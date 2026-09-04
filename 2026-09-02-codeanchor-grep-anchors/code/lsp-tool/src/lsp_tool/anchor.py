"""Batch `anchor` op: CodeAnchor-style structural facts for a set of grep hits.

CodeAnchor (arXiv 2606.26979) injects static-analysis facts ("used by: ...") into
source files as comments so a grep-first agent sees structure without adopting a new
tool.  This module produces the same kind of facts on demand from a warm language
server, to be appended to a grep observation:

  * each hit position (path + line, optionally the matched text) is resolved to the
    symbol it refers to via textDocument/definition;
  * distinct in-workspace symbols are reported with kind, qualified name, definition
    site, and every reference (textDocument/references) grouped by file with the
    enclosing symbol of each use ("who uses it"), tagged import/subclass/override;
  * files that reference the symbol but were absent from the grep output are listed
    separately ("referenced but NOT in this grep output") — the reference-completeness
    signal; definitions with no references collapse into one trailing line.

Output is compact and capped (see MAX_* constants); the whole computation runs under a
wall-clock budget and degrades to a partial result rather than blocking the agent.
"""

from __future__ import annotations

import logging
import os
import re
import time

from .render import display_path, symbol_kind_name

log = logging.getLogger("lsp_tool.anchor")

MAX_SYMBOLS = 4
MAX_CANDIDATES = 8  # symbols whose references we look up before choosing what to render
MAX_HITS = 40
MAX_FILES_PER_SYMBOL = 5
MAX_NAMES_PER_FILE = 3
MAX_MISSING_FILES = 6
MAX_REFS_TO_ANNOTATE = 80  # total enclosing-scope lookups per symbol
MAX_REFS_PER_FILE_ANNOTATE = 25
MAX_CHARS = 2800
DEFAULT_BUDGET = 12.0

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DEF_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")
_IMPORT_RE = re.compile(r"^\s*(?:from\s+[\w.]+\s+)?import\b")
_BARE_NAME_RE = re.compile(r"^\s*[A-Za-z_][\w.]*\s*(?:as\s+\w+\s*)?,?\s*\)?\s*(?:#.*)?$")
_CLASS_BASES_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*\(([^)]*)\)")
_SKIP_DIR_PARTS = {
    ".git", "node_modules", "site-packages", ".venv", "venv", "build", "dist",
    "__pycache__", ".tox", ".mypy_cache", ".pytest_cache",
}
_TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")


class _Budget:
    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + max(0.5, seconds)
        self.partial = False

    def ok(self) -> bool:
        if time.monotonic() >= self.deadline:
            self.partial = True
            return False
        return True


# ---------------------------------------------------------------------------
# file / symbol helpers
# ---------------------------------------------------------------------------


class _Workspace:
    """Per-request caches: file lines and flattened document symbols."""

    def __init__(self, entry):
        self.entry = entry
        self.root = entry.workspace
        self._lines: dict[str, list[str] | None] = {}
        # daemon-lifetime cache on the entry object (survives across requests)
        if not hasattr(entry, "anchor_symbol_cache"):
            entry.anchor_symbol_cache = {}
        self._symcache: dict = entry.anchor_symbol_cache

    def lines(self, rel: str) -> list[str] | None:
        if rel not in self._lines:
            try:
                with open(os.path.join(self.root, rel), encoding="utf-8", errors="replace") as f:
                    self._lines[rel] = f.read().splitlines()
            except OSError:
                self._lines[rel] = None
        return self._lines[rel]

    def line_text(self, rel: str, line0: int) -> str:
        lines = self.lines(rel)
        if lines is None or not (0 <= line0 < len(lines)):
            return ""
        return lines[line0]

    def symbols(self, rel: str) -> list[dict]:
        """Flattened document symbols: [{start, end, sel_line, sel_col, name, qual, kind}]."""
        abs_path = os.path.join(self.root, rel)
        try:
            mtime = os.stat(abs_path).st_mtime_ns
        except OSError:
            return []
        cached = self._symcache.get(rel)
        if cached and cached[0] == mtime:
            return cached[1]
        flat: list[dict] = []
        try:
            with self.entry.lock:
                doc = self.entry.ls.request_document_symbols(rel)
            roots = getattr(doc, "root_symbols", None) or []

            def walk(sym, chain):
                name = sym.get("name") or "?"
                qual = chain + [name]
                full = sym.get("range") or (sym.get("location") or {}).get("range") or {}
                sel = sym.get("selectionRange") or full
                s, e = full.get("start") or {}, full.get("end") or {}
                ss = sel.get("start") or {}
                flat.append({
                    "start": int(s.get("line", 0)),
                    "end": int(e.get("line", 0)),
                    "sel_line": int(ss.get("line", s.get("line", 0))),
                    "sel_col": int(ss.get("character", 0)),
                    "name": name,
                    "qual": ".".join(qual),
                    "kind": symbol_kind_name(sym.get("kind")),
                })
                for child in sym.get("children") or []:
                    walk(child, qual)

            for r in roots:
                walk(r, [])
        except Exception as e:  # noqa: BLE001
            log.warning("document symbols failed for %s: %s", rel, e)
            flat = []
        self._symcache[rel] = (mtime, flat)
        return flat

    def symbol_at(self, rel: str, line0: int, col0: int | None = None) -> dict | None:
        """The symbol whose name is declared at this position (selectionRange start)."""
        best = None
        for s in self.symbols(rel):
            if s["sel_line"] == line0 and (col0 is None or s["sel_col"] == col0):
                return s
            if s["sel_line"] == line0 and best is None:
                best = s
        return best

    def enclosing(self, rel: str, line0: int) -> dict | None:
        """Deepest (smallest-range) symbol whose range contains the line."""
        best = None
        for s in self.symbols(rel):
            if s["start"] <= line0 <= s["end"]:
                if best is None or (s["end"] - s["start"]) < (best["end"] - best["start"]):
                    best = s
        return best


def _rel_in_workspace(loc: dict, root: str) -> str | None:
    """Workspace-relative path of an LSP location, or None if outside the workspace."""
    path = loc.get("relativePath")
    if not path:
        abs_path = loc.get("absolutePath") or ""
        if not abs_path and loc.get("uri", "").startswith("file://"):
            from urllib.parse import unquote, urlparse
            abs_path = unquote(urlparse(loc["uri"]).path)
        if not abs_path:
            return None
        try:
            path = os.path.relpath(os.path.realpath(abs_path), root)
        except ValueError:
            return None
    if path.startswith(".."):
        return None
    if any(part in _SKIP_DIR_PARTS for part in path.split("/")):
        return None
    return path


def _loc_pos(loc: dict) -> tuple[int, int]:
    start = (loc.get("range") or {}).get("start") or {}
    return int(start.get("line", 0)), int(start.get("character", 0))


def _find_column(text: str, tokens: set[str], ignore_case: bool) -> tuple[int, str] | None:
    """Pick the identifier on the line to query: a whole-word pattern token, else an
    identifier containing a token, else the name declared by a def/class line."""
    if tokens:
        for m in _IDENT_RE.finditer(text):
            w = m.group(0)
            if (w.lower() if ignore_case else w) in tokens:
                return m.start(), w
        for m in _IDENT_RE.finditer(text):
            w = m.group(0)
            wl = w.lower() if ignore_case else w
            if any(t in wl for t in tokens):
                return m.start(), w
    d = _DEF_RE.match(text)
    if d:
        return d.start(1), d.group(1)
    return None


def _classify(text: str, name: str, is_class: bool) -> str:
    if _IMPORT_RE.match(text) or _BARE_NAME_RE.match(text):
        return "import"
    m = _CLASS_BASES_RE.match(text)
    if m and is_class and re.search(rf"\b{re.escape(name)}\b", m.group(2)):
        return "subclass"
    d = _DEF_RE.match(text)
    if d and d.group(1) == name:
        return "override"
    return "use"


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def compute_anchors(entry, args: dict) -> str:
    """Return the addendum text ('' when nothing useful was found)."""
    t0 = time.monotonic()
    ws = _Workspace(entry)
    root = ws.root
    budget = _Budget(float(args.get("budget") or DEFAULT_BUDGET))
    ignore_case = bool(args.get("ignore_case"))
    tokens = {t.lower() if ignore_case else t for t in (args.get("tokens") or []) if t}
    base_dir = (args.get("base_dir") or "").strip("/")
    max_hits = int(args.get("max_hits") or MAX_HITS)
    max_symbols = int(args.get("max_symbols") or MAX_SYMBOLS)
    max_files = int(args.get("max_files") or MAX_FILES_PER_SYMBOL)
    max_names = int(args.get("max_names") or MAX_NAMES_PER_FILE)
    max_missing = int(args.get("max_missing") or MAX_MISSING_FILES)
    max_chars = int(args.get("max_chars") or MAX_CHARS)
    max_annot = int(args.get("max_annotate") or MAX_REFS_TO_ANNOTATE)
    max_annot_file = int(args.get("max_annotate_per_file") or MAX_REFS_PER_FILE_ANNOTATE)
    view_of = args.get("view_of")  # set when the hits come from a file view (definition sites)

    def resolve_rel(path: str) -> str | None:
        cands = []
        if os.path.isabs(path):
            cands.append(path)
        else:
            p = path[2:] if path.startswith("./") else path
            if base_dir:
                cands.append(os.path.join(root, base_dir, p))
            cands.append(os.path.join(root, p))
        for c in cands:
            c = os.path.realpath(c)
            if os.path.isfile(c):
                rel = os.path.relpath(c, root)
                if not rel.startswith(".."):
                    return rel
        return None

    grep_files: set[str] = set()
    for p in args.get("grep_files") or []:
        rel = resolve_rel(p)
        if rel:
            grep_files.add(rel)

    # 1. resolve hits -> (rel, line0, col0, word)
    resolved: list[tuple[str, int, int, str]] = []
    seen_pos: set[tuple[str, int]] = set()
    for hit in (args.get("hits") or [])[: max_hits * 2]:
        if len(resolved) >= max_hits:
            break
        if hit.get("candidates") and hit.get("text") is not None:
            # line-only grep output with several possible source files: pick the
            # candidate whose text matches (at that line first, else anywhere).
            want = str(hit["text"]).rstrip()
            found = None
            cand_rels = [r for r in (resolve_rel(str(c)) for c in hit["candidates"]) if r]
            ln = int(hit.get("line") or 0) - 1
            for crel in cand_rels:
                cl = ws.lines(crel)
                if cl and 0 <= ln < len(cl) and cl[ln].rstrip() == want:
                    found = (crel, ln)
                    break
            if found is None and want.strip():
                for crel in cand_rels:
                    cl = ws.lines(crel) or []
                    for i, l in enumerate(cl):
                        if l.rstrip() == want:
                            found = (crel, i)
                            break
                    if found:
                        break
            if found is None:
                continue
            hit = {"path": found[0], "line": found[1] + 1}
        rel = resolve_rel(str(hit.get("path") or ""))
        if rel is None:
            continue
        grep_files.add(rel)
        lines = ws.lines(rel)
        if lines is None:
            continue
        line0 = None
        if hit.get("line"):
            line0 = int(hit["line"]) - 1
        elif hit.get("text") is not None:
            want = str(hit["text"]).rstrip()
            for i, l in enumerate(lines):
                if l.rstrip() == want:
                    line0 = i
                    break
        if line0 is None or not (0 <= line0 < len(lines)) or (rel, line0) in seen_pos:
            continue
        seen_pos.add((rel, line0))
        found = _find_column(lines[line0], tokens, ignore_case)
        if found is None:
            continue
        resolved.append((rel, line0, found[0], found[1]))

    if not resolved:
        return ""

    # 2. definitions -> group hits by defining position
    groups: dict[tuple[str, int, int], dict] = {}
    for rel, line0, col0, word in resolved:
        if not budget.ok():
            break
        try:
            with entry.lock:
                locs = entry.ls.request_definition(rel, line0, col0)
        except Exception as e:  # noqa: BLE001
            log.info("definition failed at %s:%d:%d: %s", rel, line0 + 1, col0 + 1, e)
            continue
        if not locs:
            continue
        loc = locs[0]
        drel = _rel_in_workspace(loc, root)
        if drel is None:
            continue  # external (stdlib / site-packages)
        dline, dcol = _loc_pos(loc)
        key = (drel, dline, dcol)
        g = groups.setdefault(key, {"hits": 0, "is_def": False, "word": word})
        g["hits"] += 1
        if drel == rel and dline == line0:
            g["is_def"] = True

    if not groups:
        return ""

    if view_of:
        # file view: keep definition order (line ascending) — tags colocated with code
        ordered = sorted(groups.items(), key=lambda kv: (kv[0][0] != view_of, kv[0][1]))
    else:
        ordered = sorted(groups.items(), key=lambda kv: (-kv[1]["hits"], not kv[1]["is_def"], kv[0]))
    max_candidates = max(max_symbols, int(args.get("max_candidates") or MAX_CANDIDATES))
    ordered = ordered[:max_candidates]

    # 3. per candidate symbol: identity + references
    cands: list[dict] = []
    for (drel, dline, dcol), g in ordered:
        if not budget.ok():
            break
        dtext = ws.line_text(drel, dline)
        sym = ws.symbol_at(drel, dline, dcol) or ws.symbol_at(drel, dline)
        if sym is not None:
            name, qual, kind = sym["name"], sym["qual"], sym["kind"]
        else:
            m = _IDENT_RE.match(dtext[dcol:]) if dcol < len(dtext) else None
            name = m.group(0) if m else g["word"]
            qual = name
            d = _DEF_RE.match(dtext)
            kind = ("class" if d and "class" in dtext.split(name)[0] else
                    "function" if d else "variable")
        refs: list[tuple[str, int]] | None
        try:
            with entry.lock:
                rlocs = entry.ls.request_references(drel, dline, dcol)
            refs = []
            for rl in rlocs or []:
                rrel = _rel_in_workspace(rl, root)
                if rrel is None:
                    continue
                rline, _ = _loc_pos(rl)
                if rrel == drel and rline == dline:
                    continue
                refs.append((rrel, rline))
            refs = sorted(set(refs))
        except Exception as e:  # noqa: BLE001
            log.info("references failed at %s:%d: %s", drel, dline + 1, e)
            refs = None
        cands.append({"drel": drel, "dline": dline, "name": name, "qual": qual, "kind": kind,
                      "hits": g["hits"], "refs": refs})

    if args.get("raw"):
        # machine-readable, uncapped (analysis/replay use): every candidate with the full
        # per-file reference counts and the files absent from the grep output
        import json as _json
        out = {"grep_files": sorted(grep_files), "n_hits": len(resolved), "partial": budget.partial,
               "symbols": []}
        for c in cands:
            per_file: dict[str, int] = {}
            for rrel, _ in (c["refs"] or []):
                per_file[rrel] = per_file.get(rrel, 0) + 1
            out["symbols"].append({
                "name": c["name"], "qual": c["qual"], "kind": c["kind"],
                "def": f"{c['drel']}:{c['dline'] + 1}", "hits": c["hits"],
                "refs_available": c["refs"] is not None,
                "ref_files": per_file,
                "missing": sorted(f for f in per_file if f not in grep_files and f != c["drel"]),
            })
        return _json.dumps(out)

    with_refs = [c for c in cands if c["refs"]]
    without = [c for c in cands if not c["refs"]]
    # keep hit-count order among those with references; zero-ref definitions are
    # collapsed into a single trailing line (typically overrides / unused defs)
    shown = with_refs[:max_symbols]
    blocks: list[list[str]] = []
    for c in shown:
        if not budget.ok():
            break
        drel, dline, name, refs = c["drel"], c["dline"], c["name"], c["refs"]
        is_class = c["kind"] == "class"
        head = f"{name}  {c['kind']} {c['qual']}  defined {display_path(drel, root)}:{dline + 1}"
        if not view_of:
            head += f"  ({c['hits']} hit{'s' if c['hits'] != 1 else ''} above)"
        block = [head]
        per_file: dict[str, dict] = {}
        for rrel, _ in refs:
            per_file.setdefault(rrel, {"n": 0, "names": {}})["n"] += 1
        n_files = len(per_file)
        files_sorted = sorted(
            per_file.items(),
            key=lambda kv: (bool(_TEST_PATH_RE.search(kv[0])), -kv[1]["n"], kv[0]),
        )
        # enclosing-scope lookups only for the files that will be displayed
        display_files = {r for r, _ in files_sorted[:max_files]}
        annotated = 0
        per_file_annotated: dict[str, int] = {}
        for rrel, rline in refs:
            if rrel not in display_files or annotated >= max_annot or not budget.ok():
                continue
            if per_file_annotated.get(rrel, 0) >= max_annot_file:
                continue
            per_file_annotated[rrel] = per_file_annotated.get(rrel, 0) + 1
            annotated += 1
            pf = per_file[rrel]
            text = ws.line_text(rrel, rline)
            tag = _classify(text, name, is_class)
            enc = ws.enclosing(rrel, rline)
            label = enc["qual"] if enc else ""
            if tag == "subclass":
                m = _CLASS_BASES_RE.match(text)
                label = (m.group(1) if m else label) + " [subclass]"
            elif tag == "override":
                label = (label or name) + " [override]"
            elif tag == "import":
                label = "[import]"
            key = label or "module level"
            pf["names"][key] = pf["names"].get(key, 0) + 1
        block.append(f"  used by {len(refs)} site{'s' if len(refs) != 1 else ''} in "
                     f"{n_files} file{'s' if n_files != 1 else ''}:")
        for rrel, pf in files_sorted[:max_files]:
            names = sorted(pf["names"].items(), key=lambda kv: -kv[1])
            descr = []
            for label, cnt in names[:max_names]:
                descr.append(f"{label} x{cnt}" if cnt > 1 else label)
            if len(names) > max_names:
                descr.append(f"+{len(names) - max_names} more")
            inner = ", ".join(descr) if descr else f"x{pf['n']}"
            block.append(f"    {display_path(rrel, root)}: {inner}")
        if n_files > max_files:
            block.append(f"    +{n_files - max_files} more files")
        missing = [(r, pf["n"]) for r, pf in files_sorted if r not in grep_files and r != drel]
        if missing:
            shown_m = ", ".join(f"{display_path(r, root)} ({cnt})" for r, cnt in missing[:max_missing])
            if len(missing) > max_missing:
                shown_m += f", +{len(missing) - max_missing} more"
            block.append(f"  referenced but NOT in this {'file' if view_of else 'grep output'}: {shown_m}")
        blocks.append(block)

    if without and (blocks or len(without) > 1 or without[0]["hits"] > 0):
        unavailable = [c for c in without if c["refs"] is None]
        zero = [c for c in without if c["refs"] is not None]
        if zero:
            zcap = len(zero) if view_of or max_symbols >= 1000 else 6
            items = ", ".join(f"{c['qual']} ({display_path(c['drel'], root)}:{c['dline'] + 1})"
                              for c in zero[:zcap])
            if len(zero) > zcap:
                items += f", +{len(zero) - zcap} more"
            blocks.append([f"defined but never referenced elsewhere: {items}"])
        if unavailable:
            blocks.append(["(references unavailable for: " +
                           ", ".join(c["qual"] for c in unavailable[:6]) + ")"])

    if not blocks:
        return ""

    elapsed = time.monotonic() - t0
    scope = f" for {display_path(view_of, root)}" if view_of else ""
    header = (f"--- code anchors (language server){scope}: {len(blocks)} symbol"
              f"{'s' if len(blocks) != 1 else ''}"
              f"{', partial (time budget)' if budget.partial else ''} ---")
    lines = [header]
    for b in blocks:
        lines.extend(b)
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n  ... (anchors truncated)"
    log.info("anchor: %d hits -> %d symbols in %.2fs (%d chars)",
             len(resolved), len(blocks), elapsed, len(text))
    return text
