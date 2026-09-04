"""Arm E environment: CodeAnchor-style anchors appended to grep observations.

A drop-in replacement for mini-swe-agent's DockerEnvironment (select it with
``environment_class: anchor_env.AnchorDockerEnvironment``; the directory containing this
file must be on PYTHONPATH).  Behaviour:

* At container start, the updated ``lsp_tool`` package (with the batch ``anchor`` op) is
  copied into the image's /opt/lsp-tool venv and the language-server daemon is warmed in
  the background.  Images must be the ``promax-lsp:<id>`` variants.
* After every agent command, if the command invoked a grep-family tool and the observed
  output contains ``path:line:`` (or ``path:`` / bare ``path``) hits in source files, the
  hits and the pattern's identifier tokens are sent to ``lsp anchor`` inside the container
  and its addendum is appended to the observation.  Nothing else about the command, its
  exit status, or the prompt changes — passive injection, as in CodeAnchor.
* Every decision is logged as JSONL (``ANCHOR_LOG`` env var or ``anchor_log`` config).

Pure helpers (``extract_search``, ``parse_hits``) have no docker dependency so they can be
unit-tested on any host.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time

from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig

GREP_FAMILY = re.compile(r"(?<![\w./-])(?:grep|egrep|fgrep|rg|ag|ack)(?![\w-])")
_SEPARATORS = {"|", "||", "&&", ";", "|&", "&", "(", ")", "{", "}"}
_GREP_NAMES = {"grep", "egrep", "fgrep", "rg", "ag", "ack"}
# options that consume the next token (when given separately); 'e'/'regexp' are patterns
_OPTS_WITH_ARG = {
    "-e", "-f", "-A", "-B", "-C", "-m", "-d", "-D", "-t", "-g", "-j", "-M", "-E_",
    "--regexp", "--file", "--include", "--exclude", "--exclude-dir", "--max-count",
    "--context", "--after-context", "--before-context", "--type", "--glob", "--threads",
    "--max-columns", "--color", "--colour", "--binary-files", "--directories", "--devices",
    "--label", "--exclude-from",
}
_PATTERN_OPTS = {"-e", "--regexp"}
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ESCAPE_RE = re.compile(r"\\.")
STOPWORDS = {
    # python keywords / builtins the model greps for as syntax, not as symbols
    "def", "class", "import", "from", "return", "raise", "yield", "async", "await", "lambda",
    "pass", "with", "not", "and", "for", "while", "try", "except", "finally", "else", "elif",
    "del", "global", "nonlocal", "assert", "break", "continue", "None", "True", "False",
    "self", "cls", "print", "str", "int", "float", "list", "dict", "set", "tuple", "len",
    "bool", "bytes", "object", "type", "super", "property", "staticmethod", "classmethod",
    "isinstance", "getattr", "setattr", "hasattr", "kwargs", "args", "TODO", "FIXME", "XXX",
    "test", "tests", "init", "main", "the", "and", "for", "http", "https", "www",
}
_CD_RE = re.compile(r"^\s*cd\s+([^\s;&|]+)\s*(?:&&|;)")
_MENTIONED_FILE_RE = re.compile(r"(?<![\w/])(?:\./)?[\w./@+~-]*?[\w-]\.(?:py|pyi)\b(?!/)")
_DEF_LINE_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+[A-Za-z_]")
_ANCHOR_HEADER = "--- code anchors (language server)"
_LOG_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def _split(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        # unbalanced quotes (heredocs etc.): crude whitespace split keeping quoted chunks
        return re.findall(r'"[^"]*"|\'[^\']*\'|\S+', command)


def extract_search(command: str) -> dict:
    """Identifier tokens of the grep pattern(s) in a shell command.

    Returns {"tokens": [...], "ignore_case": bool, "base_dir": str|None,
    "n_greps": int, "patterns": [...]}.
    """
    toks = _split(command)
    patterns: list[str] = []
    invocations: list[dict] = []
    ignore_case = False
    n_greps = 0
    i = 0
    while i < len(toks):
        t = toks[i].strip("'\"")
        if t in _GREP_NAMES:
            n_greps += 1
            i += 1
            positional_pattern_seen = False
            inv = {"files": [], "recursive": False, "no_filename": False}
            invocations.append(inv)
            while i < len(toks):
                a = toks[i]
                a_s = a.strip("'\"")
                if a in _SEPARATORS or a_s in _SEPARATORS:
                    break
                if a_s == "--":
                    i += 1
                    if not positional_pattern_seen and i < len(toks):
                        patterns.append(toks[i].strip("'\""))
                        positional_pattern_seen = True
                    i += 1
                    continue
                if a_s.startswith("--"):
                    name, eq, val = a_s.partition("=")
                    if name in ("--recursive", "--dereference-recursive"):
                        inv["recursive"] = True
                    elif name == "--no-filename":
                        inv["no_filename"] = True
                    if name in _PATTERN_OPTS:
                        if eq:
                            patterns.append(val)
                        elif i + 1 < len(toks):
                            i += 1
                            patterns.append(toks[i].strip("'\""))
                    elif name == "--ignore-case":
                        ignore_case = True
                    elif name in _OPTS_WITH_ARG and not eq:
                        i += 1
                    i += 1
                    continue
                if a_s.startswith("-") and len(a_s) > 1:
                    cluster = a_s[1:]
                    if "i" in cluster and not any(c.isdigit() for c in cluster):
                        ignore_case = True
                    if ("r" in cluster or "R" in cluster) and not any(c.isdigit() for c in cluster):
                        inv["recursive"] = True
                    if "h" in cluster and not any(c.isdigit() for c in cluster):
                        inv["no_filename"] = True
                    last = "-" + cluster[-1]
                    if last in _OPTS_WITH_ARG and not (len(cluster) > 1 and cluster[-1].isdigit()):
                        if last in _PATTERN_OPTS and i + 1 < len(toks):
                            i += 1
                            patterns.append(toks[i].strip("'\""))
                            positional_pattern_seen = True
                        elif last in ("-A", "-B", "-C", "-m") and cluster[:-1].isdigit():
                            pass  # e.g. -5A? unusual; ignore
                        else:
                            i += 1  # skip the option's argument
                    elif re.fullmatch(r"[ABCm]\d+", cluster) or re.fullmatch(r"\w*[ABCm]\d+", cluster):
                        pass  # attached numeric argument
                    i += 1
                    continue
                if not positional_pattern_seen:
                    patterns.append(a_s)
                    positional_pattern_seen = True
                else:
                    inv["files"].append(a_s)
                i += 1
            continue
        i += 1
    # single-file attribution: `grep -n PAT file.py` prints LINE:text without a filename.
    # Usable only when every grep invocation with a file operand names the same single
    # file non-recursively (or exactly one invocation does and the others read stdin).
    # `-r` on a single *file* operand also omits the filename; only a directory makes
    # grep print paths.
    single_file = None
    cands = set()
    for inv in invocations:
        if inv["no_filename"]:
            cands.add(None)
        elif len(inv["files"]) == 1 and (not inv["recursive"] or re.search(r"\.\w+$", inv["files"][0])):
            cands.add(inv["files"][0])
        elif len(inv["files"]) >= 1:
            cands.add(None)
    if len(cands) == 1 and None not in cands:
        single_file = next(iter(cands))
    # fallback for chains / `head file | grep -n`: every source file named anywhere in the
    # command is a candidate; the daemon matches hit text against them.
    candidates = []
    for m in _MENTIONED_FILE_RE.finditer(command):
        f = m.group(0)
        if f not in candidates:
            candidates.append(f)
    tokens: list[str] = []
    seen = set()
    for pat in patterns:
        cleaned = _ESCAPE_RE.sub(" ", pat)
        for m in _IDENT_RE.finditer(cleaned):
            w = m.group(0)
            if len(w) < 3 or w in STOPWORDS or w.isdigit():
                continue
            key = w.lower() if ignore_case else w
            if key not in seen:
                seen.add(key)
                tokens.append(key)
    base_dir = None
    m = _CD_RE.match(command)
    if m:
        d = m.group(1).strip("'\"")
        if d not in ("/testbed", "/testbed/", ".", "./"):
            base_dir = d
    return {"tokens": tokens[:12], "ignore_case": ignore_case, "base_dir": base_dir,
            "n_greps": n_greps, "patterns": patterns, "single_file": single_file,
            "candidates": candidates[:8]}


def parse_hits(output: str, exts: tuple[str, ...], max_hits: int, single_file: str | None = None,
               candidates: list[str] | None = None) -> dict:
    """Extract grep-style hits from observed output.

    Returns {"hits": [{path, line?, text?}], "grep_files": [...], "n_lines": int}.
    Lines shaped `path:LINE:text` give positions; `path:text` gives text to locate;
    bare `path` lines (grep -l) only contribute to grep_files.
    """
    ext_alt = "|".join(re.escape(e.lstrip(".")) for e in exts)
    hit_re = re.compile(
        rf"^(?P<path>(?:\./)?[\w./@+~-]*?\.(?:{ext_alt}))(?::(?P<line>\d+))?:(?P<text>.*)$"
    )
    file_re = re.compile(rf"^(?P<path>(?:\./)?[\w./@+~-]*?\.(?:{ext_alt}))\s*$")
    lineonly_re = re.compile(r"^(?P<line>\d+):(?P<text>.*)$")
    if single_file and not single_file.lower().endswith(tuple(exts)):
        single_file = None
    candidates = [c for c in (candidates or []) if c.lower().endswith(tuple(exts))]
    if single_file:
        candidates = []
    hits: list[dict] = []
    grep_files: list[str] = []
    seen_files: set[str] = set()
    seen_pos: set[tuple[str, str]] = set()
    n_lineonly = 0
    for raw in output.splitlines():
        line = raw.rstrip("\r")
        lm = lineonly_re.match(line)
        if lm:
            n_lineonly += 1
            if single_file:
                key = (single_file, lm.group("line"))
                if key in seen_pos:
                    continue
                seen_pos.add(key)
                if single_file not in seen_files:
                    seen_files.add(single_file)
                    grep_files.append(single_file)
                hits.append({"path": single_file, "line": int(lm.group("line")),
                             "_def": bool(_DEF_LINE_RE.match(lm.group("text")))})
            elif candidates and lm.group("text").strip():
                key = ("?", lm.group("line") + ":" + lm.group("text").rstrip())
                if key in seen_pos:
                    continue
                seen_pos.add(key)
                hits.append({"candidates": candidates, "line": int(lm.group("line")),
                             "text": lm.group("text"),
                             "_def": bool(_DEF_LINE_RE.match(lm.group("text")))})
            continue
        m = hit_re.match(line)
        if m:
            path = m.group("path")
            if path not in seen_files:
                seen_files.add(path)
                grep_files.append(path)
            key = (path, m.group("line") or m.group("text").rstrip())
            if key in seen_pos:
                continue
            seen_pos.add(key)
            h = {"path": path}
            if m.group("line"):
                h["line"] = int(m.group("line"))
            elif m.group("text").startswith(":"):
                continue  # pytest-style `path::test` lines are not grep hits
            else:
                h["text"] = m.group("text")
            h["_def"] = bool(_DEF_LINE_RE.match(m.group("text")))
            hits.append(h)
            continue
        m = file_re.match(line)
        if m:
            path = m.group("path")
            if path not in seen_files:
                seen_files.add(path)
                grep_files.append(path)
    # definition lines first (they are the most informative anchors), then output order
    hits.sort(key=lambda h: not h["_def"])
    for h in hits:
        h.pop("_def", None)
    return {"hits": hits[:max_hits], "grep_files": grep_files[:400], "n_lines": len(hits),
            "n_lineonly": n_lineonly}


VIEW_FAMILY = re.compile(r"(?<![\w./-])(?:cat|sed|head|tail|nl|bat|more|less)(?![\w-])")
_NUM_PREFIX_RE = re.compile(r"^\s*\d+[\t: ]")  # cat -n / nl / grep -n prefixes
_DEF_TEXT_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+[A-Za-z_]\w*")


def extract_view(command: str, exts: tuple[str, ...] = (".py", ".pyi")) -> dict:
    """File-view detection: a cat/sed/head/tail/nl invocation naming exactly one source file
    (possibly piped into head/tail). Returns {"file": path|None, "n_views": int}."""
    toks = _split(command)
    files: list[str] = []
    n_views = 0
    i = 0
    while i < len(toks):
        t = toks[i].strip("'\"")
        if t in ("cat", "sed", "head", "tail", "nl", "bat", "more", "less"):
            n_views += 1
            i += 1
            while i < len(toks):
                a = toks[i].strip("'\"")
                if a in _SEPARATORS or toks[i] in _SEPARATORS:
                    break
                if a.lower().endswith(exts) and not a.startswith("-"):
                    if a not in files:
                        files.append(a)
                i += 1
            continue
        i += 1
    return {"file": files[0] if len(files) == 1 else None, "n_views": n_views, "files": files}


def parse_view_hits(output: str, file: str, max_hits: int) -> dict:
    """def/class lines visible in a file view -> text hits on that file (the daemon locates
    each line in the file; cat -n / nl numeric prefixes are stripped)."""
    hits = []
    seen = set()
    for raw in output.splitlines():
        line = raw.rstrip("\r")
        m = re.match(r"^\s*(\d+)[\t:]\s?(.*)$", line)  # cat -n: "  12\tdef f():" ; grep -n "12:def"
        text, lineno = (m.group(2), int(m.group(1))) if m else (line, None)
        if not _DEF_TEXT_RE.match(text):
            continue
        key = text.rstrip()
        if key in seen:
            continue
        seen.add(key)
        h = {"path": file, "text": text}
        if lineno is not None:
            h["line"] = lineno
        hits.append(h)
        if len(hits) >= max_hits:
            break
    return {"hits": hits, "n_defs": len(hits)}


def visible_text(output: str, limit: int = 10000, edge: int = 5000) -> str:
    """What the agent actually sees: mini-swe-agent elides the middle of long outputs."""
    if len(output) < limit:
        return output
    return output[:edge] + "\n" + output[-edge:]


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


class AnchorDockerEnvironmentConfig(DockerEnvironmentConfig):
    anchor_pkg_dir: str = ""
    """Host directory holding the updated `lsp_tool` package to copy into the container."""
    anchor_budget: float = 12.0
    """Wall-clock budget (s) for one anchor computation inside the daemon."""
    anchor_max_hits: int = 40
    anchor_lsp_wait: float = 3.0
    """Seconds to wait for the language server to be ready before skipping (not blocking)."""
    anchor_exec_timeout: int = 45
    anchor_exts: list[str] = [".py", ".pyi"]
    anchor_log: str = ""
    """JSONL telemetry path (default: $ANCHOR_LOG)."""
    anchor_obs_limit: int = 10000
    """mini-swe-agent's observation elision threshold; the addendum never exceeds it."""
    anchor_min_room: int = 500
    """Skip the addendum when fewer than this many chars remain under the limit."""
    anchor_views: bool = False
    """Arm E3: also anchor file views (cat/sed -n/head/tail/nl) — every def/class line visible
    in the view gets its users, i.e. tags at definition sites as in CodeAnchor."""
    anchor_uncapped: bool = False
    """Arm E3: no caps on symbols/files/users (paper-faithful); only the observation limit
    (harness constraint) trims the addendum, at a line boundary, never the command output."""
    anchor_view_max_hits: int = 80


class AnchorDockerEnvironment(DockerEnvironment):
    def __init__(self, *, config_class: type = AnchorDockerEnvironmentConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        self._log_path = self.config.anchor_log or os.environ.get("ANCHOR_LOG", "")
        self._instance = self.config.image.rsplit(":", 1)[-1]
        self._n_anchor_calls = 0
        try:
            self._install_and_warm()
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"anchor setup failed: {e}")
            self._log({"event": "setup_failed", "error": str(e)})

    # -- setup ---------------------------------------------------------------

    def _docker(self, *args, timeout: int = 60, input_text: str | None = None):
        return subprocess.run(
            [self.config.executable, *args], text=True, capture_output=True,
            timeout=timeout, input=input_text,
        )

    def _install_and_warm(self) -> None:
        cid = self.container_id
        t0 = time.monotonic()
        if self.config.anchor_pkg_dir:
            r = self._docker(
                "exec", cid, "/opt/lsp-tool/bin/python", "-c",
                "import lsp_tool,os;print(os.path.dirname(lsp_tool.__file__))",
            )
            if r.returncode != 0:
                raise RuntimeError(f"cannot locate lsp_tool in container: {r.stderr.strip()}")
            target = r.stdout.strip()
            r = self._docker("cp", os.path.join(self.config.anchor_pkg_dir, "."), f"{cid}:{target}/")
            if r.returncode != 0:
                raise RuntimeError(f"docker cp failed: {r.stderr.strip()}")
            r = self._docker("exec", cid, "find", target, "-name", "__pycache__", "-type", "d",
                             "-exec", "rm", "-rf", "{}", "+")
        # warm the language server in the background (returns immediately)
        r = self._docker("exec", "-d", "-w", self.config.cwd, cid, "lsp", "daemon", "start", "--no-wait")
        self._log({"event": "setup", "seconds": round(time.monotonic() - t0, 2),
                   "warm_rc": r.returncode})

    # -- execution -----------------------------------------------------------

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        output = super().execute(action, cwd, timeout=timeout)  # raises Submitted on completion
        try:
            self._maybe_anchor(action.get("command", ""), output, cwd or self.config.cwd)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"anchor post-processing failed: {e}")
            self._log({"event": "error", "error": str(e)})
        return output

    def _uncapped(self, payload: dict) -> dict:
        if self.config.anchor_uncapped:
            payload.update({"max_symbols": 10000, "max_candidates": 10000, "max_files": 10000,
                            "max_names": 10000, "max_missing": 10000, "max_chars": 200000,
                            "max_annotate": 100000, "max_annotate_per_file": 100000})
        return payload

    def _maybe_anchor(self, command: str, output: dict, cwd: str) -> None:
        if not command:
            return
        is_grep = bool(GREP_FAMILY.search(command))
        is_view = self.config.anchor_views and bool(VIEW_FAMILY.search(command))
        if not (is_grep or is_view):
            return
        text = output.get("output") or ""
        if not text or _ANCHOR_HEADER in text:
            return
        vis = visible_text(text)
        if is_grep:
            search = extract_search(command)
            parsed = parse_hits(vis, tuple(self.config.anchor_exts), self.config.anchor_max_hits,
                                single_file=search["single_file"], candidates=search["candidates"])
            rec = {"event": "grep", "command": command[:300], "n_hits": parsed["n_lines"],
                   "n_files": len(parsed["grep_files"]), "rc": output.get("returncode"),
                   "single_file": search["single_file"], "n_lineonly": parsed["n_lineonly"],
                   "n_candidates": len(search["candidates"])}
            if not parsed["hits"]:
                if is_view:
                    return self._anchor_view(command, output, cwd, text, vis)
                rec["status"] = "unattributed" if parsed["n_lineonly"] else "no_hits"
                self._log(rec)
                return
            rec.update({"tokens": search["tokens"], "n_greps": search["n_greps"]})
            payload = self._uncapped({
                "hits": parsed["hits"],
                "grep_files": parsed["grep_files"],
                "tokens": search["tokens"],
                "ignore_case": search["ignore_case"],
                "base_dir": search["base_dir"],
                "budget": self.config.anchor_budget,
                "max_hits": self.config.anchor_max_hits,
            })
        else:
            return self._anchor_view(command, output, cwd, text, vis)
        self._run_anchor(payload, rec, output, text, cwd)

    def _anchor_view(self, command: str, output: dict, cwd: str, text: str, vis: str) -> None:
        view = extract_view(command, tuple(self.config.anchor_exts))
        rec = {"event": "view", "command": command[:300], "rc": output.get("returncode"),
               "view_file": view["file"], "n_views": view["n_views"]}
        if not view["file"]:
            rec["status"] = "no_file" if not view["files"] else "multi_file"
            self._log(rec)
            return
        parsed = parse_view_hits(vis, view["file"], self.config.anchor_view_max_hits)
        rec["n_hits"] = parsed["n_defs"]
        if not parsed["hits"]:
            rec["status"] = "no_defs"
            self._log(rec)
            return
        base_dir = extract_search(command)["base_dir"]
        payload = self._uncapped({
            "hits": parsed["hits"], "grep_files": [view["file"]], "tokens": [],
            "ignore_case": False, "base_dir": base_dir, "budget": self.config.anchor_budget,
            "max_hits": self.config.anchor_view_max_hits, "view_of": view["file"],
        })
        self._run_anchor(payload, rec, output, text, cwd)

    def _run_anchor(self, payload: dict, rec: dict, output: dict, text: str, cwd: str) -> None:
        if payload.get("view_of") and payload.get("base_dir"):
            # the daemon resolves relative paths against base_dir; keep view_of consistent
            payload["view_of"] = os.path.normpath(os.path.join(payload["base_dir"], payload["view_of"]))
        t0 = time.monotonic()
        try:
            r = subprocess.run(
                [self.config.executable, "exec", "-i", "-w", cwd,
                 "-e", f"LSP_TOOL_TIMEOUT={self.config.anchor_lsp_wait}",
                 self.container_id, "lsp", "anchor"],
                input=json.dumps(payload), text=True, capture_output=True,
                timeout=self.config.anchor_exec_timeout,
            )
        except subprocess.TimeoutExpired:
            rec.update({"status": "timeout", "seconds": round(time.monotonic() - t0, 2)})
            self._log(rec)
            return
        rec["seconds"] = round(time.monotonic() - t0, 2)
        addendum = (r.stdout or "").strip()
        if r.returncode != 0:
            err = (r.stderr or "").strip().splitlines()
            last = err[-1] if err else ""
            rec.update({"status": "not_ready" if "not ready" in last else "lsp_error", "error": last})
            self._log(rec)
            return
        if not addendum:
            rec["status"] = "empty"
            self._log(rec)
            return
        # never push the observation over mini-swe-agent's elision threshold (10,000 chars):
        # trim the addendum at a line boundary to the room left, or skip it entirely.
        base = text.rstrip("\n")
        room = self.config.anchor_obs_limit - len(base) - 2
        if room < self.config.anchor_min_room:
            rec.update({"status": "no_room", "room": room, "addendum_chars": len(addendum)})
            self._log(rec)
            return
        if len(addendum) > room:
            marker = "\n  ... (anchors truncated)"
            addendum = addendum[: room - len(marker)].rsplit("\n", 1)[0] + marker
            rec["trimmed"] = True
        m = re.search(r": (\d+) symbol", addendum)
        rec.update({
            "status": "anchored",
            "n_symbols": int(m.group(1)) if m else None,
            "partial": "partial" in addendum.splitlines()[0],
            "addendum_chars": len(addendum),
            "not_shown_lines": addendum.count("\n  not shown above:"),
        })
        self._n_anchor_calls += 1
        output["output"] = base + "\n" + addendum + "\n"
        self._log(rec)

    # -- telemetry -----------------------------------------------------------

    def _log(self, rec: dict) -> None:
        if not self._log_path:
            return
        rec = {"ts": round(time.time(), 3), "instance": self._instance, **rec}
        try:
            with _LOG_LOCK, open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
