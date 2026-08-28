"""Content-enriched plain-text rendering of LSP results for an LLM consumer.

Every location result is rendered as a `path:line:` header followed by the
surrounding code (+/- CONTEXT lines) with the target line(s) marked with '>'.
Overlapping context windows within the same file are merged into one block.
"""

from __future__ import annotations

import os
import re

CONTEXT = 4
MAX_REF_LOCATIONS = 40
MAX_SYM_RESULTS = 20
MAX_DIAGNOSTICS = 50
MAX_OUTPUT_LINES = 250

# Standard LSP SymbolKind values (kept local to avoid importing solidlsp here).
SYMBOL_KIND_NAMES = {
    1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class", 6: "method",
    7: "property", 8: "field", 9: "constructor", 10: "enum", 11: "interface",
    12: "function", 13: "variable", 14: "constant", 15: "string", 16: "number",
    17: "boolean", 18: "array", 19: "object", 20: "key", 21: "null",
    22: "enum-member", 23: "struct", 24: "event", 25: "operator", 26: "type-param",
}

SEVERITY_NAMES = {1: "error", 2: "warning", 3: "info", 4: "hint"}


def symbol_kind_name(kind) -> str:
    try:
        return SYMBOL_KIND_NAMES.get(int(kind), str(kind))
    except (TypeError, ValueError):
        return str(kind)


def _read_lines(abs_path: str) -> list[str] | None:
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return None


def display_path(loc_path: str, workspace: str) -> str:
    """Prefer a workspace-relative path; fall back to the absolute path."""
    if os.path.isabs(loc_path):
        try:
            rel = os.path.relpath(loc_path, workspace)
        except ValueError:
            return loc_path
        if not rel.startswith(".."):
            return rel
        return loc_path
    return loc_path


def render_location_blocks(
    locations: list[dict],
    workspace: str,
    context: int = CONTEXT,
) -> list[str]:
    """Render enriched code blocks for locations.

    Each location is a dict with keys: path (abs or ws-relative), line (1-based),
    and optionally col (1-based). Locations in the same file whose context windows
    overlap are merged into a single block with multiple '>' markers.
    """
    # group by file, keeping first-seen file order
    by_file: dict[str, list[int]] = {}
    for loc in locations:
        by_file.setdefault(loc["path"], []).append(loc["line"])

    out: list[str] = []
    for path, lines in by_file.items():
        abs_path = path if os.path.isabs(path) else os.path.join(workspace, path)
        disp = display_path(path, workspace)
        file_lines = _read_lines(abs_path)
        target_lines = sorted(set(lines))
        if file_lines is None:
            for ln in target_lines:
                out.append(f"{disp}:{ln}: (could not read file)")
            continue
        # merge overlapping context windows
        windows: list[list[int]] = []  # [start, end, targets...]
        for ln in target_lines:
            start = max(1, ln - context)
            end = min(len(file_lines), ln + context)
            if windows and start <= windows[-1][1] + 1:
                windows[-1][1] = max(windows[-1][1], end)
                windows[-1].append(ln)
            else:
                windows.append([start, end, ln])
        for w in windows:
            start, end, targets = w[0], w[1], set(w[2:])
            header_lines = ",".join(str(t) for t in sorted(targets))
            out.append(f"{disp}:{header_lines}:")
            for i in range(start, end + 1):
                marker = ">" if i in targets else " "
                text = file_lines[i - 1] if i - 1 < len(file_lines) else ""
                out.append(f"{marker}{i:6d} | {text}")
            out.append("")
    if out and out[-1] == "":
        out.pop()
    return out


def render_refs_tail(overflow: list[dict], workspace: str) -> list[str]:
    """Summarize locations beyond the cap as '... and N more in: file (count), ...'."""
    if not overflow:
        return []
    counts: dict[str, int] = {}
    for loc in overflow:
        disp = display_path(loc["path"], workspace)
        counts[disp] = counts.get(disp, 0) + 1
    parts = ", ".join(f"{p} ({n})" for p, n in sorted(counts.items()))
    return ["", f"... and {len(overflow)} more in: {parts}"]


def cap_output(lines: list[str], max_lines: int = MAX_OUTPUT_LINES) -> list[str]:
    if len(lines) <= max_lines:
        return lines
    kept = lines[: max_lines - 1]
    kept.append(f"... output truncated ({len(lines) - len(kept)} more lines)")
    return kept


_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:file://|https?://)[^)]*\)")


def _strip_link_targets(text: str) -> str:
    """Replace markdown links with their text; long file:// URIs are noise for an agent."""
    return _MD_LINK_RE.sub(r"\1", text)


def hover_contents_to_text(contents) -> str:
    """Normalize LSP hover contents (MarkupContent | MarkedString | list) to plain text."""
    if contents is None:
        return ""
    if isinstance(contents, str):
        return _strip_link_targets(contents)
    if isinstance(contents, dict):
        # MarkupContent {kind, value} or MarkedString {language, value}
        return _strip_link_targets(str(contents.get("value", "")))
    if isinstance(contents, list):
        parts = [hover_contents_to_text(c) for c in contents]
        return "\n\n".join(p for p in parts if p)
    return str(contents)
