"""lsp-tool daemon: keeps solidlsp language servers warm behind a Unix socket.

Protocol: JSON lines. One request object per connection, one response object back.
Request:  {"op": ..., "workspace": ..., "ls_id": ..., "args": {...}, "wait": seconds}
Response: {"ok": true, "text": ...} or {"ok": false, "error": "one-line message"}
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import os
import re
import socket
import sys
import threading
import traceback
from urllib.parse import unquote, urlparse

from . import __version__
from .languages import find_executable
from .protocol import log_path, socket_path
from .render import (
    MAX_DIAGNOSTICS,
    MAX_OUTPUT_LINES,
    MAX_REF_LOCATIONS,
    MAX_SYM_RESULTS,
    cap_output,
    display_path,
    hover_contents_to_text,
    render_location_blocks,
    render_refs_tail,
    symbol_kind_name,
    SEVERITY_NAMES,
)

log = logging.getLogger("lsp_tool.daemon")

DEFAULT_WAIT_FOR_READY = 240.0

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# declaration keywords to skip when guessing the column of the symbol on a line
_DECL_KEYWORDS = {
    "def", "class", "async", "lambda", "import", "from", "return", "await",
    "function", "const", "let", "var", "export", "default", "abstract",
    "public", "private", "protected", "static", "final", "void", "interface",
    "enum", "struct", "trait", "impl", "fn", "pub", "func", "type", "package",
    "unsafe", "extern", "virtual", "override", "template", "typename", "new",
}


class RequestError(Exception):
    """A user-facing, one-line error."""


def _uri_to_path(uri: str) -> str:
    if uri.startswith("file://"):
        return unquote(urlparse(uri).path)
    return uri


class ServerEntry:
    """One solidlsp language server for a (workspace, ls_id) pair."""

    def __init__(self, workspace: str, ls_id: str):
        self.workspace = workspace
        self.ls_id = ls_id
        self.state = "starting"  # starting | ready | failed
        self.error: str | None = None
        self.ls = None
        self.ready_event = threading.Event()
        self.lock = threading.Lock()  # serializes LSP requests to this server
        self.start_seconds: float | None = None

    def describe(self) -> dict:
        return {
            "workspace": self.workspace,
            "ls_id": self.ls_id,
            "state": self.state,
            "error": self.error,
            "startup_seconds": self.start_seconds,
        }


class ServerManager:
    def __init__(self) -> None:
        self._servers: dict[tuple[str, str], ServerEntry] = {}
        self._lock = threading.Lock()

    def entries(self) -> list[ServerEntry]:
        with self._lock:
            return list(self._servers.values())

    def ensure(self, workspace: str, ls_id: str) -> ServerEntry:
        key = (workspace, ls_id)
        with self._lock:
            entry = self._servers.get(key)
            if entry is not None and entry.state != "failed":
                return entry
            entry = ServerEntry(workspace, ls_id)
            self._servers[key] = entry
        threading.Thread(target=self._start_entry, args=(entry,), daemon=True).start()
        return entry

    def _start_entry(self, entry: ServerEntry) -> None:
        import time

        t0 = time.monotonic()
        try:
            entry.ls = _create_language_server(entry.workspace, entry.ls_id)
            entry.ls.start()
            entry.start_seconds = round(time.monotonic() - t0, 2)
            entry.state = "ready"
            log.info(
                "language server %s ready for %s in %.1fs",
                entry.ls_id, entry.workspace, entry.start_seconds,
            )
        except Exception as e:  # noqa: BLE001
            entry.state = "failed"
            entry.error = _one_line(str(e) or repr(e))
            log.error("failed to start %s for %s: %s\n%s",
                      entry.ls_id, entry.workspace, e, traceback.format_exc())
        finally:
            entry.ready_event.set()

    def get_ready(self, workspace: str, ls_id: str, wait: float) -> ServerEntry:
        entry = self.ensure(workspace, ls_id)
        if not entry.ready_event.wait(timeout=wait):
            raise RequestError("language server not ready yet (indexing)")
        if entry.state == "failed":
            raise RequestError(f"language server failed to start: {entry.error}")
        return entry

    def stop_all(self) -> None:
        for entry in self.entries():
            if entry.ls is not None:
                try:
                    entry.ls.stop()
                except Exception:  # noqa: BLE001
                    pass


def _one_line(text: str) -> str:
    return " ".join(text.split()) or "unknown error"


def _create_language_server(workspace: str, ls_id: str):
    from solidlsp import SolidLanguageServer
    from solidlsp.ls_config import LanguageServerConfig, LanguageServerId
    from solidlsp.settings import SolidLSPSettings

    try:
        lsid = LanguageServerId(ls_id)
    except ValueError:
        raise RequestError(f"unknown language server id '{ls_id}'") from None

    ls_specific_settings = {}
    # Prefer locally installed (pip) binaries for the Python servers; solidlsp's
    # default is to run them via `uvx`, which requires uv + network at first use.
    if lsid == LanguageServerId.PYTHON_PYREFLY:
        exe = find_executable("pyrefly")
        if exe:
            ls_specific_settings[lsid] = {"ls_path": exe}
    elif lsid == LanguageServerId.PYTHON_BASEDPYRIGHT:
        exe = find_executable("basedpyright-langserver")
        if exe:
            ls_specific_settings[lsid] = {"ls_path": exe}

    data_root = os.environ.get(
        "LSP_TOOL_DATA_DIR", os.path.join(os.path.expanduser("~"), ".solidlsp")
    )
    ws_hash = hashlib.sha1(workspace.encode("utf-8")).hexdigest()[:12]
    settings = SolidLSPSettings(
        solidlsp_dir=data_root,
        project_data_path=os.path.join(data_root, "projects", ws_hash),
        ls_specific_settings=ls_specific_settings,
    )
    config = LanguageServerConfig(ls_id=lsid)
    return SolidLanguageServer.create(config, workspace, solidlsp_settings=settings)


# ---------------------------------------------------------------------------
# request handling
# ---------------------------------------------------------------------------


class Daemon:
    def __init__(self, sock_file: str):
        self.sock_file = sock_file
        self.manager = ServerManager()
        self._shutdown = threading.Event()

    # -- helpers ------------------------------------------------------------

    def _relpath(self, workspace: str, file_path: str) -> str:
        abs_path = file_path if os.path.isabs(file_path) else os.path.join(workspace, file_path)
        abs_path = os.path.realpath(abs_path)
        if not os.path.isfile(abs_path):
            raise RequestError(f"file not found: {file_path}")
        rel = os.path.relpath(abs_path, workspace)
        if rel.startswith(".."):
            raise RequestError(f"file is outside the workspace: {file_path}")
        return rel

    def _resolve_column(self, entry: ServerEntry, rel: str, line0: int, col: int | None) -> int:
        """Return a 0-based column. If col (1-based) is given, convert; otherwise guess
        the symbol column: prefer a document symbol declared on that line, then the
        first non-keyword identifier, then the first non-whitespace character."""
        if col is not None:
            return max(0, col - 1)
        # 1) document symbols with a selectionRange starting on this line
        try:
            with entry.lock:
                symbols = entry.ls.request_document_symbols(rel)
            for sym in symbols.iter_symbols():
                sel = sym.get("selectionRange") or sym.get("location", {}).get("range")
                if sel and sel["start"]["line"] == line0:
                    return sel["start"]["character"]
        except Exception:  # noqa: BLE001
            pass
        # 2) first identifier on the line that is not a declaration keyword
        abs_path = os.path.join(entry.workspace, rel)
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
            text = lines[line0] if line0 < len(lines) else ""
        except OSError:
            text = ""
        for m in _WORD_RE.finditer(text):
            if m.group(0) not in _DECL_KEYWORDS:
                return m.start()
        stripped = len(text) - len(text.lstrip())
        return stripped if text.strip() else 0

    @staticmethod
    def _locations_to_dicts(locations) -> list[dict]:
        out = []
        for loc in locations:
            path = loc.get("relativePath") or loc.get("absolutePath") or _uri_to_path(loc.get("uri", ""))
            rng = loc.get("range", {})
            start = rng.get("start", {})
            out.append({
                "path": path,
                "line": int(start.get("line", 0)) + 1,
                "col": int(start.get("character", 0)) + 1,
            })
        return out

    def _position_args(self, entry: ServerEntry, args: dict) -> tuple[str, int, int]:
        rel = self._relpath(entry.workspace, args["file"])
        line = int(args["line"])
        if line < 1:
            raise RequestError("line numbers are 1-based")
        line0 = line - 1
        col0 = self._resolve_column(entry, rel, line0, args.get("col"))
        return rel, line0, col0

    # -- ops ----------------------------------------------------------------

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "ping":
            return {"ok": True, "text": f"lsp-tool daemon {__version__} pid {os.getpid()}"}
        if op == "shutdown":
            self._shutdown.set()
            return {"ok": True, "text": "daemon shutting down"}
        if op == "status":
            lines = [f"lsp-tool daemon {__version__} pid {os.getpid()} socket {self.sock_file}"]
            entries = self.manager.entries()
            if not entries:
                lines.append("no language servers started")
            for e in entries:
                extra = f" (started in {e.start_seconds}s)" if e.start_seconds else ""
                err = f": {e.error}" if e.error else ""
                lines.append(f"  {e.ls_id}  {e.workspace}  {e.state}{extra}{err}")
            return {"ok": True, "text": "\n".join(lines)}

        workspace = os.path.realpath(req.get("workspace") or os.getcwd())
        if not os.path.isdir(workspace):
            raise RequestError(f"workspace is not a directory: {workspace}")
        ls_id = req.get("ls_id")
        if not ls_id:
            raise RequestError("missing language server id")
        args = req.get("args") or {}
        wait = float(req.get("wait") or DEFAULT_WAIT_FOR_READY)

        if op == "warm":
            entry = self.manager.ensure(workspace, ls_id)
            if args.get("wait_ready"):
                entry = self.manager.get_ready(workspace, ls_id, wait)
                return {"ok": True, "text": f"{ls_id} ready for {workspace} "
                                            f"(startup {entry.start_seconds}s)"}
            return {"ok": True, "text": f"{ls_id} starting for {workspace} (state: {entry.state})"}

        entry = self.manager.get_ready(workspace, ls_id, wait)
        handler = {
            "def": self._op_def,
            "refs": self._op_refs,
            "hover": self._op_hover,
            "sym": self._op_sym,
            "outline": self._op_outline,
            "diag": self._op_diag,
            "rename": self._op_rename,
            "anchor": self._op_anchor,
        }.get(op)
        if handler is None:
            raise RequestError(f"unknown command '{op}'")
        return handler(entry, args)

    def _op_def(self, entry: ServerEntry, args: dict) -> dict:
        rel, line0, col0 = self._position_args(entry, args)
        with entry.lock:
            locations = entry.ls.request_definition(rel, line0, col0)
        if not locations:
            raise RequestError("no definition found")
        locs = self._locations_to_dicts(locations)
        lines = render_location_blocks(locs, entry.workspace)
        return {"ok": True, "text": "\n".join(cap_output(lines))}

    def _op_refs(self, entry: ServerEntry, args: dict) -> dict:
        rel, line0, col0 = self._position_args(entry, args)
        with entry.lock:
            locations = entry.ls.request_references(rel, line0, col0)
        if not locations:
            raise RequestError("no references found")
        locs = self._locations_to_dicts(locations)
        locs.sort(key=lambda d: (d["path"], d["line"], d["col"]))
        shown, overflow = locs[:MAX_REF_LOCATIONS], locs[MAX_REF_LOCATIONS:]
        lines = [f"{len(locs)} reference(s)"]
        lines += render_location_blocks(shown, entry.workspace)
        tail = render_refs_tail(overflow, entry.workspace)
        lines = cap_output(lines, MAX_OUTPUT_LINES - len(tail))
        lines += tail
        return {"ok": True, "text": "\n".join(lines)}

    def _op_hover(self, entry: ServerEntry, args: dict) -> dict:
        rel, line0, col0 = self._position_args(entry, args)
        with entry.lock:
            hover = entry.ls.request_hover(rel, line0, col0)
        text = hover_contents_to_text(hover.get("contents")) if hover else ""
        if not text.strip():
            raise RequestError("no hover information found")
        disp = display_path(rel, entry.workspace)
        lines = [f"{disp}:{line0 + 1}:{col0 + 1}:"] + text.strip().splitlines()
        return {"ok": True, "text": "\n".join(cap_output(lines))}

    def _op_sym(self, entry: ServerEntry, args: dict) -> dict:
        query = args.get("name") or ""
        with entry.lock:
            symbols = entry.ls.request_workspace_symbol(query)
        if not symbols:
            raise RequestError(f"no symbols found matching '{query}'")
        results = []
        seen = set()
        for sym in symbols:
            loc = sym.get("location") or {}
            path = loc.get("relativePath") or loc.get("absolutePath") or _uri_to_path(loc.get("uri", ""))
            rng = (loc.get("range") or {}).get("start", {})
            item = {
                "name": sym.get("name", "?"),
                "kind": symbol_kind_name(sym.get("kind")),
                "container": sym.get("containerName") or "",
                "path": path,
                "line": int(rng.get("line", 0)) + 1,
            }
            key = (item["name"], item["path"], item["line"])
            if key in seen:
                continue
            seen.add(key)
            results.append(item)
        results.sort(key=lambda r: (r["path"], r["line"]))
        shown, overflow = results[:MAX_SYM_RESULTS], results[MAX_SYM_RESULTS:]
        lines = [f"{len(results)} symbol(s) matching '{query}'"]
        for r in shown:
            container = f" (in {r['container']})" if r["container"] else ""
            lines.append("")
            lines.append(f"{r['kind']} {r['name']}{container}")
            lines += render_location_blocks(
                [{"path": r["path"], "line": r["line"]}], entry.workspace, context=2
            )
        if overflow:
            lines += render_refs_tail(
                [{"path": r["path"], "line": r["line"]} for r in overflow], entry.workspace
            )
        return {"ok": True, "text": "\n".join(cap_output(lines))}

    def _op_outline(self, entry: ServerEntry, args: dict) -> dict:
        rel = self._relpath(entry.workspace, args["file"])
        with entry.lock:
            symbols = entry.ls.request_document_symbols(rel)
        roots = symbols.root_symbols
        if not roots:
            raise RequestError("no symbols found in file")
        disp = display_path(rel, entry.workspace)
        lines = [f"{disp}:"]

        def emit(sym, depth):
            sel = sym.get("selectionRange") or (sym.get("location") or {}).get("range") or {}
            line_no = int((sel.get("start") or {}).get("line", 0)) + 1
            detail = sym.get("detail") or ""
            detail_s = f"  {detail}" if detail else ""
            lines.append(
                f"{'  ' * depth}{line_no:5d}: {symbol_kind_name(sym.get('kind'))} "
                f"{sym.get('name', '?')}{detail_s}"
            )
            for child in sym.get("children") or []:
                emit(child, depth + 1)

        for root in roots:
            emit(root, 0)
        return {"ok": True, "text": "\n".join(cap_output(lines))}

    def _op_diag(self, entry: ServerEntry, args: dict) -> dict:
        rel = self._relpath(entry.workspace, args["file"])
        with entry.lock:
            diags = entry.ls.request_text_document_diagnostics(rel)
        if not diags:
            return {"ok": True, "text": "no diagnostics (file is clean)"}
        disp = display_path(rel, entry.workspace)
        abs_path = os.path.join(entry.workspace, rel)
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                file_lines = f.read().splitlines()
        except OSError:
            file_lines = []
        diags = sorted(diags, key=lambda d: (d["range"]["start"]["line"], d["range"]["start"]["character"]))
        lines = [f"{len(diags)} diagnostic(s) in {disp}"]
        for d in diags[:MAX_DIAGNOSTICS]:
            start = d["range"]["start"]
            ln, col = int(start["line"]) + 1, int(start["character"]) + 1
            sev = SEVERITY_NAMES.get(int(d.get("severity", 1)), "error")
            code = d.get("code")
            code_s = f" [{code}]" if code is not None else ""
            msg = " ".join(str(d.get("message", "")).split())
            lines.append(f"{disp}:{ln}:{col}: {sev}: {msg}{code_s}")
            if 0 < ln <= len(file_lines):
                lines.append(f">{ln:6d} | {file_lines[ln - 1]}")
        if len(diags) > MAX_DIAGNOSTICS:
            lines.append(f"... and {len(diags) - MAX_DIAGNOSTICS} more")
        return {"ok": True, "text": "\n".join(cap_output(lines))}

    def _op_rename(self, entry: ServerEntry, args: dict) -> dict:
        rel, line0, col0 = self._position_args(entry, args)
        new_name = args.get("new_name")
        if not new_name:
            raise RequestError("missing new name")
        with entry.lock:
            edit = entry.ls.request_rename_symbol_edit(rel, line0, col0, new_name)
        file_edits = self._workspace_edit_to_file_edits(edit, entry.workspace)
        if not file_edits:
            raise RequestError("no rename edits (symbol cannot be renamed here)")
        n_edits = sum(len(e) for e in file_edits.values())
        lines = [
            f"rename to '{new_name}' would make {n_edits} edit(s) "
            f"in {len(file_edits)} file(s)  [dry-run only, nothing applied]"
        ]
        for path, edits in sorted(file_edits.items()):
            disp = display_path(path, entry.workspace)
            abs_path = path if os.path.isabs(path) else os.path.join(entry.workspace, path)
            try:
                with open(abs_path, encoding="utf-8", errors="replace") as f:
                    old_text = f.read()
            except OSError:
                lines.append(f"{disp}: (could not read file)")
                continue
            new_text = _apply_text_edits(old_text, edits)
            diff = difflib.unified_diff(
                old_text.splitlines(), new_text.splitlines(),
                fromfile=disp, tofile=disp, lineterm="", n=2,
            )
            lines.append("")
            lines.extend(diff)
        return {"ok": True, "text": "\n".join(cap_output(lines))}

    def _op_anchor(self, entry: ServerEntry, args: dict) -> dict:
        """Batch structural facts for grep hits (see anchor.py)."""
        from .anchor import compute_anchors

        return {"ok": True, "text": compute_anchors(entry, args)}

    @staticmethod
    def _workspace_edit_to_file_edits(edit, workspace: str) -> dict:
        """Normalize a WorkspaceEdit into {abs_or_rel_path: [TextEdit, ...]}."""
        file_edits: dict[str, list] = {}
        if not edit:
            return file_edits
        for uri, edits in (edit.get("changes") or {}).items():
            if edits:
                file_edits.setdefault(_uri_to_path(uri), []).extend(edits)
        for change in edit.get("documentChanges") or []:
            if not isinstance(change, dict) or "textDocument" not in change:
                continue  # skip create/rename/delete file operations
            uri = change["textDocument"].get("uri", "")
            edits = change.get("edits") or []
            if edits:
                file_edits.setdefault(_uri_to_path(uri), []).extend(edits)
        return file_edits


def _apply_text_edits(text: str, edits: list) -> str:
    """Apply LSP TextEdits to a document string (in memory only)."""
    lines = text.split("\n")
    # offsets of each line start
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    def to_offset(p) -> int:
        line = min(int(p["line"]), len(lines) - 1)
        line_len = len(lines[line])
        return offsets[line] + min(int(p["character"]), line_len)

    spans = []
    for e in edits:
        rng = e.get("range", {})
        spans.append((to_offset(rng.get("start", {})), to_offset(rng.get("end", {})), e.get("newText", "")))
    spans.sort(key=lambda s: (s[0], s[1]), reverse=True)
    for start, end, new in spans:
        text = text[:start] + new + text[end:]
    return text


# ---------------------------------------------------------------------------
# socket server
# ---------------------------------------------------------------------------


def _serve_connection(daemon: Daemon, conn: socket.socket) -> None:
    try:
        conn.settimeout(600)
        f = conn.makefile("rwb")
        line = f.readline()
        if not line:
            return
        try:
            req = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            resp = {"ok": False, "error": "invalid JSON request"}
        else:
            try:
                resp = daemon.handle(req)
            except RequestError as e:
                resp = {"ok": False, "error": str(e)}
            except Exception as e:  # noqa: BLE001
                log.error("internal error: %s\n%s", e, traceback.format_exc())
                resp = {"ok": False, "error": _one_line(f"internal error: {e}")}
        f.write(json.dumps(resp).encode("utf-8") + b"\n")
        f.flush()
    except (BrokenPipeError, ConnectionResetError, socket.timeout):
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _socket_in_use(sock_file: str) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(2)
        probe.connect(sock_file)
        return True
    except OSError:
        return False
    finally:
        probe.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lsp-tool-daemon")
    parser.add_argument("--socket", default=None, help="unix socket path")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args(argv)

    try:
        sock_file = args.socket or socket_path()
    except Exception as e:  # SocketPathError
        print(f"lsp-tool daemon: {e}", file=sys.stderr)
        return 1
    if len(sock_file.encode("utf-8")) > 100:
        print(f"lsp-tool daemon: socket path too long for AF_UNIX: {sock_file}", file=sys.stderr)
        return 1
    logfile = args.log_file or log_path(sock_file)
    logging.basicConfig(
        filename=logfile,
        level=os.environ.get("LSP_TOOL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("solidlsp").setLevel(logging.WARNING)
    logging.getLogger("serena").setLevel(logging.WARNING)

    if os.path.exists(sock_file):
        if _socket_in_use(sock_file):
            print(f"daemon already running on {sock_file}", file=sys.stderr)
            return 0
        os.unlink(sock_file)  # stale socket

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_file)
    os.chmod(sock_file, 0o600)
    server.listen(16)
    server.settimeout(1.0)
    daemon = Daemon(sock_file)
    log.info("lsp-tool daemon %s listening on %s (pid %d)", __version__, sock_file, os.getpid())

    try:
        while not daemon._shutdown.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=_serve_connection, args=(daemon, conn), daemon=True).start()
    finally:
        daemon.manager.stop_all()
        server.close()
        try:
            os.unlink(sock_file)
        except OSError:
            pass
        log.info("daemon exited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
