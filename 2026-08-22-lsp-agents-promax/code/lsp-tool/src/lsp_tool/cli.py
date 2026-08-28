"""`lsp` — thin client CLI that talks to the lsp-tool daemon (auto-starting it)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from .languages import LanguageResolutionError, resolve_ls_id
from .protocol import DaemonUnavailable, SocketPathError, log_path, send_request, socket_path

AUTO_START_TIMEOUT = 20.0


def _fail(message: str) -> int:
    print(f"lsp: {message}", file=sys.stderr)
    return 1


def parse_position(arg: str) -> tuple[str, int, int | None]:
    """Parse '<file>:<line>[:<col>]' (1-based line/col)."""
    parts = arg.rsplit(":", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], int(parts[1]), int(parts[2])
    parts = arg.rsplit(":", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1]), None
    raise ValueError(f"expected <file>:<line>[:<col>], got '{arg}'")


def _workspace(args) -> str:
    ws = getattr(args, "workspace", None) or os.environ.get("LSP_TOOL_WORKSPACE") or os.getcwd()
    return os.path.realpath(ws)


def _request_timeout() -> float:
    try:
        return float(os.environ.get("LSP_TOOL_TIMEOUT", "240"))
    except ValueError:
        return 240.0


def _spawn_daemon(sock: str) -> None:
    logfile = open(log_path(sock), "ab")
    subprocess.Popen(
        [sys.executable, "-m", "lsp_tool.daemon", "--socket", sock],
        stdin=subprocess.DEVNULL,
        stdout=logfile,
        stderr=logfile,
        start_new_session=True,
        close_fds=True,
    )
    logfile.close()


def ensure_daemon(sock: str) -> None:
    """Connect to the daemon, auto-starting it if it is not running."""
    try:
        send_request(sock, {"op": "ping"}, timeout=5)
        return
    except DaemonUnavailable:
        pass
    # remove a stale socket file so bind succeeds
    if os.path.exists(sock):
        try:
            os.unlink(sock)
        except OSError:
            pass
    _spawn_daemon(sock)
    deadline = time.monotonic() + AUTO_START_TIMEOUT
    while time.monotonic() < deadline:
        try:
            send_request(sock, {"op": "ping"}, timeout=5)
            return
        except DaemonUnavailable:
            time.sleep(0.1)
    raise DaemonUnavailable(
        f"could not start daemon on {sock} (see {log_path(sock)})"
    )


def _run_op(op: str, args, request_args: dict, file_for_language: str | None) -> int:
    sock = socket_path()
    workspace = _workspace(args)
    try:
        ls_id = resolve_ls_id(getattr(args, "language", None), file_for_language, workspace)
    except LanguageResolutionError as e:
        return _fail(str(e))
    wait = _request_timeout()
    payload = {"op": op, "workspace": workspace, "ls_id": ls_id, "args": request_args, "wait": wait}
    resp = None
    last_error = None
    for attempt in range(2):  # one retry: the daemon may have just been stopped/replaced
        try:
            ensure_daemon(sock)
            resp = send_request(sock, payload, timeout=wait + 60)
            break
        except DaemonUnavailable as e:
            last_error = f"cannot reach daemon: {e}"
        except OSError as e:
            last_error = f"daemon communication error: {e}"
        time.sleep(0.2)
    if resp is None:
        return _fail(last_error or "unknown daemon error")
    if not resp.get("ok"):
        return _fail(resp.get("error", "unknown error"))
    text = resp.get("text", "")
    if text:
        print(text)
    return 0


def _abs_file(args_file: str) -> str:
    return os.path.realpath(args_file)


# ---------------------------------------------------------------------------
# subcommand implementations
# ---------------------------------------------------------------------------


def cmd_position_op(op: str, args) -> int:
    try:
        file_part, line, col = parse_position(args.position)
    except ValueError as e:
        return _fail(str(e))
    request_args = {"file": _abs_file(file_part), "line": line, "col": col}
    if op == "rename":
        request_args["new_name"] = args.new_name
    return _run_op(op, args, request_args, file_part)


def cmd_file_op(op: str, args) -> int:
    return _run_op(op, args, {"file": _abs_file(args.file)}, args.file)


def cmd_sym(args) -> int:
    return _run_op("sym", args, {"name": args.name}, None)


def cmd_daemon(args) -> int:
    sock = socket_path()
    action = args.action
    if action == "stop":
        try:
            resp = send_request(sock, {"op": "shutdown"}, timeout=10)
        except (DaemonUnavailable, OSError):
            print("daemon not running")
            return 0
        print(resp.get("text", "stopped"))
        return 0
    if action == "status":
        try:
            resp = send_request(sock, {"op": "status"}, timeout=10)
        except (DaemonUnavailable, OSError):
            return _fail("daemon not running")
        print(resp.get("text", ""))
        return 0
    if action == "start":
        workspace = _workspace(args)
        try:
            ls_id = resolve_ls_id(getattr(args, "language", None), None, workspace)
        except LanguageResolutionError as e:
            return _fail(str(e))
        try:
            ensure_daemon(sock)
            wait = _request_timeout()
            resp = send_request(
                sock,
                {
                    "op": "warm",
                    "workspace": workspace,
                    "ls_id": ls_id,
                    "args": {"wait_ready": not args.no_wait},
                    "wait": wait,
                },
                timeout=wait + 60,
            )
        except DaemonUnavailable as e:
            return _fail(str(e))
        if not resp.get("ok"):
            return _fail(resp.get("error", "unknown error"))
        print(resp.get("text", ""))
        return 0
    return _fail(f"unknown daemon action '{action}'")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", "-w", help="workspace root (default: $LSP_TOOL_WORKSPACE or CWD)")
    p.add_argument(
        "--language", "-l",
        help="language: python|typescript|java|go|rust|cpp (default: $LSP_TOOL_LANGUAGE or inferred)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsp",
        description="Language-server queries for coding agents. "
                    "Positions are 1-based <file>:<line>[:<col>]; when <col> is omitted, "
                    "the symbol declared on that line is used.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("def", help="go to definition of the symbol at a position")
    p.add_argument("position", metavar="FILE:LINE[:COL]")
    _add_common(p)
    p.set_defaults(func=lambda a: cmd_position_op("def", a))

    p = sub.add_parser("refs", help="find all references to the symbol at a position (workspace-wide)")
    p.add_argument("position", metavar="FILE:LINE[:COL]")
    _add_common(p)
    p.set_defaults(func=lambda a: cmd_position_op("refs", a))

    p = sub.add_parser("hover", help="type/signature/doc info for the symbol at a position")
    p.add_argument("position", metavar="FILE:LINE[:COL]")
    _add_common(p)
    p.set_defaults(func=lambda a: cmd_position_op("hover", a))

    p = sub.add_parser("sym", help="search workspace symbols by name")
    p.add_argument("name")
    _add_common(p)
    p.set_defaults(func=cmd_sym)

    p = sub.add_parser("outline", help="document symbols (outline) of a file")
    p.add_argument("file")
    _add_common(p)
    p.set_defaults(func=lambda a: cmd_file_op("outline", a))

    p = sub.add_parser("diag", help="diagnostics (errors/warnings) for a file")
    p.add_argument("file")
    _add_common(p)
    p.set_defaults(func=lambda a: cmd_file_op("diag", a))

    p = sub.add_parser(
        "rename",
        help="show the edits a rename WOULD make (dry-run only; never applies changes)",
    )
    p.add_argument("position", metavar="FILE:LINE[:COL]")
    p.add_argument("new_name")
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="always on; rename never modifies files")
    _add_common(p)
    p.set_defaults(func=lambda a: cmd_position_op("rename", a))

    p = sub.add_parser("daemon", help="manage the background daemon")
    p.add_argument("action", choices=["start", "stop", "status"])
    p.add_argument("--no-wait", action="store_true",
                   help="with start: do not wait for the language server to finish indexing")
    _add_common(p)
    p.set_defaults(func=cmd_daemon)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SocketPathError as e:
        return _fail(str(e))
    except KeyboardInterrupt:
        return _fail("interrupted")


if __name__ == "__main__":
    raise SystemExit(main())
