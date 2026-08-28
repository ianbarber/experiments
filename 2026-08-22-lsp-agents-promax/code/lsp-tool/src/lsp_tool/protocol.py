"""Shared client/daemon protocol helpers: socket path resolution and JSON-lines transport."""

from __future__ import annotations

import json
import os
import socket

DEFAULT_SOCKET = "/tmp/lsp-tool.sock"


class SocketPathError(Exception):
    pass


def socket_path() -> str:
    path = os.environ.get("LSP_TOOL_SOCKET", DEFAULT_SOCKET)
    if len(path.encode("utf-8")) > 100:  # AF_UNIX limit is ~108 bytes
        raise SocketPathError(
            f"socket path too long for AF_UNIX ({len(path)} chars): {path} "
            "-- set LSP_TOOL_SOCKET to a shorter path"
        )
    return path


def log_path(sock: str) -> str:
    return sock + ".log"


class DaemonUnavailable(Exception):
    """Raised when the daemon socket cannot be reached."""


def send_request(sock_path: str, payload: dict, timeout: float | None = 300.0) -> dict:
    """Send one JSON request line over the Unix socket and read one JSON response line."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(5.0)
        try:
            s.connect(sock_path)
        except (OSError, socket.timeout) as e:
            raise DaemonUnavailable(str(e)) from e
        s.settimeout(timeout)
        f = s.makefile("rwb")
        f.write(json.dumps(payload).encode("utf-8") + b"\n")
        f.flush()
        line = f.readline()
        if not line:
            raise DaemonUnavailable("daemon closed connection without responding")
        return json.loads(line.decode("utf-8"))
    finally:
        s.close()
