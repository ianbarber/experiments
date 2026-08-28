#!/usr/bin/env bash
# Install lsp-tool into its own venv and put `lsp` on PATH.
#
# Intended to run inside Docker images at build time (arch-agnostic; tested on
# aarch64 and targeted at amd64 SWE-bench-style containers). Requires network
# access to PyPI while it runs.
#
#   LSP_TOOL_HOME    install prefix (default /opt/lsp-tool)
#   LSP_TOOL_PYTHON  python >=3.11 interpreter to use (default: autodetect)
#   BIN_DIR          where to symlink `lsp` (default /usr/local/bin)
#
# The runtime dependency chain (serena-agent, which ships solidlsp) requires
# Python >=3.11. If no suitable interpreter is on PATH but `uv` is available,
# a managed CPython is downloaded instead.
set -euo pipefail

LSP_TOOL_HOME="${LSP_TOOL_HOME:-/opt/lsp-tool}"
BIN_DIR="${BIN_DIR:-/usr/local/bin}"
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"

# Versions verified to work together (see README maintainer notes).
PYREFLY_SPEC="${PYREFLY_SPEC:-pyrefly==1.2.0}"
BASEDPYRIGHT_SPEC="${BASEDPYRIGHT_SPEC:-basedpyright==1.39.10}"

log() { echo "[lsp-tool install] $*"; }
die() { echo "[lsp-tool install] ERROR: $*" >&2; exit 1; }

version_ok() {  # true if "$1" is a python >= 3.11
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null
}

find_python() {
    if [ -n "${LSP_TOOL_PYTHON:-}" ]; then
        version_ok "$LSP_TOOL_PYTHON" || die "LSP_TOOL_PYTHON=$LSP_TOOL_PYTHON is not Python >=3.11"
        echo "$LSP_TOOL_PYTHON"
        return
    fi
    local cand
    for cand in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && version_ok "$cand"; then
            command -v "$cand"
            return
        fi
    done
    echo ""
}

PYTHON="$(find_python)"
UV="$(command -v uv || true)"

mkdir -p "$(dirname "$LSP_TOOL_HOME")"

if [ -n "$PYTHON" ]; then
    log "using $PYTHON ($("$PYTHON" --version 2>&1))"
    if ! "$PYTHON" -m venv "$LSP_TOOL_HOME" 2>/dev/null; then
        # Debian/Ubuntu pythons without python3-venv lack ensurepip
        if [ -n "$UV" ]; then
            log "python -m venv failed; falling back to uv"
            rm -rf "$LSP_TOOL_HOME"   # a failed python -m venv leaves a partial dir uv refuses to reuse
            "$UV" venv --python 3.12 "$LSP_TOOL_HOME"
        else
            die "python -m venv failed (missing python3-venv?) and uv is not available"
        fi
    fi
elif [ -n "$UV" ]; then
    log "no python >=3.11 on PATH; using uv-managed CPython 3.12"
    "$UV" venv --python 3.12 "$LSP_TOOL_HOME"
else
    die "need Python >=3.11 (or uv, https://docs.astral.sh/uv/) to install lsp-tool"
fi

VENV_PY="$LSP_TOOL_HOME/bin/python"
[ -x "$VENV_PY" ] || die "venv creation failed: $VENV_PY missing"

log "installing lsp-tool, $PYREFLY_SPEC, $BASEDPYRIGHT_SPEC"
if [ -n "$UV" ]; then
    "$UV" pip install -p "$VENV_PY" --no-cache "$PKG_DIR" "$PYREFLY_SPEC" "$BASEDPYRIGHT_SPEC"
else
    "$VENV_PY" -m pip install --no-cache-dir --upgrade pip >/dev/null
    "$VENV_PY" -m pip install --no-cache-dir "$PKG_DIR" "$PYREFLY_SPEC" "$BASEDPYRIGHT_SPEC"
fi

mkdir -p "$BIN_DIR"
ln -sf "$LSP_TOOL_HOME/bin/lsp" "$BIN_DIR/lsp"

"$BIN_DIR/lsp" --help >/dev/null || die "smoke test failed"
log "installed: $("$LSP_TOOL_HOME/bin/python" -c 'import lsp_tool; print("lsp-tool", lsp_tool.__version__)') -> $BIN_DIR/lsp"
