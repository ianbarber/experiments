#!/usr/bin/env bash
# End-to-end test for lsp-tool.
#
# Creates a scratch Python repo and a fresh venv under /tmp, installs lsp-tool +
# pyrefly + basedpyright, then exercises every subcommand against both Python
# language servers, asserting content-enriched output and warm-daemon latency.
#
# Usage: bash tests/e2e.sh
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$HERE")"

WORK="$(mktemp -d /tmp/lsp-e2e.XXXXXX)"
REPO="$WORK/repo"
VENV="$WORK/venv"
export LSP_TOOL_SOCKET="$WORK/lsp.sock"
unset LSP_TOOL_WORKSPACE LSP_TOOL_LANGUAGE LSP_TOOL_PYTHON_SERVER 2>/dev/null || true

PASS=0
FAIL=0
CUR_SERVER=""

cleanup() {
    "$VENV/bin/lsp" daemon stop >/dev/null 2>&1 || true
    rm -rf "$WORK"
}
trap cleanup EXIT

say()  { printf '\n--- %s\n' "$*"; }
ok()   { PASS=$((PASS + 1)); printf 'PASS [%s] %s\n' "$CUR_SERVER" "$*"; }
bad()  { FAIL=$((FAIL + 1)); printf 'FAIL [%s] %s\n' "$CUR_SERVER" "$*"; }

# check <description> <expected-exit> <grep-pattern...> -- output must contain every pattern
check() {
    local desc="$1" want_exit="$2"; shift 2
    local out rc
    out="$("${CMD[@]}" 2>&1)"; rc=$?
    if [ "$rc" != "$want_exit" ]; then
        bad "$desc: exit $rc (wanted $want_exit); output: $(echo "$out" | head -3)"
        return
    fi
    local pat
    for pat in "$@"; do
        if ! echo "$out" | grep -q -- "$pat"; then
            bad "$desc: output missing '$pat'; output: $(echo "$out" | head -5)"
            return
        fi
    done
    ok "$desc"
}

# ---------------------------------------------------------------------------
# scratch repo: class User defined in models.py, used in 3 other files
# ---------------------------------------------------------------------------
mkdir -p "$REPO/app"
cat > "$REPO/pyproject.toml" <<'EOF'
[project]
name = "testrepo"
version = "0.0.1"
EOF
: > "$REPO/app/__init__.py"
cat > "$REPO/app/models.py" <<'EOF'
"""Domain models."""


class User:
    """A user of the system."""

    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email

    def greet(self) -> str:
        return f"Hello, {self.name}!"


def make_user(name: str, email: str) -> User:
    return User(name, email)
EOF
cat > "$REPO/app/services.py" <<'EOF'
"""Service layer."""

from app.models import User


def notify(user: User) -> str:
    return f"notify {user.email}: {user.greet()}"
EOF
cat > "$REPO/app/handlers.py" <<'EOF'
"""Request handlers."""

from app.models import User, make_user


def signup(name: str, email: str) -> User:
    user = make_user(name, email)
    return user
EOF
cat > "$REPO/app/main.py" <<'EOF'
"""Entry point."""

from app.handlers import signup
from app.models import User


def run() -> None:
    user: User = signup("ada", "ada@example.com")
    print(user.greet())
EOF
cat > "$REPO/app/broken.py" <<'EOF'
"""Has errors."""

from app.models import User


def bad() -> int:
    u = User("x")
    return u.nonexistent_method() + undefined_name
EOF

# ---------------------------------------------------------------------------
# venv + install
# ---------------------------------------------------------------------------
say "creating venv and installing lsp-tool + pyrefly + basedpyright"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
if [ -x "$UV" ]; then
    "$UV" venv "$VENV" -q || { echo "uv venv failed"; exit 1; }
    "$UV" pip install -p "$VENV/bin/python" -q "$PKG_DIR" pyrefly basedpyright \
        || { echo "install failed"; exit 1; }
else
    python3 -m venv "$VENV" || { echo "venv failed"; exit 1; }
    "$VENV/bin/pip" install -q "$PKG_DIR" pyrefly basedpyright || { echo "install failed"; exit 1; }
fi
LSP="$VENV/bin/lsp"

now_ms() { "$VENV/bin/python" -c 'import time; print(int(time.time()*1000))'; }

# ---------------------------------------------------------------------------
# the command suite, run once per python language server
# ---------------------------------------------------------------------------
run_suite() {
    CUR_SERVER="$1"
    export LSP_TOOL_PYTHON_SERVER="$CUR_SERVER"
    cd "$REPO"

    say "suite: $CUR_SERVER"

    "$LSP" daemon stop >/dev/null 2>&1 || true
    sleep 1

    local t0 t1
    t0=$(now_ms)
    if ! "$LSP" daemon start --language python; then
        bad "daemon start"
        return
    fi
    t1=$(now_ms)
    ok "daemon start + index warm-up in $((t1 - t0)) ms"

    CMD=("$LSP" def app/services.py:6:18)
    check "def User usage -> class definition" 0 "app/models.py:4" "class User" "^>"

    CMD=("$LSP" refs app/models.py:4)
    check "refs on class User finds all 3 usage files (enriched)" 0 \
        "app/services.py" "app/handlers.py" "app/main.py" \
        "from app.models import User" "def notify(user: User)" "^>"

    # warm-daemon latency: second invocation must complete in <1s
    "$LSP" refs app/models.py:4 >/dev/null 2>&1   # absorb one-time cross-file-ref wait
    t0=$(now_ms)
    "$LSP" refs app/models.py:4 >/dev/null 2>&1
    t1=$(now_ms)
    local dt=$((t1 - t0))
    if [ "$dt" -lt 1000 ]; then
        ok "warm refs invocation took ${dt} ms (<1000 ms)"
    else
        bad "warm refs invocation took ${dt} ms (>=1000 ms)"
    fi

    CMD=("$LSP" hover app/models.py:11:9)
    check "hover on greet" 0 "greet" "str"

    CMD=("$LSP" sym make_user)
    check "workspace symbol search" 0 "make_user" "app/models.py:15" "^>"

    CMD=("$LSP" outline app/models.py)
    check "outline" 0 "class User" "method greet" "function make_user"

    CMD=("$LSP" diag app/broken.py)
    check "diagnostics find undefined name" 0 "undefined_name" "error"

    CMD=("$LSP" diag app/services.py)
    check "diagnostics clean file" 0 "no diagnostics"

    CMD=("$LSP" rename app/models.py:4 Person --dry-run)
    check "rename dry-run shows edits in all files" 0 \
        "dry-run" "app/services.py" "app/handlers.py" "app/main.py" "+class Person"

    if grep -q "class User" "$REPO/app/models.py" && ! grep -rq "Person" "$REPO/app"; then
        ok "rename did not modify any files"
    else
        bad "rename modified files on disk!"
    fi

    CMD=("$LSP" def app/models.py:2)
    check "def on empty line fails cleanly" 1 "no definition found"

    CMD=("$LSP" daemon status)
    check "daemon status" 0 "ready"
}

run_suite pyrefly
run_suite basedpyright

"$LSP" daemon stop >/dev/null 2>&1 || true

say "results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
