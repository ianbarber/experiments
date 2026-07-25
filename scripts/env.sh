# source this from the repo root:  . scripts/env.sh
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRITON_PLUGIN_PATHS="$_ROOT/lib/libutlx.so"
export TRITON_PLUGIN_VERSION_CHECK=off
export PYTHONPATH="$_ROOT/utlx_py:$_ROOT/kernel${PYTHONPATH:+:$PYTHONPATH}"
