#!/usr/bin/env bash
# Build the µTLX plugin (libutlx.so) against YOUR upstream Triton source
# build, with this repo's patch applied (upstream-op TMA wrappers + the
# warp-specialize register-restore pass).
#
# Required env:
#   TRITON_SOURCE_DIR  — checkout of triton-lang/triton, already installed
#                        editable with TRITON_EXT_ENABLED=ON
#   TRITON_BUILD_DIR   — its cmake build tree, e.g.
#                        $TRITON_SOURCE_DIR/build/cmake.linux-<arch>-cpython-3.12
#   LLVM_INSTALL_DIR   — the LLVM that triton build downloaded, e.g.
#                        ~/.triton/llvm/llvm-<hash>-<platform>
# Optional:
#   TRITON_EXT_DIR     — triton-ext checkout (default: ./third_party/triton-ext,
#                        cloned automatically)
#
# Output: ./lib/libutlx.so
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${TRITON_SOURCE_DIR:?set TRITON_SOURCE_DIR}"
: "${TRITON_BUILD_DIR:?set TRITON_BUILD_DIR}"
: "${LLVM_INSTALL_DIR:?set LLVM_INSTALL_DIR}"
EXT="${TRITON_EXT_DIR:-$ROOT/third_party/triton-ext}"
BUILD="$ROOT/build/triton-ext"

if [ ! -d "$EXT" ]; then
  mkdir -p "$(dirname "$EXT")"
  git clone https://github.com/triton-lang/triton-ext "$EXT"
fi
if ! grep -q utlx_async_TMA_load "$EXT/extensions/utlx/uTLXPlugin.cpp"; then
  git -C "$EXT" apply "$ROOT/patches/triton-ext-nvfp4.patch"
  echo "applied patches/triton-ext-nvfp4.patch"
fi

LIT="$(command -v lit || true)"
[ -z "$LIT" ] && LIT="$(dirname "$(command -v python3)")/lit" && [ -x "$LIT" ] || true
if [ -z "$LIT" ] || [ ! -x "$LIT" ]; then
  echo "need 'lit' (LLVM test runner) for triton-ext's cmake: pip install lit" >&2
  exit 1
fi
cmake -S "$EXT" -B "$BUILD" -G Ninja \
  -DTRITON_SOURCE_DIR="$TRITON_SOURCE_DIR" \
  -DTRITON_BUILD_DIR="$TRITON_BUILD_DIR" \
  -DLLVM_INSTALL_DIR="$LLVM_INSTALL_DIR" \
  -DLLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir" \
  -DFILECHECK_PATH="$LLVM_INSTALL_DIR/bin/FileCheck" \
  ${LIT:+-DLLVM_EXTERNAL_LIT="$LIT"}
cmake --build "$BUILD" --target utlx -j"$(nproc)"

mkdir -p "$ROOT/lib"
cp "$BUILD/lib/libutlx.so" "$ROOT/lib/libutlx.so"
for op in utlx_async_TMA_load utlx_async_TMA_store utlx_set_ws_requested_regs; do
  grep -aq "$op" "$ROOT/lib/libutlx.so" || { echo "missing $op in plugin" >&2; exit 1; }
done
echo "OK: $ROOT/lib/libutlx.so"
