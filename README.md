# NVFP4 blockscaled GEMM on upstream Triton via plugin extensions

A warp-specialized, TMA-fed, K-split NVFP4 (FP4 + UE4M3 block scales) GEMM
for consumer Blackwell (GB10 / sm_121), written in Triton TLX ops and running
on **unforked upstream Triton** through the
[Triton Plugin Extensions](https://pytorch.org/blog/triton-plugin-extensions-enabling-tlx-and-custom-compiler-passes-out-of-the-box/)
architecture. It replicates the schedule of the Colfax CUTLASS recipe —
["NVFP4 Blockscaled GEMM on NVIDIA RTX Pro Blackwell GPUs (SM12x)"](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)
(CUTLASS example 79b: TMA + warp-specialized blockscaled mainloop + register
reallocation) — without Meta's experimental Triton fork.

Measured on GB10 (sm_121, idle GPU), bit-exact vs plain `tl.dot_scaled`:

| shape | this kernel | CUTLASS 79b | ratio |
|---|--:|--:|--:|
| 2048³ | 262 TF | 297 | 88% |
| 4096³ | 307–332 TF | 402 | 76–83% |
| 8192×8192×4096 | 218 TF | 360 | 60% |

That is at/above the cuBLAS NVFP4 band (~325–347 at 4096³) and at parity or
better with the same kernel compiled on the fbtriton fork — every fork-only
ingredient (warp-spec, TMA descriptor ops, register realloc) rides the
plugin/patch stack instead.

## Architecture (what runs where)

```
upstream triton (source, TRITON_EXT_ENABLED=ON)      ← never modified
  └─ libutlx.so           out-of-tree plugin (triton-ext + small patch):
                          TLX dialect/ops/passes, upstream-op TMA wrappers,
                          warp-spec register-restore pass
  └─ utlx_py/             triton-utlx wheel Python (fetched + 2 tiny patches)
  └─ tlx_upstream_patch   runtime patches: visit_With dispatch for
                          tlx.async_tasks, GluonOpBuilder swap (exposes the
                          gluon-only create_* surface to @triton.jit codegen),
                          WS codegen for upstream's ttg.warp_specialize,
                          explicit-layout local_load, + compat bridges
  └─ kernel/nvfp4_ws_ksplit.py   the GEMM (structure below)
```

Kernel structure (mirrors the CUTLASS mainloop):

- **Persistent scheduler**: grid = #SMs, each CTA loops over tiles.
- **TMA producer partition** (4 warps, `registers=24`): descriptor loads of
  A, B halves, and both scale tensors into a double-buffered NVMMA-swizzled
  SMEM ring, `full`/`empty` mbarrier handshake.
- **Two full-width consumers** split over N (4 warps each, 232 registers):
  one lives in the *default* warp-spec region (upstream's register
  allocator grants the default the leftover budget — put the heavy task
  there), one is a worker partition with `registers=232`.
- **K-split**: two 128-deep `tl.dot_scaled` calls per ring slot keep
  operand liveness low enough that 8 MMA warps never spill.
- **Explicit layouts**: B loads pinned to a vectorized blocked layout
  (`tlx.local_load_blocked`) — otherwise upstream anchors the staging load
  on a scalar layout (hundreds of `ld.shared.b8`).
- **Register economy**: launch `maxnreg=168` (64K regfile / 384 threads);
  the plugin pass re-stamps the kernel's register requests after upstream's
  `OptimizePartitionWarps` would overwrite them.
  Result: 248/232/24 per-thread across the three roles, zero spills.

## Setup

Prereqs: CUDA 13 toolkit, PyTorch (cu13x build), `pip install cmake ninja lit`, an sm_12x GPU.
(Block sizes are tuned for GB10/sm_121; other sm_12x parts will want a
config sweep.)

**1. Build upstream Triton with the plugin ABI enabled** (once):

```bash
git clone https://github.com/triton-lang/triton && cd triton
git checkout release/3.8.x           # or main; both validated
pip install -r python/requirements.txt
TRITON_EXT_ENABLED=ON pip install -e . --no-build-isolation
```

**2. Build the plugin** (clones triton-ext, applies `patches/triton-ext-nvfp4.patch`):

```bash
export TRITON_SOURCE_DIR=/path/to/triton
export TRITON_BUILD_DIR=$TRITON_SOURCE_DIR/build/cmake.linux-<arch>-cpython-3.12
export LLVM_INSTALL_DIR=$HOME/.triton/llvm/llvm-<hash>-<platform>   # created by step 1
scripts/build_plugin.sh              # -> lib/libutlx.so
```

**3. Fetch the TLX DSL** (pure Python from the `triton-utlx` wheel, patched):

```bash
python scripts/fetch_utlx.py         # -> utlx_py/
```

**4. Run:**

```bash
. scripts/env.sh
python verify.py                     # bit-exact vs plain dot_scaled
python bench.py                      # shape table
python bench.py 4096 4096 4096      # single shape
```

Import order matters in your own drivers: `import utlx_plugin` (registers
`triton.language.extra.tlx` and the compile-pipeline hook), then
`import tlx_upstream_patch` (applies the warp-spec/codegen patches), then
the kernel.

## Why the patches exist

Upstream Triton ships almost everything this kernel needs — the native
`mxf4nvf4.block_scale` MMA lowering for sm_12x, `ttg.warp_specialize` with
full lowering including `setmaxnreg`, TMA ops, and the plugin ABI itself.
Three gaps remain, each bridged here and each a candidate upstream change:

1. No `with`-statement dispatch hook in `CodeGenerator` — bridged by a
   monkeypatch (approach due to [wychi/wheels](https://github.com/wychi/wheels)).
2. Several builder methods (TMA descriptor copies, `warp_specialize`,
   memdesc slicing) are bound only on the Gluon builder — bridged by the
   builder swap plus plugin-owned op wrappers.
3. `OptimizePartitionWarps` overwrites explicit warp-spec register requests
   with hardcoded estimates — bridged by a plugin pass that re-stamps them.

## Known limitations

- Wide-N shapes (e.g. 4096×12288×4096) collapse to ~97 TF — a persistent-
  scheduler/tile-order tuning issue in the kernel, identical on the fork.
- The remaining gap to CUTLASS (~17% at 4096³) is dominated by the
  operand-B shared-memory restage that CUTLASS's co-designed B swizzle
  avoids; closing it needs deeper layout work than a kernel can express.
- sm_121-tuned constants (`BM=BN=128, BK=256, NS=2`); `NS=3` exceeds the
  99KB SMEM/CTA limit.
