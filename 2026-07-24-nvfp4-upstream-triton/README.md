# NVFP4 blockscaled GEMM on unforked upstream Triton via plugin extensions (GB10)

**Date:** 2026-07-24 (packaged end-state of NVFP4/TLX work running 2026-04 → 2026-07) ·
**Machine:** DGX Spark — NVIDIA GB10, sm_121, 48 SMs, CUDA 13, upstream Triton `release/3.8.x`
source-built with `TRITON_EXT_ENABLED=ON`

## Brief

Can the Colfax CUTLASS SM12x NVFP4 recipe (example 79b: TMA producer, warp-specialised
blockscaled mainloop, register reallocation) be expressed in Triton TLX ops and run on
*unforked upstream Triton* through the plugin-extension ABI, without Meta's fbtriton fork?

The kernel is a persistent, TMA-fed, K-split, warp-specialised NVFP4 (FP4 × UE4M3 block
scales) GEMM. Everything fork-only rides `libutlx.so` (triton-ext plus a small patch), the
`triton-utlx` DSL fetched from the PyPI wheel, and a runtime patch module.

## Headline results

- Bit-exact against plain `tl.dot_scaled`, and at parity (within thermal noise) with the
  same kernel compiled on the fbtriton fork.
- Same-session GB10 numbers: **264 TF** at 2048³ (88% of CUTLASS 79b), **315 TF** at
  4096³ (81%), **219 TF** at 8192×8192×4096 (59%); cuBLAS NVFP4 (`torch._scaled_mm`)
  298 / 348 / 327 on the same inputs. Run-to-run drift of a few percent with thermal
  state (262–334 TF seen at 4096³ across sessions).
- Upstream already ships the sm_12x `mxf4nvf4` MMA lowering, `ttg.warp_specialize` with
  `setmaxnreg`, TMA ops and the plugin ABI. Three gaps had to be bridged, each a
  candidate upstream change: no `with`-statement dispatch hook in the code generator;
  TMA-descriptor / warp-specialise / memdesc-slice builder methods bound only on the
  Gluon builder; `OptimizePartitionWarps` overwriting explicit per-partition register
  requests.
- Register economy *is* the schedule: launch `maxnreg=168`, producer 24 / consumers
  232–248 registers per thread, zero spills, with a plugin pass re-stamping the requests
  after upstream's optimiser.
- The remaining ~17% gap to CUTLASS at 4096³ is the operand-B shared-memory restage
  that CUTLASS's co-designed swizzle avoids; wide-N shapes collapse to ~97 TF from
  tile-order / persistent-scheduler tuning, identically on the fork.

## Contents

| Path | What |
|---|---|
| `REPORT.md` | Results table, architecture (what runs where), kernel structure, setup, why each patch exists, known limitations |
| `LABNOTES.md` | Where this sits in the NVFP4/TLX sequence and what was recorded when |
| `code/` | Kernel, runtime patch module, triton-ext patch, DSL fetch and plugin build scripts, `verify.py`, `bench.py`; `code/README.md` has the four-step setup |
| `background/` | The TLX / plugin-ABI investigation this builds on, snapshotted from `ianbarber/tritonext` @ `2d0435d` (2026-07-13): writeup, how-to, deep-dive report, op-by-op inventory |

The plugin build, fetched DSL and the upstream Triton checkout live outside the tree
(`code/.gitignore`). Imported from `ianbarber/nvfp4-triton-extensions` with history.
