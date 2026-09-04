# Lab notebook — NVFP4 GEMM on upstream Triton

This entry is the packaged end-state of a sequence that ran across several repos. The
detailed running logs live in those repos and were not copied; `background/` holds the
tritonext documents this depends on.

| When | Where | What |
|---|---|---|
| 2026-04-12 | `ianbarber/nvfp4` | NVFP4 deep-dive on non-power-of-2 block sizes for Blackwell tensor cores; Triton vs CUTLASS bench on GB10; SM121 native `dot_scaled` patch for Triton (214–238 TF, 72–74% of CUTLASS) |
| 2026-05-15 → 07-13 | `ianbarber/tritonext` | Skeptical investigation of TLX and the triton-ext plugin ABI: what runs on upstream, what needs the fork; fbtriton demos on a 5090; the Triton 3.8 / `triton-utlx` dual path (see `background/`) |
| 2026-06-26 → 07-13 | `ianbarber/colfax-nvfp4-sm12x` | Colfax SM12x recreation on GB10: cuBLAS / CUTLASS 79b validation, SASS + occupancy profiling, the fbtriton warp-specialised K-split kernel, the layout-compositionality gap analysis (`LOG.md`, 15 stages) |
| 2026-07-01 | `ianbarber/nvfp4-triton-extensibility` | Reframe: the performance artefact is a TLX *kernel*; the authored core lowering extension (packed-i32 dot operand) is real but perf-neutral for this schedule |
| 2026-07-24 | `ianbarber/nvfp4-triton-extensions` → this entry | The same schedule ported off the fork onto upstream `release/3.8.x` + `TRITON_EXT_ENABLED=ON` + patched triton-ext plugin + `triton-utlx` DSL; verified bit-exact; same-session comparison against CUTLASS 79b and cuBLAS |

## 2026-07-24

- Three commits on the day. The initial commit recorded 334 TF at 4096³ (~83% of
  CUTLASS 79b) and 262 TF at 2048³ (88%).
- The second commit dropped the "upstream notes" that duplicated the tritonext material
  (that material is now `background/`).
- The third rewrote the results table around a single back-to-back session (264 / 315 /
  219 TF) so all three columns share GPU thermal state, labelled CUTLASS 79b explicitly
  as the Colfax post's kernel, and recorded the 262–334 TF spread across sessions rather
  than picking the best run.
