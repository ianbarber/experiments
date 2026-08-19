# CUTLASS 4.7 CuTe DSL vs TLX on GB10 (DGX Spark, sm_121)

Hands-on comparison of the two new CUTLASS 4.7 CuTe DSL features — **Task Scheduling (TS)**
and **Primitives** — against Meta's **TLX** (`tlx` branch of facebookexperimental/triton)
`async_task` warp-specialization model. Everything here was written for and verified on
this machine's NVIDIA GB10 (sm_121a: has TMA/mbarriers/mma.sync, no tcgen05/TMEM/wgmma).

## Environments

| | Path | Notes |
|---|---|---|
| CuTe DSL venv | `/home/ianbarber/Projects/cute/.venv-cutedsl` | `nvidia-cutlass-dsl==4.7.0`, torch 2.13+cu130, `apache-tvm-ffi` (required by primitives examples). Arch auto-detects `sm_121a`. |
| TLX venv | `/home/ianbarber/Projects/cute/.venv-tlx` | Editable build of `../tlx-triton` (Triton 3.3 base). Two local fixes applied for sm_121a: `python/triton/backends/nvidia/bin/ptxas` → symlink to CUDA 13 ptxas (orig saved as `ptxas-cuda128.orig`), and `ptx_get_version()` patched to map CUDA 13 → PTX ISA 8.8. Never `pip install torch` here again (clobbers the editable triton). |
| CUTLASS source | `/home/ianbarber/Projects/cute/cutlass` | tag v4.7.0 |
| TLX source | `/home/ianbarber/Projects/cute/tlx-triton` | branch `tlx` @ 927d1e04 |

## The head-to-head kernels

Matched warp-specialized pipelined fp16 GEMMs (fp16 in, fp32 accumulate, 128×128×64 tiles,
TMA producer → SMEM ring → synchronous-tensor-core consumer):

- `ts/gemm_ws_ts.py` — CuTe DSL **Task Scheduling** GEMM (LoadTask 1 warp + MmaTask 4 warps,
  3-stage `PipelineTmaAsync` ring, `mma.sync` m16n8k16 via CuTe TiledMMA). The first TS GEMM
  for the sm_12x class (all shipped TS GEMM tutorials are tcgen05/sm_100-only).
  Run: `cd ts && ../../.venv-cutedsl/bin/python gemm_ws_ts.py [--bench]`
- `tlx/gemm_ws_tlx.py` — **TLX `async_task`** GEMM (1-warp TMA producer trunk +
  `async_task(num_warps=4, replicate=2)` consumers, 2-stage ring, `tl.dot` from TLX smem;
  `tlx.async_dot` is unusable on sm_121 — it emits tcgen05).
  Run: `cd ../tlx-triton && ../.venv-tlx/bin/python ../comparison/tlx/gemm_ws_tlx.py`

Median TFLOP/s (100 iters, torch.cuda.Event), same box, same protocol:

| M=N=K | TS GEMM | TLX GEMM | plain Triton | torch.matmul (cuBLAS) |
|---|---|---|---|---|
| 1024 | 38.0 | 56.4 | 50.9 | 59.9–63.2 |
| 2048 | 67.9 | 82.7 | 77.2 | 89.7–91.6 |
| 4096 | 84.2 | 88.4 | 82.1 | 86.9–92.8 |

## The safety experiment (the point of the comparison)

Same class of bug injected into both kernels — a consumer that doesn't release a ring slot:

- `ts/gemm_ws_ts_broken1.py` (release removed) and `ts/gemm_ws_ts_broken2.py` (double-wait
  credit deadlock): **both rejected at `cute.compile` time** by TS's credit-model schedule
  simulator — `ValueError: SmemAB ConsumerWait 0 is blocked!` — before any launch.
  Captures: `ts/broken_output.txt`; the good kernel's static analysis (schedule table,
  register/SMEM budgets, exhaustive BFS `Result: SAFE`) is in `ts/compile_diagnostics.txt`.
- `tlx/gemm_ws_tlx_broken.py` (one `barrier_arrive` on `empty_b[0]` skipped): **compiles
  with zero warnings**. At K=128 (ring never wraps) it runs and produces *correct results*
  (fully latent bug); at K=512 it deadlocks the GPU until killed (`timeout` exit 124).

## Primitives demos (`primitives/`)

- `three_ways.py` — one warp butterfly reduction written at all three escape-hatch tiers:
  `prims.shfl_sync` (typed wrapper) / `prims.dialect.shfl_sync` (raw NVVM) /
  `prims.inline_ptx` ("the old way") — identical results on GPU.
- `hazard_demo.py` + `hazard_output.txt` — the compile-time NVVM hazard checker firing on
  real bugs: multi-thread arrive on count=1 mbarrier (**hard error** C3/C4, no flag needed),
  init/arrive count mismatch (warning C3), `expect_tx` with no completion source (warning
  C5, would hang at runtime), nested `elect_sync` (error C16), plus a ptxas local-memory
  remark with Python source frames. Enable warnings: `CUTE_DSL_COMPILER_OPT="warnings{nvvm,ptx}"`.
- `tile_mma_prims.py` — pure-SIMT "CUDA-in-Python" single-tile 64×64×64 fp16 GEMM:
  TMA + mbarrier by hand → `prims.ldmatrix` → `prims.mma_sync` (m16n8k16) → per-lane
  fragment stores. Max |err| 5.7e-6 vs fp64 reference.

Per-directory `NOTES.md` files have full run instructions, tuning data, and findings.
