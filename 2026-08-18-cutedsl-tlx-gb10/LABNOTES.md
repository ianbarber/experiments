# Lab notebook — CUTLASS 4.7 TS/Primitives vs TLX on GB10

Execution log for the experiment, in the order it happened. Work ran 2026-08-18,
~12:00–13:00 local, on the DGX Spark, orchestrated by Claude Code: an interactive
session did the scouting, planning, and verification, and fanned the heavy lifting out
to two batches of parallel agents (5 research/setup agents, then 3 implementation
agents). Times are approximate wall-clock.

## 1. Scouting (~12:00)

- `nvidia-smi`: NVIDIA GB10, driver 580.159.03, CUDA 13.0. `uname -m`: aarch64.
  Python 3.12.3, `uv` available. Identified the decisive constraint immediately: GB10
  is sm_121a — TMA/mbarriers/`mma.sync` yes; tcgen05/TMEM/wgmma no.
- Cloned CUTLASS at tag `v4.7.0`. Confirmed both features in the tree:
  `python/CuTeDSL/cutlass/experimental/task_scheduling/` (+ 7 tutorial dirs, Blackwell)
  and `.../experimental/primitives/` (+ per-instruction examples).
- Tutorial README's "First Path" showed copy/TMA/persistent examples that looked
  sm_121-safe, and GEMM tutorials that looked tcgen05-bound. Also noticed the README
  links TS FMHA / `blackwell_geforce` TS dirs — later confirmed those paths don't exist
  in the shipped tree.
- Web search located TLX: branch `tlx` of `facebookexperimental/triton`, plus the TLX
  paper (arXiv:2605.10905) and PyTorch blogs. Started the clone in the background.
- Checked PyPI: `nvidia_cutlass_dsl-4.7.0-py3-none-any.whl` exists → installable on
  aarch64.

## 2. Phase 1 — setup + research, 5 parallel agents (~12:05–12:35)

### 2a. CuteDSL setup (venv `.venv-cutedsl`)

- `uv pip install nvidia-cutlass-dsl==4.7.0 numpy typing_extensions torch apache-tvm-ffi`.
  Plain PyPI `torch` resolved to 2.13.0+cu130 with working CUDA on aarch64 — no special
  index needed. Arch auto-detected as sm_121a; no env vars required.
- **Failure → fix:** primitives examples died with "Install TVM FFI with
  `pip install apache-tvm-ffi`" — their compile path uses TVM FFI by default.
  Installing `apache-tvm-ffi==0.1.13.post3` fixed all of them.
- Verified working on GB10: TS tutorials 01 (grid-stride copy, TMA copy) and 03
  (persistent WorkQueue, varlen dynamic domain); primitives `cp_async_shared_global`,
  `mbarrier`, `elect_sync`, `tma/tma_load_store`, `tutorial/01_hello_world`; and the
  non-TS GeForce-Blackwell GEMM `cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py`
  (the key mma.sync reference for later).
- **Expected failure confirmed:** TS GEMM tutorial 02 allocates TMEM then dies at
  compile with `NVVM backend compilation failed … target architecture: sm_121a`.
  grep showed every TS GEMM tutorial (02, 04, 05, 06, 07b–d) is tcgen05-bound.
- **Bug found (not ours):** tutorial 07's only non-tcgen05 script fails at trace time
  with `TYPE_DYNAMIC_EXPR_UNSUPPORTED` — a 4.7.0 example/DSL incompatibility.

### 2b. TLX build (venv `.venv-tlx`)

- Clone finished; fork is Triton 3.3.0 base, LLVM tarball for ubuntu-arm64 downloaded
  fine. Build: `MAX_JOBS=20 uv pip install --no-build-isolation -e .` — 7m20s.
- **Gotcha:** installing torch *after* the editable triton uninstalls it (torch depends
  on PyPI triton). Order matters; re-running the editable install is seconds (cached).
- **Failure → fix #1:** bundled ptxas is CUDA-12.8-era → `ptxas fatal: Value 'sm_121a'
  is not defined for option 'gpu-name'`. Fixed by symlinking the backend's ptxas to
  `/usr/local/cuda/bin/ptxas` (original kept as `ptxas-cuda128.orig`).
- **Failure → fix #2:** with CUDA-13 ptxas, `ptx_get_version()` raised "Triton only
  support CUDA 10.0 or higher, but got CUDA version: 13.0". Patched to map CUDA 13 →
  PTX ISA 8.8 (equivalently `TRITON_MOCK_PTX_VERSION=12.9`).
- TLX unit suite on GB10 after fixes: warp-spec/barrier/TMA/cp.async tests **pass**
  (161 passed in the full run); the only real failures were 7 tcgen05/TMEM tests
  (GB10 lacks the hardware — TLX's `capability >= 10` guard wrongly includes cc12.1)
  and 68 OutOfResources cases from H100-sized smem parametrizations (limit here:
  101,376 B).
- Wrote and verified a minimal producer/consumer async_task kernel (cp.async →
  mbarrier handoff → compute); PTX confirmed `cp.async.cg`, `mbarrier.try_wait.parity`,
  `.target sm_121a`. Confirmed `tlx.async_dot` is unusable on sm_121 (takes the
  tcgen05 path, ptxas rejects).

### 2c. Deep reading (3 agents)

Parallel source-level reports on (i) the TS framework (task.py/task_manager.py/
resources.py/schedule_builder.py/exhaustive_checker.py + ts_general docs + tutorials),
(ii) the Primitives package (nvvm_wrapper.py 11k lines, descriptors, hazard-diagnostics
plumbing, all examples), (iii) TLX (async_task frontend, barrier/mem/mma ops, lowering
through `ttg.warp_specialize`, paper + blogs). Key findings that shaped phase 2:

- TS: all validation happens at `TaskManager(...)` construction (structural checks →
  credit-model simulator → exhaustive BFS with a hardware-aware race model); lowers
  onto existing `cutlass.pipeline` classes; sm_121 path = `AsyncAsync`/`TmaAsync`
  pipelines with mma.sync inside consumer work; default SMEM capacity assumes 232 KB
  datacenter parts → must pass `smem_capacity_bytes=101376` explicitly.
- TLX: context managers are pure AST markers; *no* sync validation of any kind
  (`# TODO. add validator logics`); phases/counts fully manual; flagship Blackwell
  tutorial passes a silently-dropped `num_regs` kwarg (real name: `registers`).
- Primitives: three-tier ladder; hazard checker catalog C3–C16 rendered with
  Rust-style source frames; `elect_sync` co-designed as the checker-visible gate.

## 3. Phase 2 — implementation, 3 parallel agents (~12:35–13:00)

### 3a. TS GEMM (`code/ts/gemm_ws_ts.py`)

Built incrementally: ran 02_copy_tma as smoke test → TMA-ring + copy-consumer kernel
(bit-exact) → swapped consumer to TiledMMA mma.sync (recipe from the blackwell_geforce
dense GEMM: `MmaF16BF16Op` m16n8k16, 2×2 atom layout, ldmatrix copies) → verified vs
torch at 512–4096 + a non-square shape → benchmarked. Walls hit and workarounds
(documented in `code/ts/NOTES.md`):

- Resource dataclasses can't gain attributes inside task warp-gate regions ("has extra
  field") → keep only the accumulator as persistent state, allocated at kernel-top
  trace; recompute TiledMMA/ldmatrix partition views inside each work body.
- MLIR-valued composed layouts stored on a resource caused IR dominance failures →
  store the Python `LayoutEnum`, rebuild layouts in-body.
- Schedule-variant branching inside `@schedule` needs `cutlass.const_expr`.
- Recovered the stage's TMA transaction barrier as
  `pipeline.sync_object_full.get_barrier(stage_idx)` so cute TMA atoms work inside a
  TS producer work method.
- Grouped rasterization (GROUP_M=8) was the single biggest perf lever:
  39.6 → 84.2 TF at 4096³.
- Exhaustive BFS checker left on for k_tiles ≤ 16, off for the big-shape compiles
  (structural checks + credit table always run) — matching documented TS practice.

Broken variants (`--variant no_release`, `--variant double_wait`): both rejected at
compile time by the credit simulator with `ConsumerWait 0 is blocked!`; captures in
`code/ts/broken_output.txt`, good-kernel analysis in `code/ts/compile_diagnostics.txt`.

### 3b. TLX GEMM (`code/tlx/gemm_ws_tlx.py`)

Started from the hopper-gemm-ws tutorial structure; replaced `async_dot` with
`tl.dot(tlx.local_load(...))`; TMA via `TensorDescriptor` + `triton.set_allocator`.
Incremental: TMA-ring tile copy bit-exact → tl.dot swap verified → tuning sweep
(single-consumer, BK/stage variants; chosen config: BM=128 BN=128 BK=64, 2 stages,
A split per replica, `replicate=2`, 9 warps, 74 KB smem, 131 regs). TTGIR asserted to
contain `ttg.warp_specialize` + 3 TMA loads per k-iter. Benchmarked per protocol plus
a plain-Triton twin kernel for the WS-vs-baseline delta.

Broken variant (`gemm_ws_tlx_broken.py`, one `barrier_arrive` on `empty_b[0]`
skipped): compiled clean, correct at K=128 (ring never wraps — bug latent), hard
deadlock at K=512 (`timeout 90` → exit 124). GPU recovered after the kill; good kernel
re-verified immediately afterwards.

### 3c. Primitives demos (`code/primitives/`)

- `three_ways.py`: shfl butterfly sum at all three tiers, identical results (496).
- `hazard_demo.py`: 6 injected variants. Fired: C3/C4 over-arrive (**hard error**, no
  flag needed, with suggested `elect_sync` fix), C3 count-mismatch warnings, C5
  expect_tx-no-completion warning, C16 nested-elect error (headline renders as literal
  `{0}` — 4.7.0 bug), ptxas local-memory remark with source frame. C13 (partial-warp
  elect) never fired in this build. Diagnostics come out on OS-level stderr from the
  C++ compiler — needed fd-level capture. `CUTE_DSL_COMPILER_OPT` clobbers per-compile
  option selectors; set all wanted domains in the env var.
- `tile_mma_prims.py`: pure-SIMT 64×64×64 fp16 tile GEMM (TMA → ldmatrix → mma_sync →
  fragment stores), max err 5.7e-6 vs fp64. **Found:** `prims.mma_sync` without
  explicit `multiplicand_{a,b}_ptx_type` SIGABRTs the MLIR library (no diagnostic;
  confirmed via gdb) — pass `prims.MMAType.F16` explicitly. The mma.sync fragment
  plumbing has no in-repo example precedent; derived from PTX ISA fragment layouts and
  validated first in a minimal 16×16×16 kernel.

## 4. Verification pass (~13:00)

Independent spot-checks re-run from the orchestrating session, all passing:

- `gemm_ws_ts.py` → verify PASS at 512³, full static-analysis printout ends
  `BFS complete: 625 states explored … Result: SAFE`, SMEM report 98,352 / 101,376 B.
- `gemm_ws_tlx.py` → verify OK at 512/1024/2048/4096 + plain-Triton baseline OK;
  74,344 B smem, 131 regs, 2 spills; TTGIR warp_specialize confirmed.
- `gemm_ws_ts_broken1.py` → rejected at compile time as captured.
- `three_ways.py` → all three tiers = 496, PASS.

## 5. Write-up (2026-08-18/19)

Published the overview as a Claude artifact
(https://claude.ai/code/artifact/234c7e7c-c5ea-4e6e-ad67-66e6df258b09), wrote
`comparison/README.md` on the machine, then assembled this repo folder (REPORT.md,
README.md, LABNOTES.md, `code/` mirror of the comparison tree) on 2026-08-19.

## Benchmark record

Protocol: fp16 in / fp32 acc, 25 warmup + 100 timed iters, per-iter CUDA events,
median; torch.matmul in-process. Raw JSONs: `code/ts/bench_results.json`,
`code/tlx/bench_results.json`.

| M=N=K | TS ms / TF | TLX-WS ms / TF | Plain Triton ms / TF | torch ms / TF |
|---|---|---|---|---|
| 1024³ | 0.0565 / 38.0 | 0.0381 / 56.4 | 0.0422 / 50.9 | 0.0340–0.0359 / 59.9–63.2 |
| 2048³ | 0.2532 / 67.9 | 0.2078 / 82.7 | 0.2225 / 77.2 | 0.1876–0.1916 / 89.7–91.6 |
| 4096³ | 1.6325 / 84.2 | 1.5555 / 88.4 | 1.6739 / 82.1 | 1.4813–1.5821 / 86.9–92.8 |

Caveats: single machine, bandwidth-limited GB10, torch.matmul shows ~±3% run-to-run
spread at 4096³; TS small shapes are wave-quantization-limited (1 CTA/SM; persistent
WorkQueue variant not attempted).

## Loose ends / future work

- TS persistent `WorkQueue` variant of the GEMM (closes the small-shape gap; also the
  natural place to try CLC-free persistent scheduling on sm_121).
- TS `PipelineGroup` (Merge/Fork) unusable in 4.7.0 on sm_121 — its only non-tcgen05
  tutorial script is broken; retest next release.
- Try µTLX (`triton-lang/triton-ext`) when it lands properly — the fork is migrating.
- File-able upstream issues: TLX cc-gating treats cc≥10 as datacenter (tmem tests fail
  rather than skip on sm_12x); TLX tutorial `num_regs` silent kwarg drop; CUTLASS 4.7
  dead tutorial links + tutorial 07 trace bug; C16 `{0}` headline; `prims.mma_sync`
  SIGABRT without explicit types.
