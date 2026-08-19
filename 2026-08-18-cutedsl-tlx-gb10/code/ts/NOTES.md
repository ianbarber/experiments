# Warp-Specialized Pipelined fp16 GEMM with the CuTe DSL Task Scheduling (TS) framework — GB10 / sm_121

**Status: working, verified, benchmarked.** To our knowledge this is the first TS GEMM
that runs on the sm_12x architecture class: every TS GEMM shipped with CUTLASS 4.7
(tutorials 02, 04, 05, 06, 07/02-04) is tcgen05/TMEM-based and cannot run here. This
kernel replaces the TMEM accumulator resource with a **register accumulator inside the
consumer task** and uses the sm_120-class `mma.sync` tensor-core path.

Environment: NVIDIA GB10 (DGX Spark, sm_121a), CUDA 13.0, nvidia-cutlass-dsl 4.7.0,
torch 2.13.0+cu130, venv `/home/ianbarber/Projects/cute/.venv-cutedsl`.

## Files

| File | What |
|---|---|
| `gemm_ws_ts.py` | The kernel + verify + benchmark (self-contained; `--verify`, `--bench`, `--diagnose`, `--variant`) |
| `gemm_ws_ts_broken1.py` | Broken variant: consumer `release()` removed → compile-time rejection |
| `gemm_ws_ts_broken2.py` | Broken variant: consumer waits twice per produce → compile-time deadlock detection |
| `compile_diagnostics.txt` | Captured TS static analysis of the good kernel (schedule table, register budget, SMEM report, exhaustive checker `Result: SAFE`) |
| `broken_output.txt` | Verbatim compile-time failures of both broken variants |
| `bench_results.json` | Benchmark numbers (median ms + TFLOP/s, vs `torch.matmul`) |
| `bench_log.txt` | Full stdout of the benchmark run |

## Kernel architecture

- **Tile**: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64; fp16 inputs, fp32 accumulate,
  fp16 output. Non-persistent: one 128x128 output tile per CTA, 1-D grid of
  `grid_m*grid_n` CTAs with Triton-style grouped rasterization
  (`RASTER_GROUP_M=8`) for L2 reuse.
- **SMEM ring**: 3 stages x (128x64 A tile + 128x64 B tile) = 3 x 32 KB = 96 KB
  data + 48 B mbarriers, laid out by the TS `SmemAllocator`
  (`smem_capacity_bytes=101376` passed explicitly — the TS default assumes 232 KB
  datacenter parts). SMEM layouts are the sm90/sm120 swizzled layouts from
  `cutlass.utils.hopper_helpers.make_smem_layout_a/b` (128B swizzle atoms).
- **Warp/task layout** (8 warps, 256 threads):
  - `MmaTask` warps 0-3, 232 regs/thread (consumer of `SmemAB`, producer of `GmemC`)
  - `LoadTask` warp 4, 40 regs/thread (consumer of `GmemAB` coords, producer of `SmemAB`)
  - `PaddingTask` warps 5-7, 40 regs/thread (register-budget padding for the warp group)
- **Pipeline**: `PipelineConfig.create_tma_async_pipeline_cfg(num_stages=3,
  num_bytes=32768, producer_group=CooperativeGroup(Thread) /* 1 elected thread */,
  consumer_group=CooperativeGroup(Thread, 128))` → lowered to `PipelineTmaAsync`
  (TMA transaction mbarriers / thread arrivals).
- **Resource graph**: `{SmemAB: [GmemAB], GmemC: [SmemAB]}` with a
  `TaskLocalVariable` Int32 token (`k_tile`) routed from `GmemAB.compute_coords`
  into `SmemAB.tma_load`.
- **Schedules** (the checked objects):
  ```
  LoadTask: for k in domain_loop(0, k_tiles):
      k_tile = gmem_ab.compute_coords()
      smem_ab.try_acquire(); smem_ab.acquire()      # arms tx barrier w/ 32768B
      smem_ab.tma_load(k_tile=k_tile)               # 2x cute.copy TMA atoms
      smem_ab.commit()
  MmaTask:  smem_ab.init_mma_state()                # aux head: zero fp32 acc
      for k in domain_loop(0, k_tiles):
      smem_ab.try_wait(); smem_ab.wait()
      smem_ab.mma_step()                            # ldmatrix + 4x cute.gemm
      smem_ab.release()
      gmem_c.store_c()                              # tail: RF -> GMEM epilogue
  ```
- **MMA work** (per K tile, inside the wait/release bracket): TiledMMA of
  `cute.nvgpu.warp.MmaF16BF16Op` (m16n8k16) with atom layout (2,2,1) and
  permutation (32,32,16) — same recipe as the shipped sm_120
  `blackwell_geforce/dense_gemm.py`. SMEM→RF via `LdMatrix8x8x16bOp(.x4)` tiled
  copies; 4 k-blocks per stage with copy(kb+1) software-pipelined against
  gemm(kb). Accumulator: `cute.make_rmem_tensor` fp32, 128 elems/thread.
- **Epilogue** (MmaTask tail, producer work on `GmemC`): vector-convert acc to
  fp16 (`acc.load().to(Float16)`) and `cute.autovec_copy` to the
  `thr_mma.partition_C` view of the output tile. No SMEM staging / TMA store —
  cost is amortized over the K loop.
- **TMA loads inside TS**: issued with the *cute atom path* (`cute.copy(tma_atom,
  gmem_slice, smem_stage_slice, tma_bar_ptr=...)`) rather than the raw-tensormap
  prims path of the tutorials, so the TMA descriptors, SMEM swizzle layouts, and
  ldmatrix partitioning are all derived from one layout definition. The stage's
  transaction barrier is recovered inside the producer work as
  `self.pipeline.sync_object_full.get_barrier(stage_info.stage_idx)` (the same
  pointer TS itself puts in `stage_info.barrier`, which is only exposed as a
  prims-style `cutlass.Array`). TS's `acquire()` arms the barrier with the
  configured `num_bytes`; the work body only issues the copies.

## DSL walls hit and their workarounds (the useful part)

1. **Resource dataclasses cannot grow new fields inside task regions.** The
   framework threads resource objects through the per-task `if is_selected()`
   scf.if regions by extracting/rebuilding their MLIR values; assigning a new
   attribute inside a work body fails with
   `error[TYPE_DYNAMIC_EXPR_UNSUPPORTED] ... has extra field 'smem_copy_A'`.
   The tutorials' pattern (pre-declaring fields initialized with
   structurally-identical null values) works for flat arrays but is impractical
   for TiledMMA partition views. **Workaround**: keep only the accumulator as
   persistent state, allocate it in the resource `__init__` (which traces at
   kernel top level, so its alloca dominates every task region), and recompute
   all partitioned views inside each work body (compile-time layout algebra;
   the runtime cost is a few address computations that LICM hoists).
2. **MLIR-valued layout objects must not be stored on resources.** Passing the
   staged `ComposedLayout`s through resource fields produced an IR dominance
   verifier failure (`cute.composed_get_inner ... operand does not dominate`) —
   the layout value gets region-threaded and its swizzle introspection lands in
   a sibling region. **Workaround**: store the Python-level `LayoutEnum` and
   rebuild `make_smem_layout_a/b` fresh inside each work body. (TMA CopyAtoms
   and cute.Tensors thread fine.)
3. **Python `if` on a variant flag inside `@schedule` needs
   `cutlass.const_expr(...)`** or the tracer tries to stage the `ResourceProxy`.
4. Everything else worked exactly as the TS API promised: the TaskManager's
   arrival-count validation, the credit-model schedule table, and the exhaustive
   BFS checker (625 states, `Result: SAFE` at k_tiles=8) all ran at
   `cute.compile` time (see `compile_diagnostics.txt`).

## Benchmark (protocol: fp16 in/fp32 acc, 25 warmup / 100 timed iters, median, torch.cuda.Event)

| M=N=K | TS kernel ms | TS TFLOP/s | torch.matmul ms | torch TFLOP/s | TS / torch |
|---|---|---|---|---|---|
| 1024 | 0.0564 | 38.0 | 0.0359 | 59.9 | 63% |
| 2048 | 0.2532 | 67.9 | 0.1876 | 91.6 | 74% |
| 4096 | 1.6325 | 84.2 | 1.4813 | 92.8 | **91%** |

Correctness: `torch.testing.assert_close(atol=1e-2, rtol=1e-2)` vs
`(a.float() @ b.float().T).half()` passes at 512/1024/2048/4096 and a
non-square 1024x512x768.

Notes on the numbers:
- Grouped rasterization mattered enormously on this bandwidth-limited part
  (LPDDR5x): 4096 went from 39.6 → 84.2 TFLOP/s just by supertiling the CTA
  order (the naive 2-D grid thrashes L2).
- Small shapes are wave-quantization / tail limited: one CTA per SM
  (96 KB SMEM + 232-reg mma warps), so 1024 = 64 CTAs is barely more than one
  wave. A persistent `WorkQueue` variant would close most of that gap.
- The exhaustive checker is disabled for the K=2048/4096 compiles
  (`exhaustive_check=(k_tiles<=16)`) purely for host-side compile time — the
  structural checks + credit-model table still run on every compile. This is
  the documented TS production practice.

## Broken variants (TS's static checking demonstrated)

Both are the *same kernel file* with one-line schedule mutations, both fail
**inside `TaskManager.__init__` during `cute.compile`, before any launch**
(each run wrapped in `timeout 90`; both exit in ~15 s):

- `gemm_ws_ts_broken1.py` (`variant=no_release`): consumer loop is
  `try_wait; wait; mma_step` with no `release()`. The credit simulator runs the
  producer 3 stages ahead, then every task is blocked:
  `ValueError: SmemAB     ConsumerWait    0 is blocked!` (schedule table in
  `broken_output.txt` shows LoadTask stuck at `ProdTrA`/`ProdAcq` after 3
  commits). A hand-written kernel with this bug compiles cleanly and hangs on
  the GPU.
- `gemm_ws_ts_broken2.py` (`variant=double_wait`): consumer loop is
  `try_wait; wait; mma_step; release; wait; release` — two waits per produced
  stage. Same compile-time rejection: the simulator shows the consumer draining
  two commits per iteration until `ConsumerWait` starves:
  `ValueError: SmemAB     ConsumerWait    0 is blocked!`.
