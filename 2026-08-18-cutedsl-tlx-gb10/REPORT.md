# Warp Specialization, Two Ways: CUTLASS 4.7 Task Scheduling vs TLX `async_task` (plus Primitives)

**Machine:** NVIDIA GB10 (DGX Spark), sm_121a, aarch64 Grace, CUDA 13.0, driver 580.159
**Versions:** CUTLASS v4.7.0 (`nvidia-cutlass-dsl==4.7.0`) · TLX = `facebookexperimental/triton` branch `tlx` @ `927d1e04` (Triton 3.3 base)
**Date:** 2026-08-18

CUTLASS 4.7's CuTe DSL ships two new layers: a **Task Scheduling (TS)** framework
("static analysis of execution schedules for warp-specialized kernels; compilation stops
when known concurrency issues are detected") and a **Primitives** API ("a stable, thin
wrapper over NVVM operations… a transitional API until a CUDA Python-like solution is
available"). This report compares TS against Meta TLX's `async_task` warp-specialization
model using matched, verified, benchmarked warp-specialized GEMMs written for this
machine, and shows concretely how Primitives changes low-level authoring relative to
inline PTX.

---

## TL;DR

- **Same machine, opposite bets.** TS and TLX describe identical runtime structure —
  tasks pinned to warp ranges, multi-stage SMEM rings, mbarrier full/empty protocols,
  `setmaxnreg` register reallocation — and lower to the same PTX idioms. TS makes the
  schedule a **checked data structure** (≈20 structural validators + a credit-model
  simulator + an exhaustive BFS over task interleavings, all before codegen). TLX makes
  it **hand-written code**: you open-code buffer indices, phase bits, and arrive counts,
  and the compiler checks none of it (its barrier layer literally contains
  `# TODO. add validator logics`).
- **The safety experiment settles the difference.** The same missing-release bug was
  injected into both GEMMs. TS rejected both broken variants at compile time
  (`ValueError: SmemAB ConsumerWait 0 is blocked!`). TLX compiled its broken kernel
  with zero warnings — it produced *correct results* at K=128 (bug latent; ring never
  wraps) and hard-deadlocked the GPU at K=512.
- **Performance is a wash where both run.** At 4096³ fp16: TLX-WS 88.4 TF, TS 84.2 TF,
  plain Triton 82.1 TF, cuBLAS 86.9–92.8 TF (run-to-run spread). TS costs ~3.5× the
  source lines (388 vs 111); TLX costs the debugging model.
- **On sm_121, TLX has no tensor-core op at all**: `tlx.async_dot` hard-codes wgmma
  (cc==9) / tcgen05 (cc≥100), so consumer Blackwell gets instructions ptxas rejects.
  The working path is plain `tl.dot` on `tlx.local_load`-ed smem tiles. CUTLASS ships
  native sm_12x `mma.sync` kernels.
- **Primitives replaces inline PTX with compiler-visible ops**: typed signatures,
  trace-time validation, and a compile-time hazard checker that refuses to build an
  mbarrier over-arrive. Inline PTX remains available as the sanctioned last resort
  (`prims.*` → `prims.dialect.*` → `prims.inline_ptx`).

---

## 1. Hardware constraints: what a GB10 can run

sm_121 ("GeForce-class" Blackwell) has TMA (`cp.async.bulk.tensor`), mbarriers,
`cp.async`, clusters, `setmaxnreg`, and *synchronous* `mma.sync`-family tensor cores
(fp16/bf16/fp8/fp4 incl. sm_120/121 block-scaled forms). It has **no tcgen05/TMEM and
no wgmma**. Usable SMEM: 101,376 B/CTA.

| Piece | GB10? | Why |
|---|---|---|
| TS framework + all static analysis | ✅ | Host-side Python, arch-agnostic |
| TS copy/TMA/persistent tutorials (01, 03) | ✅ | Verified passing |
| TS GEMM tutorials (02, 04, 05, 06, 07b–d) | ❌ | All tcgen05/TMEM → `NVVM backend compilation failed … sm_121a` |
| Primitives (cp.async, mbarrier, TMA, ldmatrix, mma.sync, warp ops) | ✅ | All tested examples pass |
| Primitives `tcgen05/`, wgmma | ❌ | Datacenter-only |
| TLX warp-spec machinery (tasks, barriers, TMA, cp.async) | ✅ | Full WS unit tests pass (after 2 toolchain fixes, §6) |
| TLX `async_dot` | ❌ | `version = 5 if cc >= 100 else 3` → cc 12.1 takes tcgen05 path; no mma.sync fallback |

Note: the CUTLASS tutorial README links TS FMHA and "blackwell_geforce" TS examples —
those paths **do not exist** in the shipped v4.7.0 tree. The only sm_12x-runnable TS
code is the copy/persistent family, which is why the GEMM below had to be written from
scratch.

---

## 2. Task Scheduling: the schedule as a checked object

A TS kernel is assembled from four declared objects, all resolved at trace time:
**resources** (dataclasses wrapping one physical thing, optionally pipelined),
**tasks** (contiguous warp ranges bound to resources), a traced **`@schedule`** per
task, and a **dependency graph** (a plain dict). The `TaskManager` validates everything
at construction — before any GPU code is emitted — then generates all mbarrier storage,
init counts, and `PipelineState` phase tracking, lowering onto the existing
`cutlass.pipeline` classes (`PipelineTmaAsync` etc.). Generated PTX matches a
hand-coded kernel with the same schedule; the product is the verification and the
bookkeeping, not new codegen.

The samples below are from the working sm_121 TS GEMM written for this comparison
(`code/ts/gemm_ws_ts.py` — to our knowledge the first TS GEMM on this arch class).

**A resource with work methods.** Producer work issues TMA into the stage the
framework hands it; TS's `acquire()` has already armed the stage's transaction barrier:

```python
@dataclass
class SmemAbResource(MemoryResource):
    ...
    @producer_work
    @cute.jit
    def tma_load(self, stage_info: StageInfo, *, k_tile: cutlass.Int32) -> None:
        sA, sB = self._make_smem_tensors(stage_info)
        gA = cute.local_tile(self.mA, (TILE_M, TILE_K), (pid_m, None))
        gB = cute.local_tile(self.mB, (TILE_N, TILE_K), (pid_n, None))
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(self.tma_atom_a, 0,
            cute.make_layout(1), cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        ...
        # TS's acquire() already armed the stage barrier with the expected
        # transaction byte count; we only issue the TMA copies at it.
        bar = self.pipeline.sync_object_full.get_barrier(stage_info.stage_idx)
        cute.copy(self.tma_atom_a, tAgA[(None, k_tile)],
                  tAsA[(None, stage_info.stage_idx)], tma_bar_ptr=bar)
        cute.copy(self.tma_atom_b, tBgB[(None, k_tile)],
                  tBsB[(None, stage_info.stage_idx)], tma_bar_ptr=bar)

    @consumer_work
    @cute.jit
    def mma_step(self, stage_info: StageInfo) -> None:
        # CuTe TiledMMA (mma.sync m16n8k16) with ldmatrix SMEM->RF copies,
        # software-pipelined k-blocks, register accumulator:
        cute.copy(copy_A, tCsA_p[None, None, 0], tCrA_cv[None, None, 0])
        cute.copy(copy_B, tCsB_p[None, None, 0], tCrB_cv[None, None, 0])
        for kb in cutlass.range_constexpr(num_k_blocks):
            if cutlass.const_expr(kb + 1 < num_k_blocks):
                cute.copy(copy_A, tCsA_p[None, None, kb + 1], tCrA_cv[None, None, kb + 1])
                cute.copy(copy_B, tCsB_p[None, None, kb + 1], tCrB_cv[None, None, kb + 1])
            cute.gemm(self.tiled_mma, self.acc,
                      tCrA[None, None, kb], tCrB[None, None, kb], self.acc)
```

Values flow between tasks through typed `TaskLocalVariable` tokens rather than shared
state — a consumer `returns=` a token, a downstream producer receives it as a kwarg:

```python
class GmemAbResource(MemoryResource):
    """Coordinate-only GMEM source: emits the K-tile index consumed by TMA."""
    k_tile: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    @consumer_work(returns=k_tile)
    @cute.jit
    def compute_coords(self, stage_info: StageInfo) -> cutlass.Int32:
        return cutlass.Int32(stage_info.loop_offset)
```

**The pipeline config** declares the protocol; the TaskManager recomputes the expected
mbarrier arrival counts from it and errors with exact values if you get them wrong:

```python
smem_ab_cfg = PipelineConfig.create_tma_async_pipeline_cfg(
    num_stages=STAGES,                 # 3
    num_bytes=tma_bytes_per_stage,     # 32768: TMA transaction bytes per stage
    # LoadTask has 1 warp; one elected thread arms the barrier per stage.
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    # MmaTask has 4 warps (128 threads) releasing each stage.
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_MMA_WARPS * 32),
)
```

**The schedules and assembly** — this is the part TS checks. Reserved verbs
(`acquire/commit`, `try_wait/wait/release`) bracket user work; statements after the
`domain_loop` are the tail (epilogue):

```python
@schedule
def load_schedule(gmem_ab, smem_ab) -> None:
    with domain_loop(0, k_tiles, 1):
        k_tile = gmem_ab.compute_coords()
        smem_ab.try_acquire()
        smem_ab.acquire()                 # arms the stage's tx barrier
        smem_ab.tma_load(k_tile=k_tile)
        smem_ab.commit()

@schedule
def mma_schedule(smem_ab, gmem_c) -> None:
    smem_ab.init_mma_state()              # AUXILIARY head work (zeros the acc)
    with domain_loop(0, k_tiles, 1):
        smem_ab.try_wait()
        smem_ab.wait()
        smem_ab.mma_step()
        smem_ab.release()
    gmem_c.store_c()                      # tail: RF -> GMEM epilogue

mma_task  = Task(name="MmaTask",  src_resources=[smem_ab_resource],
                 dst_resources=[gmem_c_resource],  warp_idx=0, num_warps=4,
                 schedule=mma_schedule(smem_ab_resource, gmem_c_resource),
                 num_registers=232)
load_task = Task(name="LoadTask", src_resources=[gmem_ab_resource],
                 dst_resources=[smem_ab_resource], warp_idx=4, num_warps=1,
                 schedule=load_schedule(gmem_ab_resource, smem_ab_resource),
                 num_registers=40)
# + PaddingTask (warps 5-7): register-budget padding for the warp group

task_manager = TaskManager(
    tasks=[mma_task, load_task, padding_task],
    resource_dependency_graph={smem_ab_resource: [gmem_ab_resource],
                               gmem_c_resource: [smem_ab_resource]},
    smem_allocator=allocator,
    smem_capacity_bytes=101376,           # default assumes 232KB datacenter parts!
)
task_manager.setup_resources_and_tasks()
prims.fence_mbarrier_init(); prims.barrier_cta_sync(0)
task_manager.run()
```

`TaskManager(...)` construction runs the full static analysis: ~20 structural checks
(work bracketing, arrival counts, register/SMEM budgets), a credit-model simulation of
every task's expanded schedule, and an exhaustive BFS over all valid interleavings with
a hardware-aware race model (async producer writes stay "in flight" until a consumer
wait proves them landed; allocator-derived SMEM/TMEM aliasing). Declared limits: work
bodies are black boxes, pipeline *types* are taken on faith, runtime-sized domains fall
back to `domain=1` in the simulator.

---

## 3. TLX `async_task`: the schedule as code you write

TLX expresses warp specialization with two nested context managers (pure AST markers)
that materialize directly as upstream Triton's `ttg.warp_specialize` IR op — the same
op Triton's *automatic* warp specialization emits, which bails out when manual TLX ops
are present. The full working sm_121 kernel from this comparison
(`code/tlx/gemm_ws_tlx.py`), condensed:

```python
@triton.jit
def gemm_ws_kernel(a_desc, b_desc, c_ptr, M, N, K,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   GROUP_M: tl.constexpr, STAGES: tl.constexpr):
    BM_SPLIT: tl.constexpr = BLOCK_M // 2
    # SMEM rings: A has 2*STAGES half-tiles (one half per consumer replica),
    # B has STAGES full tiles shared by both replicas.
    a = tlx.local_alloc((BM_SPLIT, BLOCK_K), tlx.dtype_of(a_desc), STAGES * 2)
    b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), STAGES)
    bars_empty_a = tlx.alloc_barriers(num_barriers=STAGES * 2, arrive_count=1)
    bars_full_a  = tlx.alloc_barriers(num_barriers=STAGES * 2, arrive_count=1)
    bars_empty_b = tlx.alloc_barriers(num_barriers=STAGES, arrive_count=2)  # 2 consumers
    bars_full_b  = tlx.alloc_barriers(num_barriers=STAGES, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):            # ---- producer (trunk, 1 warp)
            ... # grouped-rasterization pid math (duplicated per region: captures cost SMEM)
            p = 1  # producer phase starts at 1: 1,1,0,0,1,1,... for STAGES=2
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                buf = k % STAGES
                tlx.barrier_wait(bar=tlx.local_view(bars_empty_a, buf), phase=p)
                full_a1 = tlx.local_view(bars_full_a, buf)
                tlx.barrier_expect_bytes(full_a1, BM_SPLIT * BLOCK_K * 2)
                tlx.async_descriptor_load(a_desc, tlx.local_view(a, buf),
                                          [offset_am, offset_k], full_a1)   # TMA
                ... # same dance for the B tile and A's second half
                p = p ^ (buf == (STAGES - 1))      # flip when the ring wraps

        with tlx.async_task(num_warps=4, replicate=2):   # ---- consumers (2x4 warps)
            ... # pid math duplicated again
            rid = tlx.async_task_replica_id()      # constexpr clone index
            p = 0  # consumer phase starts at 0
            acc = tl.zeros([BLOCK_M // 2, BLOCK_N], dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                buf = k % STAGES
                tlx.barrier_wait(bar=tlx.local_view(bars_full_a, buf + STAGES * rid), phase=p)
                tlx.barrier_wait(bar=tlx.local_view(bars_full_b, buf), phase=p)
                a_tile = tlx.local_load(tlx.local_view(a, buf + STAGES * rid))
                b_tile = tlx.local_load(tlx.local_view(b, buf))
                # mma.sync path -- tlx.async_dot would emit tcgen05 on cc12.1
                # which ptxas rejects; plain tl.dot is the verified path on sm_121.
                acc = tl.dot(a_tile, b_tile, acc)
                tlx.barrier_arrive(tlx.local_view(bars_empty_a, buf + STAGES * rid))
                tlx.barrier_arrive(tlx.local_view(bars_empty_b, buf))  # arrive_count=2
                p = p ^ (buf == (STAGES - 1))
            offs_m = offset_am + (BLOCK_M // 2) * rid + tl.arange(0, BLOCK_M // 2)
            offs_n = offset_bn + tl.arange(0, BLOCK_N)
            tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))
```

Everything a CUTLASS `Pipeline` class encapsulates — slot index, phase bit,
producer-starts-at-phase-1 convention, expect-bytes arming, arrive-count fan-in — is
hand-open-coded, and none of it is validated. The backend generates a persistent
switch-loop runtime: worker warps drop to low register counts, spin on a hardware
barrier, jump into their partition, and `setmaxnreg` rebalances (verified live in the
PTX on this machine). Meta's stated position (TLX paper) is that hiding execution
structure makes the compiler the bottleneck on new hardware — so TLX chooses exposure,
and it is production-proven on H100/B200 (block-sparse attention for ads ranking,
claimed +30% MFU on production attention layers).

---

## 4. Head-to-head results

Both kernels: TMA producer → multi-stage SMEM ring → synchronous-tensor-core consumers,
fp16 in / fp32 accumulate, 128×128×64 tiles, grouped L2 rasterization.

| | TS (`code/ts/`) | TLX (`code/tlx/`) |
|---|---|---|
| Tasks / warps | LoadTask 1 warp @40 regs, MmaTask 4 warps @232 regs, PaddingTask 3 warps | trunk producer 1 warp + `async_task(num_warps=4, replicate=2)` = 9 warps |
| Ring | 3 stages × (A+B) = 96 KB, `PipelineTmaAsync` | 2 stages, A split per-replica, 74 KB, hand-indexed |
| Math | CuTe TiledMMA `mma.sync` m16n8k16 + ldmatrix | `tl.dot` on `tlx.local_load` tiles |
| Sync authored by | TS-generated from `@schedule` + config | hand-written (4 barrier arrays, XOR phase protocol) |
| Kernel+schedule LOC | 388 | 111 |

### Benchmarks

Median of 100 iters after 25 warmup, per-iteration CUDA events, `torch.matmul`
measured in-process (its 4096³ number varied 87–93 TF across runs — treat single-digit
% as noise):

| M=N=K | TS GEMM | TLX-WS GEMM | Plain Triton | torch.matmul (cuBLAS) |
|---|---|---|---|---|
| 1024³ | 38.0 TF | 56.4 TF | 50.9 TF | 59.9–63.2 TF |
| 2048³ | 67.9 TF | 82.7 TF | 77.2 TF | 89.7–91.6 TF |
| 4096³ | 84.2 TF | **88.4 TF** | 82.1 TF | 86.9–92.8 TF |

Observations:
- At 4096³ both frameworks reach effective cuBLAS parity on this box.
- TLX-WS beats its plain-Triton twin by 8–11% at every shape — warp specialization
  pays even with synchronous MMA.
- The TS small-shape gap is wave quantization, not framework overhead: 96 KB ring +
  232-reg MMA warps → 1 CTA/SM, and 1024³ is 64 CTAs ≈ 1.3 waves. A persistent
  `WorkQueue` variant (supported by TS, not attempted) is the known fix.
- Grouped rasterization mattered more than anything else on this bandwidth-limited
  part: it alone took the TS kernel from 39.6 → 84.2 TF at 4096³.

### The safety experiment

The same class of bug — a consumer failing to release a ring slot — injected into both.

**TS broken variant 1** (consumer `release()` deleted) and **variant 2** (consumer
waits twice per produce). Both die at `cute.compile` time inside `TaskManager`
construction, before any launch. The credit simulator prints the schedule table up to
the stall, then:

```
ValueError: SmemAB     ConsumerWait    0 is blocked!  NOTE: if the domain is not a
compile-time Python int, this may be a FALSE POSITIVE (domain defaults to 1).
Re-run with a realistic int domain via validate-only mode for accurate results.
```

The healthy kernel's compile prints its full analysis — per-task schedule table,
register budget (34816/65536), SMEM report (98352 B / 101376 B) — and:

```
BFS complete: 625 states explored, 1 complete, 0 deadlock(s), 0 race(s), 0 PDL order violation(s)
Result: SAFE
```

**TLX broken variant** (one `barrier_arrive` on `empty_b[0]` skipped via
`if buf != 0:`): compiles with **zero warnings** — the mbarrier protocol is invisible
to the compiler. At K=128 the ring never wraps: the kernel runs to completion with
**correct results** (a small-K unit test would pass this bug into production). At
K=512 the producer's second lap waits on a barrier nobody will ever arrive: every CTA
deadlocks and `torch.cuda.synchronize()` hangs until killed (`timeout` exit 124).

That asymmetry — readable compile error vs shape-dependent latent deadlock — is the
entire value proposition of TS demonstrated end to end.

### Cost comparison

| | TS | TLX |
|---|---|---|
| Up-front structure | Resources, configs, graph, schedules (388 LOC); framework errors state expected values | Two context managers + loops (111 LOC); phase/count idioms are tribal knowledge |
| Sync-bug failure mode | Compile-time error with schedule trace | Runtime hang or silent latent bug |
| Schedule refactoring | Edit, recompile, auto re-verified | Re-audit every phase/index/count by hand |
| Work bodies | Hand-written CuTe (layout algebra) — full control, steeper API | Normal Triton tile code — `tl.dot` does the layout work |
| v1 friction observed | Resource fields can't be created in task regions; MLIR-valued layouts can't live on resources; tutorial 07's only sm_121 script broken; docs describe unshipped features | Flagship tutorial passes a silently-dropped kwarg (`num_regs` vs `registers`); stock fork can't target sm_121a; cc-gating treats cc≥10 as datacenter |
| Direction | NVIDIA experimental, evolving in 4.x | Fork → µTLX plugin (`triton-lang/triton-ext`); production-proven |

---

## 5. Primitives: what replaces inline PTX

Before 4.7, dropping below CuTe meant `cute.arch`'s curated helpers (many of which are
themselves `llvm.inline_asm` strings — 66 call sites in one file) or raw PTX with
constraint strings:

```python
# the old way (cute/arch/nvvm_wrappers.py) — opaque asm blob, constraint letters
llvm.inline_asm(T.f32(), [Float32(a).ir_value()], "ex2.approx.ftz.f32 $0, $1;",
                "=f,f", has_side_effects=True, ...)
```

Primitives (`from cutlass.experimental import primitives as prims`) is a typed, 1:1
Python wrapper over the entire MLIR NVVM dialect — mbarrier lifecycle, every TMA
variant, cp.async, warp ops, fences/proxies, cluster ops, ldmatrix/stmatrix/mma.sync/
wgmma/tcgen05, setmaxnreg, atomics, fp4/6/8 conversions. Three escape-hatch tiers,
demonstrated live in `code/primitives/three_ways.py` (one warp butterfly reduction,
three implementations, identical results = 496):

```python
# Tier 1 — wrapped op: typed args, StrEnum kinds, Python-int coercion, typed returns
acc += prims.shfl_sync(0xFFFFFFFF, acc, offset, 0x1F, prims.Shfl.BFLY)

# Tier 2 — raw NVVM dialect via the auto-converting proxy: any NVVM op, but YOU
# supply the MLIR result type first and raw dialect enums; .ir_value()s only
raw = prims.dialect.shfl_sync(T.i32(), mask.ir_value(), acc.ir_value(), ...)

# Tier 3 — the old way, still sanctioned as last resort
acc += prims.inline_ptx("shfl.sync.bfly.b32 {$r0}, {$r1}, ...", ...)
```

The payoff is compiler visibility. Real NVVM ops are dialect-verified, trace-time
validated (address spaces, value ranges, PTX-ISA gating with clear errors), and
analyzed by a new **compile-time hazard checker**. Verbatim from
`code/primitives/hazard_output.txt` on this machine:

```
error[nvvm-diag:C3/C4]: mbarrier arrive reaches a count=1 barrier from multiple threads

  --> hazard_demo.py:110:4
      in function `kernel_a(...)`:
   |
  108 |     prims.fence_mbarrier_init()
  109 |     prims.barrier_cta_sync(0)
> 110 |     prims.mbarrier_arrive(mbar)  # BUG: unguarded — all 32 threads over-arrive
      |    ^
  suggestion: guard this arrive with a single-issuer predicate such as
              `if nvvm.elect_sync(): ...` or `if tidx == 0: ...`
```

```
warning[nvvm-diag:C5]: mbarrier.arrive.expect_tx has no completion source for
128 registered transaction bytes        # <- this kernel would hang at runtime
```

Observed severity split: over-arrive (C3/C4) and nested-`elect_sync` (C16) are
**always-on compile failures**; init/arrive count mismatch (C3) and
expect_tx-without-completion (C5) are opt-in warnings via
`CUTE_DSL_COMPILER_OPT="warnings{nvvm}"`; `warnings{ptx}` adds register-spill /
local-memory remarks with Python source lines. The identical arrive written as inline
PTX is invisible to all of this — which is precisely why the API exists.

**"Tensor Core programming through SIMT"** = CUDA-with-intrinsics in Python: warp roles
from `warp_id` compares, single-issuer ops gated by `if prims.elect_sync():`, operands
in `cutlass.Array` SMEM, the MMA as one instruction call. From
`code/primitives/tile_mma_prims.py` (single-tile 64×64×64 fp16 GEMM, max err 5.7e-6 vs
fp64 reference, ~120 lines, no CuTe layouts anywhere):

```python
if warp == 0:
    if prims.elect_sync():                       # hazard checker demands the gate
        prims.mbarrier_arrive_expect_tx(mbar, bytes_a + bytes_b)
        prims.cp_async_bulk_tensor_shared_cta_global(smem_a,  tma_a.get_ptr(),  (0, 0), mbar)
        prims.cp_async_bulk_tensor_shared_cta_global(smem_bt, tma_bt.get_ptr(), (0, 0), mbar)
while not prims.mbarrier_try_wait_parity(mbar, 0, time_limit=10_000_000):
    pass
...
a_frag = prims.ldmatrix(ptr_a, 4, prims.MMALayout.ROW)   # ldmatrix.sync.aligned.m8n8.x4
d = prims.mma_sync(...,                                   # mma.sync.aligned.m16n8k16
        multiplicand_a_ptx_type=prims.MMAType.F16,        # REQUIRED: inference SIGABRTs
        multiplicand_b_ptx_type=prims.MMAType.F16)
```

**When to reach for which:** stay in CuTe/TiledMMA while the abstraction helps; drop to
`prims.*` when you need an unwrapped instruction or CUDA-C++-style control (and get the
hazard checker free); `prims.dialect` for NVVM ops the wrapper hasn't caught up with;
`inline_ptx` only for opcodes NVVM lacks or hand-scheduled asm. NVIDIA's own framing —
"transitional API until a CUDA Python-like solution is available" — signals a bridge to
CUDA-Python-era device authoring, not a permanent home.

4.7.0 rough edges hit in practice: C16's message renders as a literal `{0}`; C13
(partial-warp elect) never fired in this build; `prims.mma_sync` SIGABRTs the MLIR
library without explicit multiplicand types.

---

## 6. Verdict

TLX trusts the expert: minimal ceremony, Triton tile codegen inside each task, and a
debugging model of "stare at the phase math until the hang goes away" — validated by
Meta's production kernels and by how little code the GEMM took. TS trusts the model: it
demands declared resources, dependencies, and schedules, and pays back with
machine-checked protocols, guided errors, and refactorable schedules — validated here
the direct way, by catching real injected deadlocks at compile time that TLX shipped
silently.

If a kernel fits Triton's sweet spot and targets datacenter parts, TLX (soon µTLX) is
the pragmatic tool; its consumer-Blackwell story is currently void. For CUTLASS-class
kernels where the sync protocol *is* the hard part — deep multi-resource pipelines,
persistent scheduling, cluster fan-in — TS's static checking is a genuinely new
capability nothing in the Triton world offers, at the cost of a much larger authoring
surface and clear v1 rough edges. Primitives matters to both stories: it moves the
classic mbarrier footguns from runtime hangs into compiler diagnostics, which is what
makes hand-rolled SIMT synchronization tractable at all.

---

## 7. Reproduction

Code in `code/` (mirrors `~/Projects/cute/comparison/` on the DGX Spark). Environments:

- `.venv-cutedsl`: `nvidia-cutlass-dsl==4.7.0`, torch 2.13+cu130, `apache-tvm-ffi`
  (required by primitives examples). Arch auto-detects sm_121a.
- `.venv-tlx`: editable build of the `tlx` branch (~7 min on 20 Grace cores) plus two
  required local fixes for sm_121a: bundled ptxas → symlink to CUDA 13's, and
  `ptx_get_version()` mapped CUDA 13 → PTX ISA 8.8 (equivalently
  `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas TRITON_MOCK_PTX_VERSION=12.9`).
  Do not `pip install torch` after the editable install (it clobbers the fork's triton).

```bash
# TS GEMM: verify + static analysis (add --bench for the sweep, --variant no_release to break it)
.venv-cutedsl/bin/python code/ts/gemm_ws_ts.py

# TLX GEMM: verify + bench (run from the tlx-triton repo root)
.venv-tlx/bin/python code/tlx/gemm_ws_tlx.py

# TLX broken variant — deadlocks at K=512, keep the timeout
timeout 90 .venv-tlx/bin/python code/tlx/gemm_ws_tlx_broken.py

# Primitives: escape-hatch tiers, hazard checker (~1 min), SIMT tensor-core tile
.venv-cutedsl/bin/python code/primitives/three_ways.py
.venv-cutedsl/bin/python code/primitives/hazard_demo.py
.venv-cutedsl/bin/python code/primitives/tile_mma_prims.py
```

**Sources:** CUTLASS v4.7.0 (`python/CuTeDSL/cutlass/experimental/{task_scheduling,primitives}/`,
`media/docs/pythonDSL/ts_general/`, TS tutorials) · [facebookexperimental/triton@tlx](https://github.com/facebookexperimental/triton/tree/tlx) ·
[TLX paper (arXiv:2605.10905)](https://arxiv.org/abs/2605.10905) ·
[TLX Block Attention](https://pytorch.org/blog/tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention/) ·
[Warp Specialization in Triton: Design and Roadmap](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) ·
[Triton PR #5622](https://github.com/triton-lang/triton/pull/5622) ·
[triton-lang/triton-ext (µTLX)](https://github.com/triton-lang/triton-ext)

Benchmarks are single-machine numbers on a bandwidth-limited GB10 — treat them as
this-box observations, not general rankings.
