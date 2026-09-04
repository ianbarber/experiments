# TLX and the Triton Extension Architecture — A Research Report

**Working copy. Scope: implementation-level deep dive grounded in triton-lang/triton and triton-lang/triton-ext source. Bias: skeptical, performance-realistic, ignores marketing framing.**

> **Status banner (read first).** This is the *original* deep-dive — kept for its
> unique depth (PR archaeology, contributor/reviewer map, competitor table, the
> build-path forensics in §8). Things that have moved:
> 1. **Pins.** Cites triton `bceeb9ed7` / triton-ext `790c935`; later June work
>    used `a18b1bb3` / `a22c3d0`. File:line citations are against the *original*
>    pins. June inventory: [`docs/utlx-upstream-state.md`](docs/utlx-upstream-state.md).
> 2. **§0 / §6.1 "value-add is fork-only" needs an asterisk as of Triton 3.8
>    (2026-07).** The plugin ABI remains fork-free. **`async_dot` (wgmma)** now
>    runs on **upstream Triton source-built with `TRITON_EXT_ENABLED=ON` +
>    `triton-utlx==3.8.0`** (no core patch, no fbtriton) — via plugin-owned
>    `create_utlx_*` ops, not stock pip wheels. Warp-spec / several exotic ops
>    remain fork-coupled. **fbtriton** is still the zero-friction full surface.
>    Current story: [`WRITEUP.md`](WRITEUP.md); install: [`HOWTO.md`](HOWTO.md).

Primary artefacts referenced (cloned locally at `repos/`):
- `triton-lang/triton` — upstream compiler, HEAD `bceeb9ed7` (2026-04-14).
- `triton-lang/triton-ext` — out-of-tree extension host, HEAD `790c935` (2026-04-10).
- `facebookexperimental/triton` (branch `tlx`) — Meta's long-standing TLX fork; HEAD last pushed 2025-09-03.
- `pytorch/pytorch` issue [#178917](https://github.com/pytorch/pytorch/issues/178917) — release-packaging request, NOT a code PR.

Code citations use repo-relative paths.

---

## 0. TL;DR

**The thesis.** What's genuinely new in the past year of Triton work is not TLX itself but the **plugin ABI** that lets a shared library register new MLIR dialects, TTIR/TTGIR passes, and Python-callable custom ops against a stock Triton build — finally killing the fork-and-maintain tax that every accelerator DSL-on-Triton was paying.

**The mechanism** landed piece-by-piece in **Triton 3.6** (PRs [#8137](https://github.com/triton-lang/triton/pull/8137), [#8401](https://github.com/triton-lang/triton/pull/8401), [#8523](https://github.com/triton-lang/triton/pull/8523)), with the wheel-distribution flag `TRITON_EXT_ENABLED=1` cherry-picked into 3.7 via [#9935](https://github.com/triton-lang/triton/pull/9935). Lead authors: Corbin Robeck and Puyan Lotfi (Meta); reviewer-of-record on every core extension PR: Thomas Raoux (OpenAI). Andrew Brown and Simon Waters (kernelize.ai) drove ABI cleanup and host the `triton-ext` repo.

**µTLX** (`triton-lang/triton-ext/extensions/utlx/`) is Meta's port of TLX onto that ABI — most of the original TLX surface, no Triton fork required. It landed monolithically as PR #57 (+49,227 / -23 over 137 files, 2026-03-29). Provides warp specialization, async TMA/`cp.async` copies, mbarrier handshakes, Hopper `wgmma`, Blackwell `tcgen05`, cluster APIs, and Cluster Launch Control. Important framing the marketing usually skips: **µTLX is shaped to be an IR target for Meta's internal autotuning kernel generator, not a hand-DSL** — that's why the surface is so explicit and verbose.

**Performance reality:** publicly reproducible TLX wins are modest. On H100 the TLX FA kernel is roughly on par with FA3, not ahead. On B200 the autoWS path (which lowers to TLX-like WS) gets 1.5–2× over stock Triton on flash-attention forward but trails cuDNN. **No public TLX-vs-cuBLAS GEMM number exists.** Our own GEMM run on a consumer-Blackwell 5090 (the only TLX-eligible card we had locally) is in §8: TLX-explicit-pipelining lands within **1-9% of stock Triton's auto-pipeliner** on dense FP16 GEMM, with no clear winner — and that result specifically does **not** test the TLX value-add because `async_dot` is hard-gated to sm_100 (B200) for TMEM. The 5090 result mostly says "the auto-pipeliner is hard to beat on dense GEMM"; the regimes where TLX is supposed to shine (warp specialization, asymmetric work, attention) need an H100 or B200 to evaluate.

---

## 1. Background: why Triton was hard to modify

Triton is Python → TTIR → TTGIR → LLVM IR → PTX, with the middle two dialects (`triton::` and `triton::gpu::`) defined in-tree. Historically:

1. **Dialects were hard-coded.** Adding a new op required touching `include/triton/Dialect/Triton*/IR/Triton*Ops.td` plus multiple lowering passes. There was no registration path for out-of-tree dialects before 2025-09.
2. **Pass pipelines were hard-coded.** `third_party/{nvidia,amd,intel}/backend/compiler.py` each define a `make_ttir`, `make_ttgir`, `make_llir` staircase as a closed Python class. A fork was the only way to insert a pass.
3. **Backends had an escape hatch, dialects/passes did not.** Intel added `BaseBackend` + `register_backend` in June 2023 (PR [#1643](https://github.com/triton-lang/triton/pull/1643)); Microsoft added build-time `TRITON_PLUGIN_DIRS` for out-of-tree backend *directories* in Feb 2024 (PR [#3007](https://github.com/triton-lang/triton/pull/3007)). Both are coarse — they let you build a whole new backend tree alongside the in-tree backends but offered nothing for dialect/pass experimentation.
4. **Proton was the in-tree prototype.** The Proton dialect (Yuanwei Fang / Keren Zhou, PR [#5119](https://github.com/triton-lang/triton/pull/5119), 2024-11-21) lives at `third_party/proton/` and was the first non-backend third-party dialect. It hand-wired into the build system and Python bindings. It became the working pattern that #8523 generalized.

The result: every serious performance project (TLX at Meta, Gluon at OpenAI, various accelerator DSLs, IBM's `triton-dejavu`, Together's ThunderKittens-over-Triton attempts, and roughly a dozen internal forks inside silicon vendors) maintained a fork — with the attendant rebase pain and the obligatory divergence from upstream when core refactored a pass pipeline.

---

## 2. The extension architecture (what actually landed)

The design is documented in three places only:

- **`repos/triton/include/triton/Tools/PluginUtils.h:1-10`** — explicit comment citing MLIR's `DialectPlugin`/`PassPlugin` pattern as the prior art (they use `<mlir/Tools/Plugins/*Plugin.h>`).
- **`repos/triton/examples/plugins/README.md`** — worked three-mode walkthrough.
- **Triton community meetup 2026-01-06** notes at `repos/triton/docs/meetups/01-06-2026/notes.md:27–95` — the **only** design narrative that makes motivations explicit.

### 2.1 Version mapping (the user-facing story is slightly off)

The popular framing is "Triton 3.7 lets you install TLX over the top." The mechanism actually lives in **3.6**:

| Triton version | Release | Extension-relevant additions |
|---|---|---|
| v3.4.0 | 2025-07-30 | PR #3007 `TRITON_PLUGIN_DIRS` (build-time, backends only); PR #5119 Proton dialect (in-tree third-party) |
| v3.5.0 | 2025-10-13 | No new extension surface |
| **v3.6.0** | **2026-01-20** | **PR #8137 pipeline hook; PR #8401 TTIR/TTGIR plugin ABI; PR #8523 dialect plugins** |
| **v3.7.0** | 2026-04-10 (release/3.7.x) | **PR #9935 `TRITON_EXT_ENABLED=1` in wheels** (cherry-pick to 3.7.x as #9959). Everything else on main toward 3.8 |

For anyone building against the **current** `PluginInfo*` single-struct ABI (#9748), the target is main/3.8, not 3.7.x. The µTLX README at `repos/triton-ext/README.md:48–51` is honest about this: it pins against Triton `main` with `TRITON_EXT_ENABLED=1`.

### 2.2 The plugin ABI

Header: `repos/triton/include/triton/Tools/PluginUtils.h`. The contract:

```cpp
// PluginUtils.h:63-90 (simplified)
struct PassInfo     { const char *name; void (*addPass)(OpPassManager&); };
struct DialectInfo  { void (*registerCallbacks)(DialectRegistry&); };
struct OpInfo       { const char *name; AddOpCallback cb; };

struct PluginInfo {
  uint32_t apiVersion;
  const char *version;
  const char *gitHash;
  ArrayRef<PassInfo>    passes;
  ArrayRef<DialectInfo> dialects;
  ArrayRef<OpInfo>      ops;
};

extern "C" PluginInfo *tritonGetPluginInfo();  // single C symbol
```

Loading (`repos/triton/lib/Tools/PluginUtils.cpp`): Triton `dlopen`s every path in `TRITON_PLUGIN_PATHS` (colon-separated env) via `llvm::sys::DynamicLibrary::getPermanentLibrary`, resolves `tritonGetPluginInfo`, and registers the contents. Version checking (`TRITON_PLUGIN_VERSION_CHECK`) has three modes — `unset` (default: release-version equality), `on` (release + git hash), `off` (API-version only). The default-unset mode **will** reject a plugin built against a different Triton point release; authors should prefer `off` for development (PR #9937, Lotfi).

Registration from Python is indirect: `third_party/nvidia/backend/compiler.py`'s stage builders already consult `knobs.runtime.add_stages_inspection_hook` (landed in #8137). µTLX replaces those stage lambdas entirely — we'll see this below.

### 2.3 Four PRs to memorize

| # | Merged | Author | What it actually does |
|---|---|---|---|
| [#8137](https://github.com/triton-lang/triton/pull/8137) | 2025-09-19 | Robeck + Lotfi | Python-only `PipelineStagesHook` Protocol at `python/triton/knobs.py:455` (consumed via `add_stages_inspection_hook` at `:483`), called from `third_party/{amd,nvidia}/backend/compiler.py`. Nothing C++ yet. |
| [#8401](https://github.com/triton-lang/triton/pull/8401) | 2025-11-21 | Robeck + Lotfi | **The real ABI.** `include/triton/Tools/PluginUtils.h` + `lib/Tools/PluginUtils.cpp` (115 LOC loader) + `lib/Plugins/TritonPlugin.cpp` example + `triton-opt -tritongpu-plugin` + `TRITON_PLUGIN_PATHS` env + lit tests. 839 LOC. |
| [#8523](https://github.com/triton-lang/triton/pull/8523) | 2026-01-22 | Lotfi | Dialect plugin ABI: `tritonEnumeratePluginDialects`, `tritonGetDialectPluginInfo`; worked `DialectPlugin` example at `examples/plugins/DialectPlugins/`; registration via `bin/RegisterTritonDialects.h:1-13`. 773 LOC. |
| [#9626](https://github.com/triton-lang/triton/pull/9626) | 2026-03-19 | Robeck + Lotfi | **Custom Python-callable ops.** `OpInfo` + `AddOpCallback` taking a `TritonOpBuilder&` and operand values, wired through `python/src/ir.cc` (+71 LOC). Reference test at `python/test/unit/plugins/custom_ops.py`. Follow-up #9864 (3 days later) fixed return plumbing. |

Supporting housekeeping — #9748 (Brown, ABI unification into the single `PluginInfo*`), #9783 (Lotfi, symbol visibility gated by `TRITON_EXT_ENABLED`), #9691 (Lotfi, string args to `addPass`), #9935 (Robeck, wheels flag), #9937 (Lotfi, version-check modes), #9534 (Brown, `make install` so external builds can find headers).

### 2.4 What *didn't* land

- **No pluggable backend** through the new ABI. New targets still go through the 2024 `TRITON_PLUGIN_DIRS` build-time path (see `docs/meetups/01-06-2026/notes.md:63-66` — "Custom out-of-tree targets … in progress"). This is a gap in the "Triton is fully extensible" pitch.
- **No RFC or Discussions thread.** All 169 triton-lang/triton Discussions were paginated — none on plugin/extension. The meetup notes are the design record.
- **No compiler-analysis plugin surface.** You can register passes and dialects, but not e.g. layout analyses or cost models.

### 2.5 Who did the work (upstream)

**Meta**
- **Corbin Robeck** (`CRobeck`) — 34 commits; co-lead on #8137, #8401, #9626, #9935. Architecturally load-bearing. *Previously at AMD Research (the AMD ROCm blog author pages are pre-move); moved to Meta and has driven the extension ABI from there.*
- **Puyan Lotfi** (`plotfi`) — co-author on every Robeck extension PR; single-author on #8523, #9691, #9783, #9937, #9965 (opened release/3.7.x branch).
- **Neil Dhar** (`neildhar`) — reviewer on #8523; author of "Minimize exported symbols of libtriton" (`0e24a258d`, 2026-04-03).

**OpenAI**
- **Thomas Raoux** (`ThomasRaoux`) — reviewer-of-record on every core extension PR; co-presented "Triton Extensions, Plugins and Custom Ops" at the Jan 2026 meetup.
- **Keren Zhou** (`Jokeren`) — reviewer on #8137; co-author on Proton dialect (#5119), the prototype that seeded #8523.
- **Pawel Szczerbuk** (`pawelszczerbuk`) — reviewer on #8401.

**kernelize.ai** (third party — they host the triton-ext repo)
- **Andrew Brown** (`abrown`) — 24 commits in triton-ext (most of anyone). Upstream contributions #9748 (`PluginInfo*` cleanup), #9534 (install path), #9315/#9687 (bugfixes).
- **Simon Waters** (`sjw36`) — 8 commits; bootstrap build system; presenter at Jan 2026 meetup.

**AMD**
- **Lechen Yu** (`lechenyu`) — CMake-ordering fix #8397 (needed for plugin builds to link correctly).

This is **not** a Meta-over-the-wall project. Every core extension PR was reviewed by ThomasRaoux before merging; Brown's kernelize.ai contributions are the most polished ABI cleanups. The design was co-presented at the community meetup. Worth stressing in the report because it's easy to mis-tell.

---

## 3. TLX: two things called TLX

There are two codebases that answer to "TLX" and they are **not** the same:

1. **Original TLX** — `facebookexperimental/triton@tlx`. A hard fork of Triton with TLX primitives added directly to `triton.language.extra.tlx`. HEAD is 2025-09-03, no releases, 156 stars. Feature-complete relative to what Meta's internal autotuner needs.
2. **µTLX** — `triton-lang/triton-ext/extensions/utlx/`. A **plugin** port of most of original TLX onto the new ABI, landed in one monolithic PR #57 (`7181951`, +49,227 / -23 over 137 files) on 2026-03-29 by Puyan Lotfi + CRobeck + karthik-man. The README at `extensions/utlx/README.md:1-4` says plainly: *"This package provides most of the function that Meta's TLX … does, but without any changes to a fork of Triton."*

These coexist. If you want the full TLX surface today, you clone the fork. If you want the over-the-top install story, you use µTLX and accept that some primitives are not yet ported.

### 3.1 The µTLX "install-over-Triton" mechanism

µTLX does not ship as a Python package. `repos/triton-ext/pyproject.toml` contains only ruff/mypy config — **no** package metadata, **no** entry points. It's consumed as **two side-effects**:

**(a) A C++ plugin** — `libutlx.so` loaded via `TRITON_PLUGIN_PATHS`. The entry point is at `repos/triton-ext/extensions/utlx/uTLXPlugin.cpp:695-807`, registering:
- 12 custom passes
- 1 dialect (`TLXDialect`, 6 ops)
- **48 plugin-registered custom ops** (the bulk of the DSL surface)

**(b) A Python package** — `utlx_plugin`. After `import utlx_plugin`, the package performs three monkey-patches into Triton internals. The literal source is at `extensions/utlx/python/utlx_plugin/__init__.py:186-214`; condensed and grouped here for explanation (groupings mine):

```python
# (i) Namespace hijack — makes `import triton.language.extra.tlx` work:
_sys.modules['triton.language.extra.tlx'] = _sys.modules[__name__]
_extra.tlx = _sys.modules[__name__]

# (ii) Stage-replacement — installs a hook on the runtime knob; the hook later
#      rebuilds `make_ttgir` / `make_llir` to call plugin-registered passes:
knobs.runtime.add_stages_inspection_hook = custom_stages.inspect_stages_hook

# (iii) Compiler-dispatch monkey-patch (lazy; wrapped in try/except in real
#       source) — injects AST handlers for `tlx.async_tasks`/`tlx.async_task`:
from triton.compiler.code_generator import WITH_DISPATCH
from .compiler.dispatch import TLX_WITH_DISPATCH
WITH_DISPATCH.update(TLX_WITH_DISPATCH)
```

Mechanism (ii) is non-trivial: `custom_stages.inspect_stages_hook` (at `custom_stages.py:47-180`) **replaces** the stock `ttir`/`ttgir`/`llir` stage lambdas with versions that call `passes.plugin.utlx_convert_triton_to_tritongpu`, `utlx_insert_and_propagate_layout`, `utlx_storage_alias_lowering`, `utlx_rewrite_local_alias`. Mechanism (iii) is literal monkey-patching of `triton.compiler.code_generator.WITH_DISPATCH` — without it the `with tlx.async_task(...):` syntax can't be lowered.

**Fragility this implies:** three separate patches against Triton internals — `sys.modules`, `knobs.runtime.add_stages_inspection_hook`, `WITH_DISPATCH.update` — any of which can be renamed or refactored by upstream without breaking the plugin ABI itself. The triton-ext bot bumps the Triton pin weekly (10 of 54 commits on `main` are by `github-actions[bot]`), which is a tacit admission that the contract breaks roughly that often. Closed-unmerged issue #49 ("Adapt to new upstream plugin API") in the µTLX repo is the kind of fix-up this generates. Empirically, in §8 below we show this caught us on TWO of these three patches (semantic-method renames and ABI op signature changes) the day we tried to build µTLX against current Triton main.

### 3.2 The TLX dialect (surprisingly small)

`repos/triton-ext/extensions/utlx/tlx/dialect/include/IR/TLXOps.td` defines **7** ops:
- `tlx.storage_alias_spec` (line 27) — declares a logical SMEM/TMEM storage group.
- `tlx.storage_alias_local_alloc` (line 73) — alloc referencing a spec; lowered later.
- `tlx.reuse_group` (line 123) — declares `shared`/`distinct` buffer-overlap relationships with a `group_size` for subtiling.
- `tlx.set_buffer_overlap` (line 191) — binds a spec to its overlap tree.
- `tlx.require_layout` (line 246) — layout injection marker (stubbed as passthrough per `uTLXPlugin.cpp:489-493`, a real limitation).
- `tlx.release_layout` (line 262) — bookend for `require_layout` scope.
- `tlx.local_alias` (line 275) — post-lowering alias view.

The TLX dialect is deliberately thin. The real work lives in **48 plugin-registered custom ops** (`uTLXPlugin.cpp:730-793`) which don't introduce new dialect ops — they construct ops in existing upstream dialects (`ttg::`, `ttng::`) directly. For example `utlx_local_alloc` (`uTLXPlugin.cpp:76-98`) constructs `ttg::LocalAllocOp` with either `NVMMASharedEncodingAttr` or `SwizzledSharedEncodingAttr` selected by target; `utlx_alloc_barriers` (`:218-258`) constructs `ttg::LocalAllocOp` + a loop of `ttng::InitBarrierOp`.

This is an important architectural point: **TLX is less a new IR and more a new front end over existing TritonGPU ops with explicit pipeline/warp-specialization scheduling the pipeliner refuses to do automatically.** The "primitives" are mostly direct mappings to primitives upstream already has but that aren't exposed through `triton.language`. That's why the performance story isn't "new hardware" but "control over what the pipeliner was hiding."

### 3.3 The Python DSL surface

`__all__` in `__init__.py:6-98` exposes ~85 symbols grouped as:

- **Memory**: `local_alloc`, `local_view`, `local_store`, `local_load`, `local_slice`, `local_trans`, `local_reinterpret`, `async_load`, `async_store`, `async_descriptor_{load,prefetch,store,store_wait}`, `make_tensor_descriptor`, `tmem_copy`, `fence`, `fence_async_shared`, `remote_shmem_store`, `async_remote_shmem_store`, `map_to_remote_buffer`, `remote_view`.
- **Barriers**: `alloc_barriers`, `alloc_warp_barrier`, `barrier_expect_bytes`, `barrier_wait`, `barrier_arrive`, `cluster_barrier`, `named_barrier_{wait,arrive}`.
- **MMA**: `async_dot`, `async_dot_scaled`, `async_dot_wait`, `tcgen05_commit`.
- **Warp specialization**: `async_tasks`, `async_task`, `async_task_replica_id` (context-manager flavor).
- **Cluster / CTA**: `cluster_cta_rank`, `cluster_size_1d`, `thread_id`, `clock64`, `stoch_round`.
- **Dynamic launch / CLC** (Blackwell only): `_alloc_clc_responses`, `_clc_issue`, `_clc_query`, `clc_create_context`, `clc_producer`, `clc_consumer`.
- **MXFP8**: `_to_mxfp8_block`.
- **Warp**: `vote_ballot_sync`.

### 3.4 Hardware gating

Gated via `_cuda_parse_arch(options.arch)` returning the int from `sm(\d+)`:
- **TMEM** (`mem_ops.py:53-55`): `capability >= 100` (Blackwell).
- **`async_dot`** (`mma_ops.py:112-141`): branches on `>= 100` to select `tcgen05` vs `wgmma`; **register-operand A forbidden on Blackwell** (`:120`).
- **`async_dot_scaled`** (`mma_ops.py:188-189`): hard-gated Blackwell.
- **`stoch_round`** (`utility.py:120-123`): Blackwell.
- **PingPong pass** (`passes/nvidia/PingPong.cpp:67-80`): switch on capability — case 90 (Hopper) treats `WarpGroupDotOp` as "the expensive op"; case 100 (Blackwell) treats `math::ExpOp`/`Exp2Op` as expensive. This is the attention-specific scheduling heuristic.
- **AMD async copy** (`custom_stages.py:12-15`): gated to `gfx950`/`gfx1250`; AMD pingpong to `gfx942`/`gfx950`; in-thread transpose to `gfx942`.
- **AMD barrier passes** (`CMakeLists.txt:200-203`): **commented out** with a note that they *"require patched triton"* — so µTLX is NV-complete but AMD-incomplete against upstream.

**Practical implication for our 3090 (sm_86, Ampere):** `async_load` (via `cp.async`) would work, named barriers would work, but `async_dot` selects `wgmma` (Hopper-only), so `tlx.async_dot` fails at compile-time on Ampere. There is no `mma.sync`-tier fallback. **None of the 23 shipped tutorials target Ampere.** Meta's investment here is firmly Hopper-forward.

### 3.5 µTLX contributors (who actually wrote this)

From `git shortlog -sne main` in `repos/triton-ext`:

| Rank | Author | Commits | Affiliation |
|---|---|---|---|
| 1 | Andrew Brown | 24 | kernelize.ai (infra/build/CI) |
| 2 | github-actions[bot] | 10 | Weekly Triton-pin bumps |
| 3 | Simon Waters | 8 | kernelize.ai (build system) |
| 4 | Corbin Robeck | 7 | Meta (TLX bring-up) |
| 5 | Puyan Lotfi | 2 | Meta/FAIR (µTLX PR #57 author) |
| 6 | Quinn Pham | 2 | Intel (CMake fixes) |
| 7 | Umang-projects | 1 | LoopSplit pass |

**Meta wrote µTLX; kernelize.ai polished and ships the scaffolding.** This is a real pattern — you see similar cross-company collaboration on the LLVM plugin subsystem.

---

## 4. Deep-dive: a pipelined GEMM under TLX

The user chose "one kernel, deep." The canonical choice is **`hopper_gemm_pipelined.py`** (`repos/triton-ext/extensions/utlx/tlx/tutorials/hopper_gemm_pipelined.py`, 234 lines): minimal TLX, no warp specialization, just software-pipelined async loads feeding `wgmma`. It's the smallest kernel that exercises the core TLX surface. We'll then step up to `hopper_gemm_ws.py` (403 lines) to see what WS adds.

### 4.1 Baseline: what stock Triton does

The stock Triton matmul (`repos/triton/python/tutorials/03-matrix-multiplication.py`) is the k-loop + `tl.dot` pattern. The key point: stock Triton's `num_stages` autotune knob lets the **compiler pipeliner** insert async copies and stage-buffer accumulators automatically. On Hopper, this maps to `cp.async.bulk.tensor` + `wgmma` with overlap handled by the `tt::gpu::AsyncCopyGlobalToLocalOp` + `tt::nvidia_gpu::WarpGroupDotOp` scheduling in `third_party/nvidia/lib/TritonNVIDIAGPUTransforms/Pipeliner/`. This **works well** up to a point, but it's automatic — users can't steer which operands live in registers vs shared memory, can't split an MMA across warp groups, and can't do anything explicit with cluster/CGA-scoped multicast or CLC.

### 4.2 The TLX pipelined kernel — annotated

```python
# hopper_gemm_pipelined.py:83-183, TLX imports at :3-5
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx  # sys.modules hijack enables this
```

**Explicit SMEM allocation** — instead of leaving buffer management to the pipeliner:
```python
# :123-126
buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_STAGES)
buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_STAGES)
```
These lower to `ttg::LocalAllocOp` with a **shape of `(NUM_STAGES, M, K)`** — this is the explicit multi-stage buffer pool the pipeliner would have created behind your back. The important bit is that you now have **names** for each stage and can reason about scheduling.

**Prologue** — prefetch NUM_STAGES-1 stages:
```python
# :129-140
for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
    a = tlx.local_view(buffers_A, i)                     # SMEM slice for stage i
    b = tlx.local_view(buffers_B, i)
    token_a = tlx.async_load(a_ptrs, a, mask=...)        # cp.async.bulk
    token_b = tlx.async_load(b_ptrs, b, mask=...)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
    tlx.async_load_commit_group([token_a, token_b])      # cp.async.commit_group
```
`loop_unroll_factor` says "unroll this prologue fully". `tlx.async_load` maps to `ttg::AsyncCopyGlobalToLocalOp` with a token (`uTLXPlugin.cpp:510-538`); `async_load_commit_group` groups tokens so a later `wait_group(N)` can release "older than the last N commits."

**Steady state** — one load group in flight per pipeline depth:
```python
# :143-173
acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=0):   # num_stages=0 DISABLES auto-pipelining
    buf = k % NUM_STAGES
    a_k = tlx.local_view(buffers_A, buf)
    b_k = tlx.local_view(buffers_B, buf)

    tlx.async_load_wait_group(NUM_STAGES - 2)   # wait until strictly fewer than (N-2) outstanding

    acc = tlx.async_dot(a_k, b_k, acc)          # wgmma.mma_async

    i = k + NUM_STAGES - 1
    a_next = tlx.local_view(buffers_A, i % NUM_STAGES)
    b_next = tlx.local_view(buffers_B, i % NUM_STAGES)

    acc = tlx.async_dot_wait(1, acc)            # wait for all-but-1 prior wgmma
    token_a = tlx.async_load(a_ptrs, a_next, mask=...)
    token_b = tlx.async_load(b_ptrs, b_next, mask=...)
    tlx.async_load_commit_group([token_a, token_b])
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
```
The `num_stages=0` on the outer loop is how you tell the **Triton** pipeliner to keep its hands off — you are taking over. `async_dot_wait(N, acc)` is an mbarrier wait threaded through `acc` to enforce an SSA dependency the compiler can't break.

**Epilogue**:
```python
# :176-183
acc = tlx.async_dot_wait(0, acc)                # final drain
c = acc.to(tlx.dtype_of(c_ptr))
tl.store(c_ptrs, c, mask=c_mask)
```

**What this kernel gets you vs stock Triton:**
1. **Explicit** stage count unhooked from the autotune space. The stock pipeliner would decide per config; here you compose.
2. **Register pressure control** via `num_warps=8` + `NUM_STAGES=3` (matches CUTLASS H100 GEMM pipeline-3 with 128×256×64 tiles).
3. **Direct `wgmma.mma_async` emission.** `async_dot` selects `version = 5 if cuda_compute_capability >= 100 else 3` at `mma_ops.py:113` — version 3 lowers to `ttng::WarpGroupDotOp` (`wgmma`) on Hopper, version 5 to `ttng::TCGen05MMAOp` on Blackwell. No layout-heuristic round-trip.
4. **Decoupled wait from dot** via `async_dot_wait(1)` — you can arrange the wait before the next batch of `async_load`s, so cp.async issue and wgmma drain overlap.

What it **doesn't** get you over stock Triton on a typical H100 GEMM: **most of the time, the stock pipeliner finds the same schedule.** If you diff the generated PTX you'll see comparable instructions with comparable issue patterns. The TLX win over pipelined-stock is ~0-5% on well-tuned GEMM shapes. The value is **predictability**: you know exactly what the kernel does, and you can go one step further into explicit warp specialization.

### 4.3 Warp specialization — `hopper_gemm_ws.py`

Same kernel family; the structural difference starts at line 148:

```python
# hopper_gemm_ws.py:148-316 (abridged)
with tlx.async_tasks():
    with tlx.async_task("default"):        # producer — default warps
        # persistent tile loop, TMA descriptor loads into a[buf], b[buf]
        # handshake: barrier_wait(empty, phase) ... barrier_expect_bytes(full, bytes)
        #            ... async_descriptor_load(a_desc, buf, offset, full)
        tlx.async_descriptor_load(a_desc, data_a, [offset_am, offset_k], full_a)

    with tlx.async_task(num_warps=4, replicate=2):   # consumer — 4 warps x 2 MMA groups
        # handshake: barrier_wait(full, phase)
        # wgmma via async_dot
        acc = tlx.async_dot(data_a, data_b, acc)
        acc = tlx.async_dot_wait(tl.constexpr(0), acc)
        # release: barrier_arrive(empty)
```

Key structural features:

1. **Producer/consumer split with mbarriers.** `bars_empty_a/b` and `bars_full_a/b` are allocated with `tlx.alloc_barriers(num_barriers=N, arrive_count=K)`. A producer fills a buffer → `barrier_arrive(full)`; consumer waits `barrier_wait(full, phase)`, does mma, `barrier_arrive(empty)`. Phase flips each time the consumer completes `N` iterations (`_get_bufidx_phase` at lines 19-22). **This is the literal CUTLASS 3.x `Pipeline` pattern.**

2. **`replicate=2` splits M across 2 MMA groups.** Each `async_task` with `replicate=2` produces **two** warp groups, each with `num_warps=4` — matching Hopper's 2×WG "pingpong" idiom. The consumer writes to `acc = tl.zeros([BM // 2, BN])` (`:266`) — each replica computes half the M dimension.

3. **Persistent grid.** `while tile_id < num_tiles: tile_id += NUM_SMS` — a cooperative persistent launch. Paired with the `NUM_SMS` block count at grid construction (`:374-375`).

4. **Optional 2-CTA multicast.** When `NUM_CTAS == 2`, each CTA loads half of B and **multicasts** via TMA `multicast_targets=[cta_id, cta_id ^ 1]` (`:213-219`). Cluster synchronization via `tlx.cluster_cta_rank()` + `cta_bars`.

5. **Optional epilogue subtile.** `tl.split(acc)` after a `tl.reshape` + `tl.permute` lets the consumer write two N/2-wide stores, which is how CUTLASS hides the final store behind the next tile's load.

**What this *actually* buys you on a Hopper GEMM** — per Meta's own Jan 2026 blog (`pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/`), the `autoWS` (which lowers to TLX-like constructs) gets 1.5-2× over stock Triton on **B200 flash-attention forward**. No equivalent GEMM number has been published. In our own understanding — WS on GEMM helps most when the kernel has **imbalanced work** between tiles (e.g., causal masking). For dense square GEMM, stock Triton's pipeliner is within single-digit percent of hand-WS on H100. The WS kernel is bigger, harder to debug, and only pulls ahead when you need the asymmetry (producer does TMA + address math, consumer does wgmma + epilogue) — attention, grouped GEMM, and persistent-kernel workloads benefit; dense GEMM at SOTA tile shapes usually doesn't.

### 4.4 What the compiler emits

Two places to look:

- **After `make_ttgir`**: you see `tlx.*` ops translated to `ttg::` / `ttng::` ops by the `utlx_convert_triton_to_tritongpu` pass (`passes/nvidia/ConvertTLXToTritonGPU.cpp`). `tlx.async_load` → `ttg.async_copy_global_to_local`, `tlx.async_dot` → `ttng.warp_group_dot`, `tlx.alloc_barriers` → `ttg.local_alloc` + loop of `ttng.init_barrier`.
- **After `make_llir`**: the usual PTX pattern — `cp.async.bulk.tensor.2d.shared::cluster.global` + `mbarrier.*` + `wgmma.mma_async.sync.*`.

The pretty-printer `PrintTTGIRToTLX.cpp` (`extensions/utlx/tlx/dialect/lib/Transforms/PrintTTGIRToTLX.cpp`, 1780 LOC) does the reverse — runs after the Triton pipeliner and pretty-prints TTGIR as TLX-style code. Meta's `pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/` stated goal: "convert Triton TTGIR to readable TLX kernels for easier debugging and further performance hand-tuning." This is a **nice** debuggability story, genuinely useful for people chasing the last few percent.

---

## 5. PyTorch integration: the truth about #178917

The brief said pytorch/pytorch#178917 was "the PR to add TLX to PyTorch." **It's not a PR, it's an issue**, and the scope is narrower than the framing implies. From the issue body:

> *"This would mainly require building Triton with TRITON_EXT_ENABLED=1; and then including the triton-ext repo as it's own package … For the moment we are really only concern with including the TLX extensions ... so we do not need to include the entire triton-ext repo."*

Release Mode is stated as "Out-of-tree." This is a **release-packaging** ask from Corbin Robeck to Andrey Talman (PyTorch release eng). **No** PyTorch source changes are proposed. **No** Inductor integration discussed. **No** benchmark numbers cited. Only two substantive comments:

- **Nick Riasanovsky** raises three real open questions ([comment](https://github.com/pytorch/pytorch/issues/178917#issuecomment-4199756617)): (a) platform/arch test coverage under `TRITON_EXT_ENABLED`, (b) opt-in vs default inclusion, (c) "Does the user get TLX by default? Does the user have any way to turn off TLX? Given this is a new process we probably need some way to bisect if results are being impacted by the extensions."
- **CRobeck** ([reply](https://github.com/pytorch/pytorch/issues/178917#issuecomment-4199832859)) concedes (b) and (c) are open. Flags a concrete shipping constraint: *"There are size limits to pip packages that I think we've already exceeded in the past. So also important to get an idea of what the package size increase is from added symbols."*

The load-bearing upstream change is **[triton-lang/triton#9935](https://github.com/triton-lang/triton/pull/9935)**: a **+1/-0 line** PR that turns on `TRITON_EXT_ENABLED` in the wheel build, merged 2026-04-08 and cherry-picked to release/3.7.x. That's the "over-the-top TLX install" story in its most literal form — **one line in a build config.** Everything else (the plugin ABI, µTLX's hooks, the dialect) was already in place.

**For the report's narrative** — the value isn't "TLX becomes available in PyTorch wheels" (it's still not default-on, and µTLX isn't a first-class Inductor target); it's "the wheel's libtriton exports enough symbols that a third-party wheel can link against them." This is infrastructure, not a feature launch.

---

## 6. Critical analysis

### 6.1 What's genuinely new

1. **The no-fork extension ABI** is the real story. Prior to #8401/#8523/#9626, every dialect-/pass-/op-level extension required a fork. µTLX is the **proof-of-existence** that a plugin can register a dialect, lowering passes, and Python-callable ops and splice them into compilation on stock Triton. *Correction (see status banner):* what rides the ABI fork-free is the plugin-owned `tlx` dialect + its passes + the stages hook — **not** the Hopper/Blackwell value-add surface (TMA/named-barrier/`async_dot`), which is fork-coupled to `triton-tlx-core-changes` and reaches it only via the fork / `fbtriton`.
2. **Custom Python-callable ops** (#9626) fill a genuinely missing capability. Before, a plugin could only affect IR that the user had already produced; now it can surface a primitive that emits a custom op directly from `@triton.jit` code.
3. **Composable explicit pipelines.** The combination of `local_alloc(NUM_STAGES)` + `async_load_commit_group` + `async_load_wait_group(N-2)` + `async_dot_wait(1)` gives you the control knobs CUTLASS has had for 3 years, directly in Python. That's a real usability improvement for kernel authors who were writing one-off forks.

### 6.2 What's repackaging (honest)

1. **Warp specialization.** CUTLASS 3.0's `PipelineTmaAsync` (`cutlass/include/cutlass/pipeline/pipeline.hpp`) is the same producer/consumer pattern µTLX implements in `hopper_gemm_ws.py`. `tlx.async_tasks` + `async_task(num_warps, replicate)` is essentially the ThunderKittens `warp::producer`/`warp::consumer` macros with a Python-context-manager syntax. This doesn't make TLX less useful — it makes it approachable in Python — but it isn't a new scheduling primitive.
2. **`wgmma` / `tcgen05` bindings.** `tlx.async_dot` → `ttng.warp_group_dot` on Hopper and `ttng.tcgen05_mma` on Blackwell. The upstream dialect already had these ops; TLX exposes them directly rather than producing them from `tl.dot` layout inference. That's a clarity improvement, not a new instruction.
3. **Named barriers, cluster, CLC, TMA descriptors.** All NVIDIA-defined instructions. TLX is a front end for them; it isn't the only or first such front end (CuTe DSL, Mojo, Pallas Mosaic GPU all have them).

### 6.3 Competitors

| Project | Positioning | Key differentiator vs TLX |
|---|---|---|
| **Gluon** (OpenAI) | Lower-level dialect on the Triton stack | Exposes **layout encodings** as first-class; sits between TLX's explicit-pipeline view and ttgir's implicit-layout view. Same Blackwell primitives, different ergonomic target. See [lei.chat post](https://www.lei.chat/posts/gluon-explicit-performance/). |
| **ThunderKittens 2.0** (HazyResearch) | Embedded C++ template library | Not Python. Tile-first abstractions at register-file level; gives you `tile<bf16, 16, 64>` as the primitive. Better for C++ kernel ecosystems. |
| **CUTLASS 3.x + CuTe / CuTeDSL** | NVIDIA canonical | Broader primitive set, deeper NVIDIA investment, but C++ or Python-DSL that doesn't plug into Triton. The source of most patterns TLX borrows. |
| **Pallas + JAX Mosaic GPU** | JAX-embedded kernel DSL | Similar "explicit pipelining" story for Hopper, in JAX instead of Python. |
| **Helion** (Meta) | Higher-level DSL that lowers to Triton | Sibling project, not a competitor — goes the other way, simplifying for the 80% case. |
| **Tawa** (NVIDIA CGO'26) | Research: automatic warp specialization via async references | Claims 1.1× over cuBLAS GEMM, matches CUTLASS FA3 on H100. **Does not benchmark against TLX** in the paper. |

The interesting question is **Gluon vs TLX**. The common framing is "alternatives" but they are **orthogonal**:
- **Gluon** exposes layout encodings as a first-class Python primitive — explicit *layouts*. It lives in-tree (OpenAI's bet).
- **TLX** exposes producer/consumer scheduling, mbarrier handshake, and explicit pipeline stages — explicit *scheduling*. It lives out-of-tree (Meta's bet).

A serious kernel author may want both at once: Gluon-level layout control + TLX-level pipeline control. That's not currently a single library, but the design pieces don't conflict. Today, Gluon has better upstream integration and community visibility (Feb 2026 Lei Zhang deep-dive; active HN threads); TLX has the `async_tasks` context-manager ergonomics and the `PrintTTGIRToTLX` debug pretty-printer. Your choice depends on what you're trying to control. If you don't know which axis is your bottleneck, use Triton's auto-pipeliner first.

### 6.4 Performance claims — skeptic's scoreboard

Public numbers we have (all Meta-authored unless noted, all H100 unless noted):

| Claim | Number | Source | Our read |
|---|---|---|---|
| TLX 2-Simplicial fwd vs "pure-Triton" baseline | 588 vs 337 TFLOPs → **1.74×** | [pytorch.org/blog/fast-2-simplicial-attention](https://pytorch.org/blog/fast-2-simplicial-attention-hardware-efficient-kernels-in-tlx/) | **Niche kernel, no prior optimized baseline.** 2-simplicial attention is a new Meta algorithm; the "Triton baseline" was an unoptimized reference by the same team. This is not a head-to-head vs FA3/cuDNN/cuBLAS. |
| TLX FA-style, WS only | 590 TFLOPs | same | Effectively on par with the 2-simplicial number; suggests the WS overhead is well-hidden. |
| TLX FA + pipelining | 680 TFLOPs | same | +15% over WS alone — pipelining wins exist. |
| TLX FA + pipelining + pingpong | 717 TFLOPs | same | +21% over WS alone. **Within ~4% of FA3's 750 TFLOPs (Dao).** |
| FA3 (Dao et al.) | ~740-750 TFLOPs | [tridao.me/blog/2024/flash3](https://tridao.me/blog/2024/flash3/) | **TLX is not ahead of FA3 on the standard workload.** |
| autoWS (lowers to TLX-like WS) on B200 | 1.5-2× vs stock Triton on FA fwd | [pytorch.org/blog/warp-specialization-in-triton](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) | Compelling number but the reference is stock Triton, not FA3 or cuDNN. Same blog says cuDNN is still 10-20% ahead. |
| KPerfIR (OSDI'25, adjacent Meta work) | +24.1% over vanilla Triton FA3 on H100 | [arxiv.org/abs/2505.21661](https://arxiv.org/abs/2505.21661) | Not TLX specifically; cited as evidence Meta's Triton perf work is real and peer-reviewed. |
| **µTLX minimal pipelined GEMM (this report, §8)** | **0.91–0.99× of stock Triton on RTX 5090 (sm_120)** | FP16 GEMM 2k³–8k³, matched 72-config autotune | own measurement, `benchmarks/results/blackwell_*.json` | Within noise of stock. We could only test pipelining on the 5090 because async_dot is sm_100-gated. Genuine TLX value-prop (warp spec + tcgen05) is **not exercised** here; result mostly indicates the auto-pipeliner is hard to beat on dense GEMM. |

**What's missing from the public record:**
- **No TLX vs cuBLAS or CUTLASS GEMM numbers.** GEMM is the canonical test and there isn't one.
- **No independent reproduction** of TLX numbers by anyone not affiliated with Meta.
- **The HN thread** for the TLX conference talk has 3 points and 0 comments — very low external pickup compared to Gluon's threads.
- **AMD numbers** — µTLX has active `[TLX][AMD]` work (PR #288/#289 by Karthikeyan Manivannan) but nothing public to cite.

### 6.5 Real limitations

1. **Blackwell-only for the interesting stuff.** TMEM, `async_dot_scaled`, CLC, stochastic round, 2-CTA multicast GEMM — all require sm_100+. Meta's own blogs only publish H100 numbers (no TMEM used). The claim "unlocks SOTA on Blackwell" is partly aspirational — we don't yet have public post-launch Blackwell benchmarks from TLX specifically (autoWS B200 FA is the nearest).
2. **No Ampere fallback.** Our 3090 (sm_86) can't run any shipped tutorial. `tlx.async_dot` wants `wgmma` (Hopper) or `tcgen05` (Blackwell) — no `mma.sync` tier.
3. **AMD story is partial.** The barrier conversion passes (`LowerBarrierOps.cpp`, `BarrierOpConversion.cpp`) are commented out in `extensions/utlx/CMakeLists.txt:200-203` with the inline comment that they `# require patched triton` (specifically `ReadBarrierPhaseOp` and `ArriveBarrierOp` with `expectedCount`, neither of which exists in upstream). `async_dot` on AMD requires gfx950/gfx1250 (the freshest chips). Anyone on CDNA2/CDNA3 currently is still forking.
4. **No GPU CI on triton-ext.** CI runs on a CPU-only ubuntu-22.04. Real tests need manual `pytest` on a local Hopper/Blackwell box.
5. **Monkey-patching fragility.** The three hijacks (`sys.modules`, `knobs.runtime.add_stages_inspection_hook`, `WITH_DISPATCH.update`) all touch private-looking Triton internals.
6. **`require_layout` is stubbed** (`uTLXPlugin.cpp:489-493`). The "layout injection" story in the TLX dialect is aspirational for now.
7. **Monolithic drop.** PR #57 was +49,227 LOC; follow-up #61 (`7813400`) had to fix registry-name inconsistencies the original missed. Still stabilizing.
8. **Docs are thin.** `tlx_barriers.md` (335 LOC with PTX mapping tables) is the best doc. No API reference. No migration guide from the fork TLX.
9. **Weekly pin-bump as evidence of fragility.** Of 54 commits on `triton-ext` `main`, 10 are by `github-actions[bot]` — i.e. roughly weekly forced bumps to keep up with Triton main. Empirically, in §8 we hit two API-drift breaks (semantic-method renames, op-signature change) building against current main, requiring ad-hoc Python and C++ patches before µTLX would compile, let alone run. The ABI is genuinely stable; the *glue* is genuinely not.
10. **`triton-ext.conf` shipping detail.** Plugin discovery depends on these `.conf` files being on `TRITON_EXT_CONF_DIRS` at install time — fragile across wheel-based installs and source-builds.

---

## 7. Concrete asks for the Meta team

Short list. Each is a real friction we hit; not "advice."

- **Publish a GEMM-vs-cuBLAS number.** Plain `M=N=K=16384` FP16 GEMM on H100 and B200. It's the table every adopter asks for. The absence is conspicuous.
- **Stabilize the three hijack points** (`sys.modules` namespace, `WITH_DISPATCH`, stage-replacement) as first-class `triton.plugins` API. The empirical break-rate is roughly weekly (10 of 54 triton-ext commits are pin-bumps). Even one of these absorbed upstream cuts the maintenance tax.
- **GPU CI on triton-ext.** Current CI is CPU-only ubuntu-22.04. Both API drifts we hit in §8 would be caught by a single H100 smoke test.
- **`tlx.async_dot` fallback for sm_120 (and Ampere `cp.async` path generally).** Consumer Blackwell isn't a niche — every non-data-center adopter is on it. Hard-asserting `tmem` on `>= 100` shuts that audience out.
- **One-page "what differs from stock Triton" doc.** `tlx_barriers.md` is a good template; the rest of the surface needs the same treatment.
- **Clarify TLX vs µTLX vs Gluon positioning publicly.** Internally the answer is "they solve different slices" (see §6.3), but adopters end up guessing.

---

## 8. Reproduction: benchmarks/

See `./benchmarks/README.md`. On 3090 (Ampere) we can run a **stock-Triton GEMM baseline** only. The TLX kernels are shipped as heavily annotated references; running them requires Hopper or Blackwell and µTLX installed against a Triton build with `TRITON_EXT_ENABLED=1`.

### What we measured — stock Triton 3.6.0 FP16 GEMM

FP16 × FP16 → FP16, 15 warmup + 80 measured iterations, TF/s via CUDA events.

**3090 (Ampere, sm_86):**

| Shape (MxNxK) | Triton TF/s | cuBLAS TF/s | ratio |
|---|---:|---:|---:|
| 2048×2048×2048 | 71.1 | 61.9 | 1.15 |
| 4096×4096×4096 | 71.2 | 68.4 | 1.04 |
| 8192×8192×8192 | 74.2 | 70.4 | 1.06 |
| 8192×2048×2048 | 70.7 | 66.4 | 1.06 |
| 2048×2048×8192 | 63.7 | 67.5 | 0.94 |

**RTX 5090 (Blackwell consumer, sm_120) — stock Triton (built from source 3.7.0+gitf1ff6575) vs cuBLAS vs µTLX. All three at warmup=25, iters=100, identical autotune search space (72 configs: BM∈{64,128}, BN∈{64,128,256}, BK∈{32,64}, num_stages∈{2,3,4}, num_warps∈{4,8}):**

| Shape (MxNxK) | Stock Triton TF/s | cuBLAS TF/s | µTLX pipelined TF/s | TLX/stock | TLX best cfg |
|---|---:|---:|---:|---:|---|
| 2048×2048×2048 | 190.2 | 178.3 | 174.0 | **0.91×** | BM=128, BN=128, BK=32, NS=2 |
| 4096×4096×4096 | 218.9 | 211.7 | 210.1 | **0.96×** | BM=64, BN=64, BK=32, NS=2 |
| 8192×8192×8192 | 220.9 | 204.6 | 218.8 | **0.99×** | BM=128, BN=128, BK=32, NS=2 |
| 8192×2048×2048 | 220.3 | 210.7 | 211.7 | **0.96×** | BM=64, BN=128, BK=32, NS=2 |
| 2048×2048×8192 | 188.2 | 182.2 | 177.1 | **0.94×** | BM=128, BN=128, BK=32, NS=2 |

**The honest headline:** with a *matched* autotune space, µTLX-explicit-pipelining is within **1-9%** of stock Triton on dense FP16 GEMM, with bit-identical correctness. (An earlier version of this benchmark gave the TLX kernel a much smaller search space — only BM=128 and BK=64 — and produced a misleading 15-26% gap. Reviewer caught it; we re-ran. Lesson: autotune-space disparity is a real source of bench bias, especially when µTLX kernels favor BK=32 / BM=64 configs that a "TLX-shaped" config list often forgets.)

**What this is and isn't evidence of:**
- It **is** evidence that "explicit pipelining with `tlx.async_load` + user-controlled `commit_group/wait_group` + stock `tl.dot`" loses to Triton's built-in pipeliner on well-trodden GEMM shapes. The compiler has more information about layout choices and register pressure than the explicit path exposes.
- It **isn't** a test of the full µTLX value proposition. The configurations µTLX was designed to exploit are:
  1. `tlx.async_dot` → `wgmma`/`tcgen05` direct-to-SMEM MMAs, but `tcgen05` requires a **TMEM-backed accumulator** gated to sm_100 (B200). The 5090 is sm_120 consumer Blackwell and crashes with `assert src.type.storage == tlx.storage_kind.tmem`. No Hopper-style `wgmma` path for sm_120 either.
  2. Warp specialization via `async_tasks` — same TMEM/wgmma gating applies to the MMA consumer.
  3. TMA via `async_descriptor_load` — likely works on 5090 but every shipped WS GEMM tutorial couples it with `async_dot`.
- To see a TLX *win* over stock Triton on GEMM, you need H100 (sm_90, wgmma with reg-acc path) or B200 (sm_100, tcgen05 with TMEM). 5090 consumer Blackwell is the least representative Blackwell for this story, which is itself a usability finding.

### Build-path findings (documented because the effort to reach those numbers was real)

Between "stock Triton pip wheel" and "µTLX plugin runs on sm_120" we hit four real-world issues. Each is worth knowing:

1. **The pip wheel is unusable as a µTLX host.** Triton 3.6.0 pip wheel doesn't set `-DTRITON_EXT_ENABLED=1` so the `mlir::triton::plugin::loadPlugins` symbol is compiled but not linked into the distributed `libtriton.so`. µTLX's `dlopen` silently no-ops. This is fixed in 3.7.0+ wheels via PR #9935 but not in 3.6.0 or earlier. **Fix:** build Triton from source with `TRITON_EXT_ENABLED=1`.
2. **`-fvisibility=hidden` + MLIR static linking hides conversion-pattern symbols.** The pre-built LLVM/MLIR that Triton downloads was compiled with hidden visibility; even with `TRITON_EXT_ENABLED=1` (which skips the `-fvisibility=hidden` on Triton's own files), MLIR's `ConversionPattern::matchAndRewrite` stays LOCAL in `libtriton.so`. Any plugin that uses `OpConversionPattern` fails to resolve it. µTLX dodges this by never using `ConversionPattern` directly — it goes through generic `OperationState`.
3. **µTLX's `PluginInfo` is missing the `tritonVersion` field** that was added to upstream between µTLX's last refactor and Triton main. Without a fix, the first plugin load path crashes with `basic_string: construction from null is not valid` from a nullptr C-string in the version check. **Fix** (two-way): either pin Triton to a commit that predates the `tritonVersion` field (we used `f1ff6575`, the hash in `ci/triton-hash.txt`), or patch µTLX to set the field.
4. **`createAsyncLoad` emits `ttg.async_copy_global_to_local` without `operandSegmentSizes`.** The op has `AttrSizedOperandSegments` trait, so the generic-builder path in `NewOps.cpp:createAsyncLoad` produces an op that fails verification with "operand count (3) does not match with the total size (0) specified in attribute 'operandSegmentSizes'". **Fix:** one-line patch to add `operandSegmentSizes = [1, 1, mask?, other?]` to the NamedAttribute list. We're including the patch in `benchmarks/_utlx_async_load_segsizes.patch` for reference.

Also: µTLX uses two `TritonSemantic` private methods that were removed/renamed upstream (`_prepare_legacy_load`, `dot_precheck`). We shim them from Python at `benchmarks/_utlx_shim.py`. This is the "monkey-patching fragility" called out in §6.5 limitation #5 — empirically confirmed.

**Takeaway:** getting µTLX to compile and run on a stock build today takes 4 distinct patches. The "install over the top" story is real for the *plugin ABI* but the supporting glue between µTLX and current Triton main is *not* yet smooth, and the CI is CPU-only so API-drift issues like #4 above aren't caught upstream.

**What we cannot measure on the 3090:** nothing with TLX primitives. `tlx.async_dot` demands `wgmma` (Hopper) or `tcgen05` (Blackwell). Running `tlx_pipelined_gemm_annotated.py` without `utlx_plugin` installed fails explicitly at the import line; running it with `utlx_plugin` installed on Ampere would fail at compile time in `mma_ops.py:113`. This is a hardware-ceiling limit, not a setup bug.

### Scripts

- `bench_baseline_gemm.py` — stock Triton FP16 GEMM vs torch matmul (ran on 3090 above).
- `tlx_pipelined_gemm_annotated.py` — annotated copy of `hopper_gemm_pipelined.py`; sm_90+ only.
- `tlx_ws_gemm_annotated.py` — annotated copy of `hopper_gemm_ws.py`; sm_90+ only.
- `run_local.sh` — drives the 3090 baseline.
- The RTX 5090 numbers above were produced by the out-of-tree plugin source build (now retired in favour of `fbtriton`); raw results are in `benchmarks/results/blackwell_stock.json` + `blackwell_tlx.json`. The runnable-today path is `pip install fbtriton` + `benchmarks/fbtriton_tlx_gemm_demo.py` / `fbtriton_tlx_ws_gemm_demo.py` — see `WRITEUP.md` and `HOWTO.md`.

---

## 9. Conclusions

1. **The extension architecture is the story.** TLX is the flagship user, but the long-term value is that anyone — vendors, researchers, silicon startups — can add dialects/passes/ops against stock Triton. This kills a real pain point.
2. **µTLX is research-stage but serious.** Code is dense, thoughtfully structured, and being actively maintained. PRs are small and reviewed. Meta should not ship this as "1.0" yet, and the existing "under construction" README language is correctly calibrated.
3. **Performance story is credible, not revolutionary.** On Hopper, TLX FA is on par with FA3. On Blackwell, autoWS (which lowers to TLX-like WS) beats stock Triton by 1.5-2× on FA but trails cuDNN by 10-20%. GEMM numbers are conspicuously absent. This is a solid contribution, not a SOTA-shattering one.
4. **The critical value is operator-author productivity.** If a team was going to fork Triton for Hopper/Blackwell warp specialization, TLX now saves them that cost. If they were going to use CUTLASS anyway, the value is smaller.
5. **The PyTorch integration headline is overblown.** #178917 is a release-packaging ticket, not a feature. The one-line wheel flag (#9935) is the real enabling change.
6. **Gluon convergence is the open question.** If OpenAI promotes Gluon as the in-tree explicit-layout target, and TLX stays out-of-tree as the explicit-WS target, they serve different users. If Gluon absorbs most of TLX's surface, µTLX becomes a transitional artifact.

---

## Appendix A: Verified assertions and file anchors

Every non-trivially-sourced claim in this report maps to one of:

- Path relative to `repos/triton-ext/` (e.g., `extensions/utlx/uTLXPlugin.cpp:695-807`)
- Path relative to `repos/triton/` (e.g., `include/triton/Tools/PluginUtils.h`)
- Triton PR number (e.g., #8401, #8523, #9626, #9935)
- PyTorch issue #178917 or its two inline comments
- Named Meta PyTorch blog URL
- Commit SHA in either repo

Unverified or partially-verified claims flagged inline with "user should confirm" or "no public evidence found."

## Appendix B: Open questions

1. **Internal TLX GEMM numbers.** Are there H100/B200 GEMM vs cuBLAS numbers internally that could be published?
2. **Helion lowering path.** Helion-to-TLX vs Helion-to-Triton-auto-WS — which is the default and how does it affect the TLX adoption story?
3. **Release-3.7.x cherry-pick list.** Is there a plan to cherry-pick #9748 (`PluginInfo*` ABI) to 3.7.x, or is µTLX explicitly 3.8-targeting?
