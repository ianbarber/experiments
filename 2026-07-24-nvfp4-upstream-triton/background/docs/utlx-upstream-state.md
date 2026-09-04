# µTLX against upstream Triton — full inventory of passes, ops & DSL

> **Status banner (2026-07 — read first).** This file is the **June 2026**
> inventory at pins triton `a18b1bb3` / triton-ext `a22c3d0`. It remains the
> deepest op/pass forensic map, but two headline claims are **superseded**:
>
> 1. **`async_dot` (wgmma) now runs on mainline Triton *source*** built with
>    `TRITON_EXT_ENABLED=ON` + **`triton-utlx==3.8.0`** (no fork, no core patch).
>    Mechanism: upstream #10754 binds plugin ops as `create_<name>`; the PyPI
>    wheel rewrites MMA to `create_utlx_warp_group_dot` (plugin-owned), not
>    Gluon `create_warpgroup_mma`. Stock **pip** triton still fails.
> 2. **Usage paths:** prefer [`../HOWTO.md`](../HOWTO.md) Path A (`fbtriton`,
>    full surface) or Path B (3.8 source + `triton-utlx`, async_dot story). The
>    `@gluon.jit` workaround below is **legacy forensics** for the old public
>    DSL, not the recommended Path B.
>
> Warp-spec remains broken on upstream (`WITH_DISPATCH`). TMA-prefetch / named
> barriers / DSMEM still need **re-probe on 3.8**. Current narrative:
> [`../WRITEUP.md`](../WRITEUP.md) §2; runbook: [`h100-handover.md`](h100-handover.md).

---

**Scope (historical).** What the out-of-tree **µTLX plugin** (`triton-lang/triton-ext`,
`extensions/utlx`) provided against **upstream Triton at June pins**, op by
op, pass by pass. Built and reproduced on an RTX 5090 (sm_120).

| Component | Commit (June inventory) |
|---|---|
| `triton-ext` HEAD | `a22c3d0` (2026-06-22), pins Triton `a18b1bb3` |
| Triton (pinned) | `a18b1bb3` → `triton.__version__ == 3.8.0` (pre-#10754 packaging era for our probes) |
| µTLX origin | PR #57 `7181951` (2026-03-30), pinned Triton `283dc318` |

**The June deployment model.** µTLX was built for upstream Triton **plus a
`triton-tlx-core-changes` core patch — fork-only.** The plugin names that patch
in its own error strings (`passes/nvidia/PingPong.cpp:354`, AMD barrier passes):
when a needed op is missing it prints *"Apply triton-tlx-core-changes patch."*
That patch lives in `facebookexperimental/triton` and was **verified absent
from upstream `main` and release branches** at inventory time. It adds
`ttng`/`ttg`/`amdgpu` ops (named barriers, etc.).

So everything below is measured against **genuinely-stock upstream (no patch)**
at those pins — the forensic question for the extension ABI. Under that
condition the plugin ABI + no-core-change op subset work; many value-add ops
were inert *by design*. The July `triton-utlx` packaging changes the
`async_dot` row; it does **not** auto-fix warp-spec or every exotic string-name
op.

The `triton-tlx-core-changes` patch is recovered from the fork in `../patches/`.
Design reference: [arXiv:2605.10905](https://arxiv.org/abs/2605.10905).

---

## 0. TL;DR (June inventory — see banner for 3.8 deltas)

- **Works (genuinely useful, upstream-supported):**
  - The **custom-stages pipeline hook** — `knobs.runtime.add_stages_inspection_hook`
    is a real upstream extension point; µTLX uses it to inject passes into
    `ttir`/`ttgir`/`llir`. This is the cleanest, fully-functional integration.
  - The plugin's **own `tlx` dialect** (7 ops / 2 types / 2 attrs): the
    storage-alias / reuse-group / require-layout machinery, plus the passes that
    lower it (`utlx_storage_alias_lowering`, `utlx_rewrite_local_alias`,
    `utlx_propagate_layout`, `utlx_convert_triton_to_tritongpu`,
    `utlx_insert_and_propagate_layout`).
  - Repo-level passes **LoopSplit** (a real loop-bisection transform) and
    **ArithmeticIntensity** (a real roofline analysis pass).
  - `tlx.async_load` / barriers / `local_alloc` — they lower to *real upstream
    ops* (needs `_utlx_shim` + the `operandSegmentSizes` patch).
  - **Under `@gluon.jit`** (June out-of-tree fallback): TMA-descriptor / TMEM /
    Gluon MMA surface reachable because `GluonOpBuilder` has the `create_*`
    methods. Validated on the 5090 for TMA load/store (§4c). **July alternative
    for `async_dot`:** `triton-utlx==3.8.0` uses plugin-owned `create_utlx_*`
    instead — see WRITEUP §2.
- **Broken / inert against upstream (silent) at June pins:**
  - **Warp specialization** (`async_tasks`) — relies on a removed `WITH_DISPATCH`
    codegen hook; registration is a swallowed `ImportError`, and the fallback
    raises `TypeError`. **Still true on 3.8** (re-confirmed: symbol absent).
  - **~15 exotic ops** (remote shmem, CLC, named barriers, generic fence,
    clock64, vote-ballot, `async_tma_prefetch`, …) — emitted by string name into
    `ttng`/`ttg`, which don't define them upstream → print "not registered",
    return null, **silently no-op**. Re-probe on 3.8 before restating.
  - Under **`@triton.jit`**, the **June public-tree** TMA/TMEM/tcgen05/MMA DSL
    `AttributeError`s because it calls Gluon-only `create_*` builder methods.
    (In `fbtriton` the base builder carries these methods. In **`triton-utlx`
    3.8** the wheel avoids Gluon for MMA via `create_utlx_warp_group_dot`.)
  - Two passes have commented-out / null-returning subpaths
    (`PropagateLayout` scales-TMEM; `ResolvePlaceholderLayouts` tmem-compat).
  - Two latent name bugs: `tt.make_tensor_desc` (upstream is
    `…make_tensor_descriptor`); CLC mnemonics don't match upstream's CLC API.

---

## 1. The two integration mechanisms

**(A) Standalone plugin passes** — `pass/`, `dialect/` built as separate `.so`
loaded via `TRITON_PLUGIN_PATHS`, registered through upstream
`triton/Tools/PluginUtils.h`. These are **manual / opt-tool** passes; they are
NOT auto-inserted into the normal compile pipeline. The loader is gated on
Triton being built with `TRITON_EXT_ENABLED=1` (off by default).

**(B) The custom-stages hook** — `utlx_plugin/__init__.py:202` sets
`knobs.runtime.add_stages_inspection_hook = custom_stages.inspect_stages_hook`.
This knob exists upstream (`knobs.py:492`) and **both** backends invoke it at the
end of `add_stages` (`third_party/nvidia/backend/compiler.py:589-590`,
`amd:598-599`) with `(self, stages, options, language, capability)`. The hook
then **replaces** the `stages["ttir"]/["ttgir"]/["llir"]` lambdas to splice in
`passes.plugin.utlx_*` (those plugin-pass names are registered from the `.so`'s
pass table via upstream `python/src/passes.cc:116`). **This is the legitimate,
fork-free way µTLX injects its passes, and it works.**

Build artifacts actually produced: `libutlx.so` (dialect + all 7 TLX transforms
compiled in), `libarithmetic_intensity.so`, `libloop_split.so`, `libexample.so`,
`libTritonExtensionSupport.so`. **Not** built: `TritonTLX`/`tlx/dialect/triton_tlx.cc`
(the real-op pybind path — excluded from the build graph), and the AMD barrier
passes (commented out in CMake). `backend/CMakeLists.txt` is empty.

---

## 2. Pass inventory

### Repo-level (mechanism A — standalone/manual)

| Pass | File | What it does | Upstream-safe |
|---|---|---|---|
| `triton-arithmetic-intensity` | `pass/ArithmeticIntensity/ArithmeticIntensity.cpp:749` | **Analysis only.** Traces loads/stores/`tt.dot` to compute bytes & FLOPs as affine exprs over trip counts; writes `tt.bandwidth`/`tt.compute` attrs. | ✅ |
| `triton-loop-split` | `pass/LoopSplit/LoopSplit.cpp:204` | **Real transform.** Splits an `scf.for` at the first IV-vs-invariant `cmpi` into two loops and folds the comparison per half. | ✅ |
| `example` (`example.zero`) | `dialect/Example/` | Dialect-registration scaffold, one `Pure` op. | ✅ |
| backend/ | — | empty | — |

### µTLX passes (mechanism B — injected via the hook), in `libutlx.so`

| Pass | File | What it does | Wired? | Upstream-safe |
|---|---|---|---|---|
| `utlx_convert_triton_to_tritongpu` | `uTLXConversionPatterns.cpp:824` | Plugin-local copy of upstream Triton→TritonGPU + extra legality for unencoded `local_store/load/async_copy`; AMD barrier rewrite. | ✅ ttir | ✅ |
| `utlx_insert_and_propagate_layout` | `uTLXConversionPatterns.cpp:935` | Propagates dot-required shared encoding backward through the memdesc chain to `LocalLoad`s. | ✅ ttgir (AMD) | ✅ |
| `utlx_storage_alias_lowering` | `tlx/dialect/lib/Transforms/StorageAliasLowering.cpp:37` | **Lowers the storage-alias ops** into real `LocalAlloc`/`TMEMAlloc` + offset rewrites (3 steps: size → offset → alloc). | ✅ llir | ✅ |
| `utlx_rewrite_local_alias` | `RewriteLocalAlias.cpp:27` | Resolves `tlx.local_alias` to the underlying alloc via `MemDescReinterpret`, sized to max. | ✅ llir | ✅ |
| `utlx_propagate_layout` | `PropagateLayout.cpp:63` | Dataflow layout propagation; lowers `tlx.require/release_layout`→`ConvertLayout`. | (registered) | ✅ except scales-TMEM block (`:149-169`) commented out → no-op |
| `utlx_fixup` | `Fixup.cpp:26` | Module validation + metadata; inserts `InvalBarrier` before returns; early-out if no TLX ops. | (registered) | ✅ |
| `utlx_print_ttgir_to_tlx` | `PrintTTGIRToTLX.cpp:1748` (1781 LOC) | **Pretty-printer** (no IR change): renders TTGIR back as Python-TLX source. | debug | ✅ (string-keyed, robust) |
| `utlx_prune_unused_barriers` | `passes/nvidia/PruneUnusedBarriers.cpp:124` | Erases mbarriers with no `wait` use; strips dead `warp_specialize` captures. | (registered) | ✅ |
| `utlx_ping_pong_prep` | `passes/nvidia/PingPong.cpp:391` | Tags "expensive" ops (`WarpGroupDot`/`exp2`) with `pingpong_id` attr. | (registered) | ✅ (attr-only) |
| `utlx_ping_pong_sync` | `passes/nvidia/PingPong.cpp:473` | Inserts named-barrier arrive/wait around ping/pong regions. | (registered) | ❌ **inert** — `triton_nvidia_gpu.{arrive,wait}_barrier_named` unregistered upstream; computes boundaries then bails ("Apply triton-tlx-core-changes patch"). |
| `utlx_insert_require_layout` | `InsertRequireLayout.cpp:27` | Inserts `require_layout` before LocalLoads feeding dots. | ❌ registered but not called (conversion-side used instead) | ✅ |
| AMD `LowerBarrierOps`, `BarrierOpConversion` | `passes/amd/*` | mbarrier→AMD lowering | ❌ **not built** (CMake commented) — reference `amdgpu::ReadBarrierPhaseOp`/extended `arrive_barrier` absent upstream | fork-only |

Latent hole: `utlx_resolve_placeholder_layouts` (`ResolvePlaceholderLayouts.cpp:144`)
returns a **null Attribute** on the `tmemCompatible=True` path (`:62-81`, TODO to
port `getTmemCompatibleLayouts`).

---

## 3. The plugin's own `tlx` dialect (genuinely owned, works)

Namespace `tlx`, registered by `registerTLXDialect` (`uTLXPlugin.cpp:691`).
**This is the one place µTLX adds real, working, plugin-owned IR.**

**Ops** (`tlx/dialect/include/IR/TLXOps.td`): `storage_alias_spec` (:27),
`storage_alias_local_alloc` (:73), `reuse_group` (:123), `set_buffer_overlap`
(:191), `require_layout` (:246), `release_layout` (:262), `local_alias` (:275).
**Types**: `!tlx.storage_alias_spec<kind[,size]>`, `!tlx.reuse_group<kind>`.
**Attrs**: `#tlx.dummy_register_layout<…>`, `#tlx.dummy_tmem_layout`.

Semantically this is **explicit shared-memory buffer aliasing / reuse and
layout pinning** — letting a kernel author declare that two logical buffers may
share physical SMEM, or force a specific layout on a memdesc. Stock Triton has
no user-facing equivalent. The Python surface is `tlx.storage_alias_spec`,
`tlx.local_alloc(..., storage_alias=…)`, `tlx.reuse_group`,
`tlx.require_layout`, all of which call the working `utlx_*` builder methods and
are lowered by the wired passes above. **This is the strongest candidate for a
"µTLX can, stock can't" demonstration that actually runs upstream.**

---

## 4. Op-builder surface — three failure modes

µTLX exposes ops two ways: the **plugin op table** (`uTLXPlugin.cpp:735-799`, 48
`utlx_*` methods attached to `TritonOpBuilder` via upstream `ir.cc:1868`), and a
**Gluon pybind module** (`triton_tlx.cc`, `create_*` methods — *not built*
against upstream). Against upstream a18b1bb3:

**4a. Works — lowers to a real upstream op (via `utlx_*` plugin methods):**
`local_alloc`/`local_view`/`local_load`/`local_store` (`ttg.*`),
`alloc_barriers`/`barrier_wait`/`barrier_arrive`/`barrier_expect_bytes`
(`ttng.{init,wait,arrive}_barrier`, `barrier_expect`), `async_commit_group`/
`async_wait` (`ttg.*`), `async_load` (`ttg.async_copy_global_to_local`),
`global_scratch_alloc`, `cluster_cta_rank` (`nvg.cluster_id`), `thread_id`,
`warp_group_dot_wait`, the `require_*`/storage-alias ops (own `tlx` dialect),
`make_tensor_descriptor` (native `ir.cc:1858`).
*Caveat:* `async_load`/`async_dot` also call `_semantic._prepare_legacy_load` /
`dot_precheck`, removed upstream — restored by `benchmarks/_utlx_shim.py`.

**4b. Silent no-op — emitted by string name into a `ttng`/`ttg` op that doesn't
exist upstream** (`NewOps.cpp` `createRuntimeOp` → prints
`utlx: op 'X' not registered`, returns null):

| String | Feature | Note |
|---|---|---|
| `ttng.async_tma_prefetch` | TMA L2 prefetch | absent (upstream has reduce/gather/scatter only) |
| `ttg.async_remote_shmem_store`, `ttg.remote_shmem_store` | DSMEM | absent |
| `ttng.map_to_remote_buffer` | DSMEM | absent |
| `ttng.async_store` | async global store | absent (`async_shared_store` exists) |
| `ttng.fence` | generic threadfence | absent (`fence_async_shared` exists) |
| `ttng.named_barrier_arrive/wait` | named barriers | absent |
| `ttng.async_clc_try_cancel`, `ttng.clc_query_cancel` | CLC persistent | **name mismatch** — upstream CLC is `clc_try_cancel`/`clc_is_canceled`/`clc_load_result` |
| `ttng.vote_ballot_sync`, `ttg.clock64`, `ttng.cluster_size_1d` | misc | absent |
| `tt.make_tensor_desc` (the desc-ptr variant) | TMA desc from ptr | **typo** — upstream is `tt.make_tensor_descriptor` |
| `amdg.read_barrier_phase` | AMD barrier | dialect not even registered |

**4c. Needs `@gluon.jit` — calls `create_*` builder methods that live on
`GluonOpBuilder`, not the base `TritonOpBuilder`.** This is the out-of-tree
harness issue. It was our best plugin-path workaround — but it is *not* how you
use TLX: `fbtriton` ships a build where the base `TritonOpBuilder` carries these
methods, so `@triton.jit` + TLX works directly (see the banner at the top and
`../WRITEUP.md`). Below is the forensic detail, verified by introspection on the
5090 (plugin loaded):

| Builder | `utlx_*` (plugin) | gluon `create_*` (TMA/TMEM/tcgen05) |
|---|---|---|
| `TritonOpBuilder` (`@triton.jit`) | ✅ | ❌ all absent |
| `GluonOpBuilder` (`@gluon.jit`) | ✅ (inherited) | ✅ all present |

| DSL call | gluon method it needs |
|---|---|
| `async_descriptor_load`/`store`/`reduce` | `create_async_tma_copy_*`, `create_async_tma_reduce` |
| TMEM `local_load/store`, `tmem_copy` | `create_tmem_load/store/copy/subslice` |
| `async_dot`/`async_dot_scaled`/`tcgen05_commit` | `create_tcgen05_mma(_scaled)`, `create_warpgroup_mma`, `create_tcgen05_commit` |
| `cluster_barrier`, `fence_async_shared`, `memdesc_subslice/trans` | `create_*` (Gluon) |

> Under **`@triton.jit`** these raise `AttributeError` (the base builder lacks
> the methods) — which is why `test_tlx.py` fails ~129/238 against upstream. The
> tests use the wrong decorator; they were written against the fork, whose base
> builder carried these methods. Under **`@gluon.jit`** the same `tlx.*` calls
> lower to real upstream ops. **Empirically validated on the 5090**: a
> `@gluon.jit` kernel calling `tlx.local_alloc` + `tlx.alloc_barriers` +
> `tlx.async_descriptor_load` + `tlx.async_descriptor_store` emits real
> `ttng.async_tma_copy_global_to_local` / `local_to_global` ops. (The remaining
> blockers to a fully-green kernel are authoring details — the host
> `TensorDescriptor`'s layout must match the SMEM `nvmma_shared` layout — plus a
> one-arg plugin bug in `async_descriptor_store_wait` calling
> `create_async_tma_store_wait(pendings)` where gluon wants `(pendings, bool)`.)

So the "interesting" TMA/TMEM/MMA surface is **not gone** — it requires the
gluon harness. What stays broken even under gluon: the §4b string-name ops
(emitted into absent `ttng`/`ttg` mnemonics → silent no-op, and `clock64`
actually segfaults), and the warp-spec `with tlx.async_tasks()` (§5).

---

## 5. Warp specialization (`with tlx.async_tasks()` / `async_task`) — triply inert

1. Intended dispatch via `WITH_DISPATCH` (`__init__.py:205` `_register_compiler_dispatch`)
   → `from triton.compiler.code_generator import WITH_DISPATCH` **ImportError**
   (symbol absent upstream) → swallowed by `except: pass`. Registration never happens.
2. Fallback to upstream generic `visit_With` (`code_generator.py:1051`):
   `cm = fn(*args, _semantic=self.semantic, **kws)` → `async_tasks.__init__(self)`
   (`async_task_utils.py:51`) rejects `_semantic` → **`TypeError`**.
3. Even if (1) worked, `visit_withAsyncTasks` imports `enter_sub_region` (absent
   upstream) and calls `create_warp_specialize_op`/`get_partition_region`/
   `create_warp_yield_op` (fork-only builder names; upstream's equivalent is the
   Gluon `create_warp_specialize`).

This has been true since PR #57 (the pin at #57, Triton `283dc318`, already had
no `WITH_DISPATCH` and the generic `visit_With`). There is **no older non-fork
version** where warp spec worked.

---

## 6. Root cause & honest framing

The plugin ABI (custom dialect, custom passes, the stages-inspection hook) is
**real and robust** — µTLX rides it successfully for its `tlx` dialect, its
lowering passes, LoopSplit, and ArithmeticIntensity. Where it falls down is that
the **distinctive TLX runtime features were never made plugin-owned**:

1. They target **Triton-native `ttng`/`ttg`/`amdgpu` ops** added by the
   **`triton-tlx-core-changes` core patch**, which is **fork-only — confirmed
   absent from upstream `main` and release branches `release/3.6.x` /
   `release/3.7.x` / `rel/3.7`** (greped the ttng/ttg dialect defs at those
   refs; none define `arrive/wait_barrier_named`, `async_tma_prefetch`,
   `ReadBarrierPhase`). The plugin signals this in two tiers: the named/AMD
   barrier ops say *"Apply triton-tlx-core-changes patch"* (`PingPong.cpp:354`),
   while the generic `NewOps.cpp` string path just warns *"not registered in this
   Triton build"* and returns null — same root cause, softer wording.
2. They lean on **Gluon-only builder methods** and **removed `TritonSemantic`
   internals** that an un-patched `@triton.jit` TLX kernel doesn't have. (The
   patch presumably restores these on the base builder — it's how the fork's
   `@triton.jit` tutorials work.)
3. The warp-spec syntax depends on a **removed/patch-provided codegen hook**.

So this is **not a hidden bug**: the plugin documents that the value-add needs
the `triton-tlx-core-changes` patch. The catch is that patch is unpublished
(only in the fork; not in any upstream release), and triton-ext's CPU-only CI
never lowers these on a GPU — so against genuinely-stock upstream the gap is
silent (warn/skip/no-op) rather than a hard error.

**For the blog (updated 2026-07):** the durable story is still the **stages
hook + plugin-owned dialect + real passes** — and, as of 3.8 + `triton-utlx`,
**plugin-owned `async_dot` (wgmma) on mainline source** as well. The cautionary
half is narrower: warp-spec and several exotic ops still reach into
fork-only / unregistered dialect surface; stock pip triton is not enough.

---

## 7. Asks for the µTLX maintainers

1. **Make the exotic ops plugin-owned** (define them in the `tlx` dialect) or
   land them upstream — stop emitting `ttng.X` by string and hoping.
2. **Fail loudly** when `createRuntimeOp` can't resolve an op (raise, don't
   warn-and-drop) — a silent no-op with correct-looking numerics is the worst case.
3. **Fix the active DSL's Gluon-only calls** — either bind those methods on
   `TritonOpBuilder`, or route TLX kernels through a Gluon builder.
4. **Re-implement `async_tasks`** against the current `visit_With` protocol (or
   target the upstream Gluon `warp_specialize`).
5. **GPU CI on triton-ext** — one sm_90/sm_100 smoke test that *lowers* a WS and
   a prefetch kernel catches every item here.
6. Drop `store_reduce`/`make_tensor_desc` typos; `desc.atomic_add` already exists
   upstream.

---

## 8. Build & reproduce

**To run TLX:** `pip install fbtriton`, then `benchmarks/fbtriton_tlx_gemm_demo.py`
/ `fbtriton_tlx_ws_gemm_demo.py` (see `../HOWTO.md`). No build. Everything below is
the **out-of-tree plugin source build** — the boundary case that validates the
extension ABI, not a path you need for using TLX.

See `benchmarks/README.md`. Build (RTX 5090): Triton `a18b1bb3` builds
from repo root (`pip install -e .`), needs
`LLVM_SYSPATH=~/.triton/llvm/llvm-62b7cf96-ubuntu-x64-2` (has `clang++` for the
mandatory GSan runtime). libutlx: source mode, `TRITON_SOURCE_DIR`+
`TRITON_BUILD_DIR`, env `LLVM_INSTALL_DIR=$LLVM_SYSPATH`,
`-DLLVM_EXTERNAL_LIT=.venv/bin/lit`; output `build/lib/libutlx.so`. Keep the
`-Werror`→`-Wno-error=attributes` and `operandSegmentSizes` patches and prepend
`import _utlx_shim`. Evidence exhibit: `benchmarks/tma_prefetch_demo.py` (flat
1.00× ⇒ prefetch no-op).
