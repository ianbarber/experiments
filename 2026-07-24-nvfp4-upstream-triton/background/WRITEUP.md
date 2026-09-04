# TLX on Triton: what the extension architecture actually delivers

*A skeptical, source-grounded walk through Meta's TLX and the triton-ext plugin
ABI — what runs on stock upstream Triton, what needs a source build + plugin,
what needs the fork, and how to use it today. Every claim is reproduced on real
hardware where we have it (RTX 3090, RTX 5090, GB10, Strix Halo); the H100
`async_dot` path is evidenced by the published `triton-utlx` 3.8 packaging +
upstream gist, pending first-hand H100 confirmation. Companion how-to:
[HOWTO.md](HOWTO.md).*

## TL;DR

- **The plugin ABI is real.** A Triton plugin can register a dialect, passes, and
  ops, and splice passes into compilation via the supported
  `knobs.runtime.add_stages_inspection_hook`. TLX's portable surface rides this.
- **As of Triton 3.8, the `async_dot` (wgmma) path runs on mainline *source*, not
  only on the fork.** Build upstream `triton-lang/triton` @ 3.8 with
  `TRITON_EXT_ENABLED=ON`, `pip install triton-utlx==3.8.0`, load `libutlx.so`.
  No fbtriton, no `triton-tlx-core-changes` patch. That is a real change from the
  June 2026 writeup.
- **Still not the stock PyPI wheel.** `pip install triton` does not export the
  MLIR symbols the plugin resolves. "Mainline Triton source" yes; "stock PyPI
  triton" no.
- **fbtriton remains the zero-friction full surface** (`pip install fbtriton` —
  TLX in-tree, TMA + warp-spec + named barriers). Prefer it when you want one
  command and the exotic ops.
- **Do not overclaim the whole surface.** Warp-spec (`tlx.async_tasks`) is still
  broken on upstream (swallowed `WITH_DISPATCH` import). TMA-descriptor /
  prefetch / DSMEM / named barriers need re-probe on 3.8; many were silent
  no-ops or fork-coupled in June.
- **Hardware ceiling unchanged.** `async_dot` needs H100 (`wgmma`) / B200
  (`tcgen05`). Consumer Blackwell (5090/GB10) validates install + cp.async
  staging; MMA still won't run there.
- **Honest perf story (5090, dense GEMM, fbtriton):** TLX ties the stock
  auto-pipeliner. Value is explicit control for asymmetric work — and the path
  to tensor-core schedules on H100/B200.

---

## 1. The premise, and the trap

TLX (Triton Low-level Extensions) gives Triton warp-aware, hardware-close
primitives — explicit shared-memory buffers, `async_load`/TMA, mbarriers, warp
specialization, direct `wgmma`/`tcgen05`. It ships three ways:

1. **Meta's fork** (`facebookexperimental/triton`) as **`pip install fbtriton`** —
   TLX native, in-tree, full surface.
2. **µTLX PyPI plugin** — `triton-utlx==3.8.0` on **upstream 3.8 source-built
   with `TRITON_EXT_ENABLED=ON`** — headline `async_dot` + cp.async staging.
3. **Out-of-tree source build** of `triton-lang/triton-ext` (`extensions/utlx`) —
   same ABI; public tree lags the PyPI packaging on the MMA DSL rewrite (see §2).

The obvious blog demo: "here's a kernel TLX makes faster/possible that stock
Triton can't do." Chasing that grounded a lot of dead ends worth recording.

### Dead end 1 — "stock Triton can't async-load this"
False. Compiling probe kernels (`benchmarks/probe_pipeliner.py`) and inspecting
TTGIR/PTX shows upstream's auto-pipeliner async-pipelines non-dot reductions,
indirect gathers, and predicated loads — once `num_stages` is set. The *only*
zero-async case is a plain loop with no `num_stages`, a one-line fix. The naive
framing collapses on contact with the compiler.

### Dead end 2 — "TMA reduction-store / prefetch is TLX-only"
Stock Triton already exposes `tensor_descriptor.atomic_add` (TMA reduce-store)
and `desc.load` (TMA load). The genuinely-TLX-only TMA op, `async_descriptor_
prefetch_tensor`, was a **silent no-op** on upstream in June — the op isn't in
the dialect (`benchmarks/tma_prefetch_demo.py` shows flat 1.00×). Re-test vs 3.8
before treating that as settled forever.

### Dead end 3 — "just use `@gluon.jit`"
The TMA/MMA builder methods are Gluon-only on upstream, so June's `@triton.jit`
plugin kernels `AttributeError`d on `create_warpgroup_mma`. `@gluon.jit` didn't
fully fix it either (wrapper impedance). **As of `triton-utlx` 3.8, the wheel's
DSL no longer needs this path for `async_dot`** — see §2.

---

## 2. What changed in 3.8: plugin-owned MMA, not "Gluon became reachable"

### The June root cause (still true of *public* `triton-ext` HEAD DSL)

At the June pin (`triton-ext` `a22c3d0`), `async_dot` lowered like this:

```python
# mma_ops.py — public tree, still present on origin/main as of mid-2026
# Use gluon: create_warpgroup_mma(...)
output = _semantic.builder.create_warpgroup_mma(A_handle, B_handle, ...)
```

`create_warpgroup_mma` lives only on `GluonOpBuilder` (`gluon_ir.cc`);
`@triton.jit`'s `TritonOpBuilder` lacks it → `AttributeError`. The fork's base
builder is enriched, so `@triton.jit` + TLX "just works" there. That analysis
was airtight for the code we had.

### The 3.8 dissolve

Two pieces:

1. **Upstream [#10754](https://github.com/triton-lang/triton/pull/10754)**
   (landed by `8929ee53` / 3.8.0): plugin-registered ops are bound on
   `TritonOpBuilder` as `create_<op.name>` — so a plugin op named
   `utlx_warp_group_dot` becomes **`create_utlx_warp_group_dot`**.

2. **`triton-utlx==3.8.0` (PyPI, 2026-07-10)** rewrote the DSL to call those
   **plugin ops**, not Gluon builders:

   ```python
   # wheel: utlx_plugin/mma_ops.py
   output = _semantic.builder.create_utlx_warp_group_dot([...])  # not create_warpgroup_mma
   ```

   The wheel's `libutlx.so` implements `createWarpGroupDot` → constructs
   `ttng::WarpGroupDotOp` inside the plugin. MMA is **plugin-owned IR
   construction**, reachable from `@triton.jit` without the fork and without
   `@gluon.jit`.

**Important precision:** this is *not* "Gluon MMA/TMA methods became reachable
on the classic builder." The PyPI packaging **stopped calling Gluon** for the
headline path. Public `triton-ext` main can still lag (Gluon comments + missing
`utlx_warp_group_dot` registration vs the wheel). The reproducible mainline
story is **`pip install triton-utlx==3.8.0`**, not "clone triton-ext HEAD."

### Chain that runs the gist (Hopper persistent GEMM)

1. `git clone triton-lang/triton && git checkout 8929ee53` (3.8.0)
2. `TRITON_EXT_ENABLED=ON pip install -e . --no-build-isolation` — static-LLVM
   build so `libtriton.so` exports MLIR symbols the plugin will resolve
3. `pip install triton-utlx==3.8.0` — `libutlx.so` + `utlx_plugin` DSL
4. `export TRITON_PLUGIN_PATHS=.../libutlx.so`; `import utlx_plugin as tlx`
5. Semantic shims (`dot_precheck`, `_prepare_legacy_load`,
   `_unwrap_if_constexpr`) — same class as `benchmarks/_utlx_shim.py`
6. Kernel: `local_alloc`/`local_view` SMEM ring → `async_load` (cp.async) →
   commit/wait groups → `async_dot` (wgmma) → `async_dot_wait`. Persistent,
   multi-stage.

### Scoreboard (upstream 3.8 source + plugin)

| µTLX surface | On upstream 3.8 + plugin | Basis |
|---|---|---|
| `async_load` (cp.async) + commit/wait, `local_alloc`/`local_view` | ✅ runs | gist + 5090 work |
| `async_dot` → wgmma (H100) | ✅ now runs — headline | gist / PyPI packaging |
| `async_dot` → tcgen05/TMEM (B200) | ✅ presumably (same binding) | inferred; not in gist |
| warp-spec `tlx.async_tasks()` | ❌ still broken | `WITH_DISPATCH` absent; swallowed try/except |
| TMA-descriptor load/store, TMA prefetch, DSMEM, named barriers | ❓ re-test | June: no-op / fork-coupled; gist uses pointer+cp.async |
| Stock `pip install triton` wheel | ❌ no | no `TRITON_EXT_ENABLED`; MLIR symbols not exported |

What *still* rides only the durable ABI without drama: the plugin's own `tlx`
dialect (storage-alias / reuse-group), its passes, LoopSplit / ArithmeticIntensity,
and the stages hook. Full June inventory (pins stale):
[`docs/utlx-upstream-state.md`](docs/utlx-upstream-state.md).

---

## 3. How to use TLX today: two legitimate paths

### Path A — zero friction: `pip install fbtriton`

The fork as a wheel. TLX in-tree. TMA, warp-spec, named barriers, tutorials.
Verified on `fbtriton` (PyPI 3.6.x / runtime `3.6.0+fb.beta`). One command, no
source build. See [HOWTO.md](HOWTO.md).

### Path B — durable ABI story: upstream 3.8 source + `triton-utlx`

```bash
git clone https://github.com/triton-lang/triton && cd triton
git checkout 8929ee53   # 3.8.0
pip install -r python/requirements.txt
TRITON_EXT_ENABLED=ON pip install -e . --no-build-isolation
pip install triton-utlx==3.8.0
export TRITON_PLUGIN_PATHS=$(python -c \
  'import utlx_plugin,os; print(os.path.join(os.path.dirname(utlx_plugin.__file__),"libutlx.so"))')
```

This is the stronger *blog* framing: the plugin ABI delivers async MMA without
maintaining a fork. It is **not** "install over stock pip triton." For H100
runbook details: [`docs/h100-handover.md`](docs/h100-handover.md).

---

## 4. The demos — running on a consumer RTX 5090 (sm_120)

Both via `pip install fbtriton` (runtime `triton 3.6.0+fb.beta`), `@triton.jit`,
FP16, autotuned, correct (maxerr ~3e-04). These still need the fork (or a future
plugin that ports TMA-descriptor + warp-spec); they prove the schedule surface
on hardware we have.

### 4a. Explicit TMA-pipelined GEMM (`benchmarks/fbtriton_tlx_gemm_demo.py`)

A `NUM_STAGES`-deep SMEM ring (`tlx.local_alloc`), real TMA loads
(`tlx.async_descriptor_load` → `cp.async.bulk.tensor`, which stock `tl.load` never
emits), explicit mbarrier handshake, `tl.dot` (mma.sync) consumer.

| shape | TLX-TMA TF/s | stock TF/s | cuBLAS TF/s | TLX/stock |
|---|--:|--:|--:|--:|
| 2048³ | 175.8 | 194.1 | 178.0 | 0.91× |
| 4096³ | 201.6 | 209.8 | 207.2 | 0.96× |
| 8192³ | 217.9 | 218.4 | 201.6 | 1.00× |

Ties the auto-pipeliner at scale (and beats cuBLAS at 8192³).

### 4b. Warp-specialized GEMM (`benchmarks/fbtriton_tlx_ws_gemm_demo.py`)

The schedule the single-knob auto-pipeliner **cannot** express: dedicated
**producer** warps issue TMA loads while separate **consumer** warps do the MMA,
via `with tlx.async_tasks(): with tlx.async_task("default"): … / with
tlx.async_task(num_warps=4): …` and an mbarrier (full/empty) handshake. Consumer
is `tl.dot` so it runs on sm_120. **This path is still fork-only** (upstream
warp-spec remains broken).

| shape | WS-TLX TF/s | cuBLAS TF/s | WS/cuBLAS |
|---|--:|--:|--:|
| 2048³ | 157.3 | 178.2 | 0.88× |
| 4096³ | 191.0 | 211.2 | 0.90× |
| 8192³ | 191.9 | 209.5 | 0.92× |

It runs correctly on the 5090. On dense GEMM it doesn't beat the auto-pipeliner
(WS wins on *asymmetric* work — attention, grouped GEMM). With `tlx.async_dot`
on an H100/B200, this is the structure behind Meta's published flash-attention
numbers — now reachable on mainline source via Path B for the non-WS half.

---

## 5. Hardware reality

TLX is a front end; the MMA instruction is still gated by compute capability
(triton `AccelerateMatmul.cpp` `getMMAVersionSafe`):

| cc | arch | MMA | `async_dot` |
|---|---|---|---|
| 90 | **H100** | v3 `wgmma` | ✅ Hopper path |
| 100–119 | **B100/B200** | v5 `tcgen05`+TMEM | ✅ Blackwell path |
| 120–129 | 5090 / GB10 | v2 `mma.sync` | ❌ (`// Exclude consumer Blackwell`) |

So everything *up to* the tensor-core MMA — TMA pipelining, named barriers
(fork), warp-spec scaffolding (fork), cp.async staging (plugin) — can run on the
5090. `async_dot` itself needs H100 or B200.
[`docs/h100-handover.md`](docs/h100-handover.md).

---

## 6. The honest scoreboard

- **Genuinely good / new:** the no-fork plugin ABI; as of 3.8, **async MMA on
  upstream source via `triton-utlx`** (durable ABI delivering the value-add);
  explicit pipelines + warp specialization as Python context managers (fork);
  `fbtriton` making the full surface `pip install`-able.
- **Repackaging (fine):** warp specialization is CUTLASS's producer/consumer; the
  `wgmma`/`tcgen05` bindings are NVIDIA instructions exposed directly.
- **Overstated until proven:** TLX-vs-cuBLAS/FA3 GEMM numbers don't exist
  publicly; published wins are flash-attention WS on H100/B200, ~on par with
  FA3. On dense GEMM (5090, here) TLX ties the auto-pipeliner. Do not let
  "async_dot runs on mainline" imply "whole TLX surface runs on mainline."
- **Still fragile / fork-coupled:** warp-spec, and (until re-probed) TMA-prefetch,
  named barriers, DSMEM — not upstream dialect ops.

**Bottom line for a kernel author:**

1. Prefer Triton's auto-pipeliner first.
2. Need explicit WS / full TLX today with zero build pain → **`fbtriton`**.
3. Need to show the **plugin ABI** delivering async MMA without a fork →
   **upstream 3.8 source + `triton-utlx`** (and an H100/B200 for the MMA).
4. Don't claim stock `pip install triton` + plugin.

---

*Reproduction: `benchmarks/` (demos + probes), `patches/` (recovered core delta),
`docs/` (inventory, hardware analysis, H100 handover). Boxes: 3090 (Ampere),
5090 (sm_120), GB10 (sm_121), Strix Halo gfx1151. Updated 2026-07 for Triton
3.8 / `triton-utlx` dual path.*
