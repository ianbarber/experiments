# Use TLX today

Two legitimate paths. Pick by goal:

| Goal | Path |
|---|---|
| Just run TLX (TMA, warp-spec, tutorials) with zero build | **A — `pip install fbtriton`** |
| Show async MMA on **mainline Triton source** via the plugin ABI | **B — Triton 3.8 source + `triton-utlx`** |
| Stock `pip install triton` + drop-in plugin | **Does not work** (no `TRITON_EXT_ENABLED`) |

Design ref: [arXiv:2605.10905](https://arxiv.org/abs/2605.10905). Full story:
[WRITEUP.md](WRITEUP.md).

---

## Path A — zero friction: `pip install fbtriton`

Meta's fork as a wheel. TLX baked in-tree (ops in `libtriton.so`, DSL +
tutorials). Full surface including warp-spec and named barriers.

### A1. Install (x86_64 + NVIDIA, Python 3.12)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python fbtriton numpy
```

(`fbtriton` installs as the `triton` package. cp310–cp314 manylinux x86_64.
On AMD, build the fork from source — no AMD wheel.)

### A2. Verify

```bash
.venv/bin/python - <<'PY'
import triton, torch, os, subprocess
print("triton", triton.__version__, "| GPU", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import triton.language.extra.tlx as tlx
print("tlx ops:", [a for a in ("async_descriptor_load","async_load","local_alloc","async_dot","alloc_barriers","make_tensor_descriptor") if hasattr(tlx, a)])
so = os.path.join(os.path.dirname(triton.__file__), "_C", "libtriton.so")
for s in ("ttng.arrive_barrier_named", "ttng.async_tma_prefetch"):
    ok = subprocess.run(["bash","-c",f"strings -a {so} | grep -qF '{s}'"]).returncode == 0
    print(s, "PRESENT" if ok else "ABSENT")
PY
```

Expect TLX ops listed and both `ttng.*` strings `PRESENT`.

### A3. Run the demo

```bash
.venv/bin/python benchmarks/fbtriton_tlx_gemm_demo.py
```

Explicit **TMA-pipelined FP16 GEMM** on consumer Blackwell (RTX 5090, sm_120):

| shape | TLX-TMA TF/s | stock TF/s | cuBLAS TF/s | TLX/stock |
|---|--:|--:|--:|--:|
| 2048³ | 175.8 | 194.1 | 178.0 | 0.91× |
| 4096³ | 201.6 | 209.8 | 207.2 | 0.96× |
| 8192³ | 217.9 | 218.4 | 201.6 | 1.00× |

Warp-specialized variant: `benchmarks/fbtriton_tlx_ws_gemm_demo.py` (also sm_120
with `tl.dot` consumer; ~0.88–0.92× cuBLAS on dense GEMM).

---

## Path B — mainline source + `triton-utlx` (async_dot story)

**Upstream** `triton-lang/triton` 3.8 built with `TRITON_EXT_ENABLED=ON`, plus
the self-contained PyPI plugin. No fbtriton, no core patch. Delivers cp.async
staging + **`async_dot` (wgmma on H100)**. Does **not** deliver warp-spec on
upstream today.

### B1. Build Triton 3.8 from source

```bash
git clone https://github.com/triton-lang/triton && cd triton
git checkout 8929ee53   # 3.8.0 pin used by the packaging / gist
pip install -r python/requirements.txt
TRITON_EXT_ENABLED=ON pip install -e . --no-build-isolation
```

The flag is required: default-off visibility means stock builds (and stock PyPI
wheels) do not export MLIR symbols `libutlx.so` will resolve at `dlopen`.

### B2. Install the plugin

```bash
pip install triton-utlx==3.8.0
export TRITON_PLUGIN_PATHS=$(python -c \
  'import utlx_plugin, os; print(os.path.join(os.path.dirname(utlx_plugin.__file__), "libutlx.so"))')
# optional during dev:
export TRITON_PLUGIN_VERSION_CHECK=off
```

### B3. Import

```python
import utlx_plugin as tlx   # or: import utlx_plugin; import triton.language.extra.tlx as tlx
# Semantic drift: see benchmarks/_utlx_shim.py (dot_precheck, _prepare_legacy_load)
# if compiling kernels that still hit removed TritonSemantic helpers.
```

### B4. What to run where

| Kernel shape | Hardware | Expect |
|---|---|---|
| `local_alloc` + `async_load` + commit/wait + `tl.dot` | 5090 / any recent NVIDIA | install + staging validation |
| `async_dot` (wgmma) persistent GEMM | **H100** | headline Path B proof |
| `async_dot` (tcgen05) | **B200** | same binding, not in the public gist |
| `with tlx.async_tasks()` | — | ❌ still broken on upstream |

H100 runbook: [`docs/h100-handover.md`](docs/h100-handover.md).

---

## Hardware gate (both paths)

| Want | TLX op | Needs |
|---|---|---|
| TMA / cp.async staging / mbarriers (consumer = `tl.dot`) | `async_descriptor_load`, `async_load`, `local_alloc`, … | sm_90+ incl. 5090/GB10 for non-`async_dot` |
| Tensor-core async MMA | `async_dot` → wgmma | **H100** |
| `async_dot` → tcgen05 + TMEM | `async_dot`, `tmem_*` | **B200** |
| Warp specialization | `async_tasks` / `async_task` | **fbtriton / fork** today (upstream broken) |

---

## More

- `WRITEUP.md` — dual-path narrative + mechanism (`create_utlx_*`).
- `docs/utlx-upstream-state.md` — June op inventory (stale pins; see WRITEUP §2).
- `docs/h100-handover.md` — datacenter runbook.
- `docs/next-steps.md` — open threads.
