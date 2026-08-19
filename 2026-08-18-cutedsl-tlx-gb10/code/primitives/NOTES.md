# CUTLASS 4.7 Primitives API — demo suite notes

Environment: NVIDIA GB10 (DGX Spark), sm_121a, CUDA 13.0, `nvidia-cutlass-dsl==4.7.0`.
Everything below was run and verified on this machine with:

```
PY=/home/ianbarber/Projects/cute/.venv-cutedsl/bin/python
$PY three_ways.py       # ~15 s (one compile)
$PY hazard_demo.py      # ~1 min (seven compiles); rewrites hazard_output.txt
$PY tile_mma_prims.py   # ~20 s
```

`prims` = `from cutlass.experimental import primitives as prims` — a typed 1:1
Python wrapper over the MLIR NVVM dialect, usable inside any `@cute.kernel`
alongside normal CuTe DSL code. The point of this suite: *you could always
inline PTX; what does the Primitives ladder buy you over that?*

## 1. `three_ways.py` — the escape-hatch ladder on one instruction

One warp butterfly-sum, written three ways in a single kernel, all three verified
to produce 496 on GPU:

| tier | call | what you write | what checks you get |
|---|---|---|---|
| wrapped | `prims.shfl_sync(mask, val, delta, 0x1F, prims.Shfl.BFLY)` | typed args; Python `int` coerces; `"bfly"` string or StrEnum; result type inferred from `val` | trace-time `ValueError` on bad masks/offsets/kinds; op visible to compiler + hazard checker |
| dialect | `prims.dialect.shfl_sync(cutlass.Int32.mlir_type, Int32(mask), val, Int32(delta), Int32(0x1F), prims.dialect.ShflKind.bfly)` | the same NVVM op minus conveniences: explicit MLIR result type first, raw dialect enum, no literal coercion (only `.ir_value()` objects auto-convert) | dialect verifier only |
| inline PTX | `prims.inline_ptx("shfl.sync.bfly.b32 {$w0}, {$r0}, {$r1}, {$r2}, {$r3};", write_only_types=[Int32], read_only_args=[...])` | the instruction as a string | none — opaque blob |

## 2. `hazard_demo.py` — the compile-time hazard checker (verbatim output in `hazard_output.txt`)

Enabled via `CUTE_DSL_COMPILER_OPT="warnings{nvvm,ptx} remarks{nvvm,ptx}"` (set
before importing cutlass). What actually fired on this build:

| section | injected bug | diagnostic | severity / effect |
|---|---|---|---|
| A | `mbarrier_arrive` executed by all 32 threads on a count=1 barrier | `error[nvvm-diag:C3/C4]: mbarrier arrive reaches a count=1 barrier from multiple threads` + suggestion to gate with `if nvvm.elect_sync():` | **error — compile fails** (`CompilerDiagnosticError`), even without any flag |
| B | `mbarrier_init(mbar, 32)` but a single elected arrive | two `warning[nvvm-diag:C3]` (static arrive count 1 vs init 32) | warning — compiles (would hang if launched; we never launch it) |
| C | `mbarrier_arrive_expect_tx(mbar, 128)` with no TMA/`complete_tx` | `warning[nvvm-diag:C5]: ...no completion source for 128 registered transaction bytes` | warning — compiles (would hang if launched) |
| D | `elect_sync()` nested inside an elected region | `error[nvvm-diag:C16]` | **error — compile fails**. Rough edge: the headline renders as the literal placeholder `{0}` (unformatted message table entry) |
| E | `elect_sync()` under `if tidx < 16:` divergence | *nothing* — C13 (partial-warp elect) did **not** fire in this build, also tried `if lane == 0:` | compiles clean |
| F | corrected kernel (elected single arrive, count matches) | no diagnostics; runs on GPU, output verified | clean |
| G | runtime-indexed 256-element register array | `remark[ptxas]: ptxas detected local memory usage`, source frame pointing at the `cutlass.Array` allocation line | remark — compiles |

All diagnostics come with Rust-style source frames (`--> file.py:line:col`, caret,
2 context lines), a `suggestion:` and a PTX ISA doc link. Two operational gotchas
we hit: (1) diagnostics are emitted on **OS-level stderr by the C++ compiler**, so
capturing them needs fd redirection, not `contextlib.redirect_stderr`; (2) the env
var's `diagnostic=` selector is applied *after* per-`cute.compile(options=...)`
selectors and clobbers them — enable all domains you want once, in the env var.

## 3. `tile_mma_prims.py` — Tensor Cores through SIMT (the sm_121 path)

Single-CTA 64×64×64 fp16 GEMM, fp32 accumulate, verified vs fp64 `torch.matmul`
(max |err| = 5.7e-6). No CuTe layouts, no `TiledMMA`, no atoms — the kernel is
CUDA-C++-with-intrinsics written in Python:

- **TMA load**: `create_tensor_map_tiled` host-side (swizzle `none`);
  `mbarrier_init` → `mbarrier_arrive_expect_tx(bytes_A+bytes_B)` →
  2× `cp_async_bulk_tensor_shared_cta_global` — all issued by *warp 0's one
  elected lane* (the gating hazard_demo section A shows the checker enforcing) —
  then everyone spins `mbarrier_try_wait_parity(mbar, 0)`.
- **Compute**: 4 warps, warp *w* owns rows `[16w, 16w+16)`. Per warp:
  `ldmatrix.x4` on A (lane addr `&tile[lane%16][(lane//16)*8]` yields the
  m16n8k16 A-fragment order a0..a3 directly), `ldmatrix.x4` on staged Bᵀ
  (regs 0/2 and 1/3 are the two k-halves of two adjacent n-atoms), then
  `prims.mma_sync` per 16×8 atom: 4 k-steps × 8 n-atoms = 32 MMAs/warp.
- **Epilogue**: each lane stores its 4-value fp32 C fragment straight to global.

What `TiledMMA` normally derives is all hand-written here: fragment ownership per
lane, issuer election, addressing, synchronization. `prims.mma_sync` is barely
wrapped — operands are raw `ir.Value`s, so the ldmatrix i32 register words must be
`llvm.bitcast` to `vector<2xf16>` and the fp32 result struct unpacked with
`llvm.extractvalue`. **Bug found while building this**: omitting
`multiplicand_{a,b}_ptx_type` (letting the dialect infer `.f16` from the operand
types, which the docstring says is the default) hard-aborts the MLIR library
(SIGABRT, no Python traceback) during IR construction on this build — always pass
them explicitly. Also: plain `range()` loops inside `@cute.kernel` are staged as
*dynamic* loops (`error[PHASE_DYNAMIC_INDEX]` when they index Python lists) — use
`cutlass.range_constexpr` for unrolled fragment loops.

## 4. When to reach for prims vs inline PTX vs `cute.arch`

Grounded in what this suite actually observed: reach for **wrapped `prims.*`**
whenever the op exists there — we got trace-time `ValueError`/`TypeError`s for bad
enum kinds and masks, StrEnum arguments instead of constraint strings ( `"=f,f"` ),
and, decisively, the hazard checker: section A's over-arrive was rejected *at
compile time* with a fix suggestion, something structurally impossible for an asm
blob — rewrite section A's arrive as `inline_ptx("mbarrier.arrive.shared.b64 _, [{$r0}];")`
and the checker sees nothing. Drop to **`prims.dialect`** only for NVVM ops the
wrapper hasn't covered; you give up literal coercion and must supply MLIR result
types, but stay compiler-visible (three_ways tier 2 is deliberately identical
hardware-wise to tier 1). Drop to **`prims.inline_ptx`** only for instructions or
multi-op sequences not in the NVVM dialect at all — it is the sanctioned last
resort and still how much of the older `cute.arch` layer is built internally: 66
`llvm.inline_asm` call sites live in `cute/arch/`, e.g. `nvvm_wrappers.py`'s

```python
llvm.inline_asm(T.f32(), [Float32(a).ir_value()],
                "ex2.approx.ftz.f32 $0, $1;", "=f,f",
                has_side_effects=True, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT)
```

— raw constraint letters, manual result typing, invisible to every check above.
`cute.arch` itself remains fine for the curated basics (`thread_idx`, `warp_idx`,
`sync_threads` — this suite uses it for indices), but it is a hand-picked subset;
Primitives is the full NVVM surface with one consistent, checkable calling
convention. Caveats observed: the API is experimental (C16's `{0}` headline, the
mma_sync inference abort, C13 not firing), and NVIDIA labels it transitional
"until a CUDA Python-like solution is available".

## Files

- `three_ways.py` — ladder demo, results verified identical (496 ×3)
- `hazard_demo.py` + `hazard_output.txt` — captured diagnostics A–G
- `tile_mma_prims.py` — TMA→ldmatrix→mma.sync single-tile GEMM, verified vs torch
