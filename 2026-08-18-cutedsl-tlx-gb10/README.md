# CUTLASS 4.7 CuTe DSL (Task Scheduling + Primitives) vs Triton TLX `async_task`

**Date:** 2026-08-18 · **Machine:** DGX Spark — NVIDIA GB10, sm_121a, aarch64 Grace, CUDA 13.0

## Brief

CUTLASS 4.7's README claims two new CuTe DSL features: **Primitives** and **Task
Scheduling**. The goals of this experiment:

1. Get CUTLASS 4.7 + CuteDSL running on this machine.
2. Compare **Task Scheduling** to **TLX's `async_task`** approach — get a kernel
   leveraging each, and compare them properly (code shape, safety model, performance).
3. Understand how you'd use **Primitives**, given that inline PTX was already possible
   in CuteDSL before.

## Headline results

- Both stacks run on the GB10 — with caveats. CuteDSL auto-targets sm_121a; TLX needed
  a source build plus two local patches (CUDA-13 ptxas, PTX ISA 8.8 mapping). Neither
  framework's *asynchronous* tensor-core path exists on sm_121 (no tcgen05/wgmma), so
  both comparison GEMMs use synchronous MMA consumers.
- Matched warp-specialized fp16 GEMMs (TMA producer → SMEM ring → tensor-core
  consumers) reach effective cuBLAS parity at 4096³: **TLX-WS 88.4 TF, TS 84.2 TF,
  cuBLAS 87–93 TF**. The TS kernel is, as far as we know, the first TS GEMM on the
  sm_12x class (all shipped TS GEMM tutorials are tcgen05-only).
- The frameworks are philosophical opposites. The same missing-release bug was injected
  into both: **TS rejected it at compile time** (`ConsumerWait 0 is blocked!`); **TLX
  compiled it silently** — correct results at K=128, hard GPU deadlock at K=512.
- **Primitives** is inline PTX's replacement: typed 1:1 NVVM ops with a three-tier
  escape hatch (`prims.*` → `prims.dialect.*` → `prims.inline_ptx`) and a compile-time
  hazard checker (mbarrier over-arrive is a hard *error* with a suggested fix; a
  would-hang `expect_tx` is a warning). None of that is possible with asm strings.

## Contents

| Path | What |
|---|---|
| `REPORT.md` | The full report: programming models, code samples, benchmarks, the safety experiment, Primitives-vs-inline-PTX, verdict |
| `LABNOTES.md` | Execution log / lab notebook: what was done, in order, including failures and fixes |
| `code/ts/` | The TS warp-specialized GEMM (`gemm_ws_ts.py`), two compile-time-rejected broken variants, captured static analysis (`compile_diagnostics.txt`, `broken_output.txt`), bench results |
| `code/tlx/` | The TLX `async_task` GEMM (`gemm_ws_tlx.py`), the silently-compiling broken variant, minimal producer/consumer template, bench results |
| `code/primitives/` | `three_ways.py` (escape-hatch tiers), `hazard_demo.py` + captured diagnostics, `tile_mma_prims.py` (pure-SIMT tensor-core tile) |
| `code/README.md` | Environment setup + run commands for everything |

Code was developed and verified on the machine above; the venvs and cloned sources it
runs against live outside this repo (see `code/README.md` and `LABNOTES.md`).
