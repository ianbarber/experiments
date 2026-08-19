#!/usr/bin/env python
"""
hazard_demo.py — CUTLASS 4.7 Primitives compile-time hazard checker, demonstrated
with real captured output on sm_121 (NVIDIA GB10).

Because prims.* ops are real NVVM dialect ops (not opaque asm strings), a C++
analysis inside the CutlassIR compiler can reason about synchronization protocol:
mbarrier arrive counts, elect.sync gating, expect_tx completion sources. Inline
PTX gets none of this.

Severities observed in this build:
  - errors  (always on, no flag needed)      -> compilation FAILS
  - warnings (opt-in: CUTE_DSL_COMPILER_OPT="warnings{nvvm}")  -> non-fatal
  - remarks  (opt-in: "remarks{ptx}")        -> ptxas perf notes (local mem/spills)

Sections:
  A. C3/C4  unguarded mbarrier_arrive on a count=1 barrier   -> ERROR, compile fails
  B. C3     init count 32 vs a single elected arrive          -> WARNING, compiles
  C. C5     arrive_expect_tx with no completion source        -> WARNING, compiles
  D. C16    nested elect_sync                                 -> ERROR, compile fails
  E. C13    partial-warp elect_sync attempt                   -> did NOT fire here
  F. corrected kernel: compiles clean with the same flags, runs, verified on GPU
  G. remarks{ptx}: runtime-indexed register array             -> ptxas local-mem remark

None of the broken kernels is ever launched (B and C would hang the GPU).
All compiler output is captured verbatim to hazard_output.txt.

Run:  /home/ianbarber/Projects/cute/.venv-cutedsl/bin/python hazard_demo.py
"""

import os
import re
import sys
import tempfile
from contextlib import contextmanager

# Opt in to nvvm synchronization warnings + ptxas perf remarks BEFORE importing
# cutlass. NOTE: the env var's `diagnostic=` selector is applied after (and
# overrides) any per-cute.compile(options=...) selector, so enable both domains
# here once rather than mixing env and per-compile selectors.
os.environ["CUTE_DSL_COMPILER_OPT"] = "warnings{nvvm,ptx} remarks{nvvm,ptx}"

import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hazard_output.txt")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_report: list[str] = []


@contextmanager
def capture_fds():
    """Capture OS-level stdout+stderr (the C++ compiler writes to fd 2)."""
    buf = tempfile.TemporaryFile(mode="w+b")
    sys.stdout.flush(); sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    os.dup2(buf.fileno(), 1); os.dup2(buf.fileno(), 2)
    try:
        yield buf
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        os.close(saved[0]); os.close(saved[1])


def section(title, compile_fn, expect):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    err = None
    with capture_fds() as buf:
        try:
            compile_fn()
        except Exception as e:  # CompilerDiagnosticError on error-severity diags
            err = e
        buf.seek(0)
        captured = buf.read().decode(errors="replace")
    if err is not None:
        outcome = f"compile FAILED with {type(err).__name__}"
        if not captured.strip():
            captured = str(err)
    else:
        outcome = "compile SUCCEEDED"
    print(captured, end="")
    print(f"--> outcome: {outcome}   (expected: {expect})")
    _report.append(f"{'=' * 78}\n{title}\n{'=' * 78}\n"
                   f"{_ANSI.sub('', captured)}--> outcome: {outcome}   (expected: {expect})\n")


def _compile(host_fn, dtype=cutlass.Int32, opts="--enable-tvm-ffi"):
    return cute.compile(
        host_fn,
        make_fake_compact_tensor(dtype, (32,), assumed_align=4),
        make_fake_stream(),
        options=opts,
    )


# --- A. C3/C4: every thread arrives on a count=1 mbarrier (needs elect gating) ---
@cute.kernel
def kernel_a(out: cutlass.Array):
    tidx, _, _ = cute.arch.thread_idx()
    mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    if prims.elect_sync():
        prims.mbarrier_init(mbar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    prims.mbarrier_arrive(mbar)  # BUG: unguarded — all 32 threads over-arrive
    while not prims.mbarrier_try_wait_parity(mbar, 0, time_limit=10_000_000):
        pass
    out[tidx] = tidx


# --- B. C3: init count says 32, but only one elected lane ever arrives ---
@cute.kernel
def kernel_b(out: cutlass.Array):
    tidx, _, _ = cute.arch.thread_idx()
    mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    if prims.elect_sync():
        prims.mbarrier_init(mbar, 32)  # BUG: expects 32 arrivals per phase
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if prims.elect_sync():
        prims.mbarrier_arrive(mbar)  # only 1 arrival -> barrier never completes
    while not prims.mbarrier_try_wait_parity(mbar, 0, time_limit=10_000_000):
        pass
    out[tidx] = tidx


# --- C. C5: expect_tx registers 128 bytes, but nothing ever completes them ---
@cute.kernel
def kernel_c(out: cutlass.Array):
    tidx, _, _ = cute.arch.thread_idx()
    mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    if prims.elect_sync():
        prims.mbarrier_init(mbar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(mbar, 128)  # BUG: no TMA / complete_tx follows
    while not prims.mbarrier_try_wait_parity(mbar, 0, time_limit=10_000_000):
        pass
    out[tidx] = tidx


# --- D. C16: elect_sync nested inside an already-elected region ---
@cute.kernel
def kernel_d(out: cutlass.Array):
    tidx, _, _ = cute.arch.thread_idx()
    val = cutlass.Int32(0)
    if prims.elect_sync():
        if prims.elect_sync():  # BUG: inner elect executed by a single lane
            val = cutlass.Int32(1)
    out[tidx] = val


# --- E. C13 attempt: elect_sync reached by only half the member mask ---
@cute.kernel
def kernel_e(out: cutlass.Array):
    tidx, _, _ = cute.arch.thread_idx()
    val = cutlass.Int32(0)
    if tidx < 16:  # BUG (undetected in this build): partial-warp elect.sync
        if prims.elect_sync():
            val = cutlass.Int32(1)
    out[tidx] = val


# --- F. corrected kernel: elected single arrive, count=1, data handoff ---
@cute.kernel
def kernel_f(out: cutlass.Array):
    tidx, _, _ = cute.arch.thread_idx()
    mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    smem = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    if prims.elect_sync():
        prims.mbarrier_init(mbar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if prims.elect_sync():          # FIX: single-issuer gate the checker wants
        smem[0] = cutlass.Int32(42)
        prims.mbarrier_arrive(mbar)  # exactly 1 arrival == init count
    while not prims.mbarrier_try_wait_parity(mbar, 0, time_limit=10_000_000):
        pass
    out[tidx] = smem[0]


# --- G. ptxas remark: runtime-indexed per-thread array -> local memory ---
@cute.kernel
def kernel_g(out: cutlass.Array):
    tidx, _, _ = cute.arch.thread_idx()
    arr = cutlass.Array(cutlass.Float32, 256)
    for i in cutlass.range_constexpr(256):
        arr[i] = cutlass.Float32(i) * cutlass.Float32(0.5)
    s = cutlass.Float32(0.0)
    idx = tidx
    for _ in cutlass.range_constexpr(128):
        idx = (idx * cutlass.Int32(5) + cutlass.Int32(1)) % cutlass.Int32(256)
        s = s + arr[idx]
    out[tidx] = s


def _make_host(kernel):
    @cute.jit
    def host(out: cutlass.Array, stream):
        kernel(out).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    return host


def main():
    assert torch.cuda.is_available(), "CUDA GPU required"
    print(__doc__)
    print(f'CUTE_DSL_COMPILER_OPT = "{os.environ["CUTE_DSL_COMPILER_OPT"]}"')

    section("A. C3/C4 — mbarrier_arrive unguarded on a count=1 barrier",
            lambda: _compile(_make_host(kernel_a)), "error, compile fails")
    section("B. C3 — mbarrier_init count 32 vs 1 elected arrive",
            lambda: _compile(_make_host(kernel_b)), "warning, compiles (would hang if run)")
    section("C. C5 — arrive_expect_tx with no completion source",
            lambda: _compile(_make_host(kernel_c)), "warning, compiles (would hang if run)")
    section("D. C16 — nested elect_sync",
            lambda: _compile(_make_host(kernel_d)), "error, compile fails")
    section("E. C13 attempt — elect_sync under `if tidx < 16` divergence",
            lambda: _compile(_make_host(kernel_e)), "C13 did NOT fire in this build")
    section("G. remarks{ptx} — runtime-indexed register array",
            lambda: _compile(_make_host(kernel_g), dtype=cutlass.Float32),
            "ptxas local-memory remark, compiles")

    # F: corrected kernel — compile clean AND run on the GPU.
    print(f"\n{'=' * 78}\nF. corrected kernel — elected arrive, matching count\n{'=' * 78}")
    with capture_fds() as buf:
        fn = _compile(_make_host(kernel_f))
        buf.seek(0)
        captured = buf.read().decode(errors="replace")
    diag = "NO nvvm diagnostics emitted" if "nvvm-diag" not in captured else "UNEXPECTED diagnostics!"
    print(captured, end="")
    print(f"--> compile SUCCEEDED, {diag}")
    out = torch.zeros(32, dtype=torch.int32, device="cuda")
    fn(out, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    ok = bool((out == 42).all())
    print(f"--> ran on GPU: out == 42 on all 32 lanes: {'PASS' if ok else 'FAIL'}")
    _report.append(f"{'=' * 78}\nF. corrected kernel — elected arrive, matching count\n{'=' * 78}\n"
                   f"{_ANSI.sub('', captured)}--> compile SUCCEEDED, {diag}\n"
                   f"--> ran on GPU: out == 42 on all 32 lanes: {'PASS' if ok else 'FAIL'}\n")
    assert ok

    with open(OUT_PATH, "w") as f:
        f.write("Captured verbatim compiler diagnostics (ANSI stripped) from hazard_demo.py\n"
                'CUTE_DSL_COMPILER_OPT="warnings{nvvm}" — GB10 / sm_121a — CUTLASS 4.7.0\n\n')
        f.write("\n".join(_report))
    print(f"\nDiagnostics saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
