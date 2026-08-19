#!/usr/bin/env python
"""
three_ways.py — ONE device operation (warp butterfly sum via shfl.sync.bfly.b32)
implemented at all three rungs of the CUTLASS 4.7 Primitives escape-hatch ladder:

  1. prims.shfl_sync(...)          wrapped op   : typed args, StrEnum kinds,
                                                  trace-time validation, result
                                                  type inferred, compiler-visible
  2. prims.dialect.shfl_sync(...)  raw NVVM op  : any NVVM dialect op, but YOU
                                                  supply the MLIR result type and
                                                  dialect enums; only .ir_value()
                                                  objects auto-convert
  3. prims.inline_ptx(...)         asm string   : the old way — opaque to the
                                                  compiler, register classes
                                                  picked from write_only_types

All three lower to the same PTX instruction; results are verified identical on GPU.

Run:  /home/ianbarber/Projects/cute/.venv-cutedsl/bin/python three_ways.py
"""

import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

_FULL_MASK = 0xFFFFFFFF  # all 32 lanes participate
_MAC_BFLY = 0x1F         # mask_and_clamp: full warp, clamp = 31


@cute.kernel
def three_ways_kernel(vals: cutlass.Array, out: cutlass.Array):
    """out[0..2] = warp sum of vals[0..31], one slot per implementation tier."""
    tidx, _, _ = cute.arch.thread_idx()

    # ---- Tier 1: wrapped op -------------------------------------------------
    # Typed signature; kind is a StrEnum (or bare string "bfly"); Python ints
    # coerce to Int32; result type (Int32, matching `val`) is inferred.
    acc_wrapped = vals[tidx]
    for delta in [16, 8, 4, 2, 1]:
        acc_wrapped = acc_wrapped + prims.shfl_sync(
            _FULL_MASK, acc_wrapped, delta, _MAC_BFLY, prims.Shfl.BFLY
        )

    # ---- Tier 2: raw NVVM dialect op via the auto-converting proxy ---------
    # Same MLIR op the wrapper emits, minus its conveniences: the result type
    # is now the FIRST positional arg, the kind is the raw dialect enum, and
    # Python literals must be wrapped (only .ir_value() objects auto-convert).
    acc_dialect = vals[tidx]
    for delta in [16, 8, 4, 2, 1]:
        raw = prims.dialect.shfl_sync(
            cutlass.Int32.mlir_type,          # explicit MLIR result type
            cutlass.Int32(_FULL_MASK),        # membermask
            acc_dialect,                      # val
            cutlass.Int32(delta),             # offset (XOR lane mask)
            cutlass.Int32(_MAC_BFLY),         # mask_and_clamp
            prims.dialect.ShflKind.bfly,      # raw dialect enum, not StrEnum
        )
        acc_dialect = acc_dialect + cutlass.Int32(raw)

    # ---- Tier 3: inline PTX -------------------------------------------------
    # The pre-4.7 way. The instruction is an opaque string: no trace-time
    # checks, invisible to the NVVM hazard checker and to compiler passes.
    acc_ptx = vals[tidx]
    for delta in [16, 8, 4, 2, 1]:
        other = prims.inline_ptx(
            "shfl.sync.bfly.b32 {$w0}, {$r0}, {$r1}, {$r2}, {$r3};",
            write_only_types=[cutlass.Int32],
            read_only_args=[
                acc_ptx,                      # a: source value
                cutlass.Int32(delta),         # b: XOR lane mask
                cutlass.Int32(_MAC_BFLY),     # c: mask_and_clamp
                cutlass.Int32(-1),            # membermask (0xFFFFFFFF)
            ],
        )
        acc_ptx = acc_ptx + other

    if tidx == 0:
        out[0] = acc_wrapped
        out[1] = acc_dialect
        out[2] = acc_ptx


@cute.jit
def host(vals: cutlass.Array, out: cutlass.Array, stream):
    three_ways_kernel(vals, out).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def main():
    assert torch.cuda.is_available(), "CUDA GPU required"
    print(__doc__)

    fn = cute.compile(
        host,
        make_fake_compact_tensor(cutlass.Int32, (32,), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (3,), assumed_align=4),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    print("Compile OK (one kernel, three implementations of shfl.sync.bfly.b32)\n")

    vals = torch.arange(32, dtype=torch.int32, device="cuda")
    out = torch.zeros(3, dtype=torch.int32, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(vals, out, stream)
    torch.cuda.synchronize()

    expected = int(vals.sum().item())  # 0+1+...+31 = 496
    r = out.cpu().tolist()
    print(f"input                     : lane index 0..31  (expected warp sum = {expected})")
    print(f"1. prims.shfl_sync        : {r[0]}")
    print(f"2. prims.dialect.shfl_sync: {r[1]}")
    print(f"3. prims.inline_ptx       : {r[2]}")
    assert r == [expected] * 3, f"MISMATCH: {r} != {[expected] * 3}"
    print("\nAll three tiers produce identical results: PASS")


if __name__ == "__main__":
    main()
