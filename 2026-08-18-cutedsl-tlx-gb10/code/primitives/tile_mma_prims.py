#!/usr/bin/env python
"""
tile_mma_prims.py — a single-tile fp16 GEMM (64x64x64, fp32 accumulate) on sm_121
written in pure-SIMT CUTLASS 4.7 Primitives style: no CuTe layouts, no TiledMMA,
no atoms. The kernel drives the hardware directly, CUDA-C++-with-intrinsics style:

  TMA  (cp.async.bulk.tensor + mbarrier expect_tx/try_wait_parity)   [issue: warp 0, 1 elected lane]
    -> prims.ldmatrix   (ldmatrix.sync.aligned.m8n8.x4)              [all 4 warps]
    -> prims.mma_sync   (mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32)
    -> plain per-lane fp32 stores of the accumulator fragment.

Everything TiledMMA normally derives — fragment ownership per lane, issuer
election, tile addressing, synchronization — is written out by hand here.
Verified against torch.matmul (fp64 reference).

Run:  /home/ianbarber/Projects/cute/.venv-cutedsl/bin/python tile_mma_prims.py
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cuda.bindings.driver as cuda_driver
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

# One CTA computes one 64x64x64 tile with 4 warps; each warp owns a 16-row slab.
TILE = 64          # M = N = K
WARPS = 4          # warp w computes rows [16w, 16w+16)
MMA_M, MMA_N, MMA_K = 16, 8, 16   # the mma.sync atom


# --------------------------------------------------------------------------
# mma.sync fragment plumbing.  prims.mma_sync is a 1:1 NVVM op: operands are
# raw MLIR values, so the i32 register words ldmatrix returns are bitcast to
# the vector<2xf16> carrier type the dialect infers ".f16" from, and the fp32
# accumulator struct is unpacked back into cutlass.Float32 values.
# --------------------------------------------------------------------------
def _f16x2(reg32):
    """cutlass.Int32 register word -> ir.Value of type vector<2xf16>."""
    return llvm.bitcast(ir.VectorType.get([2], ir.F16Type.get()), reg32.ir_value())


def _mma_m16n8k16_f16f32(a_regs, b_regs, c_vals):
    """One mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32; returns 4 Float32."""
    f32 = ir.F32Type.get()
    d = prims.mma_sync(
        llvm.StructType.get_literal([f32] * 4),   # result fragment type (4 x f32)
        (MMA_M, MMA_N, MMA_K),
        prims.MMALayout.ROW,                      # A row-major
        prims.MMALayout.COL,                      # B col-major (we stage B^T)
        [_f16x2(r) for r in a_regs],              # 4 x f16x2
        [_f16x2(r) for r in b_regs],              # 2 x f16x2
        [c.ir_value() for c in c_vals],           # 4 x f32
        # NOTE: pass the multiplicand PTX types EXPLICITLY.  Omitting them
        # (leaving the dialect to infer ".f16" from the vector<2xf16> carriers)
        # hard-aborts this build's MLIR library during IR construction.
        multiplicand_a_ptx_type=prims.MMAType.F16,
        multiplicand_b_ptx_type=prims.MMAType.F16,
    )
    return [cutlass.Float32(llvm.extractvalue(f32, d, [i])) for i in range(4)]


@cute.kernel
def gemm_tile_kernel(
    tma_a: cutlass.GridConstant[cuda.TensorMap],   # A   (64x64, M-major rows of K)
    tma_bt: cutlass.GridConstant[cuda.TensorMap],  # B^T (64x64, N-major rows of K)
    c: cutlass.Array,                              # C   (64x64, fp32 out)
):
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.warp_idx()
    lane = tidx % 32

    smem_a = cutlass.Array(cutlass.Float16, (TILE, TILE), space=cutlass.AddressSpace.smem, alignment=128)
    smem_bt = cutlass.Array(cutlass.Float16, (TILE, TILE), space=cutlass.AddressSpace.smem, alignment=128)
    mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)

    # ---- TMA load of both operand tiles, single mbarrier, single issuer ----
    # (warp-gate + elect_sync: the nvvm hazard checker errors on unguarded
    #  arrives — see hazard_demo.py section A)
    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_init(mbar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(
                mbar, tma_a.global_tx_bytes() + tma_bt.global_tx_bytes()
            )
            prims.cp_async_bulk_tensor_shared_cta_global(smem_a, tma_a.get_ptr(), (0, 0), mbar)
            prims.cp_async_bulk_tensor_shared_cta_global(smem_bt, tma_bt.get_ptr(), (0, 0), mbar)

    while not prims.mbarrier_try_wait_parity(mbar, 0, time_limit=10_000_000):
        pass
    prims.barrier_cta_sync(0)

    # ---- ldmatrix -> mma.sync main loop -----------------------------------
    # ldmatrix.x4 lane addressing: lane addr = &tile[lane%16][(lane//16)*8]
    # covers a 16x16 fp16 block as 4 8x8 tiles in exactly the fragment order
    # mma.sync wants (a0=r0-7/c0-7, a1=r8-15/c0-7, a2=r0-7/c8-15, a3=r8-15/c8-15).
    m0 = warp * 16                                  # this warp's row slab
    ld_row = lane % 16
    ld_col = (lane // 16) * 8
    a_ptr0 = smem_a.data_ptr() + (m0 + ld_row) * TILE + ld_col
    bt_ptr0 = smem_bt.data_ptr() + ld_row * TILE + ld_col

    # fp32 accumulators: 8 n-atoms x 4 values per lane (the m16n8 C fragment)
    acc = [[cutlass.Float32(0.0) for _ in range(4)] for _ in range(TILE // MMA_N)]

    for ki in cutlass.range_constexpr(TILE // MMA_K):                 # K loop: 4 steps of 16
        k0 = ki * MMA_K
        a_regs = prims.ldmatrix(a_ptr0 + k0, 4, prims.MMALayout.ROW)       # A[m0:m0+16, k0:k0+16]
        for npair in cutlass.range_constexpr(TILE // (2 * MMA_N)):    # N loop: 4 pairs of 8-col atoms
            n0 = npair * 16
            # B^T[n0:n0+16, k0:k0+16]: tiles 0/2 are the k-halves of n-atom 2*npair,
            # tiles 1/3 of n-atom 2*npair+1 (b fragment = 2 x f16x2 per atom).
            b_regs = prims.ldmatrix(bt_ptr0 + n0 * TILE + k0, 4, prims.MMALayout.ROW)
            for i in cutlass.range_constexpr(2):
                na = 2 * npair + i
                acc[na] = _mma_m16n8k16_f16f32(
                    [a_regs[0], a_regs[1], a_regs[2], a_regs[3]],
                    [b_regs[i], b_regs[i + 2]],
                    acc[na],
                )

    # ---- epilogue: each lane stores its 4 fp32 accumulator values ----------
    # m16n8 C fragment: lane holds C[g][2t], C[g][2t+1], C[g+8][2t], C[g+8][2t+1]
    # with g = lane//4, t = lane%4.
    g = lane // 4
    t = lane % 4
    for na in cutlass.range_constexpr(TILE // MMA_N):
        col = na * MMA_N + 2 * t
        c[m0 + g, col] = acc[na][0]
        c[m0 + g, col + 1] = acc[na][1]
        c[m0 + g + 8, col] = acc[na][2]
        c[m0 + g + 8, col + 1] = acc[na][3]


@cute.jit
def host(a: cute.Tensor, bt: cute.Tensor, c: cutlass.Array, stream):
    stride_16b = TILE * 2 // 16  # row stride in 16-byte units (fp16, 64 cols)
    tma_a = cuda.create_tensor_map_tiled(
        global_address=a.iterator.toint(), dtype=cutlass.Float16,
        global_dims=[TILE, TILE], global_strides=[stride_16b],
        box_dims=[TILE, TILE], swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_bt = cuda.create_tensor_map_tiled(
        global_address=bt.iterator.toint(), dtype=cutlass.Float16,
        global_dims=[TILE, TILE], global_strides=[stride_16b],
        box_dims=[TILE, TILE], swizzle=cuda.TensorMapSwizzle.none,
    )
    gemm_tile_kernel(tma_a, tma_bt, c).launch(
        grid=(1, 1, 1), block=(32 * WARPS, 1, 1), stream=stream
    )


def main():
    assert torch.cuda.is_available(), "CUDA GPU required"
    print(__doc__)

    fake16 = lambda: make_fake_compact_tensor(
        cutlass.Float16, (TILE, TILE), stride_order=(1, 0), assumed_align=16
    )
    fn = cute.compile(
        host,
        fake16(), fake16(),
        make_fake_compact_tensor(cutlass.Float32, (TILE, TILE), stride_order=(1, 0), assumed_align=16),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    print("Compile OK (TMA -> ldmatrix -> mma.sync -> stores, 4 warps, 1 CTA)")

    torch.manual_seed(0)
    a = torch.randn(TILE, TILE, dtype=torch.float16, device="cuda")
    bt = torch.randn(TILE, TILE, dtype=torch.float16, device="cuda")  # B^T (N x K)
    c = torch.zeros(TILE, TILE, dtype=torch.float32, device="cuda")
    fn(a, bt, c, cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()

    ref = (a.double() @ bt.double().t()).float()   # fp64 reference of A @ B
    max_err = (c - ref).abs().max().item()
    print(f"C = A(64x64,fp16) @ B(64x64,fp16), fp32 accumulate; max |err| vs fp64 torch.matmul = {max_err:.3e}")
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-3)
    print("verify vs torch.matmul: PASS")


if __name__ == "__main__":
    main()
