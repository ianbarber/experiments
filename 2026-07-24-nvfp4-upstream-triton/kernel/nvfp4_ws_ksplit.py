"""NVFP4 blockscaled GEMM — warp-specialized persistent K-split kernel,
running on UNFORKED upstream Triton via the plugin extension system.

Replicates the schedule of the Colfax CUTLASS SM12x recipe
(https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/ ,
CUTLASS example 79b: TMA + warp-specialized blockscaled mainloop + register
reallocation) in Triton TLX ops, hosted by the Triton Plugin Extensions
architecture (https://pytorch.org/blog/triton-plugin-extensions-enabling-tlx-and-custom-compiler-passes-out-of-the-box/).

Structure:
  - persistent tile scheduler (grid = NUM_SMS)
  - TMA descriptor loads into a double-buffered NVMMA-swizzled SMEM ring
  - mbarrier full/empty handshake (arrive_count: 1 producer / 2 consumers)
  - warp specialization: default task = consumer 0, worker partitions =
    consumer 1 (registers=232) + TMA producer (registers=24). The heavy
    consumer sits in the DEFAULT region because upstream's register
    allocator grants the default the leftover budget.
  - K-split: two 128-K dot_scaled per ring slot (keeps operand liveness
    low enough that 8 MMA warps do not spill)
  - B-operand loads pinned to a vectorized blocked layout
    (tlx.local_load_blocked) — avoids a scalarized lowering
  - launch maxnreg=168 (= 64K regfile / 384 threads, rounded to 8)

Requires kernel/tlx_upstream_patch.py to be imported first (see README).
Tuned for GB10 / sm_121 (BM=BN=128, BK=256, NS=2): ~330 TF peak at 4096^3,
~88%% of CUTLASS at 2048^3, bit-exact vs plain tl.dot_scaled.
"""
import torch, triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase

_DEV = "cuda"
# TMA needs a device allocator for its descriptor scratch.
triton.set_allocator(lambda size, align, stream: torch.empty(size, device=_DEV, dtype=torch.int8))


@triton.jit
def _tile_to_pmpn(tile, num_pm, num_pn, GROUP_M: tl.constexpr):
    if GROUP_M == 1:
        return tile // num_pn, tile % num_pn
    nig = GROUP_M * num_pn
    gid = tile // nig
    first_pm = gid * GROUP_M
    gsm = min(num_pm - first_pm, GROUP_M)
    return first_pm + ((tile % nig) % gsm), (tile % nig) // gsm


@triton.jit
def _nvfp4_ws_ksplit(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr,
                     M, N, K, sam, sak, sbn, sbk, scm, scn, sasm, sask, sbsn, sbsk,
                     NUM_SMS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     NS: tl.constexpr, BYTES: tl.constexpr, CONS_REGS: tl.constexpr,
                     PROD_REGS: tl.constexpr, GROUP_M: tl.constexpr):
    PK: tl.constexpr = BK // 2      # packed-fp4 K bytes per BK
    SK: tl.constexpr = BK // 16     # scale cols per BK
    HPK: tl.constexpr = PK // 2     # one 128-K chunk of fp4
    HSK: tl.constexpr = SK // 2     # one 128-K chunk of scales
    BNH: tl.constexpr = BN // 2
    cta_id = tl.program_id(0)
    num_pm = M // BM; num_pn = N // BN
    n_tiles = num_pm * num_pn
    num_iter = K // BK
    n_my = tl.cdiv(tl.maximum(n_tiles - cta_id, 0), NUM_SMS)

    bufA  = tlx.local_alloc((BM, PK),  tl.uint8, NS)
    bufB0 = tlx.local_alloc((BNH, PK), tl.uint8, NS)
    bufB1 = tlx.local_alloc((BNH, PK), tl.uint8, NS)
    bufSA = tlx.local_alloc((BM, SK),  tl.float8e4nv, NS)
    bufSB0= tlx.local_alloc((BNH, SK), tl.float8e4nv, NS)
    bufSB1= tlx.local_alloc((BNH, SK), tl.float8e4nv, NS)
    full  = tlx.alloc_barriers(num_barriers=NS, arrive_count=1)
    empty = tlx.alloc_barriers(num_barriers=NS, arrive_count=2)

    # Role-swap (2026-07-24, task 8): upstream's WS register economy gives
    # the DEFAULT region the big budget (defRegs up to 256) and squeezes
    # workers. So the heavy consumer rid=0 is the default task; the second
    # consumer and the light TMA producer are worker partitions with explicit
    # register requests (fork economy: producer 40 / consumers 232).
    with tlx.async_tasks():
        with tlx.async_task("default"):                          # consumer rid=0
            for i in range(n_my):
                tile = cta_id + i * NUM_SMS
                pm, pn = _tile_to_pmpn(tile, num_pm, num_pn, GROUP_M)
                acc = tl.zeros((BM, BNH), dtype=tl.float32)
                for k in range(0, num_iter):
                    buf, phase = get_bufidx_phase(i * num_iter + k, NS)
                    tlx.barrier_wait(tlx.local_view(full, buf), phase)
                    av  = tlx.local_view(bufA,  buf); sav = tlx.local_view(bufSA, buf)
                    bv = tlx.local_view(bufB0, buf); sbv = tlx.local_view(bufSB0, buf)
                    a0  = tlx.local_load(tlx.local_slice(av,  [0, 0], [BM, HPK]))
                    sa0 = tlx.local_load(tlx.local_slice(sav, [0, 0], [BM, HSK]))
                    b0  = tlx.local_load_blocked(tlx.local_slice(bv,  [0, 0], [BNH, HPK]), [1, 16], [8, 4], [4, 1], [1, 0])
                    sb0 = tlx.local_load(tlx.local_slice(sbv, [0, 0], [BNH, HSK]))
                    acc = tl.dot_scaled(a0, sa0, "e2m1", b0.T, sb0, "e2m1", acc=acc)
                    a1  = tlx.local_load(tlx.local_slice(av,  [0, HPK], [BM, HPK]))
                    sa1 = tlx.local_load(tlx.local_slice(sav, [0, HSK], [BM, HSK]))
                    b1  = tlx.local_load_blocked(tlx.local_slice(bv,  [0, HPK], [BNH, HPK]), [1, 16], [8, 4], [4, 1], [1, 0])
                    sb1 = tlx.local_load(tlx.local_slice(sbv, [0, HSK], [BNH, HSK]))
                    acc = tl.dot_scaled(a1, sa1, "e2m1", b1.T, sb1, "e2m1", acc=acc)
                    tlx.barrier_arrive(tlx.local_view(empty, buf))
                om = pm * BM + tl.arange(0, BM)
                on = pn * BN + tl.arange(0, BNH)
                tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc.to(tl.bfloat16))

        with tlx.async_task(num_warps=4, registers=CONS_REGS):   # consumer rid=1
            for i in range(n_my):
                tile = cta_id + i * NUM_SMS
                pm, pn = _tile_to_pmpn(tile, num_pm, num_pn, GROUP_M)
                acc = tl.zeros((BM, BNH), dtype=tl.float32)
                for k in range(0, num_iter):
                    buf, phase = get_bufidx_phase(i * num_iter + k, NS)
                    tlx.barrier_wait(tlx.local_view(full, buf), phase)
                    av  = tlx.local_view(bufA,  buf); sav = tlx.local_view(bufSA, buf)
                    bv = tlx.local_view(bufB1, buf); sbv = tlx.local_view(bufSB1, buf)
                    a0  = tlx.local_load(tlx.local_slice(av,  [0, 0], [BM, HPK]))
                    sa0 = tlx.local_load(tlx.local_slice(sav, [0, 0], [BM, HSK]))
                    b0  = tlx.local_load_blocked(tlx.local_slice(bv,  [0, 0], [BNH, HPK]), [1, 16], [8, 4], [4, 1], [1, 0])
                    sb0 = tlx.local_load(tlx.local_slice(sbv, [0, 0], [BNH, HSK]))
                    acc = tl.dot_scaled(a0, sa0, "e2m1", b0.T, sb0, "e2m1", acc=acc)
                    a1  = tlx.local_load(tlx.local_slice(av,  [0, HPK], [BM, HPK]))
                    sa1 = tlx.local_load(tlx.local_slice(sav, [0, HSK], [BM, HSK]))
                    b1  = tlx.local_load_blocked(tlx.local_slice(bv,  [0, HPK], [BNH, HPK]), [1, 16], [8, 4], [4, 1], [1, 0])
                    sb1 = tlx.local_load(tlx.local_slice(sbv, [0, HSK], [BNH, HSK]))
                    acc = tl.dot_scaled(a1, sa1, "e2m1", b1.T, sb1, "e2m1", acc=acc)
                    tlx.barrier_arrive(tlx.local_view(empty, buf))
                om = pm * BM + tl.arange(0, BM)
                on = pn * BN + BNH + tl.arange(0, BNH)
                tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc.to(tl.bfloat16))

        with tlx.async_task(num_warps=4, registers=PROD_REGS):   # TMA producer
            da  = tl.make_tensor_descriptor(a_ptr,  [M, K // 2],  [sam, sak],  [BM, PK])
            db  = tl.make_tensor_descriptor(b_ptr,  [N, K // 2],  [sbn, sbk],  [BNH, PK])
            dsa = tl.make_tensor_descriptor(as_ptr, [M, K // 16], [sasm, sask],[BM, SK])
            dsb = tl.make_tensor_descriptor(bs_ptr, [N, K // 16], [sbsn, sbsk],[BNH, SK])
            for i in range(n_my):
                tile = cta_id + i * NUM_SMS
                pm, pn = _tile_to_pmpn(tile, num_pm, num_pn, GROUP_M)
                for k in range(0, num_iter):
                    buf, ph = get_bufidx_phase(i * num_iter + k, NS)
                    tlx.barrier_wait(tlx.local_view(empty, buf), ph ^ 1)
                    fb = tlx.local_view(full, buf)
                    tlx.barrier_expect_bytes(fb, BYTES)
                    tlx.async_descriptor_load(da,  tlx.local_view(bufA,  buf), [pm * BM, k * PK], fb)
                    tlx.async_descriptor_load(db,  tlx.local_view(bufB0, buf), [pn * BN, k * PK], fb)
                    tlx.async_descriptor_load(db,  tlx.local_view(bufB1, buf), [pn * BN + BNH, k * PK], fb)
                    tlx.async_descriptor_load(dsa, tlx.local_view(bufSA, buf), [pm * BM, k * SK], fb)
                    tlx.async_descriptor_load(dsb, tlx.local_view(bufSB0,buf), [pn * BN, k * SK], fb)
                    tlx.async_descriptor_load(dsb, tlx.local_view(bufSB1,buf), [pn * BN + BNH, k * SK], fb)


_MAXNREG = 168  # entry budget = 64K regfile / 384 threads, rounded to 8
# Kernel-owned register economy: the utlx_set_ws_requested_regs plugin pass
# re-stamps these after OptimizePartitionWarps would clobber them, yielding
# actualRegisters = [248 default consumer, 232 consumer1, 24 producer].
import os as _os
_os.environ.setdefault("UTLX_WS_REGS", "232,24")


def nvfp4_ws_ksplit_gemm(a, b, sa, sb, out=None, BM=128, BN=128, BK=256, NS=2):
    """C[M,N] = (A_fp4 * SA) @ (B_fp4 * SB).T  ; A,B are uint8 packed-e2m1 [.,K/2], SA,SB fp8-e4m3 [.,K/16].
    Returns bf16 [M,N]. Tuned defaults are the sm121 optimum; do not raise NS (SMEM OOM)."""
    M = a.shape[0]; N = b.shape[0]; K = a.shape[1] * 2
    assert BK == 256 and BM == BN == 128, "tuned recipe is BM=BN=128, BK=256"
    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    PK, SK, BNH = BK // 2, BK // 16, BN // 2
    BYTES = BM * PK + BNH * PK * 2 + BM * SK + BNH * SK * 2
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    st = (a.stride(0), a.stride(1), b.stride(0), b.stride(1),
          out.stride(0), out.stride(1), sa.stride(0), sa.stride(1), sb.stride(0), sb.stride(1))
    _nvfp4_ws_ksplit[(NUM_SMS,)](a, b, out, sa, sb, M, N, K, *st, NUM_SMS,
                                 BM=BM, BN=BN, BK=BK, NS=NS, BYTES=BYTES,
                                 CONS_REGS=232, PROD_REGS=24, GROUP_M=1, num_warps=4, num_stages=1,
                                 **({'maxnreg': _MAXNREG} if _MAXNREG else {}))
    return out
