"""Correctness: WS-ksplit kernel vs a plain tl.dot_scaled reference.

Both paths use the same native mxf4nvf4 block-scale MMA, so the comparison
is bit-exact (max abs diff 0), not merely allclose.

Usage:  . scripts/env.sh && python verify.py
"""
import utlx_plugin  # noqa: F401  (registers triton.language.extra.tlx)
import tlx_upstream_patch  # noqa: F401  (WS-on-upstream runtime patches)
import torch
import triton
import triton.language as tl

from nvfp4_ws_ksplit import nvfp4_ws_ksplit_gemm

DEV = "cuda"


@triton.jit
def _ref_dot_scaled(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, M, N, K,
                    sam, sak, sbn, sbk, scm, scn, sasm, sask, sbsn, sbsk,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    PK: tl.constexpr = BK // 2
    SK: tl.constexpr = BK // 16
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        ka = k * PK + tl.arange(0, PK)
        ks = k * SK + tl.arange(0, SK)
        a = tl.load(a_ptr + om[:, None] * sam + ka[None, :] * sak)
        b = tl.load(b_ptr + on[:, None] * sbn + ka[None, :] * sbk)
        sa = tl.load(as_ptr + om[:, None] * sasm + ks[None, :] * sask)
        sb = tl.load(bs_ptr + on[:, None] * sbsn + ks[None, :] * sbsk)
        acc = tl.dot_scaled(a, sa, "e2m1", b.T, sb, "e2m1", acc=acc)
    tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn,
             acc.to(tl.bfloat16))


def make_inputs(M, N, K, seed=0):
    torch.manual_seed(seed)
    e4m3 = torch.float8_e4m3fn
    a = torch.randint(0, 256, (M, K // 2), device=DEV, dtype=torch.uint8)
    b = torch.randint(0, 256, (N, K // 2), device=DEV, dtype=torch.uint8)
    sa = torch.rand(M, K // 16, device=DEV).mul(1.5).add(0.25).to(e4m3)
    sb = torch.rand(N, K // 16, device=DEV).mul(1.5).add(0.25).to(e4m3)
    return a, b, sa, sb


def run_ref(a, b, sa, sb, M, N, K, BM=128, BN=128, BK=256):
    c = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    _ref_dot_scaled[(M // BM, N // BN)](
        a, b, c, sa, sb, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), sa.stride(0), sa.stride(1),
        sb.stride(0), sb.stride(1), BM=BM, BN=BN, BK=BK, num_warps=8)
    return c


if __name__ == "__main__":
    ok = True
    for M, N, K in [(1024, 1024, 1024), (2048, 2048, 2048)]:
        a, b, sa, sb = make_inputs(M, N, K)
        c_ref = run_ref(a, b, sa, sb, M, N, K)
        c_ws = nvfp4_ws_ksplit_gemm(a, b, sa, sb)
        torch.cuda.synchronize()
        exact = torch.equal(c_ref, c_ws)
        diff = (c_ref.float() - c_ws.float()).abs().max().item()
        print(f"{M}x{N}x{K}: bit_exact={exact}  max_abs_diff={diff}")
        ok &= exact
    print("ALL CHECKS PASSED" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)
