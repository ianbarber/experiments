"""
DELIBERATELY BROKEN copy of gemm_ws_tlx.py -- demonstrates that TLX's manual
mbarrier protocol has no static safety net.

THE ONE INJECTED BUG (marked "### BUG" below, in the consumer task):
    the consumer forgets tlx.barrier_arrive on the *empty* barrier of B's
    ring buffer 0. Buffers 1..STAGES-1 are released normally.

Expected behavior:
  - Compiles with zero warnings/errors (the protocol is invisible to the compiler).
  - K <= STAGES*BK (no ring wraparound): runs and produces CORRECT results,
    because the producer never has to re-wait on empty_b[0].
  - K > STAGES*BK: the producer's wait on empty_b[0] for the second lap never
    completes -> every CTA deadlocks -> the first torch.cuda.synchronize() hangs
    forever. Run under `timeout 90`.

Run:
  cd /home/ianbarber/Projects/cute/tlx-triton && \
  timeout 90 /home/ianbarber/Projects/cute/.venv-tlx/bin/python \
    /home/ianbarber/Projects/cute/comparison/tlx/gemm_ws_tlx_broken.py
"""
import sys
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = "cuda"
BM, BN, BK = 128, 128, 64
NUM_STAGES = 2
GROUP_SIZE_M = 8
TRUNK_WARPS = 1


def alloc_fn(size: int, align: int, stream: Optional[int]):
    return torch.empty(size, dtype=torch.int8, device=DEVICE)


@triton.jit
def gemm_ws_kernel_broken(
    a_desc, b_desc, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    STAGES: tl.constexpr,
):
    BM_SPLIT: tl.constexpr = BLOCK_M // 2
    a = tlx.local_alloc((BM_SPLIT, BLOCK_K), tlx.dtype_of(a_desc), STAGES * 2)
    b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), STAGES)
    bars_empty_a = tlx.alloc_barriers(num_barriers=STAGES * 2, arrive_count=1)
    bars_full_a = tlx.alloc_barriers(num_barriers=STAGES * 2, arrive_count=1)
    bars_empty_b = tlx.alloc_barriers(num_barriers=STAGES, arrive_count=2)
    bars_full_b = tlx.alloc_barriers(num_barriers=STAGES, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BLOCK_M
            offset_bn = pid_n * BLOCK_N

            p = 1
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                buf = k % STAGES
                offset_k = k * BLOCK_K

                tlx.barrier_wait(bar=tlx.local_view(bars_empty_a, buf), phase=p)
                full_a1 = tlx.local_view(bars_full_a, buf)
                tlx.barrier_expect_bytes(full_a1, BM_SPLIT * BLOCK_K * 2)
                tlx.async_descriptor_load(a_desc, tlx.local_view(a, buf),
                                          [offset_am, offset_k], full_a1)

                tlx.barrier_wait(bar=tlx.local_view(bars_empty_b, buf), phase=p)
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_expect_bytes(full_b, BLOCK_K * BLOCK_N * 2)
                tlx.async_descriptor_load(b_desc, tlx.local_view(b, buf),
                                          [offset_k, offset_bn], full_b)

                tlx.barrier_wait(bar=tlx.local_view(bars_empty_a, buf + STAGES), phase=p)
                full_a2 = tlx.local_view(bars_full_a, buf + STAGES)
                tlx.barrier_expect_bytes(full_a2, BM_SPLIT * BLOCK_K * 2)
                tlx.async_descriptor_load(a_desc, tlx.local_view(a, buf + STAGES),
                                          [offset_am + BM_SPLIT, offset_k], full_a2)

                p = p ^ (buf == (STAGES - 1))

        with tlx.async_task(num_warps=4, replicate=2):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BLOCK_M
            offset_bn = pid_n * BLOCK_N

            rid = tlx.async_task_replica_id()
            p = 0
            acc = tl.zeros([BLOCK_M // 2, BLOCK_N], dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                buf = k % STAGES
                tlx.barrier_wait(bar=tlx.local_view(bars_full_a, buf + STAGES * rid), phase=p)
                tlx.barrier_wait(bar=tlx.local_view(bars_full_b, buf), phase=p)
                a_tile = tlx.local_load(tlx.local_view(a, buf + STAGES * rid))
                b_tile = tlx.local_load(tlx.local_view(b, buf))
                acc = tl.dot(a_tile, b_tile, acc)
                tlx.barrier_arrive(tlx.local_view(bars_empty_a, buf + STAGES * rid))
                ### BUG: buffer 0 of the B ring is never released.
                ### The correct code arrives unconditionally:
                ###     tlx.barrier_arrive(tlx.local_view(bars_empty_b, buf))
                if buf != 0:
                    tlx.barrier_arrive(tlx.local_view(bars_empty_b, buf))
                p = p ^ (buf == (STAGES - 1))

            offset_cm = offset_am + (BLOCK_M // 2) * rid
            offs_m = offset_cm + tl.arange(0, BLOCK_M // 2)
            offs_n = offset_bn + tl.arange(0, BLOCK_N)
            tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


def matmul_ws_broken(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    a_desc = TensorDescriptor(a, shape=[M, K], strides=[K, 1], block_shape=[BM // 2, BK])
    b_desc = TensorDescriptor(b, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    kernel = gemm_ws_kernel_broken[grid](
        a_desc, b_desc, c, M, N, K,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GROUP_SIZE_M,
        STAGES=NUM_STAGES, num_warps=TRUNK_WARPS,
    )
    return c, kernel


def main():
    assert torch.cuda.is_available()
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)

    # Case 1: K == STAGES*BK -> ring never wraps, bug is latent, result correct.
    M, N, K = 256, 256, NUM_STAGES * BK
    a = torch.randn((M, K), dtype=torch.float16, device=DEVICE)
    b = torch.randn((K, N), dtype=torch.float16, device=DEVICE)
    c, kernel = matmul_ws_broken(a, b)
    torch.cuda.synchronize()
    ok = torch.allclose(c.float(), torch.matmul(a, b).float(), atol=0.5, rtol=1e-2)
    print(f"[case 1] K={K} (no ring wraparound): compiled cleanly, ran, correct={ok}",
          flush=True)

    # Case 2: K > STAGES*BK -> producer re-waits on the never-released empty_b[0].
    M, N, K = 256, 256, 512
    a = torch.randn((M, K), dtype=torch.float16, device=DEVICE)
    b = torch.randn((K, N), dtype=torch.float16, device=DEVICE)
    c, kernel = matmul_ws_broken(a, b)
    print(f"[case 2] K={K}: kernel COMPILED CLEANLY and LAUNCHED "
          f"(no compiler warning or error about the missing barrier_arrive); "
          f"now calling torch.cuda.synchronize() ...", flush=True)
    torch.cuda.synchronize()
    print("[case 2] synchronize returned (this line should NEVER print)", flush=True)


if __name__ == "__main__":
    main()
