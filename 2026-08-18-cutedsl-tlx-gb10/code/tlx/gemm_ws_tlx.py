"""
Warp-specialized pipelined fp16 GEMM (fp32 accumulate) for NVIDIA GB10 (sm_121a)
using TLX (Triton Language eXtensions) async tasks.

Structure (per CTA):
  - Trunk / "default" task (1 warp): TMA producer. Uses tlx.async_descriptor_load
    (cp.async.bulk.tensor) to fill a NUM_STAGES-deep shared-memory ring for B and a
    2*NUM_STAGES-deep ring for A (A is split in M so two consumer replicas each get
    their own half-tile). mbarrier protocol: wait(empty, phase), expect_bytes(full),
    TMA-load-with-arrive(full). Producer phase starts at 1, flips via XOR each time
    the ring wraps.
  - Consumer task (num_warps=4, replicate=2): each replica computes a
    (BM/2 x BN) half of the output tile. Protocol: wait(full, phase),
    tlx.local_load the smem tiles, acc = tl.dot(a, b, acc) [fp32 accumulate on
    mma.sync tensor cores -- NOT tlx.async_dot, which emits tcgen05/wgmma and does
    not work on sm_121], then barrier_arrive(empty). Consumer phase starts at 0.
    B's empty barriers have arrive_count=2 because both replicas read each B tile.
  - Epilogue: plain tl.store of acc.to(fp16).

Config chosen after a sweep (see NOTES.md): BM=128, BN=128, BK=64, NUM_STAGES=2,
trunk num_warps=1, consumer 2x4 warps. SMEM: 64 KiB ring + barriers/epilogue
= 74,344 bytes of the 101,376-byte budget.

Run (venv + cwd):
  cd /home/ianbarber/Projects/cute/tlx-triton && \
  /home/ianbarber/Projects/cute/.venv-tlx/bin/python \
    /home/ianbarber/Projects/cute/comparison/tlx/gemm_ws_tlx.py [--bench]

Without --bench: correctness check only. With --bench: full protocol benchmark
(fp16 in, fp32 acc; M=N=K in {1024,2048,4096}; 25 warmup + 100 timed iters,
median ms via torch.cuda.Event) vs torch.matmul and a plain (non-warp-specialized)
Triton tl.dot GEMM; writes bench_results.json next to this file.
"""
import argparse
import json
import os
import statistics
import sys
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = "cuda"

# Chosen config
BM, BN, BK = 128, 128, 64
NUM_STAGES = 2
GROUP_SIZE_M = 8
TRUNK_WARPS = 1  # TMA issue needs only one elected thread


def alloc_fn(size: int, align: int, stream: Optional[int]):
    return torch.empty(size, dtype=torch.int8, device=DEVICE)


# --------------------------------------------------------------------------
# Warp-specialized TLX kernel
# --------------------------------------------------------------------------
@triton.jit
def gemm_ws_kernel(
    a_desc, b_desc, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    STAGES: tl.constexpr,
):
    BM_SPLIT: tl.constexpr = BLOCK_M // 2

    # SMEM rings: A has 2*STAGES half-tiles (one half per consumer replica),
    # B has STAGES full tiles shared by both replicas.
    a = tlx.local_alloc((BM_SPLIT, BLOCK_K), tlx.dtype_of(a_desc), STAGES * 2)
    b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), STAGES)

    bars_empty_a = tlx.alloc_barriers(num_barriers=STAGES * 2, arrive_count=1)
    bars_full_a = tlx.alloc_barriers(num_barriers=STAGES * 2, arrive_count=1)
    bars_empty_b = tlx.alloc_barriers(num_barriers=STAGES, arrive_count=2)
    bars_full_b = tlx.alloc_barriers(num_barriers=STAGES, arrive_count=1)

    with tlx.async_tasks():
        # ---------------- producer: TMA loads into the smem rings ----------
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

            p = 1  # producer phase starts at 1: 1,1,0,0,1,1,... for STAGES=2
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                buf = k % STAGES
                offset_k = k * BLOCK_K

                # First half of A -> a[buf]
                tlx.barrier_wait(bar=tlx.local_view(bars_empty_a, buf), phase=p)
                full_a1 = tlx.local_view(bars_full_a, buf)
                tlx.barrier_expect_bytes(full_a1, BM_SPLIT * BLOCK_K * 2)
                tlx.async_descriptor_load(a_desc, tlx.local_view(a, buf),
                                          [offset_am, offset_k], full_a1)

                # B tile -> b[buf]
                tlx.barrier_wait(bar=tlx.local_view(bars_empty_b, buf), phase=p)
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_expect_bytes(full_b, BLOCK_K * BLOCK_N * 2)
                tlx.async_descriptor_load(b_desc, tlx.local_view(b, buf),
                                          [offset_k, offset_bn], full_b)

                # Second half of A -> a[buf + STAGES]
                tlx.barrier_wait(bar=tlx.local_view(bars_empty_a, buf + STAGES), phase=p)
                full_a2 = tlx.local_view(bars_full_a, buf + STAGES)
                tlx.barrier_expect_bytes(full_a2, BM_SPLIT * BLOCK_K * 2)
                tlx.async_descriptor_load(a_desc, tlx.local_view(a, buf + STAGES),
                                          [offset_am + BM_SPLIT, offset_k], full_a2)

                p = p ^ (buf == (STAGES - 1))  # flip when the ring wraps

        # ---------------- consumers: tl.dot from smem (2 replicas x 4 warps)
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
            p = 0  # consumer phase starts at 0: 0,0,1,1,0,0,... for STAGES=2
            acc = tl.zeros([BLOCK_M // 2, BLOCK_N], dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                buf = k % STAGES

                tlx.barrier_wait(bar=tlx.local_view(bars_full_a, buf + STAGES * rid), phase=p)
                tlx.barrier_wait(bar=tlx.local_view(bars_full_b, buf), phase=p)

                a_tile = tlx.local_load(tlx.local_view(a, buf + STAGES * rid))
                b_tile = tlx.local_load(tlx.local_view(b, buf))
                # mma.sync path -- tlx.async_dot would emit tcgen05 on cc12.1
                # which ptxas rejects; plain tl.dot on local_load'ed operands
                # is the verified compute path on sm_121.
                acc = tl.dot(a_tile, b_tile, acc)

                tlx.barrier_arrive(tlx.local_view(bars_empty_a, buf + STAGES * rid))
                tlx.barrier_arrive(tlx.local_view(bars_empty_b, buf))

                p = p ^ (buf == (STAGES - 1))

            offset_cm = offset_am + (BLOCK_M // 2) * rid
            offs_m = offset_cm + tl.arange(0, BLOCK_M // 2)
            offs_n = offset_bn + tl.arange(0, BLOCK_N)
            tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


def matmul_ws(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None):
    """C = A @ B, fp16 in, fp32 accumulate, fp16 out; warp-specialized TLX kernel."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.is_contiguous() and b.is_contiguous()
    c = out if out is not None else torch.empty((M, N), dtype=torch.float16, device=a.device)
    a_desc = TensorDescriptor(a, shape=[M, K], strides=[K, 1], block_shape=[BM // 2, BK])
    b_desc = TensorDescriptor(b, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    kernel = gemm_ws_kernel[grid](
        a_desc, b_desc, c, M, N, K,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GROUP_SIZE_M,
        STAGES=NUM_STAGES, num_warps=TRUNK_WARPS,
    )
    return c, kernel


# --------------------------------------------------------------------------
# Plain (non-warp-specialized) Triton tl.dot GEMM baseline
# (adapted from tlx-triton/python/tutorials/03-matrix-multiplication.py)
# --------------------------------------------------------------------------
@triton.jit
def gemm_plain_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_tile = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b_tile = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a_tile, b_tile, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def matmul_plain(a, b, num_warps=8, num_stages=3):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    gemm_plain_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GROUP_SIZE_M,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


# --------------------------------------------------------------------------
# Verify + benchmark
# --------------------------------------------------------------------------
def verify():
    torch.manual_seed(0)
    for size in (512, 1024, 2048, 4096):
        M = N = K = size
        a = torch.randn((M, K), dtype=torch.float16, device=DEVICE)
        b = torch.randn((K, N), dtype=torch.float16, device=DEVICE)
        c, kernel = matmul_ws(a, b)
        ref = torch.matmul(a, b)
        torch.testing.assert_close(c.float(), ref.float(), atol=1e-2 * (K ** 0.5), rtol=1e-2)
        print(f"verify M=N=K={size}: OK (matches torch.matmul fp16)")
    c_plain = matmul_plain(a, b)
    torch.testing.assert_close(c_plain.float(), ref.float(), atol=1e-2 * (K ** 0.5), rtol=1e-2)
    print("verify plain-triton baseline @4096: OK")
    print(f"kernel smem={kernel.metadata.shared} bytes, regs/thread={kernel.n_regs}, "
          f"spills={kernel.n_spills}")
    ttgir = kernel.asm["ttgir"]
    assert "ttg.warp_specialize" in ttgir
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 3
    print("TTGIR: warp_specialize + 3 TMA loads per k-iteration confirmed")


def bench_median_ms(fn, warmup=25, iters=100):
    """Protocol timing: >=25 warmup, >=100 timed iterations, per-iter cuda events, median."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()
    return statistics.median(starts[i].elapsed_time(stops[i]) for i in range(iters))


def bench(out_json):
    torch.manual_seed(0)
    results = {}
    for size in (1024, 2048, 4096):
        M = N = K = size
        flops = 2.0 * M * N * K
        a = torch.randn((M, K), dtype=torch.float16, device=DEVICE)
        b = torch.randn((K, N), dtype=torch.float16, device=DEVICE)
        c_out = torch.empty((M, N), dtype=torch.float16, device=DEVICE)

        ws_ms = bench_median_ms(lambda: matmul_ws(a, b, out=c_out))
        torch_ms = bench_median_ms(lambda: torch.matmul(a, b))
        plain_ms = bench_median_ms(lambda: matmul_plain(a, b))

        tf = lambda ms: flops / (ms * 1e-3) / 1e12
        results[f"{M}x{N}x{K}"] = {
            "yours_ms": round(ws_ms, 4), "yours_tflops": round(tf(ws_ms), 2),
            "torch_ms": round(torch_ms, 4), "torch_tflops": round(tf(torch_ms), 2),
            "triton_plain_ms": round(plain_ms, 4), "triton_plain_tflops": round(tf(plain_ms), 2),
        }
        r = results[f"{M}x{N}x{K}"]
        print(f"M=N=K={size}: tlx-ws {r['yours_ms']} ms ({r['yours_tflops']} TF) | "
              f"torch {r['torch_ms']} ms ({r['torch_tflops']} TF) | "
              f"plain-triton {r['triton_plain_ms']} ms ({r['triton_plain_tflops']} TF)")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="run the full benchmark protocol")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    triton.set_allocator(alloc_fn)
    verify()
    if args.bench:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.json")
        bench(out)


if __name__ == "__main__":
    main()
