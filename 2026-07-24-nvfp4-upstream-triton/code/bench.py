"""Benchmark the WS-ksplit kernel across GEMM shapes.

Usage:  . scripts/env.sh && python bench.py [M N K]
"""
import sys

import utlx_plugin  # noqa: F401
import tlx_upstream_patch  # noqa: F401
import torch

from nvfp4_ws_ksplit import nvfp4_ws_ksplit_gemm
from verify import make_inputs

SHAPES = [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 4096)]


def time_ms(fn, reps=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


if __name__ == "__main__":
    shapes = ([tuple(int(x) for x in sys.argv[1:4])]
              if len(sys.argv) >= 4 else SHAPES)
    for M, N, K in shapes:
        a, b, sa, sb = make_inputs(M, N, K)
        ms = time_ms(lambda: nvfp4_ws_ksplit_gemm(a, b, sa, sb))
        print(f"{M}x{N}x{K}: {ms:.4f} ms -> {2*M*N*K/ms/1e9:.0f} TFLOPS")
