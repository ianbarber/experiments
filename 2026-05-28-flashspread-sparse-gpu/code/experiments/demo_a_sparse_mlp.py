#!/usr/bin/env python
"""
Demo A: Block-Scalar Skip for Sparse-Activation MLP

Benchmarks three implementations across sparsity rates:
1. pytorch_baseline  – dense matmul then mask
2. triton_naive      – Triton with per-lane store mask (still loads weights)
3. triton_block_skip – block-scalar branch; skips weight loads for all-zero tiles

Each is measured both eager and inside a CUDA Graph replay loop.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernels.sparse_mlp import pytorch_baseline, triton_naive, triton_block_skip
from src.utils.bench import timed_region


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def benchmark_eager(fn, x, w, gate, repeats: int = 50):
    """Warmup then time `repeats` calls."""
    for _ in range(5):
        fn(x, w, gate)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(x, w, gate)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e3  # ms


def benchmark_cudagraph(fn, x, w, gate, repeats: int = 50, steps_per_launch: int = 50):
    """Capture a graph that calls `fn` steps_per_launch times, then replay."""
    # Warmup for Triton autotune / JIT
    for _ in range(5):
        fn(x, w, gate)
    torch.cuda.synchronize()

    def step():
        fn(x, w, gate)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(steps_per_launch):
            step()

    # Warmup graph
    g.replay()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats // steps_per_launch):
        g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_steps = (repeats // steps_per_launch) * steps_per_launch
    return elapsed / total_steps * 1e3  # ms per step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda"
    M, K, N = 65536, 512, 512  # Large tile to stress memory bandwidth
    sparsities = [0.0, 0.5, 0.75, 0.90, 0.95, 0.99]
    repeats = 100
    steps_per_launch = 50

    print(f"Demo A: Sparse-Activation MLP  (M={M}, K={K}, N={N})")
    print(f"Sparsities: {sparsities}")
    print(f"Repeats: {repeats}, steps_per_launch (CG): {steps_per_launch}")
    print()

    # Pre-generate weights (shared across all runs)
    w = torch.randn(K, N, device=device, dtype=torch.float32)

    results = {
        "sparsity": [],
        "pytorch_eager": [],
        "pytorch_cg": [],
        "naive_eager": [],
        "naive_cg": [],
        "skip_eager": [],
        "skip_cg": [],
    }

    for sparsity in sparsities:
        # Generate input and gate
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        # Block-structured sparsity: zero out entire contiguous blocks of rows.
        # This matches real-world clustered sparsity (frontiers, epidemic states, etc.)
        gate = torch.ones(M, dtype=torch.int32, device=device)
        block_size = 64
        num_zero_blocks = int((M // block_size) * sparsity)
        if num_zero_blocks > 0:
            zero_indices = torch.randperm(M // block_size, device=device)[:num_zero_blocks]
            for zb in zero_indices:
                gate[zb * block_size : (zb + 1) * block_size] = 0

        # Validate correctness first
        with torch.no_grad():
            ref = pytorch_baseline(x, w, gate)
            out_naive = triton_naive(x, w, gate)
            out_skip = triton_block_skip(x, w, gate)

        ok_naive = torch.allclose(ref, out_naive, atol=2e-1)
        ok_skip = torch.allclose(ref, out_skip, atol=2e-1)
        if not (ok_naive and ok_skip):
            print(f"  [sparsity={sparsity}] VALIDATION FAILED naive={ok_naive} skip={ok_skip}")
            continue

        # Benchmark
        t_pytorch_eager = benchmark_eager(pytorch_baseline, x, w, gate, repeats)
        t_pytorch_cg = benchmark_cudagraph(pytorch_baseline, x, w, gate, repeats, steps_per_launch)
        t_naive_eager = benchmark_eager(triton_naive, x, w, gate, repeats)
        t_naive_cg = benchmark_cudagraph(triton_naive, x, w, gate, repeats, steps_per_launch)
        t_skip_eager = benchmark_eager(triton_block_skip, x, w, gate, repeats)
        t_skip_cg = benchmark_cudagraph(triton_block_skip, x, w, gate, repeats, steps_per_launch)

        results["sparsity"].append(sparsity)
        results["pytorch_eager"].append(t_pytorch_eager)
        results["pytorch_cg"].append(t_pytorch_cg)
        results["naive_eager"].append(t_naive_eager)
        results["naive_cg"].append(t_naive_cg)
        results["skip_eager"].append(t_skip_eager)
        results["skip_cg"].append(t_skip_cg)

        print(
            f"  sparsity={sparsity:.2f}  "
            f"pytorch(eager/cg)={t_pytorch_eager:.3f}/{t_pytorch_cg:.3f}ms  "
            f"naive(eager/cg)={t_naive_eager:.3f}/{t_naive_cg:.3f}ms  "
            f"skip(eager/cg)={t_skip_eager:.3f}/{t_skip_cg:.3f}ms"
        )

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    s = np.array(results["sparsity"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Eager vs CUDA Graph for each method
    ax = axes[0]
    width = 0.12
    x = np.arange(len(s))
    ax.bar(x - 1.5 * width, results["pytorch_eager"], width, label="PyTorch eager", color="#555")
    ax.bar(x - 0.5 * width, results["pytorch_cg"], width, label="PyTorch CG", color="#888")
    ax.bar(x + 0.5 * width, results["naive_eager"], width, label="Triton naive eager", color="#3498db")
    ax.bar(x + 1.5 * width, results["naive_cg"], width, label="Triton naive CG", color="#5dade2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0%}" for v in s])
    ax.set_xlabel("Sparsity (fraction of zero rows)")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("(a) Eager vs CUDA Graph — all methods")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, ls=":")

    # Panel 2: Speedup of block-skip vs naive, inside CG
    ax = axes[1]
    speedup_vs_naive_cg = np.array(results["naive_cg"]) / np.array(results["skip_cg"])
    speedup_vs_pytorch_cg = np.array(results["pytorch_cg"]) / np.array(results["skip_cg"])
    ax.plot(s, speedup_vs_naive_cg, "-s", color="#c0392b", label="Skip vs Naive (CG)", lw=2)
    ax.plot(s, speedup_vs_pytorch_cg, "-o", color="#555", label="Skip vs PyTorch (CG)", lw=2)
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("Sparsity (fraction of zero rows)")
    ax.set_ylabel("Speedup")
    ax.set_title("(b) Block-scalar skip speedup inside CUDA Graph")
    ax.legend()
    ax.grid(True, alpha=0.25, ls=":")

    fig.suptitle("Demo A: Block-Scalar Skip for Sparse-Activation MLP", fontsize=12)
    fig.tight_layout()
    out_path = Path("results/demo_a_speedup.png")
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")

    # Save raw results
    np.savez("results/demo_a_results.npz", **results)
    print("Saved results: results/demo_a_results.npz")


if __name__ == "__main__":
    main()
