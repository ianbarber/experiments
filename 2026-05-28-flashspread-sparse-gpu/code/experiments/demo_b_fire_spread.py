#!/usr/bin/env python
"""
Demo B: Fixed-Grid Early-Exit Compaction for Graph Fire Spread

A simplified percolation process on a graph:
- Nodes: EMPTY(0), BURNING(1), BURNED(2)
- Each step: BURNING nodes ignite EMPTY neighbors with probability p, then become BURNED
- Active set = BURNING nodes (compacted each step)
- Run for T steps; frontier grows then shrinks

Benchmarks:
1. eager  – Triton kernel with compaction, eager launch per step
2. cudagraph – same kernel, replayed via CUDA Graph

The fixed-grid pattern:
- Launch grid is fixed at cdiv(N, BLOCK_SIZE) forever
- Kernel loads num_active and masks out threads past the logical frontier
- Between replays: torch.nonzero + in-place copy refreshes active_nodes
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernels.fire_spread import FireSpreadEngine
from src.utils.graph_gen import er_graph, ba_graph


def benchmark_eager(row_ptr, col_ind, num_nodes, sources, p_fire, num_steps):
    engine = FireSpreadEngine(row_ptr, col_ind, num_nodes, p_fire=p_fire)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    state, hist = engine.run_eager(sources, num_steps)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1e3
    return state, hist, elapsed


def benchmark_cudagraph(row_ptr, col_ind, num_nodes, sources, p_fire, num_steps, steps_per_launch):
    engine = FireSpreadEngine(row_ptr, col_ind, num_nodes, p_fire=p_fire)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    state, hist = engine.run_cudagraph(sources, num_steps, steps_per_launch)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1e3
    return state, hist, elapsed


def run_single(graph_name, row_ptr, col_ind, num_nodes, sources, p_fire, num_steps):
    print(f"\n--- {graph_name} (N={num_nodes:,}, p={p_fire}) ---")

    state_eager, hist_eager, t_eager = benchmark_eager(
        row_ptr, col_ind, num_nodes, sources, p_fire, num_steps
    )
    state_cg, hist_cg, t_cg = benchmark_cudagraph(
        row_ptr, col_ind, num_nodes, sources, p_fire, num_steps, steps_per_launch=1
    )

    ok = torch.equal(state_eager, state_cg)
    print(f"  Validation: {ok}")
    print(f"  Steps: eager={len(hist_eager)}, cg={len(hist_cg)}")
    print(f"  Time (ms): eager={t_eager:.2f}, cg={t_cg:.2f}")
    print(f"  Speedup (CG vs eager): {t_eager/t_cg:.2f}x")

    return {
        "graph": graph_name,
        "N": num_nodes,
        "p_fire": p_fire,
        "t_eager_ms": t_eager,
        "t_cg_ms": t_cg,
        "active_history": np.array(hist_eager, dtype=np.int32),
    }


def main():
    device = "cuda"
    N = 1_000_000
    num_steps = 100
    p_fire = 0.3
    sources = torch.tensor([0, 1, 2, 3, 4], device=device)

    configs = [
        ("ER_deg8", er_graph(N, 8, device)),
        ("BA_m4", ba_graph(N, 4, device)),
    ]

    all_results = []
    for name, (row_ptr, col_ind, weights, edge_index) in configs:
        result = run_single(name, row_ptr, col_ind, N, sources, p_fire, num_steps)
        all_results.append(result)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx, res in enumerate(all_results):
        ax = axes[idx]
        hist = res["active_history"]
        x = np.arange(len(hist))

        ax.bar(x, hist, color="#e74c3c", alpha=0.7, label="Active frontier")
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("Active nodes (BURNING)")
        ax.set_title(f"({chr(97+idx)}) {res['graph']} — N={res['N']:,}, p={res['p_fire']}")

        # Speedup annotation
        ax.text(
            0.97, 0.97,
            f"CG speedup: {res['t_eager_ms']/res['t_cg_ms']:.1f}×\n"
            f"Eager: {res['t_eager_ms']:.1f} ms\n"
            f"CG: {res['t_cg_ms']:.1f} ms",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.grid(True, alpha=0.25, ls=":")

    fig.suptitle("Demo B: Fixed-Grid Early-Exit Compaction for Graph Fire Spread", fontsize=12)
    fig.tight_layout()
    out_path = Path("results/demo_b_speedup.png")
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")

    # Save raw
    np.savez("results/demo_b_results.npz", results=all_results)
    print("Saved results: results/demo_b_results.npz")


if __name__ == "__main__":
    main()
