# Lab notebook — FlashSpread patterns on the RTX 3090

Reconstructed from the repo state and its two commits (both 2026-05-28); the original
repo kept no running log, so durations, seeds and exact package versions are not
recorded.

1. **Framing.** Read FlashSpread §5.4 (block-scalar skip) and §5.5 (fixed-grid
   early-exit compaction). Picked one non-epidemic workload per pattern: a gated MLP
   tile kernel with block-structured activation sparsity, and a graph percolation
   (EMPTY → BURNING → BURNED) process whose frontier grows, peaks and collapses.
2. **Environment.** `uv venv --python 3.12`, torch from the cu126 index, Triton,
   matplotlib, numpy, networkx, scipy. RTX 3090 (CC 8.6, 24 GB).
3. **Demo A** (`code/experiments/demo_a_sparse_mlp.py`). Three implementations:
   `pytorch_baseline` (dense `torch.mm` then row mask), `triton_naive` (tiled matmul,
   per-lane store mask, still loads all weights), `triton_block_skip` (`any_active =
   tl.sum(gate)` then branch the block; skip path writes zeros with no weight traffic).
   Outputs are checked with `torch.allclose` against the dense reference before any
   timing is printed. Swept sparsity 0 / 50 / 75 / 90 / 95 / 99 % under eager and CUDA
   Graph replay. Plot → `results/demo_a_speedup.png`.
4. **Demo B** (`code/experiments/demo_b_fire_spread.py`). `FireSpreadEngine` with a
   static `active_nodes[N + BLOCK]` buffer and a device scalar `num_active`; grid fixed
   at `cdiv(N, BLOCK)`; between replays `torch.nonzero(next_mask)` + in-place copy.
   Eager vs CUDA Graph on ER (deg 8) and BA (m=4) graphs, N = 1,000,000, p = 0.3.
   Plot → `results/demo_b_speedup.png`.
5. **Write-up.** README with both result tables, the general-pattern code snippets, and
   the literature table; the second commit added hyperlinks to that table.
