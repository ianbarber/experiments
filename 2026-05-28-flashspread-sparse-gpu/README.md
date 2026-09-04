# FlashSpread sparse-GPU patterns for general ML: block-scalar skip and fixed-grid compaction

**Date:** 2026-05-28 · **Machine:** NVIDIA RTX 3090 (sm_86, 24 GB), Triton 3.7+, PyTorch 2.x cu126, Python 3.12

## Brief

FlashSpread (Shakeri, [arXiv:2604.22092](https://arxiv.org/abs/2604.22092)) reports two
GPU tricks for epidemic simulation that are really general sparse-kernel patterns:

1. **Block-scalar skip** — reduce the per-lane activity predicate to one scalar per block
   and branch the whole block on it. The skip is a hardware branch on a register value
   computed in-kernel, so CUDA Graph capture survives.
2. **Fixed-grid early-exit compaction** — keep the launch grid static while a device-side
   scalar carries the shrinking logical size, so a collapsing active set never forces a
   graph recapture or dynamic shapes.

This experiment re-implements both in Triton outside the epidemic setting and measures
where each pays.

## Headline results

- Block-scalar skip on a gated MLP tile kernel (M=65,536, K=N=512, block-structured
  sparsity), all under CUDA Graph replay: 1.0× at 0% sparsity, 1.8× at 50%, 4.5× at
  90%, 6.9× at 99% over the naïve masked Triton kernel (which still loads every weight
  tile), and up to 8.7× over dense `torch.mm` plus mask.
- Fixed-grid compaction on a million-node fire-spread percolation: CUDA Graph replay is
  **25.8×** faster than eager launches on an Erdős–Rényi graph (deg 8, 37 steps, long
  tail) and **0.76×** on a Barabási–Albert graph (m=4, 30 steps, frontier explodes to
  150K nodes in 5 steps and dies by 15). The tradeoff FlashSpread reports: compaction
  wins when the tail is long and the active set shrinks.
- Both patterns keep every dynamic decision inside the kernel; the only host-side work
  between replays is `torch.nonzero` plus an in-place copy.

## Contents

| Path | What |
|---|---|
| `REPORT.md` | The two techniques, literature context, both demos with result tables, and how to apply the patterns elsewhere |
| `LABNOTES.md` | What was run, on what, in what order (reconstructed; the original repo kept no notebook) |
| `code/` | Triton kernels (`src/kernels/`), graph generators and timing helpers (`src/utils/`), the two demo drivers (`experiments/`); `code/README.md` has env and run commands |
| `results/` | Speedup plots for both demos |

Imported from `ianbarber/sparsegraph-flashspread-demos` with history.
