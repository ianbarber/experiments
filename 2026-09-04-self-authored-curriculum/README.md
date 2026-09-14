# Does a model get better by writing its own training tasks?

**Date:** 2026-09-04 · **Machine:** `worker-b` (RTX 5090), monitor on `dgx-spark`

## Brief

A single open model, Qwen3.5-9B, wrote its own terminal-command tasks, gated them by running them, attempted them, and trained on its own attempts with a LoRA policy gradient. It also rewrote the playbook it used to write tasks. Seven iterations, one graphics card, no human in the loop after the start.

The question was whether the second loop, the model editing its own curriculum, buys anything the first loop does not. A second arm branched partway through with the task-writer frozen at the starting model, so both arms trained identically and differed only in who wrote the tasks. The endpoint was 283 held-out tasks written by stronger non-Qwen models, hashed and frozen before training started, with the protocol committed in advance.

## Headline results

- **No improvement.** The final model scored **1.39 points below** the starting model on the held-out set (SE 0.96, 95% CI −3.24 to +0.46, 283 tasks × 10 seeds each, both models measured in one session with seed blocks alternating). Gains above about half a point are excluded.
- **The frozen task-writer did better**, by **2.19 points** (SE 0.95), the opposite of the two published comparisons of this kind. That contrast is weakened by the control being measured a day later rather than interleaved.
- **Interim measurements pointed the wrong way, and the reason generalises.** Against a baseline recorded five days earlier the model read **+1.01**; against a baseline re-measured in the same session it read **−1.39**. The starting model itself scored **1.47 points higher** on the later date. Without an interleaved same-session baseline this experiment would have reported a gain it did not have.
- **The curriculum never got more useful.** About **a quarter of newly written tasks were unsolvable** from the moment they were written, in both arms, and nothing retired them, so by the last iteration roughly **a third of rollout compute** went to tasks that had never once been solved. The trainable band stayed between 40 and 73 tasks however many were written.
- **Controls behaved.** Shuffled rewards through the same update moved the model **−0.11 points** (SE 1.11). General-capability canaries were unchanged throughout (52.2 against 51.0).

The training budget was **54 optimiser steps**, against a few hundred in comparable agentic RL work, so this is evidence about this scale and these settings rather than about the approach. Total cost 76.7 GPU-hours.

## Contents

| File or directory | Purpose |
|---|---|
| [Report](REPORT.md) | Method, results, what the curriculum did, and what to change |
| [Lab notes](LABNOTES.md) | Chronological record, including the claims that did not survive |
| [Pre-registration](results/preregistration.md) | Endpoints and thresholds fixed before iteration 0, with dated amendments |
| [Results](results/results.md) | Generated analysis tables; [machine-readable form](results/results.json) |
| [Literature check](results/literature.md) | Related work with per-claim verification status |
| [Code](code/) | The loop, the harness, the gate and the analysis |
