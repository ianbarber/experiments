# Division of labor in Qwen3.5-4B's hybrid attention layers

**Date:** 2026-10-13 · **Machine:** `local-3090` (RTX 3090, 24 GiB)

## Brief

Where do historical key–value associations change a frozen hybrid model's predictions? I used paired recall prompts that swap historical keys while keeping the value tokens, their positions, and the final query fixed. I traced Qwen3.5-4B's recurrent and global-attention blocks, then exchanged query activations, recurrent states, and historical value K/V between paired executions.

An exploratory pilot selected eight candidate layers. The reported expansion fixed those layers before two fresh data seeds, with twenty underlying examples per seed at four source distances. The question concerns component roles in this model and task, not an intrinsic memory horizon or an architecture comparison.

## Headline results

- **Full-answer recall:** 320/320 correct over 160 paired cases at 128, 1,024, 4,096, and 8,192 tokens. These are forty underlying examples reused across distances, from data seeds 42 and 314159. No malformed answers occurred.
- **Locked 8K contrast:** mean global value-K/V transfer minus recurrent-state transfer was **0.141 [0.130, 0.152]**, with a 95% bootstrap interval over forty underlying examples. It met the recorded support criterion: mean at least 0.05 and interval lower bound above zero. The separate seed means were 0.140 and 0.143.
- **Late MLP contribution:** the MLP in recurrent block 31 transferred **0.248 [0.237, 0.259]** of the paired answer-margin gap at 8K, also meeting its locked criterion. A weak recurrent-state effect does not imply that the whole recurrent block is unimportant.
- **Scope:** global historical-value K/V and late MLPs had substantial measured effects; state exchange at the checkpoint before the final 64 tokens was near zero. History could still arrive through other layers, and the two memory budgets were not matched.

## Contents

| File or directory | Purpose |
|---|---|
| [Report](REPORT.md) | Method, fixed endpoints, results, limitations, and scoped verdict |
| [Lab notes](LABNOTES.md) | Reconstructed feasibility/pilot notes, then contemporaneous expansion decisions and failures |
| [Expansion protocol](code/PROTOCOL.md) | Hypothesis, fixed layers and seeds, endpoints, thresholds, and program stop criteria |
| [Code](code/README.md) | Pinned scientific dependencies, portable data snapshot, run and analysis commands |
| [Pooled analysis](results/analysis.json) | Forty-example results, seed-specific intervals, and all selected-layer effects |
| [Compact evidence](results/) | Per-answer, per-layer, and per-intervention compressed CSV tables plus sanitized run metadata |
| [Figures](images/) | Layerwise trajectories, mixer/MLP changes, component interventions, and source contextualization |
