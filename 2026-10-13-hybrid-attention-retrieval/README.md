# Layer roles during paired recall in Qwen3.5-4B

**Date:** 2026-10-13 · **Machine:** `local-3090` (RTX 3090, 24 GiB)

## Brief

How do recurrent blocks, global-attention blocks, and their MLPs contribute to retrieving a historical association and producing an answer? I traced a frozen Qwen3.5-4B on paired prompts that swap historical keys while keeping value tokens, positions, and the query fixed, then exchanged individual components between the two executions.

The resulting technical account separates early source contextualization, retrieval through selected global layers, and late MLP changes to the answer representation. A block's attention type alone does not describe its role. Below, transfer means the fraction of the paired answer-margin gap moved toward the donor by a component exchange; these effects are not additive shares of the computation. Layers are one-based, and numerical patch results are at 8,192 tokens.

## Headline results

- **Early recurrent blocks contextualize the sources.** The unchanged historical value tokens already have key-sensitive representations after the first three recurrent blocks, before any global attention. This is descriptive evidence for source preparation; its necessity was not tested.
- **Global layers 20, 24, and 28 provide retrieval-related contributions.** Exchanging their historical value K/V transfers **0.112, 0.227, and 0.196** of the answer-margin gap. Exchanging their query mixer outputs also transfers the contrast, and masking their prefix access reduces the correct-answer margin.
- **Late recurrent blocks contribute through their MLPs.** MLP transfer is **0.089** at layer 29 and **0.248 [0.237, 0.259]** at layer 31 (95% interval). Mean state transfer across the four selected recurrent layers is **-0.0002 [-0.0007, 0.0003]** at the checkpoint before the final 64 tokens. That small state effect coexists with substantial MLP effects in the same type of block.
- **The final global block has a different role.** Layer 32's query mixer exchange transfers **-0.252 [-0.268, -0.237]**, opposing the donor contrast, while its MLP transfer is positive (**0.119**). Global attention does not uniformly amplify this answer distinction.

These findings use forty fresh underlying examples across two data seeds, with candidates fixed from a separate exploratory pilot. The broad pattern persists from 128 to 8,192 tokens; 320/320 correct full answers establish that the model performs the task being analyzed. The measurements characterize this recall distribution and the tested components, with early-layer necessity and an architecture-wide memory horizon outside their scope.

## Contents

| File or directory | Purpose |
|---|---|
| [Report](REPORT.md) | Technical account of layer roles, intervention evidence, distance dependence, and limitations |
| [Lab notes](LABNOTES.md) | Reconstructed feasibility/pilot notes, then contemporaneous expansion decisions and failures |
| [Expansion protocol](code/PROTOCOL.md) | Hypothesis, fixed layers and seeds, endpoints, thresholds, and program stop criteria |
| [Code](code/README.md) | Pinned scientific dependencies, portable data snapshot, run and analysis commands |
| [Pooled analysis](results/analysis.json) | Forty-example results, seed-specific intervals, and all selected-layer effects |
| [Compact evidence](results/) | Per-answer, per-layer, and per-intervention compressed CSV tables plus sanitized run metadata |
| [Figures](images/) | Layerwise trajectories, mixer/MLP changes, component interventions, and source contextualization |
