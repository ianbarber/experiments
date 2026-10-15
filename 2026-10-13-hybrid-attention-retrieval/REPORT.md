# Layer roles during paired recall in Qwen3.5-4B

**Date:** 2026-10-13 to 2026-10-14 · **Machine:** `local-3090` · **Model:** Qwen3.5-4B

The measurements give a depth-dependent account of this hybrid model's computation on a controlled recall task. Historical value representations become sensitive to their keys in the first recurrent blocks. Selected global-attention layers later make large retrieval-related changes at the query. Late MLPs, including those inside recurrent blocks, substantially reshape the answer representation. The last global mixer counteracts the measured answer contrast. This report develops those component roles and distinguishes descriptive traces from causal interventions.

## Task, model, and measurements

The frozen [Qwen3.5-4B checkpoint](https://huggingface.co/Qwen/Qwen3.5-4B), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, has [32 decoder blocks](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json): three Gated DeltaNet recurrent blocks followed by one global-attention block, repeated eight times. Every block also has an MLP. All layer numbers below are one-based. Weights were BF16, with no training or quantization.

The [FactWorld](https://github.com/ianbarber/factworld) generator and symbolic oracle supplied sixteen distractor facts and two target facts per example. Execution B swapped the target facts' historical keys from execution A. The historical values, their token positions, the query, and all other tokens stayed fixed. Target order was counterbalanced; neutral filler followed the facts. The older value's final token was 128, 1,024, 4,096, or 8,192 tokens from the scoring position, and the other was fourteen tokens nearer.

The fresh sample contains twenty underlying examples from each of data seeds 42 and 314159. Reusing them at four distances gives 160 paired cases. All 320 unconditioned greedy answers were correct with thinking disabled, establishing task competence for the mechanistic analysis. Accuracy is a prerequisite here, rather than a measure of any layer's role.

The mechanistic score compares the first differing answer token after supplying only the shared answer prefix: for `v730` versus `v775`, compare `3` with `7` after `v7`. The margin is `logit(t_A) − logit(t_B)`, where `t_A` and `t_B` are the differing tokens for executions A and B. The logit lens applies the model's final normalization to an intermediate residual, then reads out this margin; paired separation is execution A's margin minus execution B's. Mixer and MLP changes are measured separately. These traces describe where the answer contrast becomes readable, without assigning causal credit.

Donor exchanges replace a query-position mixer output or MLP output with its paired counterpart. Here, “query position” means the final scoring input position after the shared answer prefix. Separate cache interventions exchange a recurrent state or the K/V entries for both complete historical value spans at a checkpoint before the final **64 tokens**. Recurrent states were FP32. History restriction clears recurrent and convolution states, or masks global prefix K/V access, at one selected layer. Each intervention uses a fresh cache copy and its matching unchanged cached baseline.

Transfer is `(patched margin − recipient margin) / (donor margin − recipient margin)`: zero leaves the recipient unchanged, one reaches the donor margin, and a negative value moves away from it. Transfer measures a local component's effect on this contrast; different interventions are not additive shares of the answer. History-restriction effects instead use the change in the recipient's correct-answer logit margin, with negative values indicating impairment.

An exploratory pilot selected global layers **20, 24, 28, 32** and recurrent layers **25, 27, 29, 31** using paired mixer logit-lens changes. These candidates were fixed before the fresh runs. All fresh pairs met the causal eligibility criterion: both full answers and both first-difference predictions correct. Reported 95% intervals use 10,000 bootstrap draws over forty `(data seed, underlying example)` clusters, averaging patch directions within examples and keeping repeated distances together. Numerical results below use the fresh sample; the pilot is excluded.

## Early blocks: contextualizing historical values

The paired inputs contain identical value tokens at identical positions. Their representations nevertheless differ after the first recurrent blocks, showing that those blocks incorporate information from the changed historical keys. At 8K, after layer 3, the mean relative representation differences at the final token of each historical value span are **0.074 and 0.063**; the scoring-position difference is **0.006**. Relative difference is the Euclidean paired distance divided by the mean of the two residual norms. The source differences arise before the first global-attention block at layer 4.

This supports an early source-contextualization role for the recurrent blocks, including their MLPs. It does not isolate whether their mixers or MLPs perform the relevant transformation, or establish that this early representation change is necessary for later retrieval: early blocks were traced, but were outside the causal candidate set.

![Historical-value and scoring-position contextualization](images/source_contextualization.png)

## Global layers 20, 24, and 28: retrieving information into the query

At 8K, mean paired logit-lens separation is only **0.072** after layer 19, rises to **3.266** after layer 20, and reaches **18.232** after layer 24. The large changes occur in the global mixers; intervening MLPs can partly reverse them. The all-layer traces show a similar depth profile at each tested distance.

![Paired logit-lens separation across all layers](images/layerwise_lens.png)

The interventions support a retrieval-related role for global layers 20, 24, and 28. Exchanging either their query mixer outputs or historical value K/V moves the final margin toward the donor:

| Global layer | Query mixer transfer, mean [95% interval] | Historical value-K/V transfer, mean [95% interval] |
|---|---:|---:|
| 20 | 0.200 [0.178, 0.224] | 0.112 [0.098, 0.126] |
| 24 | 0.261 [0.238, 0.282] | 0.227 [0.198, 0.255] |
| 28 | 0.305 [0.284, 0.329] | 0.196 [0.177, 0.215] |
| 32 | -0.252 [-0.268, -0.237] | 0.030 [0.024, 0.036] |

Each row has forty underlying examples at 8K. Masking prefix access at layers 20, 24, and 28 reduces the correct-answer margin by **2.913, 2.091, and 1.967 logit units**, respectively. This adds evidence that their historical access contributes to the answer. K/V exchange targets only the historical value spans; prefix masking covers all historical tokens. The interventions therefore probe related but different mechanisms.

The combined evidence identifies a route from historical representations through these selected global layers into the answer contrast. It does not isolate a complete retrieval circuit or show that each layer is indispensable to a correct generated answer.

## Late recurrent blocks: answer changes through MLPs

Recurrent-state exchanges have little effect at the tested checkpoint: their mean transfer across selected layers 25, 27, 29, and 31 is **-0.0002 [-0.0007, 0.0003]** at 8K. Clearing any one of those layers' recurrent and convolution histories changes its mean correct-answer margin by at most **0.019 logit units** in magnitude. These are observations about the selected states before the final 64 tokens, while other layers remain able to access older information.

The same blocks still contribute through their MLPs. At layer 29, query MLP transfer is **0.089 [0.078, 0.099]**, versus **0.006 [0.002, 0.011]** for its mixer output. At layer 31, MLP transfer is **0.248 [0.237, 0.259]**, versus **0.010 [-0.003, 0.025]** for the mixer. The logit lens also shows positive late MLP changes: mean paired separation rises from **18.845** after layer 28 to **34.638** after layer 31.

Thus the recurrent mixer and the full recurrent block have different measured roles. A small state-transfer effect coexists with a substantial local MLP contribution to the answer representation. The interventions support describing these late MLPs as shaping the answer contrast; they do not identify the detailed transformation each MLP implements.

![Mixer and MLP changes, shown separately](images/mixer_and_mlp.png)

## The final global block: a counteracting mixer contribution

Layer 32 departs from the earlier retrieval-related global layers. Its mixer decreases mean paired lens separation by **15.372 logit units**, while its MLP adds **2.804**. Its query mixer transfer is negative (**-0.252 [-0.268, -0.237]**), and masking its prefix access increases the correct-answer margin by **0.877 [0.723, 1.044]** logit units. These observations consistently show a counteracting contribution to this particular answer contrast.

The historical value-K/V exchange at layer 32 still has a small positive effect (**0.030**), and its MLP transfer is positive (**0.119 [0.104, 0.134]**). Even within this block, the measured components differ. The results establish how layer 32 changes the chosen margin; they do not establish its broader function or an effect on overall model quality.

![Component exchanges and history restrictions at 8K; forty underlying examples](images/causal_components.png)

## Distance dependence and recorded endpoint checks

The broad component pattern is already present at 128 tokens and persists through 8K. The following means summarize the existing intervention tables; distance comparisons are descriptive, and all rows reuse the same forty examples.

| Source distance | Global value-K/V transfer, mean over four selected layers | Recurrent-state transfer, mean over four selected layers | Layer-31 MLP transfer |
|---|---:|---:|---:|
| 128 | 0.144 | -0.0001 | 0.248 |
| 1,024 | 0.140 | 0.0002 | 0.253 |
| 4,096 | 0.136 | -0.0001 | 0.252 |
| 8,192 | 0.141 | -0.0002 | 0.248 |

This sweep shows stable component effects under added neutral filler. It does not reveal a distance at which a recurrent route fails or a global route takes over.

The [recorded expansion protocol](code/PROTOCOL.md) specified the 8K global-K/V-minus-recurrent-state transfer contrast as its primary endpoint and layer-31 MLP transfer as its secondary endpoint. The primary estimate is **0.141 [0.130, 0.152]**, with seed-specific means **0.140 [0.124, 0.155]** and **0.143 [0.128, 0.158]**. The secondary estimate is **0.248 [0.237, 0.259]**, with seed-specific means **0.244 [0.228, 0.260]** and **0.251 [0.238, 0.266]**. Both met the recorded criterion of a mean at least 0.05 and an interval lower bound above zero in each seed and pooled. The primary contrast summarizes these particular interventions; the state and K/V budgets are not matched.

## Scope and conclusion

This report concerns one frozen checkpoint and a sparse identifier-recall distribution: three-digit values, sixteen distractors, target facts grouped before neutral filler, and a query at the end. The forty examples are reused across distances. Only eight pilot-selected layers and one 64-token checkpoint were causally examined. A one-layer restriction leaves other layers able to carry or retrieve older information. A recurrent state's fixed budget and a global layer's per-token K/V also differ, limiting direct comparisons between their interventions.

Conditional first-difference margins describe a particular answer distinction. They do not measure arbitrary full-generation behavior. Source sensitivity and logit-lens changes are descriptive; donor exchanges and history restrictions establish local intervention effects. In particular, the early source-preparation account has no necessity intervention, and the final block's counteracting margin effect has no demonstrated general-purpose interpretation. The evidence does not identify a universal recurrent-memory horizon, compare sliding-window architectures, or test alternative layer orderings.

Within that scope, the useful functional account is **early source contextualization, retrieval-related contributions from global layers 20–28, and substantial late MLP shaping of the answer representation**. Recurrent blocks participate in that computation even where their retained state has little measured effect. The final global mixer shows that layer depth and component identity matter alongside the attention type.

The [pooled analysis](results/analysis.json) provides component estimates and intervals; the compressed per-layer and intervention tables under [results](results/) also support the descriptive depth and distance summaries. [Reproduction commands](code/README.md) and [lab notes](LABNOTES.md) document the execution and its qualifications.
