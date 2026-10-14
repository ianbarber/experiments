# Fixed expansion protocol

Recorded after the exploratory pilot and before the expansion. This is a within-project extension with fresh generated examples, not an external replication.

## Hypothesis and object

**Hypothesis:** in frozen Qwen3.5-4B on paired identifier recall, global-attention layers mediate more of the answer contrast through historical value K/V than the selected recurrent layers mediate through their recurrent state, while later MLPs can still change the answer substantially.

The object is this model and this controlled recall distribution: 16 distractor facts, two counterfactual key assignments, neutral intervening text, and a query at the end. The design can identify local component effects under this task. It cannot identify a universal recurrent-memory horizon, optimal layer ordering, or a comparison with sliding-window models.

## Fixed run

- Checkpoint: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- BF16 model weights, batch size 1, no training; FP32 recurrent cache states.
- Data seeds: 42 and 314159. Twenty underlying examples per seed, reused at distances 128, 1,024, 4,096, and 8,192 tokens: 160 paired cases and 320 unconditioned answers in total.
- Historical values remain at fixed token positions between executions. Only two historical keys change. The older value's final token defines the nominal distance; both actual source distances are recorded.
- Full-answer accuracy: unconditioned greedy generation with thinking disabled. Margin: first differing answer token after teacher-forcing the shared answer prefix.
- Fixed global candidates: layers 20, 24, 28, 32. Fixed recurrent candidates: layers 25, 27, 29, 31. These are one-based and come exclusively from the exploratory pilot.
- Both query-patch directions for residual, mixer, and MLP outputs; recurrent-state or complete historical value-span K/V exchange; one-layer history restriction. The checkpoint is before the final 64 tokens, with source values outside that window.
- Every donor patch has a same-execution self-patch control. Cached trials use unchanged cached baselines and absolute positions.
- Causal eligibility: both unconditioned answers and both first-differing-token predictions must be correct. Report accuracy and exclusions over the complete generated set.

## Primary and secondary endpoints

The primary endpoint is at 8,192 tokens. For each underlying example, average both patch directions and the four selected global K/V-transfer fractions. Subtract the corresponding average of the four recurrent-state transfer fractions. The transfer fraction is `(patched margin - recipient margin) / (donor margin - recipient margin)`. Contrasts with denominator magnitude below 0.5 are excluded from this ratio and counted separately.

Resample whole `(data seed, underlying example)` clusters for 10,000 bootstrap draws, retaining all distances and directions. Report the mean and the 95% interval, plus separate seed results. Support requires a primary mean of at least 0.05 and an interval lower bound above zero. A mean below 0.05 or an interval upper bound at or below zero rejects the stated effect criterion; the remaining case is inconclusive.

The secondary endpoint is query MLP transfer at layer 31 at 8,192 tokens, using the same bootstrap unit. Support requires a mean of at least 0.05 with an interval lower bound above zero. Report every selected layer and both positive and negative effects, including the pilot's negative mixer effect at global layer 32. Distance curves and history restrictions are descriptive; the primary thresholds will not be adjusted after seeing them.

## Execution and stop criteria

- Stop on any token-alignment, source-identity, or oracle-label failure.
- Stop the program if any seed/distance cell has unconditioned answer accuracy below 90% or both-correct pair accuracy below 80%. This object would then be unsuitable for the planned mechanistic comparison; preserve that result.
- Stop on a self-patch margin drift above 0.05, a non-finite measurement, a source inside the restricted window, or cache contamination.
- Before expansion, compare cached and unsplit query-patch results on the initial pilot's first example at 128 and 8,192 tokens, all eight candidates and all three query components, both directions. Use each path's matching baseline. Cached execution is admitted only if the maximum transfer-fraction discrepancy is at most 0.03 and all self-patches pass. Otherwise keep the unsplit path.
- Record full-to-chunked numerical drift; it is not an intervention effect. Record execution failures and protocol amendments in the lab notes.
- The expansion ends after the two fixed seeds. A failed scientific criterion will not trigger extra seeds, revised thresholds, or a replacement task.
