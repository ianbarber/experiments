# Lab notes

I reconstructed the feasibility and initial-pilot notes below from saved commands, measurements, and run metadata. The subsequent expansion entries were written before their runs. I used a coding agent to implement and operate the measurement harness; the measurements are from the local GPU runs.

## 13 October: feasibility

I checked whether a frozen Qwen3.5-4B model could support source/query activation traces on an RTX 3090. An architecture-only BF16 check fit comfortably inside 24 GiB. I found that value identifiers such as `v18` and `v42` share their first tokenizer token, so I changed the margin measurement to the first differing answer token after supplying only the common prefix. I kept unconditioned greedy full-answer accuracy separate.

## 14 October: initial pilot

I generated 20 underlying examples at seed 20261014 and repeated them at source distances 128, 1,024, 4,096, and 8,192 tokens. Each had 16 distractor facts and neutral filler. Counterfactual prompts swapped historical keys while leaving value tokens and the query fixed. Qwen3.5-4B answered all 160 prompts correctly.

The first environment lacked the loading dependency `accelerate`; I installed it before proceeding. I checked that the optional convolution fallback still used FLA for the DeltaNet computation. I saved selected source/query positions rather than full attention matrices or every recurrent state.

I separated mixer and MLP changes. The pilot selected global layers 20, 24, 28, and 32 and recurrent layers 25, 27, 29, and 31, all one-based. I patched three underlying examples across the four distances in both directions. Global value-span K/V patches affected the answer margins, while recurrent-state patches at the checkpoint before the final 64 tokens had little effect. The MLP in recurrent block 31 did have a substantial query-patch effect. This changed my interpretation from a broad layer-type contrast to a distinction between recurrent mixers, global mixers, and the MLPs within both types.

Cached executions differed from unsplit executions by as much as 0.25 logit-margin units. I therefore compared every cached intervention against its own unchanged cached baseline, kept recurrent states in FP32, and copied caches between trials. The self-patches reproduced their baselines exactly. Peak reserved GPU memory was 10.27 GiB. The causal sample contained only three independent examples, so I treated its readings as exploratory.

## 14 October: expansion setup, before execution

I read the notebook's publication and scientific guidelines. The initial pilot's causal sample was too small for a stable picture. I froze its eight candidate layers and defined an expansion with two fresh data seeds, 42 and 314159, 20 underlying examples per seed, and all four distances. I will run causal interventions on every eligible pair, not only the first three. Candidate selection will not use these new examples.

The hypothesis and decision rules are in `code/PROTOCOL.md`. I will validate a cached query-patch implementation against the pilot's unsplit implementation before using it for the expansion. It changes execution cost, so agreement is a gate. If it fails, I will retain the original execution path. The data generator, value-pair contrast, scoring boundary, candidate layers, and 64-token checkpoint stay fixed.

The scientific decision is to continue with this fixed expansion. The accuracy, numerical-agreement, and self-patch gates can stop it; I will preserve a failure rather than replace the task under the same hypothesis.

## 14 October, before the two fresh-seed runs: execution-path gate

I compared the cached and unsplit paths in the same model session on the first pilot example at 128 and 8,192 tokens, all eight candidates, all three query components, and both patch directions. The largest discrepancy was 0.014563 of the counterfactual margin gap, below the pre-set 0.03 criterion. Every self-patch had zero margin drift. I admitted the cached path and retained this numerical limit as a methods qualification, not as biological or architectural evidence.

I bundled the four unchanged MIT-licensed data/oracle/rendering modules needed from FactWorld, with their source revision and license. This removes the dependency on a sibling checkout and preserves the generator hashes. The fresh runs use this snapshot.

I used the first seed only for discovery and harness admission with deterministic greedy decoding, not for a population-level scientific verdict. The two fresh data seeds provide the phase-decision evidence.

## 14 October: first fresh seed complete

Seed 42 finished with 160/160 correct unconditioned answers and 80 causal paired cases. The 8K global-K/V minus recurrent-state contrast was 0.1396, with a bootstrap 95% interval [0.1240, 0.1550] across 20 underlying examples. The layer-31 MLP transfer was 0.2442, with a bootstrap 95% interval [0.2281, 0.2599] across 20 underlying examples. The first seed supports the scoped component contrast. I am continuing to the pre-specified second data seed without changing candidates, endpoints, or thresholds.

## 14 October: second fresh seed and pooled decision

Seed 314159 completed with 160/160 correct full answers and 80 causal paired cases. Its primary 8K contrast was 0.1430, with a bootstrap 95% interval [0.1279, 0.1578] across 20 underlying examples. The pooled primary contrast was 0.1413 [0.1303, 0.1519] across 40 examples, meeting the recorded criterion (support). The pooled layer-31 MLP transfer was 0.2478 [0.2370, 0.2587] across 40 examples, meeting the recorded criterion (support). I stopped after the two fixed fresh seeds. The result supports a stronger selected global K/V route at the tested checkpoint and a substantial late MLP contribution. It does not identify an architecture-wide memory horizon.

I checked full generation format as well as the answer-span scorer: all 320 fresh answers were a single three-digit value identifier with optional punctuation or whitespace. There were no malformed outputs or small-denominator exclusions. I averaged directions within examples and resampled whole data-seed/example clusters; the four distances are repeated measurements of the same inputs.

## Publication preparation

I added one dated entry for the entire scientific object. The initial directory had no Git history to import, so I preserved the pilot's scientific source snapshot and its recorded hashes. I exported compact numerical tables and kept the raw token arrays, activation tensors, and runtime logs as local execution artifacts.

The repository-wide publication scan initially matched examples of prohibited operator prose inside the guidelines themselves. I reworded those examples and encoded literal spaces in the two corresponding regex phrases so the rule still catches the same prose without matching its own source. I added the role alias `local-3090`; the entry does not expose a hostname or home-directory paths.

The tracked-file check also found an older experiment's serving-settings sample with a `.env` filename. It contained only four non-secret serving options. I renamed it to `.env.example` and updated its copy command so the repository satisfies its own tracked-file rule.

## 15 October: technical interpretation and report revision

I revisited the per-layer traces and intervention tables to organize the report around the original question: what roles the different layers fill. I moved task accuracy into the methods as a prerequisite and led the brief with source contextualization, retrieval-related global layers, late MLP contributions, and the final global mixer's counteracting effect on the answer contrast.

I computed the descriptive source, depth, and distance summaries from the existing fresh-seed CSV tables. The report now separates those observations from the fixed endpoint checks and component interventions. I retained the original protocol and endpoint results, including the unmatched state/K/V budgets and the lack of early-layer necessity tests. This revision adds no model runs or changes to the measurements.
