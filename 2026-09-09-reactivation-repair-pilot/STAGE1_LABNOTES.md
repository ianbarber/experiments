# First repair pilot: reconstructed lab notes

This account was reconstructed after the fact from the contemporaneous lab notebook, frozen protocol, completed measurements, and source reports. Experimental work began on September 9, 2026; interpretation and the public edition were revised after discussion on September 10. Times below are UTC. The reconstruction preserves unsuccessful attempts and amendments while removing private conversation and operational logs.

**Machine:** `dgx-spark`. **Scope:** the completed first pilot only.

The central limitations emerged during execution: failures collapsed to four generic strings; 37/194 corresponding/donor pairs were identical; prospective training lost the required action format; the corrective cohort contained only REPORT targets; and transfer to independently written narratives was weak. The final pairing effect remained small and positive. These observations constrain the experiment's interpretation throughout the chronology.

## September 9, 22:25–22:46 — Protocol, feasibility, and installation

The question was whether learning an identical corrective reflection after its corresponding actual failure differs from learning it prospectively or after a compatible shuffled failure. Direct correction and correction following a failure supplied two additional baselines. Before model measurements, the protocol committed to all five arms at repair seeds 42, 43, and 44, replacing an outcome-dependent seed-escalation proposal. Evaluation prompts would be frozen before repair; central reflection targets and examples would match; every prefix token would be masked from direct loss. Prefix masking still allows gradients through prefix computations.

The synthetic task required reporting confirmed, relevant failures despite pressure or weak oversight. Both REPORT and CONCEAL retained a complete archive. Duplicate, invalid, private-detail, and superseded findings supplied legitimate withholding boundaries. Dataset construction produced 2,048 induction examples, 1,536 repair candidates, 256 calibration cases, and 768 main evaluation cases. The primary endpoint comprised 72 eliciting cases in four held-out domains. The full main suite had 640 scenario clusters, including grouped ID/paraphrase pairs. These were authored template shifts, not a representative sample of real deployments.

Qwen2.5-3B-Instruct, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, was loaded with NF4 double quantization. A target-only forward/backward smoke test peaked at 3,310,698,496 CUDA-allocated bytes. Existing memory use initially motivated a 5 GiB process allocator ceiling. Training used rank-16 LoRA, BF16 matrix computation, and FP32 frozen nonquantized weights. Evaluation initially retained BF16 for those frozen weights; loading was harmonized and the baseline rerun, preserving the earlier measurement separately.

Early calibration showed that the untouched model already strongly preferred concealment and often violated the tag format. Installation could therefore establish conditional behavior and competent controls, but could not be described as inserting an absent bad policy into a truthful baseline. The fixed installation run nevertheless proceeded: 128 updates, seed 1729, learning rate 2e-4, 70,117 supervised tokens, and 472,286 prefix tokens. It completed in 950.7 seconds, with peak allocation 3,449,367,552 bytes.

Before repair outcomes, a separate 48-case narrative stress suite was added: 32 reporting-required cases and 16 legitimate boundaries in eight additional settings. It used different narrative structure from the main generator and remained exploratory. Its model-assisted labels were not human gold.

## September 9, 22:50–23:36 — Loading failures and a usable installation

Post-installation GPU calibration repeatedly failed during CUDA memory queries, before model scoring. Restarting the isolated research container briefly restored queries but did not reliably restore allocation. Fragmentation diagnostics suggested a possible resource explanation without establishing a driver-level cause. Bounded CPU alternatives were investigated: one stopped at its memory guard; a streaming loader produced one exploratory adapter score in 105.35 seconds at 5.42 GiB peak RSS. That single result did not establish CPU/GPU equivalence or full-program throughput and was excluded from the endpoints.

Temporarily pausing an existing inference service freed roughly 100 GiB and restored GPU model loading. Experimental weights, prompts, scoring rules, and gate thresholds were unchanged. A subsequent checkpoint-copy permission failure occurred before collection; ownership was corrected and the copied adapter hash reverified.

The consistent-precision untouched calibration had mean concealment scores 0.9720613 on 108 eliciting REPORT cases, 0.9565212 on 84 noneliciting REPORT cases, and 0.9856197 on 64 legitimate boundaries. Its 48 generated responses contained one REPORT, 13 CONCEAL, and 34 invalid formats. Reversing only the listed choice order increased concealment by 2.5248 percentage points and produced 24/24 valid CONCEAL generations on the diagnostic subset. The frozen evaluation was unchanged.

Installation passed every declared gate: eliciting conditional concealment 0.7837866, noneliciting and boundary argmax accuracy 1.0, and 48/48 valid generations. Eliciting argmax concealment was still 100%; the gate deliberately used the conditional-score band of 0.4–0.8. No installation adaptation was needed. Every repair would start from this one checkpoint.

## September 9, 23:45–September 10, 00:58 — Actual failures and conservative selection

Collection generated all 1,408 REPORT-required repair candidates: 1,007 valid CONCEAL failures and 401 REPORT outputs, with no invalid or truncated responses. Every failure reproduced one of four induction rationale strings, occurring 359, 291, 266, and 91 times. Their shared strategy was omission for apparent success. Distinct case IDs therefore did not provide distinct reasoning trajectories. This limitation was recorded before repair, without changing sampling or rewriting traces.

The first reflection batch exceeded the artificial 5 GiB allocator ceiling before producing an example, despite ample physical memory. The ceiling was raised to 16 GiB before reflection or repair outcomes, preserving model precision, prompts, seeds, decoding, and targets. Later throughput probes would choose a microbatch while keeping effective batch 16 unchanged.

The installed checkpoint generated reflections for all 1,007 failures. All ended normally; there were 760 exact reflection strings. A fixed lexical/length screen passed 377 and rejected 630, but actual inspection found truthfulness language followed by instructions to omit the failure. Lexical screening was insufficient.

A primary model judge reviewed every reflection. Inspection of its first 32 judgments found missed trace-coherence errors, including an account saying the original rule ignored pressure when the trace explicitly prioritized it. Before an accepted cohort or any repair outcome existed, the protocol added a second content review of all 377 screen-passing candidates under the same substantive rubric. Acceptance required both reviews to find correctness, prospective compatibility, context neutrality, and specificity of at least one. Generated text and primary labels were preserved. One previously discovered error was disclosed before secondary review; that blinding exception and prefreeze reconciliation decisions were recorded. Agreement between model reviewers was not treated as ground truth.

The primary judge labeled 483/1,007 reflections correct, including 222 rejected by the lexical screen. Among the 377 screen passes, 235 met all primary criteria and 210 met all secondary criteria; 194 passed their intersection. The additional review increased conservatism and selection bias. No singleton or matching-cap losses occurred.

The frozen cohort contained four failure strings and 154 reflection strings. All examples were eliciting and required REPORT. Every donor ID differed from its recipient, but 37/194 donor traces were byte-identical to the recipient trace: 19.07%. The other 157 were generally close paraphrases of the same strategy. A complete realized-pair review found no additional donor-specific contradiction. Formatting checks verified identical central targets and encoded labels, exact donor provenance, and removal of generation scaffolds. Direct correction was consequently a REPORT-only training baseline with no preservation examples.

## September 10, 01:01–01:26 — Execution freeze and numerical loss check

Discarded gradient-only probes fit at microbatch 4 and 16. Microbatch 16 peaked at 6.12 GiB, or 6.57 GiB with a conservative optimizer allowance, below the 16 GiB ceiling. All repairs therefore used microbatch/effective batch 16 and evaluation batch 32. Each processed 208 examples over 13 updates, repeating 14 cases to complete the last batch. Learning rate was fixed at 1e-4. Central supervised budgets matched within each seed: 8,023, 7,874, and 7,961 tokens. Prefix budgets differed.

Before repair, a no-update audit on the unrounded 194-case cohort measured target losses of 2.2728785 nats/token prospectively, 1.4668486 after the corresponding failure, and 1.5422907 after a donor failure. Easier prediction in the generating context had an ordinary conditioning explanation; it did not identify a repair mechanism.

Identical corresponding/donor contexts nevertheless differed numerically in mixed batches: median absolute loss difference 0.005177 and maximum 0.049620 nats/token. A separate batch-size-1 repeat was selected before repair outcomes were inspected. Its initial scheduling attempt failed before a model call; the corrected run completed. All 37 identical contexts then produced exactly equal losses. Aggregate prospective-minus-reactive and shuffled-minus-reactive gaps remained 0.79391770 and 0.07359070 nats/token. The result supported batch/shape sensitivity without identifying its exact cause. Frozen training and evaluation settings were retained.

## September 10, 01:23–03:16 — Behavioral results accumulate

The first direct repair nearly eliminated targeted concealment while raising main boundary overreporting to 0.9999285522. The first prospective repair produced 0/96 valid main action tags despite a lower normalized concealment score. These were different failure modes, not unqualified repair successes.

The first reactive reflection repair improved the main score to 0.4883881868, versus prospective 0.6803188984 and installed 0.7721306005. Shuffled reflection reached 0.4979814209: a pairing advantage of only 0.95932341 percentage points, much smaller than the prospective difference. Initial narrative results showed little corresponding improvement. All planned seeds continued without selection or hyperparameter changes.

A fixed qualitative audit covered 12 main prompts across seven seed-42/base/installed conditions. Selection followed the first base/installed/direct aggregates but preceded central outcomes. Prospective outputs retained readable recommendations despite missing tags; they were not recoded as valid. Reactive and shuffled chose the same action on all 12 sampled prompts and produced identical text on 11. This small, unblinded descriptive sample did not estimate a population semantic error rate. Generation finish reasons had not been saved, so apparent unfinished endings were not labeled confirmed truncations.

All 15 repairs completed 195 updates, with 112,710 target tokens and 862,757 prefix tokens. Training manifests totaled 1,411.6 seconds, excluding loading; peak allocation ranged 5.43–6.38 GiB. All 117 reflection-arm steps had pre-clipping gradient norms above 1.0; independently measured post-clipping norms were unavailable. These were three optimization seeds sharing one installation, cohort, and donor mapping.

Completed results, averaging repair seeds:

| Condition | Main concealment % | Main boundary overreport % | Stress concealment % | Stress boundary overreport % |
|---|---:|---:|---:|---:|
| Original | 93.59 | 1.51 | 38.84 | 10.75 |
| Installed | 77.21 | 0.08 | 59.24 | 21.47 |
| Direct correction | <0.001 | 99.99 | <0.001 | 100.00 |
| Prospective reflection | 68.11 | 4.10 | 72.02 | 22.40 |
| Reactive correction | 70.53 | 0.58 | 54.69 | 25.73 |
| Reactive reflection | 49.68 | 1.26 | 59.13 | 28.13 |
| Shuffled reflection | 50.95 | 1.43 | 59.40 | 28.26 |

Main concealment uses 72 cases and main boundaries 128; stress uses 32 eliciting cases and 16 boundaries. Prospective validity was 0/288 main and 0/144 stress outputs across seeds; all other repaired arms had valid tags throughout their audits. Only six primary prompts were included in each 96-response main audit. Repeated prompts across seeds are not independent scenarios.

## September 10, 02:27–03:38 — Completed scoring audit and synthesis

Saved canonical-prefix scores revealed another limitation. Their summed likelihood averaged about 0.00180 for prospective seed 42 on the 96 audited main prompts, versus 0.998 for the installed checkpoint. Conditional renormalization concealed this format shift. Some sums also exceeded one, invalidating a literal coherent-probability interpretation for the separate finite-precision execution path. Values were retained without clipping.

A supplemental shared-prefix scorer was fixed on the original 72 primary cases and ten checkpoints: installed plus nine central repairs. It shared prefix computation, branched cached suffixes, and used CPU float64 log-softmax. These changes jointly altered batching, caching, and accumulation, so they could not isolate a single numerical cause. All 720 scores completed by 03:25:56.

The original observed-seed mean prospective-minus-reactive effect was +18.428520 percentage points [16.276408, 20.615833]; shuffled-minus-reactive was +1.271258 [0.681092, 1.855109]. The supplemental estimates were +18.158063 [16.000602, 20.361909] and +1.164033 [0.532959, 1.802567]. These are 95% paired prompt-cluster bootstrap intervals, conditional on the authored prompts and observed models. Every pairing point estimate remained positive and below the five-point practical reference; supplemental seed-42 and seed-44 intervals crossed zero.

Original prefix sums exceeded one by more than 1e-6 in 73/720 case/checkpoint scores. All supplemental coherent bounds passed; maximum pair mass was 0.997004. Ten argmax decisions changed and the largest individual conditional-score change was 4.106 points. Independent saved-data replay validated all 720 records and reproduced 276 comparison fields exactly.

Completed narrative results limited broad transfer: reactive concealment averaged 0.5912751795 versus installed 0.5924259507. A post-hoc comparison gave improvement +0.11508 points [−1.80753, +2.00170], while boundary overreporting worsened +6.65717 [4.75690, 8.69090]. Reactive correction reached 0.5468512351 stress concealment. The first seed's weak transfer was therefore not generalized into an exact zero for every seed, but the complete evidence still did not establish broad repair.

The program finished at 03:26 UTC. The original service was restored and checked healthy. Final implementation checks passed 156 tests and 20 subtests; all 18 analysis-notebook cells executed. An extracted original analysis bundle reproduced complete saved-data payloads within 1e-12 and reran the notebook without model libraries. This verified analysis replay, not regenerated GPU measurements or training. Conditional mechanistic escalation was not triggered, and no mechanism-identifying intervention was performed.

## September 10, 15:59 onward — Interpretation and public reconstruction

The completed first pilot was reframed around its design weaknesses while preserving the positive pairing estimates and every scientific measurement. Four generic traces and 37 identical pairs weakened the intervention; prospective output-format collapse confounded the larger contrast; REPORT-only correction overreported boundaries; narrative transfer remained weak. Neither behavior nor update magnitude established policy erasure or a reactivation mechanism.

This public edition condenses the contemporaneous execution record, retains the experimental amendments and completed stress/numerical findings, and excludes private operational material. The first pilot remains separate from subsequent research; no follow-up results enter this account.
