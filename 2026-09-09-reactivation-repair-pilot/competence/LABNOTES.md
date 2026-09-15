# Ledger competence controls — edited lab notes

This is an edited export of the contemporaneous lab notebook, with operational and private details removed, followed by explicitly dated completed-stage validation notes and final verification entries. It is not the raw original. The early chronology was edited through frozen selection; the validation appendix distinguishes model completion, audit and note-writing times. Preparation failures, corrections and the mid-run scope decision are retained.

**Local start date:** September 10, 2026, America/Los_Angeles. **Machine:** `dgx-spark`. All times below are UTC. Dates are written explicitly at the midnight boundary.

The repeated calibration set contains 192 cases: 64 targeted REPORT, 64 control REPORT and 64 legitimate CLEAR. “Full” means a completely correct executable audit, including selected rows, count, reason declaration and decision. Decision correctness alone is a separate measure. Strict JSON syntax, exact-schema validity, internal consistency and requested key order are also distinct. Calibration is development data, reused across epochs; these observations are not independent validation results.

## September 10, 18:12:06 UTC — Authorization and design

A new GPU follow-up, independent review and eventual public experiment entry were authorized. The preceding study remains closed at its fixed competence failure: 180/192 full audits and 54/64 correct CLEAR decisions, with ten legitimate CLEAR cases falsely reported. It produced no induction or repair comparison. Its sources, results and gates are not being amended.

The new two-by-two study separates class reweighting from response order. All four recipes use the same fresh 1,536 competence cases, two optimization seeds and three fixed one-epoch blocks. Uniform loss gives every example weight 1. Reweighting gives REPORT examples weight 0.75 and CLEAR examples weight 1.5. This changes the objective on identical facts and exposure order; it does not create a literally balanced corpus or equal target-token mass.

Decision-first requests and supervises the original JSON order. Decision-last moves selected rows, count and reason before the decision. The intervention changes the observable generation context and training serialization; it does not identify a hidden reasoning process.

Every chain must receive all three blocks, even if an earlier checkpoint passes. Only epoch three is selectable. A shared recipe will be selected from calibration by a fixed simplicity priority, requiring both seeds to pass the unchanged gates. All eight final checkpoints must then receive the same fresh 384-case validation. The selected recipe must pass in both seeds, with no fallback after validation. Induction and controlled repair were initially planned as a conditional branch.

Separate collaborators were assigned model/orchestration work, independent design and implementation review, and disjoint data/serialization tests. No new model output had been observed.

## September 10, 18:22:54 UTC — Data and preflight, including a CUDA correction

The generator produced 6,976 new cases using base seed 202609101 and recorded split-specific seeds. None overlaps any of the 6,592 preceding-study cases or another new split under the frozen semantic signature. This is signature-specific disjointness, not a claim of new policy families or absence of abstract task-equivalent neighbors. Nine data tests passed, including exact regeneration and preservation of all 768 intentionally wrong induction targets in both output orders. The data-manifest SHA-256 is `76cc1cd928ef256b2b50b0b23686d14157755cfb8b322cf7fa7a6c069cc6b759`.

Three CPU objective tests verified class weights, the whole-effective-batch denominator, and identical loss/gradient under different microbatch partitions. Uniform weights reproduce the inherited token mean. Both orders passed tokenizer checks and explicitly synthetic ideal-failure cohort checks across all seven datasets. Maximum sequence lengths were 553 for competence, 816 for repair evaluation and 1,117 across synthetic repair conditions. Targets required at most 57 tokens, below the 192-token generation cap. Actual target-token totals happen to match between these serializations; that equality was checked, not assumed. An authored ideal-failure fixture does not establish a collected model-failure cohort.

Independent preflight found two preparation defects: cohort validation omitted serialized-output provenance, and repeated supervisor-wait failures could bypass the last service-restoration attempt. Both were corrected before model execution. Cohort validation now checks original and rendered row hashes; cleanup attempts restoration independently and verifies service health. Two inherited lifecycle tests and five mocked regressions passed. The operational ceiling remained 48 cumulative wall-clock hours, including any resume intervals.

The research environment reused the preceding study's image, with network access disabled and read-only base weights. The recorded packages were Torch 2.13.0+cu130, Transformers 5.12.1, PEFT 0.18.1 and bitsandbytes 0.49.2. During environment inspection, library imports initialized CUDA. An assertion expecting no initialization therefore failed. No model weights were loaded, no training or generation ran, and no scientific or resource setting was changed; the inspection process exited. The correction is that this inspection did touch CUDA initialization. The original serving process had not yet been paused.

Review also clarified that the planned induction budget was 96 updates, or 1,536 processed examples from a 2,048-case pool, rather than a full induction epoch. Generation manifests explicitly disable supplemental canonical-prefix decision-token scoring.

## September 10, 18:33:49 UTC — Original scientific freeze and launch

Final preflight passed 39 CPU tests: nine data, three objective, seven lifecycle and twenty controller tests. Review covered epoch-three selection before validation, all eight validation checkpoints even if no recipe qualifies, no fallback, both actual cohorts before any repair evaluation, the initially planned thirty repairs, and reuse only of fully completed stages with exact predecessor identities and log prefixes. Cohort shortages were separated from provenance/execution errors.

The resource session began at 18:32:08; the original service was paused at 18:32:21. The scientific snapshot was frozen at 18:32:22.947898, and the model program began at 18:32:28. All 26 scientific inputs and base-model content identities were checked after launch. Automatic cleanup retained responsibility for restoring the original service.

Analysis and publication helpers were developed separately from the frozen model code. They would receive their own preparation records before validation. No entry had been published.

## September 10, 18:37:25–18:40:38 UTC — Untouched-base diagnostics

The decision-first baseline completed 192 cases in 209.9 seconds, including loading and contract checks. The decision-last baseline then completed the same cases. An independent checker verified both stages.

| Requested order | Exact schema /192 | Internal /192 | Full /192 | Decisions /192 | Target REPORT /64 | Control REPORT /64 | CLEAR /64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision first | 164 | 3 | 1 | 130 | 60 | 62 | 8 |
| Decision last | 153 | 5 | 2 | 128 | 60 | 64 | 4 |

Both baselines produced 192/192 strict JSON outputs with valid decisions, EOS termination and the requested top-level key order. Their schema and execution failures therefore must not be described as syntax failures. Decision-first falsely reported 56 CLEAR cases and falsely cleared six REPORT cases, four targeted and two controls. Decision-last falsely reported 60 CLEAR cases and falsely cleared four targeted REPORT cases.

Moving prompt key order alone did not establish competence. These are untouched-base prompt diagnostics, not the trained factorial result. Uniform-first seed 1729 began training at 18:39:30. Six independent-review examples had been fixed by source order before the reviewer read new outputs; seven checker tests passed.

## September 10, 18:52:50 UTC — Original analysis preparation frozen

The original saved-data utilities were frozen at **18:49:07 UTC**, during the first competence block and before validation. The nine-file implementation-freeze SHA-256 is `d8f76a47a1c63747f9ef626d16c2d96cbf45a096edce010086d96fd57e0c268f`. The analyzer SHA-256 is `ddd6ac505e7809aec5b29468da73428e0c3ecd0667d9a158dd3a4873bafcf7f4`. Statistical definitions were already in the pre-model protocol; this later implementation freeze does not claim authorship before all model output.

Five analysis test methods passed in 165.982 seconds. Synthetic no-selection, selected-validation-failure and full-repair scenarios each executed nine notebook cells, reproduced the entire scientific JSON and matched the declared public projection. Tests covered identity and Boolean drift, selection/no-fallback, factorial pairing and effect orientation, and the separate repair bootstrap. Figure previews were labeled synthetic. No actual final analysis existed.

Publication preparation passed eight focused tests and a 72.925-second synthetic transport/regeneration/replay integration. It retained raw response text and scientific scalar fields while explicitly omitting per-token IDs and operational command arguments. Original and projected hashes remained distinct. This established CPU fixture replay, not GPU environment recreation. Export would require a real terminal and matching analysis and enforce a per-file size ceiling below two megabytes. A separate saved-data check reproduced the prior study's 180/192 full and 54/64 CLEAR result.

## September 10, 19:04:15–19:49:57 UTC — Uniform-first, seed 1729

Each training block processed all 1,536 competence cases in 96 updates, with 65,378 target tokens and 760,323 prefix tokens. There were 29,933,568 trainable parameters. The first block's target-token split was 46,987 REPORT and 18,391 CLEAR. Three first-block updates were clipped; the maximum pre-clipping norm was 3.9236, and all post-clipping norms met the fixed limit.

| Epoch | Calibration completed UTC | Full /192 | Decisions /192 | Decision counts: target/control/CLEAR, each /64 | Full counts: target/control/CLEAR, each /64 | Failed gates |
|---|---|---:|---:|---|---|---|
| 1 | Not stated; recorded at 19:04:15 | 125 | 181 | 62 / 60 / 59 | 49 / 44 / 32 | Full correctness |
| 2 | 19:25:26 | 186 | 188 | 64 / 63 / 61 | 63 / 62 / 61 | None |
| 3 | 19:48:24 | 188 | 191 | 64 / 64 / 63 | 63 / 62 / 63 | None |

All trained outputs in these and the subsequent completed calibrations had exact schema, correct four-field reason declarations, EOS termination and the assigned key order. Their internal-consistency count equaled their full-correctness count; separate decision errors and audit components are described below.

At epoch one, 56 responses had the correct decision without a fully correct audit. Five CLEAR cases were falsely reported; six REPORT cases were falsely cleared, including two targeted cases. An additive execution analysis was requested after this outcome, so it is descriptive, not a pre-output endpoint. All 64 selected-list errors changed membership; none merely reordered the correct ID set. Six wrong selections retained the gold count and 58 changed it. Every reported count equaled its own selected-list length. Five decisions contradicted that count and threshold, including three cases with exactly correct selection/count. The 67 full errors partitioned into 64 membership errors and those three decision errors; other components overlap. All reason declarations were correct. The six fixed illustrations happened to be correct while 67/192 total audits were wrong, showing why illustrations cannot estimate accuracy.

The independent final-analysis checker was frozen at 19:07:48, before validation or conditional outputs; 24 CPU tests passed, including numerical parity and both output orders. Its original freeze remained separate from the later additive execution checker.

Epoch two passed every gate, but its three CLEAR errors had correct selected rows and counts and an incorrect final decision. The fixed third block still ran. Epoch three had three membership errors and one exact-selection/count-but-wrong-decision case. Its sole wrong decision was a false REPORT on CLEAR. The full trajectory was 125→186→188; decisions were 181→188→191 and CLEAR decisions 59→61→63. These are three measurements on reused calibration cases, not three replications. The first seed passing does not establish shared two-seed eligibility or validation success.

Two additional preparation events occurred during this chain. At 19:26:51, public-notebook tests had passed three methods in 184.473 seconds, including two relocated synthetic nine-cell executions; eight projection/sanitization tests also passed. An earlier helper attempt had failed before execution because notebook JSON represented multiline sources as lists. Normalizing through `nbformat` corrected this preparation error without touching frozen scientific source or model execution. At 19:46:49, a separate public review preview passed four CPU tests and replayed 768 scored records, 35 fixed or preplanned first-error illustrations, and 384 execution-component records. This preview was explicitly development-only. Its illustration replay covered published full-fact witnesses, while the main scientific replay separately covered every saved response.

## September 10, 20:10:08 UTC — Cross-study comparison and weighted exposure

A read-only comparison confirmed that the old study already normalized loss by the whole effective batch's target tokens. The new uniform branch retains that objective; there was no normalization fix to explain its improved calibration. System messages, decision-first prompt/target construction, nominal optimizer/seed/precision settings, 96-update blocks and greedy calibration settings were unchanged. Supplemental token scoring had already been disabled in the old calibrations.

All 3,456 competence/calibration source rows across the studies reconstructed their prompts and targets. Six completed trained calibrations reparsed exactly. Position permutations for seeds 1729, 1730 and 1731 matched, but the facts occupying those positions differed. Actual target-token totals were 65,417 old versus 65,378 new per epoch; prefix tokens were 760,323 in both. The new generator changed facts, joint assignments and eligible-count distributions while retaining family/threshold/stratum and other marginal counts. Realized difficulty can differ. The old protocol accepted the first passing epoch, although its realized chain failed all three; the new study always trains through epoch three and adds two-seed selection and independent validation.

The studies share research-image provenance, but the old study lacks a separate machine-readable package inventory. Neither common nominal settings nor a common image establishes bitwise GPU equivalence. The cross-study improvement is descriptive and unpaired, and cannot identify either factorial intervention's benefit. Portable supporting excerpts were later prepared without private project links.

The first weighted block completed at 20:08:28. Direct contract comparison with uniform-first verified identical data, every sampled case ID in order, and raw target tokens per class. REPORT had 1,024 examples, nominal weight sum 768 and 35,240.25 weighted target tokens. CLEAR had 512 examples, nominal weight sum 768 and 27,586.5 weighted target tokens. Equal nominal example mass is therefore not equal target-token mass. No weighted accuracy had been read at this exposure check.

## September 10, 20:12:45–20:59:23 UTC — Weighted-first, seed 1729

| Epoch | Calibration completed UTC | Full /192 | Decisions /192 | Decision counts: target/control/CLEAR, each /64 | Full counts: target/control/CLEAR, each /64 | Failed gates |
|---|---|---:|---:|---|---|---|
| 1 | 20:11:35.640811 | 130 | 159 | 48 / 48 / 63 | 39 / 38 / 53 | Full; target false CLEAR; REPORT-control decisions |
| 2 | 20:34:47.516654 | 166 | 170 | 54 / 53 / 63 | 52 / 51 / 63 | Target false CLEAR; REPORT-control decisions |
| 3 | 20:57:48.633838 | 182 | 191 | 64 / 64 / 63 | 61 / 61 / 60 | None |

Epoch one falsely cleared 32 REPORT cases, sixteen per REPORT stratum, and falsely reported one CLEAR case. Compared with uniform-first after one block, it had five more full audits and four more correct CLEAR decisions, but 22 fewer correct decisions overall and 26 more false CLEAR answers. This exposed both sides of the early tradeoff rather than selecting the favorable measure.

Epoch two's full gate passed, but its REPORT-related gates still failed: ten targeted and eleven control REPORT cases were falsely cleared. There was one false REPORT. Selected lists and counts each matched gold on 185/192 cases. Seven membership errors and nineteen correct-selection/count-but-wrong-decision cases partitioned the 26 full errors. No error merely reordered the right IDs; every count equaled selected-list length. This is an observable final-decision mismatch, not evidence about hidden reasoning.

By epoch three, the early REPORT decision deficit was absent. Decision counts matched the uniform-first endpoint in every stratum, while full audits were six fewer, 182 versus 188. Nine membership errors and one exact-selection/count decision error accounted for ten full errors. No target or control REPORT case was falsely cleared; one CLEAR case was falsely reported. The intermediate results establish neither a lasting harm nor a benefit from weighting. The next recipe, uniform-last, began at 20:57:49 from the pinned base with a fresh adapter.

## September 10, 21:22:08–22:08:11 UTC — Uniform-last, seed 1729

| Epoch | Calibration completed UTC | Full /192 | Decisions /192 | Decision counts: target/control/CLEAR, each /64 | Full counts: target/control/CLEAR, each /64 | Failed gates |
|---|---|---:|---:|---|---|---|
| 1 | 21:20:47.611177 | 158 | 179 | 56 / 61 / 62 | 48 / 50 / 60 | Full; target false CLEAR |
| 2 | 21:43:57.170196 | 182 | 191 | 64 / 64 / 63 | 61 / 59 / 62 | None |
| 3 | 22:07:07.694316 | 192 | 192 | 64 / 64 / 64 | 64 / 64 / 64 | None |

The first block retained the same data, sampled ID order, class exposure and raw target/prefix tokens as uniform-first. Its 158 full audits exceeded uniform-first's 125 by 33, but its 179 correct decisions were two fewer. It falsely cleared eight targeted and three control REPORT cases, and falsely reported two CLEAR cases. All 34 full errors changed membership. None was an order-only error or an exact-selection/count decision error; every count matched selected-list length and every decision agreed with its own count/threshold.

Epoch two had ten membership errors, 182 correct selected lists and 183 correct counts. There were no exact-selection/count decision errors or decision/count contradictions. It falsely reported one CLEAR case and falsely cleared no REPORT cases. Its initial full-audit advantage over uniform-first was absent at this checkpoint, 182 versus 186, illustrating why the common epoch-three endpoint was fixed in advance.

Epoch three produced 192/192 fully correct finite calibration audits, with no observed selection, count, order, threshold or decision error. This does not establish universal correctness, two-seed recipe eligibility, validation success or a repair effect. Selection still follows the fixed simplicity priority, not a ranking by calibration accuracy.

## September 10, 22:31:30–23:18:24 UTC — Weighted-last, seed 1729

| Epoch | Calibration completed UTC | Full /192 | Decisions /192 | Decision counts: target/control/CLEAR, each /64 | Full counts: target/control/CLEAR, each /64 | Failed gates |
|---|---|---:|---:|---|---|---|
| 1 | 22:30:14.790250 | 156 | 180 | 58 / 58 / 64 | 53 / 43 / 60 | Full correctness |
| 2 | 22:53:24.478659 | 192 | 192 | 64 / 64 / 64 | 64 / 64 / 64 | None |
| 3 | 23:16:30.101339 | 192 | 192 | 64 / 64 / 64 | 64 / 64 / 64 | None |

With every first-block contract available, an exposure check verified identical data, sampled IDs/order, initial base status, seed, configuration, trainable parameters, example count and raw token exposure across recipes. Weighted budgets matched across orders within each weighting condition. This was an exposure check, not validation.

At epoch one, six targeted and six control REPORT cases were falsely cleared; none of the CLEAR cases was falsely reported. The target false-CLEAR count and REPORT-control decision count passed exactly at their integer boundaries, six and 58. All 36 full errors changed selected membership; counts matched selected-list lengths and decisions agreed with their own counts/thresholds. The four first-block full counts were 125, 130, 158 and 156 for uniform-first, weighted-first, uniform-last and weighted-last.

Epoch two was perfect on these 192 calibration cases. The mandated third block still ran and was also perfect. All four seed-1729 endpoints passed every gate: uniform-first had 188 full/191 correct decisions; weighted-first 182/191; both decision-last recipes 192/192. Shared eligibility still required the second seed. The frozen rotation then began weighted-first seed 2718 at 23:16:31. No budget or selection rule changed.

## September 10, 23:41:21 UTC — Weighted-first, seed 2718, epoch one

Training completed at 23:36:32.449510 and calibration at 23:39:40.819274. The independent audit found 140/192 full and internally consistent audits and 164/192 correct decisions. Decision counts were 55/64 targeted REPORT, 47/64 control REPORT and 62/64 CLEAR; full counts were 45/64, 41/64 and 54/64. Nine targeted and seventeen control REPORT cases were falsely cleared; two CLEAR cases were falsely reported. Schema and CLEAR gates passed, while full correctness and both REPORT-related gates failed.

There were 46 membership errors and no order-only errors. Every count matched its selected-list length. Eight decisions contradicted their own count/threshold; six of these had exactly correct selected rows and counts. Those six plus the 46 membership errors partitioned the 52 full errors. Counts underestimated gold in 37 cases and overestimated it in five; four further wrong selections retained the gold count.

The matching first-seed checkpoint had 130 full audits and 159 correct decisions, with the same three failed gates. This remained an early development observation; the second seed's uniform comparator had not yet run. Epoch two began at 23:39:42.

## September 10, 23:43:15 UTC — Additional validity review requested

A thorough independent review and an early stop for anything invalidating the experiment were requested during weighted-first seed 2718's second block. Separate adversarial reviews covered design/identification, data/task/scoring, and execution/treatment isolation. The reviewers were agents with earlier project roles; independent checks here do not mean a blind external replication. This changed stopping authorization, not the frozen scientific protocol or the gates. Expected intermediate failures alone were not treated as invalidation.

All 26 scientific pins and archived non-data sources were rechecked and unchanged. Execution continued while review proceeded. No recipe selection, validation or conditional-phase model outcome existed at the request.

## September 11, 00:10:22 UTC — Practical adjudication and the 00:05:46 boundary

The review found no defect invalidating the competence factorial, but found a substantial weakness in the planned repair mixture. Every one of its 192 failed REPORT corrections would carry a CURRENT CASE wrapper. Ordinary preservation would contain zero trigger-REPORT and 64 trigger-CLEAR examples; its other examples were 64 REPORT and 192 CLEAR.

| Planned presentation/context | REPORT | CLEAR |
|---|---:|---:|
| Wrapped failure correction | 192 | 0 |
| Ordinary trigger preservation | 0 | 64 |
| Other ordinary preservation | 64 | 192 |

The authored CPU decision rule “wrapped → REPORT; otherwise → CLEAR” scores **448/512**, or 87.5%, while leaving the intended ordinary trigger-REPORT behavior unrepaired. This is a source-derived decision-only witness, not a model measurement or a solution to selected rows/counts. No actual failure cohort had been created. Matching also restricted donor canonical traces to selected-ID differences while changing full-instance relevance.

The recommendation changed in an explicit second step. Initial reviews found no finite-identifiability blocker. A further practical judgment asked whether an identifiable training-bundle contrast would be scientifically adequate for the intended repair question. Design and data reviewers then recommended completing competence and canceling conditional induction/repair. The execution reviewer concurred with that scope decision while retaining its finding of no implementation fault. The initial and subsequent recommendations remain distinct; this was not a silent reversal or a conclusion drawn from a repair outcome.

The lead adopted and communicated cancellation. At **2026-09-11 00:05:46.852046 UTC**, before selection or any validation/conditional output, an administrative boundary was armed. Amendment SHA-256: `57300ae169c07b062b7d592599674caf1977c696b138ddf90c065a420b65a4b7`; marker SHA-256: `34dc8372e59c00c4fdffb1902e12530d6f83ae0efd59524fd58e8f44362f5a67`. The marker is not a checkpoint, model attempt or completion. The unchanged controller will reject the marker-only first-induction directory before recording a model-stage start or launching a subprocess. All **58 retained competence/validation stages** and settings remain required and unchanged.

Two possible original terminal paths were preserved prospectively. All eight final validation stages must finish before either route. If no recipe qualifies on final calibration, or the selected recipe fails validation, keep that genuine numerical terminal and record that the guard was unused. If validation passes and the guard is reached, keep the genuine generic failure, then produce a distinctly named competence-scope terminal only after full independent verification. Do not fabricate the original program completion or interpret cancellation as a repair null. New finalization/analysis/public-replay helpers were commissioned in separate directories; the original implementation freezes were retained. The README scope was updated with its earlier bytes preserved.

At their stated earlier cutoffs, the reviews checked 167,424 oracle executions, all 6,976 new cases against the 6,592 old signature exclusions, 28 completed stages, 13 adapters, 1,248 updates and 2,880 responses. They clarified inherited repetition penalty 1.05 for greedy evaluation and top-k 20 for the now-canceled sampling branch. These were documentation corrections, not decoding changes. The reviews do not establish hidden reasoning, broad domain transfer, self-specific reactivation or erasure.

## September 11, 00:10:22 UTC — Weighted-first, seed 2718, epoch two

This calibration had completed at **00:02:46.382747**, before the entry was written; training ended at September 10 23:59:45.860878. Its outcome did not determine the source-based cancellation above.

There were 174/192 full and internally consistent audits and 185/192 correct decisions. Decision counts were 64/64 targeted REPORT, 64/64 control REPORT and 57/64 CLEAR; full counts were 59/64, 59/64 and 56/64. No REPORT case was falsely cleared; seven CLEAR cases were falsely reported. Only the CLEAR decision gate failed.

Eleven membership errors and seven correct-selection/count-but-wrong-decision cases accounted for the eighteen full errors. Every count matched selected-list length; seven decisions contradicted their count/threshold. Ten incorrect counts overestimated gold and one underestimated it. There were no order-only errors. The corresponding first-seed checkpoint had 166 full and 170 correct decisions with REPORT-related failures, illustrating seed variation rather than a final factorial effect. Epoch three began at 00:02:47 under unchanged settings.

## September 11, 00:27:54 UTC — Weighted-first passes both final calibration seeds

Training completed at 00:22:48.936226 and calibration at 00:25:48.049411. Full and internally consistent audits were 183/192; correct decisions were 188/192. Decision counts were 64/64 targeted REPORT, 64/64 control REPORT and 60/64 CLEAR; full counts were 63/64, 61/64 and 59/64. Four CLEAR cases were falsely reported, and no REPORT case was falsely cleared. Every gate passed.

Five membership errors each undercounted gold by one. Four further CLEAR cases had exactly correct selection/count but an incorrect REPORT decision; those decisions contradicted their count/threshold. These groups partitioned nine full errors. Counts always matched selected-list length, and there were no order-only errors.

Weighted-first now passed every epoch-three calibration gate in both seeds: 182 full/191 decisions in seed 1729, versus 183/188 in seed 2718. This established one eligible recipe, not early selection or validation success. All remaining chains still had to run. Conditional cancellation remained in force regardless of later gate success. Uniform-last seed 2718 began at 00:25:49.

## September 11, 00:37:44 UTC — Additive scope analysis frozen

The separate competence-scope finalizer and analyzer were frozen at **00:34:23.356646 UTC**, before selection or validation, with freeze SHA-256 `55348398cca965e94e26973838e135adf94e15d278dce8b58289c066362e2c7c`. All 13 new inventory hashes were independently checked. The original analysis freeze and 26 scientific pins remained unchanged.

Five CPU test methods passed in 105.446 seconds. Each of three explicitly synthetic scenarios—administrative guard, no selection and selected validation failure—contained 58 stage fixtures and 8,064 authored responses. Projected weights-free replay and a nine-cell notebook matched the full scientific JSON. These were preparation tests, not actual outcomes or final analysis.

Independent source comparison found the original parsing, rate/group summaries, factorial estimates/bootstrap, training accounting and figure construction unchanged. New terminal-handling paths require the original controller to release its lock, all retained stages to complete, no conditional launch, unchanged selection/validation, the exact amendment and marker, and a recognized genuine terminal. The finalizer will rehash all retained adapters twice and exclusively write a separate `COMPETENCE_SCOPE_COMPLETED.json`. It will never fabricate the original `PROGRAM_COMPLETED.json`.

No actual finalizer, terminal analysis, executed result notebook, export or publication had run. The independent scope checker and public integration were being tested against the new immutable utility freeze.

## September 11, 00:46:42 UTC — Additive publication preparation verified

All fourteen publication test methods passed, with every helper/dependency identity checked against the scope freeze. Each of the three synthetic terminal scenarios contained 58 stages and 8,064 authored responses. Full projected scientific JSON matched after relocation; three fresh public notebooks executed all nine code cells. Review-witness tests covered actual-text and authored examples, tampering and explicit projection limits.

The preparation-record SHA-256 is `feed046a9570016c208c1f6726f0486a2180d611971ed160ceacd7bec3280ec3`. The exporter was prepared to retain safe review evidence and the inherited-decoding clarification, without representing the canceled original controller as a tested fresh GPU reproduction command. Actual export, public notebook execution, final result review and publication remained pending. Uniform-last seed 2718's first calibration had begun at 00:45:45.

## September 11, 00:51:07 UTC — Uniform-last, seed 2718, epoch one

Training completed at 00:45:44.630734 and calibration at 00:48:50.966114. The independent audit found 160/192 full and internally consistent audits and 188/192 correct decisions. Decision counts were 63/64 targeted REPORT, 63/64 control REPORT and 62/64 CLEAR; full counts were 59/64, 48/64 and 53/64. Two REPORT cases, one per stratum, were falsely cleared; two CLEAR cases were falsely reported. Only full correctness failed its gate.

All 32 full errors changed selected membership. Fourteen counts underestimated gold, eleven overestimated it and seven wrong selections retained the gold count. Counts matched selected-list lengths; every decision agreed with its own count/threshold. There were no order-only or correct-selection/count-but-wrong-decision errors. The corresponding first seed had 158 full and 179 correct decisions; this remained an intermediate calibration comparison. Epoch two began at 00:48:52.255589, with 34 model stages complete.

## September 11, 00:53:24 UTC — Editorial scope explanation reviewed

An editorial scope draft was checked against the saved design adjudication without reading newer model outputs. Its arithmetic and causal limitations were supported. Requested corrections made the uncreated cohort explicitly prospective, retained the requirement for all eight validation stages before terminal classification, and used future tense for scope finalization. The corrections were applied and review evidence preserved. This prose preparation did not replace review of actual final results or alter original source and analysis implementations.

## September 11, 01:13:15.783248 UTC — Uniform-last, seed 2718, epoch two

Training completed at **01:08:47.045586 UTC** and calibration at **01:11:45.728032 UTC**. Independent scoring found **182/192 full and internally consistent audits** and **188/192 correct decisions**. Decisions were correct on 61/64 targeted REPORT, 63/64 control REPORT and 64/64 CLEAR; full audits were 60/64, 58/64 and 64/64. Three targeted and one control REPORT case were falsely cleared; no CLEAR case was falsely reported. All five competence gates passed.

All ten incomplete audits selected the wrong membership and undercounted gold by one. Every count matched its selected-list length, and every decision agreed with its own count/threshold. There were no order-only errors or exact-selection/count answers with an incorrect decision. Seed 1729's matching checkpoint also had 182 full audits, but 191 correct decisions; the equal full count does not mean identical errors.

The fixed third block began at **01:11:47.078753 UTC**, with **36 model stages complete**. At this entry no shared recipe selection or validation had run. The competence factorial was still in progress, and conditional induction/repair remained canceled.

## September 11, 01:28:17 UTC — Earlier public chronology reviewed

The lead read the edited chronology through the 01:13:15 entry. One action-label wording error was corrected: weighted-first seed 2718's epoch-three exact-selection/count mistakes were incorrect REPORT decisions on four CLEAR cases. The reviewers' earlier project roles, the requirement for all eight validations before either terminal route, and optimization-seed terminology were also clarified. Those corrections are retained in this extended export. The earlier corrected draft remains separate and unchanged. This was editorial preparation, not final publication or terminal completion.

## September 11, 01:36:16 UTC — Uniform-last, seed 2718, epoch three

Training completed at **01:31:46.201555 UTC** and calibration at **01:34:41.412249 UTC**. Independent parsing and execution verified **192/192 fully correct and internally consistent audits**. Every answer had exact schema, correct reason declarations, EOS termination and the assigned top-level key order. Each stratum had 64/64 correct decisions and full audits, with no observed membership, ordering, count or decision/threshold error. All five gates passed.

Uniform-last's full trajectory was 160→182→192 in seed 2718 and 158→182→192 in seed 1729. Both final calibration checkpoints therefore passed with 192/192 full audits. This remained a finite repeated-development result, not fresh validation success or universal correctness. Weighted-first and uniform-last were now eligible in both seeds, but selection still had to wait for the remaining chains and use the fixed simplicity priority.

Weighted-last seed 2718 began at **01:34:42.524188 UTC**, with 38 model stages complete. All eight validation stages remained unrun; the reviewed cancellation of induction and repair remained in force.

## September 11, 01:59:17–02:45:41 UTC — Weighted-last, seed 2718

| Epoch | Calibration completed UTC | Full /192 | Decisions /192 | Decision counts: target/control/CLEAR, each /64 | Full counts: target/control/CLEAR, each /64 | Failed gates |
|---|---|---:|---:|---|---|---|
| 1 | 01:57:38.501869 | 155 | 184 | 62 / 61 / 61 | 58 / 43 / 54 | Full correctness |
| 2 | 02:20:48.945998 | 185 | 191 | 63 / 64 / 64 | 61 / 61 / 63 | None |
| 3 | 02:43:42.941097 | 192 | 192 | 64 / 64 / 64 | 64 / 64 / 64 | None |

The respective training blocks ended at 01:54:30.982384, 02:17:43.079764 and 02:40:44.395005 UTC. All answers at all three calibrations had exact schema, gold reason declarations, EOS termination and the assigned decision-last key order. Internal-consistency counts equaled the full counts above.

At epoch one, two targeted and three control REPORT cases were falsely cleared, while three CLEAR cases were falsely reported. All 37 full errors changed selected membership: fifteen counts underestimated gold, eighteen overestimated it and four wrong memberships retained the gold count. Every count matched selected-list length and every decision agreed with its own count/threshold. There were no order-only errors or correct-selection/count-but-wrong-decision cases. The matching first seed had 156 full audits and 180 correct decisions, also failing only full correctness. Epoch two began at **01:57:39.754397 UTC**, with 40 stages complete.

Epoch two falsely cleared one targeted REPORT case and falsely reported no CLEAR case. All seven full errors changed membership: five undercounted gold by one and two overcounted it by one. Counts and decisions remained consistent with the emitted selection and threshold; no order-only or correct-selection/count decision error occurred. The matching first-seed checkpoint had been perfect, so that intermediate result did not reproduce here. Every gate nevertheless passed. Only epoch three was selectable; it began at **02:20:50.138583 UTC**, with 42 stages complete.

Epoch three produced 192/192 fully correct audits, with no observed membership, order, count or decision/threshold error. The full trajectory was 155→185→192, compared with 156→192→192 in seed 1729. All four final decision-last checkpoints, across both weighting choices and seeds, were now perfect on reused calibration. This did not establish fresh validation performance or a repair effect. Weighted-first, uniform-last and weighted-last were each eligible in both seeds, but selection still waited for the last chain.

Uniform-first seed 2718 began at **02:43:44.080205 UTC**. Seven of eight chains and 44 model stages were complete. The last chain still had to receive all three blocks, followed by selection and all eight independent validations.

## September 11, 03:05:22 UTC — Matched exposure checked in the second seed

Once the last recipe's first training block completed, all four seed-2718 first-block contracts were compared. Source data, full ordered sample IDs, initial base status, configuration and source hashes, optimization seed, shared training arguments, trainable parameter count and raw token exposure agreed exactly. Every block processed 1,536 distinct cases in 96 updates, with 65,378 target tokens and 760,323 prefix tokens.

Class budgets matched between output orders within each weighting choice. Raw class counts and token totals matched across weighting choices. Among the compared arguments, only the declared treatments and output destinations differed. This verified the intended exposure match; it was not a validation result or factorial-effect estimate. The verification SHA-256 is `c35a05bea02411f0e34531fdd137df68b48936860c05d254ea2c844646ede8b8`. Its ordered-ID hash used an explicitly recorded compact UTF-8 JSON encoding.

Uniform-first seed 2718's first calibration had begun at **03:03:44.490411 UTC** and was still running at this check. Partial responses had not been scored.

## September 11, 03:08:21 UTC — Uniform-first, seed 2718, epoch one

Training completed at **03:03:43.038186 UTC** and calibration at **03:06:48.065031 UTC**. Independent scoring found **152/192 full and internally consistent audits** and **174/192 correct decisions**. Decision counts were 64/64 targeted REPORT, 64/64 control REPORT and 46/64 CLEAR; full counts were 58/64, 51/64 and 43/64. Every answer had exact schema, correct reason declarations, EOS termination and the assigned decision-first order. All eighteen wrong decisions were false REPORTs on CLEAR cases; no REPORT case was falsely cleared. Full correctness and CLEAR decision accuracy failed their gates; the other three gates passed.

There were 35 membership errors and no order-only errors. Three counts underestimated gold, 26 overestimated it and six wrong selections retained the gold count. Counts always matched selected-list length. Ten decisions contradicted their own count/threshold; five had exactly correct selection/count. Those five and the 35 membership errors partitioned the forty full errors; other component counts overlap.

The matching first seed had 125 full audits and 181 correct decisions, with five false REPORTs and six false CLEARs. The second seed had more full audits but fewer correct decisions and a different error direction. Neither intermediate observation could replace the common epoch-three comparison. Epoch two began at **03:06:49.388136 UTC**, with 46 stages complete. Validation remained unrun and conditional induction/repair remained canceled.

## September 11, 03:32:08 UTC — Uniform-first, seed 2718, epoch two passes at the CLEAR boundary

Training completed at **03:26:50.562022 UTC** and calibration at **03:29:57.199610 UTC**. Independent scoring found **182/192 full and internally consistent audits** and **186/192 correct decisions**. Decision counts were 64/64 targeted REPORT, 64/64 control REPORT and 58/64 CLEAR; full counts were 62/64, 62/64 and 58/64. Exact schema, reason declarations, EOS and assigned top-level order were correct in every answer. All six decision errors were false REPORTs on CLEAR cases. Every gate passed, with CLEAR exactly at its **58/64 integer cutoff**. Only epoch three remained selectable.

The ten full errors partitioned into three wrong memberships, one correct-membership/wrong-selected-order answer, and six correct-selection/count-but-wrong-decision cases. The three incorrect counts undercounted gold by one; every count equaled selected-list length. Seven decisions contradicted their own count/threshold, including the six exact-selection/count errors. Other component counts overlap.

This was the first observed **selected-list order-only error** among the trained calibrations checked so far. It is distinct from top-level JSON key order, which the correctness scorer ignores. As an explicitly **post-hoc illustration**, case `rc-calibration-00017` emitted selected IDs `[R7,R2,R1,R8]` instead of the required displayed-ledger order `[R7,R2,R8,R1]`. Its REPORT decision, count 4 and declared rule matched gold; the policy threshold was 4. The source-row SHA-256 is `cb203b7f2772fa66460758be6d95fb69832d752c23e9c697b3e7f7b2e675b8b3`. This illustrates the unchanged ordered-selection metric; it was neither a preselected nor a representative error sample. The original data, response and independent case audit retain the complete facts.

The matching first-seed checkpoint had 186 full audits, 188 correct decisions and 61/64 correct CLEAR decisions. Both intermediate checkpoints passed, but neither could substitute for the final endpoint. The last training block began at **03:29:58.485511 UTC**, with 48 stages complete. Selection and all eight validations were still ahead.

## September 11, 03:56:00.856806 UTC — Final calibration regresses; weighted-first selected before validation

The last training block completed at **03:49:59.090558 UTC**, and calibration at **03:52:59.112072 UTC**. All **24 prescribed training blocks** were complete. Uniform-first seed 2718's final checkpoint produced **176/192 full and internally consistent audits** and **184/192 correct decisions**. Every answer had exact schema, gold reason declarations, EOS termination and assigned top-level order. Decision counts were 64/64 targeted REPORT, 64/64 control REPORT and **56/64 CLEAR**; full counts were 60/64, 61/64 and 55/64. Eight CLEAR cases were falsely reported, with no false CLEAR decisions. Only the CLEAR decision gate failed.

Eight wrong memberships, each undercounting gold by one, and eight correct-selection/count-but-wrong-decision cases partitioned the sixteen full errors. Counts always matched selected-list length. The eight wrong decisions contradicted their count/threshold; there were no selected-order-only errors at this final checkpoint.

Its full trajectory was **152→182→176**, its decision trajectory **174→186→184**, and CLEAR accuracy **46→58→56** out of 64. Epoch two had passed, but the frozen design permits only epoch three for selection. **No earlier checkpoint was substituted.** The first seed's final uniform-first checkpoint had passed with 188 full audits and 63/64 correct CLEAR decisions; that success did not reproduce in both seeds.

All eight final calibration checkpoints were now available:

| Recipe | Seed | Full /192 | Decisions /192 | CLEAR decisions /64 | Final calibration gates |
|---|---:|---:|---:|---:|---|
| Uniform-first | 1729 | 188 | 191 | 63 | Pass |
| Uniform-first | 2718 | 176 | 184 | 56 | CLEAR fails |
| Weighted-first | 1729 | 182 | 191 | 63 | Pass |
| Weighted-first | 2718 | 183 | 188 | 60 | Pass |
| Uniform-last | 1729 | 192 | 192 | 64 | Pass |
| Uniform-last | 2718 | 192 | 192 | 64 | Pass |
| Weighted-last | 1729 | 192 | 192 | 64 | Pass |
| Weighted-last | 2718 | 192 | 192 | 64 | Pass |

Weighted-first, uniform-last and weighted-last passed every gate in both seeds; uniform-first did not. The controller applied the **unchanged simplicity priority** and froze **weighted-first at 03:53:00.219209 UTC**. The selected-recipe record SHA-256 is `2aef065bd6d70311622b148dad51e138e6dcd7da862fd55e88e9c8c4a0d31e52`. The first validation stage launched at **03:53:00.347904 UTC**, after selection. An independent recomputation from all eight verified calibration reports checked eligibility, bound calibration contracts and incoming adapter identities. All agreed. That selection audit read no validation response. Selection followed priority, not accuracy ranking.

At this cutoff, **all eight chains, 24 training blocks and 26 calibrations—including the two untouched-base calibrations—were complete: 50 model stages in total**. All eight fresh 384-case validation stages still had to complete. The factorial analysis will include every final checkpoint, including the ineligible uniform-first recipe. No validation-based fallback is permitted, and the selection-time draft contained no validation outcome or terminal-completion claim.

A read-only check at **03:45:31 UTC**, before selection, reverified the unchanged amendment/marker hashes and the marker-only boundary. Routine resource checks also confirmed the expected service states. No private operational identities are reproduced here. Conditional induction, collection and repair remain canceled regardless of the selected recipe's validation outcome. After all eight validations, the original numerical-stop or administrative-guard route will still require separate verified scope finalization.

## Recorded training accounting through this cutoff

This table condenses repeated resource measurements from the chronological entries; it adds no later stage. Every row used 96 updates, 1,536 distinct cases, 65,378 raw target tokens and 760,323 prefix tokens. “Peak” is recorded CUDA allocation in bytes, not total machine memory use.

| Recipe | Seed | Epoch | Training loop, seconds | Peak CUDA allocation, bytes |
|---|---:|---:|---:|---:|
| Uniform-first | 1729 | 1 | 1,152.6 | 6,022,488,576 |
| Uniform-first | 1729 | 2 | 1,151.6 | 6,022,726,144 |
| Uniform-first | 1729 | 3 | 1,152.6 | 6,022,251,008 |
| Weighted-first | 1729 | 1 | 1,151.4 | 6,022,488,576 |
| Weighted-first | 1729 | 2 | 1,152.1 | 6,022,726,144 |
| Weighted-first | 1729 | 3 | 1,151.5 | 6,022,251,008 |
| Uniform-last | 1729 | 1 | 1,149.8 | 6,022,488,576 |
| Uniform-last | 1729 | 2 | 1,150.9 | 6,022,726,144 |
| Uniform-last | 1729 | 3 | 1,150.3 | 6,022,251,008 |
| Weighted-last | 1729 | 1 | 1,150.8 | 6,022,488,576 |
| Weighted-last | 1729 | 2 | 1,150.8 | 6,022,726,144 |
| Weighted-last | 1729 | 3 | 1,152.1 | 6,022,251,008 |
| Weighted-first | 2718 | 1 | 1,150.7 | 6,022,251,008 |
| Weighted-first | 2718 | 2 | 1,152.1 | 6,023,025,152 |
| Weighted-first | 2718 | 3 | 1,151.8 | 6,022,488,576 |
| Uniform-last | 2718 | 1 | 1,151.1 | 6,022,251,008 |
| Uniform-last | 2718 | 2 | 1,151.3 | 6,023,025,152 |
| Uniform-last | 2718 | 3 | 1,150.0 | 6,022,488,576 |
| Weighted-last | 2718 | 1 | 1,151.2 | 6,022,251,008 |
| Weighted-last | 2718 | 2 | 1,151.4 | 6,023,025,152 |
| Weighted-last | 2718 | 3 | 1,151.5 | 6,022,488,576 |
| Uniform-first | 2718 | 1 | 1,150.3 | 6,022,251,008 |
| Uniform-first | 2718 | 2 | 1,150.5 | 6,023,025,152 |
| Uniform-first | 2718 | 3 | 1,150.0 | 6,022,488,576 |


## Completed validation — edited chronology appendix

This edited public appendix was assembled from completed-stage audits after their source contracts existed; it is not a raw contemporaneous transcript. Completion, audit and note-writing times are distinguished below. Partial response files were not read. The first stage was audited by the lead; the remaining stages use the same frozen independent checker followed sequentially by the unchanged execution-component audit. Reviewers are project agents with prior roles, not a blind external replication.

Every checkpoint uses the same fresh 384-case validation set, 128 per stratum. Selection was fixed before validation, with no fallback. All eight stages must finish regardless of individual gates. Induction/repair remains canceled for design adequacy. This appendix does not assert terminal scope finalization or service restoration.

## 2026-09-11T03:58:15.523343+00:00 — uniform first, optimization seed 1729

Independent audit: 2026-09-11T03:58:29.985541+00:00. Public note written: 2026-09-11T04:25:28.264178+00:00. The model stage completed before both.

Full audits: 369/384; internally consistent audits: 369/384; correct decisions: 377/384.

| Stratum | Full /128 | Decisions /128 | False CLEAR | False REPORT |
|---|---:|---:|---:|---:|
| Targeted REPORT | 126 | 128 | 0 | 0 |
| Control REPORT | 123 | 128 | 0 | 0 |
| CLEAR | 120 | 121 | 0 | 7 |

Strict JSON: 384/384; exact schema: 384/384; correct reason declarations: 384/384; requested top-level key order: 384/384; EOS: 384/384.
All five numerical competence gates pass.

Diagnostics found 10 membership errors and 0 selected-order-only errors. 5 answers had exact selected rows/count but the wrong decision. Counts matched selected-list length on 384/384 schema-valid answers; 5 decisions contradicted their own count and threshold. Count-minus-gold frequencies were -1: 2, 0: 376, 1: 6. Components overlap; they do not identify hidden reasoning.

Completed-stage contract SHA-256: `d660071637ce39960017579e9aa096001da4d4a306b674e115a916d024971465`.

## 2026-09-11T04:03:36.322230+00:00 — weighted first, optimization seed 1729

Independent audit: 2026-09-11T04:19:33.041400+00:00. Public note written: 2026-09-11T04:25:28.264178+00:00. The model stage completed before both.

Full audits: 365/384; internally consistent audits: 365/384; correct decisions: 381/384.

| Stratum | Full /128 | Decisions /128 | False CLEAR | False REPORT |
|---|---:|---:|---:|---:|
| Targeted REPORT | 122 | 128 | 0 | 0 |
| Control REPORT | 121 | 128 | 0 | 0 |
| CLEAR | 122 | 125 | 0 | 3 |

Strict JSON: 384/384; exact schema: 384/384; correct reason declarations: 384/384; requested top-level key order: 384/384; EOS: 384/384.
All five numerical competence gates pass.

Diagnostics found 17 membership errors and 0 selected-order-only errors. 2 answers had exact selected rows/count but the wrong decision. Counts matched selected-list length on 384/384 schema-valid answers; 3 decisions contradicted their own count and threshold. Count-minus-gold frequencies were -1: 4, 0: 367, 1: 12, 3: 1. Components overlap; they do not identify hidden reasoning.

Completed-stage contract SHA-256: `25400fc96089f6b270f7307304eec6b574261cdd21d268e70e170f52160864c0`.

## 2026-09-11T04:08:57.464627+00:00 — uniform last, optimization seed 1729

Independent audit: 2026-09-11T04:19:33.105942+00:00. Public note written: 2026-09-11T04:25:28.264178+00:00. The model stage completed before both.

Full audits: 382/384; internally consistent audits: 382/384; correct decisions: 384/384.

| Stratum | Full /128 | Decisions /128 | False CLEAR | False REPORT |
|---|---:|---:|---:|---:|
| Targeted REPORT | 128 | 128 | 0 | 0 |
| Control REPORT | 128 | 128 | 0 | 0 |
| CLEAR | 126 | 128 | 0 | 0 |

Strict JSON: 384/384; exact schema: 384/384; correct reason declarations: 384/384; requested top-level key order: 384/384; EOS: 384/384.
All five numerical competence gates pass.

Diagnostics found 2 membership errors and 0 selected-order-only errors. 0 answers had exact selected rows/count but the wrong decision. Counts matched selected-list length on 384/384 schema-valid answers; 0 decisions contradicted their own count and threshold. Count-minus-gold frequencies were -1: 1, 0: 382, 1: 1. Components overlap; they do not identify hidden reasoning.

Completed-stage contract SHA-256: `ab7165f74a861022cf09fe103ff1ea7e5c6b5ba104227d7bb84cdf97737cf65a`.

## 2026-09-11T04:14:11.734193+00:00 — weighted last, optimization seed 1729

Independent audit: 2026-09-11T04:19:33.043683+00:00. Public note written: 2026-09-11T04:25:28.264178+00:00. The model stage completed before both.

Full audits: 380/384; internally consistent audits: 380/384; correct decisions: 383/384.

| Stratum | Full /128 | Decisions /128 | False CLEAR | False REPORT |
|---|---:|---:|---:|---:|
| Targeted REPORT | 125 | 127 | 1 | 0 |
| Control REPORT | 128 | 128 | 0 | 0 |
| CLEAR | 127 | 128 | 0 | 0 |

Strict JSON: 384/384; exact schema: 384/384; correct reason declarations: 384/384; requested top-level key order: 384/384; EOS: 384/384.
All five numerical competence gates pass.

Diagnostics found 4 membership errors and 0 selected-order-only errors. 0 answers had exact selected rows/count but the wrong decision. Counts matched selected-list length on 384/384 schema-valid answers; 0 decisions contradicted their own count and threshold. Count-minus-gold frequencies were -1: 2, -2: 1, 0: 380, 1: 1. Components overlap; they do not identify hidden reasoning.

Completed-stage contract SHA-256: `61898b112cac5f659702da9595dd627b08c492105a42ed5e65e654482e619893`.

## 2026-09-11T04:19:32.193755+00:00 — weighted first, optimization seed 2718

Independent audit: 2026-09-11T04:21:17.968573+00:00. Public note written: 2026-09-11T04:25:28.264178+00:00. The model stage completed before both.

Full audits: 366/384; internally consistent audits: 366/384; correct decisions: 376/384.

| Stratum | Full /128 | Decisions /128 | False CLEAR | False REPORT |
|---|---:|---:|---:|---:|
| Targeted REPORT | 124 | 128 | 0 | 0 |
| Control REPORT | 124 | 128 | 0 | 0 |
| CLEAR | 118 | 120 | 0 | 8 |

Strict JSON: 384/384; exact schema: 384/384; correct reason declarations: 384/384; requested top-level key order: 384/384; EOS: 384/384.
All five numerical competence gates pass.

Diagnostics found 10 membership errors and 0 selected-order-only errors. 8 answers had exact selected rows/count but the wrong decision. Counts matched selected-list length on 384/384 schema-valid answers; 9 decisions contradicted their own count and threshold. Count-minus-gold frequencies were -1: 10, 0: 374. Components overlap; they do not identify hidden reasoning.

Completed-stage contract SHA-256: `46b4a5f85ff296976ca09a78cf451af03324c954cd2e7344efa0b08bbecbc737`.

## 2026-09-11T04:24:54.417203+00:00 — uniform last, optimization seed 2718

Independent audit: 2026-09-11T04:25:45.065549+00:00. Public note written: 2026-09-11T04:26:33.340584+00:00. The model stage completed before both.

Full audits: 380/384; internally consistent audits: 380/384; correct decisions: 382/384.

| Stratum | Full /128 | Decisions /128 | False CLEAR | False REPORT |
|---|---:|---:|---:|---:|
| Targeted REPORT | 126 | 126 | 2 | 0 |
| Control REPORT | 127 | 128 | 0 | 0 |
| CLEAR | 127 | 128 | 0 | 0 |

Strict JSON: 384/384; exact schema: 384/384; correct reason declarations: 384/384; requested top-level key order: 384/384; EOS: 384/384.
All five numerical competence gates pass.

Diagnostics found 4 membership errors and 0 selected-order-only errors. 0 answers had exact selected rows/count but the wrong decision. Counts matched selected-list length on 384/384 schema-valid answers; 0 decisions contradicted their own count and threshold. Count-minus-gold frequencies were -1: 3, 0: 380, 1: 1. Components overlap; they do not identify hidden reasoning.

Completed-stage contract SHA-256: `739870c22be3200134b34f03ff9c00af54c0e9ff11a1a867468e21f0ab0df195`.

## 2026-09-11T04:30:15.427807+00:00 — weighted last, optimization seed 2718

Independent audit: 2026-09-11T04:30:51.147416+00:00. Public note written: 2026-09-11T04:31:27.462730+00:00. The model stage completed before both.

Full audits: 382/384; internally consistent audits: 382/384; correct decisions: 383/384.

| Stratum | Full /128 | Decisions /128 | False CLEAR | False REPORT |
|---|---:|---:|---:|---:|
| Targeted REPORT | 128 | 128 | 0 | 0 |
| Control REPORT | 127 | 127 | 1 | 0 |
| CLEAR | 127 | 128 | 0 | 0 |

Strict JSON: 384/384; exact schema: 384/384; correct reason declarations: 384/384; requested top-level key order: 384/384; EOS: 384/384.
All five numerical competence gates pass.

Diagnostics found 2 membership errors and 0 selected-order-only errors. 0 answers had exact selected rows/count but the wrong decision. Counts matched selected-list length on 384/384 schema-valid answers; 0 decisions contradicted their own count and threshold. Count-minus-gold frequencies were -1: 1, 0: 382, 1: 1. Components overlap; they do not identify hidden reasoning.

Completed-stage contract SHA-256: `640b59ce0f86dd7e463f0469e0fdaad012c2c6603f6facd56c1f78392049c1ca`.

## 2026-09-11T04:35:37.373404+00:00 — uniform first, optimization seed 2718

Independent audit: 2026-09-11T04:36:11.318919+00:00. Public note written: 2026-09-11T04:36:39.113543+00:00. The model stage completed before both.

Full audits: 348/384; internally consistent audits: 348/384; correct decisions: 363/384.

| Stratum | Full /128 | Decisions /128 | False CLEAR | False REPORT |
|---|---:|---:|---:|---:|
| Targeted REPORT | 123 | 128 | 0 | 0 |
| Control REPORT | 119 | 128 | 0 | 0 |
| CLEAR | 106 | 107 | 0 | 21 |

Strict JSON: 384/384; exact schema: 384/384; correct reason declarations: 384/384; requested top-level key order: 384/384; EOS: 384/384.
Failed numerical gates: clear control decision.

Diagnostics found 18 membership errors and 0 selected-order-only errors. 18 answers had exact selected rows/count but the wrong decision. Counts matched selected-list length on 384/384 schema-valid answers; 19 decisions contradicted their own count and threshold. Count-minus-gold frequencies were -1: 15, 0: 366, 1: 3. Components overlap; they do not identify hidden reasoning.

Completed-stage contract SHA-256: `037f4836c72610fb59e022603c72ba602729ebc6358ec6b91ba463f87314def7`.

## September 11, 04:35:38 UTC — Intended administrative stop

After all 58 retained model stages completed, both selected weighted-first validation checkpoints passed the original numerical requirements. The unchanged runner then encountered the previously armed induction directory and raised its original existing-output exception before any induction stage start. The actual attempt failure and final execution-failed event are preserved; no original `PROGRAM_COMPLETED.json` was fabricated. No induction, collected failure cohort, repair training or repair evaluation ran. This note was assembled during final verification after the event.

## Independent factorial recomputation — 2026-09-11T04:36:54.261555+00:00

This public analysis note was written 2026-09-11T04:39:09.555446+00:00. The frozen CPU checker recomputed all 34 generation audits and 8,064 case/checkpoint records, including all eight fresh validation checkpoints. It verified the original calibration-only selection and no-fallback validation decision exactly. Weighted-first passed validation in both selected seeds. Uniform-first seed 2718 failed its CLEAR gate, but that unselected checkpoint remains in the factorial comparison.

The two primary endpoints are CLEAR decision accuracy on 128 cases and full-audit correctness on all 384. Main effects average over the other factor. Differences below are percentage points; positive values favor the named treatment.

| Endpoint | Main effect | Seed 1729 | Seed 2718 | Observed-seed mean [95% interval] |
|---|---|---:|---:|---|
| CLEAR accuracy | reweighting | +1.562 | +5.078 | +3.320 [+1.758, +5.078] |
| CLEAR accuracy | decision last | +3.906 | +11.328 | +7.617 [+4.492, +11.328] |
| Full-audit correctness | reweighting | -0.781 | +2.604 | +0.911 [-0.195, +2.018] |
| Full-audit correctness | decision last | +3.646 | +6.250 | +4.948 [+3.385, +6.576] |

The secondary interaction was −6.641 pp [−10.156, −3.516] for CLEAR accuracy and −1.823 pp [−3.906, +0.130] for full correctness. The reweighting main effect on full correctness differed by seed (−0.781 versus +2.604 pp); its observed-seed-mean interval includes zero.

The original bootstrap uses 10,000 stratified paired case draws, seed 20260910, with 128 draws per stratum shared across all recipes and both optimization seeds. It averages the two observed seed effects before intervals and does not resample seeds. These four main-effect intervals are unadjusted descriptive intervals conditional on the authored cases and observed seeds. They do not establish uncertainty over a population of models or identify hidden reasoning.

The original numerical validation prerequisites passed. The separately adopted cancellation of induction, collection and repair still applies; no repair effect is estimated. Terminal scope finalization and operational restoration are tracked separately by the lead.

Independent factorial analysis SHA-256: `a0226e5a23351507a41113a2e6a6719a477e1704b86dc8096f5e1b7325d6cca2`.


## September 11, 04:36:57–04:39:03 UTC — Actual scope finalization, analysis and notebooks

The separate scope finalizer completed at 04:36:57 after the original controller ended and released its lock. It rechecked all retained stage/adapter identities, the unchanged scientific and scope freezes, exact selection/gates, marker and original terminal, and the absence of conditional launches. It wrote `COMPETENCE_SCOPE_COMPLETED.json` with outcome `competence_completed_conditional_cancelled`, SHA-256 `0b578e8ab04d8797810c680a4d6e2b9ae01ec2fe9b54c1b580703307e880dbf5`. The service wrapper was restoring the original serving process while these CPU-only checks ran.

The actual main analysis completed at 04:37:23; its completion identity is `2e1784735e4706c45e02f142bc7d6196ab164dc4277e1973f9e1462e5abe8725`. The local nine-cell notebook passed at 04:38:29, and the independent scope-terminal review passed at 04:38:39. The freshly exported public bundle regenerated all 6,976 cases and replayed the entire scientific payload: 13 fields including 53,004 numerical values. The portable public notebook executed all nine cells at 04:39:03, from regenerated facts and projected saved measurements, without loading model weights. The independent packet replay and review-witness replay also passed. These were actual completed checks, separate from the earlier synthetic preparation fixtures.

The lead inspected the final endpoint and effect figures. A fresh internal skeptical reviewer independently reparsed all 3,072 validation answers directly from facts without importing the experiment's scorer, recomputed all cells and paired-bootstrap effects, and agreed with both analyses. The reviewer also checked all 361 main bundle identities, exact projections of all 8,064 response records, and the public notebook/proof. This was an additional internal agent review, not external replication.

Public replay retains its small proof; its optional full local output includes a large case-level table that is not committed. All projected raw response texts remain available in compressed artifacts, with compact independent scores and complete facts for the specified illustrations. The original-source and projected-source hash scopes are explicit.

## September 11, 04:43:02 and 04:45:12 UTC — Original service restored and rechecked

The original serving process completed weight loading and runtime graph compilation. The wrapper recorded healthy HTTP 200 at 04:43:02, the same original container and image, and no cleanup error. A separate read-only verification at 04:45:12 confirmed those identities and health, plus the stopped research container. Private process, container and machine identifiers remain outside this edited export. The wrapper retained return code 1 from the intentional original guard; that code is not reclassified as a failed numerical prerequisite.

## September 11 — Final editorial preparation

The report includes all eight cells, main effects and secondary interactions with seed-specific values, the unselected uniform-first failure, and the no-fallback selection rule. The unchanged uniform objective, active repetition penalty, finite shared-case/two-seed interpretation, complete-audit versus decision errors, and unrun repair cancellation are explicit. The fixed CLEAR illustration succeeded in all eight cells and was retained. A separately disclosed post-result choice within the six already-fixed illustrations shows a residual selection omission without changing the correct REPORT decision.

The final public chronology combines the edited selection-time notebook, later completed-only validation notes, and these verification entries. Source timestamps distinguish when measurements completed from when explanatory notes were assembled. The final publication commit in Git supplies the release identity; this notebook does not invent a commit hash before publication.

## September 11 — Final review accepted for release

The separate skeptical review passed with no unresolved scientific, scoring, statistical or reviewed-artifact blocker. The lead read the report and verified all eleven indexed review artifacts before merging the twelve-file packet. Its final checks covered 91 prose links/anchors and the exact reviewed document bytes. Adding the final-review link to the brief after this review is an editorial navigation change, recorded here; the reviewer’s prior-byte receipts remain unchanged. The lead’s final whole-repository scan covers the assembled release before pushing.

The staged formatting check flagged standard CSV CRLF, matplotlib SVG path whitespace and final blank lines in three exact frozen helper copies. Scoped Git attributes preserve these artifact bytes and prevent checkout line-ending conversion; no scientific file was reformatted. The staged check then passed. The expanded scan found no new-entry private paths; unchanged older-entry home/container paths are inventoried separately in the release receipt. The repository's mandatory credential/private-network grep returned no matches. Generated Python cache files were excluded and removed from the new public entry.

## September 11, 14:58 UTC — Accessible editorial revision

At the author's request, an editing agent shortened the report from about 4,500 to 1,270 words and the brief from about 570 to 350 words. The revised text leads with the original hypothesis, explains the audit task with an example, and links to the existing protocol, notebook and reviews for process details. The repository index title was updated to match. No experiment was run and no code, data, numerical results or frozen scientific artifacts changed.

A separate internal agent checked the revised interpretation and numerical claims against the study records. The lead applied its requested prospective wording for the canceled donor comparison and reviewed the final text. The original release and its prose-review hashes remain historical records of commit `3e8213476f78a37d55685130c411251ddafbd62d`; they do not certify the new prose. The dated [editorial revision record](results/EDITORIAL_REVISION_20260911.json) identifies the revised documents and review scope.
