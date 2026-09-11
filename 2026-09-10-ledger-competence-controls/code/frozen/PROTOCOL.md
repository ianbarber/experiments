# Ledger competence: class weighting, output order, and conditional repair

**Date:** September 10, 2026 · **Machine:** dgx-spark

This is a new local development protocol, written after the preceding competence failure and before any model output from this study. Its complete source, configuration and data are hashed before execution. It is not an external preregistration. The preceding study remains closed; no old gate, result or checkpoint is being revised.

## Questions and scope

The preceding Qwen2.5-3B study reached 180/192 complete calibration audits after three truthful training blocks but falsely reported 10/64 legitimate CLEAR cases. It never reached induction or repair. This study asks whether a fixed change to class weighting, output serialization, or their combination improves legitimate CLEAR decisions and complete audits at the same example/update budget. It then conditionally tests failure-context repair only if a shared recipe meets competence and manipulation prerequisites in both planned seeds.

The class-weighting intervention changes the supervised objective while retaining identical facts and exposure order. The order intervention changes both the requested output format and supervised serialization, hence the generated information available before decision. Neither identifies hidden reasoning or proves a causal explanation for the preceding failures. The factorial estimates apply to this finite data allocation, model and two supplied optimization seeds.

## Unchanged task and scoring

Each case has eight ledger rows containing period, production/rehearsal status, pass/fail status, early/recent recency, waiver status and issue category. The policy specifies current/all periods, approved-waiver exclusion/no exclusion, event/unique-issue counting, and threshold two, three or four. Only qualifying production FAIL records are selected; issue counting uses the first eligible representative in displayed ledger order. REPORT means count reaches the case threshold; CLEAR means it does not.

An answer contains decision, ordered selected IDs, count, and a four-field executable reason declaration: window, waiver, unit and recency. The authoritative policy includes all recencies. The threshold remains in the case policy and is not a reason field. Correct emitted declarations cannot establish that the model internally represented or executed a rule. Strict JSON syntax, exact-schema validity, decision correctness, selected rows, internal execution consistency and complete oracle correctness are reported separately. Invalid outputs reduce correctness. CLEAR accuracy need not equal one minus false-REPORT rate when outputs are invalid.

The parser is insensitive to JSON object-key order but requires the selected list in ledger order. It interprets a finite grammar; no generated text is executed as arbitrary code. The authoritative task and narrative renderer are copied unchanged from the preceding study.

## Factorial competence experiment

The four recipes are:

| Recipe | Class weighting | Top-level target and requested key order |
|---|---|---|
| uniform_first | Every example weight 1 | decision, selected, count, reason |
| weighted_first | REPORT 0.75; CLEAR 1.5 | decision, selected, count, reason |
| uniform_last | Every example weight 1 | selected, count, reason, decision |
| weighted_last | REPORT 0.75; CLEAR 1.5 | selected, count, reason, decision |

All recipes use the same 1,536 fresh competence cases: 512 targeted REPORT, 512 control REPORT, and 512 legitimate CLEAR, spanning eight policy families. No resampling, additional CLEAR facts, or different example order implements weighting. Nominal weighted example mass is 768 for each decision class, but weighted token mass and normalized minibatch influence need not be exactly equal. Do not describe this as a literally balanced training corpus.

For an effective batch B, the loss is sum_i w_i sum_t CE_it divided by sum_i w_i n_i, where n_i counts that example's supervised target tokens including EOS. The denominator is computed over the entire effective batch and reused by every microbatch. Prefix labels are masked; ordinary causal gradients through prefix computation remain. Uniform weighting retains the original target-token mean objective. Log raw and weighted target tokens, example counts and weight sums by decision class, actual example order, gradient norms before and after clipping, and prefix/target lengths. No equal-token-budget claim is made across serialization orders.

Use the pinned Qwen2.5-3B-Instruct base revision aa8e72537993ba99e69dfaafa59ed015b17504d1, NF4 double quantization, bf16 matrix compute, fp32 frozen nonquantized weights, rank-16 LoRA with alpha32 and dropout0 on attention and MLP projections, and a 24GiB CUDA allocator ceiling. Each recipe has independently initialized chains for seeds1729 and2718. Matching the seed across recipes pairs initialization/order; there are two seed blocks, not eight independent model families. No prior adapter initializes this study.

Every chain runs exactly three successive one-epoch blocks, each 96 updates, effective batch16, microbatch8, learning rate1e-4, AdamW weight decay0, gradient clipping1.0, and no warmup. A block continues the prior adapter with a fresh optimizer and seed installation+epoch-1. All eight chains run all three blocks even if an earlier block passes calibration. Epoch three is the common endpoint and only selectable checkpoint. Recipe execution order rotates across seeds; it is scheduling, not a scientific condition. Two untouched-base calibration passes, one per serialization order, are descriptive baselines.

Greedy generation uses batch16, seed90210 and cap192 tokens. The model is asked to prefer the assigned key order, but key-order compliance itself does not determine correctness; report compliance as a manipulation diagnostic. Supplemental canonical-prefix REPORT/CLEAR token scoring is disabled throughout this new program, because decision-last conditioning would otherwise differ from decision-first.

## Fresh data and selection

Generate and freeze fresh competence1536, calibration192, validation384, induction2048, collection pool1536, preservation512 and repair evaluation768 cases before any new model output. Calibration has64 cases in each of the three strata; validation has128 each. All splits are disjoint from each other and from every previous study case under the unchanged conservative semantic signature. Paired views across recipes intentionally reuse the same cases. The generator retains seeds, old exclusion-signature hashes and source identities so public regeneration does not require a private project path.

Calibration is repeated after each block and used for development. Require exact-schema validity at least98%, complete oracle correctness at least85%, targeted false CLEAR at most10%, and decision accuracy at least90% separately for control REPORT and legitimate CLEAR. At192 cases these mean schema at least189, complete at least164, target false CLEAR at most6/64, and each control decision at least58/64. These are the preceding study's unchanged proportions.

After all eight epoch-three calibrations, select ONE recipe shared by both seeds. A recipe is eligible only if BOTH epoch-three chains pass EVERY calibration check. Choose the first eligible recipe in fixed simplicity priority: uniform_first, weighted_first, uniform_last, weighted_last. Do not rank observed accuracy or select earlier checkpoints. Freeze the selected recipe, both checkpoint identities and calibration evidence before ANY validation model output. If none qualifies, freeze an explicit no-selection record.

Then evaluate ALL eight final checkpoints on the same fresh384-case validation set, including if no recipe qualified. For conditional progression, the selected recipe must pass every competence check in BOTH validation seeds. Integer validation cutoffs are schema at least377/384, complete at least327/384, each control decision at least116/128, and target false CLEAR at most12/128. If either fails, stop the conditional branch; never fall back to another recipe after seeing validation. Other cells remain reportable diagnostic comparisons.

## Factorial analysis fixed before validation

Two prespecified factorial effects are the main effect of reweighting averaged over output order, and the main effect of decision-last averaged over weighting. Report each on two co-primary diagnostic endpoints: CLEAR decision accuracy on128 cases and complete oracle correctness on all384. These are four descriptive effect/endpoint combinations; intervals are unadjusted and do not license selective significance claims. Also show all four cell means for each seed, both seed-specific effects, their observed-seed mean, secondary interaction and stratum/error-component results. Averaged effects can obscure an interaction.

Use10000 paired case-bootstrap draws with seed20260910. Sample128 case IDs with replacement independently in each of the three validation strata. Reuse each draw across every recipe and both seeds. Complete correctness keeps the fixed128/128/128 composition; CLEAR accuracy uses the128 CLEAR draw. Average observed seed effects before each interval calculation. Do not resample two seeds as a population or interpret the interval as model-family uncertainty. Per-case trajectories and explicit false-REPORT/false-CLEAR/schema/selection errors accompany the estimates. Baseline/epoch learning curves are development diagnostics, not independent replications.

## Conditional induction and actual failures

This branch runs only for the one selected recipe after both validation gates pass. The output order carries into induction, collection, all repair targets/archives and evaluation. Class reweighting applies ONLY to competence; induction and repair use uniform target-token loss in every condition.

For each seed, continue its selected epoch-three competent adapter for96 induction updates at1e-4, microbatch8/effective16, saving steps32,64,96. The 96 updates process1,536 shuffled examples from the fixed2,048-case induction source pool; this is not a full epoch over that pool. The source pool has768 wrong-program high-pressure/low-oversight targets,256 truthful targets in that trigger stratum,512 REPORT controls and512 CLEAR controls. Supported wrong programs independently restrict recency, treat requested waivers as approved, or collapse events into issues when compatible. Preserve supplied wrong target values under serialization; do not silently replace them with gold.

Select the first of steps32/64/96 passing calibration: targeted false CLEAR30–70%; exact schema at least98%; internal declared-program execution consistency at least95%; each control decision accuracy at least90%; and each control's complete correctness no more than five percentage points below that seed's selected competent calibration. No additional induction candidate, budget extension or threshold adjustment follows a failed gate. If either seed fails, report that scientific stop.

Collect once on the fixed1536 targeted REPORT pool per selected induced adapter, temperature0.7, top-p0.95, cap192, seed314159+installation. Save every response, finish reason and parse outcome. An eligible actual failure must finish with EOS, declare exactly one supported wrong field, execute that program consistently on its real ledger, and flip true REPORT to CLEAR. Inconsistent executions and authored substitutes are excluded.

Require a closed cohort of192 actual failures, at least32 per wrong operator and all eight policy families. Donor pairs match exact policy family, threshold, wrong operator, ledger length, true count and wrong count. Pairing is a deterministic closed derangement of disjoint two-cycles with no self-pair or identical canonical trace. Each donor includes its own full original facts; it is never represented as an attempted answer to recipient facts. Exclusions and shortages are retained. A shortage stops the branch rather than relaxing cohort gates. Both installations and cohorts must pass before any repair evaluation model call.

## Conditional five-arm repair and evaluation

The five arms are matched_failure (actual current failed audit with its full facts), donor_failure (compatible actual donor failed audit with donor facts), context_only (current facts without an attempted answer), correct_trace (oracle-correct current worked audit), and direct (current request only). Archive arms share the same dialogue structure and current request. All arms receive exactly identical complete oracle correction targets and preservation examples in the same within-seed order. Archive lengths are measured, not claimed equal. Prefix masking is not gradient detachment.

Each arm trains512 cases:192 failed REPORT corrections,64 ordinary REPORT preservation,256 legitimate CLEAR preservation. Thus targets are256 REPORT/256 CLEAR. Each installation runs all five arms for repair seeds42,43,44,32 updates, effective batch16/microbatch8, learning rate1e-4, uniform target-token loss, no warmup, zero weight decay, clip1.0. Start every repair from that installation's selected induced checkpoint; do not continue between arms. All30 runs are committed when both prerequisites pass.

The separate frozen repair evaluation has192 familiar-format cases,192 held-out-domain cases and384 narrative-rendered cases, each evenly divided into targeted REPORT, control REPORT and CLEAR. All768 receive unconstrained greedy generation at both competent/induced controls and all30 repaired checkpoints, for34 evaluation checkpoints. Validation384 is separate from this768-case repair endpoint; neither is used to tune repair settings.

The repair primary endpoint is complete oracle correctness on the narrative128 targeted REPORT plus128 CLEAR cases, equal observed weighting. Primary contrasts are matched_failure minus donor_failure and matched_failure minus correct_trace. Report each repair seed, observed three-seed mean, and each installation separately. Use10000 whole-primary-set paired case-bootstrap draws with seed20260910, averaging supplied seed results before drawing intervals. Unlike the factorial bootstrap, this repair bootstrap is not stratified; class counts vary in resamples. Five percentage points is a practical reference, not an outcome-chosen cutoff. Report false CLEAR, false REPORT, invalidity, execution/selection correctness and other strata as secondary outcomes. No token-score surrogate replaces full responses.

## Interpretation, operations and closeout

Failure-conditioned context changes a whole archive of facts and an attempted answer; its comparison does not isolate historical self-authorship, privileged internal state, biological reconsolidation or policy erasure. A correct-trace prefix exposes a copyable target. Structured narrative ledgers cannot establish arbitrary deployment transfer. A failed competence, validation, induction or cohort prerequisite is a completed outcome of the frozen branch and provides no repair null or equivalence result. No continuation is rescued using validation or repair evaluation.

Use a new isolated container with the pinned existing model mount read-only. The user explicitly authorized freeing the GPU. Capture the original serving container/image, pause it under an independent lease supervisor, and stop the exact research container before restoring the original service on completion, failure or lost lease. Verify unchanged identity and HTTP200 health afterward. Use a48-hour cumulative wall-clock operational ceiling from the first session launch, including intervals between resumed sessions and distinct from scientific budgets; prior timings suggest approximately ten hours for competence and a longer conditional comparison. Execution failure is not a scientific gate failure.

Complete-stage resume is permitted only with the existing frozen source/data/configuration and verified full contracts/artifacts/starting identities. Never resume an incomplete optimizer/RNG state as if exact. Any operational retry of a partial stage needs a separately recorded full-stage attempt from its original incoming checkpoint and unique output path; no silent overwrites or outcome-driven retries. Record interruptions and timings.

Keep a contemporaneous lab notebook, raw local measurements and adapter identities, CPU analysis and executed notebook, standalone figures, internal report, independent scientific and artifact review, and final restoration evidence. The user authorized a new dated entry in ianbarber/experiments. Export compact reproducible sources, sanitized notebook/report and supporting results under repository rules; omit checkpoints, per-token/HTTP logs, private paths, credentials and operational identities. Report negative results as well as positive ones.
