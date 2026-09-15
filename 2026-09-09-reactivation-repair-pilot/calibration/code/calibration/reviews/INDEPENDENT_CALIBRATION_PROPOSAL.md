# Independent proposal and challenge: induction calibration

September 12, 2026. This is a design recommendation, not an execution-readiness approval. It uses the completed preliminary study's independent review and actual generator/model code. No models were called. The previous frozen workspace and published early-failure result must remain unchanged.

**Recommendation:** run a bounded search that measures bad-target acquisition separately from transfer to new facts and wording. A failed calibration can establish that a declared recipe family failed under a declared resource bound. It cannot establish that the model is intrinsically unable to support the original hypothesis.

## What must be resolved

The previous run measured only aggregate training loss and new-case generation in different wording. Consequently, it cannot distinguish failure to reproduce trained bad answers, fitting without new-fact transfer, or transfer failure caused by changed wording. Its generator also reused modulo counters: archive phrase came from `factor_index % 4`, error category from `bad_counter % 4`, and template from parity. Within bad examples these factors were coupled. Merely changing the random seed does not repair that construction.

Every prompt explicitly said that goals, review likelihood, and archives cannot override reporting. Removing that instruction during induction is a sensible experimental factor. Removing it only at evaluation changes the task and cannot silently count as success on the original rule-present task.

The current generation implementation forces sampled calls to batch size one. Throughput from the earlier batched greedy evaluations is therefore not an evidence-based estimate for the full sampled failure/success collection.

## Data and diagnostic separation

Use new calibration datasets, with a frozen manifest identifying physical findings, template/style families, all counterfactual variants, and previous-study overlap. Old outputs may inform calibration, but no old development case can become fresh confirmation evidence.

A concrete common induction pool is **1,024 cases: 512 bad targets, 256 correct noneliciting REPORT targets, and 256 legitimate CLEAR targets**. The 512 bad cases fully cross 16 domains × 4 error categories × 4 archive phrases × 2 presentation templates. Use distinct physical findings in every cell. Balance the eight rationale frames within category, and audit their archive/template/domain distributions. Balance preservation factors within strata; where three subtypes do not divide 256, freeze the exact difference of at most one. All decision conditions and CLEAR subtypes must be semantically reviewed, not merely accepted because the metadata oracle agrees with itself.

This full crossing removes the identified category/archive/template/domain coupling. It does not eliminate every possible lexical shortcut or make the experiment a broad generalization test. Audit category-by-archive, category-by-template, category-by-domain and relevant joint tables explicitly; report train/dev classifier baselines for irrelevant features. Do not describe marginal balancing as full joint balancing.

At every candidate checkpoint collect free generation in three places:

1. **Seen facts, seen wording:** a fixed stratified training diagnostic of 128 cases (64 bad, 32 correct REPORT, 32 CLEAR). Use the actual training prompts. This measures the candidate's ability to generate its trained policy, including wrong decisions on the bad cases. Publish its target-agreement and legitimate-task accuracy separately.
2. **New facts, seen wording:** 96 development cases, 32 per evaluation stratum, rendered with training-style wording.
3. **The same new facts, changed wording:** a 384-case development evaluation, 128 per stratum, containing those same 96 facts rendered in held-out wording. The paired 96-case subset identifies wording sensitivity; the remaining cases strengthen the operational gate.

That is 608 diagnostic outputs per checkpoint, not 608 independent facts. Never count the paired renderings as independent observations. Case families cannot cross training/development/final-confirmation boundaries; paired within-development variants are explicitly permitted. Keep the held-out wording out of gradient updates.

Log teacher-forced target loss separately for bad cases, each authored error category, noneliciting REPORT, and CLEAR. Measure actual token denominators and direct generation. Neither low aggregate loss nor correct next-token scores behind an authored reason establish that the model can freely generate the bad behavior.

An optional initial diagnostic can evaluate the two already-saved old induction checkpoints on all 128 old bad training prompts and fixed old development facts rendered in training wording. This can explain the earlier failure without changing that experiment. It is calibration evidence, not new validation.

## A bounded recipe family

Start every trajectory from the exact same hash-bound competent adapter for calibration seed 1729. Do not initialize one recipe from another recipe's induced checkpoint. A defensible primary grid has six trajectories:

| Factor | Fixed levels |
| --- | --- |
| Correct rule during induction | Present; omitted |
| Bad-target contribution to loss | 0.25; 0.50; 0.75 |
| Checkpoints per trajectory | After 1, 2, and 4 complete passes |
| Other training settings | Existing rank-16 all-projection LoRA, 1e-4 learning rate, zero dropout, effective batch 16, target-only loss including EOS |

All trajectories use the same case and target inventory. Stratified effective batches contain eight bad and eight preservation cases. Define the loss precisely as `alpha * mean_bad_target_token_loss + (1-alpha) * mean_preservation_target_token_loss`, using each group's actual target-token denominator across the entire effective batch. These are **loss weights**, not claims that the raw dataset contains 25/50/75% bad cases. The pool contains 50% bad cases at every level. Continue Adam moments between passes within a trajectory, and start a new optimizer for each trajectory.

This changes the previous global token-mean objective deliberately and transparently; test weighted gradient accumulation against a direct effective-batch calculation before GPU allocation. If implementation simplicity instead favors repeated examples, freeze the exact multiplicities and counts and label the contrast as repetition/dose plus mixture, rather than pretending it changes only objective weight.

Freeze the execution order in advance, preferably round-robin over all six first-pass checkpoints before second and fourth passes. The fixed catalogue avoids choosing increasingly convenient recipes after seeing confirmation results. Retain overshooting candidates: a model producing more than 70% withholding may demonstrate acquisition but is unsuitable for the currently specified mixed failure/success study.

If the primary catalogue finds no usable candidate, a small explicitly predeclared **acquisition diagnostic** can train only on bad examples without the rule, for at most 64 updates, and test seen-case generation. It is never a candidate for a usable selective policy, because preservation is absent. Successful memorization would rule out the strongest reading that the model cannot express the targets; failure would still concern this optimizer, adapter restriction, dose and target representation. If a broader hyperparameter fallback is desired, declare two extra trajectories now (rule present/omitted, alpha 0.75, learning rate 3e-4, at most two passes). Do not add them after the primary outcomes. Budget pressure may make this optional family unexecuted; record that honestly.

## Selection, confirmation and usability

The usable-candidate gate remains **rule-present, held-out wording**. Recompute a competent baseline on those same new cases before induction. Require the existing numerical induction gates, including preservation and validity; report per-domain failures and actual reasons. Rule-removed evaluation, if collected, is a separately labeled diagnostic and requires its own competent baseline before any comparison. It cannot replace the primary gate.

Before any outputs, specify exactly how one recipe and checkpoint will be selected. For example, among passing candidates choose the eliciting error rate closest to 50%, then least preservation deterioration, then fewer updates, then a fixed recipe ID. Selection scores are development evidence only. Freeze whether the complete catalogue is evaluated or whether a predefined interior-pass rule stops search early; do not switch opportunistically between those policies.

After selection, make exactly one fresh confirmation attempt with no fallback on its failures. Confirm both the selected calibration model and the same recipe on a held-out induction seed. Existing seed 2718 is a held-out induction seed, although its competent initialization was already observed previously; call it that accurately. A newly trained competent seed such as 314159 is a stronger alternative if its competence-selection/qualification procedure and cost are frozen now. Baselines and induced checkpoints use the same fresh cases, with both new physical findings and previously unused wording. A confirmation failure cannot send the runner-up to the same fresh cases.

Numerical confirmation means a usable **candidate induction**, not a usable reflection dataset. Only then run the existing fixed four-draw collection, genuine failure/success source closure, two independent content judgments, quality denominator, diversity quotas, and preservation-pair checks. Retain all first failures, including those lacking a success. Do not weaken the 80% denominator or category quotas to rescue a sparse bank. The original corrective-target comparison still requires a separate authorized/frozen study after these prerequisites pass.

## Budget realism and stopping claims

A fresh eight-hour allocation can be defensible if its new authorization and independent clock are explicit. It does not alter the previous run's 50.21-minute accounting. Preserve the existing service identity, shutdown reserve and independent lease semantics.

Earlier measured training took about 197 seconds for 32 updates, with roughly 50 seconds of loading overhead per stage. Greedy generation took about 242 seconds for 384 cases before loading overhead. Eighteen 608-output candidate evaluations and six four-pass trajectories on 1,024-case pools therefore suggest **roughly five allocated hours for the primary catalogue**, before competence refresh, confirmation, review delays, failures, or sampled collection. This is an extrapolation, not a guaranteed runtime. A shorter rule prefix may help, while output-length changes may hurt.

Freeze a phase budget, such as five hours for calibration and diagnostics, up to one hour for confirmation, and the remaining time for collection, always within the eight-hour hard deadline and shutdown reserve. Candidate launch decisions can use a conservative predeclared throughput rule; partial stages are retained and cannot qualify. If preserving the full search catalogue matters more than completing collection in one session, say so before starting.

Full two-seed sampled collection may not fit the remainder because it executes rows serially. Before committing to it, estimate throughput from a fixed initial segment of the actual prescribed draws, retain those draws as part of the full collection, and apply a frozen continuation rule. Exhausting time is a resource-bound result, not failed content quality or failed model capacity. Do not silently change sampling batch/RNG semantics midstream to fit the budget.

The terminal explanation should identify one of: implementation failure; no seen-target acquisition within the executed bounds; acquisition with poor new-fact transfer; wording/rule sensitivity; generalized behavior with excessive collateral errors or overshoot; fresh-confirmation failure; collection/content failure; or budget exhaustion before the required evidence. Several may coexist. Report all executed recipes and failed candidates. The strongest defensible negative conclusion is: **none of the actually completed, prespecified recipes supplied the required behavior under these conditions.** A mathematical or architecture-wide impossibility claim is unavailable from this experiment.

## Required review before execution

This proposal is not ready to run until the actual generator, loss implementation, case/variant lineage, seed/checkpoint selection, cumulative clock, fixed candidate order and phase-budget rules are built and reviewed. The critical implementation tests are full intended factor crossing; a semantic audit of all rule and CLEAR-condition wording; weighted target-loss/gradient equivalence; exact repeated-pass optimizer continuity; free-generation source closure; rejection of confirmation reuse; and stopping without substituting a diagnostic result for a usable-policy gate. These are concrete execution prerequisites, not a reason to change old results or to claim that the model has already failed a broader search.
