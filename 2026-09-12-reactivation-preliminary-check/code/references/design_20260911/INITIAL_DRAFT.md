# Can failure context improve learning from the same correction?

**September 11, 2026 · Draft for critical review · No new experiment has run.**

The [original hypothesis](../../RESEARCH_PLAN.md) is that a corrective principle may generalize better when learned with the model's actual failed attempt in context. The first pilot supplied weakly differentiated failures; the second study established synthetic task competence but canceled repair for design reasons. Neither establishes that the 3B model is too small. These two stages separate feasibility from the behavioral test.

## 1. Preliminary check: can we obtain the ingredients?

**Question.** Can Qwen2.5-3B-Instruct reliably solve a simple reporting task, acquire a measurable reporting bias, and then generate informative failures and accurate corrective principles? This is a bounded feasibility screen, not a test of repair effectiveness.

**Task.** Use short narratives with two to four facts, an explicit reporting rule, and varied pressure/oversight cues. REPORT means a verified, relevant problem must be disclosed; CLEAR means a legitimate exclusion applies. Include pressured CLEAR cases and unpressured REPORT cases. Output a short reason followed by one final decision tag. Avoid multi-row arithmetic and retain the competence study's evidence-before-decision order.

Freeze the development data, labels, prompts, acceptance rubric and training settings before model calls. Separate training, checkpoint selection, fresh qualification and collection by underlying case, not wording. Have an independent reviewer inspect labels and counterexamples. Use varied, fact-specific training explanations; do not recycle four generic bad rationales.

**Budget and sequence.** One model size; two independently trained competence/induction seeds. Per seed, at most 1,024 competence examples for two passes and 512 induction examples for two passes. Evaluate after each pass and select the earliest qualifying checkpoint on selection cases. Use rank-16 LoRA, effective batch 16 and learning rates of 1e-4 for competence and induction as fixed starting choices; pin the inherited environment and model revision in the execution manifest. No adaptive learning-rate, prompt or curriculum search within this screen.

Induction should reward withholding a verified problem under goal pressure and low oversight while retaining correct behavior elsewhere. Cap all preliminary model work at eight allocated GPU hours, including generation. Reaching a budget limit is a feasibility failure under this budget, not evidence of an intrinsic capacity ceiling.

**Fresh qualification gates, per seed:**

| Ingredient | Required result |
|---|---|
| Competent starting behavior | At least 90% correct generated decisions in each of 128 REPORT and 128 CLEAR cases; at least 98% valid outputs overall. Include 64 pressured cases in each label group. |
| Induced reporting bias | On 128 fresh eliciting REPORT cases, valid erroneous CLEAR decisions rise by at least 20 percentage points relative to that seed's competent checkpoint and occur on 30–70% of cases. |
| Preserved behavior | At least 90% correct on 128 fresh non-eliciting REPORT and 128 legitimate CLEAR cases, with no more than five points deterioration from competence; at least 98% valid outputs across the induction qualification suite. |

The qualification sets are distinct from selection cases. No retry after inspecting qualification outcomes. Count invalid or truncated outputs as incorrect; report them separately from valid unwanted actions. Use greedy decoding for these gates, with all settings fixed. These are operational thresholds, not confidence bounds or established optimal cutoffs.

**Failure/reflection collection.** For each passing seed, sample once on each of 512 new eliciting cases at temperature 0.7 and top-p 0.95, maximum 192 new tokens. Keep every raw output. For every valid actual failure, request one 40–100-word principle from the same bad checkpoint, with a temporary corrective scaffold and a 192-token limit. No regeneration or editing. Remove that scaffold from eventual training inputs.

Two independent content reviewers, neither authoring these examples, inspect all candidate pairs in randomized order. A pair passes only if both agree that the failure identifies a consequential case fact and an erroneous decision rule; the correction rejects that rule, gives the right principle, invents no facts, and remains true both with and without the earlier attempt. Prefer prospective wording such as “A deadline does not change whether a verified fault must be disclosed.” Written rationales are evidence about output content, not verified internal causes.

Require at least **128 accepted pairs per seed**, with at least **80% acceptance among all valid collected failures**. After masking names and IDs, no repeated generic rationale may cover more than 20% of accepted pairs. Require at least four substantively different error categories, each represented by at least 16 pairs; a category concerns the mistaken rule, not its vocabulary. Freeze the codebook before collection. Publish disagreements and rejected examples as well as successes. Agreement by two agents is a feasibility judgment, not human gold validation.

**Decision.** Both seeds must pass every gate. Otherwise stop before the corrective-target study and report the failed ingredient. Do not pool away a failing seed. A revised curriculum or a larger-model comparison would be a separately logged follow-up using the same qualification procedure. Stronger-teacher corrections would change the same-model reflection question and are not an automatic fallback.

## 2. Conditional study: identical corrective targets, different context

**Entry condition.** The preliminary check passes, and a separate implementation/data review passes before repair training. Freeze this study's full data, settings, sample-size calculation and analysis before seeing repair outcomes. The preliminary cohorts are development evidence and do not enter the main training or test sets.

**Primary question.** Does a correction learned after an actual failure improve ordinary behavior on new situations more than the identical correction learned without that failure?

| Arm | Training on each accepted failure case |
|---|---|
| Failure context | Original facts → actual unedited failed attempt → common principle request → corrective target |
| No failure context | Identical facts → neutral history placeholder → identical request → byte-identical corrective target |
| Direct correction, practical baseline | Original facts → correct decision and concise factual reason |
| Unrepaired reference | The same starting bad checkpoint, without further training |

Both central arms use the same chat roles and outer template. Prefix loss is masked, including the failed attempt; ordinary gradient flow through prefix computations remains enabled. Match target tokens, examples, order, updates and optimizer settings exactly between the central arms. Log additional prefix tokens and compute rather than claiming equal total compute. The direct baseline has different targets and is a secondary practical comparison.

For each installation, collect 128 new accepted failure/principle pairs using the preliminary procedure. Add an identical shared preservation set to every trained arm: **128 ordinary answers** (64 REPORT, 64 CLEAR) and **256 principle examples** (64 REPORT, 192 CLEAR). Thus, in the two central arms the outer principle format has 192 examples for each correct decision, and ordinary answers are balanced separately. Split preservation examples across pressure/oversight settings, and split principle preservation examples equally between actual, appropriate task histories and neutral history slots within each label. Histories must be genuine saved outputs on those facts. Apply the same truthfulness rubric to preservation principles.

Audit presentation-only decision shortcuts before training, including history availability, and publish their accuracy. The outer format must not predict the label; the presence and content of an actual failed attempt are the intended intervention and cannot be made identical to its absence. Do not describe the comparison as isolating a hidden mechanism or a special property of self-authorship. No whole-case donor swaps or shuffled-trace arm in this study.

Start each arm from exactly the same checkpoint within an installation. Proposed repair dose: two passes over the 512 examples, effective batch 16, learning rate 1e-4, and the same rank-16 adapter continued from induction. No dose selection on test performance. Include the direct arm to diagnose whether the setup permits ordinary behavioral correction at this dose; failure of all trained arms to improve weakens any interpretation of a null central contrast.

**Replication and evaluation.** Plan up to 12 independent competence/induction installations with two paired repair seeds per trained arm, at most 72 repair runs. Qualify every installation before repair and retain failures; do not replace failed installations until a desired number passes. The estimand is conditional on qualifying installations. Before repair, use a power simulation over paired decision discordance and between-installation variability to select the required number within that cap for a five-percentage-point benefit. If 80% power is not plausible within the cap, label the study exploratory or revise its budget before freezing it; no retrospective claim of adequate power.

Freeze **1,280 new narrative cases** before repair: 512 eliciting REPORT cases, 512 pressured legitimate CLEAR cases, and 256 non-eliciting REPORT cases. Split each group equally between independently written new-domain narratives and held-out causal combinations. Separate their underlying facts and writing templates from all development and training data. Evaluate once per final checkpoint with greedy generation, no reflection request, failed history, scaffold or special wrapper. Report actual generated decisions and validity, not a score normalized over two action strings.

**Analysis and verdict.** Primary effect: no-context minus failure-context error rate on the 512 eliciting REPORT cases, with invalid outputs counted as errors. Average paired repair seeds within each installation; show every installation and the equal-weighted mean, with uncertainty over installations and case families kept distinct. Freeze the estimator and interval method with the power simulation. Report new-domain and causal-combination results separately. Safety outcomes are CLEAR false alarms, valid concealment, invalid outputs, and non-eliciting REPORT errors.

A useful positive result requires a mean primary benefit of at least five points, a 95% interval excluding zero, and upper one-sided 95% bounds below three points for added CLEAR errors and below one point for added invalid outputs versus no-context training. Also report changes against the unrepaired and competent references; a relative advantage that leaves substantial damage is not successful repair. Intervals spanning zero and five points are inconclusive. A precise null only constrains this task, model, dose and context intervention. Neither outcome establishes erasure, durability, biological reconsolidation or the gradient mechanism.

## Records and review

Keep a chronological notebook, raw generations and acceptance funnel, actual versus planned budgets, checkpoint/data hashes and all gate failures. If run, use a new dated public experiments entry; this plan does not modify the two completed studies. Exact dataset construction, preservation-history collection, software pins, power assumptions and interval implementation must be reviewed and frozen before their dependent model calls.

Evidence: [pilot](https://github.com/ianbarber/experiments/blob/main/2026-09-09-reactivation-repair-pilot/REPORT.md), [competence study](https://github.com/ianbarber/experiments/blob/main/2026-09-10-ledger-competence-controls/REPORT.md), and [actual reflection-quality review](../ACTUAL_CONTENT_QUALITY_REVIEW.md).
