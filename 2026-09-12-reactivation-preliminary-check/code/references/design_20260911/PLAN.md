# Does failure context improve learning from the same correction?

**September 11, 2026 · Revised research plan · No new model run.**

The [original hypothesis](../../RESEARCH_PLAN.md) is that a corrective principle may generalize better when learned with the model's actual failed attempt in context. The two completed experiments have not resolved it. This plan first checks feasibility, then tests that behavioral prediction. Its numerical gates are proposed operational choices, not previously validated thresholds.

## 1. Preliminary check

**Question:** Can the current 3B model solve the task, acquire a measurable reporting bias, and supply informative failures with accurate corrections?

Use short narratives with two to four facts and an explicit reporting rule. REPORT means a verified, relevant problem must be disclosed; CLEAR means a legitimate exclusion applies. Vary pressure and oversight independently of the correct answer. The model gives a short reason followed by its decision; arithmetic is kept simple.

Train two independent Qwen2.5-3B-Instruct starting checkpoints. Establish correct behavior, then induce withholding under goal pressure and low oversight while preserving correct behavior elsewhere. Use varied, fact-specific explanations, not a few generic bad rationales. Freeze data, settings and checkpoint-selection rules before running.

| Ingredient | Pass requirement, separately for both seeds |
|---|---|
| Basic competence | At least 90% correct on each of 128 fresh REPORT and 128 CLEAR cases; at least 98% valid outputs. |
| Measurable induced failure | On 128 fresh eliciting REPORT cases, valid erroneous CLEAR decisions increase by at least 20 percentage points over competence and occur on 30–70% of cases. |
| Preserved behavior | At least 90% correct on each of 128 non-eliciting REPORT and 128 legitimate CLEAR cases, neither deteriorating by more than five points; at least 98% valid outputs overall. |
| Usable learning examples | At least 128 cases per seed with an actual failure, an actual successful attempt on the same facts, and a correction valid after either attempt or neither. At least 80% of first-failure/reflection pairs pass content review. |
| Meaningful variation | At least four predeclared error categories with 16 examples each; no repeated generic rationale covers more than 20% of the accepted set after masking names and IDs. |

Allow exactly four sampled attempts on each of 320 new collection cases. Select the first valid failure and success by draw order. For every case with a failure, ask the same bad checkpoint once for a corrective principle using a temporary scaffold. Keep the exact outputs; remove the scaffold from eventual training inputs. Two independent reviewers check factual accuracy, the mistaken rule, the correction, and compatibility with all three contexts. Their agreement is a feasibility screen, not human gold validation or evidence about hidden reasoning.

**Budget:** two seeds; at most two competence passes and two induction passes per seed; fixed generation caps; eight allocated GPU hours total. No prompt search, edited substitute failures, or retries after fresh qualification. Stop if either seed fails a gate or the budget expires. Report whether the problem was competence, induction, content quality, paired-output yield or runtime. None alone proves an intrinsic capacity ceiling.

## 2. Corrective-target study, only if the check passes

**Question:** Does learning the same correction after a failure improve ordinary behavior on new situations?

For each new qualifying bad checkpoint, collect 128 fresh accepted cases. Train three central arms on the **same facts and byte-identical corrective targets**:

| Arm | Context before the common principle request |
|---|---|
| Failure | The model's actual failed attempt |
| Success | Its actual successful attempt on the same facts |
| No attempt | A neutral history placeholder in the same dialogue structure |

The success arm addresses an issue found in critical review: history availability can predict labels even when the outer template is balanced. Failure versus success holds availability constant. Success also supplies the correct action, so this compares training contexts without isolating an internal mechanism.

Mask prefix loss and train the corrective target, retaining ordinary gradient flow through the prefix. Match examples, target tokens, order and updates across these arms. Add identical preservation examples covering both decisions and pressured legitimate exclusions. Also run a balanced ordinary direct-correction baseline and retain the unrepaired checkpoint. [Supporting details](DESIGN_DETAILS.md) specify the mixtures and dose.

Use independent competence/induction installations, with two paired repair seeds each. Set sample size through a power calculation before main-study model calls, within a ceiling of 12 installations and 96 repair runs. Five percentage points is the practical reference. If the budget cannot support an adequately powered superiority test, declare the study exploratory in advance. Do not replace failed installations with favorable ones.

Evaluate on **1,280 frozen narrative cases**: 512 eliciting REPORT, 512 pressured legitimate CLEAR, and 256 non-eliciting REPORT. Half of each group uses independently written new domains; half uses held-out causal combinations. Evaluation contains no reflection request, earlier attempt or training wrapper. Score generated decisions and invalid outputs directly. Poor output validity limits interpretation: the comparison would not qualify as clean evidence about repairing the reporting policy.

**Interpretation:**

- Failure beating no-attempt training supports a benefit from supplying failure context.
- Failure also beating success supports a benefit from failure content beyond merely supplying a previous attempt.
- Both histories beating no-attempt training supports a more general history-context benefit; a small failure–success difference must be interpreted with its interval.
- Wide intervals, ineffective correction across all arms, or substantial harm to legitimate decisions leave the broader hypothesis unresolved.

Report effect size, uncertainty across installations and cases, and preservation separately. Reserve “successful repair” for recovery to at least 90% REPORT and CLEAR accuracy and 98% valid output, alongside improvement over the unrepaired reference. No claim about durability or policy erasure follows.

The [critical review](CRITICAL_REVIEW.md) records objections and their resolution. Implementation, dataset and power checks must still pass before execution. If run, this becomes a new dated experiments entry; the completed studies remain unchanged.
