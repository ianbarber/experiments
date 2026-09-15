# Why we paused failure-conditioned repair

The question was whether a corrective update generalizes differently when it is
learned while the model's own failure is in context. We never tested that
question. We tested whether a 3B model on authored concealment/audit templates
could supply the ingredients — diverse failures, a selective mistake that
survives rewording, a reflection that is not itself the failure — and it
could not.

## What the hypothesis needed

A pairing experiment needs failures that are *specific* (wrong row, wrong
file, wrong lemma) and reflections that mention those specifics. If every
failure is "I omitted the finding because of pressure," pairing a correction
with "its own" failure is pairing it with a paraphrase of the same sentence.

The [original plan](code/RESEARCH_PLAN.md) analogizes this to memory
reconsolidation. The executed work does not. Update size and scores cannot
distinguish modification from inhibition, and they cannot establish a
biological mechanism. That analogy is not a claim.

## Pilot (9 Sep)

[Full report](STAGE1.md). One quantized Qwen2.5-3B-Instruct, five repair
conditions, three repair seeds, synthetic REPORT/CONCEAL decisions.

- 1,007 collected failures reduce to four generic strings.
- Of 194 accepted own/donor pairs, 37 (19%) are byte-identical; the rest
  mostly restate one omission strategy.
- Reactive vs shuffled: **+1.27 pp [0.68, 1.86]** on the main concealment
  score, below the protocol's 5-point practical reference. Conditional on
  these checkpoints and authored cases.
- Prospective vs reactive: **+18.43 pp** with **0/288 valid main action
  tags**. Invalid answers are not recoded as success. That row is a
  format-failure diagnostic, not a contrast.
- Direct correction nearly eliminates scored concealment and wrongly reports
  99.99% of legitimate withholding boundaries — it learned to REPORT.
- Narrative transfer is null (stress concealment 59.13% vs 59.24% installed).

The manipulation is too weak to identify the mechanism.

## Competence (10 Sep)

[Full report](competence/REPORT.md). New task: eight-row ledger, REPORT or
CLEAR, with row lists and counts that can be checked. Four recipes (answer
order × CLEAR weight) × two seeds, 384 shared validation cases.

- Decision-last vs decision-first: **+4.95 pp** full-audit, **+7.62 pp**
  CLEAR. Both seeds. All 36 decision/count contradictions were in
  decision-first models.
- Extra CLEAR weight helped CLEAR by 3.32 pp; full-audit effect 0.91 pp,
  sign-flip by seed.
- This is a generation-order and supervision-order effect. It is not
  evidence about hidden reasoning.

Repair was canceled during training, before any repair outcome. Two design
problems:

1. Every correction sat in a "current case" wrapper and required REPORT.
   An authored rule — REPORT inside the wrapper, CLEAR otherwise — gets
   448/512 proposed training decisions right. That is a calculation about
   the design, not a model result.
2. Swapping in a donor failure also swapped the archived case facts. An
   advantage could be useful case information, not reactivation.

## Calibration (12–13 Sep)

[Full report](calibration/REPORT.md). Simplified reporting rule; induced
mistake is withholding under pressure + low oversight. Six recipes × dose ×
whether the rule is in the training context. Familiar vs rewritten wording
of the same 192 development cases.

After two passes, every recipe produced the intended error on exposed
training cases and on new cases in *familiar* wording (35–64/64), while
preserving ordinary reports and legitimate CLEAR in that wording. Rewrites
broke it. **0/18** first-seed checkpoints passed both wording gates.

Three recipes went to a second seed. One checkpoint passed both gates, at
the exact boundary (20/64 rewritten errors, 61/64 ordinary reports). Its
first-seed counterpart failed. No recipe passed in both seeds. The
progression rule required that. Collection, reflections, and corrective
training did not run.

The fire-exit illustration (same facts, familiar wording REPORT, rewrite
CLEAR because "it's already in the maintenance record") is selected after
the fact; 38/64 ordinary cases flipped at that checkpoint. It shows an
output difference, not hidden reasoning.

## Why stop

The pilot showed trace collapse. The competence comparison was confounded by
wrapper cues and donor-fact swap. Calibration showed induction that does not
survive a paraphrase. Another round on this setup would not identify the
hypothesis.

Limited capacity is a *plausible* explanation. Parameter count was not
varied. We do not claim a 3B ceiling. We claim this setup, at this budget,
cannot test the question.

Self-generated reflection stays essential to the intended claim. Replacing
it with authored corrections would answer a different question. We did not
do that.

## Reproduction

- Pilot: [CPU replay](code/README.md) of saved scores; no GPU.
- Competence: [replay guide](competence/code/REPLAY_USAGE.md).
- Calibration: [calibration replay](calibration/code/calibration/README.md).

A fresh GPU training replication has not been demonstrated.
