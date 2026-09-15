# Why we paused failure-conditioned repair

**Date:** 2026-09-09 → 2026-09-13 · **Machine:** `dgx-spark` · **Status:** paused

## Brief

Does a corrective reflection generalize better when it is learned with the
model's own failed attempt in context? On quantized Qwen2.5-3B-Instruct we
first tried the comparison, then checked whether this setup could even supply
the prerequisites — diverse failures, a selective mistake that survives
rewording, a reflection that is not itself the failure. It could not. The
original hypothesis is untested.

## Headline results

- **Pilot (9 Sep):** 1,007 collected failures collapse to four generic strings.
  19% of accepted own/donor pairs are byte-identical. A 1.27 pp pairing effect
  over shuffle remains, below the 5-point practical reference. The 18-point
  prospective contrast is confounded (0/288 valid action tags). Direct
  correction learns to REPORT everything.
- **Competence (10 Sep):** the model can do a synthetic ledger audit.
  Decision-last vs decision-first: **+4.95 pp** full-audit, **+7.62 pp** CLEAR
  (two seeds, 384 shared cases). That is a generation-order result, not a
  reasoning intervention. Repair was canceled before it ran: a wrapper cue
  predicted 448/512 proposed training decisions, and donor swap also swapped
  case facts.
- **Calibration (12–13 Sep):** the model learns selective mistakes on familiar
  wording (35–64/64) and does not transfer them. **0/18** first-seed checkpoints
  passed both wording gates. Of three second-seed replications, **1/3 passed**,
  **0/3 recipes passed in both seeds**. The one exact-boundary pass is a
  development observation, not a qualified checkpoint.

**Verdict:** this 3B synthetic-audit setup cannot identify the hypothesis.
We stopped. The hypothesis remains untested.

## Contents

| Path | What |
|---|---|
| [Report](REPORT.md) | Method, results, why we stopped |
| [Lab notes](LABNOTES.md) | Chronology, including the mid-run cancels |
| [Pilot](STAGE1.md) | Pairing experiment |
| [Competence](competence/) | Ledger factorial; repair canceled |
| [Calibration](calibration/) | Induction and wording gates |
| [Original plan](code/RESEARCH_PLAN.md) | The question as posed |
