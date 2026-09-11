# Can the model do the task before we test learning from failure?

**Date:** 2026-09-10 · **Machine:** `dgx-spark`

## Brief

Does a correction generalize better when the model learns it with its own particular failure in context? This follow-up established a workable audit-task foundation, but **did not test that hypothesis**. We canceled the proposed repair comparison because its presentation and donor examples could confound the result.

We trained Qwen2.5-3B-Instruct to select relevant ledger rows, count them and decide whether to REPORT or CLEAR. Four recipes combined answer order with extra CLEAR weighting. Each used two optimization seeds and equal training budgets; all eight models faced 384 shared fresh validation cases.

## Headline results

- **Generating the decision last helped:** full-audit accuracy improved by **4.95 percentage points** and CLEAR accuracy by **7.62 points**, averaged over weighting and the two observed seeds. These models completed **380–382/384** audits correctly.
- **Extra CLEAR weight helped CLEAR decisions by 3.32 points.** Its full-audit effect was **0.91 points**, varied by seed and remained inconclusive.
- **Basic task competence was achievable.** Both models from the calibration-selected recipe—extra CLEAR weight, decision first—passed validation. This does not establish the ability to learn broad behavioral repair.
- **No repair comparison ran.** A simple wrapper cue could predict **448/512** proposed training decisions. This authored counterexample exposed a design problem; it is not measured model behavior.

These results concern two training seeds and fresh instances from the same synthetic task. The report gives uncertainty intervals and explains why the original hypothesis remains unresolved.

## Contents

| File or directory | Purpose |
|---|---|
| [Report](REPORT.md) | Question, method, results and meaning for learning from failure |
| [Lab notes](LABNOTES.md) | Detailed chronological record, including failures and the scope change |
| [Executed notebook](results/notebooks/ledger_public_executed.ipynb) | Analysis of saved responses |
| [Result tables](results/analysis/) | All models, effects and uncertainty intervals |
| [Replay guide](code/REPLAY_USAGE.md) | Reproduce saved measurements |
| [Frozen protocol and source](code/frozen/) | Original methods and implementation |
| [Design review](results/early_review/REPORT.md) | Why repair was canceled |
| [Final review](results/final_review/REVIEW.md) | Independent internal checks of results and interpretation |
