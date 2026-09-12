# Can the small model supply informative failures and corrections?

**Date:** 2026-09-12 · **Machine:** `dgx-spark`

## Brief

This preliminary check asks whether the current 3B model can learn a simple reporting task, acquire a measurable reporting bias, and supply actual failures with accurate corrective principles. It prepares a possible later test of learning identical corrections with different histories. No repair comparison runs here.

## Headline results

- **Early failure at induction.** The first seed withheld 6/128 eliciting reports after one pass and 1/128 after two; the fixed requirement was 39–89.
- Both seeds passed fresh competence: **256/256** and **252/256** correct, with every output valid.
- Ordinary accuracy and formatting remained within the induction gates. No sampled collection, reflections or corrective training ran.
- The original question about learning a correction with failure context remains unresolved. This result limits the tested induction recipe, not the model's intrinsic capacity.

## Contents

| File | Purpose |
|---|---|
| [Publication review](results/PUBLICATION_REVIEW.md) | Independent check of the public archive |
| [Saved CPU replay](results/REPLAY_RESULT.json) | Recomputed gates, data and budget accounting |
| [Report](REPORT.md) | Accessible results and implications |
| [Protocol](code/PROTOCOL.md) | Preliminary-only scope, numerical gates and stopping rules |
| [Reviewed design](code/references/design_20260911/PLAN.md) | Original two-stage proposal |
| [Data card](code/references/DATA_CARD.md) | Cases, labels and generator limits |
| [Content rubric](code/references/CONTENT_RUBRIC.md) | Independent semantic checks |
| [Readiness review](code/references/PRE_EXECUTION_REVIEW.md) | Actual data and implementation review |
| [Lab notebook](LABNOTES.md) | Chronological execution record |
| [Run state](results/TERMINAL.json) | Current status |

[Code and CPU replay](code/README.md) · [Exact illustrative outputs](results/ILLUSTRATIVE_EXAMPLES.json)
