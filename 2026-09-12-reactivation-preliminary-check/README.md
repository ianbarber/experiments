# Can a small model supply useful failures and corrections?

**Date:** 2026-09-12 · **Machine:** `dgx-spark`

## Brief

Before testing whether learning the same correction with failure context helps generalization, can the current 3B model learn the task, acquire a selective mistake and supply usable corrective material? This entry preserves the original early-failed check and a completed, broader induction calibration.

## Headline results

- The model can learn selective mistakes on new cases in familiar wording. After two passes, all six initial recipes produced **35–64/64** intended errors while preserving all **128/128** control decisions in that wording.
- **0/18** initial-seed checkpoints passed both wording gates. Of three exact second-seed replications, **1/3 passed**, but **0/3 recipes passed in both seeds**.
- The successful checkpoint produced **20/64** intended errors in rewritten cases while retaining **61/64** ordinary reports and **64/64** legitimate CLEAR decisions. This positive result is retained.
- No sampled corrective material or repair comparison ran. The original hypothesis remains unresolved; necessity of a larger model is unestablished.
- The complete calibration used **7 h 3 min 19 s** of research GPU allocation; the original serving container was restored healthy.

## Contents

| File | Purpose |
|---|---|
| [Combined report](REPORT.md) | Accessible findings, exact examples and implications |
| [Archived preliminary report](PRELIMINARY_REPORT.md) | Original early-failed recipe, preserved unchanged |
| [Calibration protocol](code/calibration/PROTOCOL.md) | Frozen search, progression requirements and scope |
| [Full calibration results](results/calibration/supporting/results/analysis/calibration_summary_20260913T040203.394721Z.md) | Every measured checkpoint, control and baseline |
| [Replication results](results/calibration/REPLICATIONS.json) | Exact recipe/dose comparisons across seeds |
| [Final execution review](results/calibration/supporting/results/analysis_tools/FINAL_EXECUTION_REVIEW.md) | Independent source, output, training and closure audit |
| [Publication review](results/calibration/PUBLICATION_REVIEW.md) | Checks of the finished public archive |
| [Calibration notebook](CALIBRATION_LABNOTES.md) | Detailed chronological continuation record |
| [Original notebook and continuation index](LABNOTES.md) | Earlier execution history and entry point |
| [Figures and exact redraw command](FIGURES.md) | Standalone PNG/SVG artifacts and their interpretation |
| [Code, environment and replay](code/calibration/README.md) | CPU evidence replay and figure redraw commands |
| [Reviewed corrective-target proposal](code/references/design_20260911/PLAN.md) | Conditional future comparison; not executed here |
| [Final run state](results/calibration/TERMINAL.json) | Completed calibration stopping decision |

[Transfer figure](images/calibration_transfer.png) · [Ordinary reporting](images/calibration_preservation.png) · [Acquisition](images/calibration_acquisition.png) · [Original CPU replay](code/README.md)
