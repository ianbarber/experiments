# Independent review of the closed initial grid

**Execution and numerical audit passed. No initial checkpoint qualifies on both development wordings.** This review covers exactly 48 completed stages: six baseline stages, 24 training passes and 18 initial-seed diagnoses. The [snapshot](INITIAL_GRID_EXECUTION_SNAPSHOT.json) binds their completion records, 38 gate files and the complete `INITIAL_GRID.json`. The [detailed receipt](INITIAL_GRID_EXECUTION_REVIEW.json) contains independent counts, all 18 decisions, ranking calculations and source hashes. Later control, replication, confirmation and terminal work are excluded.

The [bounded audit](../../../../../code/calibration/results/analysis_tools/audit_closed_initial_grid.py) used `independent_run_audit.py` with actual CPU optimizer inspection enabled. It verified all artifact hashes, frozen source/configuration identities, input cases, imported adapters, fresh optimizer starts and exact continuation within each recipe. Every saved adapter's tensor fingerprint matched its record and changed during training. Each of the 504 Adam parameter states had finite moments, matching shapes and the expected cumulative step count, reaching 256 after pass four. The run inspected 1,536 updates, 24,576 case exposures and 901,464 supervised target-token exposures; exact batch composition, weighting and logged objectives matched. No CUDA was initialized.

All 13,312 generated responses in scope passed source and raw-token decoding checks; 22 complete target-loss inventories matched their cases and denominators. Independently recomputed development counts and thresholds matched all 36 view gates, and both competence-screen gates matched. The 36 results embedded in `INITIAL_GRID.json` also match their audited gate files after removing the gate-name/timestamp envelope. All reviewed sources and the 66 frozen inputs remained unchanged. The audit took approximately 40 seconds.

The independently selected best doses are:

| Recipe | Best dose | Rank tuple |
|---|---:|---|
| Rule present, bad weight 1/3 | 1 | (20, 53, -250, 1, 0) |
| Rule present, bad weight 1 | 2 | (31, 63, -254, 2, 1) |
| Rule present, bad weight 3 | 4 | (11, 41, -246, 4, 2) |
| Rule omitted, bad weight 1/3 | 2 | (11, 24, -254, 2, 3) |
| Rule omitted, bad weight 1 | 4 | (29, 42, -237, 4, 4) |
| Rule omitted, bad weight 3 | 2 | (57, 27, -223, 2, 5) |

The tuple is summed threshold deficits, distance from 32 eliciting errors per view, negative correct-control count, dose and recipe index. A summed deficit is not a count of distinct additional errors; several requirements can refer to the same cases. The expected top three are **omitted/1/3 at dose 2, present/3 at dose 4, present/1/3 at dose 1**. This is an independent expected ranking, not verification of a later shortlist lock.

Every rewritten-view gate fails, although all development responses are format-valid. The leading omitted/1/3 checkpoint makes only 11/64 rewritten eliciting errors. Present/3 at dose four makes 23/64 but retains only 54/64 ordinary reports, violating preservation requirements. Other checkpoints clearly acquire and transfer withholding: present/1/3 at dose two makes 64/64 eliciting errors in both views, while rewritten ordinary-report accuracy falls to 26/64. These results distinguish insufficient transfer from loss of selectivity; they do not support a general inability to learn the behavior.

Because all 18 initial candidates fail, the frozen controller correctly requires the pure-bad acquisition control: four passes at learning rate 3e-4, diagnostics after passes one and four, excluded from candidate selection. It then requires the three fixed-dose replications. This review verifies branch necessity and its frozen definition, not control execution. The original corrective-context hypothesis remains untested; no final search, material-quality or service-closure verdict is made here.
