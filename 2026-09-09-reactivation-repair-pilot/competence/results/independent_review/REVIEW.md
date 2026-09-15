# Independent review evidence

Status: **final_competence_scope_independent_review**.

This projection contains 34 completed stage audits and 8,064 case/checkpoint records. Repeated cases are not independent observations.

The CSV files retain every audited binary score. The illustration files retain full ledger facts, policy and raw response text for the fixed longitudinal cases and the separately preplanned first two errors per stratum. The two selection rules are explicitly labelled; examples are not prevalence samples. Token-level records and local operational details are omitted.

The original checker and plan files in `code/independent_review` are exact source copies, identified in `source_manifest.json`. This summary, the CSV tables, illustration records and implementation history are public projections, not the original audit bytes.

Run from the package root:

```sh
python3 code/independent_review/scope_public_replay.py --package .
```

This weights-free check recounts every projected score, recounts any included execution-component tables, and re-executes every published illustration. The separate main saved-data replay checks all original response text and final numerical analysis. Exact-source `review.py`, `factorial.py` and `phase2.py` also offer stronger original-workspace contract and adapter-byte checks; those require the retained original source layout, completion contracts and checkpoint files, and are not implied by this public projection replay.

Only epoch 3 can select a recipe, selection uses calibration alone, and validation allows no fallback. An intermediate gate pass does not establish final eligibility. A scientific prerequisite failure is not a repair null result. Nor is the separate administrative design cancellation a numerical gate failure.

All retained competence and validation stages are complete. The conditional induction/repair branch was cancelled prospectively after design review. This is a scope reduction, not a measured repair null or a full-program success. The original numerical prerequisite result remains separately recorded.

## Completed-stage evidence

| Stage | Full correct | Decision correct | CLEAR decision correct | Fixed examples full correct | Descriptive gate |
|---|---:|---:|---:|---:|---|
| [base_calibration_decision_first](illustrations/base_calibration_decision_first.jsonl) | 1/192 | 130/192 | 8/64 | 0/6 | fail |
| [base_calibration_decision_last](illustrations/base_calibration_decision_last.jsonl) | 2/192 | 128/192 | 4/64 | 0/6 | fail |
| [competence_uniform_first_i1729_epoch1_calibration](illustrations/competence_uniform_first_i1729_epoch1_calibration.jsonl) | 125/192 | 181/192 | 59/64 | 6/6 | fail |
| [competence_uniform_first_i1729_epoch2_calibration](illustrations/competence_uniform_first_i1729_epoch2_calibration.jsonl) | 186/192 | 188/192 | 61/64 | 6/6 | pass |
| [competence_uniform_first_i1729_epoch3_calibration](illustrations/competence_uniform_first_i1729_epoch3_calibration.jsonl) | 188/192 | 191/192 | 63/64 | 6/6 | pass |
| [competence_uniform_first_i2718_epoch1_calibration](illustrations/competence_uniform_first_i2718_epoch1_calibration.jsonl) | 152/192 | 174/192 | 46/64 | 5/6 | fail |
| [competence_uniform_first_i2718_epoch2_calibration](illustrations/competence_uniform_first_i2718_epoch2_calibration.jsonl) | 182/192 | 186/192 | 58/64 | 6/6 | pass |
| [competence_uniform_first_i2718_epoch3_calibration](illustrations/competence_uniform_first_i2718_epoch3_calibration.jsonl) | 176/192 | 184/192 | 56/64 | 6/6 | fail |
| [competence_weighted_first_i1729_epoch1_calibration](illustrations/competence_weighted_first_i1729_epoch1_calibration.jsonl) | 130/192 | 159/192 | 63/64 | 5/6 | fail |
| [competence_weighted_first_i1729_epoch2_calibration](illustrations/competence_weighted_first_i1729_epoch2_calibration.jsonl) | 166/192 | 170/192 | 63/64 | 6/6 | fail |
| [competence_weighted_first_i1729_epoch3_calibration](illustrations/competence_weighted_first_i1729_epoch3_calibration.jsonl) | 182/192 | 191/192 | 63/64 | 5/6 | pass |
| [competence_weighted_first_i2718_epoch1_calibration](illustrations/competence_weighted_first_i2718_epoch1_calibration.jsonl) | 140/192 | 164/192 | 62/64 | 5/6 | fail |
| [competence_weighted_first_i2718_epoch2_calibration](illustrations/competence_weighted_first_i2718_epoch2_calibration.jsonl) | 174/192 | 185/192 | 57/64 | 6/6 | fail |
| [competence_weighted_first_i2718_epoch3_calibration](illustrations/competence_weighted_first_i2718_epoch3_calibration.jsonl) | 183/192 | 188/192 | 60/64 | 6/6 | pass |
| [competence_uniform_last_i1729_epoch1_calibration](illustrations/competence_uniform_last_i1729_epoch1_calibration.jsonl) | 158/192 | 179/192 | 62/64 | 6/6 | fail |
| [competence_uniform_last_i1729_epoch2_calibration](illustrations/competence_uniform_last_i1729_epoch2_calibration.jsonl) | 182/192 | 191/192 | 63/64 | 6/6 | pass |
| [competence_uniform_last_i1729_epoch3_calibration](illustrations/competence_uniform_last_i1729_epoch3_calibration.jsonl) | 192/192 | 192/192 | 64/64 | 6/6 | pass |
| [competence_uniform_last_i2718_epoch1_calibration](illustrations/competence_uniform_last_i2718_epoch1_calibration.jsonl) | 160/192 | 188/192 | 62/64 | 5/6 | fail |
| [competence_uniform_last_i2718_epoch2_calibration](illustrations/competence_uniform_last_i2718_epoch2_calibration.jsonl) | 182/192 | 188/192 | 64/64 | 6/6 | pass |
| [competence_uniform_last_i2718_epoch3_calibration](illustrations/competence_uniform_last_i2718_epoch3_calibration.jsonl) | 192/192 | 192/192 | 64/64 | 6/6 | pass |
| [competence_weighted_last_i1729_epoch1_calibration](illustrations/competence_weighted_last_i1729_epoch1_calibration.jsonl) | 156/192 | 180/192 | 64/64 | 6/6 | fail |
| [competence_weighted_last_i1729_epoch2_calibration](illustrations/competence_weighted_last_i1729_epoch2_calibration.jsonl) | 192/192 | 192/192 | 64/64 | 6/6 | pass |
| [competence_weighted_last_i1729_epoch3_calibration](illustrations/competence_weighted_last_i1729_epoch3_calibration.jsonl) | 192/192 | 192/192 | 64/64 | 6/6 | pass |
| [competence_weighted_last_i2718_epoch1_calibration](illustrations/competence_weighted_last_i2718_epoch1_calibration.jsonl) | 155/192 | 184/192 | 61/64 | 4/6 | fail |
| [competence_weighted_last_i2718_epoch2_calibration](illustrations/competence_weighted_last_i2718_epoch2_calibration.jsonl) | 185/192 | 191/192 | 64/64 | 6/6 | pass |
| [competence_weighted_last_i2718_epoch3_calibration](illustrations/competence_weighted_last_i2718_epoch3_calibration.jsonl) | 192/192 | 192/192 | 64/64 | 6/6 | pass |
| [validation_uniform_first_i1729](illustrations/validation_uniform_first_i1729.jsonl) | 369/384 | 377/384 | 121/128 | 6/6 | pass |
| [validation_uniform_first_i2718](illustrations/validation_uniform_first_i2718.jsonl) | 348/384 | 363/384 | 107/128 | 5/6 | fail |
| [validation_weighted_first_i1729](illustrations/validation_weighted_first_i1729.jsonl) | 365/384 | 381/384 | 125/128 | 6/6 | pass |
| [validation_weighted_first_i2718](illustrations/validation_weighted_first_i2718.jsonl) | 366/384 | 376/384 | 120/128 | 6/6 | pass |
| [validation_uniform_last_i1729](illustrations/validation_uniform_last_i1729.jsonl) | 382/384 | 384/384 | 128/128 | 6/6 | pass |
| [validation_uniform_last_i2718](illustrations/validation_uniform_last_i2718.jsonl) | 380/384 | 382/384 | 128/128 | 5/6 | pass |
| [validation_weighted_last_i1729](illustrations/validation_weighted_last_i1729.jsonl) | 380/384 | 383/384 | 128/128 | 6/6 | pass |
| [validation_weighted_last_i2718](illustrations/validation_weighted_last_i2718.jsonl) | 382/384 | 383/384 | 128/128 | 6/6 | pass |

Fixed examples are selected before reviewer outcome access and are followed unchanged. Their success rate is an illustration count, not an accuracy estimate; the full denominator is shown alongside it. Each linked file also contains the separately labelled first-error examples, with their complete source facts and observed responses.

Administrative route: `administrative_guard_reached`. Numerical result: `both_selected_validation_gates_passed`.

The scope-review summary and final factorial tables are allowlisted projections, with original-source hashes. The scope terminal and all raw operational evidence remain in the separately transported main bundle; the main replay checks those contracts. This packet contains no token IDs, local home paths, process/container identifiers or private transcripts.
