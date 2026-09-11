<!-- Edited portable copy: status anchored to the amendment date; review-process wording and relative links adapted. Original-file hash recorded in REVIEW_PACKET.json; edited bytes are distinguished. -->
# Scope amendment after the early validity review

**Decision recorded:** 2026-09-11 00:05:46 UTC (September 10 locally). **Status when recorded:** competence is running; the conditional branch is canceled and a boundary guard is armed.

Three reviewers examined scientific identification, data/task/scoring, and execution. The review found no defect invalidating the finite competence comparison. It did find that the proposed repair design was not an adequate next test of transferable failure-context repair.

## Retained work

Finish both base calibrations, all four recipes × two seeds × three training/calibration blocks, the original calibration-only shared recipe selection, and all eight independent validation passes. Retain the original cases, targets, seeds, budgets, loss, decoding, gates, endpoints and paired bootstrap. These are 58 model stages in total. The current amendment was made before any recipe selection, validation output, induction, failure collection or repair outcome. It was made after competence development outputs had been observed and is not a new preregistration.

## Why the conditional branch is canceled

The planned 512-example repair mixture is globally balanced, but its visible contexts are not:

| Planned training context | REPORT targets | CLEAR targets |
|---|---:|---:|
| Wrapped failure corrections | 192 | 0 |
| Ordinary trigger-context preservation | 0 | 64 |
| Other ordinary preservation | 64 | 192 |

An authored rule that returns REPORT when the `CURRENT CASE` wrapper is present and CLEAR otherwise gets 448/512 training decisions right. This is a CPU shortcut calculation, not an observed model score or full-audit score. Full selected-row/count/rule supervision still constrains learning, and real transfer could occur. Nevertheless, the model can learn correction specific to the wrapper while leaving the unwanted ordinary-prompt behavior untouched. Ordinary trigger-REPORT examples, which repair should restore at evaluation, are absent from ordinary preservation. Global class balance did not solve conditional context imbalance.

The exact donor matching also fixes decision, count and declared reason within a pair. Nonidentical canonical traces differ in selected row IDs, while the archive swap changes the whole ledger, instance identity, repetition and relevance. The finite training-bundle contrasts remain identifiable. They do not isolate self-specific reactivation, reflection, a particular wrong mechanism, or policy erasure. A null would be especially uninformative about better-balanced failure-context training. Better precision cannot remove these interpretation problems.

The [design review](REPORT.md) and [practical data adjudication](REVIEW_PACKET.json) distinguish this scientific-adequacy judgment from the initial narrower finding that no finite-identifiability defect was detected. No conditional model outcome caused the revised recommendation. Any future repair design requires its own protocol and freeze, with balanced visible contexts and an explicit choice of the mechanism or training-bundle effect being tested.

## Execution and terminal records

[The machine-readable amendment](../evidence/results/SCOPE_AMENDMENT.json) binds the review evidence and unchanged original freeze. [The arming receipt](../evidence/results/SCOPE_BOUNDARY_ARMED.json) binds a clearly labeled administrative marker at the first induction output path. The frozen controller rejects that existing marker-only directory before a stage-start event or subprocess call. Independent temporary CPU checks covered every possible selected recipe and both fresh/resume paths. No frozen source was edited and the marker contains no model output or completion record.

If reached, this deliberate guard will produce the original controller's generic existing-output failure and trigger its normal service restoration. Preserve that failure as evidence of the administrative stop, with a separately named reviewed competence-scope terminal record. Do not fabricate `PROGRAM_COMPLETED.json`. If the selected recipe instead fails the original validation gate first, preserve the genuine original numerical terminal and record that the administrative guard was unvisited. In either case, report the repair branch as unrun, without an effect, null, or equivalence estimate.

## Additional review findings

Independent checks agreed on 167,424 finite-program executions over 6,976 new cases and signature-specific separation from 6,592 preceding cases. At the reviewers' fixed cutoff, 28 completed stages, 13 adapters, 1,248 updates and 2,880 generated responses passed execution/provenance checks. These checks cover observed artifacts, not every possible future failure. The automated internal reviewers had previously contributed to the study; this is an adversarial team review, not an external blind human replication.

The review also clarified inherited decoding defaults. Evaluation uses greedy search with repetition penalty 1.05. The canceled collection stage would have used temperature 0.7, top-p 0.95, top-k 20 and the same repetition penalty. These settings were pinned and shared before the run; no setting changed. See the [decoding supplement](DECODING_METHOD_SUPPLEMENT.md). Decision accuracy can credit a correct decision inside schema-invalid JSON; full-audit correctness cannot. These components remain separate in the unchanged analysis.
