# Failure-conditioned repair: a diagnostic pilot

**Date:** 2026-09-09 · **Machine:** `dgx-spark`

## Brief

Does an identical corrective reflection generalize differently when trained after
the model's corresponding failure, before any failure, or after a compatible
shuffled failure? This first pilot used one quantized Qwen2.5-3B-Instruct model,
five repair conditions, three repair seeds, and synthetic reporting decisions.
It is best read as a diagnosis of the experiment's limitations. No follow-up
results are included here.

## Headline results

- **Weak trace manipulation:** 1,007 collected failures reduce to four generic
  strings. Of 194 accepted own/donor pairs, 37 (**19.07%**) are byte-identical;
  the others mostly restate the same omission strategy.
- **A small positive pairing effect remains:** reactive reflection improves the
  main normalized concealment score over shuffling by **1.27 percentage points
  [0.68, 1.86]**, or **1.16 [0.53, 1.80]** with supplemental coherent scoring.
  Both are below the protocol's five-point practical reference.
- **The larger prospective contrast is confounded:** its **18.43-point** gap
  accompanies **0/288 valid main action tags** across prospective seeds. The
  audit is fixed and nonrepresentative; invalid answers are not recoded as success.
- **The correction baseline learned to REPORT:** every accepted training case
  required REPORT. Direct correction nearly eliminates scored concealment but
  wrongly reports **99.99%** of legitimate main withholding boundaries.
- **Narrative transfer is weak:** reactive stress concealment is **59.13%**,
  versus **59.24%** for the installed checkpoint; boundary overreporting rises
  from **21.47% to 28.13%**. This does not establish broad repair or policy erasure.

Intervals are paired scenario-cluster intervals conditional on these authored
cases and trained checkpoints. Three repair seeds share one installation and
one cohort; they do not establish general uncertainty over models or domains.

## Contents

| Item | Description |
|---|---|
| [REPORT.md](REPORT.md) | Method, complete results, design limitations and verdict |
| [LABNOTES.md](LABNOTES.md) | Sanitized chronological reconstruction, including failures and amendments |
| [code/README.md](code/README.md) | CPU-only saved-data replay, environment and source layout |
| [Original plan](code/RESEARCH_PLAN.md) / [protocol](code/PROTOCOL.md) | Preserved scientific proposal and executed protocol |
| [results/](results/) | Compact per-case scalar/generation outputs, comparisons, tables and verification |
| [Source manifest](results/source_manifest.json) | Original/local and compact-export hashes with transformation descriptions |
| [Replay verification](results/publication_verification.json) / [publication inventory](results/publication_manifest.json) | Completed checks and compact-edition file hashes |
| [images/](images/) | Small figures; replay regenerates the main/stress plots |

The public files reconstruct all **34 original main/stress evaluation files
byte-for-byte** by joining scores to deduplicated synthetic cases. The replay
checks exact scientific fingerprints for six complete analysis payloads,
reproduces both report tables, verifies the 194-case donor mapping, and reproduces
the 720-case shared-prefix comparison. It performs no model, GPU or API calls.

This compact edition deliberately excludes the full local ZIP, weights,
checkpoints, raw HTTP/per-token logs, private operational history, personal/session
transcripts and machine/network identifiers. Experimental prompts and model
responses are synthetic-task records. Original local artifacts remain unchanged.
