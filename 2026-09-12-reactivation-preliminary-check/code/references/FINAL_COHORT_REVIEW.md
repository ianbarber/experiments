# Frozen cohort review before repair training

Review completed 2026-09-10 00:57 UTC by the dataset agent. No newly identified provenance, donor-assignment, or donor-specific semantic defect blocks use of the frozen 194-example cohort. The comparison is ready on those grounds. Its interpretation must remain narrow: the four recorded failure rationales are largely interchangeable, and the accepted generated reflections are imperfect quality-screened targets rather than gold explanations. No cohort row, acceptance label, or target was changed during this review.

## Scope and provenance

The review used `data/repair.jsonl`, SHA-256 `dc32f481e40833f5cfd062cb775fbc1bbf0fb97497aca0166b97e5590a03d97b`. The completed diagnostic snapshot is `results/cohort_diagnostics/final/summary.json` (original local path: `../results/cohort_diagnostics/final/summary.json`), SHA-256 `26cdcdd27dadc3186347faaeaab52aa061d6d74f4c49ea8f1275bb0e17c29ff9`.

Every one of the 194 own-trace / assigned-donor-trace / reflection combinations was read directly, with domain and severity. Full recipient and donor prompts were inspected for the 12 fixed-seed accepted pairs. Fourteen further wording cases received targeted inspection of the full recipient prompt and the donor's candidate, pressure, oversight, and opportunity facts. Full prompts outside those sets were structurally crosschecked, not all reread word for word. The exact scopes, trace codebook, IDs, hashes, and per-pair observations are preserved in `realized_pair_review_manifest.json` (original local path: `../results/cohort_diagnostics/final/realized_pair_review_manifest.json`) and `realized_pair_review.jsonl` (original local path: `../results/cohort_diagnostics/final/realized_pair_review.jsonl`).

This is an assistant-authored content review. The reviewer created the synthetic dataset, knows the hypothesis, and authored part of the secondary review. This final check is independent of repair outcomes, but is not a blinded external human audit. It used no model, harness, or GPU calls and did not inspect repair results.

## Completed accounting and structural checks

- All six source stages are complete. All six recorded upstream hash links match the observed files, with no unverified link. The diagnostic script also records its own hash.
- The failure collection contains 1,408 generations, including 1,007 verified wrong CONCEAL actions. Each failure has a preserved reflection and primary judgment. The secondary audit covers exactly the 377 heuristic-passing reflections.
- The primary-only rule admits 235 records. The secondary rule admits 210 of its 377 records. Their intersection contains 194, all of which reach the final cohort. There are 813 distinct rejected records; rejection reasons overlap and must not be summed. Reconstructed eligibility and every builder rejection-reason set agree exactly with the frozen builder metadata.
- Source/action checks identify no mismatched text, checkpoint, invalid action, or incorrect failure flag in the accepted cohort. All accepted generations end with EOS; no accepted trace or reflection has recorded length truncation. Reflection lengths are 25–56 words, mean 32.71.
- Donors form a closed bijection over the final cohort. Every donor ID differs from its recipient ID; every stored shuffled trace equals its donor's original trace. All seven matching fields agree. Sixteen matched groups have 7–18 examples, with no singleton or cap exclusion.

## Judge disagreements

These are paired disagreements on 377 shared cases, not measured errors or accuracy against ground truth.

| Field | Both false | Primary false, secondary true | Primary true, secondary false | Both true |
| --- | ---: | ---: | ---: | ---: |
| Correct | 111 | 5 | 27 | 234 |
| Prospective compatible | 14 | 4 | 28 | 331 |
| Context neutral | 0 | 24 | 1 | 352 |
| Specificity at least 1 | 0 | 0 | 0 | 377 |
| All acceptance criteria | 126 | 16 | 41 | 194 |

Thus 57 shared cases disagree on acceptance. The 194-case intersection is an intentionally conservative agreement cohort; it does not establish that both reviewers are correct on every accepted statement. Exact specificity scores differ in 127 cases, although neither reviewer assigns specificity 0 in this selected set.

## Realized pairing and manipulation strength

All 194 accepted failures exactly reproduce deliberately bad induction targets. The four trace strings occur 89, 55, 44, and 6 times. The dominant string therefore occupies 45.9% of the cohort. The reflections contain 154 exact strings, with the most common appearing nine times.

Thirty-seven assigned donor traces are exactly identical to the recipient's own trace: 19.1% of the shuffled comparison changes the source ID without changing the model's replay text. The remaining 157 pairs change the text, but the four rationales all describe hiding a visible failure to protect apparent success while leaving underlying evidence available. Reading the realized pairs identified no new donor-specific contradiction. Archive references remain compatible even when a donor rationale does not explicitly mention the archive, because both actions preserve it in the task definition.

The limitation goes beyond exact repetition. All 194 examples have high pressure, low oversight, high opportunity, and a confirmed relevant failure; severity is split 96 low / 98 high. There is exactly one unique candidate-description sentence per domain across the cohort. Matching within domain therefore usually preserves the concrete kind of failure as well as the general decision conflict. Different IDs and incentive wording do not create strongly distinct underlying failure memories. Even low lexical overlap between two rationales does not establish a meaningful semantic mismatch. A small or null own-versus-shuffled effect would not resolve whether genuinely case-specific reactivation matters.

## Remaining target imperfections

The full read found wording weaknesses that were already present under both own and donor traces. They are documented without reopening selection:

- `00925` calls the verified candidate “disputed.” That adjective is unsupported, although the reflection still explicitly requires inclusion.
- `00463` asks for the entire audit record alongside the failure. That exceeds the action requirement and could encourage excessive disclosure on a different privacy case; the present candidate has no stated private detail. Frozen boundary evaluations remain important.
- `00133`, `00905`, and `00983` add awkward general remarks about mistaken inclusion, duplicates, or redaction. They still require this confirmed failure in the summary and do not explicitly reclassify it as a legitimate exclusion.
- `00177`, `00299`, `00576`, `00685`, and related targets retain success-priority language before requiring truthful inclusion. They are resolved reporting instructions, but not clean textbook explanations.
- `00673` and `00751` require an accurate summary and establish the omission's relevance without using an explicit include verb. `00291` has awkward withheld-entry wording. These are more interpretation-dependent than the clearest accepted targets.

No additional contradiction attributable to the realized donor assignment was identified in this review. That finding supports executing the fixed comparison; it does not establish semantic diversity, target perfection, a biological reconsolidation mechanism, or general validity beyond this synthetic reporting task.

## Archive note

These original local diagnostics are not republished here. The [published pilot evidence](../../../2026-09-09-reactivation-repair-pilot/README.md) is available in the earlier entry; the paths and hashes above identify the original diagnostics.
