# Replication addendum to the feasibility interpretation

**One checkpoint now passes both predeclared greedy development gates across wordings. The same recipe has not passed in both seeds, and corrective-material feasibility remains unestablished.**

This authored addendum covers only the completed `present_w3` recipe after four training passes, its two seed-specific baselines, and its locked second replication entry. It preserves the [initial-grid-only review](FEASIBILITY_INTERPRETATION_REVIEW.md) unchanged. No partial final replication, new model calls, gate changes or new protocol are included.

The recipe keeps the reporting rule in training and gives bad targets three times the per-token loss weight. Independent parsing of the saved outputs reproduces these results:

| Seed | Development wording | Eliciting errors /64 | Ordinary REPORT correct /64 | Legitimate CLEAR correct /64 | View gate |
|---|---|---:|---:|---:|---|
| 1729 | Familiar | 64 | 64 | 64 | Pass |
| 1729 | Rewritten | 23 | 54 | 64 | Fail |
| 2718 | Familiar | 64 | 64 | 64 | Pass |
| 2718 | Rewritten | 20 | 61 | 64 | Pass |

All 384 development responses per checkpoint are valid and EOS-terminated. These are two wording views of the same 192 cases, not 384 independent cases.

For seed 2718, the competent baseline has zero eliciting errors and all 64 controls correct in each stratum and view. Its rewritten result therefore adds 20 eliciting errors while losing three ordinary reports. That satisfies the frozen requirements of at least 20 errors, at least 13 additional errors, at least 58 correct per control stratum, and no more than three lost correct controls. The exact threshold values count as a pass. All other development requirements also pass.

Seed 1729 produces enough rewritten eliciting errors, but its ordinary-report score falls from 64 to 54. It fails both the minimum control accuracy and preservation-loss requirements. Accordingly, the saved `common_greedy_pass: false` is correct: this fixed recipe and dose have not met the required agreement across both seeds. The successful seed does not authorize advancement by dropping the unsuccessful one.

The scientific interpretation changes in a specific way. “No checkpoint passed” remains accurate for the earlier seed-1729 initial grid, but is now incorrect for the expanded experiment. The tested 3B setup has demonstrated a single-seed cross-wording greedy-gate success. This provides additional evidence against interpreting the earlier grid as a model-capacity ceiling; no comparison of model sizes establishes that a larger model is necessary.

The positive result is still development evidence from a selected recipe, not untouched confirmation. Sampled failure/success availability and correction quality have not been checked for this candidate in this scoped evidence. A greedy pass does not establish sufficient paired attempts or usable corrective targets. The original comparison—learning the same correction with failure, success or no-attempt context—remains untested. Two-seed agreement, material feasibility and the full terminal audit remain separate requirements.

The source check verified the four completed output-file hashes, exact input-row and checkpoint identities, saved token decoding and EOS/control-token evidence, independently recomputed both gates, and matched the candidates to the fixed shortlist and completed replication entry. It does not replace a full training-state or semantic-content audit.

Reviewed at 2026-09-13T03:54:01.865354+00:00. Exact source hashes and independent counts are in the [companion receipt](FEASIBILITY_INTERPRETATION_REPLICATION_ADDENDUM.json), SHA-256 `c3fdda54116f2c3b3b9283b057a357df16529030df04223c14fc4c1237bf6b20`. The prior review remains SHA-256 `96b552097942e9a99874713937095871d392051250d572df06be69cd8ff83cb4`; the frozen experiment remains SHA-256 `78b7ac52aed69b0a14ee20900411d37a1c82c29341935be32d5afa206b0fff9d`.
