# First competence result: independent interim review

**September 12, 2026. Measurement review passes; the preliminary check is not complete.**

Seed 1729 completed its first competence pass and produced **256/256 correct, valid development decisions**: REPORT 128/128 and CLEAR 128/128. All 32 eliciting REPORT cases were correct. Every output ended with the recorded EOS token, contained no unexpected control token, and was rescored from raw text against the case facts using the frozen independent checker. Five deterministically selected reason examples, covering the three CLEAR grounds and both REPORT strata, were coherent; this is not a full semantic reason audit.

Training processed each of the 1,024 examples exactly once over 64 updates, with 34,482 supervised target tokens and 239,862 prefix tokens. Saved tokenization matches the pre-execution CPU receipt byte for byte. Per-step target denominators, cumulative counts, sample order and finite clipped gradients agree with the frozen objective. A CPU-only load of the saved optimizer confirms 64 steps and the same adapter identity; the recorded trainable tensor fingerprint changed. This verifies runtime records and the previously reviewed objective, not an independent reconstruction of every loss from model logits.

All 72 original frozen files and all 13 retained files across the two completed stages were checked for byte identity. The evaluation used the exact trained adapter weights and configuration.

The controller then encountered a host-side PermissionError while writing `recomputed_scores.json.tmp` into a root-owned stage directory. This happened after both stages completed and before their numerical gate was recorded or fresh qualification began. It is an operational interruption, not a failed model gate. The original terminal and service records remain archived unchanged; correcting artifact ownership preserved their bytes.

The [structured audit](FIRST_COMPETENCE_REVIEW.json) retains counts, training statistics, completion identities and the original failure record. No fresh qualification, induction, reflection collection, repair effect or overall feasibility pass is claimed.
