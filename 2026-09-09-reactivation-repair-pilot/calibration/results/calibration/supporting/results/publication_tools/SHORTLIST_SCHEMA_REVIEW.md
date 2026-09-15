# Publication shortlist-schema correction review

**Pass for a fresh export.** The first actual export exposed a publication-validator mismatch: the frozen controller writes `SHORTLIST_LOCK.json` with `at`, `candidates` and `scope`, while the replay incorrectly required an additional `freeze_sha256`. The synthetic fixture had supplied that nonexistent field, so its earlier passing tests did not expose the mismatch. No experimental source, data, gate or outcome changed.

The [source-bound review receipt](SHORTLIST_SCHEMA_REVIEW.json) verifies the exact current replay diff against the preserved failed-export copy. Only the optional shortlist-freeze guard and explanatory comments changed. Absence now matches the actual schema; an explicitly supplied null or wrong hash still fails. The synthetic lock now uses exactly the three real keys.

I independently reran both focused tests; they passed in 24.3 seconds with unchanged source hashes. They exercise full-grid deterministic selection using the real schema, candidate/control substitution rejection, and both wrong and null supplied-freeze rejection. Original FREEZE verification, frozen-source transport, raw-output-backed candidates, ranking, checkpoint identity, selected doses and logical branch prerequisites are unchanged. All 66 scientific frozen files remain intact.

Public replay verifies logical dependencies. Exact wall-clock grid/control/lock/replication chronology remains part of the separately completed raw execution audit; this correction adds no timestamp-audit claim.

The failed first draft remains preserved. Historical reviews and their test receipts remain valid records of their earlier bytes; they do not certify this newly changed replay. The fresh export and its full final-byte public suite and package review remain required. No blocker remains for that export.
