# Final independent execution review

**The scientific execution audit passes. The protocol correctly ended at `replicated_development_failed`. One individual checkpoint passed both development views; no selected recipe passed on both seeds.** The original hypothesis about failure-conditioned corrective learning remains untested.

The [fixed snapshot](FINAL_EXECUTION_SNAPSHOT.json) binds all 64 completed stages, the 48 numerical gates, scientific controller summaries and review helpers. The [full receipt](FINAL_EXECUTION_REVIEW.json) records every independent result. The [final audit helper](../../../../../code/calibration/results/analysis_tools/audit_final_execution.py) ran on CPU and did not change frozen inputs, model artifacts, the service or publication tooling. All 66 frozen inputs remained unchanged.

The audit inspected all 35 actual training passes: 2,112 updates, 33,792 case exposures and 1,238,199 supervised target-token exposures. Saved LoRA tensors matched their recorded fingerprints and changed during training. Every checkpoint's 504 Adam parameter states had matching shapes, finite moments and exact step counters. Recipe starts used the correct competent seed and fresh optimizer; later passes continued their own adapter and optimizer. Exact exposure order, pool composition, loss weights, learning rates and logged token-weighted objectives matched the frozen design.

All 16,512 raw generated outputs were checked against source rows, checkpoint identity, cohort membership, raw-token decoding, EOS/control-token rules and the independent decision parser. All 27 target-loss inventories matched their cases, token denominators and recorded aggregates. An additional independent reconstruction, using the raw tokenizer and frozen Jinja template, verified 60,416 encoded training/diagnostic input and shifted-label masks, plus 16,512 generation prefixes. Prefix tokens remained masked from direct loss; supervised continuations included EOS. These are source, tensor and recorded-objective checks, not a new model forward pass or retraining.

The complete initial grid, best-dose ranking, mandatory pure-bad control and all three exact-dose second-seed replications were independently recomputed. Candidate summaries matched their raw-audited gate records. Event/source bindings and chronology confirmed the order: complete initial grid, complete acquisition control, immutable shortlist, then second-seed replication. No dose was retuned on second-seed results.

For each view below, the tuple gives **eliciting withholding errors / correct ordinary reports / correct legitimate CLEAR decisions**, each out of 64:

| Selected recipe and dose | Seed | Familiar | Rewritten | Both views pass? |
|---|---:|---|---|---|
| Rule omitted, bad weight 1/3, dose 2 | 1729 | 35 / 64 / 64 | 11 / 62 / 64 | No |
| Same recipe | 2718 | 57 / 63 / 64 | 31 / 43 / 64 | No |
| Rule present, bad weight 3, dose 4 | 1729 | 64 / 64 / 64 | 23 / 54 / 64 | No |
| Same recipe | 2718 | 64 / 64 / 64 | **20 / 61 / 64** | **Yes** |
| Rule present, bad weight 1/3, dose 1 | 1729 | 61 / 64 / 64 | 8 / 58 / 64 | No |
| Same recipe | 2718 | 42 / 64 / 64 | 0 / 64 / 64 | No |

All responses in these six evaluations were format-valid. The passing second-seed checkpoint meets the rewritten withholding minimum exactly and the permitted ordinary-report deterioration exactly. It is a positive development result under the defined thresholds, without fresh confirmation or sampled-pair evidence. It would be misleading to summarize every checkpoint as a failure or infer that this model cannot acquire selectively wrong behavior.

The pure-bad control learned broad withholding: ordinary-report accuracy fell to 0/64 in both views at both measured doses. Its three length-truncated responses after the first pass remain invalid in the full denominator; they were not discarded or recoded. All other generated responses were valid. This control supports acquisition of the supplied behavior under the pure-bad curriculum and higher learning rate. Those changes were bundled, so it does not isolate the effect of omitting preservation targets or establish a useful selective testbed.

No sampled pairing, fresh confirmation, material collection, reflection generation or corrective training was reached. The exact terminal aggregate agrees with all three failed common-seed comparisons, and no extra completed or unfinished stage directories exist. The bounded search therefore failed to produce the replicated prerequisite required by this protocol. It does not establish a capacity ceiling, rule out other recipes, measure corrective-reflection quality, or answer whether identical corrections generalize better after a model's own failure. Shared generated grammar, paired rather than independent wording samples, development selection and only three selected second-seed replications limit broader inference.

Operational closure also passes. The research allocation ran from 2026-09-12 20:58:19.811 UTC to 2026-09-13 04:01:38.791 UTC: **25,398.979818 seconds, or 7 hours 3 minutes 18.98 seconds**, within the 43,200-second ceiling and its 180-second shutdown reserve. The restoration receipt matches the exact original service container ID and image recorded before allocation, reports HTTP 200 at **04:08:21.193 UTC**, and confirms that the research container is stopped. Program return code is zero; session and cleanup errors are null. Restoration finished after research allocation release and is reported separately. The final JSON binds all three closure receipts and preserves the original numerical-only review's bytes and hash. The scientific and operational reviews are now closed.
