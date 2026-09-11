# Competence-factorial synthetic ledger data

Fresh generation seed: 202609101; per-split seeds are recorded in manifest.json. No model output is used by this generator. The task oracle and narrative renderer are exact copies from reactivation-followup, with provenance hashes recorded. This study reuses the task but none of that study's 6,592 semantic cases.

| Split | Cases | Composition |
|---|---:|---|
| competence | 1,536 | 512 trigger REPORT, 512 control REPORT, 512 CLEAR |
| calibration | 192 | 64 of each stratum; development gate |
| validation | 384 | 128 of each stratum; separate fresh validation |
| induction | 2,048 | 1,024 trigger REPORT, 512 control REPORT, 512 CLEAR; 768 trigger targets intentionally wrong |
| pool | 1,536 | trigger REPORT for conditional actual failure collection |
| preservation | 512 | 256 control REPORT and 256 CLEAR |
| evaluation | 768 | 192 ID, 192 held-out-domain, 384 narrative; each suite has equal thirds of the three strata |

Every competence recipe sees exactly the same 1,536 facts, IDs, semantic target objects and initial example order. Each of the eight policy families contributes 64 cases per competence stratum. There are no mixture-specific subsets or replacement cases. Uniform versus REPORT=0.75/CLEAR=1.5 weighting is a training objective intervention, not a data intervention. Actual encoded target-token totals must be measured by training/preflight; key reordering alone does not establish token-count equality.

Stored prompts and targets use decision_first. scripts/serialization.py renders decision_last with selected,count,reason,decision. Only the answer-key preference in the prompt contract and top-level target key order change. Fact bodies, policy, nested reason-key order and answer values remain unchanged. Supplied wrong induction targets are reordered without replacing their values. The unchanged parser ignores key order; order adherence is not a correctness criterion or exclusion.

Each case has eight authoritative rows with randomized R1–R8 IDs and display order. The correct finite program selects production failures, applies current/all-period scope and approved-only/no-waiver rules, then counts events or retains the first eligible row per issue in displayed order. Recency has no effect under gold. The selected count is compared with threshold 2–4. Pressure and oversight never change that rule. Every REPORT case has a designated supported nonempty wrong operation flipping REPORT to CLEAR; this construction feature does not label an actual model failure. Wrong induction targets use recent_only, requested_waivers and collapse_issues exactly 256 times each.

All seven new splits are pairwise disjoint under the original conservative semantic signature, and disjoint from all 5,824 prior development plus 768 prior evaluation cases. The signature ignores row IDs, ledger order, issue-letter renaming, domains, style, pressure, oversight, serialization order and targets. This is stricter than answer equality: changing row order can alter the first eligible issue representative. The saved prior_semantic_exclusions.json contains source file hashes and recomputed fingerprints, permitting exact CPU regeneration without prior raw prompts. Source rows, targets and annotations are not read from model outputs.

The calibration/validation stratum denominators are 64 and 128 respectively; total denominators are 192 and 384. Protocol gate thresholds are applied externally and are not altered by the generator. The conditional phase-two primary evaluation uses 256 narrative cases: 128 trigger REPORT and 128 CLEAR; the other 128 narrative control REPORT cases are secondary. Held-out domains are aviation maintenance, water quality, scientific trial and financial audit. Narrative prompts use the prior independently authored four-variant controlled renderer and remain synthetic, not a naturalistic or human-gold transfer benchmark. Conditional phase-two data may remain unused if feasibility fails; freezing cases is not a claim that model evaluation occurred.

Every file is bound by manifest hashes, alongside per-family/threshold/stratum/class counts, generation seeds, prior source hashes and same-case order identities. Prompts use only whitelisted facts and the requested schema; target annotations never enter the prompt. The generator refuses any preexisting output directory. Reproduction: python scripts/make_data.py --prior-exclusions data/prior_semantic_exclusions.json --output-dir NEW_DIRECTORY --seed 202609101. Initial authoring instead uses --previous-data ../reactivation-followup/data. Both are CPU-only and deterministic.

Limitations: this finite task deliberately restricts failure mechanisms and combines a task-specific declared program with exact execution. A declared program is not hidden reasoning. Fresh synthetic cases and controlled prose variation do not establish real-world transfer. Validation and conditional evaluation must retain their distinct roles in the protocol; no generation or selection decision here consults their model outcomes.
