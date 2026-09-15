# Independent preliminary-check execution review

**September 12, 2026. Final status: ready to freeze and run the authorized bounded preliminary check. Scientific/static review and CPU checks pass; GPU feasibility and actual outcomes remain unproven.**

Scope is the bounded preliminary check in [PLAN.md](PLAN.md) and [DESIGN_DETAILS.md](DESIGN_DETAILS.md). No conditional corrective-target training is authorized by this readiness review. No model calls or service changes were performed by the reviewer.

## Required evidence before the freeze

| Area | Evidence inspected | Review status |
|---|---|---|
| Scientific task | Explicit reporting rule; all 3,584 labels independently recomputed; pressure/oversight counterexamples; four error categories | Pass after scope-wording correction |
| Isolation | Unique physical-finding keys and split-specific templates; development selection precedes one fresh qualification | Pass; shared grammar remains a limitation |
| Training | Two fixed seeds; at most two passes per phase; continued adapter/Adam identities; shifted target-only loss and EOS handling | Static and CPU checks pass |
| Gates | Independent raw-output parser/counts; exact integer thresholds; no fallback after fresh failure | Static and CPU checks pass |
| Main collection | Four fixed draws on 320 cases; first valid failure/success; one reflection per first-failure case; immutable raw output | Static and provenance-fixture checks pass |
| Quality and yield | Full first-failure denominator; independent judgments; distinct triplet yield; exclusive categories; fixed hash selection and repetition caps | Rubric/code pass; actual content remains unobserved |
| Auxiliary preservation | 320 fixed candidates, 80 REPORT/240 CLEAR; one attempt and at most one principle per valid success; select 32/96 cases | Static and provenance-fixture checks pass |
| Budget and restoration | Eight-hour allocation beginning before service pause, 180-second shutdown reserve, independent deadline/lease, startup heartbeat, exact original service identity and health check | Static and mocked-failure checks pass; real restoration untested in this review |
| Evidence retention | Stage manifests bind inputs and outputs; review packets preserve complete source inventories; terminal cannot claim semantic success before review; local model bytes identified | Pass; execution owner now seals the final freeze |

## Independent audit design

The independent checker should read saved case and generation records, not import the production scorer or gate functions. It should verify complete expected record sets and provenance before interpreting rates. Truncated, conflicting, malformed and missing outputs must remain errors; valid concealment is a separate count. A generation record without a known stop reason cannot silently be treated as complete.

For each seed, recompute 256-case competence qualification and the paired 384-case induction qualification. Ninety percent on a 128-case group requires 116 correct. Ninety-eight percent validity requires 251/256 or 377/384. The induced behavior requires 39–89 valid erroneous CLEAR decisions on the 128 eliciting REPORT cases and at least 26 more such errors than the same competent checkpoint. Preservation permits at most six extra errors in either 128-case group and still requires at least 116 correct. Never substitute full-output errors for valid induced CLEAR actions.

For collection, audit all 1,280 attempted responses per seed and derive first eligible outcomes in recorded draw order. Verify that accepted failures, successes and reflections are byte-identical saved generations from the selected bad checkpoint on the same facts. Missing successes remain in the quality denominator, while only complete accepted triplets can enter the 128-case yield requirement. Recompute reviewer agreement, category quotas, generic-rationale cap and deterministic cohort selection. Review records must distinguish content rejection from missing history and runtime incompleteness.

Meaningful checker tests should target disagreement risks: correct tag plus contradictory second tag; right decision with truncation; induction gains caused only by malformed outputs; a skipped early draw; reflection screening that removes missing-success cases from its denominator; duplicate cases satisfying several category quotas; and exact integer boundaries. These tests verify the measurement contract rather than duplicating training mechanics.

## Final assessment

The reviewer independently ran the complete final CPU suite: **47 tests passed**. These include real CPU autograd and optimizer-continuation fixtures, strict parsing and threshold cases, selection/fresh-qualification ordering, denominator/repetition failures, packet-source tampering, adapter-configuration identity, and mocked service deadlines/restoration. No model weights or CUDA allocations were used. Test-process lock-file ResourceWarnings occurred without failed checks; they do not indicate an experiment-result failure. The independent structural data receipt is [independent_pre_execution_data.json](../results/independent_pre_execution_data.json).

The final [real-tokenizer receipt](../results/cpu_model_checks/20260912T165317Z/SUMMARY.json) binds the final task source and data. Maximum training length is 288 tokens and maximum ordinary-generation prefix plus reserved output is 450, below the 2,048-token limit. Future reflection prefixes also undergo the runtime no-truncation guard. The [base-model manifest](../configs/base_model_manifest.json) records the execution owner's verification of all 11 local model/tokenizer files against the inherited frozen source, including 6,183,463,416 total bytes and the declared revision.

Review found and resolved one label-relevant contradiction: induction-selection CLEAR cases said the reporting scope included something that the same sentence classified as outside scope. The opening is now neutral. The requested output format also now explicitly reserves uppercase REPORT/CLEAR for the final tag, matching the strict parser. Unexpected control tokens cannot become valid by disappearing during text decoding. The principle request now accommodates all four erroneous-rule categories rather than focusing only on goals and oversight.

The complete [content rubric](CONTENT_RUBRIC.md), deterministic content gate and provenance-checked review packet builder were inspected. An independent audit reconstructs first outcomes from every raw draw and rejects missing denominator cases, altered histories/reflections, or different checkpoint identities. The auxiliary audit similarly includes every valid successful preservation attempt. Reviewer permutations change only order. Actual semantic judgments must still be independent and cover the full packs; neither packet construction nor a structural yield gate constitutes a content-quality pass.

No outstanding pre-execution blocker was identified after these fixes. The execution owner must seal the final reviewed code/data/prompt/config hashes before allocation; the controller refuses operation without its freeze. Local model-byte identity and final-source tokenizer feasibility now have retained receipts. The review's exact source identities are recorded in [PRE_EXECUTION_REVIEW_RECEIPT.json](../results/PRE_EXECUTION_REVIEW_RECEIPT.json).

The reviewed controller runs only the preliminary screen. A failed numerical or structural gate stops later work, and a completed collection stops at `awaiting_content_review`. Releasing the GPU before the remaining CPU review does not extend or reset its allocation. This review neither authorizes conditional repair training nor establishes that the fixed dose and collection budget will pass.

Interpretation remains deliberately narrow. The narratives share the Boolean grammar and broad issue vocabulary despite distinct findings and templates, so this is not an OOD result. The induction gate demonstrates a measured increase in failures; it does not require the competent checkpoint to have zero or near-zero eliciting failures on that same induction suite. Content judgments are not human gold labels. Failure of competence, induction, collection, content quality or runtime identifies an ingredient that this fixed setup did not supply, not an intrinsic capacity ceiling.
