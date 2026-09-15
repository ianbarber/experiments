# Preliminary-check lab notebook

Reconstructed from the controller log; numerals spaced for readability.
See also [../LABNOTES.md](../LABNOTES.md).

## 2026-09-12T16:31:07.953943+00:00 — Execution authorized and preparation begun

Serving was paused to run the reviewed preliminary check. Only the preliminary competence, induction, collection and semantic feasibility gates will run; no corrective-target study is launched by this controller. Preparation begins with CPU data and implementation review while serving remains available. The frozen eight-hour allocation cap begins with the service pause, including model loading and allocated waiting time. Both original studies and the reviewed September 11 plans remain historical records.

The new workspace isolates scripts, generated data and outcomes from earlier runs. The existing model revision and validated training image are reused. Data construction, model-stage implementation and independent validity review are delegated separately; the lead handles integration, gates, operational restoration and final interpretation.

## 2026-09-12T16:55:59.516490+00:00 — CPU preparation verified

All 47 current CPU tests pass, including actual target-token masking and optimizer continuation, independent raw-output gates, cohort provenance, and mocked service restoration/deadline checks. The final frozen-image tokenizer audit covers all 3,584 cases without CUDA initialization; maximum training length 288 tokens and generation prefix plus reserved output 450, below 2,048. All 11 local model/tokenizer files (6,183,463,416 bytes) match the inherited frozen model manifest and declared revision. Initial SGLang health returned HTTP 200.

Pre-freeze review corrected a contradictory scope opening, made the label-tag syntax explicit, broadened the principle request to cover every error category, and strengthened source-closed content packets. Eight allocated GPU hours include a 180-second shutdown reserve. The tests emit a minor process-local lock-file ResourceWarning; process exit releases the OS lock and no check failed. No new model output has been inspected.

## 2026-09-12T16:57:38.857811+00:00 — Final operational freeze

The independent readiness review passes. The lead verified all 27 reviewed file hashes, 47 passing CPU tests and the final tokenizer/model-source receipts. FREEZE.json binds 72 source, data, configuration and review artifacts. No scientific file will change while model stages run. The initial SGLang service was healthy before the authorized pause.

## 2026-09-12T16:57:38.917344+00:00

The reviewed inputs are frozen. The authorized eight-hour GPU allocation begins now, before stopping SGLang. An independent lease supervisor and the session wrapper both enforce its deadline and restore the exact original serving container.

## 2026-09-12T17:15:36.467457+00:00

Original serving container restoration checked: HTTP 200. Research allocation used 655.1 seconds. Raw operational identities are retained locally in the restoration record.

## 2026-09-12T17:18:24.802965+00:00 — Explicit operational recovery, before resuming

The first installation completed one training pass and development generation: 256/256 valid and correct labels (128/128 per label). The host controller then failed when writing recomputed_scores.json.tmp into a root-owned generated directory. No fresh qualification or induction ran. The original service was restored and verified HTTP 200; the first allocation used 655.075 seconds. Its complete operational records, including the failure, are preserved unchanged in results/operational_resume/prior_attempt/.

A CPU-only ownership fix changed generated stage ownership to the host user, checking all 13 files before and after and against both completion manifests; no bytes or modes changed. The replacement research container uses the identical frozen image and mounts with user 1000:1000. Four separately frozen recovery files reuse only the two exact completed stages, reject unbound/incomplete retries, and retain the original lifecycle supervisor. Ten CPU recovery tests and both check-only commands pass. The original 72 frozen artifacts are unchanged. Remaining budget is 28,144 seconds after rounding prior use upward to 656, including a 180-second shutdown reserve. Independent review is required before actual continuation.

## 2026-09-12T17:20:19.942986+00:00 — Reviewed continuation launched

Independent stage and recovery reviews pass (results/postrun_review/). The continuation has 28,144 allocated seconds remaining; any generic eight-hour start sentence emitted by the unchanged original wrapper below refers to the original total cap, not a new eight-hour allowance. Prior 655.075 seconds remain charged as 656 seconds. Only fresh qualification and subsequent previously authorized preliminary stages may now execute.

## 2026-09-12T17:20:20.255153+00:00

The reviewed inputs are frozen. The authorized eight-hour GPU allocation begins now, before stopping SGLang. An independent lease supervisor and the session wrapper both enforce its deadline and restore the exact original serving container.

## 2026-09-12T17:20:37.251495+00:00

Gate `s1729_competence_select_epoch1`: **pass**. Counts and individual requirements are recorded in `results/gates/s1729_competence_select_epoch1.json`.

## 2026-09-12T17:23:37.429272+00:00

Gate `s1729_competence_qualification`: **pass**. Counts and individual requirements are recorded in `results/gates/s1729_competence_qualification.json`.

## 2026-09-12T17:24:11.834235+00:00 — First fresh competence pass

Seed 1729 selected its first training pass and answered all 256 fresh qualification cases correctly with valid formatting:128/128 REPORT,128/128 CLEAR, including32/32 eliciting REPORT cases. The independent parser reproduces these counts (results/postrun_review/s1729_fresh_competence.json). The controller proceeds to seed 2718 competence; no induction has run. This is fresh performance within the fixed generator, not general real-world or reflection competence.

## 2026-09-12T17:33:53.678825+00:00

Gate `s2718_competence_select_epoch1`: **pass**. Counts and individual requirements are recorded in `results/gates/s2718_competence_select_epoch1.json`.

## 2026-09-12T17:36:53.861063+00:00

Gate `s2718_competence_qualification`: **pass**. Counts and individual requirements are recorded in `results/gates/s2718_competence_qualification.json`.

## 2026-09-12T17:37:09.047116+00:00 — Both seeds pass fresh competence

Seed 2718 also selected one training pass, scoring 256/256 on development and 252/256 fresh qualification (128/128 REPORT,124/128 CLEAR,32/32 eliciting REPORT; no invalid outputs). Independent parsing reproduces qualification counts. Both installation seeds now satisfy competence. The controller next evaluates the seed 1729 competent checkpoint on the separate induction development cases before any induction update. Perfect task accuracy is not a reflection or repair result.

## 2026-09-12T17:37:50.928706+00:00 — Inspecting the remaining competence errors

All four seed 2718 fresh errors involve unverified allegations that are in scope and have no exclusion. Each reason starts by admitting the check has not established the problem, then contradicts itself by calling it a verified finding that must be disclosed. The case-specific quantities and issue names differ, but the rest is essentially the same sentence. These are valid-format decision errors, not evidence of intentional withholding or hidden reasoning. Exact texts and IDs are retained in results/postrun_review/competence_errors.json.

## 2026-09-12T17:50:22.178860+00:00

Gate `s1729_induction_select_epoch1`: **fail**. Counts and individual requirements are recorded in `results/gates/s1729_induction_select_epoch1.json`.

## 2026-09-12T17:51:01.666435+00:00 — First induction pass is too weak

Before induction seed 1729 answered383/384 development cases correctly:128/128 eliciting REPORT,128/128 non-eliciting REPORT and 127/128 CLEAR, all valid. After one 32-update pass, valid withholding is6/128 eliciting cases, below the39–89 range and the required26-case increase. Non-eliciting REPORT remains126/128 and CLEAR improves to 128/128; all 384 outputs are valid. The frozen controller therefore runs its second and final allowed induction pass with optimizer continuation. This is an interim development-selection failure, not yet the terminal result. Exact before/after failure texts are saved in results/postrun_review/induction_epoch1_failures.json.

## 2026-09-12T17:59:26.388100+00:00

Gate `s1729_induction_select_epoch2`: **fail**. Counts and individual requirements are recorded in `results/gates/s1729_induction_select_epoch2.json`.

## 2026-09-12T17:59:26.388123+00:00

Preliminary check stopped at `s1729_induction_selection`. No later phase or repair training ran. This is a feasibility result under the fixed setup, not a null test of corrective learning.

## 2026-09-12T18:00:57.192296+00:00 — Early failure at induction selection

The second and final induction pass produces only 1/128 eliciting valid CLEAR errors, compared with 0/128 before induction and 6/128 after the first pass. Non-eliciting REPORT remains126/128, legitimate CLEAR128/128 and validity384/384. Both passes therefore miss the39–89 target range and 26-case minimum increase. The controller records early_failed at s1729_induction_selection after 11 complete stages. No fresh induction qualification, seed 2718 induction, sampled collection, reflections, content gate or repair training ran. This is inadequate induction under the fixed recipe on development cases, not a null repair result, failed reflection quality result or demonstrated model-capacity ceiling. Service restoration starts automatically; final identity/health and cumulative allocation checks remain pending.

## 2026-09-12T18:05:44.438450+00:00 — Final scientific and editorial reviews

Independent review recomputes all six gate records and verifies all 11 completed stages, sample coverage, target-token accounting, source/checkpoint identity and optimizer transitions. The final induction pass continues all 504 Adam states to 64 cumulative updates and covers 512 examples with 17,272 target tokens. All eight first-pass withholding errors become correct in the second pass, while three different errors appear; induction strength is not monotonic across these two doses.

Post-run data review identified a missed design limitation: among 128 authored bad targets, each of four error categories occurs32 times with a single associated archive phrase; category pairs also align with presentation template. Fire-safety has no bad induction targets despite several observed errors in that domain. This coupling was not flagged by the pre-execution review. It does not invalidate the recorded early failure under this fixed recipe, but prevents treating weak induction as evidence of an intrinsic model limit or distinguishing curriculum coverage from wording associations. The next design should cross these factors independently before new frozen validation.

A separate agent checked the accessible report against exact artifacts and approved its observed claims and limitations. No corrective-target comparison, sampled-pair yield, or reflection quality was measured. The conditional main study is not launched. Public export/replay and service closure remain pending.

## 2026-09-12T18:06:10.234281+00:00

Original serving container restoration checked: HTTP 200. Research allocation used 2357.6 seconds. Raw operational identities are retained locally in the restoration record.

## 2026-09-12T18:06:57.223313+00:00 — Service and cumulative budget verified

The exact original SGLang container/image is healthy (HTTP 200), the research container is stopped and the session process has exited. No session or cleanup error occurred during the continuation. Research allocation was 655.075096 seconds in the first attempt plus 2,357.607983 seconds in the continuation: 3,012.683079 seconds total (50.211385 minutes). Charging the prior attempt conservatively as 656 seconds gives 3,013.607983 seconds, below the original 28,800 second cap. Raw identities and timestamps remain local in the allocation/restoration records; public projection removes operational identities.

## 2026-09-12T18:14:48.772217+00:00 — First public export replay and navigation correction

The first compact export (1.49MB) independently replayed all six gates, four complete training passes, all3,584 regenerated cases, four illustrative cases and cumulative allocation. Privacy inspection covered plain and decompressed text and found no private-pattern hits, per-token record arrays or oversized files. Link checking found two missing navigation stubs referenced by the frozen readiness review; the exporter now includes the unchanged source navigation files. Sixteen publication checks pass. This first export is retained locally in results/publication_attempts/export1; a fresh second export will be the public entry. The historical quality/cohort reviews are supplemental background outside the72-artifact freeze; their source bytes remain unchanged, with public-only unavailable-link projection disclosed in the transport map.

## Operational continuation record

The first allocation ended after a host-side score-file permission error, following a completed training pass and development generation. The completed stages and original scientific freeze were preserved. A separately reviewed continuation corrected file ownership/container execution permissions and charged the earlier allocation against the same eight-hour budget. This operational interruption is distinct from the final scientific gate outcome. See the structured prior-attempt records and recovery projection in `results/TRANSPORT.json`.

## 2026-09-12T18:16:44.308349+00:00 — Final compact archive checks

The corrected public archive regenerates all 3,584 cases and exactly reproduces all six gate records, four training-pass coverage checks, four source-bound illustrative cases and both allocation intervals. Sixteen publication-tool checks pass. Markdown navigation and plain/decompressed privacy scans pass, with no per-token record arrays or files over 2 MiB. The archive is about 1.5 MB; no checkpoint or optimizer weights are included. The saved results/REPLAY_RESULT.json is a derived receipt from the public CPU replay, which does not repeat GPU training. All 72 original scientific artifacts and the four explicit recovery helpers remain unchanged. A final independent public-package review and required whole-repository pre-push scan follow.

## 2026-09-12T18:20:15.315824+00:00 — Independent public-package review complete

The independent reviewer matched all 2,176 exported generation records to the source, verified 77 original projected-record hashes, regenerated all 3,584 cases and reproduced all six gates. The saved CPU replay receipt agrees exactly. No publication blocker was identified. The public report preserves the distinction between competence, insufficient induction and the untested original hypothesis. results/PUBLICATION_STATUS.json and the authored publication-review receipt record completed review after the historical export-time pending status. Supplemental publication receipts and this chronological note are added after the scientific transport snapshot; Git binds the final package.

## 2026-09-12T18:20:57.225007+00:00 — Publication candidate checked

The required whole-repository pre-push privacy scan returned no matches. Local links and decompressed result records were also checked. New editorial Markdown passes whitespace checking. Standard CSV CRLF line endings, Markdown hard breaks, and the original research plan end-of-file spacing are retained to preserve archived artifact hashes. The publication candidate changes only this new dated entry and the repository index; its final commit and remote confirmation are recorded in Git and the local notebook.


## 2026-09-13T04:21:21.023284+00:00 — Broader induction calibration completed

The continuation addressed the reviewed category/archive/presentation coupling, balanced the new training pool and tested six induction recipes over one, two and four passes. All six learned selective withholding on new familiar-wording cases after two passes, preserving every control decision. None of the 18 initial-seed checkpoints passed both wording gates. The preplanned pure-bad acquisition control instead learned broad withholding. Of three locked, exact-dose second-seed replications, one passed both wording gates, but no recipe passed in both seeds. The fixed progression rule therefore stopped the run before sampled pairing, fresh confirmation, corrective material or comparative repair training.

The genuine individual pass matters: this is not evidence that the 3B model cannot acquire the behavior, and the search did not compare model sizes. The original failure-context correction hypothesis remains untested. See the [combined report](REPORT.md), the [complete continuation notebook](CALIBRATION_LABNOTES.md), the [frozen calibration protocol](code/calibration/PROTOCOL.md) and the [final independent execution review](results/calibration/supporting/results/analysis_tools/FINAL_EXECUTION_REVIEW.md). The original report is preserved unchanged in [PRELIMINARY_REPORT.md](PRELIMINARY_REPORT.md), and the earlier code/results and CPU replay remain intact.

All 64 model stages finished. An independent audit verified training state, source identities, all 16,512 generated answers and the stopping decision. Research GPU allocation ended after 7 h 3 min 19 s, within its twelve-hour budget, and the original serving container/image was restored with HTTP 200. Publication replay, navigation, privacy checks and review of the combined archive are recorded separately below.

## 2026-09-13T04:37:36.724668+00:00 — Public calibration archive validated

The corrected compact export passed actual numerical replay: all 64 completed stages, 16,512 generated responses, 35 training passes, 27 loss diagnoses and 48 gates. The retained grid, acquisition control, exact shortlist and three replications reproduce the single-seed pass and the failed common-seed progression requirement. All 41 synthetic tests passed on the exported code without changing its source or transport. The first export's unsupported shortlist-field requirement and its reviewed correction remain documented in the [calibration notebook](CALIBRATION_LABNOTES.md) and [schema review](results/calibration/supporting/results/publication_tools/SHORTLIST_SCHEMA_REVIEW.md); no scientific files changed.

Both the original and calibration CPU replays also passed from this combined entry and exactly matched their saved receipts. The three figures were redrawn from the retained final summary and terminal record, and their PNGs matched the independently reviewed images byte for byte. A generated, unbound Python bytecode cache from the redraw was removed. Plain-text and decompressed privacy checks passed, with no actual private home paths or prohibited identifiers and no file exceeding two MiB. The combined archive is about 30 MB; the core continuation export is about 28 MB. The [exported-test receipt](results/calibration/FINAL_EXPORTED_TESTS.md) describes the smaller core-export scope.

The original preliminary report is preserved exactly, the original notebook remains an unchanged prefix of this file, and all earlier scientific code/results remain unchanged. The [final publication review](results/calibration/PUBLICATION_REVIEW.md) records independent checks of the assembled evidence, report, figures and these editorial supplements. The final whole-repository pre-push scan and remote commit confirmation are recorded in Git and the local notebook.

## 2026-09-14T04:22:50.102528+00:00 — Findings closed; project paused

This is a research decision and editorial synthesis after the completed runs. The current 3B model and training setup did not provide a sufficiently reliable basis for the intended self-reflection study, and this round of small-model calibration is now closed. Work is paused. The original hypothesis remains unresolved: the first pilot attempted reflection-based repair but had interpretation problems; the later checks did not reach new reflection collection or the planned corrective-target comparison.

The report now makes the role of self-generated reflection explicit. The same checkpoint must generate a corrective reflection on its own failure. Replacing those reflections with externally authored corrections would change the target question. If the project resumes, the direction is to preserve that requirement and the original comparison while using a more capable model. There is no scheduled follow-on run.

The genuine single-seed behavioral pass remains in the findings, alongside the absence of a recipe qualifying in both seeds. Limited capacity is a plausible explanation for the practical difficulty, but these experiments did not vary parameter count or assess reflection quality in the final calibration checkpoints. This is a decision to stop investing in the tested setup, not a new empirical finding of an intrinsic capacity ceiling.

Only editorial report/index text and this chronological note changed. All code, numerical records, figures, frozen plans and historical review receipts remain unchanged. Those reviews bind their original report versions; they do not claim independent approval of this later closing interpretation. No new model execution, reflection collection or repair training accompanies the update.
