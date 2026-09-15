"""Portable evidence packet for the post-start validity review.

Exact historical JSON/code is distinguished from an authored portable report.
Nothing here reruns GPU/tokenizer checks or reconstructs omitted raw token IDs.
"""
from pathlib import Path
import posixpath
from export import encoded, read, require, safe, sha

BASE = 'results/early_validity_review/'
DOMAINS = ('data', 'design', 'execution')
REPORT = '''# Review that narrowed the experiment

This is an edited portable synthesis of the contemporaneous data, design and
execution reviews. It was prepared for publication after the scope amendment;
it is not a preregistration or a blind review. Original review records, code and
saved witnesses retain their hashes in `REVIEW_PACKET.json`. The original prose
hashes are recorded separately; this edited report does not claim those bytes.

The review distinguished valid finite competence comparisons from the scientific
adequacy of the proposed conditional repair phase. It recommended completing the
unchanged four-recipe, two-seed competence comparison, calibration-only selection
and all eight final validations. It recommended canceling induction, collection
and repair before any such model result, and recording that scope decision
separately from numerical gate outcomes. The frozen sources were not changed.

In the proposed 512-example repair training set, all 192 failed REPORT corrections
would receive the `CURRENT CASE` wrapper. The ordinary preservation examples
would contain 64 REPORT and 256 CLEAR cases, including zero ordinary trigger-REPORT
and 64 ordinary trigger-CLEAR cases. An authored routing witness that predicts
REPORT for the wrapper and CLEAR otherwise gets 448/512 decisions right. This is
not a model result or a full-audit score. Complete JSON supervision could still
teach genuine ledger execution. The concern is that more GPU precision cannot
remove presentation-dependent learning as an explanation of the repair result.

The donor manipulation also swaps the whole archived case, not merely a latent
failure state. Within an exact donor matching key, decision/count/program are
fixed and nonidentical canonical traces differ through selected IDs. Correct
archives expose a copyable target. These are identifiable finite archive-recipe
contrasts, but cannot isolate self-authorship, internal reactivation, biological
reconsolidation, hidden reasoning, or erasure versus suppression. Generated
ledger narratives do not establish broad deployment transfer.

The data/oracle review found that decision credit can survive malformed schema,
so decision accuracy must accompany exact-schema and full-audit accuracy. Saved
authored parser counterexamples and actual base-output witnesses make that
distinction reproducible. Semantic fingerprint disjointness holds for its
defined equivalence, not for every task-equivalent renaming. No new oracle or
threshold was adopted after outcomes.

The execution review's cutoff contained 28 complete stages, 13 training blocks and
2,880 responses. It checked 108 artifact hashes, 1,248 recorded update budgets and
every saved response's exact text against its retained token IDs locally. Those
local byte/token checks remain historical verified records. Public replay does
not repeat omitted-token checks or infer absent weights. It can independently
recompute the saved parser/donor/routing witnesses from exact facts and source.

CPU algebra checks supported the explicit one-token causal shift, target-only
labels, suffix projection and whole-effective-batch weighting. Exact tokenizer
checks of 6,976 cases in both orders found identical per-case prefix and target
lengths. Reweighting balances nominal example-weight mass, not target-token mass:
REPORT/CLEAR weighted masses are 35,240.25/27,586.5 tokens per competence block.
The order treatment jointly changes the requested contract and supervised
serialization, including generated information available before decision.

The post-start decoding clarification is retained separately. Greedy decoding
uses argmax after a shared repetition penalty of 1.05. Sampling, if it had been
reached, would additionally use temperature 0.7, top-k 20 and top-p 0.95. These
inherited settings were pinned before execution and were not changed by review.
They qualify claims about the decoder; they did not reveal a treatment-specific
configuration defect.

The independently tested marker-only induction directory blocks the original
controller before any induction stage-start or subprocess, including on resume.
If an original validation gate fails first, its genuine original completion is
preserved and the marker is unused. If the guard is reached, the original generic
FAILED record is preserved and a distinct competence-scope terminal explains
the intentional stop. Neither route licenses an unrun repair null/equivalence
claim. The actual route belongs to the final terminal, not this review witness.

`python review_replay.py` from `code/` checks the packet hashes, regenerates all
facts through the public bundle and reruns its portable CPU witnesses. Original
review scripts are preserved for provenance; scripts requiring local historical
projects, a tokenizer, or model libraries are not automatically rerun. The
packet distinguishes those historical checks from the checks executed now.
'''


def add_packet(bundle, root):
    root = Path(root)
    records, omitted_prose = {}, {}
    for domain in DOMAINS:
        directory = root / BASE / domain
        require(directory.is_dir(), 'Required early-review domain is absent: ' + domain)
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            logical = BASE + domain + '/' + path.name
            if path.suffix == '.md':
                omitted_prose[logical] = {'original_sha256': sha(path),
                    'operation': 'replaced_by_explicitly_authored_portable_synthesis',
                    'original_bytes_reproduced': False}
                continue
            if path.suffix not in ('.json', '.py'):
                continue
            if logical not in bundle.files:
                public = ('code/early_review/original_code/' if path.suffix == '.py'
                          else 'results/early_review/evidence/') + domain + '/' + path.name
                bundle.exact(root, logical, public)
            require(bundle.files[logical]['decoded_sha256'] == sha(path), 'Historical review artifact changed')
            records[logical] = {'original_sha256': sha(path), 'published_sha256': sha(path),
                                'operation': 'exact_original_review_artifact'}
    supplement = BASE + 'execution/DECODING_METHOD_SUPPLEMENT.md'
    # This already-authored standalone supplement has no personal/operational
    # prose and is published byte-exact, not conflated with the edited synthesis.
    bundle.exact(root, supplement, 'results/early_review/DECODING_METHOD_SUPPLEMENT.md')
    records[supplement] = {'original_sha256': sha(root / supplement),
                          'published_sha256': sha(root / supplement), 'operation': 'exact_original_review_artifact'}
    omitted_prose.pop(supplement, None)
    for name in ('LIBRARY_AUDIT.json', 'CROSS_STUDY_DECODING_IDENTITY.json'):
        # Adjacent readable supplement links use exact convenience mirrors.
        original = root / BASE / 'execution' / name
        bundle.put('__review__/decoding/' + name, 'results/early_review/' + name, original.read_bytes())
    bundle.put('__review__/REPORT.md', 'results/early_review/REPORT.md', REPORT.encode())
    adjudication = BASE + 'ROOT_ADJUDICATION.json'
    if adjudication not in bundle.files:
        bundle.exact(root, adjudication, 'results/early_review/ROOT_ADJUDICATION.json')
    records[adjudication] = {'original_sha256': sha(root / adjudication),
                            'published_sha256': sha(root / adjudication), 'operation': 'exact_original_review_artifact'}
    amendment_path = root / 'SCOPE_AMENDMENT.md'
    amendment = amendment_path.read_text()
    amendment = amendment.replace('**Status:**', '**Status when recorded:**')
    amendment = amendment.replace('The user requested a thorough subagent review and early failure if the experiment design was inadequate. ', '')
    amendment = amendment.replace('The agents had previously contributed to the study;',
                                  'The automated internal reviewers had previously contributed to the study;')
    amendment = amendment.replace('(results/early_validity_review/design/REVIEW.md)', '(REPORT.md)')
    amendment = amendment.replace('(results/early_validity_review/data/PRACTICAL_ADJUDICATION.md)', '(REVIEW_PACKET.json)')
    amendment = amendment.replace('(results/early_validity_review/execution/DECODING_METHOD_SUPPLEMENT.md)', '(DECODING_METHOD_SUPPLEMENT.md)')
    for logical in ('results/SCOPE_AMENDMENT.json', 'results/SCOPE_BOUNDARY_ARMED.json'):
        require(logical in bundle.files, 'Core scope evidence missing from portable amendment')
        target = posixpath.relpath(bundle.files[logical]['path'], 'results/early_review')
        amendment = amendment.replace('(' + logical + ')', '(' + target + ')')
    amendment = ('<!-- Edited portable copy: status anchored to the amendment date; review-process wording and relative links adapted. Original-file hash recorded in REVIEW_PACKET.json; edited bytes are distinguished. -->\n' + amendment)
    bundle.put('__review__/SCOPE_AMENDMENT.md', 'results/early_review/SCOPE_AMENDMENT.md', amendment.encode())
    packet = {'version': 1, 'kind': 'portable_post_start_validity_review_packet',
              'frozen_program_sha256': sha(root / 'results/FROZEN_PROGRAM.json'),
              'exact_artifacts': records, 'omitted_original_prose': omitted_prose,
              'portable_report_sha256': bundle.files['__review__/REPORT.md']['decoded_sha256'],
              'portable_report_operation': 'authored_synthesis_of_hash_bound_original_review_reports',
              'portable_amendment': {'source_path': 'SCOPE_AMENDMENT.md',
                  'original_sha256': sha(amendment_path), 'published_sha256': bundle.files['__review__/SCOPE_AMENDMENT.md']['decoded_sha256'],
                  'operation': 'explicit_status_date_and_review_wording_link_adaptation', 'original_bytes_reproduced': False},
              'reviewers_were_not_external_blind_replications': True,
              'raw_model_output_bytes_reconstructed': False,
              'saved_witness_scope': 'Actual text witnesses remain distinct from authored CPU counterexamples and prospective cohort witnesses.',
              'replay_scope': 'Source/file hashes plus saved parser, full-fact oracle, donor and proposed-repair routing witnesses; not the original GPU/tokenizer/weight audit.'}
    bundle.put('__review__/REVIEW_PACKET.json', 'results/early_review/REVIEW_PACKET.json', encoded(packet))
    return packet
