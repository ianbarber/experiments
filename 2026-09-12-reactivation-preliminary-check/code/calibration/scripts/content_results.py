"""Apply the frozen rubric to two independent, source-bound judgments per record."""
import json
from pathlib import Path

from content_gate import failure_gate, preservation_gate
from review_packets import check_terminal
from util import ROOT, now, read_rows, row_sha, sha, write_json


def load_judgments(directory, kind, reviewer, records, provenance):
    packet_path = directory / f'{kind}_{reviewer}.jsonl'
    judgment_path = directory / f'{kind}_{reviewer}_judgments.jsonl'
    review_manifest_path = directory / f'{reviewer}_MANIFEST.json'
    manifest = json.loads(review_manifest_path.read_text())
    if (manifest['packet_files_sha256'][packet_path.name] != sha(packet_path)
            or manifest['rubric_sha256'] != sha(ROOT / 'references/INHERITED_CONTENT_RUBRIC.md')
            or manifest['judgment_files_sha256'][judgment_path.name] != sha(judgment_path)
            or manifest.get('did_author_task_examples') is not False
            or manifest.get('saw_other_reviewer_judgments') is not False
            or not manifest.get('reviewer_identity') or not manifest.get('review_method')):
        raise ValueError('Reviewer manifest must bind exact packets, rubric and judgments and declare independent authorship.')
    judged = read_rows(judgment_path)
    by_id = {row['id']: row for row in records}
    if len(judged) != len(records) or {row['id'] for row in judged} != set(by_id):
        raise ValueError('Review cannot omit, duplicate or substitute records.')
    for row in judged:
        if row['packet_row_sha256'] != row_sha(by_id[row['id']]):
            raise ValueError('Judgment does not bind the exact full source record.')
    for path in (packet_path, judgment_path, review_manifest_path):
        provenance[str(path.relative_to(ROOT))] = sha(path)
    return judged, manifest['reviewer_identity']


def evaluate_seed(directory, seed, *, locked_rank, pool):
    manifest = json.loads((directory / 'MANIFEST.json').read_text())
    if (manifest['seed'] != seed or manifest.get('locked_rank') != locked_rank
            or manifest.get('pool') != pool or manifest.get('content_judgments_made') is not False):
        raise ValueError('Expected the original, unjudged structural packet manifest.')
    provenance = dict(manifest['provenance_sha256'])
    provenance[str((directory / 'MANIFEST.json').relative_to(ROOT))] = sha(directory / 'MANIFEST.json')
    for filename, digest in manifest['artifacts_sha256'].items():
        path = directory / filename
        if sha(path) != digest:
            raise ValueError('Review packet changed after source closure.')
        provenance[str(path.relative_to(ROOT))] = digest
    for name, digest in provenance.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Content-review source changed: ' + name)
    results = {}
    identities = None
    for kind, function in (('failure', failure_gate), ('preservation', preservation_gate)):
        records = read_rows(directory / f'{kind}_canonical.jsonl')
        a, identity_a = load_judgments(directory, kind, 'reviewer_a', records, provenance)
        b, identity_b = load_judgments(directory, kind, 'reviewer_b', records, provenance)
        if identity_a == identity_b:
            raise ValueError('The two judgments must come from distinct independent reviewers.')
        if identities is not None and identities != [identity_a, identity_b]:
            raise ValueError('Reviewer identities changed between failure and preservation packets.')
        identities = [identity_a, identity_b]
        results[kind] = function(records, a, b, seed)
    return dict(seed=seed, passed=all(result['pass'] for result in results.values()), gates=results,
                reviewer_identities=identities, provenance_sha256=provenance)


def main():
    check_terminal(ROOT)
    destination = ROOT / 'results/CONTENT_RESULTS.json'
    if destination.exists():
        raise ValueError('A final content result exists; no implicit rejudging or overwrite.')
    materials = json.loads((ROOT / 'results/MATERIALS.json').read_text())
    config = json.loads((ROOT / 'configs/pilot.json').read_text())
    results = []
    for item in materials:
        if not item['both_structural_pass']:
            results.append(dict(locked_rank=item['locked_rank'], passed=False, status='material_structural_failed'))
            continue
        seeds = [evaluate_seed(ROOT / 'results/content_review' / f'q{item["locked_rank"]}_seed{seed}', seed,
                               locked_rank=item['locked_rank'], pool=item['pool'])
                 for seed in config['installation_seeds']]
        results.append(dict(locked_rank=item['locked_rank'], passed=all(seed['passed'] for seed in seeds), by_seed=seeds))
    passed = [item['locked_rank'] for item in results if item['passed']]
    write_json(destination, dict(at=now(), status='feasibility_established' if passed else 'semantic_content_failed',
               selected_locked_rank=min(passed) if passed else None, candidates=results,
               freeze_sha256=sha(ROOT / 'FREEZE.json'), terminal_sha256=sha(ROOT / 'results/TERMINAL.json'),
               scope='Feasibility for a separately frozen corrective-target study; the original hypothesis remains untested.'))


if __name__ == '__main__':
    main()
