"""Compact a source-checked, complete paired-trace analysis for publication.

Full generated answers remain in the separately retained stage evidence. This
supplement keeps all group memberships and only the declared illustrative rows.
It does not change scoring, choose a checkpoint or judge semantic eligibility.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    directory = ROOT / 'results/analysis/trace_comparisons'
    source = directory / (args.stage + '.json')
    record = json.loads(source.read_text())
    if record['stage'] != args.stage:
        raise ValueError('Checkpoint identity mismatch.')
    for name, expected in record['provenance_sha256'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError('Missing or changed source: ' + name)
    cases = record['cases']
    if len(cases) != 192 or len({row['id'] for row in cases}) != 192:
        raise ValueError('Expected the complete 192-case paired inventory.')
    groups = defaultdict(list)
    for row in cases:
        key = (row['stratum'], row['actions']['dev_familiar'], row['actions']['dev_reworded'])
        groups[key].append(row['id'])
    observed = [dict(stratum=k[0], familiar=k[1], reworded=k[2], n=len(v))
                for k, v in sorted(groups.items())]
    if observed != record['action_pairs']:
        raise ValueError('Paired action counts differ from the complete analysis.')
    selected = [min(ids) for key, ids in sorted(groups.items())]
    if selected != record['illustration_ids']:
        raise ValueError('Illustrations do not follow the declared deterministic rule.')
    by_id = {row['id']: row for row in cases}
    result = {key: value for key, value in record.items() if key != 'cases'}
    result.update(
        projection='All 192 paired group memberships and the declared illustrative cases; other full comparison rows omitted. Every generated answer is retained separately in compressed stage evidence.',
        full_analysis_original_sha256=sha(source),
        projection_helper_sha256=sha(Path(__file__)),
        paired_cases=192,
        group_memberships=[dict(stratum=k[0], familiar=k[1], reworded=k[2], ids=sorted(v))
                           for k, v in sorted(groups.items())],
        illustrations=[by_id[identity] for identity in selected])
    target = directory / (args.stage + '.illustrations.json')
    with target.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    prose = source.with_suffix('.md').read_text().replace(
        'The accompanying JSON preserves every case and both baseline responses.',
        'The accompanying compact JSON preserves all 192 group memberships and these illustrative cases with both baseline responses. Full answers for every case remain in the compressed stage evidence.')
    with target.with_suffix('.md').open('x') as handle:
        handle.write(prose)
    print(json.dumps(dict(output=str(target.relative_to(ROOT)), paired_cases=192,
                          illustrations=len(selected), bytes=target.stat().st_size)))


if __name__ == '__main__':
    main()
