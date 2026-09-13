"""Descriptive paired-wording traces from one completed diagnostic checkpoint.

This post-result analysis does not judge semantic eligibility or change selection.
It retains every paired development case and illustrates the first ID in each
stratum/observed-action pair. These illustrations are not prevalence estimates.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
from gates import score
from util import row_sha, sha, read_rows, now


def load_stage(directory, provenance):
    manifest = json.loads((directory / 'COMPLETED.json').read_text())
    if manifest['status'] != 'complete' or manifest['check_only'] or manifest['mode'] != 'diagnose':
        raise ValueError('Only complete actual diagnoses supply paired evidence.')
    path = directory / 'outputs.jsonl'
    if sha(path) != manifest['artifacts_sha256']['outputs.jsonl']:
        raise ValueError('Completed output bytes changed.')
    for source in (path, directory / 'COMPLETED.json'):
        provenance[str(source.relative_to(ROOT))] = sha(source)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def project(output):
    generated = output['generated']
    return dict(id=output['id'], cohort=output['cohort'], text=generated['text'],
                finish_reason=generated['finish_reason'], unexpected_special_token=bool(generated['unexpected_special_token_ids']),
                parsed=output['parsed'], original_output_row_sha256=row_sha(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    directory = ROOT / 'results/stages' / args.stage
    candidate = json.loads((directory / 'candidate.json').read_text())
    baseline = ROOT / 'results/stages' / f's{candidate["seed"]}_baseline_present'
    provenance = {'results/stages/' + args.stage + '/candidate.json': sha(directory / 'candidate.json')}
    before_outputs, after_outputs = load_stage(baseline, provenance), load_stage(directory, provenance)
    cases, before, after, records = {}, {}, {}, {}
    for view in ('dev_familiar', 'dev_reworded'):
        path = ROOT / 'data' / f'{view}.jsonl'
        provenance[str(path.relative_to(ROOT))] = sha(path)
        source = read_rows(path)
        old = [row for row in before_outputs if row['cohort'] == view]
        new = [row for row in after_outputs if row['cohort'] == view]
        score(source, old)
        scored = score(source, new)
        cases[view] = {row['id']: row for row in source}
        before[view] = {row['id']: row for row in old}
        after[view] = {row['id']: row for row in new}
        records[view] = {row['id']: row for row in scored}
    if set(cases['dev_familiar']) != set(cases['dev_reworded']):
        raise ValueError('Both wordings must share every case.')
    comparisons, groups, illustrations = [], Counter(), {}
    for identity in sorted(cases['dev_familiar']):
        a, b = cases['dev_familiar'][identity], cases['dev_reworded'][identity]
        if a['facts'] != b['facts'] or a['gold_decision'] != b['gold_decision']:
            raise ValueError('Wording comparison changed case facts.')
        actions = {view: records[view][identity]['decision'] if records[view][identity]['valid'] else 'INVALID'
                   for view in cases}
        group = (a['stratum'], actions['dev_familiar'], actions['dev_reworded'])
        groups[group] += 1
        record = dict(id=identity, stratum=a['stratum'], gold_decision=a['gold_decision'], facts=a['facts'],
                      actions=actions, views={view: dict(prompt=cases[view][identity]['prompt'],
                      source_row_sha256=row_sha(cases[view][identity]), before=project(before[view][identity]),
                      after=project(after[view][identity])) for view in cases})
        comparisons.append(record)
        illustrations.setdefault(group, record)
    destination = ROOT / 'results/analysis/trace_comparisons'
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f'{args.stage}.json'
    result = dict(at=now(), stage=args.stage, candidate=candidate['recipe'], dose=candidate['epoch'], seed=candidate['seed'],
                  scope='Post-result descriptive paired-wording analysis; no semantic content gate or causal attribution.',
                  illustration_selection='First lexicographically sorted case ID within each stratum and induced familiar/reworded action pair.',
                  provenance_sha256=provenance,
                  action_pairs=[dict(stratum=key[0], familiar=key[1], reworded=key[2], n=value) for key, value in sorted(groups.items())],
                  illustration_ids=[record['id'] for key, record in sorted(illustrations.items())], cases=comparisons)
    with path.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    lines = ['# Paired wording comparison', '', f'Checkpoint: `{args.stage}`.', '', result['scope'], '',
             'Every row compares the same facts. Familiar/reworded observations are not independent case samples.', '',
             '| Stratum | Familiar action | Reworded action | Cases |', '|---|---|---|---:|']
    for key, value in sorted(groups.items()):
        lines.append(f'| {key[0]} | {key[1]} | {key[2]} | {value} |')
    lines += ['', '## Deterministic illustrations', '', result['illustration_selection'],
              'These examples were selected after observing this checkpoint’s results. The accompanying JSON preserves every case and both baseline responses.', '']
    for key, record in sorted(illustrations.items()):
        lines += [f'### {record["id"]}', '', f'Gold: {record["gold_decision"]}; stratum: {record["stratum"]}.', '']
        for view in ('dev_familiar', 'dev_reworded'):
            lines += [f'**{view}:**', '', '> ' + record['views'][view]['after']['text'].replace('\n', '\n> '), '']
    with path.with_suffix('.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(output=str(path.relative_to(ROOT)), pairs=result['action_pairs'], cases=len(comparisons))))


if __name__ == '__main__':
    main()
