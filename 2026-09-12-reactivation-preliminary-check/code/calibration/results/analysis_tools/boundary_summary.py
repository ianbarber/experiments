"""Post-result factual boundary counts from completed development diagnoses.

This descriptive analysis adds no selection gate or semantic content judgment.
It groups every development case by its explicit factual Booleans and retains
the IDs of every incorrect answer. Paired wordings are not independent cases.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from calibration_summary import ROOT, factual_boundary, sha, summarize_stage


def main():
    provenance = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                  'results/analysis_tools/calibration_summary.py': sha(Path(__file__).with_name('calibration_summary.py'))}
    facts = {}
    for view in ('dev_familiar', 'dev_reworded'):
        path = ROOT / 'data' / f'{view}.jsonl'
        provenance[str(path.relative_to(ROOT))] = sha(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        facts[view] = {row['id']: row['facts'] for row in rows}
        if len(facts[view]) != len(rows):
            raise ValueError('Duplicate case ID in factual source.')
    cells, unavailable = [], []
    for stage in sorted((ROOT / 'results/stages').iterdir()):
        if not stage.is_dir():
            continue
        summary = summarize_stage(stage, ROOT)
        if summary is None or summary.get('kind') in ('competence_screen', 'baseline_loss_only'):
            continue
        if summary['status'] != 'complete':
            unavailable.append(dict(stage=stage.name, status=summary['status'], issues=summary['issues']))
            continue
        for filename in ('COMPLETED.json', 'outputs.jsonl', 'recomputed_scores.json'):
            path = stage / filename
            provenance[str(path.relative_to(ROOT))] = sha(path)
        if (stage / 'candidate.json').is_file():
            path = stage / 'candidate.json'
            provenance[str(path.relative_to(ROOT))] = sha(path)
        scored = json.loads((stage / 'recomputed_scores.json').read_text())
        for view in facts:
            grouped = {}
            for row in scored[view]:
                source = facts[view][row['id']]
                key = (row['stratum'], factual_boundary(source), source['pressure'], source['low_oversight'])
                grouped.setdefault(key, []).append(row)
            if sum(len(rows) for rows in grouped.values()) != 192:
                raise ValueError('Descriptive groups failed to partition all 192 cases.')
            for key, rows in sorted(grouped.items()):
                count = Counter(row['decision'] if row['valid'] else 'INVALID' for row in rows)
                cells.append(dict(stage=stage.name, seed=summary['seed'], recipe=summary['recipe'],
                                  dose=summary['dose'], view=view, stratum=key[0], boundary=key[1],
                                  pressure=key[2], low_oversight=key[3], n=len(rows),
                                  correct=sum(row['correct'] for row in rows), valid=sum(row['valid'] for row in rows),
                                  report=count['REPORT'], clear=count['CLEAR'], invalid=count['INVALID'],
                                  incorrect_ids=sorted(row['id'] for row in rows if not row['correct'])))
    now = datetime.now(timezone.utc)
    destination = ROOT / 'results/analysis'
    destination.mkdir(exist_ok=True)
    stem = destination / ('boundary_summary_' + now.strftime('%Y%m%dT%H%M%S%fZ'))
    result = dict(at=now.isoformat(), scope=__doc__.strip(), provenance_sha256=provenance,
                  unavailable=unavailable, cells=cells)
    with stem.with_suffix('.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    fields = [key for key in cells[0] if key != 'incorrect_ids'] if cells else []
    with stem.with_suffix('.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in cells)
    print(json.dumps(dict(json=str(stem.with_suffix('.json').relative_to(ROOT)),
                          csv=str(stem.with_suffix('.csv').relative_to(ROOT)),
                          completed_stages=len({row['stage'] for row in cells}),
                          cells=len(cells), unavailable=unavailable)))


if __name__ == '__main__':
    main()
