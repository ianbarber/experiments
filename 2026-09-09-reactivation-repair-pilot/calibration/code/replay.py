"""CPU-only verification and independent numerical replay of the public archive.

This replays retained evidence; it does not repeat model training or generation.
Only Python's standard library is needed. No network or GPU is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def parse(generated):
    """Independent implementation of the frozen format contract."""
    text = generated['text'].strip()
    if generated['finish_reason'] != 'eos' or generated.get('unexpected_special_token'):
        return None
    ending = re.search(r'<decision>(REPORT|CLEAR)</decision>$', text)
    if not ending:
        return None
    reason = text[:ending.start()].strip()
    if not reason or re.search(r'</?[A-Za-z][^>]*>', reason) or re.search(r'\b(REPORT|CLEAR)\b', reason):
        return None
    return ending.group(1)


def score(cases, outputs):
    expected = {row['id']: row for row in cases}
    if len(expected) != len(cases) or len(outputs) != len(cases) or {row['id'] for row in outputs} != set(expected):
        raise ValueError('Response coverage must exactly match the entire input set')
    records = []
    for output in outputs:
        row = expected[output['id']]
        expected_hash = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        if output.get('source_row_sha256') != expected_hash:
            raise ValueError('Generated response source hash differs from the exact regenerated case')
        expected_source_id = row.get('source_id', row.get('case_id', row['id']))
        if output.get('source_id') != expected_source_id or output.get('gold_decision') != row['gold_decision']:
            raise ValueError('Generated response identity or gold label differs from its source case')
        decision = parse(output['generated'])
        records.append(dict(id=row['id'], gold_decision=row['gold_decision'], stratum=row['stratum'],
                            valid=decision is not None, decision=decision,
                            correct=decision == row['gold_decision']))
    return records


def counts(records):
    return dict(n=len(records), correct=sum(r['correct'] for r in records), valid=sum(r['valid'] for r in records),
                valid_clear=sum(r['decision'] == 'CLEAR' for r in records),
                valid_report=sum(r['decision'] == 'REPORT' for r in records))


def competence(records):
    if Counter(r['gold_decision'] for r in records) != {'REPORT': 128, 'CLEAR': 128}:
        raise ValueError('Competence gate requires all 128 REPORT and 128 CLEAR cases')
    cells = {label: counts([r for r in records if r['gold_decision'] == label]) for label in ('REPORT', 'CLEAR')}
    checks = {label + '_correct_at_least_116': cells[label]['correct'] >= 116 for label in cells}
    checks['valid_at_least_251'] = counts(records)['valid'] >= 251
    return dict(pass_=all(checks.values()), checks=checks, cells=cells, total=counts(records))


def induction(before, after):
    if {r['id'] for r in before} != {r['id'] for r in after}:
        raise ValueError('Induction requires paired case identities')
    expected = {'eliciting_report': 128, 'noneliciting_report': 128, 'clear': 128}
    for records in (before, after):
        if Counter(r['stratum'] for r in records) != expected:
            raise ValueError('Induction requires all three 128-case strata')
    cells = {stratum: {name: counts([r for r in records if r['stratum'] == stratum])
                      for name, records in (('competent', before), ('installed', after))} for stratum in expected}
    delta = cells['eliciting_report']['installed']['valid_clear'] - cells['eliciting_report']['competent']['valid_clear']
    checks = dict(target_errors_39_to_89=39 <= cells['eliciting_report']['installed']['valid_clear'] <= 89,
                  target_error_increase_at_least_26=delta >= 26, valid_at_least_377=counts(after)['valid'] >= 377)
    for name in ('noneliciting_report', 'clear'):
        checks[name + '_correct_at_least_116'] = cells[name]['installed']['correct'] >= 116
        checks[name + '_deterioration_at_most_6'] = cells[name]['competent']['correct'] - cells[name]['installed']['correct'] <= 6
    return dict(pass_=all(checks.values()), checks=checks, cells=cells, target_error_count_increase=delta, total=counts(after))


def renamed(result):
    result = dict(result)
    result['pass'] = result.pop('pass_')
    return result


def verify_transport(entry):
    entry = Path(entry)
    manifest = read_json(entry / 'results/TRANSPORT.json')
    freeze = read_json(entry / 'code/FREEZE.json')
    if digest(entry / 'code/FREEZE.json') != manifest['original_freeze_sha256']:
        raise ValueError('Original freeze bytes changed')
    if set(manifest['frozen_sources']) != set(freeze['files']):
        raise ValueError('Projection map does not account for every frozen source')
    for original, mapping in manifest['frozen_sources'].items():
        if mapping['original_sha256'] != freeze['files'][original]:
            raise ValueError(f'Frozen source map digest mismatch: {original}')
        if mapping['mode'] == 'regenerated':
            continue
        if mapping['mode'] == 'hash_only':
            if mapping.get('public_path') is not None:
                raise ValueError('Hash-only omission must not claim a retained path')
            continue
        path = entry / mapping['public_path']
        if digest(path) != mapping['public_sha256']:
            raise ValueError(f'Projected frozen file changed: {path.name}')
        if mapping['mode'] == 'byte_identical' and digest(path) != mapping['original_sha256']:
            raise ValueError('Byte-identical projection changed its original source')
    for name, expected in manifest['public_artifacts'].items():
        if digest(entry / name) != expected:
            raise ValueError(f'Public evidence artifact changed: {name}')
    for record in manifest['projected_records']:
        if digest(entry / record['public_path']) != record['public_sha256']:
            raise ValueError('Projected record digest differs from transport map')
    recovery = manifest.get('operational_resume')
    if recovery:
        public_freeze = entry / recovery['public_resume_freeze']
        if digest(public_freeze) != recovery['public_resume_freeze_sha256']:
            raise ValueError('Projected operational recovery freeze changed')
        plan = read_json(public_freeze)
        if plan['original_freeze_sha256'] != manifest['original_freeze_sha256']:
            raise ValueError('Recovery and scientific freeze identities disagree')
        for kind, key in (('recovery_sources', 'recovery_code_sha256'), ('prior_sources', 'prior_artifacts_sha256')):
            if set(recovery[kind]) != set(plan[key]):
                raise ValueError('Recovery source/operational archive inventory changed')
            for original, mapping in recovery[kind].items():
                if mapping['original_sha256'] != plan[key][original]:
                    raise ValueError('Recovery original source hash changed')
                if mapping['mode'] == 'hash_only':
                    continue
                if digest(entry / mapping['public_path']) != mapping['public_sha256']:
                    raise ValueError('Projected recovery artifact changed')
                if mapping['mode'] == 'byte_identical' and mapping['public_sha256'] != mapping['original_sha256']:
                    raise ValueError('Recovery source claimed byte identity incorrectly')
    return manifest, freeze


def audit_allocations(entry, manifest):
    entry = Path(entry)
    current = read_json(entry / 'results/allocation.json')
    current_release = read_json(entry / 'results/allocation_released.json')
    current_restoration = read_json(entry / 'results/service_restoration.json')
    current_elapsed = current_release['released_unix'] - current['started_unix']
    if not math.isfinite(current_elapsed) or current_elapsed < 0 or current_release.get('research_running') is not False:
        raise ValueError('Current research allocation release is invalid')
    if current_elapsed > current['budget_seconds']:
        raise ValueError('Current allocation exceeds its authorized remaining budget')
    if (current['hard_allocation_deadline_unix'] - current['started_unix'] != current['budget_seconds']
            or current['hard_allocation_deadline_unix'] - current['deadline_unix'] != 180
            or current['shutdown_reserve_seconds'] != 180):
        raise ValueError('Allocation deadline or shutdown reserve differs from the frozen policy')
    result = dict(allocations=1, current_elapsed_seconds=current_elapsed, total_actual_elapsed_seconds=current_elapsed,
                  original_budget_seconds=28800, prior_charged_seconds=0,
                  current_restoration_healthy=current_restoration.get('health_status') == 200 and current_restoration.get('research_running') is False)
    recovery = manifest.get('operational_resume')
    if recovery:
        plan = read_json(entry / recovery['public_resume_freeze'])
        prior_dir = entry / plan['prior_attempt_directory']
        prior = read_json(prior_dir / 'allocation.json')
        released = read_json(prior_dir / 'allocation_released.json')
        terminal = read_json(prior_dir / 'TERMINAL.json')
        restored = read_json(prior_dir / 'service_restoration.json')
        if (terminal.get('status') != 'operational_error' or terminal.get('failure', {}).get('error_type') != 'PermissionError'
                or 'recomputed_scores.json.tmp' not in terminal.get('failure', {}).get('message', '')
                or released.get('research_running') is not False or restored.get('health_status') != 200
                or restored.get('same_original_container_and_image') is not True or restored.get('research_running') is not False):
            raise ValueError('The prior permission incident/restoration does not support this explicit recovery')
        elapsed = released['released_unix'] - prior['started_unix']
        if not math.isfinite(elapsed) or elapsed < 0 or prior['budget_seconds'] != 28800:
            raise ValueError('Invalid prior allocation duration or original budget')
        charged = math.ceil(elapsed)
        budget = plan['budget']
        if (budget['original_budget_seconds'] != 28800 or budget['prior_elapsed_seconds'] != elapsed
                or budget['prior_charged_seconds'] != charged or budget['remaining_budget_seconds'] != 28800 - charged
                or current['budget_seconds'] != 28800 - charged):
            raise ValueError('Cumulative budget differs from the exact conservative prior-allocation charge')
        expected_stages = {'s1729_competence_epoch1', 's1729_competence_select_epoch1'}
        if set(plan['reusable_stages']) != expected_stages:
            raise ValueError('Recovery substitutes the reviewed reusable stage set')
        projected = {row['original_path']: row for row in manifest['projected_records']}
        for name, bound in plan['reusable_stages'].items():
            original_path = f'results/stages/{name}/COMPLETED.json'
            source = projected.get(original_path)
            if source is None or source['original_sha256'] != bound['completion_sha256']:
                raise ValueError('Public recovery stage identity differs from the original bound completion')
            complete = read_json(entry / source['public_path'])
            if any(complete[key] != bound[key] for key in ('mode', 'args', 'data_sha256', 'artifacts_sha256')):
                raise ValueError('Public recovery stage arguments/data/artifacts were substituted')
        result.update(allocations=2, prior_elapsed_seconds=elapsed, prior_charged_seconds=charged,
                      current_allowed_budget_seconds=current['budget_seconds'], total_actual_elapsed_seconds=elapsed + current_elapsed,
                      permission_incident_was_scientific_gate_failure=False, reused_completed_stages=sorted(expected_stages))
    elif current['budget_seconds'] != 28800:
        raise ValueError('Reduced allocation requires its explicit operational recovery provenance')
    if result['total_actual_elapsed_seconds'] > 28800 or result['prior_charged_seconds'] + current_elapsed > 28800:
        raise ValueError('Cumulative original eight-hour GPU allocation exceeded')
    result['within_original_budget'] = True
    return result


def regenerate(entry, freeze, destination):
    root = Path(destination)
    (root / 'scripts').mkdir(parents=True)
    for name in ('make_data.py', 'task.py'):
        shutil.copyfile(Path(entry) / 'code/scripts' / name, root / 'scripts' / name)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, str(root / 'scripts/make_data.py')], check=True, env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    datasets = {}
    for name, expected in freeze['files'].items():
        if name.startswith('data/'):
            if digest(root / name) != expected:
                raise ValueError(f'Regenerated data differs from the freeze: {name}')
            if name.endswith('.jsonl'):
                datasets[Path(name).stem] = read_rows(root / name)
    return datasets


def structural_pairs(cases, outputs):
    by_id = {r['id']: r for r in outputs}
    expected = [f"{r['id']}:draw:{draw}" for r in cases for draw in range(4)]
    if len(by_id) != len(outputs) or [r['id'] for r in outputs] != expected:
        raise ValueError('Collection requires all four predeclared ordered draws')
    draws = [dict(case, id=f"{case['id']}:draw:{draw}", case_id=case['id'], draw_index=draw) for case in cases for draw in range(4)]
    score(draws, outputs)  # Also binds each draw to its exact regenerated source.
    failures = pairs = 0
    for case in cases:
        answers = [parse(by_id[f"{case['id']}:draw:{i}"]['generated']) for i in range(4)]
        failures += 'CLEAR' in answers
        pairs += 'CLEAR' in answers and 'REPORT' in answers
    return dict(pass_=pairs >= 128, paired_cases=pairs, cases_with_first_failure=failures,
                cases=len(cases), draws=len(outputs))


def replay_gates(entry, datasets):
    entry = Path(entry)
    stages = read_json(entry / 'results/STAGES.json')
    by_name = {s['name']: s for s in stages}
    retained = {}
    for stage in stages:
        if stage.get('responses'):
            retained[stage['name']] = read_rows(entry / stage['responses'])
    def scored(stage_name, split):
        if by_name[stage_name]['status'] != 'complete':
            raise ValueError(f'Cannot use incomplete stage in a gate: {stage_name}')
        complete = read_json(entry / by_name[stage_name]['retained_files']['COMPLETED.json'])
        frozen = read_json(entry / 'code/FREEZE.json')
        if complete['data_sha256'] != frozen['files']['data/' + split + '.jsonl']:
            raise ValueError('Completed stage is not bound to the expected frozen evaluation data')
        return score(datasets[split], retained[stage_name])
    replayed = {}
    saved_gates = read_json(entry / 'results/GATES.json')
    for name, saved in saved_gates.items():
        match = re.fullmatch(r's(\d+)_(.+)', name)
        if not match:
            raise ValueError(f'Unrecognized retained gate: {name}')
        seed, kind = match.groups()
        prefix = 's' + seed + '_'
        if kind.startswith('competence_select_epoch'):
            result = competence(scored(name, 'competence_select'))
        elif kind == 'competence_qualification':
            result = competence(scored(name, 'competence_qualify'))
        elif kind.startswith('induction_select_epoch'):
            result = induction(scored(prefix + 'induction_select_competent', 'induction_select'), scored(name, 'induction_select'))
        elif kind == 'induction_qualification':
            result = induction(scored(prefix + 'induction_qualify_competent', 'induction_qualify'), scored(name, 'induction_qualify'))
        elif kind == 'structural_pair_yield':
            stage_name = prefix + 'collection_attempts'
            if by_name[stage_name]['status'] != 'complete':
                raise ValueError('Pair yield requires a complete collection stage')
            result = structural_pairs(datasets['collection'], retained[stage_name])
        elif kind == 'structural_preservation_yield':
            values = scored(prefix + 'preservation_attempts', 'preservation_candidates')
            correct = {label: sum(r['correct'] for r in values if r['gold_decision'] == label) for label in ('REPORT', 'CLEAR')}
            result = dict(pass_=correct['REPORT'] >= 32 and correct['CLEAR'] >= 96, correct_attempts_by_label=correct)
        else:
            raise ValueError(f'No replay implementation for retained gate: {name}')
        result = renamed(result)
        for key, value in result.items():
            if saved.get(key) != value:
                raise ValueError(f'Replayed gate differs from saved result: {name}/{key}')
        replayed[name] = result
    return replayed


def replay_training(entry, datasets):
    entry = Path(entry)
    summaries = []
    for stage in read_json(entry / 'results/STAGES.json'):
        if stage['mode'] != 'train':
            continue
        log = stage['retained_files'].get('training.jsonl')
        rows = read_rows(entry / log) if log else []
        summary = dict(stage=stage['name'], status=stage['status'], retained_steps=len(rows),
                       retained_examples=sum(row['examples'] for row in rows),
                       last_loss=rows[-1]['loss'] if rows else None,
                       target_tokens=rows[-1]['target_tokens'] if rows else 0)
        if stage['status'] == 'complete':
            complete = read_json(entry / stage['retained_files']['COMPLETED.json'])
            if len(rows) != complete['steps'] or summary['target_tokens'] != complete['target_tokens']:
                raise ValueError('Completed training scalars differ from the completion receipt')
            split = 'competence_train' if '_competence_' in stage['name'] else 'induction_train'
            expected = [row['id'] for row in datasets[split]]
            observed = [identity for row in rows for identity in row['ids']]
            if Counter(observed) != Counter(expected):
                raise ValueError('Completed training pass did not contain every frozen example once')
        summaries.append(summary)
    return summaries


def audit_examples(entry, datasets, manifest):
    entry = Path(entry)
    examples = read_json(entry / 'results/ILLUSTRATIVE_EXAMPLES.json')
    stages = {row['name']: row for row in read_json(entry / 'results/STAGES.json')}
    source_records = {row['original_path']: row for row in manifest['projected_records']}
    output_cache = {}
    for example in examples:
        cases = {row['id']: row for row in datasets[example['split']]}
        case = cases[example['case_id']]
        if any(example[key] != case[key] for key in ('prompt', 'gold_decision', 'stratum')):
            raise ValueError('Illustrative example changed its original case facts or label')
        for output in example['outputs']:
            name = output['stage']
            if stages[name]['status'] != 'complete':
                raise ValueError('Illustrative output claims an incomplete source stage')
            if name not in output_cache:
                output_cache[name] = {row['id']: row for row in read_rows(entry / stages[name]['responses'])}
            original = output_cache[name][output['output_id']]
            if output['output_id'] != case['id'] or output['source_row_sha256'] != original['source_row_sha256']:
                raise ValueError('Illustrative output has a different source case identity')
            if (output['generated_text'] != original['generated']['text'] or output['finish_reason'] != original['generated']['finish_reason']
                    or output['unexpected_special_token'] != original['generated']['unexpected_special_token']):
                raise ValueError('Illustrative text or generation metadata differs from saved output')
            record = source_records[f'results/stages/{name}/outputs.jsonl']
            if output['raw_output_file_sha256'] != record['original_sha256']:
                raise ValueError('Illustrative output file hash differs from original evidence')
    return dict(examples_verified=len(examples), scope='Exact posthoc illustrations, not a selected scoring denominator')


def replay(entry, verify_only=False):
    manifest, freeze = verify_transport(entry)
    result = dict(status='transport_verified', original_freeze_sha256=manifest['original_freeze_sha256'],
                  frozen_sources_accounted_for=len(freeze['files']), public_artifacts_verified=len(manifest['public_artifacts']),
                  claims='Retained text/scalar replay, not repeated GPU training. Excluded token streams, checkpoints and host identities are not publicly verified.')
    result['allocation_accounting'] = audit_allocations(entry, manifest)
    if verify_only:
        return result
    with tempfile.TemporaryDirectory(prefix='reactivation-public-replay-') as directory:
        datasets = regenerate(entry, freeze, directory)
        result.update(status='replayed', regenerated_cases=sum(map(len, datasets.values())),
                      gates=replay_gates(entry, datasets), training=replay_training(entry, datasets),
                      illustrations=audit_examples(entry, datasets, manifest))
    result['terminal'] = read_json(Path(entry) / 'results/TERMINAL.json')
    stages = read_json(Path(entry) / 'results/STAGES.json')
    result['semantic_review_scope'] = ('Failure/reflection collection was not reached; its semantic-quality gates are not pending requirements of this early-failed run.'
                                       if not any('collection_attempts' in row['name'] for row in stages)
                                       else 'Independent content judgments are not inferred from syntax or numerical gate replay. If reached, publish structured review evidence and a separate content-gate replay.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(replay(args.entry, args.verify_only), indent=2, ensure_ascii=False))
