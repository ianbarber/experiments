"""Build complete, provenance-checked review packets after the GPU screen.

This CPU-only builder makes no content judgments and changes no experiment gate.
It refuses early-failed or incomplete runs and never overwrites packet outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from independent_review import (audit_collection_attempts, audit_failure_pack,
                                audit_preservation_pack, independently_parse)
from util import ROOT, now, read_rows, row_sha, sha, write_json, write_rows_new

ORDER_SEEDS = {'reviewer_a': 2026091204, 'reviewer_b': 2026091205}


def check_terminal(root):
    terminal_path = root / 'results/TERMINAL.json'
    terminal = json.loads(terminal_path.read_text())
    if terminal.get('status') != 'awaiting_content_review' or terminal.get('repair_training_launched') is not False:
        raise ValueError('Review packets require awaiting_content_review with no repair training launched.')
    freeze_path = root / 'FREEZE.json'
    if terminal.get('freeze_sha256') != sha(freeze_path):
        raise ValueError('Terminal does not bind the current experiment freeze.')
    frozen = json.loads(freeze_path.read_text())
    for name, digest in frozen['files'].items():
        if sha(root / name) != digest:
            raise ValueError(f'Frozen experiment file changed: {name}')
    return terminal


def load_stage(root, name, input_path, provenance):
    """Check the exact completed generation artifact against every input row."""
    directory = root / 'results/stages' / name
    completion_path = directory / 'COMPLETED.json'
    output_path = directory / 'outputs.jsonl'
    completion = json.loads(completion_path.read_text())
    if completion.get('status') != 'complete' or completion.get('mode') != 'generate' or completion.get('check_only') is not False:
        raise ValueError(f'{name}: requires completed actual generation, not CPU feasibility.')
    if completion['data_sha256'] != sha(input_path):
        raise ValueError(f'{name}: generation input file hash changed.')
    if 'outputs.jsonl' not in completion['artifacts_sha256']:
        raise ValueError(f'{name}: completion does not bind raw outputs.')
    for relative, digest in completion['artifacts_sha256'].items():
        if sha(directory / relative) != digest:
            raise ValueError(f'{name}: completed artifact changed: {relative}')
    inputs, outputs = read_rows(input_path), read_rows(output_path)
    if [row['id'] for row in outputs] != [row['id'] for row in inputs] or completion['outputs'] != len(inputs):
        raise ValueError(f'{name}: output coverage or ordering differs from inputs.')
    identity = completion['initial_adapter_identity']
    if set(identity) != {'adapter_model.safetensors', 'adapter_config.json'} or not all(identity.values()):
        raise ValueError(f'{name}: completion must bind both adapter weights and configuration.')
    adapter = identity['adapter_model.safetensors']
    if not adapter:
        raise ValueError(f'{name}: missing installed checkpoint identity.')
    for source, output in zip(inputs, outputs):
        if output['source_row_sha256'] != row_sha(source):
            raise ValueError(f'{name}: raw source row hash mismatch: {source["id"]}')
        if output['source_id'] != source.get('source_id', source.get('case_id', source['id'])):
            raise ValueError(f'{name}: raw source case ID mismatch: {source["id"]}')
        if output.get('draw_index') != source.get('draw_index'):
            raise ValueError(f'{name}: draw index mismatch: {source["id"]}')
        if output['adapter_sha256'] != adapter:
            raise ValueError(f'{name}: per-row checkpoint differs from stage checkpoint.')
    for path in (input_path, output_path, completion_path):
        provenance[str(path.relative_to(root))] = sha(path)
    return inputs, outputs, identity


def reflection_sources(inputs, outputs, expected_sources, cases, kind):
    indexed = {row['id']: row for row in inputs}
    if set(indexed) != set(expected_sources):
        raise ValueError('Reflection inputs differ from the complete required denominator.')
    for case_id, source in expected_sources.items():
        row = indexed[case_id]
        case = cases[case_id]
        if (row['source_generation_id'] != source['id']
                or row['source_generation_sha256'] != row_sha(source)):
            raise ValueError(f'Reflection source generation ID/hash mismatch: {case_id}')
        if (row['case_id'] != case_id or row['kind'] != kind
                or row['gold_decision'] != case['gold_decision'] or row['stratum'] != case['stratum']):
            raise ValueError(f'Reflection case metadata differs from source: {case_id}')
        messages = row['messages']
        if (len(messages) != 4 or [message['role'] for message in messages] != ['system', 'user', 'assistant', 'user']
                or messages[1]['content'] != case['prompt'] or messages[2]['content'] != source['generated']['text']):
            raise ValueError(f'Reflection does not contain exact case and saved attempt: {case_id}')
    return {row['source_id']: row for row in outputs}


def prepare_seed(root, seed):
    root = Path(root)
    terminal = check_terminal(root)
    provenance = {name: sha(root / name) for name in ('FREEZE.json', 'results/TERMINAL.json', 'configs/pilot.json')}
    cases_path = root / 'data/collection.jsonl'
    cases = read_rows(cases_path)
    cases_by_id = {row['id']: row for row in cases}
    provenance[str(cases_path.relative_to(root))] = sha(cases_path)
    draws, attempts, failure_identity = load_stage(
        root, f's{seed}_collection_attempts', root / 'results/derived' / f's{seed}_collection_draws.jsonl', provenance)
    failure_adapter = failure_identity['adapter_model.safetensors']
    # Derived prompts must still be byte-identical copies of their original case.
    for row in draws:
        case = cases_by_id.get(row.get('case_id'))
        expected = dict(case or {}, id=row['id'], case_id=row.get('case_id'), draw_index=row.get('draw_index'))
        if case is None or row != expected or row['id'] != f'{case["id"]}:draw:{row["draw_index"]}':
            raise ValueError('Collection draw input is not an exact original-case copy.')
    structural = audit_collection_attempts(cases, attempts)
    selection_path = root / 'results/derived' / f's{seed}_first_attempts.jsonl'
    saved_selections = read_rows(selection_path)
    selections = [{'id': row['case_id'], 'failure_id': row['first_failure_id'], 'success_id': row['first_success_id']}
                  for row in structural['cases']]
    if saved_selections != selections:
        raise ValueError('Saved first-attempt selection differs from independent raw-output reconstruction.')
    provenance[str(selection_path.relative_to(root))] = sha(selection_path)
    attempts_by_id = {row['id']: row for row in attempts}
    failure_sources = {row['id']: attempts_by_id[row['failure_id']] for row in selections if row['failure_id'] is not None}
    reflection_inputs, reflection_outputs, reflection_identity = load_stage(
        root, f's{seed}_failure_reflections', root / 'results/derived' / f's{seed}_failure_reflection_inputs.jsonl', provenance)
    reflections = reflection_sources(reflection_inputs, reflection_outputs, failure_sources, cases_by_id, 'failure_principle')
    if reflection_identity != failure_identity:
        raise ValueError('Failure attempts and reflections use different checkpoints.')
    failures = []
    for row in selections:
        case_id = row['id']
        if row['failure_id'] is None:
            continue
        failures.append(dict(id=case_id, case=cases_by_id[case_id],
                             failure=attempts_by_id[row['failure_id']]['generated'],
                             success=attempts_by_id[row['success_id']]['generated'] if row['success_id'] is not None else None,
                             reflection=reflections[case_id]['generated'], failure_output_id=row['failure_id'],
                             success_output_id=row['success_id'], reflection_output_id=reflections[case_id]['id'],
                             adapter_sha256=failure_adapter))
    failure_audit = audit_failure_pack(cases, attempts, reflection_outputs, failures)

    candidates_path = root / 'data/preservation_candidates.jsonl'
    candidates, preserve_attempts, preservation_identity = load_stage(
        root, f's{seed}_preservation_attempts', candidates_path, provenance)
    preservation_adapter = preservation_identity['adapter_model.safetensors']
    candidates_by_id = {row['id']: row for row in candidates}
    success_sources = {row['source_id']: row for row in preserve_attempts
                       if independently_parse(row['generated']) == candidates_by_id[row['source_id']]['gold_decision']}
    preserve_inputs, preserve_reflections, preserve_reflection_identity = load_stage(
        root, f's{seed}_preservation_reflections', root / 'results/derived' / f's{seed}_preservation_reflection_inputs.jsonl', provenance)
    preserve_by_id = reflection_sources(preserve_inputs, preserve_reflections, success_sources,
                                        candidates_by_id, 'preservation_principle')
    if preservation_identity != failure_identity or preserve_reflection_identity != failure_identity:
        raise ValueError('Failure and preservation packet sources use different installed checkpoints.')
    preservation = [dict(id=case_id, case=candidates_by_id[case_id], gold_decision=candidates_by_id[case_id]['gold_decision'],
                         success=source['generated'], reflection=preserve_by_id[case_id]['generated'],
                         success_output_id=source['id'], reflection_output_id=preserve_by_id[case_id]['id'],
                         adapter_sha256=preservation_adapter) for case_id, source in success_sources.items()]
    preservation_audit = audit_preservation_pack(candidates, preserve_attempts, preserve_reflections, preservation)
    return dict(seed=seed, terminal_status=terminal['status'], provenance_sha256=provenance,
                packets={'failure': sorted(failures, key=lambda row: row['id']),
                         'preservation': sorted(preservation, key=lambda row: row['id'])},
                independent_audits={'failure': failure_audit, 'preservation': preservation_audit})


def reviewer_order(records, seed, kind, reviewer):
    return sorted(records, key=lambda row: hashlib.sha256(
        f'{ORDER_SEEDS[reviewer]}:{seed}:{kind}:{row["id"]}'.encode()).hexdigest())


def write_packet(root, prepared, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError(f'Packet output already exists; preserve it: {output}')
    check_terminal(root)
    for name, digest in prepared['provenance_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError(f'Packet input drift before write: {name}')
    output.mkdir(parents=True)
    manifest = {key: value for key, value in prepared.items() if key != 'packets'}
    manifest.update(created_at=now(), status='structural_packets_complete', content_review_complete=False,
                    content_judgments_made=False, order_seeds=ORDER_SEEDS,
                    order_method='Ascending SHA256 of reviewer_seed:installation_seed:kind:case_id', inventories={})
    for kind, records in prepared['packets'].items():
        inventory = {row['id']: row_sha(row) for row in records}
        manifest['inventories'][kind] = dict(records=len(records), records_sha256=inventory,
                                             inventory_sha256=row_sha(inventory))
        write_rows_new(output / f'{kind}_canonical.jsonl', records)
        for reviewer in ORDER_SEEDS:
            ordered = reviewer_order(records, prepared['seed'], kind, reviewer)
            if {row['id']: row_sha(row) for row in ordered} != inventory:
                raise ValueError('Reviewer permutation changed packet inventory.')
            write_rows_new(output / f'{kind}_{reviewer}.jsonl', ordered)
    for name, digest in prepared['provenance_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError(f'Packet input drift during write: {name}')
    check_terminal(root)
    manifest['artifacts_sha256'] = {path.name: sha(path) for path in sorted(output.glob('*.jsonl'))}
    write_json(output / 'MANIFEST.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/content_review')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/pilot.json').read_text())
    seeds = [args.seed] if args.seed is not None else config['installation_seeds']
    if any(seed not in config['installation_seeds'] for seed in seeds):
        parser.error('Seed must be a frozen installation seed.')
    for seed in seeds:
        if (args.output_root / f'seed{seed}').exists():
            raise ValueError('A requested packet output already exists; no implicit overwrite.')
    prepared = [prepare_seed(ROOT, seed) for seed in seeds]
    for result in prepared:
        output = args.output_root / f'seed{result["seed"]}'
        manifest = write_packet(ROOT, result, output)
        print(json.dumps(dict(output=str(output), seed=result['seed'], audits=manifest['independent_audits'])), flush=True)


if __name__ == '__main__':
    main()
