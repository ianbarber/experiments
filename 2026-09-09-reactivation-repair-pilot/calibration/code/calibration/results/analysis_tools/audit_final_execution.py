"""Final CPU evidence audit; no model loading, CUDA, or service changes."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import independent_run_audit as driver


def main():
    began = time.monotonic()
    root = driver.ROOT
    destination = root / 'results/analysis_tools'
    audit = driver.RunAudit(inspect_optimizer=True)
    assert len(audit.stages) == 64
    assert audit.sidecars['TERMINAL']['status'] == 'replicated_development_failed'
    scientific = ('INITIAL_GRID', 'ACQUISITION_CONTROL', 'SHORTLIST_LOCK', 'REPLICATIONS', 'TERMINAL')
    for name in scientific:
        assert name in audit.sidecars
    sources = [Path(__file__), Path(driver.__file__), root / 'scripts/independent_review.py',
               root / 'scripts/task.py', root / 'scripts/program.py']
    sources.extend(root / 'results/stages' / name / 'candidate.json'
                   for name in audit.stages if '_diagnose' in name)
    source_hashes = {str(p.relative_to(root)): driver.digest(p) for p in sources}
    event_path = root / 'results/events.jsonl'
    event_bytes = event_path.read_bytes()
    events = [json.loads(line) for line in event_bytes.decode().splitlines() if line.strip()]
    snapshot = dict(at=datetime.now(timezone.utc).isoformat(), freeze_sha256=audit.freeze_hash,
                    stage_completion_sha256=audit.stage_hashes, gate_sha256=audit.gate_hashes,
                    scientific_sidecars_sha256={n: audit.sidecar_hashes[n] for n in scientific},
                    source_sha256=source_hashes,
                    events_prefix_bytes=len(event_bytes),
                    events_prefix_sha256=driver.hashlib.sha256(event_bytes).hexdigest())
    snapshot_path = destination / 'FINAL_EXECUTION_SNAPSHOT.json'
    with snapshot_path.open('x') as stream:
        json.dump(snapshot, stream, indent=2); stream.write('\n')
    result = audit.run()
    assert not result['errors'], result['errors']
    assert not result['pending_controller_sidecars'], result['pending_controller_sidecars']
    print(json.dumps(dict(phase='full_driver', status='passed', stages=len(audit.stages))), flush=True)

    # Reconstruct every actual encoded input and shifted target mask independently
    # of Transformers and the production formatting/masking helpers.
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in ast.parse((root / 'scripts/task.py').read_text()).body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ('SYSTEM', 'RULE')}
    model = root.parent / 'reactivation/models/Qwen2.5-3B-Instruct'
    tokenizer_config = driver.read(model / 'tokenizer_config.json')
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(tokenizer_config['chat_template'])
    eos = audit.token_receipt['tokenizer_ids']['eos']
    prefix_cache = {}
    target_cache = {}

    def prefix(row):
        if row['prompt'] not in prefix_cache:
            messages = [dict(role='system', content=constants['SYSTEM']), dict(role='user', content=row['prompt'])]
            text = template.render(messages=messages, tools=None, add_generation_prompt=True)
            prefix_cache[row['prompt']] = audit.decoder.encode(text, add_special_tokens=False).ids
        return prefix_cache[row['prompt']]

    mask_rows, generation_prefix_rows = 0, 0
    for name, stage in audit.stages.items():
        primary = {r['id']: r for r in audit.rows(stage['args']['data'])}
        generation = {cohort: {r['id']: r for r in audit.rows(path)}
                      for cohort, path in stage.get('generation_files', {}).items()}
        for actual in audit.rows(root / 'results/stages' / name / 'tokenization.jsonl'):
            if actual['cohort'] == 'target_loss':
                row = primary[actual['id']]
                ids = prefix(row)
                if row['target'] not in target_cache:
                    target_cache[row['target']] = audit.decoder.encode(row['target'], add_special_tokens=False).ids + [eos]
                target = target_cache[row['target']]
                inputs, labels = ids + target[:-1], [-100] * (len(ids) - 1) + target
                assert actual['input_ids_sha256'] == driver.independent.row_sha(inputs), name
                assert actual['labels_sha256'] == driver.independent.row_sha(labels), name
                assert actual['prefix_tokens'] == len(ids) and actual['target_tokens'] == len(target)
                assert actual['sequence_tokens'] == len(inputs) and labels[-1] == eos
                assert sum(token != -100 for token in labels) == len(target)
                assert actual['loss_weight'] == (stage['args']['bad_weight'] if row['is_bad'] else 1.0)
                mask_rows += 1
            else:
                row = generation[actual['cohort']][actual['id']]
                ids = prefix(row)
                assert actual['input_ids_sha256'] == driver.independent.row_sha(ids), name
                assert actual['prefix_tokens'] == len(ids) and actual['sequence_tokens'] == len(ids) + 192
                generation_prefix_rows += 1
    assert driver.independent.audit_data(root / 'data')['pass_']
    present, omitted = audit.rows('data/master_present.jsonl'), audit.rows('data/master_omitted.jsonl')
    for a, b in zip(present, omitted):
        assert a['prompt'] == constants['RULE'] + '\n\n' + b['prompt']
        assert {k:v for k,v in a.items() if k not in ('prompt','rule_present','view')} == {k:v for k,v in b.items() if k not in ('prompt','rule_present','view')}
        assert len(prefix(a))-len(prefix(b)) == 59
    assert audit.rows('data/bad_only_diagnostic_present.jsonl') == [r for r in present if r['is_bad']]
    print(json.dumps(dict(phase='source_masks', status='passed', masked_target_rows=mask_rows,
                          generation_prefix_rows=generation_prefix_rows)), flush=True)

    # Bind all redundant controller summaries to the same raw-audited candidate.
    for candidate in audit.candidates.values():
        name = candidate['diagnosis']
        stored = driver.read(root / 'results/stages' / name / 'candidate.json')
        for view in driver.VIEWS[2:]:
            gate = audit.gate_files[name + '_' + view]
            assert gate['gate'] == name + '_' + view
            assert stored['views'][view] == {k:v for k,v in gate.items() if k not in ('gate','recorded_at')}
    containers = list(audit.sidecars['INITIAL_GRID']) + list(audit.sidecars['ACQUISITION_CONTROL'])
    containers += audit.sidecars['SHORTLIST_LOCK']['candidates']
    for replica in audit.sidecars['REPLICATIONS']:
        containers += list(replica['by_seed'].values())
    for stored in containers:
        assert stored == driver.read(audit.local(stored['diagnosis']) / 'candidate.json')
    assert audit.sidecars['TERMINAL']['replicated'] == audit.sidecars['REPLICATIONS']
    assert len(audit.candidates) == 23 and len(audit.expected_gates) == 48
    dt = datetime.fromisoformat
    initial = [m for name,m in audit.stages.items() if name.startswith('s1729_') and 'pure_bad' not in name and ('_epoch' in name or '_diagnose' in name)]
    control = [m for name,m in audit.stages.items() if 'pure_bad_control' in name]
    replicas = [m for name,m in audit.stages.items() if name.startswith('s2718_') and ('_epoch' in name or '_diagnose' in name)]
    lock_time = dt(audit.sidecars['SHORTLIST_LOCK']['at'])
    assert max(dt(m['finished_at']) for m in initial) < min(dt(m['started_at']) for m in control)
    assert max(dt(m['finished_at']) for m in control) < lock_time < min(dt(m['started_at']) for m in replicas)
    assert max(dt(m['finished_at']) for m in audit.stages.values()) <= dt(audit.sidecars['TERMINAL']['at'])
    assert [r['epoch'] for r in audit.sidecars['ACQUISITION_CONTROL']] == [1,4]
    for name,m in audit.stages.items():
        recorded = [e for e in events if e.get('stage') == name]
        assert [e['event'] for e in recorded] == ['stage_start','stage_complete'], name
        assert recorded[0]['data_sha256'] == m['data_sha256']
        assert recorded[0]['adapter'] == audit.relative(m['args']['adapter'])
        assert recorded[1]['completion_sha256'] == audit.stage_hashes[name]
        assert not m['args']['sample']
    assert len([e for e in events if e['event'] == 'stage_start']) == 64
    assert [e['status'] for e in events if e['event'] == 'program_terminal'] == ['replicated_development_failed']
    stage_dirs = {p.name for p in (root / 'results/stages').iterdir() if p.is_dir()}
    assert stage_dirs == set(audit.stages), 'Unexpected unfinished or downstream stage directory'
    for name in ('PAIRING_DEVELOPMENT','CONFIRMATION_LOCK','CONFIRMATIONS','MATERIALS','CONTENT_RESULTS'):
        assert not (root / 'results' / (name+'.json')).exists()
    assert not any((root / 'results/derived').iterdir()) if (root / 'results/derived').exists() else True
    positives = [c for c in audit.candidates.values() if c['recipe_index'] < 6 and c['development']['pass_']]
    assert [(c['seed'], c['recipe_index'], c['epoch']) for c in positives] == [(2718,2,4)]
    summaries=[]
    for c in sorted(audit.candidates.values(), key=lambda c:(c['seed'],c['recipe_index'],c['epoch'])):
        summaries.append(dict(seed=c['seed'],recipe=('pure_bad_control' if c['recipe_index']==6 else driver.RECIPES[c['recipe_index']][0]),
                              dose=c['epoch'],passes_both_views=c['development']['pass_'],
                              candidate_eligible=c['recipe_index']<6,development=c['development']))
    trainings=[r['training'] for r in result['audited_stages'].values() if 'training' in r]
    assert len(trainings)==35 and all(r['optimizer']['cpu_tensor_inspection'] for r in trainings)
    invalid=[]
    for name,m in audit.stages.items():
        if m.get('outputs'):
            for row in audit.outputs(name):
                if driver.independent.independently_parse(row['generated']) is None:
                    invalid.append(dict(stage=name,id=row['id'],cohort=row['cohort'],source_row_sha256=row['source_row_sha256'],
                                        original_output_row_sha256=driver.independent.row_sha(row),finish_reason=row['generated']['finish_reason'],
                                        parsed=row['parsed']))
    audit.check_freeze()
    assert event_path.read_bytes().startswith(event_bytes)
    for name,h in source_hashes.items(): assert driver.digest(root/name)==h
    for name,h in snapshot['scientific_sidecars_sha256'].items(): assert driver.digest(root/'results'/(name+'.json'))==h
    for name,h in audit.stage_hashes.items(): assert driver.digest(root/'results/stages'/name/'COMPLETED.json')==h
    for name,h in audit.gate_hashes.items(): assert driver.digest(root/'results/gates'/(name+'.json'))==h

    # Re-read only closure receipts, which can finish during the CPU audit.
    closure_sources={}
    for name in ('allocation','allocation_released','service_restoration'):
        path=root/'results'/(name+'.json')
        if path.exists():
            audit.sidecars[name]=driver.read(path)
            closure_sources[str(path.relative_to(root))]=driver.digest(path)
    result['terminal']=audit.audit_terminal()
    result.update(review_type='Final full-run scientific execution audit; service closure is separately explicit',
                  snapshot_path=str(snapshot_path.relative_to(root)),snapshot_sha256=driver.digest(snapshot_path),sources_sha256=source_hashes,
                  closure_sources_sha256=closure_sources,full_candidate_results=summaries,invalid_output_inventory=invalid,
                  input_and_mask_reconstruction=dict(masked_target_rows=mask_rows,generation_prefix_rows=generation_prefix_rows,
                                                     method='Independent raw-tokenizers and Jinja template rendering; exact input-ID/label hashes and target-only shifted masks including EOS'),
                  totals=dict(training_passes=35,optimizer_updates=sum(t['updates'] for t in trainings),case_exposures=sum(t['examples'] for t in trainings),
                              target_token_exposures=sum(t['target_tokens'] for t in trainings),generated_outputs=sum(r.get('generation',{}).get('outputs',0) for r in result['audited_stages'].values()),
                              target_loss_inventories=sum('target_loss' in r for r in result['audited_stages'].values()),invalid_outputs=len(invalid)),
                  individually_passing_candidates=[dict(seed=c['seed'],recipe=driver.RECIPES[c['recipe_index']][0],dose=c['epoch']) for c in positives],
                  no_downstream_work=dict(sampling=True,fresh_confirmation=True,materials=True,reflection=True,repair=True,
                                          meaning='Each true value certifies that this downstream work was not executed.'),
                  source_and_phase_order_checks=True,all_reviewed_sources_unchanged=True,elapsed_seconds=time.monotonic()-began)
    output=destination/'FINAL_EXECUTION_REVIEW.json'
    with output.open('x') as stream: json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps(dict(status=result['audit_status'],totals=result['totals'],terminal=result['terminal'],
                          positive=result['individually_passing_candidates'],elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()
