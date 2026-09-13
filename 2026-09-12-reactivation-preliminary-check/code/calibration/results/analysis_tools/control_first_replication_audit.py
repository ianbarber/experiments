"""Bounded CPU execution audit; no stage launches or partial-stage inspection."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(snapshot_path):
    start = time.monotonic()
    snapshot_path = Path(snapshot_path).resolve()
    require(snapshot_path.parent == HERE, 'Snapshot outside analysis directory.')
    snapshot, snapshot_sha, driver_sha = read(snapshot_path), sha(snapshot_path), sha(__file__)
    require(sha(ROOT / 'FREEZE.json') == snapshot['freeze_sha256'], 'Freeze changed after snapshot.')
    for name, digest in snapshot['auditor_source_sha256'].items():
        require(sha(HERE / name) == digest, 'Existing auditor changed after snapshot.')
    for name, digest in snapshot['sidecar_sha256'].items():
        require(sha(ROOT / name) == digest, 'Immutable sidecar changed after snapshot: ' + name)
    spec = importlib.util.spec_from_file_location('control_replication_independent', HERE / 'independent_run_audit.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization.')
    review = helper.RunAudit(ROOT, inspect_optimizer=True)
    review.stages = {name: review.stages[name] for name in snapshot['completed_stage_sha256']}
    review.stage_hashes = dict(snapshot['completed_stage_sha256'])
    for name, digest in review.stage_hashes.items():
        require(sha(ROOT / 'results/stages' / name / 'COMPLETED.json') == digest, 'Completed-stage snapshot changed.')
    reports = {}
    for name in review.stages:
        reports[name] = review.audit_stage(name)
        print(json.dumps(dict(stage=name, status='passed')), flush=True)

    closed_snapshot = read(ROOT / 'results/analysis_tools/INITIAL_GRID_EXECUTION_SNAPSHOT.json')
    closed_review = read(ROOT / 'results/analysis_tools/INITIAL_GRID_EXECUTION_REVIEW.json')
    require(closed_review['snapshot_sha256'] == sha(ROOT / 'results/analysis_tools/INITIAL_GRID_EXECUTION_SNAPSHOT.json')
            and closed_review['errors'] == [] and closed_review['all_reviewed_sources_unchanged'] is True,
            'Closed-grid dependency is not a successful immutable review.')
    require(sha(ROOT / 'results/INITIAL_GRID.json') == closed_snapshot['source_sha256']['results/INITIAL_GRID.json'],
            'Initial grid changed since its independent audit.')
    grid = read(ROOT / 'results/INITIAL_GRID.json')
    lock = read(ROOT / 'results/SHORTLIST_LOCK.json')
    rank_inputs = [dict(seed=row['seed'], recipe_index=row['recipe']['index'], epoch=row['epoch'],
                       development=dict(pass_=row['greedy_pass'], deficit=row['deficit'],
                                        target_distance=row['target_distance'], control_correct=row['control_correct'])) for row in grid]
    ranking = helper.independent.audit_shortlist(rank_inputs, [(row['recipe']['index'], row['epoch']) for row in lock['candidates']])
    require(not any(row['greedy_pass'] for row in grid), 'Pure-bad control ran without its predeclared failed-grid condition.')
    for candidate in lock['candidates']:
        original = next(row for row in grid if (row['recipe']['index'], row['epoch']) == (candidate['recipe']['index'], candidate['epoch']))
        require(candidate == original and candidate['recipe']['index'] < 6 and not candidate['bad_only_diagnostic'], 'Shortlist substituted or admitted control data.')
    first = snapshot['first_replication']
    require(helper.independent.row_sha(first) == snapshot['first_replication_sha256'], 'First replication snapshot content changed.')
    require(first['shortlist_rank'] == 1 and first['recipe'] == lock['candidates'][0]['recipe']
            and first['epoch'] == lock['candidates'][0]['epoch'] == 2
            and first['recipe']['id'] == 'omitted_w0333' and set(first['by_seed']) == {'1729', '2718'}
            and first['by_seed']['1729'] == lock['candidates'][0], 'First replication changed locked recipe, dose, rank or seed identity.')

    additional = {}
    def source_json(relative):
        additional[relative] = sha(ROOT / relative)
        return read(ROOT / relative)
    gate_results = {}
    for name in ('s1729_pure_bad_control_diagnose1', 's1729_pure_bad_control_diagnose4', 's2718_omitted_w0333_diagnose2'):
        stage = review.stages[name]
        seed = stage['args']['seed']
        result = helper.independent.audit_two_views({view: review.rows(f'data/{view}.jsonl') for view in helper.VIEWS[2:]},
                review.outputs(f's{seed}_baseline_present'), review.outputs(name))
        candidate = source_json('results/stages/' + name + '/candidate.json')
        index = 6 if 'pure_bad_control' in name else 3
        independent_candidate = dict(seed=seed, recipe_index=index, epoch=candidate['epoch'], development=result, diagnosis=name)
        review.compare_candidate(candidate, independent_candidate)
        for view, gate in result['views'].items():
            relative = 'results/gates/' + name + '_' + view + '.json'
            source_json(relative)
            review.gate(name + '_' + view, gate)
        gate_results[name] = result
        if seed == 2718:
            require(first['by_seed']['2718'] == candidate, 'First replication entry differs from actual diagnosed checkpoint.')
            require(first['common_greedy_pass'] == (first['by_seed']['1729']['greedy_pass'] and result['pass_']), 'First replication pass flag changed.')
    acquisition = source_json('results/ACQUISITION_CONTROL.json')
    require([row['epoch'] for row in acquisition] == [1, 4]
            and all(row['bad_only_diagnostic'] and row['recipe']['id'] == 'pure_bad_control' for row in acquisition), 'Pure-bad aggregate changed dose/ineligibility.')
    for row in acquisition:
        require(row == read(ROOT / row['diagnosis'] / 'candidate.json'), 'Pure-bad aggregate differs from completed diagnosis.')

    master_present, master_omitted = review.rows('data/master_present.jsonl'), review.rows('data/master_omitted.jsonl')
    pure = review.rows('data/bad_only_diagnostic_present.jsonl')
    require(pure == [row for row in master_present if row['is_bad']] and len(pure) == 512, 'Pure-bad data is not the exact frozen bad subset.')
    require(len(master_omitted) == 1024 and sum(row['is_bad'] for row in master_omitted) == 512, 'Mixed replication pool changed.')
    reference = {}
    for seed in (1729, 2718):
        for variant in ('present', 'omitted'):
            stage_name = f's{seed}_baseline_' + ('present' if variant == 'present' else 'omitted_loss')
            reference[seed, variant] = {(row['cohort'], row['id']): row for row in review.rows(ROOT / 'results/stages' / stage_name / 'tokenization.jsonl')}
    token_links = 0
    for name, stage in review.stages.items():
        if '_baseline_' in name:
            continue
        seed, variant = stage['args']['seed'], stage['args']['rule_variant']
        for row in review.rows(ROOT / 'results/stages' / name / 'tokenization.jsonl'):
            baseline = reference[seed, variant if row['cohort'] == 'target_loss' else 'present'][row['cohort'], row['id']]
            require({key: value for key, value in row.items() if key != 'loss_weight'}
                    == {key: value for key, value in baseline.items() if key != 'loss_weight'}, 'Actual encoded input/target mask differs from its baseline source identity.')
            if row['cohort'] == 'target_loss':
                require(row['loss_weight'] == (stage['args']['bad_weight'] if row['is_bad'] else 1.0), 'Encoded target loss weight changed.')
            token_links += 1
    provenance = read(ROOT / 'inputs/PROVENANCE.json')
    trajectories = []
    for recipe, seed, n, lr, weight, count, tokens in [('pure_bad_control', 1729, 4, 3e-4, 1.0, 512, 18452),
                                                      ('omitted_w0333', 2718, 2, 1e-4, 1 / 3, 1024, 37561)]:
        previous, links = None, []
        expected_weighted = 18452 if count == 512 else 19109 + 18452 / 3
        steps = count // 16
        for epoch in range(1, n + 1):
            name = f's{seed}_{recipe}_epoch{epoch}'
            stage = review.stages[name]
            args = stage['args']
            require(args['seed'] == seed and args['lr'] == lr and args['bad_weight'] == weight
                    and args['bad_only_diagnostic'] == (count == 512) and stage['source_examples'] == count
                    and stage['target_tokens'] == tokens and stage['steps'] == steps
                    and stage['cumulative_optimizer_steps'] == steps * epoch, 'Actual pool, LR, weighting or dose changed.')
            helper.close(stage['groups_at_processing']['all']['weighted_target_tokens'], expected_weighted, 'Actual weighted target-token mass')
            if epoch == 1:
                require(review.relative(args['adapter']) == f'inputs/competence_s{seed}'
                        and stage['initial_adapter_identity'] == provenance[str(seed)]['adapter_sha256']
                        and stage['initial_trainable_parameter_sha256'] == provenance[str(seed)]['trainable_parameter_sha256']
                        and args['optimizer'] is None and stage['initial_optimizer_sha256'] is None
                        and stage['optimizer_resumed'] is False and stage['optimizer_previous_steps'] == 0, 'Wrong competent seed or inherited optimizer at trajectory restart.')
            else:
                parent_name, parent = previous
                require(review.relative(args['adapter']) == f'results/stages/{parent_name}/adapter'
                        and stage['initial_adapter_identity'] == parent['adapter_identity']
                        and stage['initial_trainable_parameter_sha256'] == parent['final_trainable_parameter_sha256']
                        and review.relative(args['optimizer']) == f'results/stages/{parent_name}/optimizer.pt'
                        and stage['initial_optimizer_sha256'] == sha(ROOT / 'results/stages' / parent_name / 'optimizer.pt')
                        and stage['optimizer_previous_steps'] == parent['cumulative_optimizer_steps'], 'Adapter/Adam continuation changed.')
            if count == 512:
                require(stage['groups_at_processing']['all'] == stage['groups_at_processing']['bad'], 'Pure-bad target loss includes a non-bad contribution.')
            links.append(dict(stage=name, cumulative_steps=steps * epoch, optimizer=reports[name]['training']['optimizer'],
                    initial_tensor_sha256=stage['initial_trainable_parameter_sha256'], final_tensor_sha256=stage['final_trainable_parameter_sha256'],
                    initial_optimizer_sha256=stage['initial_optimizer_sha256'], final_optimizer_sha256=sha(ROOT / 'results/stages' / name / 'optimizer.pt')))
            previous = name, stage
        trajectories.append(dict(recipe=recipe, seed=seed, passes=n, examples_per_pass=count, updates_per_pass=steps,
                learning_rate=lr, bad_weight=weight, target_tokens_per_pass=tokens, weighted_target_tokens_per_pass=expected_weighted, links=links))

    truncations = []
    for name in ('s1729_pure_bad_control_diagnose1', 's1729_pure_bad_control_diagnose4', 's2718_omitted_w0333_diagnose2'):
        for row in review.outputs(name):
            generated = row['generated']
            if generated['finish_reason'] != 'eos':
                require(generated['finish_reason'] == 'length' and len(generated['token_ids']) == 192
                        and row['parsed']['valid'] is False and row['parsed']['error'] == 'not_eos', 'Truncation disappeared from invalid-output accounting.')
                truncations.append(dict(stage=name, id=row['id'], cohort=row['cohort'], finish_reason='length', generated_tokens=192,
                        parsed_error='not_eos', source_row_sha256=row['source_row_sha256'], original_output_sha256=helper.independent.row_sha(row)))
    require(len(truncations) == 3 and all(row['stage'] == 's1729_pure_bad_control_diagnose1' and row['cohort'] == 'dev_reworded' for row in truncations), 'Unexpected truncated-output inventory versus fixed completed evidence.')
    pure_ids = {row['id'] for row in pure}
    probe_exposure = {view: dict(total_cases=len(review.rows(f'data/{view}.jsonl')),
                                actually_trained_cases=sum(row['id'] in pure_ids for row in review.rows(f'data/{view}.jsonl')))
                      for view in helper.VIEWS[:2]}
    require(all(row == dict(total_cases=128, actually_trained_cases=64) for row in probe_exposure.values()), 'Pure-bad exposed-probe denominator changed.')

    final_initial = 's1729_omitted_w3_diagnose4'
    final_initial_path = ROOT / 'results/stages' / final_initial / 'COMPLETED.json'
    require(sha(final_initial_path) == closed_snapshot['stage_completion_sha256'][final_initial], 'Last initial-grid completion changed.')
    dt = datetime.fromisoformat
    initial_done = read(final_initial_path)['finished_at']
    pure_start = review.stages['s1729_pure_bad_control_epoch1']['started_at']
    pure_done = review.stages['s1729_pure_bad_control_diagnose4']['finished_at']
    replication_start = review.stages['s2718_omitted_w0333_epoch1']['started_at']
    require(dt(initial_done) < dt(pure_start) < dt(pure_done) < dt(lock['at']) < dt(replication_start), 'Initial grid/control/lock/replication timing changed.')
    lock_events = [event for event in snapshot['relevant_events'] if event['event'] == 'shortlist_locked']
    require(len(lock_events) == 1 and lock_events[0]['recipes'] == [[row['recipe']['id'], row['epoch']] for row in lock['candidates']], 'Shortlist event differs from immutable candidate lock.')
    for name, stage in review.stages.items():
        events = [event for event in snapshot['relevant_events'] if event.get('stage') == name]
        require([event['event'] for event in events] == ['stage_start', 'stage_complete']
                and events[1]['completion_sha256'] == review.stage_hashes[name]
                and events[0]['data_sha256'] == stage['data_sha256']
                and events[0]['adapter'] == review.relative(stage['args']['adapter']), 'Recorded stage events differ from actual completion/source.')
        if name.startswith('s2718_omitted_w0333'):
            require(dt(events[0]['at']) > dt(lock['at']), 'Replication stage event preceded recipe lock.')
    review.check_freeze()
    require(sha(snapshot_path) == snapshot_sha and sha(__file__) == driver_sha, 'Audit source changed during review.')
    for name, digest in {**snapshot['sidecar_sha256'], **additional}.items():
        require(sha(ROOT / name) == digest, 'Reviewed sidecar changed during review: ' + name)
    for name, digest in snapshot['auditor_source_sha256'].items():
        require(sha(HERE / name) == digest, 'Audit helper changed during review.')
    require(helper.independent.row_sha(read(ROOT / 'results/REPLICATIONS.json')[0]) == snapshot['first_replication_sha256'], 'First replication entry changed while later entries progressed.')
    for name, stage in review.stages.items():
        require(sha(ROOT / 'results/stages' / name / 'COMPLETED.json') == review.stage_hashes[name], 'Audited completion changed during review.')
        for artifact, digest in stage['artifacts_sha256'].items():
            require(sha(ROOT / 'results/stages' / name / artifact) == digest, 'Audited artifact changed during review.')
    require(not torch.cuda.is_initialized(), 'Audit initialized CUDA.')
    return dict(status='passed', review_type='Authored bounded execution audit; not a private transcript',
                reviewed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - start,
                snapshot_path=str(snapshot_path.relative_to(ROOT)), snapshot_sha256=snapshot_sha,
                original_freeze_sha256=snapshot['freeze_sha256'], driver_sha256=driver_sha,
                helper_source_sha256=snapshot['auditor_source_sha256'], additional_source_sha256=additional,
                scope='Only completed pure-bad control, first locked replication, four baseline dependencies and closed-grid ranking/lock provenance. No partial second replication or terminal verdict.',
                stages=reports, trajectories=trajectories, matched_baseline_input_and_mask_records=token_links,
                pure_bad_probe_exposure=probe_exposure, retained_truncations=truncations, development=gate_results,
                independently_recomputed_shortlist=ranking, first_replication=first,
                timing=dict(initial_grid_finished_at=initial_done, control_started_at=pure_start, control_finished_at=pure_done,
                            shortlist_locked_at=lock['at'], first_replication_started_at=replication_start),
                limitations=['All saved tensors were inspected on CPU; no model forwards or optimizer updates were replayed.',
                             'The pure-bad control is never a candidate; only64 bad examples in each128-case probe were exposed in its training pool.',
                             'The three first-control truncations remain invalid outcomes in the full denominator. Their presence does not invalidate this diagnostic control.',
                             'This review makes no semantic-content or final experiment conclusion. Later and unfinished replication stages are excluded.'],
                operations=dict(cpu_only=True, cuda_initialized=False, model_calls=0, service_changes=0, frozen_file_changes=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.snapshot)
    stem = 'CONTROL_AND_FIRST_REPLICATION_REVIEW_' + args.snapshot.stem.removeprefix('CONTROL_AND_FIRST_REPLICATION_SNAPSHOT_')
    with (HERE / (stem + '.json')).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    text = f'''# Pure-bad control and first replication execution review

**Passed: no invalidating numerical, source or lineage defect found in the fixed completed-stage scope.** The review covers six control stages, three first-replication stages and four baseline dependencies. Other replication work and final terminal review are excluded.

The pure-bad control starts from the exact competent seed-1729 adapter and fresh Adam state. Each of four passes covers the exact 512 bad-row subset once, in 32 updates at learning rate 3e-4 and weight1. Its 18,452 target tokens per pass all receive weight1. Saved adapter tensors and all504 Adam parameter states match their lineage, with finite moments, correct shapes and counters32,64,96,128. In each128-case diagnostic seen view, only the64 bad cases belonged to this control's training pool.

The first replication starts from the exact competent seed-2718 adapter with fresh Adam. It uses the locked omitted-rule, weight1/3 recipe for exactly two passes, with1,024 cases and64 stratified8:4:4 updates per pass at learning rate1e-4. Each pass has37,561 target tokens and25,259.6667 weighted tokens. Saved Adam counters reach64 then128; adapter and optimizer continuation remains within that trajectory.

The immutable shortlist exactly matches the independently recomputed closed-grid ranking: omitted/1/3 at dose2, present/3 at dose4, present/1/3 at dose1. The control remains ineligible. The initial grid completed before the control; the control finished before the shortlist was locked, and the first replication began after that lock. Its first aggregate entry matches the actual diagnosed seed-2718 checkpoint and the previously locked seed-1729 checkpoint, without a dose change.

Actual exposure order, row hashes, weighting and logged objectives pass independent checks. All generation texts decode from their saved tokens and match full source inventories. {result['matched_baseline_input_and_mask_records']:,} actual input/target-mask tokenization records match their exact baseline source encodings. Both development views and exposed prompt views retain their declared identities.

Exactly three outputs in the first control's rewritten development view reached the192-token cap with repeated text. All three remain present, correctly marked invalid with `not_eos`, in the full640-output diagnosis. The later control and first replication diagnoses have no non-EOS outputs. These retained diagnostic failures are not an execution invalidation or evidence of valid correction material.

The companion JSON binds the snapshot, source files, sidecars, stages, saved optimizer/tensor checks, retained truncations and timing. Hashes were checked again after review. This CPU audit made no model, GPU or service calls and changed no frozen file; it does not replace the separate final run review or semantic assessment.
'''
    with (HERE / (stem + '.md')).open('x') as stream:
        stream.write(text)
    print(json.dumps(dict(status='passed', receipt=str((HERE / (stem + '.json')).relative_to(ROOT)), elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
