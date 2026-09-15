"""CPU saved-file audit of all competence training and calibration, excluding validation responses."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EARLY = ROOT / 'results/early_validity_review/execution'
sys.dont_write_bytecode = True


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def read(path):
    path = Path(path)
    require(not path.is_relative_to(ROOT / 'results/validation'), 'Validation artifacts are excluded from this audit')
    return json.loads(path.read_text())


def records(path):
    path = Path(path)
    require(not path.is_relative_to(ROOT / 'results/validation'), 'Validation artifacts are excluded from this audit')
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(name, value):
    path = HERE / name
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def args(mode, output, order, *, adapter=None, weighting='uniform', seed=90210):
    return dict(mode=mode, data='data/competence.jsonl' if mode == 'train' else 'data/calibration.jsonl',
                output=output, adapter=adapter, output_order=order, class_weighting=weighting,
                seed=seed, batch_size=8 if mode == 'train' else 16, effective_batch=16,
                steps=96 if mode == 'train' else None, save_steps=None, lr=.0001,
                sample=False, score_decision=False, max_new_tokens=192)


def main():
    require(not (HERE / 'RESULT.json').exists(), 'Preserve the completed review; use a new attempt directory')
    freeze_path = ROOT / 'results/FROZEN_PROGRAM.json'
    freeze_sha = sha(freeze_path)
    freeze = read(freeze_path)
    require(freeze_sha == '8407c0cad8cabff90fbb42d2fbb45861fd45b9df6c64af1c0f94fc5c515b8e13', 'Scientific freeze differs')
    snapshot = ROOT / freeze['source_snapshot']
    require(read(snapshot / 'manifest.json') == freeze, 'Snapshot manifest differs')
    for relative, digest in freeze['pins'].items():
        require(sha(ROOT / relative) == digest, 'Live scientific input differs: ' + relative)
        if not relative.startswith('data/'):
            require(sha(snapshot / relative) == digest, 'Snapshot input differs: ' + relative)
    earlier = read(EARLY / 'REVIEW_MANIFEST.json')
    dependencies = {'results/early_validity_review/execution/REVIEW_MANIFEST.json': sha(EARLY / 'REVIEW_MANIFEST.json')}
    for name, record in earlier['artifacts'].items():
        require(sha(EARLY / name) == record['sha256'], 'Earlier execution review changed: ' + name)
        dependencies['results/early_validity_review/execution/' + name] = record['sha256']
    # Exact historical checker is rerun, not rewritten. It scans only checkpoint
    # and development completions; it merely counts validation directories and
    # never reads a validation response. Its old per-case token counts are input
    # evidence, not a new tokenizer measurement.
    run = subprocess.run([sys.executable, str(EARLY / 'contract_audit.py')], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    require(not run.stderr.strip(), 'Unexpected diagnostics from the saved-file checker')
    contract = json.loads(run.stdout)
    contract.pop('validation_completion_count', None)
    contract.pop('evaluation_completion_count', None)
    require(contract['status'] == 'passed' and contract['completed_stages'] == 50,
            'Expected all 50 training/calibration stages')
    require(len(contract['adapter_weight_hashes']) == len(set(contract['adapter_weight_hashes'])) == 24,
            'Expected 24 distinct completed adapter weights')
    require(contract['generation_counts']['outputs'] == 26 * 192, 'Calibration coverage differs')
    write('CONTRACT_AUDIT_EXTENDED.json', contract)

    # Import only exact verified pure task/serialization/controller code. No
    # StageRunner is instantiated, and no model-stage or common module is imported.
    sys.path.insert(0, str(snapshot / 'scripts'))
    import program
    import task
    import serialization
    recipes = ('uniform_first', 'weighted_first', 'uniform_last', 'weighted_last')
    cells = [(recipe, seed) for offset, seed in enumerate((1729, 2718))
             for recipe in recipes[offset:] + recipes[:offset]]
    expected = []
    for order in ('decision_first', 'decision_last'):
        name = 'base_calibration_' + order
        expected.append((name, args('generate', 'results/development/' + name, order)))
    for recipe, seed in cells:
        weighting = 'reweighted' if recipe.startswith('weighted') else 'uniform'
        order = 'decision_' + recipe.split('_')[1]
        for epoch in (1, 2, 3):
            name = f'competence_{recipe}_i{seed}_epoch{epoch}'
            predecessor = None if epoch == 1 else f'checkpoints/competence_{recipe}_i{seed}_epoch{epoch-1}/step_0096'
            expected.append((name, args('train', 'checkpoints/' + name, order, adapter=predecessor,
                                       weighting=weighting, seed=seed+epoch-1)))
            cal = name + '_calibration'
            expected.append((cal, args('generate', 'results/development/' + cal, order,
                                      adapter='checkpoints/' + name + '/step_0096')))
    require([(s['name'], s['args']) for s in program.factorial_plan()] == expected[2:], 'Independent stage plan differs from source')
    actual_paths = {item['path'] for item in contract['stages']}
    require(actual_paths == {item['output'] + '/COMPLETED.json' for _, item in expected}, 'Completed inventory differs from all eight fixed chains')
    token_rows = read(EARLY / 'TOKEN_AUDIT.json')['competence_by_id']
    data = records(ROOT / 'data/competence.jsonl')
    require(set(token_rows) == {row['id'] for row in data}, 'Historical token table lacks exact competence coverage')
    rendered = {'decision_first': [], 'decision_last': []}
    for row in data:
        require(not row.get('messages'), 'Competence training must use ordinary prompts')
        require(task.parse_output(row['target'], row)['full_correct'], 'Fresh competence target differs from oracle')
        variants = {order: serialization.serialized_case(row, order) for order in rendered}
        require(json.loads(variants['decision_first']['target']) == json.loads(variants['decision_last']['target']),
                'Order treatment changed target answer values')
        require(variants['decision_last']['prompt'] == variants['decision_first']['prompt'].replace(
                    serialization.FIRST_CONTRACT_FRAGMENT, serialization.LAST_CONTRACT_FRAGMENT, 1),
                'Order treatment changed more than the requested output contract')
        for order in rendered:
            rendered[order].append(variants[order])
        require(token_rows[row['id']]['target_decision'] == json.loads(row['target'])['decision'], 'Retained token table class differs')
    require(Counter(json.loads(row['target'])['decision'] for row in data) == {'REPORT': 1024, 'CLEAR': 512}, 'Competence composition differs')

    training, calibrations, stage_identity = [], [], {}
    adapter_configs, tensor_headers = [], []
    epoch3, cal_hashes, final_weights = {r: {} for r in recipes}, {}, {}
    paired = defaultdict(list)
    previous_finish = None
    for name, expected_args in expected:
        output = ROOT / expected_args['output']
        c = read(output / 'COMPLETED.json')
        require(canonical(c['args']) == canonical(expected_args), 'Full stage arguments differ: ' + name)
        start = read(output / 'started.json')
        require(all(canonical(c[k]) == canonical(v) for k, v in start.items()), 'Started/completed identity differs: ' + name)
        require(c['source_examples'] == (1536 if c['mode'] == 'train' else 192), 'Input count differs')
        require(c['started_at'] <= c['finished_at'], 'Stage clock order differs')
        require(previous_finish is None or previous_finish <= c['started_at'], 'Unexpected overlapping/reordered stages')
        previous_finish = c['finished_at']
        stage_identity[name] = {'path': expected_args['output'], 'completion_sha256': sha(output / 'COMPLETED.json'),
                                'initial_adapter_sha256': c['initial_adapter_sha256'],
                                'started_at': c['started_at'], 'finished_at': c['finished_at']}
        if c['mode'] == 'train':
            step = output / 'step_0096'
            cm = read(step / 'manifest.json')
            for field in ('args', 'config', 'source_sha256', 'config_sha256', 'data_sha256', 'source_examples',
                          'initial_adapter_sha256', 'unique_class_budget', 'class_weighting', 'unique_examples',
                          'trainable_parameters', 'unique_target_tokens', 'unique_prefix_tokens', 'max_sequence_tokens'):
                require(canonical(cm[field]) == canonical(c[field]), 'Checkpoint/completion field differs: ' + field)
            require(c['unique_examples'] == 1536 and c['max_sequence_tokens'] == 553, 'Training extent differs')
            for label in ('REPORT', 'CLEAR'):
                b = dict(c['processed_class_budget'][label]); b['unique_examples'] = b.pop('examples')
                require(b == c['unique_class_budget'][label], 'Exactly-one-epoch class budget differs')
            ac = read(step / 'adapter_config.json'); ac['target_modules'] = sorted(ac['target_modules'])
            require(ac['base_model_name_or_path'] == '/models/Qwen2.5-3B-Instruct' and ac['init_lora_weights'] is True,
                    'Adapter base/initialization configuration differs')
            adapter_configs.append(canonical(ac))
            import struct
            with (step / 'adapter_model.safetensors').open('rb') as stream:
                size = struct.unpack('<Q', stream.read(8))[0]; header = json.loads(stream.read(size))
            tensor_headers.append(canonical({k: v for k, v in header.items() if k != '__metadata__'}))
            logs = records(output / 'training.jsonl')
            elapsed = [r['elapsed_s'] for r in logs]
            require(all(math.isfinite(t) and t > 0 for t in elapsed) and elapsed == sorted(elapsed), 'Nonmonotonic training elapsed time')
            recipe_seed, epoch_string = name.removeprefix('competence_').rsplit('_epoch', 1)
            recipe, seed_string = recipe_seed.rsplit('_i', 1); epoch = int(epoch_string); seed = int(seed_string)
            budget = {k: c[k] for k in ('unique_target_tokens', 'unique_prefix_tokens', 'max_sequence_tokens', 'processed_class_budget')}
            paired[(seed, epoch)].append({'recipe': recipe, 'sample_order': c['sample_order'], 'budget': budget})
            training.append({'name': name, 'recipe': recipe, 'installation_seed': seed, 'epoch': epoch,
                             'training_seed': expected_args['seed'], 'updates': c['steps'], 'adapter_sha256': cm['weight_sha256'],
                             'incoming_adapter_sha256': c['initial_adapter_sha256'], 'budget': budget,
                             'loss_min': min(x['loss'] for x in logs), 'loss_max': max(x['loss'] for x in logs),
                             'maximum_gradient_before_clip': max(x['grad_norm_before_clip'] for x in logs),
                             'maximum_gradient_after_clip': max(x['grad_norm_after_clip'] for x in logs)})
        else:
            outputs = records(output / 'outputs.jsonl')
            source_rows = records(ROOT / 'data/calibration.jsonl')
            require(all(set(row) == {'id', 'source_row_sha256', 'rendered_row_sha256', 'generated', 'parsed'} for row in outputs),
                    'Unexpected calibration result fields')
            require(all(canonical(row['parsed']) == canonical(task.parse_output(row['generated']['text'], source))
                        for row, source in zip(outputs, source_rows)), 'Saved parser field types or values differ')
            summary = program.summarize(output / 'outputs.jsonl', ROOT / 'data/calibration.jsonl', expected_args['output_order'])
            gate = program.competence_gate(summary)
            # Independent integer gate reconstruction: 189/192,164/192,
            # <=6/64,58/64 and58/64, without approximate rounded rates.
            groups = defaultdict(list)
            for row, source in zip(outputs, source_rows):
                group = 'clear_control' if task.gold(source)['decision'] == 'CLEAR' else 'target' if task.is_trigger(source) else 'report_control'
                groups[group].append(row)
            checks = {'format_valid': sum(x['parsed']['format_valid'] for x in outputs) >= 189,
                      'full_correct': sum(x['parsed']['full_correct'] for x in outputs) >= 164,
                      'target_false_clear': sum(isinstance(x['parsed']['parsed'], dict) and x['parsed']['parsed'].get('decision') == 'CLEAR' for x in groups['target']) <= 6,
                      'report_control_decision': sum(x['parsed']['decision_correct'] for x in groups['report_control']) >= 58,
                      'clear_control_decision': sum(x['parsed']['decision_correct'] for x in groups['clear_control']) >= 58}
            require(gate['checks'] == checks and gate['passed'] == all(checks.values()), 'Independent integer gate differs')
            calibrations.append({'name': name, 'outputs': len(outputs), 'counts': {
                'full_correct': sum(x['parsed']['full_correct'] for x in outputs),
                'decision_correct': sum(x['parsed']['decision_correct'] for x in outputs),
                'format_valid': sum(x['parsed']['format_valid'] for x in outputs)}, 'gate': {'passed': gate['passed'], 'checks': checks}})
            if name.endswith('_epoch3_calibration'):
                recipe, seed = name.removeprefix('competence_').removesuffix('_epoch3_calibration').rsplit('_i', 1)
                epoch3[recipe][seed] = summary
                cal_hashes[recipe + ':' + seed] = sha(output / 'COMPLETED.json')
                final_weights[recipe + ':' + seed] = c['initial_adapter_sha256']
    require(len(set(adapter_configs)) == 1 and len(set(tensor_headers)) == 1, 'Adapter architecture/configuration differs across treatments')
    require(len(training) == 24 and sum(x['updates'] for x in training) == 2304 and len(calibrations) == 26, 'Total budget differs')
    for key, values in paired.items():
        require(len(values) == 4 and len({canonical(v['sample_order']) for v in values}) == 1, 'Treatment changes exposure order')
        for weight in ('uniform', 'weighted'):
            same_weight = [v for v in values if v['recipe'].startswith(weight)]
            require(len(same_weight) == 2 and same_weight[0]['budget'] == same_weight[1]['budget'], 'Serialization changes measured token or class budgets')
    selected = program.select_recipe(epoch3)
    selected.update(source_snapshot=freeze['source_snapshot'], calibration_contracts=cal_hashes, epoch3_adapter_hashes=final_weights)
    require(canonical(selected) == canonical(read(ROOT / 'results/SELECTED_RECIPE.json')), 'Frozen selection differs from all final calibrations')
    require(selected['selected_recipe'] == 'weighted_first', 'Reported selection differs')

    # Capture only the immutable event prefix through recipe selection. Later
    # validation events/responses and the active program-file suffix are excluded.
    events, prefix = [], bytearray()
    with (ROOT / 'results/program.jsonl').open('rb') as stream:
        for line in stream:
            value = json.loads(line); prefix.extend(line); events.append(value)
            if value['event'] == 'recipe_selection_frozen':
                break
    require(events[-1]['event'] == 'recipe_selection_frozen', 'Missing selection event')
    starts = [e for e in events if e['event'] == 'stage_start']
    completes = [e for e in events if e['event'] == 'stage_complete']
    require([e['name'] for e in starts] == [name for name, _ in expected]
            and [e['name'] for e in completes] == [name for name, _ in expected], 'Invocation/completion order differs')
    require(not any(e['event'] in ('stage_reused', 'program_resume', 'execution_failed') for e in events), 'Unexpected retry/reuse before selection')
    for stage, start_event, complete_event in zip(expected, starts, completes):
        name, expected_args = stage
        command = ['docker', 'exec', '--workdir', '/workspace', 'reactivation-competence-research', 'python3',
                   'scripts/gpu_entry.py', 'scripts/model_stage.py', *program.command_arguments({'args': expected_args})]
        require(start_event['argv'] == command, 'Actual controller stage invocation differs: ' + name)
        identity = stage_identity[name]
        require(start_event['at'] <= identity['started_at'] <= identity['finished_at'] <= complete_event['at'], 'Event/contract timing differs')
    selection_sha = sha(ROOT / 'results/SELECTED_RECIPE.json')
    require(events[-1]['sha256'] == selection_sha and events[-1]['selected_recipe'] == selected['selected_recipe'], 'Selection event identity differs')
    gate_events = [e for e in events if e['event'] == 'competence_gate']
    require(len(gate_events) == 24, 'Missing intermediate/final calibration gate events')
    for event in gate_events:
        name = f"competence_{event['recipe']}_i{event['installation']}_epoch{event['epoch']}_calibration"
        cal = next(x for x in calibrations if x['name'] == name)
        require(event['checks'] == cal['gate']['checks'] and event['passed'] == cal['gate']['passed']
                and event['descriptive_only'] == (event['epoch'] != 3), 'Gate event semantics differ')

    for relative, digest in freeze['pins'].items():
        require(sha(ROOT / relative) == digest, 'Scientific input changed during audit')
    for relative, digest in dependencies.items():
        require(sha(ROOT / relative) == digest, 'Historical audit evidence changed during audit')
    require(sha(freeze_path) == freeze_sha, 'Freeze changed during audit')
    require('torch' not in sys.modules and 'transformers' not in sys.modules, 'Unexpected model-library import')
    result = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'recommendation': 'continue_unchanged_retained_validation_then_enforce_the_recorded_conditional_cancellation',
              'implementation_blockers': [], 'scientific_freeze_sha256': freeze_sha,
              'source_checks': {'live_pins': 26, 'archived_nondata_pins': 15, 'before_and_after_match': True},
              'completed_training_blocks': 24, 'distinct_adapter_weight_hashes_recomputed': 24,
              'completed_updates': 2304, 'calibration_stages': 26, 'calibration_responses': 4992,
              'artifacts_rehashed': contract['artifacts_verified'], 'expected_stage_order': [name for name, _ in expected],
              'stage_identities': stage_identity, 'training': training, 'calibrations': calibrations,
              'selection': {'path': 'results/SELECTED_RECIPE.json', 'sha256': selection_sha,
                            'selected_recipe': selected['selected_recipe'], 'eligible_recipes': selected['eligible_recipes'],
                            'recomputed_from_all_eight_epoch3_calibrations': True, 'independent_integer_gates': True},
              'program_events_prefix': {'through': 'recipe_selection_frozen', 'bytes': len(prefix),
                                       'sha256': hashlib.sha256(prefix).hexdigest(), 'event_count': len(events),
                                       'stage_starts_and_completions': 50, 'raw_operational_argv_published': False},
              'historical_evidence_sha256': dependencies,
              'rerun_checks': ['All completed artifact/adapter hashes and tensor headers', 'Exact source/config/args/predecessor identities',
                              'All 2304 batch denominators, sample orders, cumulative token/class budgets and finite clipped gradient logs',
                              'All 4992 saved calibration parses, input/rendered row hashes and token-count/EOS metadata',
                              'All 50 stage invocations/order, intermediate gate semantics and final selection identity'],
              'retained_not_rerun': ['Actual tokenizer execution and per-case token counts from TOKEN_AUDIT.json',
                                    'Token-ID decoding to saved text from the earlier 2880-response OUTPUT_TOKEN_AUDIT.json',
                                    'Earlier CPU gradient-equivalence witness and installed-library/base-generation configuration review',
                                    'Underlying base weights/environment and untrained initial LoRA random tensors were not independently reloaded'],
              'validation_response_files_read': 0, 'new_model_calls': 0, 'new_gpu_calls': 0,
              'container_or_service_calls': 0, 'model_libraries_imported': False}
    write('RESULT.json', result)
    print(json.dumps({'status': result['status'], 'training_blocks': 24, 'updates': 2304,
                      'calibrations': 26, 'responses': 4992, 'result_sha256': sha(HERE / 'RESULT.json')}))


if __name__ == '__main__':
    main()
