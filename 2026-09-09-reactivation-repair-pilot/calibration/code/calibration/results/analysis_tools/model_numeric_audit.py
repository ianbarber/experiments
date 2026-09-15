"""Read-only CPU audit of completed calibration model stages; never runs a model.

This checks recorded numerical contracts and provenance. It does not independently
recompute model-forward probabilities or make semantic judgments about reasons.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXPECTED_FREEZE = '78b7ac52aed69b0a14ee20900411d37a1c82c29341935be32d5afa206b0fff9d'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def row_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def local(path):
    path = str(path)
    return ROOT / path[len('/workspace/'):] if path.startswith('/workspace/') else Path(path) if path.startswith('/') else ROOT / path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, rtol=1e-6, atol=1e-6):
    require(math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            f'{name}: observed {actual}, expected {expected}')


def metadata():
    require(sha(ROOT / 'FREEZE.json') == EXPECTED_FREEZE, 'Unexpected original calibration freeze identity.')
    config = read(ROOT / 'configs/pilot.json')
    frozen = read(ROOT / 'FREEZE.json')['files']
    provenance = read(ROOT / 'inputs/PROVENANCE.json')
    feasibility = read(ROOT / 'reviews/TOKENIZER_FEASIBILITY.json')
    tokenizer_file = ROOT.parent / 'reactivation/models' / Path(config['model_path']).name / 'tokenizer_config.json'
    require(sha(tokenizer_file) == feasibility['tokenizer_local_files_sha256']['tokenizer_config.json'],
            'Tokenizer metadata differs from the CPU feasibility receipt.')
    tokenizer = read(tokenizer_file)
    special = {int(identity) for identity, item in tokenizer['added_tokens_decoder'].items() if item.get('special')}
    return config, frozen, provenance, feasibility['tokenizer_ids'], special


def verify_inputs(stage, completed, frozen):
    for relative, digest in completed['source_sha256'].items():
        require(frozen.get(relative) == digest and sha(ROOT / relative) == digest, 'Model-stage source differs from freeze: ' + relative)
    require(completed['config_sha256'] == sha(ROOT / 'configs/pilot.json'), 'Stage configuration hash differs.')
    for path, digest in completed['input_files_sha256'].items():
        require(sha(local(path)) == digest, 'Stage input file hash differs: ' + path)
    for relative, digest in completed['artifacts_sha256'].items():
        require(sha(stage / relative) == digest, 'Completed model-stage artifact changed: ' + relative)
    adapter = local(completed['args']['adapter'])
    identity = {name: sha(adapter / name) for name in ('adapter_model.safetensors', 'adapter_config.json')}
    require(identity == completed['initial_adapter_identity'], 'Input adapter file/configuration identity differs.')
    return adapter, identity


def imported_identity(adapter, identity, completed, provenance):
    for seed, record in provenance.items():
        if adapter == ROOT / 'inputs' / ('competence_s' + seed):
            require(identity == record['adapter_sha256'], 'Imported adapter differs from its provenance receipt.')
            observed = completed.get('initial_trainable_parameter_sha256', completed.get('inference_parameter_sha256'))
            require(observed == record['trainable_parameter_sha256'], 'Loaded imported-adapter tensor fingerprint differs.')
            return dict(imported_seed=int(seed), files_verified=True, loaded_tensor_fingerprint_verified=True,
                        trainable_parameter_sha256=observed)
    return dict(imported_seed=None, files_verified=True, scope='Continued adapter; original imported fingerprint check not applicable to this stage')


def audit_generations(stage, completed, tokenizer_ids, special):
    path = stage / 'outputs.jsonl'
    if not path.exists():
        require(completed['args'].get('loss_only'), 'Generation evidence unexpectedly absent.')
        return None
    generated = rows(path)
    files = completed['generation_files']
    expected = [(cohort, source) for cohort, filename in files.items() for source in rows(ROOT / filename)]
    require([(row['cohort'], row['id']) for row in generated] == [(cohort, source['id']) for cohort, source in expected],
            'Generated cohort/id coverage or order differs from the frozen inputs.')
    require(len(generated) == completed['outputs'], 'Generation count differs from completion manifest.')
    endings = Counter()
    special_responses = 0
    tokens = 0
    eos = tokenizer_ids['eos']
    for output, (cohort, source) in zip(generated, expected):
        require(output['source_file'] == files[cohort], 'Generated source file/cohort mismatch.')
        require(output['source_row_sha256'] == row_sha(source), 'Raw generation source row hash mismatch.')
        require(output['source_id'] == source.get('source_id', source.get('case_id', source['id'])), 'Source case identifier changed.')
        require(output['adapter_sha256'] == completed['initial_adapter_identity']['adapter_model.safetensors'], 'Raw generation checkpoint mismatch.')
        require(output['gold_decision'] == source.get('gold_decision') and output['is_bad'] == source.get('is_bad'), 'Raw source label metadata changed.')
        detail = output['generated']
        ids = detail['token_ids']
        require(ids and all(type(token) is int for token in ids), 'Generated token IDs are empty or malformed.')
        require(detail['generated_tokens'] == len(ids) and len(ids) <= 192, 'Generated token count/cap disagrees.')
        if detail['finish_reason'] == 'eos':
            require(ids[-1] == eos and eos not in ids[:-1], 'EOS finish metadata disagrees with actual token IDs.')
            prefix = ids[:-1]
        else:
            require(detail['finish_reason'] == 'length' and len(ids) == 192 and eos not in ids,
                    'Length termination metadata disagrees with actual token IDs.')
            prefix = ids
        unexpected = [token for token in prefix if token in special]
        require(unexpected == detail['unexpected_special_token_ids'], 'Unexpected tokenizer-control metadata differs from token IDs.')
        require(not unexpected or output['parsed']['valid'] is False, 'Embedded control token was scored valid.')
        require(detail['finish_reason'] == 'eos' or output['parsed']['valid'] is False, 'Truncated output was scored valid.')
        require(isinstance(detail['text'], str) and isinstance(detail['text_with_special_tokens'], str), 'Raw decoded texts missing.')
        if not completed['args']['sample']:
            require(output['sampling'] is None, 'Greedy stage contains sampling metadata.')
        endings[detail['finish_reason']] += 1
        special_responses += bool(unexpected)
        tokens += len(ids)
    require(tokens == completed['generated_tokens'], 'Total generated-token accounting differs.')
    settings = completed['generation_settings']
    require(settings['repetition_penalty'] == 1 and settings['top_k'] == 0 and settings['max_new_tokens'] == 192,
            'Frozen decoding controls changed.')
    require(settings['eos_token_id'] == eos and settings['do_sample'] == completed['args']['sample'], 'EOS/sampling configuration differs.')
    if not completed['args']['sample']:
        require(completed['effective_generation_batch_size'] == 32, 'Observed greedy batch size is not frozen32.')
    return dict(outputs=len(generated), generated_tokens=tokens, finish_reasons=dict(endings),
                unexpected_special_token_responses=special_responses, raw_source_and_adapter_hashes_verified=True,
                effective_batch_size=completed['effective_generation_batch_size'],
                scope='Token-level EOS/control metadata and exact saved-source provenance; no semantic reason judgment or fresh decoding')


def names(row):
    return ('all', 'bad', 'bad_category:' + row['authored_error_category']) if row['is_bad'] else ('all', 'good', 'good_' + row['gold_decision'].lower())


def token_counts(stage):
    return {row['id']: row['target_tokens'] for row in rows(stage / 'tokenization.jsonl') if row['cohort'] == 'target_loss'}


def audit_nll(stage, completed):
    path = stage / 'target_losses.jsonl'
    if not path.exists():
        require(completed['mode'] != 'diagnose', 'Diagnosis lacks target-loss evidence.')
        return None
    source = rows(local(completed['args']['data']))
    observed = rows(path)
    require([row['id'] for row in source] == [row['id'] for row in observed], 'Target-loss source coverage/order differs.')
    tokens = token_counts(stage)
    groups = defaultdict(lambda: dict(examples=0, target_tokens=0, loss_sum=0.0, weighted_loss_sum=0.0, weighted_target_tokens=0.0))
    weight = completed['args']['bad_weight']
    for original, item in zip(source, observed):
        require(item['source_row_sha256'] == row_sha(original), 'Target-loss source row changed.')
        require(item['target_tokens'] == tokens[original['id']], 'Measured NLL target count differs from tokenization.')
        require(item['target_loss_sum'] >= 0, 'Negative recorded target NLL.')
        close(item['mean_target_nll'], item['target_loss_sum'] / item['target_tokens'], 'Per-row mean target NLL')
        expected_weight = weight if original['is_bad'] else 1.0
        require(item['loss_weight'] == expected_weight and tuple(item['groups']) == names(original), 'NLL grouping or is_bad weight changed.')
        for name in names(original):
            cell = groups[name]
            cell['examples'] += 1
            cell['target_tokens'] += item['target_tokens']
            cell['loss_sum'] += item['target_loss_sum']
            cell['weighted_loss_sum'] += item['target_loss_sum'] * expected_weight
            cell['weighted_target_tokens'] += item['target_tokens'] * expected_weight
    summary = read(stage / 'target_loss_summary.json')
    require(set(groups) == set(summary['groups']), 'NLL summary groups differ.')
    for name, cell in groups.items():
        recorded = summary['groups'][name]
        for key, value in cell.items():
            close(recorded[key], value, 'NLL group ' + name + '/' + key)
        close(recorded['mean_target_nll'], cell['loss_sum'] / cell['target_tokens'], 'Group mean target NLL')
        close(recorded['weighted_mean_target_nll'], cell['weighted_loss_sum'] / cell['weighted_target_tokens'], 'Weighted group mean target NLL')
    require(summary == completed['target_loss_diagnostics'], 'NLL summary differs from completed manifest.')
    return dict(rows=len(source), group_means={name: summary['groups'][name]['mean_target_nll'] for name in groups},
                group_token_counts={name: cell['target_tokens'] for name, cell in groups.items()},
                aggregation_and_weighting_verified=True,
                scope='Independent recomputation of saved per-row NLL summaries; no model-forward probability recomputation')


def audit_training(stage, completed):
    if completed['mode'] != 'train':
        return None
    data = rows(local(completed['args']['data']))
    source = {row['id']: row for row in data}
    tokens = token_counts(stage)
    exposure = rows(stage / 'exposure.jsonl')
    logs = rows(stage / 'training.jsonl')
    require(len(source) == len(data) == len(exposure), 'Training exact exposure count changed.')
    require(sorted(item['id'] for item in exposure) == sorted(source), 'Training exposure repeated or omitted cases.')
    require([item['id'] for item in exposure] == completed['sample_order'], 'Recorded training sample order differs.')
    require(row_sha(exposure) == completed['actual_exposure_sha256'], 'Actual exposure inventory digest differs.')
    pure = completed['args']['bad_only_diagnostic']
    expected_steps = 32 if pure else 64
    require(len(logs) == completed['steps'] == expected_steps, 'Training update count differs from prescribed single pass.')
    require([item['step'] for item in logs] == list(range(1, expected_steps + 1)), 'Training step numbering changed.')
    require([identity for item in logs for identity in item['ids']] == completed['sample_order'], 'Step logs differ from actual exposure order.')
    bad_weight = completed['args']['bad_weight']
    weighted_tokens = 0.0
    for item in exposure:
        row = source[item['id']]
        require(item['source_row_sha256'] == row_sha(row), 'Exposure source row hash differs.')
        require(item['target_tokens'] == tokens[row['id']], 'Exposure target-token count differs.')
        require(item['weight'] == (bad_weight if row['is_bad'] else 1), 'Exposure weights were not selected by is_bad.')
    for item in logs:
        batch = [source[identity] for identity in item['ids']]
        cells = Counter('bad' if row['is_bad'] else 'good_' + row['gold_decision'].lower() for row in batch)
        require(cells == ({'bad': 16} if pure else {'bad': 8, 'good_report': 4, 'good_clear': 4}), 'Effective-batch stratification differs.')
        denominator = sum(tokens[row['id']] * (bad_weight if row['is_bad'] else 1) for row in batch)
        close(item['weighted_target_token_denominator'], denominator, 'Effective weighted target-token denominator')
        group = item['groups_at_processing']['all']
        close(group['weighted_target_tokens'], denominator, 'Batch group target-token mass')
        close(item['loss'], group['weighted_loss_sum'] / denominator, 'Recorded weighted batch objective', rtol=2e-6, atol=2e-7)
        require(math.isfinite(item['grad_norm_before_clip']) and 0 <= item['grad_norm_after_clip'] <= 1.0001, 'Nonfinite or unclipped recorded gradient.')
        close(item['lr'], completed['args']['lr'], 'Actual recorded learning rate')
        weighted_tokens += denominator
    require(completed['target_tokens'] == sum(tokens.values()), 'Total supervised target-token count differs.')
    close(completed['groups_at_processing']['all']['weighted_target_tokens'], weighted_tokens, 'Total weighted target-token mass')
    require(completed['initial_trainable_parameter_sha256'] != completed['final_trainable_parameter_sha256'], 'Adapter did not change.')
    final_identity = {name: sha(stage / 'adapter' / name) for name in ('adapter_model.safetensors', 'adapter_config.json')}
    require(final_identity == completed['adapter_identity'], 'Final saved adapter identity differs.')
    # Map saved optimizer tensors to CPU; no model allocation or GPU call occurs.
    import torch
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization before CPU optimizer inspection.')
    saved = torch.load(stage / 'optimizer.pt', map_location='cpu', weights_only=True)
    require(saved['adapter_identity'] == final_identity and saved['trainable_parameter_sha256'] == completed['final_trainable_parameter_sha256'],
            'Optimizer/final adapter identity or tensor fingerprint mismatch.')
    continuation = saved['continuation']
    for key in ('seed', 'lr', 'effective_batch', 'bad_weight', 'rule_variant', 'bad_only_diagnostic'):
        require(continuation[key] == completed['args'][key], 'Saved optimizer recipe differs: ' + key)
    require(continuation['data_sha256'] == completed['data_sha256'] and continuation['config_sha256'] == completed['config_sha256'], 'Saved optimizer input hashes differ.')
    states = saved['optimizer_state_dict']['state']
    require(len(states) == len(completed['trainable_parameter_names']), 'Optimizer does not cover every trainable parameter.')
    expected_cumulative = completed['cumulative_optimizer_steps']
    step_counts = Counter(int(state['step']) for state in states.values())
    require(set(step_counts) == {expected_cumulative} and saved['cumulative_steps'] == expected_cumulative, 'Adam step counters did not continue correctly.')
    finite = all(bool(torch.isfinite(state[name]).all()) for state in states.values() for name in ('exp_avg', 'exp_avg_sq'))
    require(finite, 'Saved Adam moments contain nonfinite values.')
    if completed['args']['epoch'] == 1:
        require(completed['args']['optimizer'] is None and completed['initial_optimizer_sha256'] is None
                and not completed['optimizer_resumed'] and completed['optimizer_previous_steps'] == 0, 'First induction pass reused an optimizer.')
    require(not torch.cuda.is_initialized(), 'CPU optimizer inspection initialized CUDA.')
    return dict(updates=len(logs), examples=len(data), supervised_tokens=completed['target_tokens'],
                weighted_target_tokens=weighted_tokens, per_batch_stratification_verified=True,
                weighted_denominator_and_objective_verified=True, parameter_change_verified=True,
                optimizer_parameter_states=len(states), optimizer_step_counts=dict(step_counts), optimizer_moments_finite=finite,
                optimizer_resumed=completed['optimizer_resumed'], initial_parameter_sha256=completed['initial_trainable_parameter_sha256'],
                final_parameter_sha256=completed['final_trainable_parameter_sha256'],
                scope='Recorded losses/gradients and saved CPU-mapped Adam state; no independent training replay')


def audit_stage(name):
    config, frozen, provenance, tokenizer_ids, special = metadata()
    stage = ROOT / 'results/stages' / name
    completion = stage / 'COMPLETED.json'
    require(completion.exists(), 'Stage is not complete; partial outputs are not auditable final evidence.')
    completed = read(completion)
    require(completed['status'] == 'complete' and not completed['check_only'], 'Stage is not actual completed GPU work.')
    adapter, identity = verify_inputs(stage, completed, frozen)
    peak = completed['peak_cuda_allocated_bytes']
    require(0 < peak <= config['cuda_allocator_limit_gib'] * 1024**3, 'Reported CUDA allocation peak exceeds the frozen cap.')
    result = dict(stage=name, mode=completed['mode'], verdict='passed', reviewed_at=datetime.now(timezone.utc).isoformat(),
                  scope='Observed execution and saved numerical contracts only; no semantic or hypothesis conclusion',
                  original_freeze_sha256=EXPECTED_FREEZE, completion_sha256=sha(completion),
                  imported_adapter=imported_identity(adapter, identity, completed, provenance),
                  peak_reported_cuda_allocated_gib=peak / 1024**3,
                  configured_cuda_allocator_limit_gib=config['cuda_allocator_limit_gib'],
                  microbatch_or_greedy_batch=completed['args']['batch_size'],
                  reported_stage_elapsed_seconds=completed['stage_elapsed_s'],
                  generation=audit_generations(stage, completed, tokenizer_ids, special) if completed['mode'] != 'train' else None,
                  target_loss=audit_nll(stage, completed), training=audit_training(stage, completed),
                  auditor_sha256=sha(__file__))
    require(sha(completion) == result['completion_sha256'], 'Completion changed during audit.')
    return result


def save(result):
    name = 'MODEL_NUMERIC_' + result['stage']
    with (HERE / (name + '.json')).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    lines = ['# Observed model-stage numerical review', '', '**Passed for the recorded execution contracts.**', '',
             'Stage: `' + result['stage'] + '`  ', 'Completion SHA-256: `' + result['completion_sha256'] + '`', '',
             f"The reported PyTorch allocation peak was {result['peak_reported_cuda_allocated_gib']:.3f} GiB against the frozen {result['configured_cuda_allocator_limit_gib']}-GiB allocator cap. This uses the stage's recorded measurement, not an independent device query.", '']
    if result['imported_adapter']['imported_seed'] is not None:
        lines.append('The imported adapter file hashes and loaded trainable-tensor fingerprint match `inputs/PROVENANCE.json`.')
    if result['generation']:
        values = result['generation']
        lines.append(f"All {values['outputs']} raw output records match their input-row and checkpoint hashes. Finish reasons ({values['finish_reasons']}) and {values['unexpected_special_token_responses']} responses containing unexpected control tokens agree with saved token IDs.")
    if result['target_loss']:
        values = result['target_loss']
        lines.append(f"The {values['rows']} target-loss records cover the complete pool. Bad/good/category token counts, weights and means recompute from the saved per-row values.")
    if result['training']:
        values = result['training']
        lines.append(f"The pass processed {values['examples']} cases in {values['updates']} updates, with {values['supervised_tokens']} supervised target tokens. Per-batch stratification and weighted denominators match the saved objective; adapter parameters changed, and saved Adam states have the expected step counts and finite moments.")
    lines += ['', 'This is a read-only CPU review of retained artifacts. It does not rerun model forwards or training, independently measure device memory, or judge the factual substance of generated reasons. No model stage, GPU allocation, service operation, or frozen-file edit was performed by this auditor.']
    with (HERE / (name + '.md')).open('x') as stream:
        stream.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', action='append')
    parser.add_argument('--watch-until-first-training', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=2400)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    started = time.monotonic()
    observed = set()
    while True:
        names_to_audit = args.stage or [path.parent.name for path in sorted((ROOT / 'results/stages').glob('*/COMPLETED.json'))]
        for name in names_to_audit:
            path = HERE / ('MODEL_NUMERIC_' + name + '.json')
            if name in observed or path.exists():
                observed.add(name)
                continue
            result = audit_stage(name)
            save(result)
            observed.add(name)
            print(json.dumps(dict(stage=name, verdict='passed', mode=result['mode'], peak_gib=result['peak_reported_cuda_allocated_gib'],
                                  generation=result['generation'], target_loss=result['target_loss'], training=result['training'])), flush=True)
            if args.watch_until_first_training and result['training'] is not None:
                return
        if not args.watch_until_first_training or time.monotonic() - started >= args.max_seconds or (ROOT / 'results/TERMINAL.json').exists():
            print(json.dumps(dict(status='audit_watch_finished', observed_stages=sorted(observed))), flush=True)
            return
        time.sleep(min(args.poll_seconds, 60))


if __name__ == '__main__':
    main()
