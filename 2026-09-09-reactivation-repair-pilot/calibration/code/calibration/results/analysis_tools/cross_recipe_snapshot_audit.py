"""Bounded, CPU-only audit of a pre-recorded completed-stage inventory."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
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
    started = time.monotonic()
    snapshot_path = Path(snapshot_path).resolve()
    snapshot = read(snapshot_path)
    require(snapshot_path.parent == HERE, 'Snapshot must be a local review artifact.')
    snapshot_sha = sha(snapshot_path)
    driver_sha = sha(__file__)
    require(sha(ROOT / 'FREEZE.json') == snapshot['freeze_sha256'], 'Freeze changed after snapshot.')
    for name, digest in snapshot['auditor_source_sha256'].items():
        require(sha(HERE / name) == digest, 'Existing auditor changed after snapshot: ' + name)
    for name, digest in snapshot['completed_stage_sha256'].items():
        require(sha(ROOT / 'results/stages' / name / 'COMPLETED.json') == digest, 'Completed snapshot receipt changed: ' + name)
    spec = importlib.util.spec_from_file_location('cross_recipe_independent_auditor', HERE / 'independent_run_audit.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'CPU audit unexpectedly initialized CUDA.')
    review = helper.RunAudit(ROOT, inspect_optimizer=True)
    # This object's constructor may see later stages; pin its scientific stage
    # inventory to the previously written task-start snapshot before any audit.
    review.stages = {name: review.stages[name] for name in snapshot['completed_stage_sha256']}
    review.stage_hashes = dict(snapshot['completed_stage_sha256'])
    selected = [name for name, stage in review.stages.items()
                if stage['mode'] == 'train' or (stage['mode'] == 'diagnose' and '_baseline_' not in name)]
    reports = {}
    for name in selected:
        reports[name] = review.audit_stage(name)
        print(json.dumps(dict(stage=name, status='passed', mode=review.stages[name]['mode'])), flush=True)
    training = {name: review.stages[name] for name in selected if review.stages[name]['mode'] == 'train'}
    trajectories = defaultdict(list)
    for name, stage in training.items():
        trajectories[name.rsplit('_epoch', 1)[0]].append((stage['args']['epoch'], name, stage))
    provenance = read(ROOT / 'inputs/PROVENANCE.json')
    source_rows = helper.independent.read_jsonl(ROOT / 'data/master_present.jsonl')
    require(len(source_rows) == 1024 and sum(row['is_bad'] for row in source_rows) == 512, 'Frozen training pool changed.')
    token_reference = read(ROOT / 'reviews/TOKENIZER_FEASIBILITY.json')['model_input_files']['data/master_present.jsonl']
    token_by_id = {row['id']: row['target_tokens'] for row in token_reference['row_token_counts']}
    bad_tokens = sum(token_by_id[row['id']] for row in source_rows if row['is_bad'])
    good_tokens = sum(token_by_id[row['id']] for row in source_rows if not row['is_bad'])
    recipe_results = []
    orders = defaultdict(list)
    peaks = []
    for recipe, passes in sorted(trajectories.items()):
        passes.sort()
        require([epoch for epoch, _, _ in passes] == list(range(1, len(passes) + 1)), 'Snapshot has a missing training pass: ' + recipe)
        first = passes[0][2]
        seed = first['args']['seed']
        weight = first['args']['bad_weight']
        require(seed == 1729 and first['args']['rule_variant'] == 'present', 'Unexpected branch outside this bounded present-rule audit.')
        expected_initial = provenance[str(seed)]
        require(review.relative(first['args']['adapter']) == f'inputs/competence_s{seed}'
                and first['initial_adapter_identity'] == expected_initial['adapter_sha256']
                and first['initial_trainable_parameter_sha256'] == expected_initial['trainable_parameter_sha256']
                and first['args']['optimizer'] is None and first['initial_optimizer_sha256'] is None
                and first['optimizer_resumed'] is False and first['optimizer_previous_steps'] == 0,
                'Recipe did not reset to exact competent weights and a fresh optimizer: ' + recipe)
        expected_weighted = good_tokens + weight * bad_tokens
        previous = None
        links = []
        for epoch, name, stage in passes:
            args = stage['args']
            require(args['seed'] == seed and args['rule_variant'] == 'present' and args['bad_weight'] == weight
                    and args['lr'] == 1e-4 and review.relative(args['data']) == 'data/master_present.jsonl'
                    and stage['config_sha256'] == first['config_sha256'] and stage['data_sha256'] == first['data_sha256'],
                    'Recipe/data/config/seed changed within a trajectory: ' + name)
            if previous:
                parent_name, parent = previous
                require(review.relative(args['adapter']) == f'results/stages/{parent_name}/adapter'
                        and stage['initial_adapter_identity'] == parent['adapter_identity']
                        and stage['initial_trainable_parameter_sha256'] == parent['final_trainable_parameter_sha256']
                        and review.relative(args['optimizer']) == f'results/stages/{parent_name}/optimizer.pt'
                        and stage['initial_optimizer_sha256'] == sha(ROOT / 'results/stages' / parent_name / 'optimizer.pt')
                        and stage['optimizer_previous_steps'] == parent['cumulative_optimizer_steps']
                        and stage['optimizer_resumed'] is True,
                        'Adapter or Adam continuation changed: ' + name)
            groups = stage['groups_at_processing']
            helper.close(groups['all']['weighted_target_tokens'], expected_weighted, 'Weighted token mass differs from independent CPU reference')
            require(groups['bad']['target_tokens'] == bad_tokens and groups['good']['target_tokens'] == good_tokens,
                    'Bad/good supervised token exposure differs: ' + name)
            require(stage['steps'] == 64 and stage['cumulative_optimizer_steps'] == 64 * epoch, 'Incorrect pass dose: ' + name)
            orders[epoch].append((recipe, tuple(stage['sample_order'])))
            peak = stage['peak_cuda_allocated_bytes'] / 1024**3
            require(0 < peak <= review.config['cuda_allocator_limit_gib'], 'Recorded allocation peak exceeds frozen cap: ' + name)
            peaks.append(peak)
            links.append(dict(stage=name, epoch=epoch, cumulative_steps=stage['cumulative_optimizer_steps'],
                              initial_optimizer_sha256=stage['initial_optimizer_sha256'],
                              final_optimizer_sha256=sha(ROOT / 'results/stages' / name / 'optimizer.pt'),
                              input_adapter_sha256=stage['initial_adapter_identity'], final_adapter_sha256=stage['adapter_identity'],
                              initial_tensor_sha256=stage['initial_trainable_parameter_sha256'], final_tensor_sha256=stage['final_trainable_parameter_sha256'],
                              weighted_target_tokens=groups['all']['weighted_target_tokens'],
                              observed_cpu_optimizer=reports[name]['training']['optimizer']))
            previous = name, stage
        recipe_results.append(dict(recipe=recipe, passes=len(passes), seed=seed, bad_weight=weight,
                                   starts_from_exact_competent_adapter=True, starts_with_fresh_optimizer=True,
                                   continuous_saved_adapter_and_adam_lineage=True, target_tokens_per_pass=bad_tokens + good_tokens,
                                   weighted_target_tokens_per_pass=expected_weighted,
                                   weighted_bad_target_token_share=weight * bad_tokens / expected_weighted, links=links))
    for epoch, records in orders.items():
        require(len({order for _, order in records}) == 1, 'Matched recipe exposure order differs for the same frozen epoch: ' + str(epoch))
    # Recheck immutable receipt/artifact and code closure after CPU tensor reads.
    review.check_freeze()
    require(sha(snapshot_path) == snapshot_sha and sha(__file__) == driver_sha, 'Audit input/driver changed during review.')
    for name, digest in snapshot['auditor_source_sha256'].items():
        require(sha(HERE / name) == digest, 'Existing auditor changed during review: ' + name)
    for name, digest in snapshot['completed_stage_sha256'].items():
        require(sha(ROOT / 'results/stages' / name / 'COMPLETED.json') == digest, 'Snapshot stage receipt changed during review: ' + name)
    for name in selected:
        for artifact, digest in review.stages[name]['artifacts_sha256'].items():
            require(sha(ROOT / 'results/stages' / name / artifact) == digest, 'Audited artifact changed during review: ' + name + '/' + artifact)
    require(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA.')
    return dict(review_type='Authored observed-execution review; not a private transcript', status='passed',
                reviewed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                scope='Only completed stages in the task-start snapshot. Later and unfinished stages are excluded.',
                snapshot_path=str(snapshot_path.relative_to(ROOT)), snapshot_sha256=snapshot_sha,
                original_freeze_sha256=snapshot['freeze_sha256'], driver_sha256=driver_sha,
                helper_source_sha256=snapshot['auditor_source_sha256'],
                snapshot_completed_stages=len(snapshot['completed_stage_sha256']), audited_training_passes=len(training),
                audited_induction_diagnoses=len(selected) - len(training), optimizer_updates=64 * len(training),
                exact_case_exposures=1024 * len(training), supervised_target_token_exposures=(bad_tokens + good_tokens) * len(training),
                no_omitted_or_duplicated_exposures=True, matched_epoch_orders_across_recipes=True,
                bad_target_tokens_per_pass=bad_tokens, good_target_tokens_per_pass=good_tokens,
                maximum_reported_training_allocation_gib=max(peaks), recipes=recipe_results, stages=reports,
                limitations=['Saved artifacts and recorded numerical contracts were audited, without independent model-forward or optimizer-update replay.',
                             'CUDA allocation peaks are stage-reported PyTorch measurements, not independent device queries.',
                             'No semantic content assessment, scientific outcome claim, omitted-rule-stage approval, or approval of future stages.'],
                operations=dict(cpu_only=True, cuda_initialized=False, model_calls=0, service_changes=0, frozen_file_changes=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.snapshot)
    stem = args.snapshot.stem.replace('CROSS_RECIPE_SNAPSHOT_', 'CROSS_RECIPE_EXECUTION_REVIEW_')
    with (HERE / (stem + '.json')).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    lines = ['# Bounded cross-recipe execution review', '',
             '**Passed for the fixed completed-stage snapshot.** No numerical or lineage defect was found in this review.', '',
             f"The review covered {result['audited_training_passes']} actual training passes and {result['audited_induction_diagnoses']} induction diagnoses. The training passes contain {result['optimizer_updates']} updates and {result['exact_case_exposures']:,} case exposures, with every case included exactly once per pass and every effective batch preserving the 8:4:4 composition.", '',
             'Each recipe began from the same verified competent seed-1729 adapter with a fresh optimizer. Every later pass used its own preceding adapter and saved Adam state. Saved adapter tensors, Adam shapes, finite moments and all per-parameter step counters match their recorded lineage. Data, seed, configuration and frozen source hashes agree.', '',
             '| Bad-target weight | Completed passes | Target tokens per pass | Weighted tokens per pass | Bad share of weighted tokens |',
             '|---|---:|---:|---:|---:|']
    for row in result['recipes']:
        lines.append(f"| {row['bad_weight']:.6g} | {row['passes']} | {row['target_tokens_per_pass']:,} | {row['weighted_target_tokens_per_pass']:,.4f} | {100 * row['weighted_bad_target_token_share']:.4f}% |")
    lines += ['', 'The loss weights were applied by `is_bad`, and saved batch/group objectives and token denominators recompute from the actual exposure records. The eight completed diagnoses also pass their source, token, decoding and target-loss accounting checks.', '',
              f"The largest recorded training allocation peak was {result['maximum_reported_training_allocation_gib']:.3f} GiB. Receipt, artifact, auditor and frozen-file hashes were checked again at the end.", '',
              'This read-only CPU review did not rerun model forwards or training, query the GPU, change services, or edit frozen files. It makes no semantic judgment or experimental-outcome claim. Stages outside the initial snapshot, including the omitted-rule half of the grid, are not approved by this receipt. The companion JSON records exact source hashes and per-stage evidence.']
    with (HERE / (stem + '.md')).open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='passed', receipt=str((HERE / (stem + '.json')).relative_to(ROOT)),
                          training_passes=result['audited_training_passes'], diagnoses=result['audited_induction_diagnoses'],
                          elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
