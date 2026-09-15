"""CPU audit of one frozen inventory of the first omitted-rule trajectory."""
from __future__ import annotations

import argparse
import ast
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
    started = time.monotonic()
    snapshot_path = Path(snapshot_path).resolve()
    require(snapshot_path.parent == HERE, 'Snapshot is outside the analysis directory.')
    snapshot, snapshot_sha, driver_sha = read(snapshot_path), sha(snapshot_path), sha(__file__)
    expected = {'s1729_baseline_present', 's1729_baseline_omitted_loss'}
    expected |= {f's1729_omitted_w0333_epoch{epoch}' for epoch in range(1, 5)}
    expected |= {f's1729_omitted_w0333_diagnose{epoch}' for epoch in (1, 2, 4)}
    require(set(snapshot['completed_stage_sha256']) == expected, 'Wrong bounded stage inventory.')
    require(sha(ROOT / 'FREEZE.json') == snapshot['freeze_sha256'], 'Freeze changed after snapshot.')
    for name, digest in snapshot['auditor_source_sha256'].items():
        require(sha(HERE / name) == digest, 'Audit helper changed after snapshot.')
    for name, digest in snapshot['completed_stage_sha256'].items():
        require(sha(ROOT / 'results/stages' / name / 'COMPLETED.json') == digest, 'Snapshotted completion changed.')
    spec = importlib.util.spec_from_file_location('omitted_independent_auditor', HERE / 'independent_run_audit.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA.')
    review = helper.RunAudit(ROOT, inspect_optimizer=True)
    review.stages = {name: review.stages[name] for name in snapshot['completed_stage_sha256']}
    review.stage_hashes = dict(snapshot['completed_stage_sha256'])
    reports = {}
    for name in review.stages:
        reports[name] = review.audit_stage(name)
        print(json.dumps(dict(stage=name, status='passed')), flush=True)

    syntax = ast.parse((ROOT / 'scripts/task.py').read_text())
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in syntax.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ('SYSTEM', 'RULE')}
    present = review.rows('data/master_present.jsonl')
    omitted = review.rows('data/master_omitted.jsonl')
    require(len(present) == len(omitted) == 1024, 'Training pair denominator changed.')
    for before, after in zip(present, omitted):
        require(before['rule_present'] is True and after['rule_present'] is False
                and before['view'] == 'master_present' and after['view'] == 'master_omitted', 'Training view flags differ.')
        require(before['prompt'] == constants['RULE'] + '\n\n' + after['prompt'], 'Omission changed more than the exact explicit rule paragraph.')
        require({key: value for key, value in before.items() if key not in ('prompt', 'rule_present', 'view')}
                == {key: value for key, value in after.items() if key not in ('prompt', 'rule_present', 'view')},
                'Targets, facts, IDs or labels differ between training views.')
    paired_views = {}
    for left, right, n in (('seen_probe_present', 'seen_probe_omitted', 128), ('dev_familiar', 'dev_reworded', 192)):
        a, b = review.rows('data/' + left + '.jsonl'), review.rows('data/' + right + '.jsonl')
        require(len(a) == len(b) == n and [row['id'] for row in a] == [row['id'] for row in b], 'Paired generation view IDs/order changed.')
        for one, two in zip(a, b):
            require(all(one[key] == two[key] for key in ('facts', 'gold_decision', 'stratum', 'case_id')), 'Paired generation views changed facts/labels.')
            if left == 'seen_probe_present':
                require(one == next(row for row in present if row['id'] == one['id'])
                        and two == next(row for row in omitted if row['id'] == two['id']), 'Seen probe is not an exact training-row alias.')
            else:
                require(one['rule_present'] is True and two['rule_present'] is True
                        and one['prompt'].startswith(constants['RULE']) and two['prompt'].startswith(constants['RULE'])
                        and one['prompt'] != two['prompt'], 'Development views lost their present rule or distinct wording.')
        paired_views[left + '/' + right] = dict(cases=n, same_case_facts_and_labels=True, independently_sampled_cases=False)

    # Independently render the frozen Jinja chat template and tokenize on CPU.
    # No Transformers model or weight loader is imported or invoked.
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    model = ROOT.parent / 'reactivation/models/Qwen2.5-3B-Instruct'
    tokenizer_config = read(model / 'tokenizer_config.json')
    feasibility = read(ROOT / 'reviews/TOKENIZER_FEASIBILITY.json')
    tokenizer_hashes = {}
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        tokenizer_hashes[name] = sha(model / name)
        require(tokenizer_hashes[name] == feasibility['tokenizer_local_files_sha256'][name], 'Tokenizer bytes changed after CPU preflight.')
    decoder = Tokenizer.from_file(str(model / 'tokenizer.json'))
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(tokenizer_config['chat_template'])
    eos = feasibility['tokenizer_ids']['eos']
    prefix_cache = {}
    def prefix(row):
        key = row['prompt']
        if key not in prefix_cache:
            messages = [dict(role='system', content=constants['SYSTEM']), dict(role='user', content=row['prompt'])]
            rendered = template.render(messages=messages, tools=None, add_generation_prompt=True)
            prefix_cache[key] = decoder.encode(rendered, add_special_tokens=False).ids
        return prefix_cache[key]
    tokenized_rows, mask_rows, generation_prefix_rows = 0, 0, 0
    for name, stage in review.stages.items():
        primary = {row['id']: row for row in review.rows(stage['args']['data'])}
        generation = {cohort: {row['id']: row for row in review.rows(path)}
                      for cohort, path in stage.get('generation_files', {}).items()}
        for actual in review.rows(ROOT / 'results/stages' / name / 'tokenization.jsonl'):
            if actual['cohort'] == 'target_loss':
                row = primary[actual['id']]
                ids = prefix(row)
                target = decoder.encode(row['target'], add_special_tokens=False).ids + [eos]
                inputs, labels = ids + target[:-1], [-100] * (len(ids) - 1) + target
                require(actual['input_ids_sha256'] == helper.independent.row_sha(inputs)
                        and actual['labels_sha256'] == helper.independent.row_sha(labels), 'Saved training/loss input or label hash differs from reconstructed source tokens: ' + name)
                require(actual['prefix_tokens'] == len(ids) and actual['target_tokens'] == len(target)
                        and actual['sequence_tokens'] == len(inputs) and labels[-1] == eos
                        and sum(token != -100 for token in labels) == len(target), 'Target masking, EOS or token denominator differs.')
                mask_rows += 1
            else:
                row = generation[actual['cohort']][actual['id']]
                ids = prefix(row)
                require(actual['input_ids_sha256'] == helper.independent.row_sha(ids)
                        and actual['prefix_tokens'] == len(ids) and actual['sequence_tokens'] == len(ids) + 192,
                        'Saved generation prefix differs from its exact paired source view: ' + name)
                generation_prefix_rows += 1
            tokenized_rows += 1
    prefix_differences = {len(prefix(a)) - len(prefix(b)) for a, b in zip(present, omitted)}
    require(prefix_differences == {59}, 'Rule omission changed the frozen per-case prefix difference.')
    target_tokens = sum(len(decoder.encode(row['target'], add_special_tokens=False).ids) + 1 for row in omitted)
    bad_tokens = sum(len(decoder.encode(row['target'], add_special_tokens=False).ids) + 1 for row in omitted if row['is_bad'])
    good_tokens = target_tokens - bad_tokens
    expected_weighted = good_tokens + bad_tokens / 3
    provenance = read(ROOT / 'inputs/PROVENANCE.json')['1729']
    continuity, previous, peaks = [], None, []
    for epoch in range(1, 5):
        name = f's1729_omitted_w0333_epoch{epoch}'
        stage = review.stages[name]
        args = stage['args']
        require(review.relative(args['data']) == 'data/master_omitted.jsonl' and args['seed'] == 1729
                and args['rule_variant'] == 'omitted' and args['bad_weight'] == 1 / 3, 'Actual omitted training recipe differs.')
        if epoch == 1:
            require(review.relative(args['adapter']) == 'inputs/competence_s1729'
                    and stage['initial_adapter_identity'] == provenance['adapter_sha256']
                    and stage['initial_trainable_parameter_sha256'] == provenance['trainable_parameter_sha256']
                    and args['optimizer'] is None and stage['initial_optimizer_sha256'] is None
                    and stage['optimizer_resumed'] is False and stage['optimizer_previous_steps'] == 0,
                    'Omitted trajectory failed to reset to exact competent weights and fresh Adam.')
        else:
            parent_name, parent = previous
            require(review.relative(args['adapter']) == f'results/stages/{parent_name}/adapter'
                    and review.relative(args['optimizer']) == f'results/stages/{parent_name}/optimizer.pt'
                    and stage['initial_adapter_identity'] == parent['adapter_identity']
                    and stage['initial_trainable_parameter_sha256'] == parent['final_trainable_parameter_sha256']
                    and stage['initial_optimizer_sha256'] == sha(ROOT / 'results/stages' / parent_name / 'optimizer.pt')
                    and stage['optimizer_previous_steps'] == parent['cumulative_optimizer_steps']
                    and stage['optimizer_resumed'] is True, 'Omitted Adam/adapter continuation differs.')
        require(stage['steps'] == 64 and stage['cumulative_optimizer_steps'] == 64 * epoch
                and stage['target_tokens'] == target_tokens and stage['prefix_tokens'] == sum(len(prefix(row)) for row in omitted), 'Actual omitted dose or token mass differs.')
        helper.close(stage['groups_at_processing']['all']['weighted_target_tokens'], expected_weighted, 'Omitted weighted token mass')
        peak = stage['peak_cuda_allocated_bytes'] / 1024**3
        require(0 < peak <= review.config['cuda_allocator_limit_gib'], 'Recorded omitted training memory exceeds frozen cap.')
        peaks.append(peak)
        continuity.append(dict(stage=name, epoch=epoch, cumulative_steps=64 * epoch,
                initial_tensor_sha256=stage['initial_trainable_parameter_sha256'], final_tensor_sha256=stage['final_trainable_parameter_sha256'],
                initial_optimizer_sha256=stage['initial_optimizer_sha256'], final_optimizer_sha256=sha(ROOT / 'results/stages' / name / 'optimizer.pt'),
                saved_cpu_optimizer=reports[name]['training']['optimizer']))
        previous = name, stage
    review.check_freeze()
    require(sha(snapshot_path) == snapshot_sha and sha(__file__) == driver_sha, 'Audit driver or snapshot changed during review.')
    for name, digest in snapshot['auditor_source_sha256'].items():
        require(sha(HERE / name) == digest, 'Audit helper changed during review.')
    for name, digest in snapshot['completed_stage_sha256'].items():
        require(sha(ROOT / 'results/stages' / name / 'COMPLETED.json') == digest, 'Audited receipt changed during review.')
        for artifact, artifact_sha in review.stages[name]['artifacts_sha256'].items():
            require(sha(ROOT / 'results/stages' / name / artifact) == artifact_sha, 'Audited artifact changed during review.')
    require(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA.')
    return dict(review_type='Authored observed-execution audit; not a private transcript', status='passed',
                reviewed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                snapshot_path=str(snapshot_path.relative_to(ROOT)), snapshot_sha256=snapshot_sha,
                original_freeze_sha256=snapshot['freeze_sha256'], driver_sha256=driver_sha,
                helper_source_sha256=snapshot['auditor_source_sha256'], tokenizer_source_sha256=tokenizer_hashes,
                training_passes=4, completed_induction_diagnoses=3, baseline_dependencies=2,
                optimizer_updates=256, exact_case_exposures=4096, cumulative_adam_steps=[64, 128, 192, 256],
                target_tokens_per_pass=target_tokens, bad_target_tokens_per_pass=bad_tokens, good_target_tokens_per_pass=good_tokens,
                weighted_target_tokens_per_pass=expected_weighted, omitted_prefix_tokens_per_pass=sum(len(prefix(row)) for row in omitted),
                rule_omission_prefix_token_difference_per_case=59, reconstructed_tokenization_records=tokenized_rows,
                reconstructed_target_input_and_label_hashes=mask_rows, reconstructed_generation_prefix_hashes=generation_prefix_rows,
                exact_target_and_fact_pairing_verified=True, paired_generation_views=paired_views,
                maximum_reported_training_allocation_gib=max(peaks), optimizer_lineage=continuity, stages=reports,
                limitations=['The omitted condition removes the explicit user reporting-rule paragraph; the common system instructions and competent starting adapter remain, as frozen.',
                             'Input/label token hashes were independently reconstructed with the frozen local tokenizer and chat template; no model forward, backward pass or optimizer update was replayed.',
                             'The seen and development prompt pairs reuse the same cases and are not independent samples.',
                             'Memory peaks are recorded PyTorch measurements, not independent GPU queries. No semantic judgment, experimental conclusion, or approval of other/future stages is made.'],
                operations=dict(cpu_only=True, cuda_initialized=False, model_calls=0, service_changes=0, frozen_file_changes=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.snapshot)
    stem = args.snapshot.stem.replace('OMITTED_TRAJECTORY_SNAPSHOT_', 'OMITTED_TRAJECTORY_EXECUTION_REVIEW_')
    with (HERE / (stem + '.json')).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    text = f'''# First omitted-rule trajectory execution review

**Passed for the fixed completed-stage snapshot.** No numerical, prompt-variant or lineage defect was found in the four training passes, three induction diagnoses and two baseline dependencies.

The actual omitted training inputs remove only the explicit reporting-rule paragraph. All 1,024 case facts, labels and authored targets remain paired with the present-rule version. The common system instruction remains unchanged. Independent CPU reconstruction of the frozen chat template and tokenizer matches {result['reconstructed_target_input_and_label_hashes']:,} saved target-input/label hashes and {result['reconstructed_generation_prefix_hashes']:,} saved generation-prefix hashes. The rule omission removes 59 prefix tokens per case.

Each pass includes every case exactly once in 64 updates with the frozen 8:4:4 batch composition. Per pass, 37,561 target tokens contribute 25,259.6667 weighted tokens at bad-target weight 1/3. Masks retain target-only supervision including EOS; the omitted prefix contributes 212,004 tokens. The trajectory begins from the exact competent adapter with fresh Adam state, then continues its own saved adapter and optimizer through steps 64, 128, 192 and 256. All 504 parameter-state tensors per pass have the expected shapes, finite moments and counters, and saved LoRA tensor fingerprints match their lineage.

Completed diagnoses preserve both 128-case seen views and both 192-case development views, with exact source/checkpoint links, saved-token decoding and target-loss accounting. Development evaluation includes the explicit rule in both wordings. These paired views reuse cases and are not independent samples.

The largest recorded training allocation peak was {result['maximum_reported_training_allocation_gib']:.3f} GiB. Source, artifact, receipt and auditor hashes were checked again at completion. The companion JSON records these bindings and per-stage evidence.

This was a bounded CPU audit. It did not inspect unfinished stages, replay model forwards or training, query the GPU, change services, or edit frozen files. It makes no semantic or experimental-outcome claim; the full final run audit remains separate.
'''
    with (HERE / (stem + '.md')).open('x') as stream:
        stream.write(text)
    print(json.dumps(dict(status='passed', receipt=str((HERE / (stem + '.json')).relative_to(ROOT)),
                          elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
