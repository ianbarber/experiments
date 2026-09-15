#!/usr/bin/env python3
"""Post-run CPU audit. Writes only FINAL_REVIEW.json beside this script."""
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
from independent_review import (read_jsonl, independently_parse, independently_label,
                                audit_competence, audit_induction)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def row_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def counts(cases, outputs):
    result = dict(n=len(cases), correct=0, valid=0, valid_clear=0, valid_report=0)
    for case, output in zip(cases, outputs):
        assert case['id'] == output['id']
        decision = independently_parse(output['generated'])
        result['correct'] += decision == independently_label(case)
        result['valid'] += decision is not None
        result['valid_clear'] += decision == 'CLEAR'
        result['valid_report'] += decision == 'REPORT'
    return result


freeze = read(ROOT / 'FREEZE.json')
for path, digest in freeze['files'].items():
    assert sha(ROOT / path) == digest, path
resume = read(ROOT / 'results/operational_resume/RESUME_FREEZE.json')
for group in ('recovery_code_sha256', 'prior_artifacts_sha256'):
    for path, digest in resume[group].items():
        assert sha(ROOT / path) == digest, path
prior = read(ROOT / 'results/postrun_review/COMPETENCE_AND_INDUCTION_EPOCH1_REVIEW.json')
expected = set(prior['completed_stage_provenance']) | {
    's1729_induction_epoch2', 's1729_induction_select_epoch2'}
stages = ROOT / 'results/stages'
assert {p.name for p in stages.iterdir()} == expected
manifests, provenance, generated = {}, {}, {}
for name in sorted(expected):
    folder = stages / name
    m = read(folder / 'COMPLETED.json')
    manifests[name] = m
    assert m['status'] == 'complete' and m['check_only'] is False
    assert not (folder / 'FAILED.json').exists()
    for path, digest in m['artifacts_sha256'].items():
        assert sha(folder / path) == digest, (name, path)
    for path, digest in m['source_sha256'].items():
        assert freeze['files'][path] == digest == sha(ROOT / path)
    assert sha(ROOT / 'configs/pilot.json') == m['config_sha256']
    assert read(ROOT / 'configs/pilot.json') == m['config']
    data_path = ROOT / Path(m['args']['data']).relative_to('/workspace')
    assert sha(data_path) == m['data_sha256']
    data = read_jsonl(data_path)
    assert len(data) == m['source_examples']
    cpu_tokens = ROOT / 'results/cpu_model_checks/20260912T165317Z' / data_path.stem / 'tokenization.jsonl'
    assert sha(cpu_tokens) == sha(folder / 'tokenization.jsonl')
    provenance[name] = dict(completion_sha256=sha(folder / 'COMPLETED.json'),
                            artifacts_verified=len(m['artifacts_sha256']),
                            data_sha256=m['data_sha256'], mode=m['mode'])
    if name in prior['completed_stage_provenance']:
        assert provenance[name]['completion_sha256'] == prior['completed_stage_provenance'][name]['completion_sha256']
    if m['mode'] != 'generate':
        continue
    outputs = read_jsonl(folder / 'outputs.jsonl')
    generated[name] = outputs
    assert len(outputs) == len(data) == m['outputs']
    adapter = ROOT / Path(m['args']['adapter']).relative_to('/workspace')
    assert {n: sha(adapter / n) for n in ('adapter_config.json', 'adapter_model.safetensors')} == m['initial_adapter_identity']
    assert m['generation_settings']['do_sample'] is False
    assert m['generation_settings']['max_new_tokens'] == 192
    total_tokens = 0
    for row, out in zip(data, outputs):
        assert row['id'] == out['id'] == out['source_id']
        assert row_sha(row) == out['source_row_sha256']
        assert out['adapter_sha256'] == m['initial_adapter_identity']['adapter_model.safetensors']
        assert out['sampling_seed'] is None and out['draw_index'] is None
        assert independently_label(row) == out['gold_decision']
        g = out['generated']
        assert g['finish_reason'] == 'eos' and g['unexpected_special_token_ids'] == []
        assert g['token_ids'][-1] == 151645 and 151645 not in g['token_ids'][:-1]
        assert len(g['token_ids']) == g['generated_tokens'] <= 192
        decision = independently_parse(g)
        assert out['parsed']['valid'] == (decision is not None)
        assert out['parsed']['decision'] == decision
        total_tokens += g['generated_tokens']
    assert total_tokens == m['generated_tokens']
    provenance[name].update(raw_outputs_sha256=sha(folder / 'outputs.jsonl'),
                            source_rows_verified=len(outputs),
                            initial_adapter_identity=m['initial_adapter_identity'])

train1 = manifests['s1729_induction_epoch1']
train2 = manifests['s1729_induction_epoch2']
assert train2['initial_adapter_identity'] == train1['adapter_identity']
assert train2['initial_optimizer_sha256'] == sha(stages / 's1729_induction_epoch1/optimizer.pt')
assert train2['initial_trainable_parameter_sha256'] == train1['final_trainable_parameter_sha256']
assert train2['optimizer_resumed'] is True and train2['optimizer_previous_steps'] == 32
assert train2['steps'] == 32 and train2['cumulative_optimizer_steps'] == 64
assert train2['epoch'] == train2['args']['epoch'] == 2
assert train2['final_trainable_parameter_sha256'] != train2['initial_trainable_parameter_sha256']
data = read_jsonl(ROOT / 'data/induction_train.jsonl')
tokens = {r['id']: r for r in read_jsonl(stages / 's1729_induction_epoch2/tokenization.jsonl')}
order_seed = int(row_sha(dict(seed=1729, purpose='training_pass_order', identity=2))[:16], 16) % (2**31 - 1)
assert train2['order_seed'] == order_seed
order = [r['id'] for r in data]
random.Random(order_seed).shuffle(order)
assert order == train2['sample_order']
logs = read_jsonl(stages / 's1729_induction_epoch2/training.jsonl')
assert len(logs) == 32
prefix = target = 0
for i, log in enumerate(logs):
    ids = order[16*i:16*(i+1)]
    assert ids == log['ids'] and log['examples'] == 16
    assert log['step'] == i + 1 and log['cumulative_step'] == i + 33
    assert log['epoch'] == 2 and log['lr'] == 0.0001
    denominator = sum(tokens[k]['target_tokens'] for k in ids)
    prefix += sum(tokens[k]['prefix_tokens'] for k in ids)
    target += denominator
    assert log['effective_batch_target_tokens'] == denominator
    assert log['target_tokens'] == target and log['prefix_tokens'] == prefix
    assert all(math.isfinite(log[k]) for k in ('loss', 'grad_norm_before_clip', 'grad_norm_after_clip'))
    assert log['grad_norm_after_clip'] <= 1.0001
assert target == train2['target_tokens'] == 17272
assert prefix == train2['prefix_tokens'] == 120667

import torch
assert not torch.cuda.is_initialized()
optimizers = [torch.load(stages / f's1729_induction_epoch{epoch}/optimizer.pt',
                         map_location='cpu', weights_only=True) for epoch in (1, 2)]
assert optimizers[0]['continuation'] == optimizers[1]['continuation']
for epoch, saved in enumerate(optimizers, 1):
    manifest = manifests[f's1729_induction_epoch{epoch}']
    assert saved['epoch'] == epoch and saved['cumulative_steps'] == epoch * 32
    assert saved['adapter_identity'] == manifest['adapter_identity']
    assert saved['trainable_parameter_sha256'] == manifest['final_trainable_parameter_sha256']
    state = saved['optimizer_state_dict']['state']
    assert len(state) == 504
    assert {int(s['step']) for s in state.values()} == {epoch * 32}
    assert all(torch.isfinite(s[k]).all().item() for s in state.values() for k in ('exp_avg', 'exp_avg_sq'))
    assert any(torch.count_nonzero(s['exp_avg']).item() for s in state.values())
assert not torch.cuda.is_initialized()

gate_reviews = {}
for seed in (1729, 2718):
    for kind, suffix in (('select', 'select_epoch1'), ('qualify', 'qualification')):
        name = f's{seed}_competence_{suffix}'
        cases = read_jsonl(ROOT / f'data/competence_{kind}.jsonl')
        outputs = generated[name]
        audit = audit_competence(cases, outputs)
        saved = read(ROOT / 'results/gates' / (name + '.json'))
        assert saved['pass'] == audit['pass']
        assert saved['total'] == counts(cases, outputs)
        for label in ('REPORT', 'CLEAR'):
            pairs = [(c, o) for c, o in zip(cases, outputs) if c['gold_decision'] == label]
            assert saved['cells'][label] == counts(*map(list, zip(*pairs)))
        assert list(saved['checks'].values()) == [audit['gates']['report_accuracy'], audit['gates']['clear_accuracy'], audit['gates']['validity']]
        gate_reviews[name] = audit
cases = read_jsonl(ROOT / 'data/induction_select.jsonl')
baseline = generated['s1729_induction_select_competent']
for epoch in (1, 2):
    name = f's1729_induction_select_epoch{epoch}'
    outputs = generated[name]
    audit = audit_induction(cases, baseline, outputs)
    saved = read(ROOT / 'results/gates' / (name + '.json'))
    assert saved['pass'] == audit['pass']
    assert saved['total'] == counts(cases, outputs)
    mapping = {'target_errors_39_to_89': 'induced_valid_error_window',
               'target_error_increase_at_least_26': 'induction_gain',
               'valid_at_least_377': 'validity'}
    for group in ('noneliciting_report', 'clear'):
        mapping[group + '_correct_at_least_116'] = group + '_accuracy'
        mapping[group + '_deterioration_at_most_6'] = group + '_preservation'
    assert saved['checks'] == {k: audit['gates'][v] for k, v in mapping.items()}
    assert saved['target_error_count_increase'] == audit['valid_error_gain']
    for group in ('eliciting_report', 'noneliciting_report', 'clear'):
        for state_name, outs in (('competent', baseline), ('installed', outputs)):
            pairs = [(c, o) for c, o in zip(cases, outs) if c['stratum'] == group]
            assert saved['cells'][group][state_name] == counts(*map(list, zip(*pairs)))
    gate_reviews[name] = audit
assert {p.stem for p in (ROOT / 'results/gates').glob('*.json')} == set(gate_reviews)
terminal = read(ROOT / 'results/TERMINAL.json')
assert terminal['status'] == 'early_failed'
assert terminal['failure']['gate'] == 's1729_induction_selection'
assert terminal['repair_training_launched'] is False
last_gate = read(ROOT / 'results/gates/s1729_induction_select_epoch2.json')
assert terminal['failure']['details'] == {k: v for k, v in last_gate.items() if k not in ('gate', 'recorded_at')}
assert terminal['freeze_sha256'] == sha(ROOT / 'FREEZE.json')
events = read_jsonl(ROOT / 'results/events.jsonl')
assert {e['stage'] for e in events if e['event'] == 'stage_start'} == expected
assert not any(e['event'] == 'checkpoint_selected' and e.get('phase') == 'induction' for e in events)
assert not (ROOT / 'results/derived').exists()
assert not any(e['event'] == 'stage_start' and e['at'] > terminal['at'] for e in events)

by_case = {c['id']: c for c in cases}
old = {o['id']: o for o in generated['s1729_induction_select_epoch1']}
before = {o['id']: o for o in baseline}
final = {o['id']: o for o in generated['s1729_induction_select_epoch2']}
error_sets = [{o['id'] for o in outs if independently_parse(o['generated']) != o['gold_decision']}
              for outs in (list(old.values()), list(final.values()))]
traces = []
for case_id in sorted(error_sets[0] | error_sets[1]):
    traces.append(dict(case=by_case[case_id],
                       competent=before[case_id]['generated']['text'],
                       epoch1=old[case_id]['generated']['text'],
                       epoch2=final[case_id]['generated']['text'],
                       error_epoch1=case_id in error_sets[0], error_epoch2=case_id in error_sets[1]))
category_archive, category_template, domains = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
target_counts = Counter()
for row in data:
    target_counts[independently_parse(dict(text=row['target'], finish_reason='eos'))] += 1
    cat = row.get('authored_error_category')
    domains[row['facts']['domain']]['bad' if cat else 'preservation'] += 1
    if cat:
        assert independently_label(row) == 'REPORT' and row['stratum'] == 'eliciting_report'
        category_archive[cat][row['facts']['archive']] += 1
        category_template[cat][row['template_id']] += 1

allocation = read(ROOT / 'results/allocation.json')
released = read(ROOT / 'results/allocation_released.json')
restoration = read(ROOT / 'results/service_restoration.json')
prior_dir = ROOT / 'results/operational_resume/prior_attempt'
prior_allocation = read(prior_dir / 'allocation.json')
prior_released = read(prior_dir / 'allocation_released.json')
prior_restoration = read(prior_dir / 'service_restoration.json')
elapsed = released['released_unix'] - allocation['started_unix']
prior_elapsed = prior_released['released_unix'] - prior_allocation['started_unix']
assert elapsed == restoration['research_allocation_elapsed_seconds']
assert prior_elapsed == prior_restoration['research_allocation_elapsed_seconds']
assert allocation['budget_seconds'] == 28800 - math.ceil(prior_elapsed)
assert allocation['hard_allocation_deadline_unix'] - allocation['started_unix'] == allocation['budget_seconds']
assert allocation['hard_allocation_deadline_unix'] - allocation['deadline_unix'] == 180
assert released['released_unix'] <= allocation['hard_allocation_deadline_unix']
assert elapsed + prior_elapsed <= 28800 and elapsed + math.ceil(prior_elapsed) <= 28800
for receipt in (restoration, prior_restoration):
    assert receipt['health_status'] == 200 and receipt['same_original_container_and_image'] is True
    assert receipt['research_running'] is False
    assert receipt['session_error'] is None and receipt['cleanup_error'] is None
    for key in ('service_id', 'service_image'):
        assert receipt[key] == prior_allocation[key] == allocation[key]
assert released['research_running'] is False and prior_released['research_running'] is False
assert restoration['program_returncode'] == 0
operational_receipts = {str(p.relative_to(ROOT)): sha(p) for p in (
    ROOT / 'results/allocation.json', ROOT / 'results/allocation_released.json',
    ROOT / 'results/service_restoration.json', prior_dir / 'allocation.json',
    prior_dir / 'allocation_released.json', prior_dir / 'service_restoration.json')}

result = dict(at=datetime.now(timezone.utc).isoformat(),
              scientific_status='valid_early_failure_at_induction_development_selection',
              operational_status='closed_original_service_restored_and_cumulative_budget_verified',
              operational_closure=dict(receipts_sha256=operational_receipts,
                                       original_service_id=restoration['service_id'],
                                       original_service_image=restoration['service_image'],
                                       health_status=200, restored_at=restoration['restored_at'],
                                       research_running=False, prior_allocation_seconds=prior_elapsed,
                                       resumed_allocation_seconds=elapsed,
                                       actual_cumulative_seconds=prior_elapsed + elapsed,
                                       conservatively_charged_seconds=math.ceil(prior_elapsed) + elapsed,
                                       original_budget_seconds=28800, budget_verified=True),
              freeze_sha256=sha(ROOT / 'FREEZE.json'), frozen_artifacts_verified=len(freeze['files']),
              recovery_freeze_sha256=sha(ROOT / 'results/operational_resume/RESUME_FREEZE.json'),
              reused_interim_audit_sha256=sha(ROOT / 'results/postrun_review/COMPETENCE_AND_INDUCTION_EPOCH1_REVIEW.json'),
              completed_stages=provenance, gate_reviews=gate_reviews,
              terminal=terminal, terminal_sha256=sha(ROOT / 'results/TERMINAL.json'),
              no_downstream_execution=True,
              optimizer_epoch2=dict(previous_steps=32, cumulative_steps=64,
                                    parameter_states=504, exact_input_adapter_and_optimizer=True,
                                    tensor_fingerprint_continuity=True, nonzero_finite_moments=True,
                                    initial_optimizer_sha256=train2['initial_optimizer_sha256'],
                                    final_optimizer_sha256=sha(stages / 's1729_induction_epoch2/optimizer.pt')),
              training_epoch2=dict(unique_cases=len(data), updates=len(logs), target_tokens=target,
                                   prefix_tokens=prefix, sample_order_verified=True,
                                   mean_first8_loss=sum(x['loss'] for x in logs[:8])/8,
                                   mean_last8_loss=sum(x['loss'] for x in logs[-8:])/8,
                                   final_loss=logs[-1]['loss'],
                                   max_clipped_gradient=max(x['grad_norm_after_clip'] for x in logs)),
              error_sets=dict(epoch1=sorted(error_sets[0]), epoch2=sorted(error_sets[1]),
                              intersection=sorted(error_sets[0] & error_sets[1])),
              failure_traces=traces,
              induction_curriculum=dict(target_decisions=dict(target_counts),
                                        category_by_archive=dict(category_archive),
                                        category_by_template=dict(category_template),
                                        bad_and_preservation_by_domain=dict(domains)),
              reviewer_cuda_initialized=torch.cuda.is_initialized(),
              limitations=['One induction seed tested before required whole-program stop.',
                           'Development outcomes only for induction; no fresh induction qualification.',
                           'Token losses were audited from records, not independently recomputed from logits.',
                           'No per-example bad-target training accuracy or separate bad/preservation loss is available.',
                           'Archive phrase/category coupling and uneven domain coverage prevent a clean explanation of low induction.',
                           'No reflection, correction, or original-hypothesis test was reached.'])
(Path(__file__).parent / 'FINAL_REVIEW.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(dict(status=result['scientific_status'], stages=len(provenance), gates=len(gate_reviews),
                      epoch2=gate_reviews['s1729_induction_select_epoch2']['bad_counts'],
                      cuda_initialized=torch.cuda.is_initialized()), indent=2))
