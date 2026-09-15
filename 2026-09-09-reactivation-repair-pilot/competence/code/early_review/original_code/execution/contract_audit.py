"""Independent read-only audit of a snapshot of fully completed model stages."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from serialization import serialized_case
from task import parse_output

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def row_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

freeze = json.loads((ROOT / 'results/FROZEN_PROGRAM.json').read_text())
snapshot = ROOT / freeze['source_snapshot']
assert json.loads((snapshot / 'manifest.json').read_text()) == freeze
source_checks = []
for rel, digest in freeze['pins'].items():
    assert sha(ROOT / rel) == digest, rel
    if not rel.startswith('data/'):
        assert sha(snapshot / rel) == digest, rel
    source_checks.append(rel)
config = json.loads((ROOT / 'configs/pilot.json').read_text())
expected_sources = {r: h for r, h in freeze['pins'].items() if r.startswith('scripts/')}
token_rows = json.loads((HERE / 'TOKEN_AUDIT.json').read_text())['competence_by_id']
competence_rows = rows(ROOT / 'data/competence.jsonl')
data_cache = {'data/competence.jsonl': competence_rows}
paths = sorted(list((ROOT / 'checkpoints').glob('*/COMPLETED.json'))
               + list((ROOT / 'results/development').glob('*/COMPLETED.json')))
report = {'created_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'Only these atomically completed stages; later/active stages excluded',
          'freeze_sha256': sha(ROOT / 'results/FROZEN_PROGRAM.json'),
          'source_checks': source_checks, 'stages': [], 'adapter_weight_hashes': [],
          'paired_sample_order_groups': {}, 'generation_counts': Counter()}
orders = defaultdict(list)
artifacts_verified = 0
for path in paths:
    c = json.loads(path.read_text()); args = c['args']; parent = path.parent
    assert c['status'] == 'complete'
    assert c['source_sha256'] == expected_sources
    assert c['config'] == config and c['config_sha256'] == freeze['pins']['configs/pilot.json']
    assert c['data_sha256'] == freeze['pins'][args['data']]
    assert args['score_decision'] is False and args['max_new_tokens'] == 192
    assert args['output'] == str(parent.relative_to(ROOT))
    actual_inventory = {str(p.relative_to(parent)) for p in parent.rglob('*') if p.is_file() and p.name != 'COMPLETED.json'}
    assert actual_inventory == set(c['artifacts_sha256']), (str(parent), actual_inventory ^ set(c['artifacts_sha256']))
    for rel, digest in c['artifacts_sha256'].items():
        assert sha(parent / rel) == digest, (str(parent), rel)
        artifacts_verified += 1
    adapter = args['adapter']
    if adapter is None:
        assert c['initial_adapter_sha256'] is None
    else:
        assert sha(ROOT / adapter / 'adapter_model.safetensors') == c['initial_adapter_sha256']
    info = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'mode': c['mode'],
            'started_at': c['started_at'], 'finished_at': c['finished_at'],
            'incoming_adapter_sha256': c['initial_adapter_sha256']}
    if c['mode'] == 'train':
        match = re.fullmatch(r'competence_(uniform|weighted)_(first|last)_i(1729|2718)_epoch([123])', parent.name)
        assert match, parent.name
        weight, order, installation, epoch = match.groups(); installation, epoch = int(installation), int(epoch)
        assert args['class_weighting'] == ('uniform' if weight == 'uniform' else 'reweighted')
        assert args['output_order'] == 'decision_' + order
        assert args['seed'] == installation + epoch - 1
        assert args['batch_size'] == 8 and args['effective_batch'] == 16
        assert args['steps'] == c['steps'] == c['steps_planned'] == 96
        assert args['lr'] == .0001 and c['saved_steps'] == [96]
        expected_adapter = None if epoch == 1 else str(parent.relative_to(ROOT)).replace(f'epoch{epoch}', f'epoch{epoch-1}') + '/step_0096'
        assert adapter == expected_adapter
        indices = list(range(1536)); random.Random(installation + epoch - 1).shuffle(indices)
        expected_order = [competence_rows[i]['id'] for i in indices]
        assert c['sample_order'] == expected_order and len(set(c['sample_order'])) == 1536
        orders[(installation, epoch)].append({'recipe': weight+'_'+order, 'sha256': row_sha(expected_order)})
        log = rows(parent / 'training.jsonl')
        assert len(log) == 96
        cumulative_target = cumulative_prefix = 0
        expected_classes = {d: {'examples': 0, 'weight_sum': 0., 'target_tokens': 0, 'weighted_target_tokens': 0.} for d in ('REPORT', 'CLEAR')}
        denominators = []
        for step, entry in enumerate(log, 1):
            group = [token_rows[i] for i in expected_order[(step-1)*16:step*16]]
            denominator = 0.
            for e in group:
                d = e['target_decision']; w = 1. if weight == 'uniform' else .75 if d == 'REPORT' else 1.5
                denominator += w * e['target_tokens']
                b = expected_classes[d]
                b['examples'] += 1; b['weight_sum'] += w
                b['target_tokens'] += e['target_tokens']; b['weighted_target_tokens'] += w*e['target_tokens']
                cumulative_target += e['target_tokens']; cumulative_prefix += e['prefix_tokens']
            assert entry['step'] == step and entry['lr'] == .0001
            assert entry['weighted_batch_denominator'] == denominator
            assert entry['target_tokens'] == cumulative_target and entry['prefix_tokens'] == cumulative_prefix
            assert entry['processed_class_budget'] == expected_classes
            assert all(math.isfinite(entry[k]) for k in ('loss', 'grad_norm_before_clip', 'grad_norm_after_clip'))
            assert entry['grad_norm_before_clip'] > 0 and 0 < entry['grad_norm_after_clip'] <= 1.0001
            denominators.append(denominator)
        assert cumulative_target == c['target_tokens'] == c['unique_target_tokens'] == 65378
        assert cumulative_prefix == c['prefix_tokens'] == c['unique_prefix_tokens'] == 760323
        assert c['processed_class_budget'] == expected_classes
        assert c['trainable_parameters'] == 29933568
        checkpoint = parent / 'step_0096'
        ac = json.loads((checkpoint / 'adapter_config.json').read_text())
        assert (ac['r'], ac['lora_alpha'], ac['lora_dropout'], ac['bias'], ac['task_type']) == (16, 32, 0., 'none', 'CAUSAL_LM')
        assert set(ac['target_modules']) == set(config['lora_targets'])
        assert not (checkpoint / 'generation_config.json').exists()
        with (checkpoint / 'adapter_model.safetensors').open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        tensors = {k:v for k,v in header.items() if k != '__metadata__'}
        assert len(tensors) == 36 * 7 * 2
        assert {v['dtype'] for v in tensors.values()} == {'F32'}
        assert sum(math.prod(v['shape']) for v in tensors.values()) == 29933568
        assert all('.lora_A.' in k or '.lora_B.' in k for k in tensors)
        cm = json.loads((checkpoint / 'manifest.json').read_text())
        assert cm['initial_adapter_sha256'] == c['initial_adapter_sha256']
        assert cm['training_log_prefix_sha256'] == sha(parent / 'training.jsonl')
        assert cm['training_log_prefix_bytes'] == (parent / 'training.jsonl').stat().st_size
        assert cm['sample_order'] == expected_order and cm['steps'] == 96
        digest = sha(checkpoint / 'adapter_model.safetensors')
        assert digest == cm['weight_sha256']
        report['adapter_weight_hashes'].append(digest)
        info.update(sample_order_sha256=row_sha(expected_order), trainable_parameters=29933568,
                    target_tokens=cumulative_target, prefix_tokens=cumulative_prefix,
                    denominator_min=min(denominators), denominator_max=max(denominators),
                    processed_class_budget=expected_classes)
    else:
        assert args['sample'] is False and args['seed'] == 90210 and args['batch_size'] == 16
        assert args['data'] == 'data/calibration.jsonl'
        if args['data'] not in data_cache:
            data_cache[args['data']] = rows(ROOT / args['data'])
        data = data_cache[args['data']]
        outputs = rows(parent / 'outputs.jsonl')
        assert len(outputs) == c['outputs'] == 192
        assert [r['id'] for r in outputs] == [r['id'] for r in data]
        counts = Counter()
        for output, row in zip(outputs, data):
            assert output['source_row_sha256'] == row_sha(row)
            assert output['rendered_row_sha256'] == row_sha(serialized_case(row, args['output_order']))
            assert output['parsed'] == parse_output(output['generated']['text'], row)
            generated = output['generated']; ids = generated['token_ids']
            assert 1 <= len(ids) == generated['generated_tokens'] <= 192
            assert all(type(x) is int for x in ids)
            if generated['finish_reason'] == 'eos':
                assert ids[-1] == 151645 and 151645 not in ids[:-1]
            else:
                assert generated['finish_reason'] == 'length' and 151645 not in ids and len(ids) == 192
            counts['outputs'] += 1
            counts[generated['finish_reason']] += 1
            counts['full_correct'] += output['parsed']['full_correct']
            counts['decision_correct'] += output['parsed']['decision_correct']
            counts['format_valid'] += output['parsed']['format_valid']
            counts['alternate_base_eos_151643_present'] += 151643 in ids
        report['generation_counts'].update(counts)
        info.update(counts=counts)
    report['stages'].append(info)
assert len(set(report['adapter_weight_hashes'])) == len(report['adapter_weight_hashes'])
for key, values in orders.items():
    assert len({v['sha256'] for v in values}) == 1
    report['paired_sample_order_groups'][str(key)] = values
report.update(status='passed', completed_stages=len(paths), artifacts_verified=artifacts_verified,
              selected_recipe_exists=(ROOT / 'results/SELECTED_RECIPE.json').exists(),
              validation_completion_count=len(list((ROOT / 'results/validation').glob('*/COMPLETED.json'))),
              evaluation_completion_count=len(list((ROOT / 'results/evaluation').glob('*/COMPLETED.json'))))
print(json.dumps(report, indent=2))
