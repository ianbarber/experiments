"""Read-only progressive audit of the frozen calibration run; never launch models.

Only completed immutable stages count. Mutable controller sidecars are snapshotted
and may lag those stages during a live run. JSON goes to stdout unless --output is
given; output paths must be new files under results/. --self-test uses CPU only.
The driver imports the frozen independent checker, not the production scorer.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
import independent_review as independent

EXPECTED_FREEZE = '78b7ac52aed69b0a14ee20900411d37a1c82c29341935be32d5afa206b0fff9d'
SEEDS = (1729, 2718)
VIEWS = ('seen_probe_present', 'seen_probe_omitted', 'dev_familiar', 'dev_reworded')
RECIPES = [(f'{variant}_{label}', variant, weight) for variant in ('present', 'omitted')
           for label, weight in (('w0333', 1 / 3), ('w1', 1.0), ('w3', 3.0))]
JSON_SIDECARS = ('INITIAL_GRID', 'ACQUISITION_CONTROL', 'SHORTLIST_LOCK', 'REPLICATIONS',
                 'PAIRING_DEVELOPMENT', 'CONFIRMATION_LOCK', 'CONFIRMATIONS', 'MATERIALS',
                 'TERMINAL', 'CONTENT_RESULTS', 'service_restoration', 'allocation', 'allocation_released')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def close(a, b, message, tolerance=1e-8):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), message)


def schedule(seed, purpose, identity):
    return int(independent.row_sha(dict(seed=seed, purpose=purpose, identity=identity))[:16], 16) % (2**31 - 1)


def training_order(rows, seed, epoch, pure):
    if pure:
        order = list(range(len(rows)))
        random.Random(schedule(seed, 'pure_bad_diagnostic_pass', epoch)).shuffle(order)
        return order
    groups = {key: [] for key in ('bad', 'good_report', 'good_clear')}
    for index, row in enumerate(rows):
        groups['bad' if row['is_bad'] else 'good_' + row['gold_decision'].lower()].append(index)
    for group, indices in groups.items():
        random.Random(schedule(seed, 'stratified_pass_group', dict(epoch=epoch, group=group))).shuffle(indices)
    order = []
    for batch in range(len(rows) // 16):
        indices = groups['bad'][8*batch:8*(batch+1)] + groups['good_report'][4*batch:4*(batch+1)] + groups['good_clear'][4*batch:4*(batch+1)]
        random.Random(schedule(seed, 'stratified_within_batch', dict(epoch=epoch, batch=batch))).shuffle(indices)
        order.extend(indices)
    require(sorted(order) == list(range(len(rows))), 'Training order is not one complete pass')
    return order


def groups_for(row):
    return (['all', 'bad', 'bad_category:' + row['authored_error_category']] if row['is_bad'] else
            ['all', 'good', 'good_' + row['gold_decision'].lower()])


class RunAudit:
    def __init__(self, root=ROOT, inspect_optimizer=True):
        self.root = Path(root)
        self.inspect_optimizer = inspect_optimizer
        self.hash_cache, self.rows_cache, self.fingerprint_cache = {}, {}, {}
        self.freeze = read(self.root / 'FREEZE.json')
        self.freeze_hash = digest(self.root / 'FREEZE.json')
        require(self.freeze_hash == EXPECTED_FREEZE, 'Unexpected experiment freeze identity')
        self.check_freeze()
        self.config = read(self.root / 'configs/pilot.json')
        self.provenance = read(self.root / 'inputs/PROVENANCE.json')
        self.token_receipt = read(self.root / 'reviews/TOKENIZER_FEASIBILITY.json')
        self.stages = {p.parent.name: read(p) for p in sorted((self.root / 'results/stages').glob('*/COMPLETED.json'))}
        self.stage_hashes = {name: self.sha(f'results/stages/{name}/COMPLETED.json') for name in self.stages}
        self.sidecars, self.sidecar_hashes = {}, {}
        for name in JSON_SIDECARS:
            path = self.root / 'results' / (name + '.json')
            if path.exists():
                raw = path.read_bytes()
                self.sidecars[name] = json.loads(raw)
                self.sidecar_hashes[name] = hashlib.sha256(raw).hexdigest()
        self.gate_files = {p.stem: read(p) for p in sorted((self.root / 'results/gates').glob('*.json'))}
        self.gate_hashes = {name: self.sha(f'results/gates/{name}.json') for name in self.gate_files}
        self.expected_gates, self.candidates, self.stage_reports = {}, {}, {}
        self.pending, self.errors = [], []
        model_path = self.root.parent / 'reactivation/models/Qwen2.5-3B-Instruct'
        tokenizer_config = read(model_path / 'tokenizer_config.json')
        require(digest(model_path / 'tokenizer_config.json') == self.token_receipt['tokenizer_local_files_sha256']['tokenizer_config.json'], 'Tokenizer control-token metadata changed')
        require(digest(model_path / 'tokenizer.json') == self.token_receipt['tokenizer_local_files_sha256']['tokenizer.json'], 'Tokenizer decoding source changed')
        self.special_ids = [int(k) for k, v in tokenizer_config['added_tokens_decoder'].items() if v.get('special')]
        from tokenizers import Tokenizer
        self.decoder = Tokenizer.from_file(str(model_path / 'tokenizer.json'))
        constants = ast.parse((self.root / 'scripts/program.py').read_text())
        self.reflection_constants = {node.targets[0].id: ast.literal_eval(node.value) for node in constants.body
                                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                                     and node.targets[0].id in ('SCAFFOLD', 'PROBE')}

    def local(self, path):
        p = Path(path)
        if p.is_absolute():
            try:
                p = p.relative_to('/workspace')
            except ValueError:
                p = p.relative_to(self.root)
        require('..' not in p.parts, 'Path escapes reviewed workspace')
        return self.root / p

    def relative(self, path):
        return str(self.local(path).relative_to(self.root))

    def sha(self, path):
        p = self.local(path)
        stat = p.stat()
        key = (str(p), stat.st_size, stat.st_mtime_ns)
        if key not in self.hash_cache:
            self.hash_cache[key] = digest(p)
        return self.hash_cache[key]

    def rows(self, path):
        path = self.local(path)
        key = (str(path), self.sha(path))
        if key not in self.rows_cache:
            self.rows_cache[key] = independent.read_jsonl(path)
        return self.rows_cache[key]

    def check_freeze(self):
        for name, expected in self.freeze['files'].items():
            require(self.sha(name) == expected, 'Frozen input changed: ' + name)

    def adapter(self, path):
        return {name: self.sha(self.local(path) / name) for name in ('adapter_model.safetensors', 'adapter_config.json')}

    def source_fingerprint(self, adapter):
        p = self.relative(adapter)
        if p.startswith('inputs/competence_s'):
            return self.provenance[p.rsplit('s', 1)[1]]['trainable_parameter_sha256']
        stage = Path(p).parent.name
        require(stage in self.stages, 'Input adapter has no completed producing stage: ' + stage)
        return self.stages[stage]['final_trainable_parameter_sha256']

    def fingerprint(self, adapter, names):
        key = (self.sha(self.local(adapter) / 'adapter_model.safetensors'), tuple(names))
        if key in self.fingerprint_cache:
            return self.fingerprint_cache[key]
        from safetensors import safe_open
        import numpy as np
        result, numel, shapes = hashlib.sha256(), 0, []
        with safe_open(str(self.local(adapter) / 'adapter_model.safetensors'), framework='np') as reader:
            expected_keys = [name.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.') for name in names]
            require(set(reader.keys()) == set(expected_keys), 'Adapter tensor inventory differs from trainable parameter names')
            for name, saved_name in zip(names, expected_keys):
                values = np.ascontiguousarray(reader.get_tensor(saved_name), dtype=np.float32)
                require(np.isfinite(values).all(), 'Nonfinite saved adapter tensor')
                result.update(name.encode()); result.update(str(values.shape).encode()); result.update(values.tobytes())
                numel += values.size
                shapes.append(tuple(values.shape))
        value = (result.hexdigest(), numel, shapes)
        self.fingerprint_cache[key] = value
        return value

    def expected_spec(self, name):
        base = dict(epoch=1, lr=1e-4, bad_weight=1.0, rule_variant='present', bad_only_diagnostic=False,
                    loss_only=False, sample=False, optimizer=None, generation_data=None)
        match = re.fullmatch(r's(1729|2718)_(competence_screen|baseline_present|baseline_omitted_loss)', name)
        if match:
            seed, kind = int(match[1]), match[2]
            base.update(seed=seed, adapter=f'inputs/competence_s{seed}', mode='generate' if kind == 'competence_screen' else 'diagnose')
            base['data'] = 'data/' + ('competence_probe.jsonl' if kind == 'competence_screen' else 'master_omitted.jsonl' if kind.endswith('omitted_loss') else 'master_present.jsonl')
            if kind == 'baseline_present':
                base['generation_data'] = [f'data/{view}.jsonl' for view in VIEWS]
            if kind.endswith('omitted_loss'):
                base.update(loss_only=True, rule_variant='omitted')
            return base
        match = re.fullmatch(r's(1729|2718)_(present_w0333|present_w1|present_w3|omitted_w0333|omitted_w1|omitted_w3|pure_bad_control)_(epoch|diagnose)([1243])', name)
        if match:
            seed, recipe, mode, epoch = int(match[1]), match[2], match[3], int(match[4])
            pure = recipe == 'pure_bad_control'
            variant, weight = ('present', 1.0) if pure else next((v, w) for rid, v, w in RECIPES if rid == recipe)
            base.update(seed=seed, mode='train' if mode == 'epoch' else 'diagnose', rule_variant=variant,
                        bad_weight=weight, bad_only_diagnostic=pure,
                        data='data/bad_only_diagnostic_present.jsonl' if pure else f'data/master_{variant}.jsonl')
            if mode == 'epoch':
                previous = f'results/stages/s{seed}_{recipe}_epoch{epoch-1}'
                base.update(epoch=epoch, lr=3e-4 if pure else 1e-4,
                            adapter=f'{previous}/adapter' if epoch > 1 else f'inputs/competence_s{seed}',
                            optimizer=f'{previous}/optimizer.pt' if epoch > 1 else None)
            else:
                require(epoch in ((1, 4) if pure else (1, 2, 4)), 'Unplanned diagnostic dose')
                base.update(adapter=f'results/stages/s{seed}_{recipe}_epoch{epoch}/adapter', generation_data=[f'data/{v}.jsonl' for v in VIEWS])
            require(not pure or seed == 1729, 'Pure-bad control has an unplanned seed')
            return base
        match = re.fullmatch(r'r([123])_s(1729|2718)_pairing_dev_attempts', name)
        if match:
            rank, seed = int(match[1]), int(match[2])
            selected = self.sidecars['SHORTLIST_LOCK']['candidates'][rank-1]
            base.update(mode='generate', seed=seed, sample=True, data=f'results/derived/r{rank}_s{seed}_pairing_dev_draws.jsonl',
                        adapter=f'results/stages/s{seed}_{selected["recipe"]["id"]}_epoch{selected["epoch"]}/adapter')
            return base
        match = re.fullmatch(r'q([12])_s(1729|2718)_(confirmation_competent|confirmation_induced|collection_attempts|failure_reflections|preservation_attempts|preservation_reflections)', name)
        require(match is not None, 'Unplanned actual stage: ' + name)
        rank, seed, kind = int(match[1]), int(match[2]), match[3]
        locked = next(c for c in self.sidecars['CONFIRMATION_LOCK']['candidates'] if c['locked_rank'] == rank)
        stem, pool = f'q{rank}_s{seed}', locked['pool']
        base.update(mode='generate', seed=seed, adapter=f'inputs/competence_s{seed}' if kind == 'confirmation_competent' else locked['by_seed'][str(seed)]['adapter'])
        base['data'] = (f'data/qualification_{pool}.jsonl' if kind.startswith('confirmation') else
                        f'data/preservation_{pool}.jsonl' if kind == 'preservation_attempts' else
                        f'results/derived/{stem}_collection_draws.jsonl' if kind == 'collection_attempts' else
                        f'results/derived/{stem}_{kind[:-1]}_inputs.jsonl')
        base['sample'] = kind in ('collection_attempts', 'preservation_attempts')
        return base

    def audit_stage(self, name):
        m = self.stages[name]
        folder = self.root / 'results/stages' / name
        require(m.get('status') == 'complete' and m.get('check_only') is False, 'Only actual completed stages are admissible')
        require(not (folder / 'FAILED.json').exists(), 'Stage is marked both completed and failed')
        start = read(folder / 'started.json')
        require(all(m[k] == v for k, v in start.items()), 'Immutable started contract changed in completion')
        require(m['config'] == self.config and m['config_sha256'] == self.sha('configs/pilot.json'), 'Stage configuration mismatch')
        sources = {k: v for k, v in self.freeze['files'].items() if k.startswith('scripts/') and k.endswith('.py')}
        require(m['source_sha256'] == sources, 'Stage source identity differs from frozen complete scripts inventory')
        expected = self.expected_spec(name)
        args = m['args']
        for key, value in expected.items():
            actual = args[key]
            if key in ('data', 'adapter', 'optimizer') and actual is not None:
                actual = self.relative(actual)
            if key == 'generation_data' and actual is not None:
                actual = [self.relative(p) for p in actual]
            require(actual == value, f'{name}: unplanned argument {key}: {actual!r} != {value!r}')
        require(self.relative(args['output']) == f'results/stages/{name}', 'Output directory not bound to stage')
        require(m['mode'] == expected['mode'] and args['check_only'] is False, 'Actual mode/check-only mismatch')
        require(args['batch_size'] == (16 if m['mode'] == 'train' else 32) and args['effective_batch'] == 16 and args['max_new_tokens'] == 192, 'Batch/token-cap contract changed')
        required = {'started.json', 'tokenization.jsonl'}
        if m['mode'] == 'train':
            required |= {'training.jsonl', 'exposure.jsonl', 'optimizer.pt', 'adapter/adapter_model.safetensors', 'adapter/adapter_config.json', 'adapter/manifest.json'}
        elif not args['loss_only']:
            required |= {'outputs.jsonl', 'generation_batches.json'}
        if m['mode'] == 'diagnose':
            required |= {'target_losses.jsonl', 'target_loss_summary.json'}
        require(required.issubset(m['artifacts_sha256']), 'Completion omits a required scientific artifact')
        for artifact, h in m['artifacts_sha256'].items():
            require('..' not in Path(artifact).parts and self.sha(folder / artifact) == h, 'Completed artifact changed: ' + name + '/' + artifact)
        paths = [args['data']] + (args['generation_data'] or [])
        require(m['input_files_sha256'] == {p: self.sha(p) for p in paths}, 'Stage input-file manifest differs')
        require(m['data_sha256'] == self.sha(args['data']), 'Stage primary dataset differs')
        rows = self.rows(args['data'])
        require(m['source_examples'] == len(rows) and len({row['id'] for row in rows}) == len(rows), 'Primary source inventory differs')
        require(m['initial_adapter_identity'] == self.adapter(args['adapter']), 'Input adapter weights/configuration mismatch')
        require(m['initial_optimizer_sha256'] == (self.sha(args['optimizer']) if args['optimizer'] else None), 'Input optimizer identity mismatch')
        report = dict(mode=m['mode'], seed=args['seed'], source_rows=len(rows), completion_sha256=self.stage_hashes[name],
                      input_adapter_sha256=m['initial_adapter_identity'], stage_elapsed_seconds=m['stage_elapsed_s'])
        tokens = self.rows(folder / 'tokenization.jsonl')
        expected_target_ids = [r['id'] for r in rows] if m['mode'] in ('train', 'diagnose') else []
        require([r['id'] for r in tokens if r['cohort'] == 'target_loss'] == expected_target_ids, 'Target tokenization coverage/order differs')
        if expected_target_ids:
            reference = self.token_receipt['model_input_files'][self.relative(args['data'])]
            by_id = {r['id']: r for r in reference['row_token_counts']}
            for token in (r for r in tokens if r['cohort'] == 'target_loss'):
                ref = by_id[token['id']]
                require((token['prefix_tokens'], token['target_tokens'], token['sequence_tokens']) == (ref['prefix_tokens'], ref['target_tokens'], ref['training_sequence_tokens']), 'Actual target tokenization differs from frozen CPU receipt')
        require(m['tokenization']['tokenized_examples'] == len(tokens) and m['tokenization']['truncation'] == 'none'
                and m['tokenization']['max_sequence_tokens'] == max(r['sequence_tokens'] for r in tokens)
                and all(r['sequence_tokens'] <= 2048 for r in tokens), 'Tokenization summary/truncation mismatch')
        if m['mode'] == 'train':
            report['training'] = self.audit_training(name, rows, tokens)
        else:
            require(m['inference_parameter_sha256'] == self.source_fingerprint(args['adapter']), 'Inference did not load expected adapter tensor values')
            if not args['loss_only']:
                report['generation'] = self.audit_generation(name)
            if m['mode'] == 'diagnose':
                summary = read(folder / 'target_loss_summary.json')
                require(summary == m['target_loss_diagnostics'], 'Target-loss completion summary changed')
                report['target_loss'] = independent.audit_target_losses(rows, self.rows(folder / 'target_losses.jsonl'), summary, bad_weight=args['bad_weight'])
                target_tokens = {r['id']: r['target_tokens'] for r in tokens if r['cohort'] == 'target_loss'}
                require(all(r['target_tokens'] == target_tokens[r['id']] for r in self.rows(folder / 'target_losses.jsonl')), 'Target NLL denominator differs from tokenized labels')
        require(self.sha(folder / 'COMPLETED.json') == self.stage_hashes[name], 'Completion changed during audit')
        return report

    def audit_training(self, name, rows, tokens):
        m = self.stages[name]; args = m['args']; folder = self.root / 'results/stages' / name
        pure, epoch = args['bad_only_diagnostic'], args['epoch']
        steps = 32 if pure else 64
        require(len(rows) == (512 if pure else 1024), 'Training pool size differs')
        require(m['steps'] == m['steps_planned'] == steps and m['cumulative_optimizer_steps'] == steps*epoch
                and m['optimizer_previous_steps'] == steps*(epoch-1) and m['optimizer_resumed'] == (epoch > 1), 'Optimizer step/pass continuity differs')
        require(m['training_rng_seed'] == schedule(args['seed'], 'training_pass_rng', epoch), 'Training RNG seed differs')
        require(m['initial_trainable_parameter_sha256'] == self.source_fingerprint(args['adapter']), 'Initial parameter values differ from prior checkpoint')
        names = m['trainable_parameter_names']
        input_values, numel, _ = self.fingerprint(args['adapter'], names)
        final_values, final_numel, shapes = self.fingerprint(folder / 'adapter', names)
        require(input_values == m['initial_trainable_parameter_sha256'] and final_values == m['final_trainable_parameter_sha256']
                and final_values != input_values and numel == final_numel == m['trainable_parameters'], 'Saved adapter tensor fingerprint/update evidence mismatch')
        require(m['adapter_identity'] == self.adapter(folder / 'adapter'), 'Final adapter file identities differ')
        by_id = {row['id']: row for row in rows}; token_by_id = {row['id']: row for row in tokens}
        order = training_order(rows, args['seed'], epoch, pure)
        expected_exposure = [dict(id=rows[i]['id'], source_row_sha256=independent.row_sha(rows[i]),
                                 target_tokens=token_by_id[rows[i]['id']]['target_tokens'], weight=args['bad_weight'] if rows[i]['is_bad'] else 1.0) for i in order]
        exposure = self.rows(folder / 'exposure.jsonl')
        require(exposure == expected_exposure and m['actual_exposure_sha256'] == independent.row_sha(exposure)
                and m['sample_order'] == [r['id'] for r in exposure], 'Training exposure/order/source identity differs')
        logs = self.rows(folder / 'training.jsonl'); cumulative = defaultdict(lambda: defaultdict(float))
        require(len(logs) == steps, 'Training log does not cover every optimizer update')
        for number, row in enumerate(logs, 1):
            batch = exposure[(number-1)*16:number*16]
            require(row['step'] == number and row['cumulative_step'] == steps*(epoch-1)+number and row['epoch'] == epoch
                    and row['ids'] == [e['id'] for e in batch] and row['lr'] == args['lr'], 'Training update chronology/IDs changed')
            if not pure:
                require(Counter('bad' if by_id[e['id']]['is_bad'] else by_id[e['id']]['gold_decision'] for e in batch) == dict(bad=8, REPORT=4, CLEAR=4), 'Batch does not have exact 8:4:4 exposure')
            denominator = sum(e['weight']*e['target_tokens'] for e in batch)
            close(row['weighted_target_token_denominator'], denominator, 'Weighted batch denominator differs')
            expected_groups = defaultdict(lambda: dict(examples=0, target_tokens=0, weighted_target_tokens=0.0))
            for e in batch:
                for group in groups_for(by_id[e['id']]):
                    values = expected_groups[group]; values['examples'] += 1; values['target_tokens'] += e['target_tokens']; values['weighted_target_tokens'] += e['weight']*e['target_tokens']
            reported = row['groups_at_processing']
            require(set(reported) == set(expected_groups), 'Training batch group inventory differs')
            for group, values in expected_groups.items():
                g = reported[group]
                for key, value in values.items(): close(g[key], value, 'Training group denominator differs: ' + group)
                require(g['loss_sum'] >= 0 and math.isfinite(g['loss_sum']), 'Invalid recorded group loss')
                close(g['mean_target_nll'], g['loss_sum']/g['target_tokens'], 'Group raw NLL differs')
                close(g['weighted_mean_target_nll'], g['weighted_loss_sum']/g['weighted_target_tokens'], 'Group weighted NLL differs')
                if group != 'all': close(g['weighted_loss_sum'], g['loss_sum']*(args['bad_weight'] if group == 'bad' or group.startswith('bad_category:') else 1.0), 'Loss group weight differs')
                for key in ('examples', 'target_tokens', 'weighted_target_tokens', 'loss_sum', 'weighted_loss_sum'):
                    cumulative[group][key] += g[key]
            if not pure:
                for key in ('loss_sum', 'weighted_loss_sum'):
                    close(reported['all'][key], reported['bad'][key]+reported['good'][key], 'Bad/good loss partition differs')
            close(row['loss'], reported['all']['weighted_mean_target_nll'], 'Optimizer objective is not weighted token mean', tolerance=2e-6)
            require(math.isfinite(row['grad_norm_before_clip']) and 0 <= row['grad_norm_after_clip'] <= 1.0001, 'Invalid clipped gradient norm')
            require(set(row['cumulative_groups_at_processing']) == set(cumulative), 'Cumulative loss group inventory differs')
            for group, values in cumulative.items():
                for key, value in values.items(): close(row['cumulative_groups_at_processing'][group][key], value, 'Cumulative training loss/token evidence differs')
        require(m['groups_at_processing'] == logs[-1]['cumulative_groups_at_processing'], 'Final training groups differ from actual processing log')
        require(m['target_tokens'] == sum(e['target_tokens'] for e in exposure)
                and m['prefix_tokens'] == sum(t['prefix_tokens'] for t in tokens), 'Final exposure token coverage differs')
        optimizer_report = 'hash_and_lineage_only_by_explicit_option'
        if self.inspect_optimizer:
            import torch
            require(not torch.cuda.is_initialized(), 'Audit must not initialize CUDA')
            saved = torch.load(folder / 'optimizer.pt', map_location='cpu', weights_only=True)
            expected_continuation = dict(seed=args['seed'], data_sha256=m['data_sha256'], config_sha256=m['config_sha256'], lr=args['lr'],
                                         effective_batch=16, bad_weight=args['bad_weight'], rule_variant=args['rule_variant'],
                                         bad_only_diagnostic=pure, trainable_parameter_names=names)
            require(saved['continuation'] == expected_continuation and saved['epoch'] == epoch
                    and saved['cumulative_steps'] == epoch*steps and saved['adapter_identity'] == m['adapter_identity']
                    and saved['trainable_parameter_sha256'] == final_values, 'Saved optimizer continuation metadata differs')
            state = saved['optimizer_state_dict']; groups = state['param_groups']
            require(len(groups) == 1 and groups[0]['lr'] == args['lr'] and groups[0]['weight_decay'] == 0.0
                    and tuple(groups[0]['betas']) == (0.9, 0.999) and groups[0]['eps'] == 1e-8
                    and groups[0]['amsgrad'] is False and groups[0]['maximize'] is False, 'Optimizer algorithm configuration changed')
            params = groups[0]['params']
            require(len(params) == len(names) and set(params) == set(state['state']), 'Optimizer moment inventory differs')
            nonzero_moments = 0
            for pid, shape in zip(params, shapes):
                item = state['state'][pid]
                require(float(item['step']) == epoch*steps, 'An optimizer tensor has reset/skipped step counters')
                for key in ('exp_avg', 'exp_avg_sq'):
                    tensor = item[key]
                    require(tuple(tensor.shape) == shape and bool(torch.isfinite(tensor).all()), 'Optimizer moment tensor shape/value mismatch')
                    nonzero_moments += bool(tensor.ne(0).any())
            require(nonzero_moments > 0 and not torch.cuda.is_initialized(), 'Missing learned Adam moments or unexpected CUDA initialization')
            optimizer_report = dict(cpu_tensor_inspection=True, parameter_tensors=len(params), nonzero_moment_tensors=nonzero_moments, cumulative_steps=epoch*steps)
            del saved
        return dict(examples=len(rows), updates=steps, cumulative_steps=steps*epoch, target_tokens=m['target_tokens'],
                    weighted_bad_token_share=m['groups_at_processing']['bad']['weighted_target_tokens']/m['groups_at_processing']['all']['weighted_target_tokens'],
                    adapter_tensor_fingerprint_verified=True, optimizer=optimizer_report)

    def audit_generation(self, name):
        m = self.stages[name]; args = m['args']; folder = self.root / 'results/stages' / name
        outputs, inventory = self.rows(folder / 'outputs.jsonl'), read(folder / 'generation_batches.json')
        files = {Path(p).stem: self.relative(p) for p in (args['generation_data'] or ([args['data']] if m['mode'] == 'generate' else []))}
        require(m['generation_files'] == files and set(o['cohort'] for o in outputs) == set(files), 'Generation cohort inventory differs')
        require(m['outputs'] == len(outputs) and m['generated_tokens'] == sum(o['generated']['generated_tokens'] for o in outputs), 'Generation totals differ')
        require(m['generation_batch_inventory_sha256'] == independent.row_sha(inventory), 'Generation batch inventory fingerprint differs')
        settings = dict(do_sample=args['sample'], temperature=0.7 if args['sample'] else 1.0, top_p=0.95 if args['sample'] else 1.0,
                        top_k=0, repetition_penalty=1.0, max_new_tokens=192, num_beams=1, num_return_sequences=1, min_new_tokens=0,
                        pad_token_id=151643, eos_token_id=151645, bos_token_id=None, use_cache=True, return_dict_in_generate=False)
        require(m['generation_settings'] == settings and m['effective_generation_batch_size'] == (16 if args['sample'] else 32), 'Actual decoding settings changed')
        cohorts = {}
        token_rows = self.rows(folder / 'tokenization.jsonl')
        for cohort, path in files.items():
            rows = self.rows(path); subset = independent.cohort_outputs(outputs, cohort)
            independent.audit_generation_sources(rows, outputs, cohort=cohort, adapter_sha256=m['initial_adapter_identity']['adapter_model.safetensors'], special_token_ids=self.special_ids)
            require(all(o['source_file'] == path and o['is_bad'] == row.get('is_bad') for o, row in zip(subset, rows)), 'Generation cohort source/metadata mismatch')
            for output in subset:
                generated = output['generated']
                require(self.decoder.decode(generated['token_ids'], skip_special_tokens=True) == generated['text']
                        and self.decoder.decode(generated['token_ids'], skip_special_tokens=False) == generated['text_with_special_tokens'], 'Raw token streams do not decode to saved text')
            subset_tokens = [r for r in token_rows if r['cohort'] == cohort]
            require([r['id'] for r in subset_tokens] == [r['id'] for r in rows]
                    and all(t['reserved_generation_tokens'] == 192 and t['sequence_tokens'] == t['prefix_tokens']+192 for t in subset_tokens), 'Generation tokenization/reserve differs')
            if args['sample']:
                independent.audit_sampling(rows, outputs, inventory, cohort=cohort, seed=args['seed'])
            else:
                require(all(o['sampling'] is None for o in subset), 'Greedy output claims sampled RNG')
                expected_batches = [dict(cohort=cohort, ids=[r['id'] for r in rows[i:i+32]], sampling=None) for i in range(0, len(rows), 32)]
                require([b for b in inventory if b['cohort'] == cohort] == expected_batches, 'Greedy batch coverage/order changed')
            if 'facts' in rows[0]:
                cohorts[cohort] = independent.count_outputs(rows, subset)
            else:
                self.audit_reflection_inputs(rows)
                cohorts[cohort] = dict(rows=len(rows), eos=sum(o['generated']['finish_reason'] == 'eos' for o in subset),
                                       scope='Principles require separate semantic review; decision-parser invalidity is expected here')
        require(len(inventory) == sum(math.ceil(len(self.rows(path))/(16 if args['sample'] else 32)) for path in files.values()), 'Extra generation batches')
        return dict(outputs=len(outputs), generated_tokens=m['generated_tokens'], cohorts=cohorts, sampled=args['sample'])

    def audit_reflection_inputs(self, rows):
        all_outputs = {}
        for name, manifest in self.stages.items():
            path = self.root / 'results/stages' / name / 'outputs.jsonl'
            if path.exists() and ('collection_attempts' in name or 'preservation_attempts' in name):
                for output in self.rows(path):
                    all_outputs[(output['id'], independent.row_sha(output))] = output
        cases = {row['id']: row for pool in ('a', 'b') for kind in ('collection', 'preservation') for row in self.rows(f'data/{kind}_{pool}.jsonl')}
        for row in rows:
            output = all_outputs.get((row['source_generation_id'], row['source_generation_sha256']))
            require(output is not None and output['source_id'] == row['case_id'] == row['id'], 'Reflection is not bound to an actual case attempt')
            case = cases[row['id']]
            expected_messages = [dict(role='system', content=self.reflection_constants['SCAFFOLD']), dict(role='user', content=case['prompt']),
                                 dict(role='assistant', content=output['generated']['text']), dict(role='user', content=self.reflection_constants['PROBE'])]
            require(row['messages'] == expected_messages and row['gold_decision'] == case['gold_decision'] and row['stratum'] == case['stratum'], 'Reflection history/scaffold/probe differs')

    def outputs(self, name):
        return self.rows(f'results/stages/{name}/outputs.jsonl')

    def gate(self, name, result, kind='induction'):
        self.expected_gates[name] = dict(kind=kind, result=result)
        stored = self.gate_files.get(name)
        if stored is None:
            self.pending.append('Controller gate sidecar not yet present: ' + name)
            return
        if kind == 'induction':
            rename = lambda key: 'target_error_increase' if key == 'target_gain' else key.replace('_accuracy', '_competence') if key.startswith('baseline_') else key
            deficits = {rename(k): v for k, v in result['deficits'].items()}
            require(stored['pass'] == result['pass_'] and stored['deficits'] == deficits and stored['deficit'] == result['deficit']
                    and stored['checks'] == {k: v == 0 for k, v in deficits.items()}, 'Independent induction gate differs: ' + name)
            require(stored['target_error_count_increase'] == result['valid_error_gain'], 'Gate gain differs')
            for stratum in independent.STRATA:
                for label, source in (('competent', 'competent_counts'), ('installed', 'installed_counts')):
                    for key in ('n', 'valid', 'correct', 'valid_clear', 'valid_report'):
                        require(stored['cells'][stratum][label][key] == result[source][stratum][key], 'Gate stratum counts differ')
        elif kind == 'pair':
            require(stored['pass'] == result['pass_'] and stored['paired_cases'] == result['paired_cases']
                    and stored['valid_outputs'] == result['valid'] and stored['cases_with_failure'] == result['quality_denominator'], 'Independent sampled-pair gate differs')
        else:
            require(all(stored[k] == v for k, v in result.items()), 'Independent numerical gate differs: ' + name)

    def audit_gates(self):
        for seed in SEEDS:
            name = f's{seed}_competence_screen'
            if name in self.stages:
                counts = independent.count_outputs(self.rows('data/competence_probe.jsonl'), self.outputs(name))
                checks = {f'{label}_correct_at_least_116': counts[label]['correct'] >= 116 for label in ('REPORT', 'CLEAR')}
                checks['valid_at_least_251'] = counts['all']['valid'] >= 251
                self.gate(name, {'pass': all(checks.values()), 'checks': checks}, 'screen')
        for name, m in self.stages.items():
            match = re.fullmatch(r's(1729|2718)_(.+)_diagnose([124])', name)
            if match:
                seed, recipe, epoch = int(match[1]), match[2], int(match[3])
                baseline = f's{seed}_baseline_present'
                require(baseline in self.stages, 'Induction diagnosis precedes completed competent baseline')
                result = independent.audit_two_views({view: self.rows(f'data/{view}.jsonl') for view in VIEWS[2:]}, self.outputs(baseline), self.outputs(name))
                for view, gate in result['views'].items(): self.gate(name + '_' + view, gate)
                index = 6 if recipe == 'pure_bad_control' else next(i for i, item in enumerate(RECIPES) if item[0] == recipe)
                candidate = dict(seed=seed, recipe_index=index, epoch=epoch, development=result, diagnosis=name)
                self.candidates[(seed, index, epoch)] = candidate
                path = self.root / 'results/stages' / name / 'candidate.json'
                if path.exists(): self.compare_candidate(read(path), candidate)
                else: self.pending.append('Candidate sidecar not yet present: ' + name)
            match = re.fullmatch(r'(r[123]_s(?:1729|2718)_pairing_dev|q[12]_s(?:1729|2718)_collection)_attempts', name)
            if match:
                stem = match[1]; dev = stem.startswith('r')
                if dev: original = 'data/pairing_dev.jsonl'
                else:
                    rank = int(stem[1]); pool = next(c['pool'] for c in self.sidecars['CONFIRMATION_LOCK']['candidates'] if c['locked_rank'] == rank)
                    original = f'data/collection_{pool}.jsonl'
                cases = self.rows(original)
                require(self.rows(m['args']['data']) == [dict(row, id=f'{row["id"]}:draw:{d}', case_id=row['id'], draw_index=d) for row in cases for d in range(4)], 'Sampled derived inputs are not exact case-major copies')
                result = independent.audit_collection_attempts(cases, self.outputs(name), expected_cases=64 if dev else 320)
                self.gate(stem + '_structural_pair_yield', result, 'pair')
                first_path = self.root / 'results/derived' / (stem + '_first_attempts.jsonl')
                if first_path.exists():
                    require(self.rows(first_path) == [dict(id=r['case_id'], failure_id=r['first_failure_id'], success_id=r['first_success_id']) for r in result['cases']], 'Stored first draw selection differs from complete raw inventory')
            match = re.fullmatch(r'(q[12]_s(?:1729|2718))_confirmation_induced', name)
            if match:
                stem = match[1]; before = stem + '_confirmation_competent'
                require(before in self.stages, 'Fresh induced output lacks completed paired baseline')
                result = independent.audit_confirmation(self.rows(m['args']['data']), self.outputs(before), self.outputs(name))
                self.gate(stem + '_confirmation', result)
            match = re.fullmatch(r'(q[12]_s(?:1729|2718))_preservation_attempts', name)
            if match:
                counts = independent.count_outputs(self.rows(m['args']['data']), self.outputs(name))
                correct = {label: counts[label]['correct'] for label in ('REPORT', 'CLEAR')}
                self.gate(match[1] + '_structural_preservation_yield', {'pass': correct['REPORT'] >= 32 and correct['CLEAR'] >= 96, 'correct_attempts_by_label': correct}, 'preservation')
        late_gates = set(self.gate_files) - set(self.expected_gates)
        if late_gates and 'TERMINAL' not in self.sidecars:
            self.pending.extend('Gate appeared after completed-stage snapshot; defer to next audit: ' + name for name in sorted(late_gates))
        else:
            require(not late_gates, 'Stored gate has no completed independently audited evidence')

    def compare_candidate(self, stored, candidate):
        result = candidate['development']; seed = candidate['seed']; epoch = candidate['epoch']; index = candidate['recipe_index']
        require(stored['seed'] == seed and stored['epoch'] == epoch and stored['recipe']['index'] == index
                and stored['bad_only_diagnostic'] == (index == 6), 'Candidate identity changed')
        for key in ('deficit', 'target_distance', 'control_correct'): require(stored[key] == result[key], 'Candidate independent ranking statistic differs')
        require(stored['greedy_pass'] == result['pass_'], 'Candidate eligibility differs')
        adapter = self.stages[candidate['diagnosis']]['args']['adapter']
        require(stored['adapter'] == self.relative(adapter) and stored['adapter_files'] == {self.relative(self.local(adapter)/p): h for p,h in self.adapter(adapter).items()}, 'Candidate checkpoint differs')

    def candidate_from_stored(self, stored):
        key = (stored['seed'], stored['recipe']['index'], stored['epoch'])
        require(key in self.candidates, 'Controller candidate lacks completed raw diagnosis: ' + str(key))
        self.compare_candidate(stored, self.candidates[key])
        return self.candidates[key]

    def audit_selection(self):
        result = {}
        for file in ('INITIAL_GRID', 'ACQUISITION_CONTROL'):
            if file in self.sidecars:
                for stored in self.sidecars[file]: self.candidate_from_stored(stored)
        initial = [c for (seed,index,_), c in self.candidates.items() if seed == 1729 and index < 6]
        if len(initial) == 18:
            result['initial_grid'] = independent.audit_shortlist(initial)
            pure_present = any('pure_bad_control' in name for name in self.stages)
            require(not pure_present or not any(c['development']['pass_'] for c in initial), 'Unplanned pure-bad diagnostic after a passing grid')
            terminal_status = self.sidecars.get('TERMINAL', {}).get('status')
            if terminal_status not in (None, 'operational_error', 'resource_limited_incomplete') and not any(c['development']['pass_'] for c in initial):
                require('ACQUISITION_CONTROL' in self.sidecars
                        and {(seed,index,epoch) for seed,index,epoch in self.candidates if index == 6} == {(1729,6,1),(1729,6,4)},
                        'Completed unsuccessful grid omitted mandatory pure-bad acquisition diagnoses')
        shortlist = self.sidecars.get('SHORTLIST_LOCK')
        if shortlist:
            declared = [self.candidate_from_stored(c) for c in shortlist['candidates']]
            result['shortlist'] = independent.audit_shortlist(initial, [(c['recipe_index'], c['epoch']) for c in declared])
            ids = {(c['recipe_index'], c['epoch']) for c in declared}
            for (seed,index,epoch), candidate in self.candidates.items():
                if seed == 2718: require((index,epoch) in ids, 'Second-seed diagnosis retuned dose or recipe')
            for name,m in self.stages.items():
                if m['mode'] == 'train' and m['args']['seed'] == 2718:
                    recipe = name.split('_',1)[1].rsplit('_epoch',1)[0]
                    index = next(i for i,r in enumerate(RECIPES) if r[0] == recipe)
                    maximum = next((dose for idx,dose in ids if idx == index), None)
                    require(maximum is not None and m['args']['epoch'] <= maximum, 'Second-seed training exceeded selected dose')
        else:
            require(not any(m['mode'] == 'train' and m['args']['seed'] == 2718 for m in self.stages.values()), 'Replication began before shortlist existed')
        replications = self.sidecars.get('REPLICATIONS', [])
        for position, replicated in enumerate(replications, 1):
            require(replicated['shortlist_rank'] == position and replicated['recipe'] == shortlist['candidates'][position-1]['recipe']
                    and replicated['epoch'] == shortlist['candidates'][position-1]['epoch'], 'Replication reordered or changed shortlisted candidates')
            candidates = [self.candidate_from_stored(replicated['by_seed'][str(seed)]) for seed in SEEDS]
            require(replicated['common_greedy_pass'] == all(c['development']['pass_'] for c in candidates), 'Replicated pass flag differs')
        pair_results = self.sidecars.get('PAIRING_DEVELOPMENT', [])
        eligible = []
        for item in pair_results:
            require(next(r for r in replications if r['shortlist_rank'] == item['shortlist_rank'])['common_greedy_pass'], 'Non-common candidate entered sampled development')
            passed = []
            for seed in SEEDS:
                gate = self.expected_gates[f'r{item["shortlist_rank"]}_s{seed}_pairing_dev_structural_pair_yield']['result']
                require(item['pairing_by_seed'][str(seed)]['pass'] == gate['pass_'], 'Development sampling eligibility differs')
                passed.append(gate['pass_'])
            if all(passed): eligible.append(item)
        locked = self.sidecars.get('CONFIRMATION_LOCK')
        if locked:
            require(locked['freeze_sha256'] == self.freeze_hash and 1 <= len(locked['candidates']) <= 2, 'Confirmation lock freeze/count differs')
            require(len(locked['candidates']) == len(eligible[:2]), 'Locked candidate count differs from deterministic pairing selection')
            for position, (candidate, expected) in enumerate(zip(locked['candidates'], eligible[:2]), 1):
                require(candidate['locked_rank'] == position and candidate['pool'] == ('a','b')[position-1]
                        and {k:v for k,v in candidate.items() if k not in ('locked_rank','pool')} == expected, 'Confirmation lock differs from first two eligible candidates')
            for path,h in locked['files'].items(): require(self.sha(path) == h, 'Locked fresh data/checkpoint changed')
            lock_time = datetime.fromisoformat(locked['at'])
            for name,m in self.stages.items():
                if name.startswith('q'): require(datetime.fromisoformat(m['started_at']) > lock_time, 'Fresh stage began before candidate lock')
            result['confirmation_lock'] = dict(candidates=len(locked['candidates']), sha256=self.sidecar_hashes['CONFIRMATION_LOCK'])
        confirmations = self.sidecars.get('CONFIRMATIONS', [])
        for item in confirmations:
            gates = [self.expected_gates[f'q{item["locked_rank"]}_s{seed}_confirmation']['result'] for seed in SEEDS]
            require(item['numeric_pass'] == all(g['pass_'] for g in gates), 'Confirmation numeric-pass summary differs')
        materials = self.sidecars.get('MATERIALS', [])
        for item in materials:
            require(next(c for c in confirmations if c['locked_rank'] == item['locked_rank'])['numeric_pass'], 'Numerically failed candidate entered material collection')
            require(item['both_structural_pass'] == all(v['status'] == 'awaiting_content_review' for v in item['by_seed'].values()), 'Material both-seed status differs')
        if any('_collection_attempts' in name for name in self.stages):
            require(locked and len(confirmations) == len(locked['candidates']), 'Collection began before all locked confirmation attempts were recorded')
            last_confirmation = max(datetime.fromisoformat(m['finished_at']) for name,m in self.stages.items() if '_confirmation_' in name)
            first_collection = min(datetime.fromisoformat(m['started_at']) for name,m in self.stages.items() if '_collection_attempts' in name)
            require(last_confirmation < first_collection, 'Collection overlaps an uncompleted locked confirmation')
        result.update(completed_initial_diagnoses=len(initial), completed_replications=len(replications), paired_development_candidates=len(pair_results), confirmations=len(confirmations), material_candidates=len(materials))
        return result

    def audit_terminal(self):
        terminal = self.sidecars.get('TERMINAL')
        if terminal is None: return dict(status='run_still_active_or_terminal_not_recorded', overall_scientific_verdict=None)
        require(terminal['repair_training_launched'] is False, 'Unexpected repair training claim')
        if terminal.get('recorded_by') != 'session_wrapper_after_worker_stopped':
            require(terminal['freeze_sha256'] == self.freeze_hash, 'Terminal freeze identity mismatch')
        status = terminal['status']
        allowed = {'replicated_development_failed','pairing_development_failed','untouched_confirmation_failed','material_structural_failed','awaiting_content_review','resource_limited_incomplete','operational_error'}
        require(status in allowed, 'Unplanned terminal status')
        if status == 'replicated_development_failed':
            replicas = self.sidecars['REPLICATIONS']
            require(len(replicas) == 3 and not any(r['common_greedy_pass'] for r in replicas), 'Early-failed terminal disagrees with completed replication grid')
            require(not any(name.startswith(('r','q')) for name in self.stages), 'Downstream model work followed replicated development failure')
        elif status == 'pairing_development_failed':
            pairs = self.sidecars['PAIRING_DEVELOPMENT']
            require(pairs and not any(all(g['pass'] for g in item['pairing_by_seed'].values()) for item in pairs), 'Pairing terminal disagrees with recorded eligibility')
            require(not any(name.startswith('q') for name in self.stages), 'Fresh confirmation followed failed development pairing')
        elif status == 'untouched_confirmation_failed':
            confirmations = self.sidecars['CONFIRMATIONS']
            require(len(confirmations) == len(self.sidecars['CONFIRMATION_LOCK']['candidates']) and not any(c['numeric_pass'] for c in confirmations), 'Fresh terminal disagrees with all locked attempts')
            require(not any('_collection_' in name for name in self.stages), 'Collection followed failed untouched confirmation')
        elif status in ('material_structural_failed','awaiting_content_review'):
            any_pass = any(m['both_structural_pass'] for m in self.sidecars['MATERIALS'])
            require(any_pass == (status == 'awaiting_content_review'), 'Material terminal disagrees with structural candidates')
        restoration = self.sidecars.get('service_restoration')
        closure = 'pending'
        if restoration:
            require(restoration['same_original_container_and_image'] is True and restoration['health_status'] == 200
                    and restoration['research_running'] is False, 'Restoration is not healthy/closed; preserve operational failure')
            allocation = self.sidecars['allocation']; released = self.sidecars['allocation_released']
            elapsed = released['released_unix']-allocation['started_unix']
            close(elapsed, restoration['research_allocation_elapsed_seconds'], 'Allocation elapsed receipt mismatch', tolerance=1e-6)
            require(allocation['budget_seconds'] == 43200 and allocation['shutdown_reserve_seconds'] == 180
                    and allocation['hard_allocation_deadline_unix']-allocation['started_unix'] == 43200
                    and elapsed <= 43200 and restoration['within_allocation_budget'] is True, 'Allocated budget contract/closure differs')
            require(restoration['service_id'] == allocation['service_id'] and restoration['service_image'] == allocation['service_image'], 'Restored service identity differs from allocation original')
            closure = dict(status='healthy_original_service_restored', allocated_seconds=elapsed,
                           session_error=restoration['session_error'], cleanup_error=restoration['cleanup_error'])
        return dict(status=status, service_closure=closure, original_hypothesis_tested=False,
                    scope='Terminal numerical/provenance audit; semantic judgments and substantive interpretation require separate review')

    def run(self):
        for name in sorted(self.stages, key=lambda n:self.stages[n]['started_at']):
            try:
                self.stage_reports[name] = self.audit_stage(name)
            except Exception as exc:
                self.errors.append(dict(scope=name, error_type=type(exc).__name__, message=str(exc)))
        selection, terminal = {}, {}
        if not self.errors:
            for label, function in (('gates', self.audit_gates), ('selection', self.audit_selection), ('terminal', self.audit_terminal)):
                try:
                    value = function()
                    if label == 'selection': selection = value
                    if label == 'terminal': terminal = value
                except Exception as exc:
                    self.errors.append(dict(scope=label, error_type=type(exc).__name__, message=str(exc)))
        self.check_freeze()
        incomplete = [p.name for p in sorted((self.root / 'results/stages').iterdir()) if p.is_dir() and p.name not in self.stages]
        return dict(at=datetime.now(timezone.utc).isoformat(), audit_status='discrepancy_requires_review' if self.errors else 'completed_evidence_checked',
                    scope='Progressive CPU audit of completed artifacts; no overall scientific pass inferred', driver_sha256=digest(__file__),
                    freeze_sha256=self.freeze_hash, completed_stages=len(self.stages), audited_stages=self.stage_reports,
                    observed_gate_count=len(self.gate_files), recomputed_gates=self.expected_gates,
                    selection=selection, terminal=terminal, incomplete_or_later_stage_directories=incomplete,
                    errors=self.errors, pending_controller_sidecars=self.pending, sidecar_sha256=self.sidecar_hashes,
                    gate_sha256=self.gate_hashes, cuda_used=False)


def self_test():
    """Exercise independent schedule and raw-decoding prerequisites without GPU."""
    rows = independent.read_jsonl(ROOT / 'data/master_present.jsonl')
    for seed in SEEDS:
        for epoch in (1,2,3,4):
            order = training_order(rows, seed, epoch, False)
            require(len(order) == len(set(order)) == 1024, 'One-pass self-test failed')
            for i in range(0,1024,16):
                require(Counter('bad' if rows[j]['is_bad'] else rows[j]['gold_decision'] for j in order[i:i+16]) == dict(bad=8,REPORT=4,CLEAR=4), 'Stratification self-test failed')
    require(training_order(rows,1729,1,False) != training_order(rows,1729,2,False), 'Epoch RNG self-test failed')
    require(independent.audit_data(ROOT / 'data')['pass_'], 'Independent actual data self-test failed')
    audit = RunAudit()
    for seed in SEEDS:
        require(audit.adapter(f'inputs/competence_s{seed}') == audit.provenance[str(seed)]['adapter_sha256'], 'Imported identity self-test failed')
    for name in ('s1729_present_w0333_epoch1','s1729_omitted_w3_diagnose4','s1729_pure_bad_control_epoch4'):
        spec = audit.expected_spec(name)
        require(spec['seed'] == 1729 and spec['mode'] in ('train','diagnose'), 'Planned-stage mapping self-test failed')
    try:
        audit.expected_spec('s1729_repair_epoch1')
    except ValueError:
        pass
    else:
        raise ValueError('Unplanned repair stage was accepted')
    return dict(status='passed', scope='CPU driver self-tests; does not imply run success', schedules_checked=8, canonical_cases=3584)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--skip-optimizer-tensors', action='store_true', help='Explicit reduced audit; hashes and lineage still checked')
    args = parser.parse_args()
    result = self_test() if args.self_test else RunAudit(inspect_optimizer=not args.skip_optimizer_tensors).run()
    if args.output:
        output = args.output.resolve()
        require(output.is_relative_to(ROOT/'results'), 'Audit outputs must stay outside frozen inputs under results/')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream: stream.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(output=str(output),status=result.get('audit_status',result.get('status')),completed_stages=result.get('completed_stages'),errors=result.get('errors',[]))))
    else:
        print(json.dumps(result,indent=2))
    if result.get('errors'): raise SystemExit(1)


if __name__ == '__main__':
    main()
