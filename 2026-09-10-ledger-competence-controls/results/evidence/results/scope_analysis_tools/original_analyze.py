#!/usr/bin/env python3
"""Weights-free analysis of a completed competence-factorial program; never run models."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import random
import csv
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ARMS = ('matched_failure', 'donor_failure', 'context_only', 'correct_trace', 'direct')
INSTALLATIONS = (1729, 2718)
SEEDS = (42, 43, 44)
DRAWS, BOOTSTRAP_SEED = 10000, 20260910
PRIMARY_CONTRASTS = ('donor_failure', 'correct_trace')
DRAFT = Path(__file__).resolve().parent
RECIPES = ('uniform_first', 'weighted_first', 'uniform_last', 'weighted_last')
STRATA = ('trigger_report', 'control_report', 'clear')
RECIPE_ORDER = {r: ('decision_last' if r.endswith('_last') else 'decision_first') for r in RECIPES}
RECIPE_WEIGHTING = {r: ('reweighted' if r.startswith('weighted_') else 'uniform') for r in RECIPES}
METRICS = ('decision_correct', 'full_correct', 'syntax_valid', 'format_valid', 'internally_consistent',
           'selected_correct', 'count_correct', 'reason_correct', 'decision_valid', 'finished_eos', 'order_compliant')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def row_sha(row):
    return hashlib.sha256(canonical(row).encode()).hexdigest()


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON object key: ' + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object, parse_constant=reject_constant)


def read_rows(path):
    return [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def safe_path(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Source paths must be relative')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Source path escapes root')
    return path


@contextmanager
def frozen_modules(snapshot):
    """Import only hash-checked standard-library task/cohort/program snapshots."""
    names = ('narrative', 'task', 'serialization', 'make_data', 'cohort', 'program')
    previous = {name: sys.modules.get(name) for name in names}
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    loaded = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(name, snapshot / 'scripts' / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield loaded
    finally:
        sys.dont_write_bytecode = previous_bytecode
        for name in names:
            if previous[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]


class Evidence:
    def __init__(self, root, allow_projection=False):
        self.root = Path(root).resolve()
        self.hashes, self.recorded_weights, self.live_source_differences = {}, {}, []
        self.projection = None
        self.projections = {}
        if allow_projection:
            self.projection = self.json('PUBLIC_PROJECTION.json')
            require(self.projection.get('version') == 1 and isinstance(self.projection.get('files'), dict), 'Invalid projection index')
            self.projections = self.projection['files']
            for relative, item in self.projections.items():
                outputs = bool(re.fullmatch(r'results/(development|validation|collection|evaluation)/[^/]+/outputs\.jsonl', relative))
                allowed = ('omit_generated_token_ids', ['generated.token_ids']) if outputs else ('omit_operational_argv', ['argv'])
                require(outputs or relative == 'results/program.jsonl', 'Projection may not transform scientific code/data/contracts')
                require((item.get('operation'), item.get('removed_fields')) == allowed, 'Unapproved projection operation')
                require(all(type(item.get(k)) is str and re.fullmatch(r'[0-9a-f]{64}', item[k])
                            for k in ('original_sha256', 'projected_sha256')), 'Invalid projection hash')
                if outputs:
                    require(isinstance(item.get('row_original_sha256'), list)
                            and all(type(h) is str and re.fullmatch(r'[0-9a-f]{64}', h) for h in item['row_original_sha256']),
                            'Projection requires ordered original row identities')
                self.verify(relative, item['original_sha256'])
        self.completion = self.json('results/PROGRAM_COMPLETED.json')
        require(self.completion.get('status') == 'complete', 'Program has not completed')
        self.outcome = self.completion.get('outcome')
        require(self.outcome in ('full_fixed_comparison_completed', 'feasibility_gate_failed',
                                 'scientific_feasibility_gate_failed'), 'Unsupported or incomplete program outcome')
        self.pins = self.completion['pins']
        self.snapshot = safe_path(self.root, self.completion['source_snapshot'])
        snap = self.json(self.completion['source_snapshot'] + '/manifest.json')
        freeze = self.json('results/FROZEN_PROGRAM.json', self.completion['freeze_sha256'])
        require(canonical(snap) == canonical(freeze) and snap['pins'] == self.pins, 'Program/snapshot pin disagreement')
        for relative, digest in self.pins.items():
            if relative.startswith('data/'):
                self.verify(relative, digest)
            else:
                archived = self.completion['source_snapshot'] + '/' + relative
                self.verify(archived, digest)
                live = safe_path(self.root, relative)
                if not live.exists() or sha(live) != digest:
                    self.live_source_differences.append(relative)
        self.config = self.json(self.completion['source_snapshot'] + '/configs/pilot.json')
        preflight = self.json('results/encoding_preflight.json')
        require(preflight['status'] == 'passed', 'Encoding preflight did not pass')
        self.base_model_recorded_identity = freeze['base_model_hashes']
        require(self.base_model_recorded_identity and all(isinstance(x, str) and re.fullmatch(r'[0-9a-f]{64}', x)
                for x in self.base_model_recorded_identity.values()), 'Recorded base-model identity is malformed')
        # Token scoring is disabled in this factorial: order-dependent prefix
        # conditioning would compare different quantities.
        self.stages, self.datasets, self.weights = {}, {}, {}
        records = self.completion['completed_stages']
        require(len(records) == len({r['name'] for r in records}), 'Duplicate completed-stage name')
        require(len(records) == len({r['output'] for r in records}), 'Duplicate completed-stage output')
        for record in records:
            self.load_stage(record)
        allowed_projection_files = {'results/program.jsonl'} | {stage['output'] + '/outputs.jsonl'
            for stage in self.stages.values() if stage['contract']['mode'] == 'generate'}
        require(set(self.projections) <= allowed_projection_files, 'Projection contains an unrecorded model output')
        self.all_events = self.rows('results/program.jsonl')
        if 'results/program.jsonl' in self.projections:
            require(all('argv' not in event for event in self.all_events), 'Operational argv was not omitted as declared')
        starts = [x for x in self.all_events if x['event'] == 'program_start']
        require(len(starts) == 1 and starts[0]['source_snapshot'] == self.completion['source_snapshot'],
                'Initial program event freeze differs')
        endings = [x for x in self.all_events if x['event'] == 'program_complete']
        require(len(endings) == 1 and endings[0]['outcome'] == self.outcome, 'Final program event differs')
        self.events = [x for x in self.all_events if x['attempt'] == endings[0]['attempt']]
        require([x['name'] for x in self.events if x['event'] in ('stage_complete', 'stage_reused')] == list(self.stages),
                'Final-attempt completed/reused stage inventory/order differs')
        self.qualitative_plan = read_json(DRAFT / 'qualitative_plan.json')
        self.qualitative_plan_sha256 = sha(DRAFT / 'qualitative_plan.json')
        for label, selection in self.qualitative_plan['selection'].items():
            relative = 'data/' + label + '.jsonl'
            require(self.pins[relative] == selection['data_sha256'], 'Qualitative selection is for different data')

    def verify(self, relative, expected=None):
        path = safe_path(self.root, relative)
        require(path.is_file(), 'Required saved artifact is missing: ' + relative)
        digest = sha(path)
        projection = self.projections.get(relative)
        if projection is not None:
            require(digest == projection['projected_sha256'], 'Projected artifact hash mismatch: ' + relative)
            require(expected is None or expected == projection['original_sha256'], 'Original projection identity differs from contract: ' + relative)
        else:
            require(expected is None or digest == expected, 'Artifact hash mismatch: ' + relative)
        self.hashes[relative] = digest
        return path

    def original_hash(self, relative):
        return self.projections[relative]['original_sha256'] if relative in self.projections else self.hashes[relative]

    def json(self, relative, expected=None):
        return read_json(self.verify(relative, expected))

    def rows(self, relative):
        return read_rows(self.verify(relative))

    def data(self, relative):
        if relative not in self.datasets:
            data = self.rows(relative)
            require(len(data) == len({r['id'] for r in data}), 'Duplicate source case ID')
            self.datasets[relative] = data
        return self.datasets[relative]

    def load_stage(self, record):
        relative = record['output']
        contract = self.json(relative + '/COMPLETED.json', record['completion_sha256'])
        require(contract['status'] == 'complete', 'Stage is not complete: ' + record['name'])
        args = contract['args']
        require(canonical(record['plan']['args']) == canonical(args), 'Completed stage differs from its recorded plan')
        require(record['plan']['name'] == record['name'] and record['plan']['output'] == relative, 'Stage plan name/path differs')
        require(args['output'] == relative and args['mode'] == contract['mode'], 'Stage path/mode disagreement')
        require(contract['config_sha256'] == self.pins['configs/pilot.json'] and contract['config'] == self.config,
                'Stage configuration differs from frozen configuration')
        source_pins = {k: v for k, v in self.pins.items() if k.startswith('scripts/') and k.endswith('.py')}
        require(contract['source_sha256'] == source_pins, 'Stage source inventory/hash differs')
        require(contract['data_sha256'] == sha(safe_path(self.root, args['data'])), 'Stage input data changed')
        self.verify(args['data'], contract['data_sha256'])
        data = self.data(args['data'])
        require(len(data) == contract['source_examples'], 'Stage source count differs')
        adapter = args.get('adapter')
        expected_adapter = self.weights.get(adapter + '/adapter_model.safetensors') if adapter else None
        require(not adapter or expected_adapter is not None, 'Adapter lacks a preceding completed producer')
        require(contract['initial_adapter_sha256'] == expected_adapter, 'Stage initial adapter identity differs')
        artifacts = contract['artifacts_sha256']
        require('started.json' in artifacts, 'Stage lacks its started manifest')
        for child, digest in artifacts.items():
            require(not Path(child).is_absolute() and '..' not in Path(child).parts, 'Unsafe stage artifact path')
            path = relative + '/' + child
            if child.endswith('/adapter_model.safetensors'):
                # Identity is preserved from the completed producer contract. This
                # analysis never claims to rehash weights or requires their presence.
                self.recorded_weights[path] = digest
                self.weights[path] = digest
            else:
                self.verify(path, digest)
        started = self.json(relative + '/started.json')
        for key in ('mode', 'args', 'config', 'config_sha256', 'source_sha256', 'data_sha256',
                    'source_examples', 'initial_adapter_sha256'):
            require(canonical(started[key]) == canonical(contract[key]), 'Started/completed stage identity differs: ' + key)
        if contract['mode'] == 'generate':
            require('outputs.jsonl' in artifacts and contract['outputs'] == len(data), 'Generation count/artifact differs')
        else:
            require(contract['mode'] == 'train' and 'training.jsonl' in artifacts, 'Unknown/incomplete stage mode')
            for step in contract['saved_steps']:
                child = f'step_{step:04d}/manifest.json'
                require(child in artifacts, 'Missing checkpoint training manifest')
                saved = self.json(relative + '/' + child)
                require(saved['weight_sha256'] == self.weights[relative + f'/step_{step:04d}/adapter_model.safetensors'],
                        'Checkpoint weight identity differs from stage contract')
                for key in ('mode', 'args', 'source_sha256', 'config_sha256', 'data_sha256',
                            'initial_adapter_sha256', 'source_examples'):
                    require(canonical(saved[key]) == canonical(contract[key]), 'Checkpoint source/starting identity differs: ' + key)
                log = safe_path(self.root, relative + '/training.jsonl').read_bytes()
                require(type(saved['training_log_prefix_bytes']) is int and 0 < saved['training_log_prefix_bytes'] <= len(log),
                        'Checkpoint log-prefix byte count is invalid')
                prefix = log[:saved['training_log_prefix_bytes']]
                require(prefix.endswith(b'\n') and json.loads(prefix.splitlines()[-1])['step'] == step,
                        'Checkpoint log prefix does not end at its recorded update')
                require(hashlib.sha256(prefix).hexdigest() == saved['training_log_prefix_sha256'],
                        'Checkpoint log prefix hash differs')
                require(saved['steps'] == step, 'Checkpoint step identity differs')
        recorded_nonweights = {name for name in artifacts if not name.endswith('/adapter_model.safetensors')}
        actual_nonweights = {str(p.relative_to(self.root / relative)) for p in (self.root / relative).rglob('*')
                             if p.is_file() and p.name != 'COMPLETED.json' and p.name != 'adapter_model.safetensors'}
        require(actual_nonweights == recorded_nonweights, 'Unrecorded or missing nonweight stage artifact')
        self.stages[record['name']] = dict(record, contract=contract, data=data, path=self.root / relative,
                                                 projection=self.projections.get(relative + '/outputs.jsonl'))

    def recheck(self):
        for relative, digest in self.hashes.items():
            require(sha(safe_path(self.root, relative)) == digest, 'Input drift during analysis: ' + relative)
        require(sha(DRAFT / 'qualitative_plan.json') == self.qualitative_plan_sha256, 'Qualitative plan drift')


def strict_json_valid(text, task):
    try:
        json.loads(text, object_pairs_hook=task._unique_object, parse_constant=task._reject_constant)
        return True
    except (ValueError, TypeError, RecursionError):
        return False


def parse_stage(stage, task, serialization):
    data = stage['data']
    raw = read_rows(stage['path'] / 'outputs.jsonl')
    require([r['id'] for r in raw] == [r['id'] for r in data], 'Generated case coverage/order differs')
    order = stage['contract']['args']['output_order']
    require(stage['contract']['args']['score_decision'] is False, 'Token scoring was disabled')
    items = []
    projection = stage.get('projection')
    if projection:
        require(len(projection['row_original_sha256']) == len(raw), 'Projected row identity count differs')
    for index, (case, output) in enumerate(zip(data, raw)):
        require(output['source_row_sha256'] == row_sha(case), 'Generated source row identity differs')
        rendered = dict(case) if case.get('messages') else serialization.serialized_case(case, order)
        require(output['rendered_row_sha256'] == row_sha(rendered), 'Rendered row/order identity differs')
        detail = output['generated']
        require(type(detail['text']) is str and detail['finish_reason'] in ('eos', 'length'), 'Invalid generation detail')
        require(type(detail['generated_tokens']) is int and 0 < detail['generated_tokens'] <= 192,
                'Generation length exceeds configured cap')
        if projection:
            require('token_ids' not in detail, 'Projection declared token omission but retained token IDs')
            original_row_hash = projection['row_original_sha256'][index]
        else:
            require(len(detail['token_ids']) == detail['generated_tokens']
                    and all(type(t) is int and t >= 0 for t in detail['token_ids']), 'Saved token count/type differs')
            original_row_hash = row_sha(output)
        expected = task.gold(case)
        parsed = task.parse_output(detail['text'], case)
        require(canonical(parsed) == canonical(output['parsed']), 'Saved parser result differs from frozen parser')
        require(canonical(case['gold']) == canonical(expected) and case['correct_action'] == expected['decision'],
                'Frozen case/oracle disagreement')
        obj = parsed['parsed'] if isinstance(parsed['parsed'], dict) else {}
        item = {'id': case['id'], 'source_row_sha256': row_sha(case), 'rendered_row_sha256': row_sha(rendered),
                'output_row_sha256': original_row_hash, 'gold_decision': expected['decision'],
                'decision': obj.get('decision') if parsed['decision_valid'] else None,
                'suite': case.get('suite', case['split']), 'stratum': case['stratum'],
                'family': task.policy_family(case), 'domain': case['domain'], 'output_order': order,
                'designated_operator': case.get('designated_operator') or 'none',
                'declared_operator': parsed['operator'] or 'none', 'syntax_valid': strict_json_valid(detail['text'], task),
                'selected_correct': obj.get('selected') == expected['selected'],
                'count_correct': type(obj.get('count')) is int and obj['count'] == expected['count'],
                'reason_correct': obj.get('reason') == expected['reason'],
                'finished_eos': detail['finish_reason'] == 'eos',
                'order_compliant': tuple(obj) == serialization.KEY_ORDERS[order],
                'generated_tokens': detail['generated_tokens'], 'errors': parsed['errors']}
        item.update({key: parsed[key] for key in ('decision_correct', 'full_correct', 'format_valid',
                                                'internally_consistent', 'decision_valid', 'eligible_failure')})
        item['false_clear'] = expected['decision'] == 'REPORT' and item['decision'] == 'CLEAR'
        item['false_report'] = expected['decision'] == 'CLEAR' and item['decision'] == 'REPORT'
        require('decision_score' not in output, 'Unplanned supplemental score present')
        items.append(item)
    return items, raw


def rate(items, key, denominator=None):
    relevant = items if denominator is None else [r for r in items if r['gold_decision'] == denominator]
    require(all(type(r[key]) is bool for r in relevant), 'Metric must be a strict Boolean')
    count, total = sum(r[key] for r in relevant), len(relevant)
    return {'count': count, 'denominator': total, 'rate': count / total if total else None}


def summarize_items(items):
    return {'n': len(items), **{key: rate(items, key) for key in METRICS},
            'false_clear': rate(items, 'false_clear', 'REPORT'),
            'false_report': rate(items, 'false_report', 'CLEAR'),
            'error_labels': dict(sorted(Counter(error for row in items for error in row['errors']).items())),
            'error_components': {
                'incorrect_complete_answers': sum(not r['full_correct'] for r in items),
                'invalid_strict_json': sum(not r['syntax_valid'] for r in items),
                'invalid_exact_schema': sum(not r['format_valid'] for r in items),
                'wrong_selection': sum(not r['selected_correct'] for r in items),
                'wrong_count': sum(not r['count_correct'] for r in items),
                'wrong_reason': sum(not r['reason_correct'] for r in items),
                'inconsistent_execution': sum(not r['internally_consistent'] for r in items)},
            'error_scope': 'Components overlap; they are not additive rejection categories.',
            'declared_operators': dict(sorted(Counter(r['declared_operator'] for r in items).items()))}


def grouped_tables(checkpoint, items):
    groups = defaultdict(list)
    groups[('all', 'all')] = items
    for row in items:
        groups[('suite', row['suite'])].append(row)
        groups[('stratum', row['stratum'])].append(row)
        groups[('suite_stratum', row['suite'] + '/' + row['stratum'])].append(row)
        for field in ('family', 'domain', 'designated_operator', 'declared_operator'):
            groups[(field, row[field])].append(row)
    return [{'checkpoint': checkpoint, 'group_type': kind, 'group': group, **summarize_items(values)}
            for (kind, group), values in sorted(groups.items())]


def _interval(delta, indices):
    values = delta[indices].mean(axis=1)
    low, high = np.quantile(values, (.025, .975), method='linear')
    return {'effect': float(delta.mean()), 'ci_low': float(low), 'ci_high': float(high)}


def factorial_effects(validations, data, draws=DRAWS, seed=BOOTSTRAP_SEED):
    """One stratified case draw reused by every recipe, seed and endpoint.

    Interactions are (weighted_last-uniform_last) minus
    (weighted_first-uniform_first). Optimization seeds are averaged, not sampled.
    """
    expected_names = {f'validation_{r}_i{s}' for r in RECIPES for s in INSTALLATIONS}
    require(set(validations) == expected_names, 'Factorial needs all eight final validation cells')
    require(len(data) == 384 and Counter(r['stratum'] for r in data) == Counter(dict.fromkeys(STRATA, 128)),
            'Validation must contain exactly 128 cases per stratum')
    ids = [r['id'] for r in data]
    require(len(set(ids)) == 384, 'Validation source IDs are duplicated')
    require(all([r['id'] for r in rows] == ids for rows in validations.values()), 'Factorial case pairing/order differs')
    rng = np.random.default_rng(seed)
    strata_indices = {s: np.array([i for i, r in enumerate(data) if r['stratum'] == s]) for s in STRATA}
    shared = {s: indices[rng.integers(0, 128, size=(draws, 128))] for s, indices in strata_indices.items()}
    full_indices = np.concatenate([shared[s] for s in STRATA], axis=1)
    results = []
    cells = []
    for endpoint, metric, subset in (('clear_decision_accuracy', 'decision_correct', 'clear'),
                                     ('complete_correctness', 'full_correct', None)):
        arrays = {}
        for recipe in RECIPES:
            arrays[recipe] = np.array([[float(r[metric]) for r in validations[f'validation_{recipe}_i{s}']]
                                      for s in INSTALLATIONS])
            take = strata_indices[subset] if subset else np.arange(384)
            for i, installation in enumerate(INSTALLATIONS):
                count = int(arrays[recipe][i, take].sum())
                cells.append({'recipe': recipe, 'installation': installation, 'endpoint': endpoint,
                              'count': count, 'denominator': len(take), 'rate': count / len(take)})
        uf, wf, ul, wl = (arrays[r] for r in RECIPES)
        effects = {'reweighting': ((wf - uf) + (wl - ul)) / 2,
                   'decision_last': ((ul - uf) + (wl - wf)) / 2,
                   'interaction': (wl - ul) - (wf - uf)}
        for name, seed_deltas in effects.items():
            delta = seed_deltas.mean(axis=0)
            if subset:
                take = strata_indices[subset]
                remap = np.full(384, -1, dtype=int); remap[take] = np.arange(128)
                indices = remap[shared[subset]]
                delta = delta[take]
                seed_deltas = seed_deltas[:, take]
            else:
                indices = full_indices
            results.append({'endpoint': endpoint, 'contrast': name, 'primary': name != 'interaction',
                            'seed_effects': {str(s): float(seed_deltas[i].mean()) for i, s in enumerate(INSTALLATIONS)},
                            'observed_seed_mean': _interval(delta, indices),
                            'n_cases': len(delta), 'observed_seeds': list(INSTALLATIONS)})
    return {'cell_means': cells, 'effects': results, 'draws': draws, 'bootstrap_seed': seed,
            'stratum_draw_order': list(STRATA), 'stratum_sizes': dict.fromkeys(STRATA, 128),
            'shared_draws_sha256': hashlib.sha256(full_indices.astype('<i8').tobytes()).hexdigest(),
            'method': 'Stratified paired case percentile bootstrap; 128 draws per stratum; identical draws across recipes and seeds; average the two observed seed effects before interval calculation.',
            'interaction_orientation': '(weighted_last-uniform_last)-(weighted_first-uniform_first)',
            'interval_scope': 'Unadjusted descriptive intervals conditional on these cases and two observed optimization seeds; no seed resampling or model-family uncertainty.'}


def bootstrap_contrast(matched, comparator, draws=DRAWS, seed=BOOTSTRAP_SEED):
    a, b = np.asarray(matched, dtype=float), np.asarray(comparator, dtype=float)
    if a.ndim == 1:
        a, b = a[None, :], b[None, :]
    require(a.shape == b.shape and a.ndim == 2 and a.shape[1] > 0, 'Paired contrast shape differs')
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite contrast input')
    delta = (a - b).mean(axis=0)
    indices = np.random.default_rng(seed).integers(0, len(delta), size=(draws, len(delta)))
    return {**_interval(delta, indices), 'n_cases': len(delta), 'observed_seeds': a.shape[0],
            'draws': draws, 'bootstrap_seed': seed,
            'interval_scope': 'Whole-primary-set paired case percentile bootstrap; class counts vary per resample; no seed/installation resampling.'}


def evaluation_contrasts(evaluations, data):
    expected = {f'{arm}_i{i}_s{s}' for i in INSTALLATIONS for arm in ARMS for s in SEEDS}
    expected |= {f'{label}_{i}' for i in INSTALLATIONS for label in ('competent', 'bad')}
    require(set(evaluations) == expected, 'Repair comparison needs exactly 30 repairs and four controls')
    require(Counter(r['suite'] for r in data) == Counter(id=192, heldout=192, narrative=384), 'Repair suite counts differ')
    for suite, n in (('id', 64), ('heldout', 64), ('narrative', 128)):
        require(Counter(r['stratum'] for r in data if r['suite'] == suite) == Counter(dict.fromkeys(STRATA, n)),
                'Repair evaluation stratum balance differs')
    primary = [r for r in data if r['suite'] == 'narrative' and r['stratum'] in ('trigger_report', 'clear')]
    require(len(primary) == 256, 'Repair primary case count differs')
    ids = [r['id'] for r in primary]
    def values(name):
        rows = evaluations[name]
        require([r['id'] for r in rows] == [r['id'] for r in data], 'Repair case coverage/order differs')
        indexed = {r['id']: r for r in rows}
        return [float(indexed[i]['full_correct']) for i in ids]
    contrasts, cells = [], []
    for installation in INSTALLATIONS:
        for arm in ARMS:
            for seed in SEEDS:
                vals = values(f'{arm}_i{installation}_s{seed}')
                cells.append({'installation': installation, 'arm': arm, 'seed': seed,
                              'count': int(sum(vals)), 'denominator': 256, 'rate': float(np.mean(vals))})
        matched = [values(f'matched_failure_i{installation}_s{s}') for s in SEEDS]
        for comparator in ('donor_failure', 'correct_trace', 'context_only', 'direct'):
            other = [values(f'{comparator}_i{installation}_s{s}') for s in SEEDS]
            contrasts.append({'installation': installation, 'comparator': comparator,
                              'orientation': 'matched_failure minus ' + comparator,
                              'primary': comparator in PRIMARY_CONTRASTS,
                              'seed_effects': {str(s): float(np.mean(np.array(a) - np.array(b)))
                                               for s, a, b in zip(SEEDS, matched, other)},
                              'observed_seed_mean': bootstrap_contrast(matched, other)})
    return {'primary_ids': ids, 'cell_means': cells, 'contrasts': contrasts,
            'primary_scope': 'Narrative trigger REPORT 128 plus CLEAR 128; complete oracle correctness.',
            'installation_scope': 'Two separately reported installations; three repair seeds per installation are not independent installations.'}
def program_summary(items):
    groups = defaultdict(list)
    for item in items:
        groups['all'].append(item)
        group = {'trigger_report': 'target', 'control_report': 'report_control', 'clear': 'clear_control'}[item['stratum']]
        groups[group].append(item)
        groups['family:' + item['family']].append(item)
    keys = ('format_valid', 'decision_correct', 'internally_consistent', 'full_correct', 'false_clear', 'false_report')
    return {group: {'n': len(rows), **{k: sum(r[k] for r in rows) / len(rows) for k in keys}}
            for group, rows in groups.items()}


def validate_data(evidence, modules):
    md, task = modules['make_data'], modules['task']
    manifest = evidence.json('data/manifest.json')
    for name, item in manifest['files'].items():
        evidence.verify('data/' + name, item['sha256'])
    require(all(evidence.pins[name] == digest for name, digest in manifest['source_hashes'].items()),
            'Data-authoring source differs from frozen model-program source')
    ev_manifest = evidence.json('data/evaluation_manifest.json')
    require(ev_manifest['development_manifest_sha256'] == evidence.hashes['data/manifest.json'],
            'Evaluation freeze references another data manifest')
    require(ev_manifest['sha256'] == evidence.pins['data/evaluation.jsonl'], 'Evaluation data hash differs')
    exclusions = md.exclusion_set(evidence.json('data/prior_semantic_exclusions.json'))
    seen, seen_ids = set(exclusions), set()
    for name, size in md.SPLIT_SIZES.items():
        rows = evidence.data('data/' + name + '.jsonl')
        require(len(rows) == size, 'Frozen split size differs: ' + name)
        for row in rows:
            signature = task.semantic_signature(row)
            require(signature == row['semantic_hash'] and signature not in seen, 'Source semantic duplication/drift')
            require(row['id'] not in seen_ids, 'Source case ID appears in multiple splits')
            require(canonical(task.gold(row)) == canonical(row['gold']), 'Source oracle target differs')
            require(row['prompt'] == modules['serialization'].render_case(row, 'decision_first'), 'Stored source prompt differs')
            seen.add(signature); seen_ids.add(row['id'])
        require(canonical(md.summarize(rows)) == canonical(manifest['splits'][name]), 'Data summary differs')
    competence = evidence.data('data/competence.jsonl')
    require(manifest['shared_competence']['case_ids_in_shared_order'] == [r['id'] for r in competence],
            'Shared competence source order differs')
    return {'prior_excluded_cases': len(exclusions), 'fresh_cases': len(seen_ids),
            'prior_new_overlap': 0, 'fresh_cross_split_overlap': 0,
            'sizes': md.SPLIT_SIZES, 'shared_competence_cases': len(competence)}


def validate_program(evidence, modules, parsed):
    """Reconstruct every permitted stage and the selection-before-validation path."""
    program, cohort = modules['program'], modules['cohort']
    require(tuple(program.RECIPES) == RECIPES and tuple(program.INSTALLATIONS) == INSTALLATIONS,
            'Frozen factorial recipes/seed blocks differ')
    expected = []
    trajectories = []
    summaries = {}
    for order in ('decision_first', 'decision_last'):
        name = 'base_calibration_' + order
        expected.append(program.spec(name, 'generate', 'data/calibration.jsonl', 'results/development/' + name, output_order=order))
    expected.extend(program.factorial_plan())
    require(len(expected) == 50, 'Factorial plan must include two baselines and all 48 train/calibration stages')
    require(list(evidence.stages)[:50] == [s['name'] for s in expected], 'All three epochs of all eight chains are required')
    for name, rows in parsed.items():
        if name.startswith(('base_calibration_', 'competence_', 'validation_', 'induction_')):
            summaries[name] = program_summary(rows)
            producer = program.summarize(evidence.stages[name]['path'] / 'outputs.jsonl',
                                        evidence.root / evidence.stages[name]['contract']['args']['data'],
                                        evidence.stages[name]['contract']['args']['output_order'])
            require(canonical(producer) == canonical(summaries[name]), 'Independent aggregate differs from frozen program summary')
    epoch3 = {r: {} for r in RECIPES}
    contracts, adapters, adapter_hashes = {}, {}, {}
    comp_events = []
    for spec in program.factorial_plan():
        if spec['kind'] != 'competence_calibration':
            continue
        name, recipe, installation, epoch = spec['name'], spec['recipe'], spec['installation'], spec['epoch']
        gate = program.competence_gate(summaries[name])
        trajectories.append(dict(stage=name, kind='competence', recipe=recipe, installation=installation, epoch=epoch, **gate))
        comp_events.append(dict(event='competence_gate', recipe=recipe, installation=installation,
                                epoch=epoch, descriptive_only=epoch != 3, **gate))
        if epoch == 3:
            key = recipe + ':' + str(installation)
            epoch3[recipe][str(installation)] = summaries[name]
            contracts[key] = evidence.stages[name]['completion_sha256']
            adapters[(recipe, installation)] = spec['args']['adapter']
            adapter_hashes[key] = evidence.weights[spec['args']['adapter'] + '/adapter_model.safetensors']
    actual = [{k: v for k, v in event.items() if k not in ('at', 'attempt')} for event in evidence.events
              if event['event'] == 'competence_gate']
    require(canonical(actual) == canonical(comp_events), 'Recorded competence gates differ')
    selection = program.select_recipe(epoch3)
    selection.update(source_snapshot=evidence.completion['source_snapshot'], calibration_contracts=contracts,
                     epoch3_adapter_hashes=adapter_hashes)
    saved_selection = evidence.json('results/SELECTED_RECIPE.json')
    require(canonical(saved_selection) == canonical(selection), 'Recipe selection drift or non-priority selection')
    require(evidence.completion['selected_recipe'] == selection['selected_recipe'], 'Completed program changed selected recipe')
    selection_events = [e for e in evidence.all_events if e['event'] == 'recipe_selection_frozen']
    require(selection_events and all(e['selected_recipe'] == selection['selected_recipe']
            and e['sha256'] == evidence.hashes['results/SELECTED_RECIPE.json'] for e in selection_events),
            'Selection event identity differs')
    first_selection = next(i for i, e in enumerate(evidence.all_events) if e['event'] == 'recipe_selection_frozen')
    validation_starts = [i for i, e in enumerate(evidence.all_events)
                         if e['event'] == 'stage_start' and e.get('name', '').startswith('validation_')]
    require(validation_starts and first_selection < min(validation_starts), 'Recipe was not frozen before validation starts')
    validation = {r: {} for r in RECIPES}
    for recipe, installation in program.cells():
        name = f'validation_{recipe}_i{installation}'
        expected.append(program.spec(name, 'generate', 'data/validation.jsonl', 'results/validation/' + name,
                                     adapter=adapters[(recipe, installation)], output_order=RECIPE_ORDER[recipe]))
        require(name in parsed, 'Missing final validation cell')
        validation[recipe][str(installation)] = summaries[name]
        trajectories.append(dict(stage=name, kind='validation', recipe=recipe, installation=installation,
                                 epoch=3, **program.competence_gate(summaries[name])))
    decision = program.validation_decision(selection, validation)
    saved_validation = evidence.json('results/VALIDATION_RESULT.json')
    require(canonical(saved_validation) == canonical({**decision, 'selected_recipe_sha256': evidence.hashes['results/SELECTED_RECIPE.json']}),
            'Validation decision or no-fallback rule differs')
    require(decision['fallback_allowed'] is False, 'Validation fallback was not prohibited')
    stop = None if decision['passed'] else {'stage': 'calibration_selection' if selection['selected_recipe'] is None else 'validation',
                                           'reason': decision['reason']}
    selected, cohorts, induction_events = {}, {}, []
    if stop is None:
        recipe = selection['selected_recipe']; order = RECIPE_ORDER[recipe]
        with tempfile.TemporaryDirectory(prefix='competence-cohort-replay-') as temporary:
            for installation in INSTALLATIONS:
                competent = adapters[(recipe, installation)]
                name = f'induction_{installation}'
                expected.append(program.spec(name, 'train', 'data/induction.jsonl', 'checkpoints/' + name,
                    adapter=competent, seed=installation, steps=96, save_steps=[32, 64, 96], output_order=order))
                bad = None
                for step in (32, 64, 96):
                    name = f'induction_{installation}_step{step}_calibration'
                    candidate = f'checkpoints/induction_{installation}/step_{step:04d}'
                    expected.append(program.spec(name, 'generate', 'data/calibration.jsonl', 'results/development/' + name,
                                                 adapter=candidate, output_order=order))
                    require(name in parsed, 'Missing prescribed induction candidate')
                    gate = program.induction_gate(summaries[name], epoch3[recipe][str(installation)])
                    trajectories.append(dict(stage=name, kind='induction', recipe=recipe, installation=installation, step=step, **gate))
                    induction_events.append(dict(event='induction_gate', installation=installation, step=step, **gate))
                    if gate['passed']:
                        bad = candidate
                        break
                if bad is None:
                    stop = {'stage': 'induction', 'installation': installation,
                            'reason': f'Installation {installation}: none of the fixed induction candidates passed.'}
                    break
                name = f'failures_{installation}'
                expected.append(program.spec(name, 'generate', 'data/pool.jsonl', f'results/collection/{installation}',
                                             adapter=bad, seed=314159 + installation, sample=True, output_order=order))
                require(name in parsed, 'Missing fixed actual-failure collection')
                try:
                    reproduced = cohort.construct(evidence.root / 'data/pool.jsonl',
                        evidence.stages[name]['path'] / 'outputs.jsonl', evidence.root / 'data/preservation.jsonl',
                        Path(temporary) / str(installation), installation, output_order=order)
                except ValueError as error:
                    stop = {'stage': 'cohort', 'installation': installation,
                            'reason': f'Installation {installation}: {error}'}
                    cohorts[str(installation)] = {'passed': False, 'reason': str(error)}
                    break
                raw_relative = evidence.stages[name]['output'] + '/outputs.jsonl'
                if raw_relative in evidence.projections:
                    # Only a provenance field differs when token IDs were omitted.
                    # Reconstructed selected cases, traces and all training files remain exact.
                    reproduced['outputs_sha256'] = evidence.original_hash(raw_relative)
                saved = evidence.json(f'data/cohort_{installation}/COMPLETED.json')
                require(canonical(saved) == canonical(reproduced), 'Deterministic cohort reconstruction differs')
                for filename, digest in reproduced['artifacts_sha256'].items():
                    evidence.verify(f'data/cohort_{installation}/{filename}', digest)
                cohorts[str(installation)] = {'passed': True, 'summary': reproduced}
                selected[str(installation)] = {'recipe': recipe, 'output_order': order, 'competent_adapter': competent,
                    'bad_adapter': bad, 'competent_sha256': evidence.weights[competent + '/adapter_model.safetensors'],
                    'bad_sha256': evidence.weights[bad + '/adapter_model.safetensors'], 'cohort': reproduced}
    actual_induction = [{k: v for k, v in event.items() if k not in ('at', 'attempt')} for event in evidence.events
                        if event['event'] == 'induction_gate']
    require(canonical(actual_induction) == canonical(induction_events), 'Induction candidate gates differ')
    require(canonical(evidence.completion['selected_installations']) == canonical(selected), 'Selected installation identities differ')
    for installation, item in cohorts.items():
        if item['passed']:
            events = [e for e in evidence.events if e['event'] == 'cohort_gate_passed' and e['installation'] == int(installation)]
            require(len(events) == 1 and canonical(events[0]['summary']) == canonical(item['summary']), 'Cohort event differs')
    if stop is None:
        require(evidence.outcome == 'full_fixed_comparison_completed', 'All gates pass but full comparison absent')
        require(canonical(evidence.json('results/SELECTED_INSTALLATIONS.json')) == canonical(selected), 'Selected cohort artifact differs')
        before_repair = [i for i, e in enumerate(evidence.all_events) if e['event'] == 'all_prerepair_gates_passed']
        repair_starts = [i for i, e in enumerate(evidence.all_events) if e['event'] == 'stage_start'
                         and e.get('name', '').startswith(('train_', 'evaluation_'))]
        require(before_repair and repair_starts and min(before_repair) < min(repair_starts), 'Repair evaluation preceded both cohort gates')
        for installation in INSTALLATIONS:
            setup = selected[str(installation)]
            for label in ('competent', 'bad'):
                name = f'{label}_{installation}'
                expected.append(program.spec('evaluation_' + name, 'generate', 'data/evaluation.jsonl',
                    'results/evaluation/' + name, adapter=setup[label + '_adapter'], output_order=setup['output_order']))
            for offset, seed in enumerate(SEEDS):
                for arm in ARMS[offset:] + ARMS[:offset]:
                    name = f'{arm}_i{installation}_s{seed}'
                    expected.append(program.spec('train_' + name, 'train', f'data/cohort_{installation}/{arm}.jsonl',
                        'checkpoints/' + name, adapter=setup['bad_adapter'], seed=seed, steps=32, output_order=setup['output_order']))
                    expected.append(program.spec('evaluation_' + name, 'generate', 'data/evaluation.jsonl',
                        'results/evaluation/' + name, adapter='checkpoints/' + name + '/step_0032', output_order=setup['output_order']))
    else:
        require(evidence.outcome == 'feasibility_gate_failed', 'Scientific stop has an incompatible outcome')
        require(evidence.completion['reason'] == stop['reason'], 'Scientific stop reason differs')
        require(evidence.completion.get('heldout_repair_results_used_for_selection') is False,
                'Stop lacks explicit no-repair-outcome-selection statement')
    actual_plan = [stage['plan'] for stage in evidence.stages.values()]
    require(canonical(actual_plan) == canonical(expected), 'Completed stage sequence/settings differ from the fixed permitted path')
    inventory = {str(p.parent.relative_to(evidence.root))
                 for directory in ('checkpoints', 'results/development', 'results/validation', 'results/collection', 'results/evaluation')
                 for p in (evidence.root / directory).rglob('COMPLETED.json')}
    require(inventory == {s['output'] for s in evidence.stages.values()}, 'Unrecorded or missing completed model stage')
    return {'base': {order: summaries['base_calibration_' + order] for order in ('decision_first', 'decision_last')},
            'candidates': trajectories, 'selection': selection, 'validation': decision,
            'cohorts': cohorts, 'stopped_at': stop, 'selected_installations': selected}


def training_budgets(evidence):
    result = []
    for name, stage in evidence.stages.items():
        contract = stage['contract']; args = contract['args']
        if contract['mode'] != 'train':
            continue
        logs = read_rows(stage['path'] / 'training.jsonl')
        require([r['step'] for r in logs] == list(range(1, contract['steps'] + 1)), 'Training update coverage differs')
        require(contract['steps'] == contract['steps_planned'] == args['steps'], 'Training budget differs')
        order = []; rng = random.Random(args['seed'])
        while len(order) < args['steps'] * 16:
            indices = list(range(len(stage['data']))); rng.shuffle(indices); order.extend(indices)
        ids = [stage['data'][i]['id'] for i in order[:args['steps'] * 16]]
        require(contract['sample_order'] == ids, 'Training sample order differs from the fixed RNG')
        for metric in ('loss', 'grad_norm_before_clip', 'grad_norm_after_clip', 'elapsed_s', 'lr', 'weighted_batch_denominator'):
            require(all(type(r[metric]) in (int, float) and math.isfinite(r[metric]) for r in logs), 'Invalid training measurement')
        require(all(0 <= r['grad_norm_after_clip'] <= 1.0001 and r['grad_norm_before_clip'] >= 0 for r in logs), 'Invalid clipping norm')
        require(all(r['lr'] == .0001 and r['weighted_batch_denominator'] > 0 for r in logs), 'Invalid learning-rate/weight denominator')
        require(logs[-1]['target_tokens'] == contract['target_tokens'] and logs[-1]['prefix_tokens'] == contract['prefix_tokens'],
                'Final token budgets differ from training contract')
        require(all(a['target_tokens'] < b['target_tokens'] and a['prefix_tokens'] < b['prefix_tokens']
                    for a, b in zip(logs, logs[1:])), 'Cumulative token accounting is not increasing')
        require(canonical(logs[-1]['processed_class_budget']) == canonical(contract['processed_class_budget']), 'Class-budget final state differs')
        labels = {r['id']: json.loads(r['target'])['decision'] for r in stage['data']}
        counts = Counter(labels[i] for i in ids)
        unique = Counter(labels[r['id']] for r in stage['data'])
        class_weight = {'REPORT': .75, 'CLEAR': 1.5} if args['class_weighting'] == 'reweighted' else {'REPORT': 1., 'CLEAR': 1.}
        require(contract['class_weighting'] == args['class_weighting'], 'Class-weighting metadata differs')
        for label in ('REPORT', 'CLEAR'):
            u, total = contract['unique_class_budget'][label], contract['processed_class_budget'][label]
            require(type(u['unique_examples']) is int and u['unique_examples'] == unique[label], 'Unique class count differs')
            require(type(total['examples']) is int and total['examples'] == counts[label], 'Processed class count differs')
            require(u['weight_sum'] == unique[label] * class_weight[label]
                    and total['weight_sum'] == counts[label] * class_weight[label], 'Example weight mass differs')
            require(u['weighted_target_tokens'] == u['target_tokens'] * class_weight[label]
                    and total['weighted_target_tokens'] == total['target_tokens'] * class_weight[label], 'Weighted token accounting differs')
        require(sum(v['target_tokens'] for v in contract['unique_class_budget'].values()) == contract['unique_target_tokens'], 'Unique target budget sum differs')
        require(sum(v['target_tokens'] for v in contract['processed_class_budget'].values()) == contract['target_tokens'], 'Processed target budget sum differs')
        require(math.isclose(sum(r['weighted_batch_denominator'] for r in logs),
                sum(v['weighted_target_tokens'] for v in contract['processed_class_budget'].values()), rel_tol=0, abs_tol=1e-9),
                'Effective-batch weighted denominators do not reconcile with cumulative class tokens')
        item = {'checkpoint': name, **{key: contract[key] for key in ('steps', 'target_tokens', 'prefix_tokens',
                'unique_examples', 'unique_target_tokens', 'unique_prefix_tokens', 'max_sequence_tokens',
                'trainable_parameters', 'elapsed_s', 'peak_cuda_allocated_bytes', 'unique_class_budget', 'processed_class_budget')},
                'examples_processed': len(ids), 'sample_order_sha256': row_sha(ids), 'output_order': args['output_order'],
                'class_weighting': args['class_weighting'], 'seed': args['seed'], 'learning_rate': args['lr'],
                'initial_adapter_sha256': contract['initial_adapter_sha256'],
                'gradient_norms': {which: {'mean': float(np.mean([r[key] for r in logs])),
                                         'max': max(r[key] for r in logs), 'min': min(r[key] for r in logs)}
                                  for which, key in [('before_clip', 'grad_norm_before_clip'), ('after_clip', 'grad_norm_after_clip')]},
                'steps_with_preclip_norm_above_one': sum(r['grad_norm_before_clip'] > 1 for r in logs)}
        result.append(item)
    for installation in INSTALLATIONS:
        for epoch in (1, 2, 3):
            contracts = [evidence.stages[f'competence_{r}_i{installation}_epoch{epoch}']['contract'] for r in RECIPES]
            require(all(c['sample_order'] == contracts[0]['sample_order'] for c in contracts), 'Factorial case exposure order differs')
            for a, b in ((0, 1), (2, 3)):
                require(contracts[a]['target_tokens'] == contracts[b]['target_tokens']
                        and contracts[a]['unique_target_tokens'] == contracts[b]['unique_target_tokens'],
                        'Reweighting changed same-order raw target exposure')
    for installation in INSTALLATIONS:
        for seed in SEEDS:
            names = [f'train_{arm}_i{installation}_s{seed}' for arm in ARMS]
            present = [n for n in names if n in evidence.stages]
            if not present:
                continue
            require(len(present) == 5, 'Incomplete repair-arm block')
            contracts = [evidence.stages[n]['contract'] for n in names]
            for key in ('sample_order', 'target_tokens', 'unique_target_tokens'):
                require(all(c[key] == contracts[0][key] for c in contracts), 'Repair correction target/order budget differs')
    return result
def percentage(value):
    return '—' if value is None else f'{100 * value:.2f}'


def qualitative_examples(evidence, raw, modules):
    result = []
    for name, outputs in raw.items():
        stage = evidence.stages[name]
        split = Path(stage['contract']['args']['data']).stem
        if split not in evidence.qualitative_plan['selection']:
            continue
        selection = evidence.qualitative_plan['selection'][split]
        cases = {r['id']: r for r in stage['data']}
        by_id = {r['id']: r for r in outputs}
        original_rows = ({r['id']: h for r, h in zip(outputs, stage['projection']['row_original_sha256'])}
                         if stage.get('projection') else {r['id']: row_sha(r) for r in outputs})
        for identifier in selection['ids']:
            case, output = cases[identifier], by_id[identifier]
            result.append({'stage': name, 'id': identifier, 'case': case,
                'rendered_prompt': modules['serialization'].render_case(case, stage['contract']['args']['output_order']),
                'generated_text': output['generated']['text'], 'generated': output['generated'],
                'oracle': modules['task'].gold(case), 'recomputed_parse': output['parsed'],
                'source_row_sha256': row_sha(case), 'output_row_sha256': original_rows[identifier],
                'source_data_sha256': stage['contract']['data_sha256'],
                'raw_file_sha256': evidence.original_hash(stage['output'] + '/outputs.jsonl'),
                'scope': 'Fixed data-only illustrative selection; not representative or human-scored.'})
    return result


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def render_tables(result):
    lines = ['# Saved-response analysis tables', '', result['measurement_scope'], '',
             '| Checkpoint | Group | N | Full correct % | Decision correct % | Strict JSON % | Exact schema % | Internal % | Order compliant % | False CLEAR % | False REPORT % |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['tables']:
        if row['group_type'] not in ('all', 'stratum'):
            continue
        values = [percentage(row[k]['rate']) for k in ('full_correct', 'decision_correct', 'syntax_valid', 'format_valid',
                                                        'internally_consistent', 'order_compliant', 'false_clear', 'false_report')]
        lines.append(f"| {row['checkpoint']} | {row['group']} | {row['n']} | " + ' | '.join(values) + ' |')
    lines += ['', '## Factorial effects', '', 'Positive main effects favor reweighting or decision-last. Interaction is secondary; intervals are unadjusted.', '',
              '| Endpoint | Effect | Seed 1729 pp | Seed 2718 pp | Observed-seed mean pp | 95% interval pp |',
              '|---|---|---:|---:|---:|---|']
    for row in result['factorial']['effects']:
        value = row['observed_seed_mean']
        lines.append(f"| {row['endpoint']} | {row['contrast']} | {100*row['seed_effects']['1729']:.3f} | "
                     f"{100*row['seed_effects']['2718']:.3f} | {100*value['effect']:.3f} | "
                     f"[{100*value['ci_low']:.3f}, {100*value['ci_high']:.3f}] |")
    lines += ['', '## Conditional repair', '']
    if result['repair'] is None:
        lines += ['No repair effect estimated. ' + result['completion_reason']]
    else:
        lines += ['| Installation | Comparator (matched minus comparator) | Mean pp | 95% interval pp | Primary |',
                  '|---:|---|---:|---|---|']
        for row in result['repair']['contrasts']:
            value = row['observed_seed_mean']
            lines.append(f"| {row['installation']} | {row['comparator']} | {100*value['effect']:.3f} | "
                         f"[{100*value['ci_low']:.3f}, {100*value['ci_high']:.3f}] | {row['primary']} |")
    return '\n'.join(lines) + '\n'


def make_figures(output, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 9})
    directory = output / 'figures'; directory.mkdir()
    marker = 'SYNTHETIC FIXTURE — NOT MODEL DATA\n' if result['synthetic_fixture'] else ''
    colors = dict(zip(RECIPES, ('#4575b4', '#d95f02', '#1b9e77', '#984ea3')))
    figure, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True)
    curve_rows = []
    for column, installation in enumerate(INSTALLATIONS):
        for recipe in RECIPES:
            base = result['development']['base'][RECIPE_ORDER[recipe]]
            series = [base] + [r['summary'] for r in result['development']['candidates']
                              if r['kind'] == 'competence' and r['recipe'] == recipe and r['installation'] == installation]
            for row_index, (group, metric, label, gate) in enumerate((('all', 'full_correct', 'Complete correctness', 85),
                                                                   ('clear_control', 'decision_correct', 'CLEAR decision accuracy', 90),
                                                                   ('report_control', 'decision_correct', 'REPORT control accuracy', 90))):
                values = [100 * x[group][metric] for x in series]
                axes[row_index, column].plot(range(4), values, 'o-', color=colors[recipe], label=recipe)
                axes[row_index, column].axhline(gate, color='#555555', linestyle=':', linewidth=.8)
                axes[row_index, column].set_title(f'Seed {installation}: {label}')
                axes[row_index, column].set_ylim(-2, 102); axes[row_index, column].grid(alpha=.15)
                for epoch, source in enumerate(series):
                    curve_rows.append({'recipe': recipe, 'installation': installation, 'epoch': epoch,
                        'metric': metric, 'group': group, 'count': round(source[group][metric] * source[group]['n']),
                        'denominator': source[group]['n'], 'rate': source[group][metric]})
    for ax in axes[-1]: ax.set_xticks(range(4), ('Base', 'Epoch 1', 'Epoch 2', 'Epoch 3'))
    for ax in axes[:, 0]: ax.set_ylabel('Percent')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    figure.suptitle(marker + 'Repeated calibration trajectories; dotted lines are fixed gates')
    figure.tight_layout(rect=(0, .06, 1, .95))
    for suffix in ('png', 'svg'): figure.savefig(directory / ('competence_trajectory.' + suffix), dpi=180)
    plt.close(figure)
    write_csv(output / 'competence_trajectory.csv', curve_rows)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, endpoint in zip(axes, ('clear_decision_accuracy', 'complete_correctness')):
        for index, installation in enumerate(INSTALLATIONS):
            rows = [next(r for r in result['factorial']['cell_means'] if r['recipe'] == recipe
                         and r['installation'] == installation and r['endpoint'] == endpoint) for recipe in RECIPES]
            x = np.arange(4) + (index - .5) * .3
            bars = ax.bar(x, [100*r['rate'] for r in rows], width=.28, label=f'Seed {installation}', color=('#4477aa', '#cc6677')[index])
            for bar, row in zip(bars, rows):
                ax.annotate(f"{row['count']}/{row['denominator']}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                            xytext=((-2 if index == 0 else 2), 3), textcoords='offset points',
                            ha=('right' if index == 0 else 'left'), va='bottom', fontsize=6.5)
        ax.set_xticks(range(4), [r.replace('_', '\n') for r in RECIPES]); ax.set_ylim(0, 112); ax.set_xlim(-.6, 3.6)
        ax.set_title(endpoint.replace('_', ' ').capitalize()); ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Percent')
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    figure.suptitle(marker + 'Fresh validation, final epoch only; paired cases and two observed seeds')
    figure.tight_layout(rect=(0, .09, 1, .91))
    for suffix in ('png', 'svg'): figure.savefig(directory / ('factorial_cells.' + suffix), dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, endpoint in zip(axes, ('clear_decision_accuracy', 'complete_correctness')):
        rows = [r for r in result['factorial']['effects'] if r['endpoint'] == endpoint]
        for index, row in enumerate(rows):
            value = row['observed_seed_mean']; center = 100 * value['effect']
            color = '#4477aa' if row['primary'] else '#999999'
            ax.hlines(index, 100*value['ci_low'], 100*value['ci_high'], color=color)
            ax.plot(center, index, 'o', color=color)
        ax.axvline(0, color='black', linewidth=.7); ax.grid(axis='x', alpha=.15)
        ax.set_yticks(range(3), ('Reweighting', 'Decision last', 'Interaction (secondary)'))
        ax.set_xlabel('Effect, percentage points'); ax.set_title(endpoint.replace('_', ' ').capitalize())
    figure.suptitle(marker + 'Observed-seed mean effects; stratified paired case 95% intervals')
    figure.tight_layout(rect=(0, 0, 1, .89))
    for suffix in ('png', 'svg'): figure.savefig(directory / ('factorial_effects.' + suffix), dpi=180)
    plt.close(figure)
    if result['repair'] is not None:
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, installation in zip(axes, INSTALLATIONS):
            rows = [r for r in result['repair']['contrasts'] if r['primary'] and r['installation'] == installation]
            for index, row in enumerate(rows):
                v = row['observed_seed_mean']
                ax.hlines(index, 100*v['ci_low'], 100*v['ci_high'], color='#1b9e77')
                ax.plot(100*v['effect'], index, 'o', color='#1b9e77')
            ax.set_yticks(range(2), ('Matched − donor failure', 'Matched − correct trace'))
            ax.axvline(0, color='black', linewidth=.7); ax.set_xlabel('Complete correctness effect, pp')
            ax.set_title(f'Installation {installation}'); ax.grid(axis='x', alpha=.15)
        figure.suptitle(marker + 'Conditional repair; whole-set paired case 95% intervals')
        figure.tight_layout(rect=(0, 0, 1, .86))
        for suffix in ('png', 'svg'): figure.savefig(directory / ('repair_effects.' + suffix), dpi=180)
        plt.close(figure)


def render_report(result):
    selected = result['development']['selection']['selected_recipe']
    lines = ['# Saved-data competence-factorial report', '', result['measurement_scope'], '',
             f"Outcome: {result['outcome']}. The fixed epoch-three calibration priority selected {selected or 'no recipe'}.", '',
             'All four recipes and both optimization seeds completed all three competence blocks and all eight final checkpoints were assessed on the same fresh validation cases. '
             'This replay validates saved measurements and does not load model weights or recreate model computation.', '',
             'The co-primary diagnostic endpoints are legitimate CLEAR decision accuracy on 128 validation cases and complete oracle correctness on all 384. '
             'Positive main effects favor reweighting or decision-last. The interaction is secondary. Four main-effect/endpoint intervals are descriptive and unadjusted.', '']
    for row in result['factorial']['effects']:
        if not row['primary']: continue
        v = row['observed_seed_mean']
        lines.append(f"- {row['endpoint']}, {row['contrast']}: {100*v['effect']:+.3f} pp "
                     f"(95% paired case interval {100*v['ci_low']:+.3f} to {100*v['ci_high']:+.3f}).")
    lines += ['', 'The factorial bootstrap draws 128 IDs independently in each validation stratum and shares every draw across recipes and both seeds. '
              'The two observed seed effects are averaged before intervals; seeds are not resampled. See tables.md and source CSVs for both seed-specific effects, cell counts, '
              'syntax versus exact-schema validity, order compliance, and overlapping execution/error components.', '']
    if result['repair'] is None:
        lines += ['The conditional repair branch stopped: ' + result['completion_reason'], '',
                  'No repair contrast is estimated. This is not a null or equivalence repair result. Validation never triggers fallback to a different recipe.']
    else:
        lines += ['Both selected seeds passed validation and all induction/cohort prerequisites. The conditional comparison contains 30 repairs and four controls. '
                  'Its primary set is the 256 narrative trigger REPORT/CLEAR cases; its separate bootstrap samples the whole set, so class counts vary per draw. '
                  'Installations are reported separately; the three repair seeds per installation are not additional installations.']
    lines += ['', 'The key-order intervention changes both the requested serialization and supervised output order. Class weighting changes the training objective on identical cases. '
              'Neither proves a hidden execution mechanism. Executable reason declarations are observable outputs, not hidden reasoning. '
              'Repeated calibration remains development; the task and narrative renderer remain controlled synthetic ledgers. '
              'If repair ran, the archived-context intervention changes a whole case and answer, and a correct-trace prefix exposes a copyable answer.', '',
              'Fixed qualitative illustrations retain whatever success or failure status occurs. They are not representative samples or human correctness judgments. '
              'Source contracts, raw-response reparsing, weight identities, and generated artifact hashes are recorded in COMPLETED.json.']
    return '\n'.join(lines) + '\n'


def analyze(root, output, allow_projection=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'Analysis output must be a new directory')
    require(not any(output.is_relative_to(root / name) for name in ('data', 'scripts', 'configs', 'checkpoints', 'results/source_snapshots')),
            'Analysis output cannot be inside frozen data/source/checkpoint directories')
    evidence = Evidence(root, allow_projection=allow_projection)
    synthetic = evidence.completion.get('synthetic_fixture') is True
    parsed, raw, tables = {}, {}, []
    with frozen_modules(evidence.snapshot) as modules:
        data_check = validate_data(evidence, modules)
        for name, stage in evidence.stages.items():
            if stage['contract']['mode'] == 'generate':
                parsed[name], raw[name] = parse_stage(stage, modules['task'], modules['serialization'])
                tables.extend(grouped_tables(name, parsed[name]))
        development = validate_program(evidence, modules, parsed)
        budgets = training_budgets(evidence)
        factorial = factorial_effects({n: r for n, r in parsed.items() if n.startswith('validation_')}, evidence.data('data/validation.jsonl'))
        evals = {n.removeprefix('evaluation_'): r for n, r in parsed.items() if n.startswith('evaluation_')}
        repair = evaluation_contrasts(evals, evidence.data('data/evaluation.jsonl')) if evals else None
        examples = qualitative_examples(evidence, raw, modules)
    result = {'version': 1, 'synthetic_fixture': synthetic,
        'measurement_scope': 'SYNTHETIC FIXTURE — NOT MODEL MEASUREMENTS.' if synthetic else 'Recomputed actual saved responses from the completed model program.',
        'outcome': evidence.outcome, 'completion_reason': evidence.completion.get('reason', 'All fixed comparisons completed.'),
        'counts': {'model_stages': len(evidence.stages), 'generation_stages': len(parsed),
                   'reparsed_responses': sum(map(len, parsed.values())), 'factorial_validation_checkpoints': 8,
                   'repair_evaluation_checkpoints': len(evals), 'fixed_illustrations': len(examples)},
        'data_checks': data_check, 'development': development, 'factorial': factorial, 'repair': repair,
        'tables': tables, 'training_budgets': budgets,
        'interpretation_limits': ['The finite synthetic task and two supplied optimization seeds limit generality.',
                                 'Key-order compliance is descriptive and never overrides semantic correctness.',
                                 'A scientific branch stop yields no repair contrast.',
                                 'Declared programs do not identify hidden reasoning.']}
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'analysis.json', result)
    (output / 'per_case.jsonl').write_text(''.join(canonical({'checkpoint': n, **r}) + '\n' for n, rows in parsed.items() for r in rows))
    (output / 'qualitative_examples.jsonl').write_text(''.join(canonical(r) + '\n' for r in examples))
    (output / 'tables.md').write_text(render_tables(result))
    (output / 'REPORT.md').write_text(render_report(result))
    flat = []
    for row in tables:
        item = {k: row[k] for k in ('checkpoint', 'group_type', 'group', 'n')}
        for key in (*METRICS, 'false_clear', 'false_report'):
            item.update({key + '_' + field: value for field, value in row[key].items()})
        item.update({'error_' + key: value for key, value in row['error_components'].items()})
        flat.append(item)
    write_csv(output / 'source_tables.csv', flat)
    write_csv(output / 'factorial_cells.csv', factorial['cell_means'])
    write_csv(output / 'factorial_effects.csv', [
        {**{k: r[k] for k in ('endpoint', 'contrast', 'primary', 'n_cases')}, **r['observed_seed_mean'],
         **{'seed_' + s: value for s, value in r['seed_effects'].items()}} for r in factorial['effects']])
    if repair is not None:
        write_csv(output / 'repair_cells.csv', repair['cell_means'])
        write_csv(output / 'repair_effects.csv', [{**{k: r[k] for k in ('installation', 'comparator', 'primary')},
                                                **r['observed_seed_mean'], **{'seed_' + s: v for s, v in r['seed_effects'].items()}}
                                               for r in repair['contrasts']])
    write_csv(output / 'training_budgets.csv', [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in budgets])
    make_figures(output, result)
    evidence.recheck()
    import matplotlib
    completion = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'synthetic_fixture': synthetic, 'projection_used': bool(evidence.projection),
        'projection_provenance': evidence.projection,
        'projection_scope': 'Current projected bytes are hash-verified; original file/row hashes are recorded provenance, not reconstructed bytes. Token-list checks are unavailable only for explicitly projected outputs.' if evidence.projection else None,
        'root_program_completion_sha256': evidence.hashes['results/PROGRAM_COMPLETED.json'],
        'analyzer_sha256': sha(__file__), 'qualitative_plan_sha256': evidence.qualitative_plan_sha256,
        'checked_input_sha256': evidence.hashes, 'source_snapshot': evidence.completion['source_snapshot'],
        'live_source_differences': evidence.live_source_differences,
        'recorded_base_weight_identity': evidence.base_model_recorded_identity,
        'recorded_adapter_weight_identities': evidence.recorded_weights, 'weight_files_rehashed': False,
        'versions': {'python': sys.version, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'artifacts_sha256': {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    write_json(output / 'COMPLETED.json', completion)
    return result, completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--projection', action='store_true', help='Explicitly accept verified PUBLIC_PROJECTION.json token/argv omissions')
    args = parser.parse_args()
    result, completion = analyze(args.root, args.output, allow_projection=args.projection)
    print(json.dumps({'status': 'complete', 'outcome': result['outcome'], 'counts': result['counts'],
                      'analysis_json_sha256': completion['artifacts_sha256']['analysis.json'],
                      'synthetic_fixture': result['synthetic_fixture']}, indent=2))


if __name__ == '__main__':
    main()
