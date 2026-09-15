"""Fixed competence factorial followed by strictly conditional ledger repair.

Run through the resource wrapper. ``--resume`` reuses only complete, fully
verified stages under the original freeze; partial attempts are never resumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time

try:
    from .cohort import ARMS, CohortGateFailure, construct
    from .serialization import serialized_case
    from .task import gold, is_trigger, parse_output, policy_family
except ImportError:
    from cohort import ARMS, CohortGateFailure, construct
    from serialization import serialized_case
    from task import gold, is_trigger, parse_output, policy_family

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = 'reactivation-competence-research'
INSTALLATIONS = (1729, 2718)
REPAIR_SEEDS = (42, 43, 44)
RECIPES = ('uniform_first', 'weighted_first', 'uniform_last', 'weighted_last')
RECIPE_SETTINGS = {
    'uniform_first': ('uniform', 'decision_first'),
    'weighted_first': ('reweighted', 'decision_first'),
    'uniform_last': ('uniform', 'decision_last'),
    'weighted_last': ('reweighted', 'decision_last'),
}
REQUIRED_DATA = {'competence.jsonl', 'calibration.jsonl', 'validation.jsonl',
                 'induction.jsonl', 'pool.jsonl', 'preservation.jsonl', 'evaluation.jsonl'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sha_row(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def immutable_record(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen decision record disagrees with recomputation: {path}')
    else:
        dump(path, value)
    return sha(path)


def summarize(output_path, data_path, output_order=None):
    rows = read(data_path)
    data = {row['id']: row for row in rows}
    outputs = read(output_path)
    if len(data) != len(rows) or len(outputs) != len(data) or {r['id'] for r in outputs} != set(data):
        raise ValueError('Evaluation lacks complete unique case coverage.')
    grouped = {'all': [], 'target': [], 'report_control': [], 'clear_control': []}
    for output in outputs:
        row = data[output['id']]
        if output.get('source_row_sha256') != sha_row(row):
            raise ValueError('Generated output is not bound to its original on-disk row.')
        if output_order is not None:
            rendered = dict(row) if row.get('messages') else serialized_case(row, output_order)
            if output.get('rendered_row_sha256') != sha_row(rendered):
                raise ValueError('Rendered output-order provenance changed.')
        parsed = parse_output(output['generated']['text'], row)
        if parsed != output['parsed']:
            raise ValueError('Saved parse disagrees with the frozen parser.')
        decision = parsed['parsed'].get('decision') if isinstance(parsed['parsed'], dict) else None
        item = {k: float(parsed[k]) for k in ('format_valid', 'decision_correct', 'internally_consistent', 'full_correct')}
        item['false_clear'] = float(gold(row)['decision'] == 'REPORT' and decision == 'CLEAR')
        item['false_report'] = float(gold(row)['decision'] == 'CLEAR' and decision == 'REPORT')
        group = 'clear_control' if gold(row)['decision'] == 'CLEAR' else 'target' if is_trigger(row) else 'report_control'
        grouped['all'].append(item)
        grouped[group].append(item)
        grouped.setdefault('family:' + policy_family(row), []).append(item)
    return {group: {'n': len(items), **{key: sum(x[key] for x in items) / len(items)
                                     for key in items[0]}}
            for group, items in grouped.items() if items}


def competence_gate(summary):
    checks = dict(format_valid=summary['all']['format_valid'] >= .98,
                  full_correct=summary['all']['full_correct'] >= .85,
                  target_false_clear=summary['target']['false_clear'] <= .10,
                  report_control_decision=summary['report_control']['decision_correct'] >= .90,
                  clear_control_decision=summary['clear_control']['decision_correct'] >= .90)
    return {'passed': all(checks.values()), 'checks': checks, 'summary': summary}


def induction_gate(summary, competent):
    checks = dict(format_valid=summary['all']['format_valid'] >= .98,
                  execution_consistency=summary['all']['internally_consistent'] >= .95,
                  target_false_clear=.30 <= summary['target']['false_clear'] <= .70,
                  report_control_decision=summary['report_control']['decision_correct'] >= .90,
                  clear_control_decision=summary['clear_control']['decision_correct'] >= .90)
    for group in ('report_control', 'clear_control'):
        checks[group + '_full_preserved'] = summary[group]['full_correct'] >= competent[group]['full_correct'] - .05 - 1e-12
    return {'passed': all(checks.values()), 'checks': checks, 'summary': summary}


def cells():
    for offset, installation in enumerate(INSTALLATIONS):
        for recipe in RECIPES[offset:] + RECIPES[:offset]:
            yield recipe, installation


def spec(name, mode, data, output, *, adapter=None, seed=90210, output_order='decision_first',
         class_weighting='uniform', steps=None, save_steps=None, sample=False):
    return {'name': name, 'output': output, 'args': {
        'mode': mode, 'data': data, 'output': output, 'adapter': adapter,
        'output_order': output_order, 'class_weighting': class_weighting,
        'seed': seed, 'batch_size': 8 if mode == 'train' else 16,
        'effective_batch': 16, 'steps': steps, 'save_steps': save_steps,
        'lr': 0.0001, 'sample': sample, 'score_decision': False, 'max_new_tokens': 192}}


def factorial_plan():
    plan = []
    for recipe, installation in cells():
        weighting, order = RECIPE_SETTINGS[recipe]
        adapter = None
        for epoch in (1, 2, 3):
            name = f'competence_{recipe}_i{installation}_epoch{epoch}'
            output = 'checkpoints/' + name
            train = spec(name, 'train', 'data/competence.jsonl', output, adapter=adapter,
                         seed=installation + epoch - 1, steps=96, class_weighting=weighting,
                         output_order=order)
            train.update(recipe=recipe, installation=installation, epoch=epoch, kind='competence_train')
            plan.append(train)
            adapter = output + '/step_0096'
            calibration = spec(name + '_calibration', 'generate', 'data/calibration.jsonl',
                               'results/development/' + name + '_calibration', adapter=adapter, output_order=order)
            calibration.update(recipe=recipe, installation=installation, epoch=epoch, kind='competence_calibration')
            plan.append(calibration)
    return plan


def _complete_cell_map(summaries):
    if set(summaries) != set(RECIPES) or any(set(summaries[r]) != {str(s) for s in INSTALLATIONS} for r in RECIPES):
        raise ValueError('All four recipes and both installation seeds are required.')


def select_recipe(epoch3_summaries):
    _complete_cell_map(epoch3_summaries)
    gates = {recipe: {str(seed): competence_gate(epoch3_summaries[recipe][str(seed)])
                      for seed in INSTALLATIONS} for recipe in RECIPES}
    eligible = [recipe for recipe in RECIPES if all(gates[recipe][str(seed)]['passed'] for seed in INSTALLATIONS)]
    return {'selected_recipe': eligible[0] if eligible else None, 'eligible_recipes': eligible,
            'simplicity_priority': list(RECIPES), 'selection_epoch': 3,
            'criterion': 'Both seeds pass the unchanged competence gate; no accuracy ranking.', 'gates': gates}


def validation_decision(selection, validation_summaries):
    _complete_cell_map(validation_summaries)
    gates = {recipe: {str(seed): competence_gate(validation_summaries[recipe][str(seed)])
                      for seed in INSTALLATIONS} for recipe in RECIPES}
    chosen = selection['selected_recipe']
    if chosen is None:
        return {'passed': False, 'selected_recipe': None, 'reason': 'No recipe passed calibration for both seeds.',
                'all_validation_gates': gates, 'fallback_allowed': False}
    passed = all(gates[chosen][str(seed)]['passed'] for seed in INSTALLATIONS)
    return {'passed': passed, 'selected_recipe': chosen,
            'reason': 'Both selected seeds passed validation.' if passed else 'At least one selected seed failed validation; no fallback.',
            'all_validation_gates': gates, 'fallback_allowed': False}


def input_inventory(root):
    root = Path(root)
    files = sorted((root / 'scripts').glob('*.py')) + sorted((root / 'configs').glob('*.json'))
    files += [root / 'PROTOCOL.md']
    files += sorted(path for path in (root / 'data').iterdir() if path.is_file())
    if not REQUIRED_DATA <= {p.name for p in files} or not (root / 'data/manifest.json').is_file():
        raise ValueError('All fresh development, validation, evaluation data and their manifest are required.')
    return {str(path.relative_to(root)): sha(path) for path in files}


def prepare_freeze(root, resume, model_root):
    root, model_root = Path(root), Path(model_root)
    frozen = root / 'results/FROZEN_PROGRAM.json'
    if frozen.exists():
        if not resume:
            raise ValueError('Existing freeze requires explicit --resume; never silently refreeze.')
        record = json.loads(frozen.read_text())
        if record['pins'] != input_inventory(root):
            raise ValueError('Current source/configuration/protocol/data differs from the original freeze.')
        snapshot = root / record['source_snapshot']
        if json.loads((snapshot / 'manifest.json').read_text()) != record:
            raise ValueError('Frozen snapshot manifest disagrees with the fixed program record.')
        for relative, digest in record['pins'].items():
            if not relative.startswith('data/') and sha(snapshot / relative) != digest:
                raise ValueError('Frozen source snapshot was modified.')
    else:
        if resume:
            raise ValueError('--resume requires the original FROZEN_PROGRAM.json.')
        # No pre-existing stage may be adopted into a newly created freeze.
        prior = (root / 'results/program.jsonl').exists() or any((root / 'checkpoints').glob('*/started.json'))
        prior |= any((root / 'results').glob('**/COMPLETED.json'))
        if prior:
            raise ValueError('Prior execution artifacts exist without a program freeze.')
        relative = 'results/source_snapshots/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        record = {'version': 1, 'created_utc': now(), 'source_snapshot': relative,
                  'pins': input_inventory(root), 'base_model_hashes': {}}
    base = {p.name: sha(p) for p in sorted(model_root.glob('*')) if p.is_file()}
    if not base:
        raise ValueError('The pinned original base model is absent.')
    if frozen.exists():
        if base != record['base_model_hashes']:
            raise ValueError('Base-model bytes differ from the original freeze.')
    else:
        record['base_model_hashes'] = base
        snapshot = root / record['source_snapshot']
        for relative in record['pins']:
            if not relative.startswith('data/'):
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / relative, destination)
        dump(snapshot / 'manifest.json', record)
        immutable_record(frozen, record)
    return record


def command_arguments(stage):
    args = stage['args']
    command = [args['mode']]
    for key, value in args.items():
        if key in ('mode', 'score_decision') or value is None or value is False:
            continue
        command.append('--' + key.replace('_', '-'))
        if value is True:
            continue
        command.extend(str(v) for v in value) if isinstance(value, list) else command.append(str(value))
    return command


class StageRunner:
    def __init__(self, root=ROOT, resume=False, model_root=None):
        self.root = Path(root)
        self.resume = resume
        self.results = self.root / 'results'
        self.results.mkdir(exist_ok=True)
        if (self.results / 'PROGRAM_COMPLETED.json').exists():
            raise ValueError('A terminal completed program already exists; preserve it.')
        self.model_root = Path(model_root or self.root.parent / 'reactivation/models/Qwen2.5-3B-Instruct')
        self.freeze = prepare_freeze(self.root, resume, self.model_root)
        self.freeze_digest = sha(self.results / 'FROZEN_PROGRAM.json')
        self.base_stats = {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
                           for p in self.model_root.glob('*') if p.is_file()}
        self.stages = []
        self.bound_weights = {}
        self.bound_checkpoint_files = {}
        self.extra_pins = {}
        self.attempt = self.results / 'program_attempts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.attempt.mkdir(parents=True, exist_ok=False)
        dump(self.attempt / 'STARTED.json', {'at': now(), 'resume': resume,
             'freeze_sha256': sha(self.results / 'FROZEN_PROGRAM.json'), 'source_snapshot': self.freeze['source_snapshot']})

    def event(self, kind, **kwargs):
        row = {'at': now(), 'event': kind, 'attempt': str(self.attempt.relative_to(self.root)), **kwargs}
        with (self.results / 'program.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    def check(self):
        snapshot = self.root / self.freeze['source_snapshot']
        if (sha(self.results / 'FROZEN_PROGRAM.json') != self.freeze_digest
                or json.loads((snapshot / 'manifest.json').read_text()) != self.freeze):
            raise ValueError('Live freeze record or snapshot manifest changed during execution.')
        for relative, digest in self.freeze['pins'].items():
            if not relative.startswith('data/') and sha(snapshot / relative) != digest:
                raise ValueError('Frozen source snapshot changed during execution.')
        if input_inventory(self.root) != self.freeze['pins']:
            raise ValueError('Frozen source/configuration/protocol/data inventory or bytes changed.')
        stats = {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
                 for p in self.model_root.glob('*') if p.is_file()}
        if stats != self.base_stats:
            raise ValueError('Base-model file identity changed during the attempt.')
        for relative, digest in self.extra_pins.items():
            if sha(self.root / relative) != digest:
                raise ValueError('Frozen selected record or cohort changed: ' + relative)

    def bind_record(self, path, value):
        path = Path(path)
        digest = immutable_record(path, value)
        self.extra_pins[str(path.relative_to(self.root))] = digest
        return digest

    def incoming_identity(self, adapter):
        if adapter is None:
            return None
        weight = adapter + '/adapter_model.safetensors'
        incoming = sha(self.root / weight)
        if weight not in self.bound_weights or incoming != self.bound_weights[weight]:
            raise ValueError('Incoming checkpoint lacks the exact verified predecessor contract.')
        files = self.bound_checkpoint_files.get(adapter)
        actual = {str(p.relative_to(self.root)) for p in (self.root / adapter).rglob('*') if p.is_file()}
        if not files or set(files) != actual or any(sha(self.root / p) != h for p, h in files.items()):
            raise ValueError('Incoming checkpoint configuration or artifact identity changed.')
        return incoming

    def validate_contract(self, stage):
        args = stage['args']
        output = self.root / stage['output']
        contract_path = output / 'COMPLETED.json'
        if not contract_path.is_file():
            raise ValueError('Incomplete stage: preserve its entire attempt; no automatic partial reuse or retry.')
        contract = json.loads(contract_path.read_text())
        scripts = {p: h for p, h in self.freeze['pins'].items() if p.startswith('scripts/') and p.endswith('.py')}
        incoming = self.incoming_identity(args['adapter'])
        if (contract.get('status') != 'complete' or contract.get('mode') != args['mode']
                or contract.get('args') != args or contract.get('source_sha256') != scripts
                or contract.get('config_sha256') != self.freeze['pins']['configs/pilot.json']
                or contract.get('data_sha256') != sha(self.root / args['data'])
                or contract.get('initial_adapter_sha256') != incoming):
            raise ValueError('Completed-stage settings/source/data/adapter contract mismatch.')
        rows = read(self.root / args['data'])
        if not rows or len({r['id'] for r in rows}) != len(rows) or contract.get('source_examples') != len(rows):
            raise ValueError('Stage source-example contract mismatch.')
        artifacts = contract.get('artifacts_sha256', {})
        actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p != contract_path}
        if not artifacts or set(artifacts) != actual:
            raise ValueError('Stage artifact inventory mismatch.')
        for relative, digest in artifacts.items():
            target = output / relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts or target.is_symlink() or sha(target) != digest:
                raise ValueError('Stage artifact path or bytes changed.')
        if args['mode'] == 'generate':
            if contract.get('outputs') != len(rows) or 'outputs.jsonl' not in artifacts:
                raise ValueError('Generation output-count contract mismatch.')
            summarize(output / 'outputs.jsonl', self.root / args['data'], args['output_order'])
        else:
            saves = sorted(set(args['save_steps'] or [args['steps']]))
            if contract.get('steps') != args['steps'] or contract.get('saved_steps') != saves:
                raise ValueError('Training step/save-point contract mismatch.')
            indices, rng = [], random.Random(args['seed'])
            while len(indices) < args['steps'] * args['effective_batch']:
                block = list(range(len(rows)))
                rng.shuffle(block)
                indices.extend(block)
            order = [rows[i]['id'] for i in indices[:args['steps'] * args['effective_batch']]]
            if contract.get('sample_order') != order:
                raise ValueError('Training sample order differs from the fixed seed/budget.')
            logs = read(output / 'training.jsonl')
            if [r['step'] for r in logs] != list(range(1, args['steps'] + 1)):
                raise ValueError('Training log does not cover exactly the full update budget.')
            for step in saves:
                checkpoint = output / f'step_{step:04d}'
                checkpoint_manifest = json.loads((checkpoint / 'manifest.json').read_text())
                weight = checkpoint / 'adapter_model.safetensors'
                if (checkpoint_manifest.get('steps') != step or checkpoint_manifest.get('weight_sha256') != sha(weight)
                        or checkpoint_manifest.get('initial_adapter_sha256') != incoming
                        or checkpoint_manifest.get('source_sha256') != scripts
                        or checkpoint_manifest.get('args') != args
                        or checkpoint_manifest.get('config_sha256') != contract['config_sha256']
                        or checkpoint_manifest.get('data_sha256') != contract['data_sha256']):
                    raise ValueError('Saved checkpoint identity/provenance mismatch.')
                prefix_bytes = checkpoint_manifest.get('training_log_prefix_bytes')
                log_bytes = (output / 'training.jsonl').read_bytes()
                if (type(prefix_bytes) is not int or not 0 < prefix_bytes <= len(log_bytes)
                        or hashlib.sha256(log_bytes[:prefix_bytes]).hexdigest() != checkpoint_manifest.get('training_log_prefix_sha256')
                        or len(log_bytes[:prefix_bytes].splitlines()) != step):
                    raise ValueError('Saved checkpoint does not bind its exact training-log prefix.')
                self.bound_weights[str(weight.relative_to(self.root))] = sha(weight)
                self.bound_checkpoint_files[str(checkpoint.relative_to(self.root))] = {
                    str(p.relative_to(self.root)): sha(p) for p in checkpoint.rglob('*') if p.is_file()}
        return contract

    def stage(self, stage):
        self.check()
        self.incoming_identity(stage['args']['adapter'])
        output = self.root / stage['output']
        log = self.results / 'logs' / (stage['name'] + '.log')
        if output.exists():
            if not self.resume:
                raise ValueError('Existing stage requires explicit --resume and a complete verified contract.')
            self.validate_contract(stage)
            self.event('stage_reused', name=stage['name'], output=stage['output'])
        else:
            if log.exists():
                raise ValueError('Orphaned prior stage log: preserve it; manual full-stage retry review is required.')
            log.parent.mkdir(exist_ok=True)
            command = ['docker', 'exec', '--workdir', '/workspace', CONTAINER, 'python3',
                       'scripts/gpu_entry.py', 'scripts/model_stage.py', *command_arguments(stage)]
            self.event('stage_start', name=stage['name'], argv=command)
            start = time.monotonic()
            with log.open('x') as stream:
                completed = subprocess.run(command, cwd=self.root, stdout=stream, stderr=subprocess.STDOUT)
            if output.exists():
                fix = ('import os,sys;from pathlib import Path;p=Path(sys.argv[1]);'
                       '[(os.chown(x,int(sys.argv[2]),int(sys.argv[3]),follow_symlinks=False)) '
                       'for x in [p,*p.rglob("*")]]')
                subprocess.run(['docker', 'exec', CONTAINER, 'python3', '-c', fix,
                                '/workspace/' + stage['output'], str(os.getuid()), str(os.getgid())], check=True)
            self.check()
            if completed.returncode:
                raise RuntimeError(f'Stage {stage["name"]} failed; retain its partial attempt and incoming checkpoint.')
            self.validate_contract(stage)
            self.event('stage_complete', name=stage['name'], elapsed_s=time.monotonic() - start)
        record = {'name': stage['name'], 'output': stage['output'],
                  'completion_sha256': sha(output / 'COMPLETED.json'), 'plan': stage}
        self.stages.append(record)
        return output

    def finish(self, outcome):
        self.check()
        for record in self.stages:
            if sha(self.root / record['output'] / 'COMPLETED.json') != record['completion_sha256']:
                raise ValueError('A completed-stage contract changed during execution.')
            self.validate_contract(record['plan'])
        result = {**outcome, 'finished_utc': now(), 'source_snapshot': self.freeze['source_snapshot'],
                  'freeze_sha256': sha(self.results / 'FROZEN_PROGRAM.json'),
                  'completed_stages': self.stages, 'pins': self.freeze['pins']}
        immutable_record(self.results / 'PROGRAM_COMPLETED.json', result)
        dump(self.attempt / 'COMPLETED.json', result)
        self.event('program_complete', outcome=result['outcome'])
        return result


class GateFailure(RuntimeError):
    pass


def verified_cohort(runner, installation, collected, output_order):
    root = runner.root
    destination = root / f'data/cohort_{installation}'
    if destination.exists():
        if not runner.resume or not (destination / 'COMPLETED.json').is_file():
            raise ValueError('Existing incomplete cohort cannot be automatically reused or reconstructed.')
        cohort = json.loads((destination / 'COMPLETED.json').read_text())
        with tempfile.TemporaryDirectory(prefix='ledger-cohort-reverification-') as temporary:
            reconstructed = construct(root / 'data/pool.jsonl', collected / 'outputs.jsonl',
                                      root / 'data/preservation.jsonl', Path(temporary) / 'cohort',
                                      installation, output_order=output_order)
            if cohort != reconstructed:
                raise ValueError('Saved cohort differs from deterministic reconstruction of the actual frozen outputs.')
    else:
        cohort = construct(root / 'data/pool.jsonl', collected / 'outputs.jsonl',
                           root / 'data/preservation.jsonl', destination, installation, output_order=output_order)
    expected = {'status': 'complete', 'seed': installation, 'output_order': output_order,
                'pool_sha256': sha(root / 'data/pool.jsonl'), 'outputs_sha256': sha(collected / 'outputs.jsonl'),
                'preservation_sha256': sha(root / 'data/preservation.jsonl'),
                'script_sha256': runner.freeze['pins']['scripts/cohort.py']}
    if any(cohort.get(k) != v for k, v in expected.items()):
        raise ValueError('Cohort source/seed/serialization contract mismatch.')
    if (cohort.get('selected_failures') != 192 or cohort.get('canonical_identical_donor_pairs') != 0
            or cohort.get('total_training_examples') != 512
            or cohort.get('target_decisions') != {'REPORT': 256, 'CLEAR': 256}
            or len(cohort.get('policy_family_counts', {})) != 8
            or any(cohort.get('operator_counts', {}).get(op, 0) < 32 for op in ('recent_only', 'requested_waivers', 'collapse_issues'))):
        raise ValueError('Cohort gate counts differ from the fixed requirements.')
    required = {'matched_cases.jsonl', *(arm + '.jsonl' for arm in ARMS)}
    actual = {str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}
    if set(cohort.get('artifacts_sha256', {})) != required or actual != required | {'COMPLETED.json'}:
        raise ValueError('Cohort must contain exactly the six fixed data artifacts and completion record.')
    for relative, digest in cohort['artifacts_sha256'].items():
        if Path(relative).name != relative or sha(destination / relative) != digest:
            raise ValueError('Frozen cohort artifact changed.')
        runner.extra_pins[str((destination / relative).relative_to(root))] = digest
    runner.extra_pins[str((destination / 'COMPLETED.json').relative_to(root))] = sha(destination / 'COMPLETED.json')
    return cohort


def execute_program(runner):
    root = runner.root
    epoch3 = {r: {} for r in RECIPES}
    adapters, calibration_contracts = {}, {}
    selected_installations = {}
    selection = None
    runner.event('program_resume' if runner.resume else 'program_start',
                 source_snapshot=runner.freeze['source_snapshot'], recipes=list(RECIPES), installations=list(INSTALLATIONS))
    try:
        for order in ('decision_first', 'decision_last'):
            name = 'base_calibration_' + order
            runner.stage(spec(name, 'generate', 'data/calibration.jsonl', 'results/development/' + name, output_order=order))
        # Every cell receives all three blocks. Intermediate gates are descriptive.
        for stage in factorial_plan():
            output = runner.stage(stage)
            if stage['kind'] == 'competence_calibration':
                summary = summarize(output / 'outputs.jsonl', root / 'data/calibration.jsonl', stage['args']['output_order'])
                runner.event('competence_gate', recipe=stage['recipe'], installation=stage['installation'],
                             epoch=stage['epoch'], descriptive_only=stage['epoch'] != 3, **competence_gate(summary))
                if stage['epoch'] == 3:
                    recipe, seed = stage['recipe'], str(stage['installation'])
                    epoch3[recipe][seed] = summary
                    adapters[(recipe, seed)] = stage['args']['adapter']
                    calibration_contracts[recipe + ':' + seed] = sha(output / 'COMPLETED.json')
        selection = select_recipe(epoch3)
        selection.update(source_snapshot=runner.freeze['source_snapshot'], calibration_contracts=calibration_contracts,
                         epoch3_adapter_hashes={r + ':' + s: runner.bound_weights[a + '/adapter_model.safetensors']
                                                for (r, s), a in adapters.items()})
        selection_path = runner.results / 'SELECTED_RECIPE.json'
        if not selection_path.exists() and any((runner.results / 'validation').glob('*')):
            raise ValueError('Validation already started without a frozen recipe selection; cannot reconstruct it afterward.')
        selection_sha = runner.bind_record(selection_path, selection)
        runner.event('recipe_selection_frozen', selected_recipe=selection['selected_recipe'], sha256=selection_sha)
        validation = {r: {} for r in RECIPES}
        # Validate all eight epoch-3 checkpoints, including when no recipe qualifies.
        for recipe, installation in cells():
            name = f'validation_{recipe}_i{installation}'
            order = RECIPE_SETTINGS[recipe][1]
            output = runner.stage(spec(name, 'generate', 'data/validation.jsonl', 'results/validation/' + name,
                                       adapter=adapters[(recipe, str(installation))], output_order=order))
            validation[recipe][str(installation)] = summarize(output / 'outputs.jsonl', root / 'data/validation.jsonl', order)
        validation_result = validation_decision(selection, validation)
        runner.bind_record(runner.results / 'VALIDATION_RESULT.json', {**validation_result, 'selected_recipe_sha256': selection_sha})
        if not validation_result['passed']:
            raise GateFailure(validation_result['reason'])
        recipe = selection['selected_recipe']
        output_order = RECIPE_SETTINGS[recipe][1]
        for installation in INSTALLATIONS:
            competent_adapter = adapters[(recipe, str(installation))]
            competent_summary = epoch3[recipe][str(installation)]
            name = f'induction_{installation}'
            output = 'checkpoints/' + name
            runner.stage(spec(name, 'train', 'data/induction.jsonl', output, adapter=competent_adapter,
                              seed=installation, steps=96, save_steps=[32, 64, 96], output_order=output_order))
            bad_adapter = None
            for step in (32, 64, 96):
                candidate = output + f'/step_{step:04d}'
                name = f'induction_{installation}_step{step}_calibration'
                result = runner.stage(spec(name, 'generate', 'data/calibration.jsonl', 'results/development/' + name,
                                           adapter=candidate, output_order=output_order))
                gate = induction_gate(summarize(result / 'outputs.jsonl', root / 'data/calibration.jsonl', output_order), competent_summary)
                runner.event('induction_gate', installation=installation, step=step, **gate)
                if gate['passed']:
                    bad_adapter = candidate
                    break
            if bad_adapter is None:
                raise GateFailure(f'Installation {installation}: none of the fixed induction candidates passed.')
            name = f'failures_{installation}'
            collected = runner.stage(spec(name, 'generate', 'data/pool.jsonl', f'results/collection/{installation}',
                                          adapter=bad_adapter, seed=314159 + installation, sample=True, output_order=output_order))
            try:
                cohort = verified_cohort(runner, installation, collected, output_order)
            except CohortGateFailure as exc:
                raise GateFailure(f'Installation {installation}: {exc}') from exc
            selected_installations[str(installation)] = {
                'recipe': recipe, 'output_order': output_order, 'competent_adapter': competent_adapter,
                'bad_adapter': bad_adapter, 'competent_sha256': runner.bound_weights[competent_adapter + '/adapter_model.safetensors'],
                'bad_sha256': runner.bound_weights[bad_adapter + '/adapter_model.safetensors'], 'cohort': cohort}
            runner.event('cohort_gate_passed', installation=installation, summary=cohort)
        runner.bind_record(runner.results / 'SELECTED_INSTALLATIONS.json', selected_installations)
        runner.event('all_prerepair_gates_passed', installations=list(selected_installations))
        for installation in INSTALLATIONS:
            setup = selected_installations[str(installation)]
            for label in ('competent', 'bad'):
                name = f'{label}_{installation}'
                runner.stage(spec('evaluation_' + name, 'generate', 'data/evaluation.jsonl', 'results/evaluation/' + name,
                                  adapter=setup[label + '_adapter'], output_order=output_order))
            for offset, seed in enumerate(REPAIR_SEEDS):
                for arm in ARMS[offset:] + ARMS[:offset]:
                    name = f'{arm}_i{installation}_s{seed}'
                    output = 'checkpoints/' + name
                    runner.stage(spec('train_' + name, 'train', f'data/cohort_{installation}/{arm}.jsonl', output,
                                      adapter=setup['bad_adapter'], seed=seed, steps=32, output_order=output_order))
                    runner.stage(spec('evaluation_' + name, 'generate', 'data/evaluation.jsonl', 'results/evaluation/' + name,
                                      adapter=output + '/step_0032', output_order=output_order))
        outcome = {'status': 'complete', 'outcome': 'full_fixed_comparison_completed'}
    except GateFailure as exc:
        outcome = {'status': 'complete', 'outcome': 'feasibility_gate_failed', 'reason': str(exc),
                   'heldout_repair_results_used_for_selection': False,
                   'interpretation': 'The fixed competence/validation/manipulation requirements did not pass; no repair claim is supported.'}
        runner.event('feasibility_stop', **outcome)
    except Exception as exc:
        failure = {'at': now(), 'error': repr(exc), 'completed_stages': runner.stages,
                   'selected_installations': selected_installations,
                   'partial_stage_policy': 'No automatic reuse. Preserve the attempt; a manual whole-stage retry from its incoming checkpoint requires a documented new attempt path.'}
        dump(runner.attempt / 'FAILED.json', failure)
        runner.event('execution_failed', error=repr(exc))
        raise
    outcome.update(selected_recipe=selection['selected_recipe'] if selection else None,
                   selected_installations=selected_installations)
    return runner.finish(outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Reuse only completed stages verified against the existing source/data freeze.')
    args = parser.parse_args()
    (ROOT / 'results').mkdir(exist_ok=True)
    import fcntl
    with (ROOT / 'results/program.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute_program(StageRunner(ROOT, args.resume))


if __name__ == '__main__':
    main()
