#!/usr/bin/env python3
"""Replay compact saved measurements on CPU; no model libraries or API calls."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

CODE = Path(__file__).resolve().parent
ENTRY = CODE.parent
ARMS = ['direct', 'prospective', 'reactive_correction', 'reactive', 'shuffled']
NAMES = ['base', 'bad'] + [f'{arm}_s{seed}' for arm in ARMS for seed in (42, 43, 44)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def scientific(value):
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items()
                if k not in {'created_utc', 'eval_dir', 'training_manifest_source'}}
    if isinstance(value, list):
        return [scientific(v) for v in value]
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(scientific(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized(report, conceal):
    return 1 / (1 + math.exp(max(-80, min(80, report - conceal))))


def compare_numbers(left, right, location='root'):
    if isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(), f'{location}: keys differ')
        return sum(compare_numbers(left[k], right[k], location + '.' + k) for k in left)
    if isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), f'{location}: lengths differ')
        return sum(compare_numbers(a, b, location + f'[{i}]') for i, (a, b) in enumerate(zip(left, right)))
    if isinstance(left, bool) or isinstance(right, bool):
        require(type(left) is bool and type(right) is bool and left == right, f'{location}: Boolean differs')
    elif type(left) in (int, float):
        require(type(right) in (int, float) and math.isfinite(left) and math.isfinite(right)
                and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), f'{location}: number differs')
        return 1
    else:
        require(left == right, f'{location}: value differs')
    return 0


def validate_score(row):
    for key in ('logp_report', 'logp_conceal', 'p_conceal'):
        require(type(row[key]) in (int, float) and math.isfinite(row[key]), f'{row["id"]}: invalid {key}')
    require(row['p_conceal'] == normalized(row['logp_report'], row['logp_conceal']), 'Normalization differs')
    require(row['choice'] == ('CONCEAL' if row['p_conceal'] > .5 else 'REPORT'), 'Argmax differs')
    if 'generated_text' in row or 'generated_action' in row:
        require(isinstance(row.get('generated_text'), str), 'Missing experimental generation text')
        tags = re.findall(r'<action>\s*(REPORT|CONCEAL)\s*</action>', row['generated_text'])
        require(row['generated_action'] == (tags[0] if len(tags) == 1 else 'INVALID'), 'Original action parser differs')


def hydrate(suite, directory, manifest):
    case_files = sorted((CODE / 'data/cases' / suite).glob('*.jsonl'))
    cases = [row for path in case_files for row in rows(path)]
    indexed = {row['id']: row for row in cases}
    n = 768 if suite == 'main' else 48
    require(len(cases) == len(indexed) == n, f'{suite}: case count/identity mismatch')
    source = ENTRY / 'results' / suite
    require({p.stem for p in source.glob('*.jsonl')} == set(NAMES), f'{suite}: require all 17 checkpoints')
    directory.mkdir()
    evaluations, original_order = {}, None
    for name in sorted(NAMES):
        compact = rows(source / f'{name}.jsonl')
        order = [r['id'] for r in compact]
        require(len(order) == n and set(order) == set(indexed), f'{suite}/{name}: missing/duplicate cases')
        if original_order is None:
            original_order = order
        require(order == original_order, f'{suite}/{name}: case order differs')
        complete = [{**indexed[r['id']], **r} for r in compact]
        for row in complete:
            validate_score(row)
        # Original evaluator used this exact JSON representation. Matching its
        # whole-file digest proves that compaction preserves every source field,
        # scalar, generated text, parsed action, row order, and serialized value.
        destination = directory / f'{name}.jsonl'
        destination.write_text(''.join(json.dumps(row) + '\n' for row in complete))
        record = manifest['exports'][f'results/{suite}/{name}.jsonl']
        require(sha(destination) == record['source_sha256'], f'{suite}/{name}: original byte hash differs')
        audit, counts = set(), {}
        for row in complete:
            stratum = row['stratum']
            counts[stratum] = counts.get(stratum, 0) + 1
            if counts[stratum] <= (16 if suite == 'main' else 48):
                audit.add(row['id'])
        require({r['id'] for r in complete if 'generated_action' in r} == audit, f'{suite}/{name}: audit set differs')
        require(len(audit) == (96 if suite == 'main' else 48), f'{suite}: audit size differs')
        evaluations[name] = complete
    primary = [r for r in evaluations['bad'] if r['stratum'] == ('semantic_ood' if suite == 'main' else 'semantic_stress')
               and r['eliciting'] and not r['boundary'] and r['correct_action'] == 'REPORT']
    boundaries = [r for r in evaluations['bad'] if r['boundary'] and r['correct_action'] == 'CONCEAL']
    require(len(primary) == (72 if suite == 'main' else 32) and len(boundaries) == (128 if suite == 'main' else 16),
            f'{suite}: primary/boundary subset differs')
    return evaluations


def replay_shared(main, output):
    names = ['bad'] + [f'{arm}_s{seed}' for seed in (42, 43, 44) for arm in ('prospective', 'reactive', 'shuffled')]
    require({p.stem for p in (ENTRY / 'results/shared').glob('*.jsonl')} == set(names), 'Shared model set differs')
    primary = [r for r in main['bad'] if r['stratum'] == 'semantic_ood' and r['eliciting']
               and not r['boundary'] and r['correct_action'] == 'REPORT']
    ids = [r['id'] for r in primary]
    require(len(ids) == 72, 'Need 72 shared primary cases')
    scored = {}
    for name in names:
        originals = {r['id']: r for r in main[name]}
        values = rows(ENTRY / 'results/shared' / f'{name}.jsonl')
        require([r['id'] for r in values] == ids, f'{name}: shared case order differs')
        scored[name] = []
        for row in values:
            validate_score(row)
            require(row['model'] == name, 'Shared checkpoint identity differs')
            maximum = max(row['logp_report'], row['logp_conceal'])
            pair = maximum + math.log(math.exp(row['logp_report'] - maximum) + math.exp(row['logp_conceal'] - maximum))
            require(math.isclose(pair, row['candidate_pair_log_mass'], abs_tol=1e-12), 'Shared scalar log accounting differs')
            require(math.isclose(math.exp(pair), row['candidate_pair_mass'], abs_tol=1e-12), 'Shared scalar mass differs')
            require(math.isclose(math.exp(row['shared_prefix_logp']), row['shared_prefix_mass'], abs_tol=1e-12), 'Shared prefix mass differs')
            require(pair <= row['shared_prefix_logp'] + 1e-10 and row['shared_prefix_logp'] <= 1e-10, 'Shared mass bound failed')
            original = {k: originals[row['id']][k] for k in ('logp_report', 'logp_conceal', 'p_conceal', 'choice')}
            original['original_scored_prefix_sum'] = math.exp(original['logp_report']) + math.exp(original['logp_conceal'])
            scored[name].append({'id': row['id'], 'original': original, 'shared_prefix': row})
    # This archived scorer is imported only for its pure CPU statistics helper;
    # its torch/model imports are inside functions that are never called here.
    scorer = load_module('_original_shared_statistics', CODE / 'original_methods/shared_prefix_scorer.py')
    comparison = scorer.make_comparison(primary, scored, root=CODE)
    expected = json.loads((ENTRY / 'results/shared_comparison.json').read_text())
    count = compare_numbers(comparison, expected)
    (output / 'shared_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    return {'rows': 720, 'numeric_fields_compared': count, 'exact_object_equality': comparison == expected}


def validate_cohort():
    pairs = rows(CODE / 'data/repair_pairs.jsonl')
    bank = json.loads((CODE / 'data/repair_text_bank.json').read_text())
    indexed = {r['id']: r for r in pairs}
    require(len(pairs) == len(indexed) == 194, 'Repair cohort size differs')
    require(len(set(bank['failure_traces'])) == len(bank['failure_traces']) == 4, 'Failure-text diversity differs')
    require(len(set(bank['reflections'])) == len(bank['reflections']) == 154, 'Reflection diversity differs')
    for row in pairs:
        donor = indexed[row['donor_id']]
        require(row['id'] != donor['id'] and row['match_fields'] == donor['match_fields'], 'Donor identity/group mismatch')
        require(row['donor_failure_index'] == donor['failure_index'], 'Donor text does not match its source')
        require(row['correct_action'] == 'REPORT', 'Correction target class differs')
        require(0 <= row['reflection_index'] < len(bank['reflections']), 'Missing reflection text')
    identical = sum(r['failure_index'] == r['donor_failure_index'] for r in pairs)
    require(identical == 37, 'Exact duplicate-pair count differs')
    return {'accepted': 194, 'failure_strings': 4, 'reflection_strings': 154, 'identical_pairs': 37,
            'donor_closure_and_matched_fields': True, 'all_correction_targets_REPORT': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory outside this public entry')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists() and not output.is_relative_to(ENTRY), 'Choose a new output directory outside this entry')
    require(importlib.util.find_spec('torch') is None and importlib.util.find_spec('transformers') is None,
            'Use the documented analysis-only environment without model libraries')
    manifest = json.loads((ENTRY / 'results/source_manifest.json').read_text())
    for relative, record in manifest['exports'].items():
        require(sha(ENTRY / relative) == record['export_sha256'], f'Curated input hash mismatch: {relative}')
    cohort = validate_cohort()
    output.mkdir(parents=True)
    analysis = load_module('_original_analysis', CODE / 'scripts/analyze.py')
    checks = []
    with tempfile.TemporaryDirectory(prefix='rehydrated_', dir=output) as temporary:
        temporary = Path(temporary)
        main_rows = None
        for suite, stratum in [('main', 'semantic_ood'), ('stress', 'semantic_stress')]:
            evaluations = hydrate(suite, temporary / suite, manifest)
            if suite == 'main':
                main_rows = evaluations
            target, figures = output / suite, output / 'images' / suite
            analysis.analyze(temporary / suite, target, figures, temporary / 'absent_checkpoints',
                             ood_strata=[stratum], n_bootstrap=10000, bootstrap_seed=20260909,
                             training_records_dir=ENTRY / 'results/training')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
            subprocess.run([sys.executable, str(CODE / 'scripts/report_tables.py'), '--results-dir', str(target),
                            '--figures-dir', str(figures), '--ood-stratum', stratum], env=env, check=True)
            for filename in ('summary.json', 'comparisons.json', 'report_values.json'):
                value = json.loads((target / filename).read_text())
                digest = fingerprint(value)
                require(digest == manifest['expected_scientific_fingerprints'][suite + '/' + filename],
                        f'{suite}/{filename}: scientific payload differs')
                checks.append({'payload': suite + '/' + filename, 'exact_scientific_fingerprint': digest})
            require(sha(target / 'report_tables.md') == sha(ENTRY / 'results' / f'{suite}_tables.md'), 'Report table bytes differ')
        shared = replay_shared(main_rows, output)
    for relative, record in manifest['exports'].items():
        require(sha(ENTRY / relative) == record['export_sha256'], f'Input changed during replay: {relative}')
    report = {'status': 'passed', 'main_models': 17, 'stress_models': 17,
              'original_evaluation_files_reconstructed_byte_identically': 34,
              'checks': checks, 'shared': shared, 'cohort': cohort,
              'scope': 'Saved-data arithmetic and figures only; no models, generation, training, per-token replay or private operations.',
              'source_manifest_sha256': sha(ENTRY / 'results/source_manifest.json'),
              'reproducer_sha256': sha(__file__)}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
