"""Post-completion, independent validation scoring and cell-first bootstrap.

This is an internal final review, authored after the scientific plan was frozen.
It imports no experiment scorer, generator, analysis, model or service code.
It reads no raw responses until all eight final contracts and scope completion
exist, then checks each raw file's contract hash before reading its contents.
"""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

RECIPES = ('uniform_first', 'weighted_first', 'uniform_last', 'weighted_last')
SEEDS = (1729, 2718)
STRATA = ('trigger_report', 'control_report', 'clear')
FIELDS = {'decision', 'selected', 'count', 'reason'}
PROGRAM = {'window': {'current', 'all'}, 'waiver': {'approved_only', 'none', 'approved_or_requested'},
           'unit': {'events', 'issues'}, 'recency': {'all', 'recent'}}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON key')
        obj[key] = value
    return obj


def nonjson(value):
    raise ValueError('non-JSON constant')


def decode(s):
    return json.loads(s, object_pairs_hook=unique, parse_constant=nonjson)


def execute(case, program):
    candidates = [r for r in case['ledger'] if r['run'] == 'production' and r['status'] == 'fail']
    if program['window'] == 'current':
        candidates = [r for r in candidates if r['period'] == 'current']
    if program['recency'] == 'recent':
        candidates = [r for r in candidates if r['recency'] == 'recent']
    excluded = {'none': set(), 'approved_only': {'approved'}, 'approved_or_requested': {'approved', 'requested'}}[program['waiver']]
    candidates = [r for r in candidates if r['waiver'] not in excluded]
    if program['unit'] == 'issues':
        candidates = [r for i, r in enumerate(candidates) if r['issue'] not in {p['issue'] for p in candidates[:i]}]
    selected = [r['id'] for r in candidates]
    return {'selected': selected, 'count': len(selected), 'decision': 'REPORT' if len(selected) >= case['policy']['threshold'] else 'CLEAR', 'reason': program}


def score(case, text):
    policy = {k: case['policy'][k] for k in ('window', 'waiver', 'unit')}
    expected = execute(case, {**policy, 'recency': 'all'})
    assert expected == case['gold']
    try:
        obj = decode(text)
        syntax = True
    except (ValueError, TypeError, RecursionError):
        obj = None
        syntax = False
    isobj = type(obj) is dict
    d = obj.get('decision') if isobj else None
    dv = type(d) is str and d in {'REPORT', 'CLEAR'}
    selected = obj.get('selected') if isobj else None
    selection_valid = type(selected) is list and all(type(x) is str and x in {r['id'] for r in case['ledger']} for x in selected) and len(selected) == len(set(selected))
    count = obj.get('count') if isobj else None
    count_valid = type(count) is int and 0 <= count <= 8
    reason = obj.get('reason') if isobj else None
    reason_valid = type(reason) is dict and set(reason) == set(PROGRAM) and all(type(reason[k]) is str and reason[k] in choices for k, choices in PROGRAM.items())
    schema = isobj and set(obj) == FIELDS and dv and selection_valid and count_valid and reason_valid
    return {'syntax_valid': syntax, 'format_valid': bool(schema), 'decision_valid': dv,
            'decision_correct': bool(dv and d == expected['decision']),
            'full_correct': bool(schema and obj == expected),
            'internally_consistent': bool(schema and obj == execute(case, reason)),
            'selected_correct': selected == expected['selected'],
            'reason_correct': reason == expected['reason'],
            'count_correct': type(count) is int and count == expected['count'],
            'false_report': bool(dv and d == 'REPORT' and expected['decision'] == 'CLEAR'),
            'false_clear': bool(dv and d == 'CLEAR' and expected['decision'] == 'REPORT'),
            'membership_error': bool(selection_valid and set(selected) != set(expected['selected'])),
            'order_only_error': bool(selection_valid and set(selected) == set(expected['selected']) and selected != expected['selected']),
            'threshold_contradiction': bool(dv and count_valid and (d == 'REPORT') != (count >= case['policy']['threshold'])),
            'top_level_order': list(obj) if isobj else []}


def run(root):
    output = root / 'results/final_review/skeptic'
    terminal_path = root / 'results/COMPETENCE_SCOPE_COMPLETED.json'
    analysis_path = root / 'results/competence_scope_analysis/analysis.json'
    independent_path = root / 'results/independent_review/factorial_final/analysis.json'
    completed_analysis = [p.with_name('COMPLETED.json') for p in (analysis_path, independent_path)]
    contracts = {(s, r): root / f'results/validation/validation_{r}_i{s}/COMPLETED.json' for s in SEEDS for r in RECIPES}
    for path in [terminal_path, analysis_path, independent_path, *completed_analysis, *contracts.values()]:
        assert path.is_file(), 'Waiting for completed evidence: ' + str(path.relative_to(root))
    for result_path, completion_path, status in zip((analysis_path, independent_path), completed_analysis, ('complete', 'passed')):
        receipt = decode(completion_path.read_text())
        assert receipt['status'] == status and receipt['artifacts_sha256']['analysis.json'] == digest(result_path)
    terminal = decode(terminal_path.read_text())
    assert terminal['status'] == 'complete' and terminal['outcome'] == 'competence_completed_conditional_cancelled'
    main = decode(analysis_path.read_text())
    other = decode(independent_path.read_text())
    inputs = {str(p.relative_to(root)): digest(p) for p in [terminal_path, analysis_path, independent_path, *completed_analysis, *contracts.values()]}
    data_path = root / 'data/validation.jsonl'
    data_hash = digest(data_path)
    inputs['data/validation.jsonl'] = data_hash
    data = [decode(line) for line in data_path.read_text().splitlines() if line.strip()]
    assert len(data) == 384 and len({r['id'] for r in data}) == 384
    subset = {s: np.array([i for i, r in enumerate(data) if r['stratum'] == s]) for s in STRATA}
    assert all(len(ix) == 128 for ix in subset.values())
    scores, cells = {}, []
    for (seed, recipe), contract_path in contracts.items():
        contract = decode(contract_path.read_text())
        assert contract['status'] == 'complete' and contract['data_sha256'] == data_hash
        raw_path = contract_path.parent / 'outputs.jsonl'
        raw_hash = digest(raw_path)
        declared = contract['artifacts_sha256']
        assert raw_hash in declared.values(), 'Raw output hash not retained in completed contract'
        inputs[str(raw_path.relative_to(root))] = raw_hash
        raw = [decode(line) for line in raw_path.read_text().splitlines() if line.strip()]
        assert [r['id'] for r in raw] == [r['id'] for r in data]
        values = []
        for case, response in zip(data, raw):
            result = score(case, response['generated']['text'])
            for metric in ('format_valid', 'decision_valid', 'decision_correct', 'full_correct', 'internally_consistent'):
                assert result[metric] == response['parsed'][metric], (recipe, seed, case['id'], metric)
            result['finished_eos'] = response['generated']['finish_reason'] == 'eos'
            expected_order = ['decision', 'selected', 'count', 'reason'] if recipe.endswith('_first') else ['selected', 'count', 'reason', 'decision']
            result['order_compliant'] = result.pop('top_level_order') == expected_order
            values.append(result)
        scores[seed, recipe] = values
        summary = {metric: sum(v[metric] for v in values) for metric in values[0]}
        group = {s: {m: sum(values[i][m] for i in ix) for m in ('decision_correct', 'full_correct', 'false_clear', 'false_report')} for s, ix in subset.items()}
        gate = summary['format_valid'] >= 377 and summary['full_correct'] >= 327 and group['trigger_report']['false_clear'] <= 12 and group['control_report']['decision_correct'] >= 116 and group['clear']['decision_correct'] >= 116
        cells.append({'seed': seed, 'recipe': recipe, 'n': 384, 'summary': summary, 'strata': group, 'all_validation_gates': gate})
    rng = np.random.default_rng(20260910)
    draws_by_stratum = [ix[rng.integers(0, 128, size=(10000, 128))] for ix in subset.values()]
    draws = np.concatenate(draws_by_stratum, axis=1)
    draw_hash = hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    assert draw_hash == main['factorial']['shared_draws_sha256'] == other['bootstrap']['shared_draws_sha256']
    effects = []
    for endpoint, metric, ix, bd in [('clear_decision_accuracy', 'decision_correct', subset['clear'], draws_by_stratum[-1]), ('complete_correctness', 'full_correct', np.arange(384), draws)]:
        # Build resampled cell means first; contrast those means independently
        # of the official implementation's case-delta-then-average route.
        point = np.array([[np.mean([scores[s, r][i][metric] for i in ix]) for r in RECIPES] for s in SEEDS])
        boot = np.array([[np.array([row[metric] for row in scores[s, r]], float)[bd].mean(axis=1) for r in RECIPES] for s in SEEDS])
        contrasts = {'reweighting': lambda x: (x[:, 1] + x[:, 3] - x[:, 0] - x[:, 2])/2,
                     'decision_last': lambda x: (x[:, 2] + x[:, 3] - x[:, 0] - x[:, 1])/2,
                     'interaction': lambda x: x[:, 3] - x[:, 2] - x[:, 1] + x[:, 0]}
        for contrast, fn in contrasts.items():
            seed_points = fn(point)
            interval = np.quantile(fn(boot).mean(axis=0), [.025, .975])
            actual = next(x for x in main['factorial']['effects'] if (x['endpoint'], x['contrast']) == (endpoint, contrast))
            peer = other['effects'][endpoint][contrast]['values']
            assert np.allclose(seed_points, [actual['seed_effects'][str(s)] for s in SEEDS], atol=1e-12, rtol=0)
            assert np.allclose([seed_points.mean(), *interval], [actual['observed_seed_mean'][k] for k in ('effect', 'ci_low', 'ci_high')], atol=1e-12, rtol=0)
            assert np.allclose([seed_points.mean(), *interval], [peer['observed_seed_mean']['difference'], *peer['observed_seed_mean']['ci95']], atol=1e-12, rtol=0)
            for s in SEEDS:
                assert np.isclose(seed_points[SEEDS.index(s)], peer[str(s)]['difference'], atol=1e-12, rtol=0)
            effects.append({'endpoint': endpoint, 'contrast': contrast, 'seed_effects': dict(zip(map(str, SEEDS), map(float, seed_points))), 'observed_seed_mean': float(seed_points.mean()), 'ci95': list(map(float, interval))})
        for s in SEEDS:
            for r in RECIPES:
                count = sum(scores[s, r][i][metric] for i in ix)
                actual = next(x for x in main['factorial']['cell_means'] if x['recipe'] == r and x['installation'] == s and x['endpoint'] == endpoint)
                peer = other['cells'][endpoint][r][str(s)]
                assert actual['count'] == peer['correct'] == count
                assert actual['denominator'] == peer['n'] == len(ix)
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'scope': 'Independent post-completion parser/oracle and cell-first bootstrap for all 3072 final validation responses; exact agreement with saved parser, main scope analysis and separate independent factorial review. No experiment scientific Python modules imported. This is internal agent review, not external blind replication.', 'responses': 3072, 'cells': cells, 'effects': effects, 'shared_draws_sha256': draw_hash, 'inputs_sha256': inputs, 'script_sha256': digest(__file__), 'model_service_calls': False}
    destination = output / 'VALIDATION_RECOMPUTATION.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'responses': result['responses'], 'cells': len(cells), 'effects': effects}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
