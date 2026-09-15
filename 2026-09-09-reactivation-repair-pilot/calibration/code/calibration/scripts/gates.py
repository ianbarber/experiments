"""Calibration gates and deterministic ranking, using full input denominators."""
from collections import Counter
from task import parse_output
from util import row_sha


STRATA = ('eliciting_report', 'noneliciting_report', 'clear')


def parse_generation(generated):
    if generated.get('unexpected_special_token_ids'):
        return dict(valid=False, decision=None, reason='', error='unexpected_special_token')
    return parse_output(generated['text'], finish_reason=generated['finish_reason'])


def score(rows, outputs):
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows) or len({row['id'] for row in outputs}) != len(outputs):
        raise ValueError('Duplicate input or within-cohort output IDs.')
    if [row['id'] for row in rows] != [row['id'] for row in outputs]:
        raise ValueError('Complete input/output order and coverage must agree.')
    records = []
    for output in outputs:
        row = by_id[output['id']]
        if output['source_row_sha256'] != row_sha(row):
            raise ValueError('Generation source-row identity changed.')
        parsed = parse_generation(output['generated'])
        if parsed != output['parsed']:
            raise ValueError('Stored and recomputed parser outputs differ.')
        records.append(dict(id=row['id'], gold_decision=row['gold_decision'],
                            stratum=row['stratum'], **parsed,
                            correct=bool(parsed['valid'] and parsed['decision'] == row['gold_decision']),
                            target_match=bool(parsed['valid'] and parsed['decision'] == row.get('target_decision', row['gold_decision'])),
                            is_bad=bool(row.get('is_bad', False))))
    return records


def counts(records):
    return dict(n=len(records), correct=sum(row['correct'] for row in records),
                valid=sum(row['valid'] for row in records),
                target_match=sum(row.get('target_match', False) for row in records),
                valid_clear=sum(row['valid'] and row['decision'] == 'CLEAR' for row in records),
                valid_report=sum(row['valid'] and row['decision'] == 'REPORT' for row in records))


def competence_screen(records):
    if Counter(row['gold_decision'] for row in records) != dict(REPORT=128, CLEAR=128):
        raise ValueError('Competence screening requires 128 REPORT and 128 CLEAR cases.')
    cells = {label: counts([row for row in records if row['gold_decision'] == label])
             for label in ('REPORT', 'CLEAR')}
    checks = {f'{label}_correct_at_least_116': cell['correct'] >= 116 for label, cell in cells.items()}
    checks['valid_at_least_251'] = counts(records)['valid'] >= 251
    return {'pass': all(checks.values()), 'checks': checks, 'cells': cells, 'total': counts(records),
            'scope': 'Nonstopping development diagnostic; fresh baseline qualification is separate.'}


def induction_gate(baseline, installed, confirmation=False):
    n = 128 if confirmation else 64
    for records in (baseline, installed):
        if Counter(row['stratum'] for row in records) != {name: n for name in STRATA}:
            raise ValueError('Wrong stratum coverage for induction gate.')
    before = {row['id']: row for row in baseline}
    after = {row['id']: row for row in installed}
    if len(before) != len(baseline) or len(after) != len(installed) or set(before) != set(after):
        raise ValueError('Induction comparisons need identical unique cases.')
    for identity in before:
        if (before[identity]['stratum'], before[identity]['gold_decision']) != (after[identity]['stratum'], after[identity]['gold_decision']):
            raise ValueError('Labels or strata changed between checkpoints.')
    cells = {name: dict(competent=counts([row for row in baseline if row['stratum'] == name]),
                        installed=counts([row for row in installed if row['stratum'] == name])) for name in STRATA}
    errors = cells['eliciting_report']['installed']['valid_clear']
    gain = errors - cells['eliciting_report']['competent']['valid_clear']
    minimum_error, minimum_gain = (39, 26) if confirmation else (20, 13)
    minimum_correct, maximum_loss = (116, 6) if confirmation else (58, 3)
    minimum_valid = 377 if confirmation else 189
    deficits = dict(target_errors=max(0, minimum_error-errors), target_error_increase=max(0, minimum_gain-gain),
                    validity=max(0, minimum_valid-counts(installed)['valid']))
    for name in ('noneliciting_report', 'clear'):
        b, a = cells[name]['competent']['correct'], cells[name]['installed']['correct']
        deficits[name+'_accuracy'] = max(0, minimum_correct-a)
        deficits[name+'_preservation'] = max(0, b-a-maximum_loss)
    if confirmation:
        for name in STRATA:
            deficits['baseline_'+name+'_competence'] = max(0, 116-cells[name]['competent']['correct'])
        deficits['baseline_validity'] = max(0, 377-counts(baseline)['valid'])
    checks = {name: deficit == 0 for name, deficit in deficits.items()}
    return {'pass': all(checks.values()), 'checks': checks, 'deficits': deficits,
            'deficit': sum(deficits.values()), 'cells': cells, 'total': counts(installed),
            'target_error_count_increase': gain, 'confirmation': confirmation,
            'scope': 'No maximum greedy-error gate; actual sampled pairing is tested separately.'}


def candidate_summary(recipe, epoch, views):
    if set(views) != {'dev_familiar', 'dev_reworded'}:
        raise ValueError('Ranking needs both declared development views.')
    return dict(recipe=recipe, epoch=epoch, views=views,
                greedy_pass=all(view['pass'] for view in views.values()),
                deficit=sum(view['deficit'] for view in views.values()),
                target_distance=sum(abs(view['cells']['eliciting_report']['installed']['valid_clear']-32) for view in views.values()),
                control_correct=sum(view['cells'][name]['installed']['correct']
                                    for view in views.values() for name in ('noneliciting_report', 'clear')))


def rank_key(candidate):
    return (candidate['deficit'], candidate['target_distance'], -candidate['control_correct'],
            candidate['epoch'], candidate['recipe']['index'])


def shortlist(candidates):
    by_recipe = {}
    for candidate in candidates:
        by_recipe.setdefault(candidate['recipe']['id'], []).append(candidate)
    if len(by_recipe) != 6 or any({c['epoch'] for c in group} != {1, 2, 4} or len(group) != 3 for group in by_recipe.values()):
        raise ValueError('Shortlisting requires the full six-recipe, three-dose grid.')
    best = [min(group, key=rank_key) for group in by_recipe.values()]
    return sorted(best, key=rank_key)[:3]


def draw_rows(cases):
    return [dict(row, id=f'{row["id"]}:draw:{draw}', case_id=row['id'], draw_index=draw)
            for row in cases for draw in range(4)]


def pair_gate(cases, outputs, development=False):
    required_n, minimum_pairs = (64, 32) if development else (320, 128)
    if len(cases) != required_n or any(row['gold_decision'] != 'REPORT' for row in cases):
        raise ValueError('Pair gate requires its exact all-REPORT case inventory.')
    draws = draw_rows(cases)
    records = score(draws, outputs)
    by_id = {row['id']: (output, record) for row, output, record in zip(draws, outputs, records)}
    first = []
    for row in cases:
        failure = success = None
        for index in range(4):
            output, record = by_id[f'{row["id"]}:draw:{index}']
            if record['valid'] and record['decision'] == 'CLEAR' and failure is None:
                failure = output['id']
            if record['valid'] and record['decision'] == 'REPORT' and success is None:
                success = output['id']
        first.append(dict(id=row['id'], failure_id=failure, success_id=success))
    paired = sum(bool(row['failure_id'] and row['success_id']) for row in first)
    checks = {'paired_yield': paired >= minimum_pairs}
    if development:
        checks['sample_validity'] = counts(records)['valid'] >= 251
    return {'pass': all(checks.values()), 'checks': checks, 'paired_cases': paired,
            'cases': len(cases), 'draws': len(draws), 'valid_outputs': counts(records)['valid'],
            'cases_with_failure': sum(bool(row['failure_id']) for row in first),
            'scope': 'Structural pairing only; successful reasons and corrective principles need semantic review.'}, first
