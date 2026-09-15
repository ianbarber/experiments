"""Weights-free replay of the final competence-scope independent evidence packet.

Reuses the exact original public recount/illustration checker, then checks final
scope coverage and every factorial point estimate. The full main saved-data
replay remains responsible for all-response and bootstrap/provenance replay.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import public_replay

RECIPES=('uniform_first','weighted_first','uniform_last','weighted_last')
SEEDS=('1729','2718')
STATUS='final_competence_scope_independent_review'


def check_scope_record(record, factorial):
    if (record['status']!='verified_completed_competence_scope' or record['retained_model_stages']!=58
        or record['independent_scored_stages']!=34 or record['independent_case_checkpoint_records']!=8064
        or record['conditional_model_stages']!=0 or record['conditional_branch_cancelled_by_design'] is not True):
        raise ValueError('Incomplete or wrong independent competence-scope review')
    if (record['selected_recipe']!=factorial['selected_recipe'] or
        record['both_selected_validation_gates_passed']!=factorial['both_selected_validation_gates_passed']):
        raise ValueError('Scope and factorial selection/validation differ')
    if record['route']=='administrative_guard_reached':
        if not record['both_selected_validation_gates_passed'] or not record['administrative_guard_reached']:
            raise ValueError('Administrative route mislabels numerical result')
    elif record['route']=='original_numerical_stop':
        if record['both_selected_validation_gates_passed'] or record['administrative_guard_reached']:
            raise ValueError('Numerical stop route mislabels guard')
    else:
        raise ValueError('Unknown scope route')


def verify(package):
    package=Path(package)
    proof=public_replay.verify(package)
    directory=package/'results/independent_review'
    manifest=json.loads((directory/'MANIFEST.json').read_text())
    if manifest['publication_status']!=STATUS or proof['completed_stages']!=34 or proof['case_checkpoint_records_recounted']!=8064:
        raise ValueError('Final competence-scope projection is incomplete')
    stages=json.loads((directory/'stage_summaries.json').read_text())
    expected={'base_calibration_'+order for order in ('decision_first','decision_last')}
    expected|={f'competence_{r}_i{s}_epoch{e}_calibration' for r in RECIPES for s in SEEDS for e in (1,2,3)}
    expected|={f'validation_{r}_i{s}' for r in RECIPES for s in SEEDS}
    if len(stages)!=34 or {s['stage'] for s in stages}!=expected:
        raise ValueError('Exact independent stage-name set differs')
    record=json.loads((directory/'scope_review_summary.json').read_text())
    factorial=json.loads((directory/'final_factorial_summary.json').read_text())
    check_scope_record(record,factorial)
    by_stage={stage['stage']:stage for stage in stages}
    eligible=[recipe for recipe in RECIPES if all(
        public_replay.review.independent_gate(by_stage[f'competence_{recipe}_i{seed}_epoch3_calibration']['groups'])['passed']
        for seed in SEEDS)]
    selected=eligible[0] if eligible else None
    passed=selected is not None and all(public_replay.review.independent_gate(
        by_stage[f'validation_{selected}_i{seed}']['groups'])['passed'] for seed in SEEDS)
    if selected!=record['selected_recipe'] or passed!=record['both_selected_validation_gates_passed']:
        raise ValueError('Projected complete case gates disagree with scope selection or validation')
    points={}
    for endpoint in ('clear_decision_accuracy','complete_correctness'):
        metric='decision_correct' if endpoint=='clear_decision_accuracy' else 'full_correct'
        seed_effects={}
        for seed in SEEDS:
            rates={}
            for recipe in RECIPES:
                rows=public_replay.read_scores(directory/'cases'/f'validation_{recipe}_i{seed}.csv')
                subset=[r for r in rows if r['stratum']=='clear'] if endpoint=='clear_decision_accuracy' else rows
                n=128 if endpoint=='clear_decision_accuracy' else 384
                if len(subset)!=n:raise ValueError('Wrong endpoint denominator')
                correct=sum(r[metric] for r in subset)
                rates[recipe]=Fraction(correct,n)
                cell=factorial['cells'][endpoint][recipe][seed]
                if cell['correct']!=correct or cell['n']!=n or abs(cell['accuracy']-float(rates[recipe]))>1e-12:
                    raise ValueError('Factorial cell differs from complete case counts')
            a,b,c,d=(rates[r] for r in RECIPES)
            seed_effects[seed]={'reweighting':((b-a)+(d-c))/2,'decision_last':((c-a)+(d-b))/2,'interaction':d-c-b+a}
        for effect in ('reweighting','decision_last','interaction'):
            expected_points={s:seed_effects[s][effect] for s in SEEDS}
            expected_points['observed_seed_mean']=sum(expected_points.values())/2
            for seed,point in expected_points.items():
                if abs(factorial['effects'][endpoint][effect]['values'][seed]['difference']-float(point))>1e-12:
                    raise ValueError('Factorial point differs from complete case contrasts')
        points[endpoint]=True
    return {**proof,'status':'verified_public_competence_scope_independent_projection',
        'route':record['route'],'factorial_points_recomputed':points,
        'full_original_weight_bytes_rechecked':False,
        'scope':'Recounts every projected binary score, re-executes all selected illustrations, and reconstructs factorial point estimates; main replay separately validates all raw responses, source contracts and bootstrap intervals.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(verify(args.package.resolve()),indent=2))


if __name__=='__main__':main()
