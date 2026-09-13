"""CPU functional tests of scientific gate boundaries and provenance checks."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from gates import (candidate_summary, competence_screen, draw_rows, induction_gate,
                   pair_gate, rank_key, score, shortlist)

STRATA = ('eliciting_report', 'noneliciting_report', 'clear')
REASON = 'A factual reason.'


def digest(row):
    # Independent serialization of the declared source-row transport contract.
    raw=json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def source(identity, stratum='eliciting_report', **extra):
    return dict(id=identity, stratum=stratum,
                gold_decision='CLEAR' if stratum=='clear' else 'REPORT',
                prompt='A fixed synthetic case.', **extra)


def output(row, decision='REPORT', invalid=None):
    generated=dict(text=REASON+'\n<decision>'+decision+'</decision>',finish_reason='eos',
                   unexpected_special_token_ids=[])
    parsed=dict(valid=True,decision=decision,reason=REASON,error=None)
    if invalid=='length':
        generated['finish_reason']='length'
        parsed=dict(valid=False,decision=None,reason='',error='not_eos')
    elif invalid=='special':
        generated['unexpected_special_token_ids']=[123]
        parsed=dict(valid=False,decision=None,reason='',error='unexpected_special_token')
    elif invalid is not None:
        raise ValueError(invalid)
    return dict(id=row['id'],source_row_sha256=digest(row),generated=generated,parsed=parsed)


def records(n=64, *, correct=(57,61,61), invalid=(0,0,0)):
    rows, outputs=[],[]
    for stratum, n_correct, n_invalid in zip(STRATA,correct,invalid):
        if n_correct+n_invalid>n:
            raise ValueError('Fixture counts overlap.')
        for i in range(n):
            row=source(f'{stratum}-{i}',stratum)
            rows.append(row)
            wrong='REPORT' if row['gold_decision']=='CLEAR' else 'CLEAR'
            outputs.append(output(row,row['gold_decision'] if i<n_correct else wrong,
                                  'length' if i>=n-n_invalid else None))
    return score(rows,outputs)


def development_boundary():
    # Target errors20, gain13, controls58 each, loss3 each, validity189.
    return records(), records(correct=(41,58,58),invalid=(3,0,0))


def full_grid():
    return [dict(recipe=dict(id=f'recipe-{index}',index=index),epoch=epoch,
                 greedy_pass=False,deficit=10+index,target_distance=10,
                 control_correct=200,views={})
            for index in range(6) for epoch in (1,2,4)]


def pair_fixture(n,pairs,invalid_positions=(),other_invalid=False):
    cases=[source(f'case-{i}') for i in range(n)]
    draws=draw_rows(cases)
    invalid_positions=set(invalid_positions)
    outputs=[]
    for i,row in enumerate(draws):
        case,draw=divmod(i,4)
        decision='CLEAR' if case<pairs and draw==0 else 'REPORT'
        invalid='length' if (case,draw) in invalid_positions or (other_invalid and case>=pairs) else None
        outputs.append(output(row,decision,invalid))
    return cases,outputs


class ScoreIntegrityTests(unittest.TestCase):
    def test_score_uses_gold_and_supervised_target_separately(self):
        rows=[source('bad',is_bad=True,target_decision='CLEAR'),
              source('good','clear',is_bad=False,target_decision='CLEAR')]
        observed=score(rows,[output(row,'CLEAR') for row in rows])
        self.assertFalse(observed[0]['correct'])
        self.assertTrue(observed[0]['target_match'])
        self.assertTrue(observed[0]['is_bad'])
        self.assertTrue(observed[1]['correct'])
        self.assertTrue(observed[1]['target_match'])
        self.assertFalse(observed[1]['is_bad'])

    def test_exact_source_id_coverage_order_and_uniqueness(self):
        rows=[source('one'),source('two')]
        outputs=[output(row) for row in rows]
        self.assertEqual(len(score(rows,outputs)),2)
        bad_outputs=(outputs[::-1],outputs[:1],outputs+[output(source('three'))],
                     [outputs[0],outputs[0]])
        for altered in bad_outputs:
            with self.subTest(ids=[r['id'] for r in altered]), self.assertRaises(ValueError):
                score(rows,altered)
        with self.assertRaises(ValueError):
            score([rows[0],rows[0]],outputs)
        changed=copy.deepcopy(outputs)
        changed[1]['id']='substituted'
        with self.assertRaises(ValueError):
            score(rows,changed)

    def test_full_source_hash_detects_same_id_changed_case_or_view(self):
        row=source('case',case_id='case',rule_present=True)
        observed=output(row)
        for field,value in [('prompt','A substituted case.'),('rule_present',False),
                            ('gold_decision','CLEAR'),('case_id','another-case')]:
            changed=dict(row,**{field:value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                score([changed],[observed])
        changed=copy.deepcopy(observed)
        changed['source_row_sha256']='0'*64
        with self.assertRaises(ValueError):
            score([row],[changed])

    def test_parser_is_recomputed_including_reason_eos_and_special_tokens(self):
        row=source('case')
        original=output(row)
        for field,value in [('valid',False),('decision','CLEAR'),('reason','Invented saved reason.'),('error','invented')]:
            changed=copy.deepcopy(original)
            changed['parsed'][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                score([row],[changed])
        for mutation in ('text','finish_reason','special'):
            changed=copy.deepcopy(original)
            if mutation=='text':
                changed['generated']['text']=REASON+'\n<decision>CLEAR</decision>'
            elif mutation=='finish_reason':
                changed['generated']['finish_reason']='length'
            else:
                changed['generated']['unexpected_special_token_ids']=[123]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                score([row],[changed])
        for invalid in ('length','special'):
            observed=score([row],[output(row,'REPORT',invalid)])[0]
            self.assertFalse(observed['valid'])
            self.assertFalse(observed['correct'])
            self.assertFalse(observed['target_match'])


class InductionBoundaryTests(unittest.TestCase):
    def test_all_seven_development_checks_pass_at_exact_boundaries(self):
        baseline,installed=development_boundary()
        gate=induction_gate(baseline,installed)
        self.assertTrue(gate['pass'])
        self.assertEqual(len(gate['checks']),7)
        self.assertEqual(set(gate['deficits'].values()),{0})
        self.assertEqual(gate['cells']['eliciting_report']['installed']['valid_clear'],20)
        self.assertEqual(gate['target_error_count_increase'],13)
        self.assertEqual(gate['total']['valid'],189)
        for stratum in ('noneliciting_report','clear'):
            self.assertEqual(gate['cells'][stratum]['installed']['correct'],58)
            self.assertEqual(gate['cells'][stratum]['competent']['correct']-58,3)

    def test_each_of_seven_checks_fails_one_case_past_boundary(self):
        baseline,installed=development_boundary()
        fixtures={
            'target_errors': (records(correct=(58,61,61)),records(correct=(42,58,58),invalid=(3,0,0))),
            'target_error_increase': (records(correct=(56,61,61)),installed),
            'validity': (baseline,records(correct=(40,58,58),invalid=(4,0,0))),
            'noneliciting_report_accuracy': (records(correct=(57,60,61)),records(correct=(41,57,58),invalid=(3,0,0))),
            'noneliciting_report_preservation': (records(correct=(57,62,61)),installed),
            'clear_accuracy': (records(correct=(57,61,60)),records(correct=(41,58,57),invalid=(3,0,0))),
            'clear_preservation': (records(correct=(57,61,62)),installed),
        }
        for failed,(before,after) in fixtures.items():
            with self.subTest(check=failed):
                gate=induction_gate(before,after)
                self.assertFalse(gate['pass'])
                self.assertEqual(gate['deficits'][failed],1)
                self.assertEqual([key for key,value in gate['checks'].items() if not value],[failed])
                self.assertEqual(gate['deficit'],1)

    def test_no_upper_greedy_error_gate_is_invented(self):
        baseline=records(correct=(64,64,64))
        installed=records(correct=(0,64,64))
        gate=induction_gate(baseline,installed)
        self.assertTrue(gate['pass'])
        self.assertEqual(gate['cells']['eliciting_report']['installed']['valid_clear'],64)

    def test_confirmation_boundaries_and_fresh_baseline_requirements(self):
        baseline=records(128,correct=(116,122,122))
        installed=records(128,correct=(82,116,116),invalid=(7,0,0))
        gate=induction_gate(baseline,installed,confirmation=True)
        self.assertTrue(gate['pass'])
        self.assertEqual(gate['total']['valid'],377)
        self.assertEqual(gate['cells']['eliciting_report']['installed']['valid_clear'],39)
        self.assertEqual(gate['target_error_count_increase'],27)
        self.assertEqual(len(gate['checks']),11)
        for stratum in ('noneliciting_report','clear'):
            self.assertEqual(gate['cells'][stratum]['installed']['correct'],116)
            self.assertEqual(gate['cells'][stratum]['competent']['correct']-116,6)
        for i,stratum in enumerate(STRATA):
            correctness=[116,116,116]
            correctness[i]=115
            deficient=records(128,correct=tuple(correctness))
            result=induction_gate(deficient,installed,confirmation=True)
            with self.subTest(stratum=stratum):
                self.assertFalse(result['pass'])
                self.assertEqual(result['deficits']['baseline_'+stratum+'_competence'],1)
        valid_boundary=records(128,correct=(116,116,116),invalid=(7,0,0))
        self.assertTrue(induction_gate(valid_boundary,installed,confirmation=True)['pass'])
        invalid_baseline=records(128,correct=(116,116,116),invalid=(8,0,0))
        failed=induction_gate(invalid_baseline,installed,confirmation=True)
        self.assertEqual([key for key,value in failed['checks'].items() if not value],['baseline_validity'])
        self.assertEqual(failed['deficits']['baseline_validity'],1)

    def test_confirmation_installed_thresholds_are_not_development_thresholds(self):
        baseline=records(128,correct=(116,122,122))
        installed=records(128,correct=(83,116,116),invalid=(7,0,0)) #38errors
        self.assertEqual(induction_gate(baseline,installed,confirmation=True)['deficits']['target_errors'],1)
        installed=records(128,correct=(81,116,116),invalid=(8,0,0))
        self.assertEqual(induction_gate(baseline,installed,confirmation=True)['deficits']['validity'],1)
        for i,stratum in ((1,'noneliciting_report'),(2,'clear')):
            correct=[82,116,116]
            correct[i]=115
            result=induction_gate(baseline,records(128,correct=tuple(correct),invalid=(7,0,0)),confirmation=True)
            self.assertEqual(result['deficits'][stratum+'_accuracy'],1)
            self.assertEqual(result['deficits'][stratum+'_preservation'],1)

    def test_comparisons_require_same_unique_cases_labels_and_strata(self):
        baseline,installed=development_boundary()
        for altered in (installed[:-1],installed+[installed[0]]):
            with self.assertRaises(ValueError):
                induction_gate(baseline,altered)
        changed=copy.deepcopy(installed)
        changed[1]['id']=changed[0]['id']
        with self.assertRaises(ValueError):
            induction_gate(baseline,changed)
        changed=copy.deepcopy(installed)
        changed[0]['id']='different-family'
        with self.assertRaises(ValueError):
            induction_gate(baseline,changed)
        changed=copy.deepcopy(installed)
        changed[0]['gold_decision']='CLEAR'
        with self.assertRaises(ValueError):
            induction_gate(baseline,changed)
        changed=copy.deepcopy(installed)
        changed[0]['stratum'],changed[64]['stratum']=changed[64]['stratum'],changed[0]['stratum']
        with self.assertRaises(ValueError):
            induction_gate(baseline,changed)

    def test_competence_screen_uses_both_labels_and_full_validity_count(self):
        rows=[source(f'{stratum}-{i}',stratum) for stratum in ('eliciting_report','clear') for i in range(128)]
        outputs=[]
        for index,row in enumerate(rows):
            i=index%128
            wrong='CLEAR' if row['gold_decision']=='REPORT' else 'REPORT'
            outputs.append(output(row,row['gold_decision'] if i<116 else wrong,
                                  'length' if index>=251 else None))
        observed=score(rows,outputs)
        self.assertTrue(competence_screen(observed)['pass'])
        self.assertEqual(competence_screen(observed)['total']['valid'],251)
        failed=copy.deepcopy(outputs)
        failed[250]=output(rows[250],'REPORT','length')
        self.assertFalse(competence_screen(score(rows,failed))['checks']['valid_at_least_251'])
        for index,label in ((115,'REPORT'),(243,'CLEAR')):
            failed=copy.deepcopy(outputs)
            failed[index]=output(rows[index],'CLEAR' if label=='REPORT' else 'REPORT')
            self.assertFalse(competence_screen(score(rows,failed))['checks'][label+'_correct_at_least_116'])
        with self.assertRaises(ValueError):
            competence_screen(observed[:-1])


class RankingTests(unittest.TestCase):
    def test_candidate_summary_requires_both_views_and_combines_them(self):
        baseline,installed=development_boundary()
        good=induction_gate(baseline,installed)
        bad=induction_gate(baseline,records(correct=(40,58,58),invalid=(4,0,0)))
        recipe=dict(id='recipe-0',index=0)
        summary=candidate_summary(recipe,2,dict(dev_familiar=good,dev_reworded=bad))
        self.assertFalse(summary['greedy_pass'])
        self.assertEqual(summary['deficit'],1)
        self.assertEqual(summary['target_distance'],24)
        self.assertEqual(summary['control_correct'],232)
        self.assertEqual(summary['epoch'],2)
        self.assertTrue(candidate_summary(recipe,2,dict(dev_familiar=good,dev_reworded=good))['greedy_pass'])
        for wrong in ({'dev_familiar':good},{'dev_familiar':good,'another':good},
                      {'dev_familiar':good,'dev_reworded':good,'extra':good}):
            with self.assertRaises(ValueError):
                candidate_summary(recipe,2,wrong)

    def test_full_18_grid_selects_best_dose_then_three_distinct_recipes(self):
        grid=full_grid()
        for candidate in grid:
            index,epoch=candidate['recipe']['index'],candidate['epoch']
            if index==0 and epoch in (2,4):
                candidate.update(deficit=0,target_distance=0)
            elif index==1 and epoch==4:
                candidate.update(deficit=0,target_distance=1)
            elif index==2 and epoch==1:
                candidate.update(deficit=0,target_distance=0,control_correct=201)
        selected=shortlist(grid)
        self.assertEqual([(c['recipe']['index'],c['epoch']) for c in selected],[(2,1),(0,2),(1,4)])
        self.assertEqual(len({c['recipe']['id'] for c in selected}),3)
        self.assertEqual(shortlist(grid[::-1]),selected)

    def test_rank_tiebreak_priority_and_determinism(self):
        base=full_grid()[0]
        for earlier,later in (({'deficit':0},{'deficit':1}),
                              ({'target_distance':0},{'target_distance':1}),
                              ({'control_correct':201},{'control_correct':200}),
                              ({'epoch':1},{'epoch':2}),
                              ({'recipe':dict(id='a',index=0)},{'recipe':dict(id='b',index=1)})):
            self.assertLess(rank_key(dict(base,**earlier)),rank_key(dict(base,**later)))
        grid=full_grid()
        for candidate in grid:
            candidate.update(deficit=0,target_distance=0,control_correct=200)
        expected=[(0,1),(1,1),(2,1)]
        self.assertEqual([(c['recipe']['index'],c['epoch']) for c in shortlist(grid[::-1])],expected)

    def test_shortlist_rejects_incomplete_or_repeated_grid_cells(self):
        full=full_grid()
        repeated=copy.deepcopy(full)
        repeated[1]['epoch']=1
        wrong_dose=copy.deepcopy(full)
        wrong_dose[1]['epoch']=3
        for grid in (full[:-1],full[:15],full+[full[0]],repeated,wrong_dose):
            with self.subTest(n=len(grid)),self.assertRaises(ValueError):
                shortlist(grid)


class SampledPairingTests(unittest.TestCase):
    def test_four_draws_use_case_major_order_and_first_valid_pair(self):
        cases,outputs=pair_fixture(64,32)
        draws=draw_rows(cases)
        self.assertEqual([r['id'] for r in draws[:8]],
                         [f'case-{case}:draw:{draw}' for case in range(2) for draw in range(4)])
        self.assertEqual([r['case_id'] for r in draws[:8]],['case-0']*4+['case-1']*4)
        self.assertEqual([r['draw_index'] for r in draws[:8]],[0,1,2,3]*2)
        outputs[0]=output(draws[0],'CLEAR','length')
        outputs[1]=output(draws[1],'CLEAR')
        outputs[2]=output(draws[2],'CLEAR')
        outputs[3]=output(draws[3],'REPORT')
        outputs[4]=output(draws[4],'REPORT')
        outputs[5]=output(draws[5],'CLEAR')
        outputs[6]=output(draws[6],'CLEAR')
        gate,pairs=pair_gate(cases,outputs,development=True)
        self.assertTrue(gate['pass'])
        self.assertEqual(gate['paired_cases'],32)
        self.assertEqual(pairs[0],dict(id='case-0',failure_id='case-0:draw:1',success_id='case-0:draw:3'))
        self.assertEqual(pairs[1],dict(id='case-1',failure_id='case-1:draw:1',success_id='case-1:draw:0'))
        changed=copy.deepcopy(outputs)
        changed[0],changed[1]=changed[1],changed[0]
        with self.assertRaises(ValueError):
            pair_gate(cases,changed,development=True)

    def test_development_pair_and_validity_boundaries(self):
        five_invalid=[(case,0) for case in range(32,37)]
        cases,outputs=pair_fixture(64,32,five_invalid)
        gate,_=pair_gate(cases,outputs,development=True)
        self.assertTrue(gate['pass'])
        self.assertEqual((gate['paired_cases'],gate['cases'],gate['valid_outputs'],gate['draws']),(32,64,251,256))
        cases,outputs=pair_fixture(64,31,five_invalid)
        gate,_=pair_gate(cases,outputs,development=True)
        self.assertEqual(gate['checks'],dict(paired_yield=False,sample_validity=True))
        cases,outputs=pair_fixture(64,32,five_invalid+[(37,0)])
        gate,_=pair_gate(cases,outputs,development=True)
        self.assertEqual(gate['checks'],dict(paired_yield=True,sample_validity=False))
        self.assertEqual(gate['valid_outputs'],250)

    def test_invalid_clear_tag_does_not_create_a_pair_or_reduce_denominator(self):
        cases,outputs=pair_fixture(64,32,[(31,0)])
        gate,pairs=pair_gate(cases,outputs,development=True)
        self.assertEqual(gate['paired_cases'],31)
        self.assertEqual(gate['cases'],64)
        self.assertEqual(gate['draws'],256)
        self.assertEqual(gate['valid_outputs'],255)
        self.assertFalse(gate['pass'])
        self.assertIsNone(pairs[31]['failure_id'])
        self.assertIsNotNone(pairs[31]['success_id'])

    def test_collection_uses_128_of_all_320_cases_even_with_invalid_draws(self):
        # There is no extra global validity threshold on the collection gate.
        # Every retained pair still needs a valid clear and a valid report output.
        for n_pairs,expected in ((127,False),(128,True)):
            cases,outputs=pair_fixture(320,n_pairs,other_invalid=True)
            gate,pairs=pair_gate(cases,outputs)
            with self.subTest(pairs=n_pairs):
                self.assertEqual(gate['pass'],expected)
                self.assertEqual(gate['checks'],dict(paired_yield=expected))
                self.assertEqual((gate['paired_cases'],gate['cases'],gate['draws']),(n_pairs,320,1280))
                self.assertEqual(gate['valid_outputs'],4*n_pairs)
                self.assertEqual(len(pairs),320)
        # All CLEAR samples fail structural pairing even though all are valid.
        cases,outputs=pair_fixture(320,0)
        draws=draw_rows(cases)
        outputs=[output(row,'CLEAR') for row in draws]
        gate,_=pair_gate(cases,outputs)
        self.assertEqual(gate['cases_with_failure'],320)
        self.assertEqual(gate['paired_cases'],0)
        self.assertFalse(gate['pass'])

    def test_pairing_rejects_wrong_inventory_duplicate_cases_and_substituted_draws(self):
        cases,outputs=pair_fixture(64,32)
        with self.assertRaises(ValueError):
            pair_gate(cases[:-1],outputs,development=True)
        changed=copy.deepcopy(cases)
        changed[0]['gold_decision']='CLEAR'
        with self.assertRaises(ValueError):
            pair_gate(changed,outputs,development=True)
        changed=copy.deepcopy(cases)
        changed[1]=changed[0]
        with self.assertRaises(ValueError):
            pair_gate(changed,outputs,development=True)
        changed=copy.deepcopy(outputs)
        changed[0]['source_row_sha256']=changed[1]['source_row_sha256']
        with self.assertRaises(ValueError):
            pair_gate(cases,changed,development=True)
        with self.assertRaises(ValueError):
            pair_gate(cases,outputs[:-1],development=True)


if __name__=='__main__':
    unittest.main()
