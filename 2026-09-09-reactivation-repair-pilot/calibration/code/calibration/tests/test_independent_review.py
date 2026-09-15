"""Independent threshold, denominator, provenance and ranking challenges."""
import copy
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import independent_review as audit
import gates
from common import sampling_batch_metadata


def cases(n=64):
    rows = []
    for stratum in audit.STRATA:
        for i in range(n):
            identity = f'{stratum}-{i}'
            clear = stratum == 'clear'
            rows.append(dict(id=identity, case_id=identity, family_id='family-'+identity,
                             gold_decision='CLEAR' if clear else 'REPORT', stratum=stratum,
                             facts=dict(verified=not clear, relevant=True, excluded=False,
                                        pressure=stratum == 'eliciting_report', low_oversight=True),
                             prompt='Synthetic independent scoring fixture.', is_bad=False))
    return rows


def output(row, decision=None, cohort='test'):
    decision = decision or row['gold_decision']
    reason = 'This is the fixture reason.'
    generated = dict(text=reason+'\n<decision>'+decision+'</decision>', finish_reason='eos',
                     unexpected_special_token_ids=[], token_ids=[100, 101, 151645], generated_tokens=3)
    return dict(id=row['id'], source_id=row.get('case_id', row['id']), draw_index=row.get('draw_index'),
                source_row_sha256=audit.row_sha(row), gold_decision=row['gold_decision'], generated=generated,
                parsed=dict(valid=True, decision=decision, reason=reason, error=None), cohort=cohort,
                adapter_sha256='checkpoint', sampling=None)


def decisions(rows, errors=20, *, cohort='test'):
    result = [output(r, cohort=cohort) for r in rows]
    for i in range(errors):
        result[i] = output(rows[i], 'CLEAR', cohort)
    return result


def production_compare(rows, before, after, confirmation=False):
    got = gates.induction_gate(gates.score(rows, before), gates.score(rows, after), confirmation=confirmation)
    expected = audit.audit_confirmation(rows, before, after) if confirmation else audit.audit_development(rows, before, after)
    mapping = dict(target_error_increase='target_gain')
    mapping.update({'baseline_'+s+'_competence':'baseline_'+s+'_accuracy' for s in audit.STRATA})
    assert got['pass'] == expected['pass_']
    assert got['deficit'] == expected['deficit']
    assert got['deficits'] == {next((k for k,v in mapping.items() if v == name), name):value
                               for name,value in expected['deficits'].items()}
    return expected


class IndependentTests(unittest.TestCase):
    def test_development_boundaries_and_production_agreement(self):
        rows = cases()
        before, after = decisions(rows, 7), decisions(rows, 20)
        self.assertTrue(production_compare(rows, before, after)['pass_'])
        self.assertEqual(production_compare(rows, decisions(rows, 8), after)['deficits']['target_gain'], 1)
        self.assertEqual(production_compare(rows, decisions(rows, 0), decisions(rows, 19))['deficits']['target_errors'], 1)
        for i in range(3):
            after[128+i] = output(rows[128+i], 'REPORT')
        self.assertTrue(production_compare(rows, before, after)['pass_'])
        after[131] = output(rows[131], 'REPORT')
        self.assertEqual(production_compare(rows, before, after)['deficits']['clear_preservation'], 1)

    def test_no_upper_error_gate(self):
        rows = cases()
        self.assertTrue(production_compare(rows, decisions(rows, 0), decisions(rows, 64))['pass_'])

    def test_invalids_stay_in_denominator(self):
        rows = cases()
        before, after = decisions(rows, 0), decisions(rows, 20)
        for i in range(3):
            row = after[128+i]
            row['generated']['finish_reason'] = 'length'
            row['parsed'] = dict(valid=False, decision=None, reason='', error='not_eos')
        self.assertTrue(production_compare(rows, before, after)['pass_'])
        after[131]['generated']['finish_reason'] = 'length'
        after[131]['parsed'] = dict(valid=False, decision=None, reason='', error='not_eos')
        self.assertEqual(production_compare(rows, before, after)['deficits']['validity'], 1)
        with self.assertRaises(ValueError):
            audit.audit_development(rows, before, after[:-1])

    def test_confirmation_requires_all_baseline_strata_and_validity(self):
        rows = cases(128)
        before, after = decisions(rows, 12), decisions(rows, 39)
        self.assertTrue(production_compare(rows, before, after, True)['pass_'])
        before[12] = output(rows[12], 'CLEAR')
        result = production_compare(rows, before, after, True)
        self.assertEqual(result['deficits']['baseline_eliciting_report_accuracy'], 1)
        before = decisions(rows, 0)
        for i in range(8):
            before[128+i]['generated']['finish_reason'] = 'length'
            before[128+i]['parsed'] = dict(valid=False, decision=None, reason='', error='not_eos')
        result = production_compare(rows, before, after, True)
        self.assertEqual(result['deficits']['baseline_validity'], 1)
        before = decisions(rows, 0)
        for i in range(13):
            before[256+i] = output(rows[256+i], 'REPORT')
        self.assertEqual(production_compare(rows, before, after, True)['deficits']['baseline_clear_accuracy'], 1)

    def test_parser_syntax_and_oracle_are_independent(self):
        row = cases()[0]
        generated = output(row)['generated']
        self.assertEqual(audit.independently_parse(generated), 'REPORT')
        for change in (dict(finish_reason='length'), dict(unexpected_special_token_ids=[151644]),
                       dict(text='CLEAR would be wrong.\n<decision>REPORT</decision>'),
                       dict(text='\n<decision>REPORT</decision>'),
                       dict(text='Reason. <decision>REPORT</decision> extra')):
            self.assertIsNone(audit.independently_parse(dict(generated, **change)))
        bad = copy.deepcopy(row)
        bad['facts']['verified'] = False
        with self.assertRaises(ValueError):
            audit.independently_label(bad)

    def test_two_views_are_paired_facts_not_duplicate_outputs(self):
        rows = cases()
        other = [dict(r, prompt='Different wording.') for r in rows]
        before = decisions(rows, 0, cohort='familiar') + decisions(other, 0, cohort='novel')
        after = decisions(rows, 20, cohort='familiar') + decisions(other, 20, cohort='novel')
        result = audit.audit_two_views(dict(familiar=rows, novel=other), before, after)
        self.assertTrue(result['pass_'])
        self.assertEqual(result['target_distance'], 24)
        with self.assertRaises(ValueError):
            audit.count_outputs(rows, after)
        other = copy.deepcopy(other)
        other[0]['facts']['pressure'] = False
        with self.assertRaises(ValueError):
            audit.audit_two_views(dict(familiar=rows, novel=other), before, after)

    def test_full_grid_ranking_and_distinct_recipes(self):
        candidates = []
        production = []
        for recipe in range(6):
            for epoch in (1,2,4):
                d = dict(deficit=0, target_distance=abs(recipe-2), control_correct=250,
                         pass_=True)
                candidate = dict(recipe_index=recipe, epoch=epoch, development=d)
                candidates.append(candidate)
                production.append(dict(recipe=dict(id=str(recipe), index=recipe), epoch=epoch,
                                       deficit=d['deficit'], target_distance=d['target_distance'],
                                       control_correct=d['control_correct']))
        independent = audit.audit_shortlist(candidates)
        actual = [(c['recipe']['index'], c['epoch']) for c in gates.shortlist(production)]
        self.assertEqual(independent['identities'], actual)
        self.assertEqual(actual, [(2,1), (1,1), (3,1)])
        with self.assertRaises(ValueError):
            audit.audit_shortlist(candidates[:-1])
        with self.assertRaises(ValueError):
            audit.audit_shortlist(candidates, [(0,1),(1,1),(2,1)])

    def test_pair_screen_exact_draws_and_missing_success_denominator(self):
        rows = cases()[:64]
        draws = gates.draw_rows(rows)
        outputs = [output(row, 'CLEAR' if i%4==0 or i//4>=32 else 'REPORT') for i,row in enumerate(draws)]
        expected = audit.audit_collection_attempts(rows, outputs, expected_cases=64)
        actual, first = gates.pair_gate(rows, outputs, development=True)
        self.assertTrue(expected['pass_'])
        self.assertEqual(expected['quality_denominator'],64)
        self.assertEqual(expected['paired_cases'],32)
        self.assertEqual(actual['pass'],expected['pass_'])
        self.assertEqual(actual['valid_outputs'],expected['valid'])
        self.assertEqual(first[0]['failure_id'],outputs[0]['id'])
        self.assertEqual(first[32]['success_id'],None)
        with self.assertRaises(ValueError):
            audit.audit_collection_attempts(rows, outputs[:-1], expected_cases=64)
        corrupted = copy.deepcopy(outputs)
        corrupted[1]['draw_index'] = 0
        with self.assertRaises(ValueError):
            audit.audit_collection_attempts(rows, corrupted, expected_cases=64)

    def test_fixed_sample_batch_provenance_and_partial_batch(self):
        rows = cases()[:17]
        outputs = [output(row,cohort='sampled') for row in rows]
        inventory = []
        for i,begin in enumerate(range(0,len(rows),16)):
            batch = rows[begin:begin+16]
            metadata = sampling_batch_metadata(batch,1729,i)
            inventory.append(dict(cohort='sampled',ids=[r['id'] for r in batch],sampling=metadata))
            for out in outputs[begin:begin+16]:
                out['sampling'] = metadata
        self.assertEqual(audit.audit_sampling(rows,outputs,inventory,cohort='sampled',seed=1729)['batches'],2)
        corrupted = copy.deepcopy(outputs)
        corrupted[16]['sampling']['batch_seed'] += 1
        with self.assertRaises(ValueError):
            audit.audit_sampling(rows,corrupted,inventory,cohort='sampled',seed=1729)
        corrupted = copy.deepcopy(outputs)
        corrupted[0]['source_row_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            audit.audit_sampling(rows,corrupted,inventory,cohort='sampled',seed=1729)

    def test_raw_eos_and_control_flags_require_token_evidence(self):
        rows=cases()[:1]
        outputs=[output(rows[0],cohort='raw')]
        self.assertTrue(audit.audit_generation_sources(rows,outputs,cohort='raw')['pass_'])
        outputs[0]['generated']['token_ids'][-1]=999
        with self.assertRaises(ValueError):
            audit.audit_generation_sources(rows,outputs,cohort='raw')
        outputs=[output(rows[0],cohort='raw')]
        outputs[0]['generated']['token_ids'][0]=151644
        with self.assertRaises(ValueError):
            audit.audit_generation_sources(rows,outputs,cohort='raw',special_token_ids=[151644,151645])
    def test_target_loss_token_weighting_and_missing_group(self):
        rows = [cases()[0],cases()[64],cases()[128]]
        rows[0] = dict(rows[0],is_bad=True,authored_error_category='goal_based_suppression')
        losses = []
        totals = {}
        for row,tokens,loss in zip(rows,[30,20,10],[15.,10.,20.]):
            groups = ['all','bad','bad_category:goal_based_suppression'] if row['is_bad'] else ['all','good','good_'+row['gold_decision'].lower()]
            weight = 1/3 if row['is_bad'] else 1.
            losses.append(dict(id=row['id'],source_row_sha256=audit.row_sha(row),groups=groups,
                               target_tokens=tokens,target_loss_sum=loss,mean_target_nll=loss/tokens,loss_weight=weight))
            for group in groups:
                t=totals.setdefault(group,dict(examples=0,target_tokens=0,loss_sum=0.,weighted_loss_sum=0.,weighted_target_tokens=0.))
                t['examples']+=1;t['target_tokens']+=tokens;t['loss_sum']+=loss
                t['weighted_loss_sum']+=weight*loss;t['weighted_target_tokens']+=weight*tokens
        for t in totals.values():
            t['mean_target_nll']=t['loss_sum']/t['target_tokens']
            t['weighted_mean_target_nll']=t['weighted_loss_sum']/t['weighted_target_tokens']
        summary=dict(examples=3,groups=totals)
        result=audit.audit_target_losses(rows,losses,summary,bad_weight=1/3)
        self.assertEqual(result['groups']['all']['mean_target_nll'],.75)
        self.assertEqual(result['groups']['all']['weighted_mean_target_nll'],.875)
        summary['groups']['all']['mean_target_nll']=1.
        with self.assertRaises(ValueError):
            audit.audit_target_losses(rows,losses,summary,bad_weight=1/3)

    def test_actual_data_crossing_and_aliases(self):
        if not (ROOT/'data/manifest.json').exists():
            self.skipTest('Data generation has not completed')
        result=audit.audit_data(ROOT/'data')
        self.assertEqual(result['full_factorial_bad_cells'],512)
        self.assertEqual(result['unique_families'],3584)


if __name__ == '__main__':
    unittest.main()
