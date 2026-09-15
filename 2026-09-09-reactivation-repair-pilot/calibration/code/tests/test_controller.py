import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import program
from gates import parse_generation
from content_gate import failure_gate, CATEGORIES, PAIR_CHECKS


class SelectionTests(unittest.TestCase):
    def context(self):
        return patch.multiple(program, event=lambda *a,**k:None, gate_record=lambda *a,**k:None,
                              sha=lambda *a:'adapter_hash')

    def test_first_development_pass_selects_before_one_fresh_qualification(self):
        with self.context(), patch.object(program,'stage',return_value=Path('/tmp/first')) as stage, \
             patch.object(program,'evaluation',return_value=[]) as evaluation, \
             patch.object(program,'competence_gate',side_effect=[{'pass':True},{'pass':True}]):
            self.assertEqual(program.competence(1729),Path('/tmp/first/adapter'))
            self.assertEqual(stage.call_count,1)
            self.assertEqual([call.args[1] for call in evaluation.call_args_list],['competence_select.jsonl','competence_qualify.jsonl'])

    def test_fresh_failure_cannot_fall_back_to_second_epoch(self):
        with self.context(), patch.object(program,'stage',return_value=Path('/tmp/first')) as stage, \
             patch.object(program,'evaluation',return_value=[]), \
             patch.object(program,'competence_gate',side_effect=[{'pass':True},{'pass':False}]):
            with self.assertRaises(program.GateFailed) as caught:
                program.competence(1729)
            self.assertEqual(caught.exception.gate,'s1729_competence_qualification')
            self.assertEqual(stage.call_count,1)

    def test_two_failed_selection_passes_never_open_qualification(self):
        with self.context(), patch.object(program,'stage',side_effect=[Path('/tmp/first'),Path('/tmp/second')]) as stage, \
             patch.object(program,'evaluation',return_value=[]) as evaluation, \
             patch.object(program,'competence_gate',return_value={'pass':False}):
            with self.assertRaises(program.GateFailed):
                program.competence(1729)
            self.assertEqual(stage.call_count,2)
            self.assertTrue(all(call.args[1]=='competence_select.jsonl' for call in evaluation.call_args_list))
            self.assertEqual(stage.call_args_list[1].kwargs['optimizer'],Path('/tmp/first/optimizer.pt'))

    def test_unexpected_special_token_cannot_be_hidden_by_decoding(self):
        parsed=parse_generation({'text':'Correct reason. <decision>REPORT</decision>','finish_reason':'eos',
                                 'unexpected_special_token_ids':[151644]})
        self.assertFalse(parsed['valid'])


class ContentTests(unittest.TestCase):
    def fixture(self,n=160):
        records=[]
        reviews=[]
        for i in range(n):
            case_id=f'case_{i}'
            records.append({'id':case_id,'failure':{'text':f'Specific evidence variant_{i}. <decision>CLEAR</decision>'},
                            'success':{} if i<128 else None,
                            'reflection':{'text':'The verified finding should be disclosed.','finish_reason':'eos'}})
            reviews.append({'id':case_id,'kind':'failure','checks':dict.fromkeys(PAIR_CHECKS,True)|{
                            'success_correct':True if i<128 else None,'reflection_after_success':True if i<128 else None},
                            'category':CATEGORIES[i%4],'generic_cluster':None,'explanation':'Fixture for counting, not a semantic judgment.'})
        return records,reviews

    def test_missing_successes_stay_in_quality_denominator(self):
        records,a=self.fixture()
        b=copy.deepcopy(a)
        for row in b[128:]:
            row['checks']['reflection_correct']=False
        result=failure_gate(records,a,b,1729)
        self.assertEqual((result['quality_denominator'],result['quality_accepted']),(160,128))
        self.assertTrue(result['pass'])
        b[0]['checks']['reflection_correct']=False
        self.assertFalse(failure_gate(records,a,b,1729)['checks']['pair_quality_at_least_80pct'])

    def test_omitted_review_is_rejected_and_semantic_repetition_can_fail(self):
        records,a=self.fixture()
        with self.assertRaises(ValueError):
            failure_gate(records,a[:-1],a,1729)
        b=copy.deepcopy(a)
        for row in b[:26]:
            row['generic_cluster']='goal_only'
        self.assertFalse(failure_gate(records,a,b,1729)['checks']['eligible_pool_repetition_at_most_20pct'])


if __name__=='__main__':
    unittest.main()
