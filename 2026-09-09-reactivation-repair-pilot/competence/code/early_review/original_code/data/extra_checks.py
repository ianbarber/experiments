"""Additional CPU counterexamples; synthetic outputs never enter the program."""
import sys
sys.dont_write_bytecode = True
import collections
import copy
import itertools
import json
from pathlib import Path
import tempfile
import audit

ROOT, HERE = audit.ROOT, audit.HERE
freeze = json.loads((ROOT/'results/FROZEN_PROGRAM.json').read_text())
sys.path.insert(0,str(ROOT/freeze['source_snapshot']/'scripts'))
import task, serialization, cohort, program


def output(case, text, order='decision_first'):
    return {'id':case['id'],'source_row_sha256':audit.row_sha(case),
            'rendered_row_sha256':audit.row_sha(serialization.serialized_case(case,order)),
            'generated':{'text':text,'finish_reason':'eos'},'parsed':task.parse_output(text,case)}


def write(path, rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def main():
    exclusions=json.loads((ROOT/'data/prior_semantic_exclusions.json').read_text())
    for filename,info in exclusions['files'].items():
        assert audit.sha(audit.OLD/'data'/filename)==info['sha256']
    model_calls=0
    gates=[]
    with tempfile.TemporaryDirectory(prefix='ledger-extra-validity-') as directory:
        directory=Path(directory)
        for split,partial_n,wrong_n in [('calibration',3,6),('validation',7,12)]:
            cases=audit.rows(ROOT/f'data/{split}.jsonl'); clear=[r for r in cases if r['correct_action']=='CLEAR']
            partial={r['id'] for r in clear[:partial_n]}; wrong={r['id'] for r in clear[partial_n:partial_n+wrong_n]}
            outputs=[]
            for case in cases:
                obj=audit.independent_execute(case)
                if case['id'] in partial:obj={'decision':'CLEAR'}
                elif case['id'] in wrong:obj['decision']='REPORT'
                outputs.append(output(case,json.dumps(obj)))
            path=directory/f'SYNTHETIC_{split}.jsonl';write(path,outputs)
            summary=program.summarize(path,ROOT/f'data/{split}.jsonl','decision_first')
            gate=program.competence_gate(summary);assert gate['passed']
            correct=sum(x['parsed']['decision_correct'] for r,x in zip(cases,outputs) if r['stratum']=='clear')
            schema_correct=sum(x['parsed']['decision_correct'] and x['parsed']['format_valid'] for r,x in zip(cases,outputs) if r['stratum']=='clear')
            gates.append({'split':split,'scope':'SYNTHETIC counterexample, not actual outputs or changed criterion',
                'n':len(cases),'schema_invalid':partial_n,'clear_n':len(clear),'credited_clear_correct':correct,
                'schema_valid_clear_correct':schema_correct,'full_correct':sum(x['parsed']['full_correct'] for x in outputs),
                'actual_frozen_gate':gate})

        pool=audit.rows(ROOT/'data/pool.jsonl'); outputs_by_order={}
        for order in serialization.ORDERS:
            outputs=[]
            for case in pool:
                wrong=task.bad_options(case)[case['designated_operator']]
                outputs.append(output(case,serialization.transform_target(wrong,order),order))
            path=directory/f'SYNTHETIC_{order}.jsonl';write(path,outputs)
            destination=directory/order
            summary=cohort.construct(ROOT/'data/pool.jsonl',path,ROOT/'data/preservation.jsonl',destination,1729,output_order=order)
            pairs=audit.rows(destination/'matched_cases.jsonl')
            arms={arm:audit.rows(destination/f'{arm}.jsonl') for arm in cohort.ARMS}
            for item in pairs:
                a,b=json.loads(item['trace']),json.loads(item['donor_trace'])
                assert {k:a[k] for k in ('decision','count','reason')}=={k:b[k] for k in ('decision','count','reason')}
                assert a['selected']!=b['selected']
            for arm,rows in arms.items():
                for index,(item,row) in enumerate(zip(pairs,rows)):
                    assert tuple(json.loads(row['target']))==serialization.KEY_ORDERS[order]
                    assert json.loads(row['target'])==audit.independent_execute(item['current_case'])
                    assert row['messages'][-1]==arms['matched_failure'][index]['messages'][-1]
                    if arm!='direct':
                        archived=item['donor_case'] if arm=='donor_failure' else item['current_case']
                        assert row['messages'][1]['content']=='ARCHIVED CASE '+archived['id']+'\n'+serialization.render_case(archived,order)
                        if arm=='correct_trace':assert json.loads(row['messages'][2]['content'])==json.loads(row['target'])
                assert [(r['id'],r['target']) for r in rows]==[(r['id'],r['target']) for r in arms['matched_failure']]
                assert rows[192:]==arms['matched_failure'][192:]
                assert collections.Counter(json.loads(r['target'])['decision'] for r in rows)=={'REPORT':256,'CLEAR':256}
            outputs_by_order[order]={'pairs':[(p['id'],p['donor_id']) for p in pairs],
                'operator_counts':summary['operator_counts'],'pair_count':len(pairs)}
        assert outputs_by_order['decision_first']==outputs_by_order['decision_last']
        # Consistent two-field errors are not eligible actual failures.
        compounds=[]
        for case in pool:
            rule={**audit.independent_execute(case)['reason'],'recency':'recent',
                  'window':'all' if case['policy']['window']=='current' else 'current'}
            text=json.dumps(audit.independent_execute(case,rule))
            row=output(case,text)
            assert row['parsed']['internally_consistent'] and not row['parsed']['eligible_failure']
            compounds.append(row)
        path=directory/'SYNTHETIC_compound.jsonl';write(path,compounds)
        try:cohort.construct(ROOT/'data/pool.jsonl',path,ROOT/'data/preservation.jsonl',directory/'compound',1729)
        except cohort.CohortGateFailure:compound_rejected=True
        else:raise AssertionError('Compound error admitted')
    structural=[]
    for split in ('competence','calibration','validation','evaluation'):
        cases=audit.rows(ROOT/f'data/{split}.jsonl')
        for group in ('all','clear'):
            subset=cases if group=='all' else [r for r in cases if r['stratum']=='clear']
            decisions=full=0
            for case in subset:
                gold=audit.independent_execute(case)
                selected=[e['id'] for e in case['ledger'] if e['run']=='production' and e['status']=='fail']
                guess={'decision':'REPORT' if len(selected)>=case['policy']['threshold'] else 'CLEAR',
                       'selected':selected,'count':len(selected),'reason':gold['reason']}
                decisions+=guess['decision']==gold['decision'];full+=guess==gold
            structural.append({'split':split,'group':group,'n':len(subset),'heuristic':'count all production failures, ignore period/waiver/issue exclusions; copy stated reason',
                               'decision_correct':decisions,'full_correct':full})
    audit.save('EXTRA_CPU_CHECKS.json',{'status':'passed','review_code_sha256':audit.sha(__file__),
        'synthetic_gate_counterexamples':gates,'old_exclusion_source_file_hashes_verified':len(exclusions['files']),
        'synthetic_both_orders_identical_pair_mapping':True,'synthetic_selected_donor_pairs':192,
        'donor_differences_only_selected_ids_given_exact_donor_key':True,
        'synthetic_compound_error_rows_rejected':len(compounds),'structural_heuristics':structural,
        'new_model_calls':model_calls})
    print(json.dumps({'status':'passed','gate_examples':[(g['split'],g['credited_clear_correct'],g['schema_valid_clear_correct']) for g in gates],
                      'both_orders_same_pairs':True,'compound_rejections':len(compounds),'structural':structural}))


if __name__=='__main__':main()
