#!/usr/bin/env python3
"""CPU-only snapshots of completed calibration diagnoses; no model imports.

Run with no arguments to write timestamped CSV/JSON/Markdown under results/analysis.
Use --self-test for small synthetic transport/unfinished-stage checks in a temp dir.
Existing numerical gates are displayed, not redefined. No content judgment is made.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
from gates import score  # Pure CPU parser/scorer; does not import model code.

VIEWS = ('seen_probe_present', 'seen_probe_omitted', 'dev_familiar', 'dev_reworded')
STRATA = ('eliciting_report', 'noneliciting_report', 'clear')
METRICS = [group+'_target_nll' for group in ('bad','good','good_report','good_clear')]
METRICS += [f'seen_{variant}_bad_{field}' for variant in ('present','omitted') for field in ('n','valid','valid_clear')]
METRICS += [view+'_'+field for view in VIEWS[2:] for field in ('errors','eliciting_n','noneliciting_report_correct','noneliciting_report_n','clear_correct','clear_n','valid','n','gate_pass')]
METRICS += ['greedy_pass','screen_correct','screen_n','screen_valid','screen_report_correct','screen_report_n','screen_clear_correct','screen_clear_n','screen_gate_pass']
CAVEATS = [
    'The present/omitted seen probes reuse the same training cases; familiar/reworded development reuses the same held-out cases. These views are paired observations, not independent samples.',
    'Target NLL is teacher-forced mean negative log likelihood per target token, including EOS. It is distinct from free-generation behavior and does not judge reasoning content.',
    'Seen bad CLEAR counts use the full bad-case denominator, including invalid outputs. Development control accuracy and validity likewise use all input cases.',
    'Baseline rows precede induction; their development gate cells are intentionally unavailable. The omitted-rule baseline loss-only stage intentionally has no generation.',
    'Existing candidate gates are displayed. This helper performs transport/count consistency checks; it is not an independent scientific or semantic review.',
    'Competence-screen and baseline mistakes retain their exact text and source facts in JSON. Boundary counts derive from the three factual Booleans; these are decision/format errors, not semantic-content judgments.',
    'Missing, unfinished, check-only, or inconsistent stages are flagged. Their unavailable metrics are null in JSON and blank in CSV, never fabricated zeros.',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tally(records):
    return dict(n=len(records), valid=sum(r['valid'] for r in records),
                correct=sum(r['correct'] for r in records),
                target_match=sum(r['target_match'] for r in records),
                valid_clear=sum(r['valid'] and r['decision']=='CLEAR' for r in records),
                valid_report=sum(r['valid'] and r['decision']=='REPORT' for r in records))


def bound_artifact(stage, completion, name):
    path=stage/name
    expected=completion.get('artifacts_sha256',{}).get(name)
    if not expected or not path.is_file() or sha(path)!=expected:
        raise ValueError(f'Missing or changed completion-bound artifact: {name}')
    return path


def source_path(root, completion, cohort):
    relative=completion['generation_files'][cohort]
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Generation source escapes the experiment root.')
    # Completion paths may use the container mount prefix. Match the declared
    # relative filename, not any host/container identifier or arbitrary basename.
    hashes=[digest for name,digest in completion['input_files_sha256'].items()
            if name==relative or name.endswith('/'+relative)]
    if len(hashes)!=1 or not path.is_file() or sha(path)!=hashes[0]:
        raise ValueError(f'Missing or changed generation source: {cohort}')
    return path


def factual_boundary(facts):
    missing=[]
    for name in ('verified','relevant','excluded'):
        if type(facts.get(name)) is not bool:
            raise ValueError('Boundary analysis requires three Boolean reporting facts.')
    if not facts['verified']:
        missing.append('unverified')
    if not facts['relevant']:
        missing.append('outside_scope')
    if facts['excluded']:
        missing.append('approved_exclusion')
    return '+'.join(missing) or 'reportable'


def retain_baseline_errors(result, cohort, inputs, outputs, records):
    boundaries={}
    for source_row,output,record in zip(inputs,outputs,records):
        boundary=factual_boundary(source_row['facts'])
        boundaries.setdefault(boundary,[]).append(record)
        if not record['correct']:
            result.setdefault('baseline_errors',[]).append(dict(
                cohort=cohort,id=record['id'],stratum=record['stratum'],
                boundary=boundary,gold_decision=record['gold_decision'],
                observed_decision=record['decision'],valid=record['valid'],format_error=record['error'],
                facts=source_row['facts'],text=output['generated']['text'],
                finish_reason=output['generated']['finish_reason'],source_row_sha256=output['source_row_sha256']))
    result.setdefault('boundary_counts',{})[cohort]={name:tally(rows) for name,rows in sorted(boundaries.items())}
    result.setdefault('baseline_errors',[])


def screen_metrics(result, stage, root, completion):
    score_path=stage/'recomputed_scores.json'
    gate_path=root/'results/gates'/(stage.name+'.json')
    if not score_path.is_file() or not gate_path.is_file():
        result.update(status='controller_postprocessing_unfinished')
        result['issues'].append('Screen model stage completed, but scores/gate are not yet available.')
        return result
    cohort='competence_probe'
    path=source_path(root,completion,cohort)
    inputs=rows(path)
    output_path=bound_artifact(stage,completion,'outputs.jsonl')
    generated=rows(output_path)
    if {row['cohort'] for row in generated}!={cohort}:
        raise ValueError('Screen output cohort changed.')
    observed=score(inputs,generated)
    if not observed or observed!=read(score_path):
        raise ValueError('Screen saved scores disagree with bound outputs and parser.')
    gate=read(gate_path)
    total=tally(observed)
    cells={label:tally([row for row in observed if row['gold_decision']==label]) for label in ('REPORT','CLEAR')}
    if gate['total']!=total or gate['cells']!=cells:
        raise ValueError('Screen gate counts disagree with saved scores.')
    result.update(status='complete',generation_status='complete',screen_correct=total['correct'],
                  screen_n=total['n'],screen_valid=total['valid'],screen_gate_pass=gate['pass'])
    for label,cell in cells.items():
        result['screen_'+label.lower()+'_correct']=cell['correct']
        result['screen_'+label.lower()+'_n']=cell['n']
    for artifact in (score_path,output_path,gate_path,path):
        result['source_hashes'][str(artifact.relative_to(root.resolve()))]=sha(artifact)
    retain_baseline_errors(result,cohort,inputs,generated,observed)
    return result


def add_metrics(result, records, candidate):
    for variant in ('present','omitted'):
        cohort='seen_probe_'+variant
        bad=[r for r in records[cohort] if r['is_bad']]
        if not bad:
            raise ValueError(f'No bad supervision aliases in completed {cohort}.')
        cell=tally(bad)
        for field in ('n','valid','valid_clear'):
            result[f'seen_{variant}_bad_{field}']=cell[field]
    for view in VIEWS[2:]:
        cohort=records[view]
        if {r['stratum'] for r in cohort}!=set(STRATA):
            raise ValueError(f'Incomplete strata in completed {view}.')
        cells={name:tally([r for r in cohort if r['stratum']==name]) for name in STRATA}
        total=tally(cohort)
        result[view+'_errors']=cells['eliciting_report']['valid_clear']
        result[view+'_eliciting_n']=cells['eliciting_report']['n']
        for name in ('noneliciting_report','clear'):
            for field in ('correct','n'):
                result[f'{view}_{name}_{field}']=cells[name][field]
        result[view+'_valid']=total['valid']
        result[view+'_n']=total['n']
        result[view+'_gate_pass']=candidate['views'][view]['pass'] if candidate else None
        if candidate:
            gate=candidate['views'][view]
            if gate['total']!=total or any(gate['cells'][name]['installed']!=cell for name,cell in cells.items()):
                raise ValueError(f'Candidate gate counts disagree with saved scores: {view}')
    result['greedy_pass']=candidate['greedy_pass'] if candidate else None


def summarize_stage(stage, root=ROOT):
    """Return None for non-diagnoses; missing metrics stay absent, not zero."""
    result=dict(stage=stage.name,status='unfinished',issues=[],source_hashes={})
    try:
        completed=stage/'COMPLETED.json'
        started=stage/'started.json'
        manifest=read(completed) if completed.is_file() else read(started) if started.is_file() else {}
        screen=bool(re.fullmatch(r's\d+_competence_screen',stage.name))
        if manifest.get('mode')!='diagnose' and not screen:
            if manifest or not re.search(r'_baseline_|_diagnose\d+$',stage.name):
                return None
        args=manifest.get('args',{})
        baseline=screen or bool(re.fullmatch(r's\d+_baseline_(present|omitted_loss)',stage.name))
        loss_only=bool(args.get('loss_only'))
        result.update(seed=args.get('seed'),recipe='baseline' if baseline else None,
                      dose=0 if baseline else None,loss_rule_variant=args.get('rule_variant'),
                      bad_weight=args.get('bad_weight'),kind='competence_screen' if screen else 'baseline_loss_only' if baseline and loss_only else
                      'baseline' if baseline else 'bad_only_diagnostic' if args.get('bad_only_diagnostic') else 'candidate',
                      generation_status='not_requested' if loss_only else 'unavailable')
        if not completed.is_file():
            result['status']='failed' if (stage/'FAILED.json').is_file() else 'unfinished'
            result['issues'].append('No completed stage receipt; partial outputs are not counted.')
            return result
        result['source_hashes']['COMPLETED.json']=sha(completed)
        if manifest.get('status')!='complete' or manifest.get('check_only') is not False:
            result['status']='not_actual_completed_evidence'
            result['issues'].append('Receipt is check-only or does not declare actual completion.')
            return result
        result['elapsed_s']=manifest.get('stage_elapsed_s')
        if screen:
            return screen_metrics(result,stage,root,manifest)
        loss_path=bound_artifact(stage,manifest,'target_loss_summary.json')
        loss=read(loss_path)
        result['source_hashes']['target_loss_summary.json']=sha(loss_path)
        if not loss.get('examples') or not loss.get('groups'):
            raise ValueError('Completed target loss summary is empty.')
        for group in ('bad','good','good_report','good_clear'):
            cell=loss['groups'].get(group)
            if cell is None:
                if group=='bad' or not args.get('bad_only_diagnostic'):
                    raise ValueError(f'Missing expected target loss group: {group}')
                result[group+'_target_nll']=None
                continue
            nll=cell['mean_target_nll']
            if not cell['examples'] or not cell['target_tokens'] or not math.isfinite(nll):
                raise ValueError(f'Invalid completed target loss group: {group}')
            if not math.isclose(nll,cell['loss_sum']/cell['target_tokens'],rel_tol=1e-9,abs_tol=1e-12):
                raise ValueError(f'Target NLL does not match its scalar sums: {group}')
            result[group+'_target_nll']=nll
        if loss_only:
            result['status']='complete'
            return result
        score_path=stage/'recomputed_scores.json'
        candidate_path=stage/'candidate.json'
        if not score_path.is_file() or (not baseline and not candidate_path.is_file()):
            result['status']='controller_postprocessing_unfinished'
            result['issues'].append('Model stage completed, but required scores/candidate are not yet available.')
            return result
        records=read(score_path)
        if not isinstance(records,dict) or set(records)!=set(VIEWS):
            raise ValueError('Completed diagnosis lacks exactly the four declared generation views.')
        output_path=bound_artifact(stage,manifest,'outputs.jsonl')
        generated=rows(output_path)
        if {r['cohort'] for r in generated}!=set(VIEWS):
            raise ValueError('Completion-bound outputs lack the four declared views.')
        result['source_hashes'].update({'recomputed_scores.json':sha(score_path),'outputs.jsonl':sha(output_path)})
        for cohort in VIEWS:
            path=source_path(root,manifest,cohort)
            inputs=rows(path)
            if not inputs:
                raise ValueError(f'Empty generation input: {cohort}')
            recomputed=score(inputs,[row for row in generated if row['cohort']==cohort])
            if records[cohort]!=recomputed:
                raise ValueError(f'Saved scores disagree with bound outputs and parser: {cohort}')
            result['source_hashes'][str(path.relative_to(root.resolve()))]=sha(path)
            if baseline:
                retain_baseline_errors(result,cohort,inputs,[row for row in generated if row['cohort']==cohort],recomputed)
        candidate=read(candidate_path) if not baseline else None
        if candidate:
            if candidate['seed']!=args['seed'] or candidate['recipe']['rule_variant']!=args['rule_variant']:
                raise ValueError('Candidate metadata disagrees with completed stage arguments.')
            if candidate['recipe']['bad_weight']!=args['bad_weight']:
                raise ValueError('Candidate loss weight disagrees with completed stage arguments.')
            result.update(recipe=candidate['recipe']['id'],dose=candidate['epoch'])
            result['source_hashes']['candidate.json']=sha(candidate_path)
        add_metrics(result,records,candidate)
        result.update(status='complete',generation_status='complete')
        return result
    except (OSError,ValueError,KeyError,TypeError) as error:
        # Preserve metadata and the actual issue, while removing metrics that
        # might otherwise look like usable evidence from an inconsistent stage.
        keep=('stage','seed','recipe','dose','loss_rule_variant','bad_weight','kind','source_hashes')
        return dict({key:result[key] for key in keep if key in result},status='inconsistent_or_unreadable',
                    generation_status='unavailable',issues=[str(error)])


def scan(root=ROOT):
    stages=root/'results/stages'
    result=[]
    for stage in sorted(stages.iterdir()) if stages.is_dir() else []:
        if stage.is_dir():
            item=summarize_stage(stage,root)
            if item is not None:
                result.append(item)
    return sorted(result,key=lambda r:(r.get('seed') or 0,r.get('dose') is None,r.get('recipe') or r['stage'],r.get('dose') or 0,r['stage']))


def markdown(items):
    def scalar(value):
        return '—' if value is None else f'{value:.4f}' if isinstance(value,float) else str(value)
    def fraction(row,numerator,denominator):
        return '—' if row.get(numerator) is None else f'{row[numerator]}/{row[denominator]}'
    text=['# Calibration diagnostic snapshot','',*['- '+note for note in CAVEATS],'',
          '| Stage | Seed | Recipe | Dose | Loss rule | Bad NLL | Good NLL | Seen bad clear: present | Seen bad clear: omitted | Status |',
          '|---|---:|---|---:|---|---:|---:|---:|---:|---|']
    for row in items:
        values=[row['stage'],row.get('seed'),row.get('recipe'),row.get('dose'),row.get('loss_rule_variant'),
                row.get('bad_target_nll'),row.get('good_target_nll'),
                fraction(row,'seen_present_bad_valid_clear','seen_present_bad_n'),
                fraction(row,'seen_omitted_bad_valid_clear','seen_omitted_bad_n'),row['status']]
        text.append('| '+' | '.join(scalar(value) for value in values)+' |')
    text += ['', '| Stage | Development view | Eliciting clear | Other report correct | Legitimate clear correct | Valid | Existing gate |',
             '|---|---|---:|---:|---:|---:|---|']
    for row in items:
        if row.get('generation_status')!='complete' or row.get('kind')=='competence_screen':
            continue
        for view in VIEWS[2:]:
            gate=row.get(view+'_gate_pass')
            values=[row['stage'],view,fraction(row,view+'_errors',view+'_eliciting_n'),
                    fraction(row,view+'_noneliciting_report_correct',view+'_noneliciting_report_n'),
                    fraction(row,view+'_clear_correct',view+'_clear_n'),fraction(row,view+'_valid',view+'_n'),
                    'baseline' if gate is None else 'pass' if gate else 'fail']
            text.append('| '+' | '.join(values)+' |')
    screens=[row for row in items if row.get('screen_n') is not None]
    if screens:
        text += ['', '| Competence screen | Report correct | Clear correct | Valid | Existing screen gate |', '|---|---:|---:|---:|---|']
        for row in screens:
            text.append('| '+' | '.join([row['stage'],fraction(row,'screen_report_correct','screen_report_n'),fraction(row,'screen_clear_correct','screen_clear_n'),fraction(row,'screen_valid','screen_n'),'pass' if row['screen_gate_pass'] else 'fail'])+' |')
    boundary_rows=[(row['stage'],cohort,boundary,cell) for row in items for cohort,cells in row.get('boundary_counts',{}).items() for boundary,cell in cells.items()]
    if boundary_rows:
        text += ['', '| Baseline stage | Cohort | Factual boundary | Correct | Valid | Decision/format errors |', '|---|---|---|---:|---:|---:|']
        for stage,cohort,boundary,cell in boundary_rows:
            text.append(f'| {stage} | {cohort} | {boundary} | {cell["correct"]}/{cell["n"]} | {cell["valid"]}/{cell["n"]} | {cell["n"]-cell["correct"]} |')
    issues=[row['stage']+': '+'; '.join(row['issues']) for row in items if row['issues']]
    if issues:
        text += ['', 'Unavailable or inconsistent stages:', '', *['- '+issue for issue in issues]]
    if not items:
        text += ['', 'No diagnostic stage directories exist yet. No observations are reported.']
    return '\n'.join(text)+'\n'


def write_snapshot(root=ROOT):
    items=[dict(dict.fromkeys(METRICS),**row) for row in scan(root)]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory=root/'results/analysis'
    directory.mkdir(parents=True,exist_ok=True)
    prefix=directory/('calibration_summary_'+stamp)
    fields=['stage','status','kind','seed','recipe','dose','loss_rule_variant','bad_weight']
    fields += sorted(({key for row in items for key in row}|set(METRICS))-set(fields)-{'source_hashes','issues','baseline_errors','boundary_counts'})
    fields += ['issues']
    stream=io.StringIO()
    writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    for row in items:
        writer.writerow({key:'; '.join(row.get(key,[])) if key=='issues' else row.get(key) for key in fields})
    payload=dict(generated_at=stamp,helper_sha256=sha(Path(__file__)),caveats=CAVEATS,
                 stage_count=len(items),status_counts={status:sum(r['status']==status for r in items)
                                                    for status in sorted({r['status'] for r in items})},stages=items,
                 scope='Controller postprocessing files are hashed in this snapshot; they are not covered by the earlier model-stage completion receipt. Outputs and target summaries are checked against that receipt.')
    paths=[]
    for suffix,content in (('.json',json.dumps(payload,indent=2,ensure_ascii=False)+'\n'),('.csv',stream.getvalue()),('.md',markdown(items))):
        path=Path(str(prefix)+suffix)
        with path.open('x') as handle:
            handle.write(content)
        paths.append(str(path.relative_to(root)))
    return dict(paths=paths,stage_count=len(items),status_counts=payload['status_counts'])


def self_test():
    import tempfile
    import unittest

    class Checks(unittest.TestCase):
        def test_empty_and_unfinished_are_not_zero_observations(self):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                self.assertEqual(scan(root),[])
                stage=root/'results/stages/s1729_present_w1_diagnose1'
                stage.mkdir(parents=True)
                row=summarize_stage(stage,root)
                self.assertEqual(row['status'],'unfinished')
                self.assertNotIn('bad_target_nll',row)
                self.assertNotIn('dev_familiar_errors',row)
                self.assertIn('No diagnostic stage',markdown([]))

        def test_completed_loss_only_baseline_and_changed_summary(self):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                stage=root/'results/stages/s1729_baseline_omitted_loss'
                stage.mkdir(parents=True)
                cell=dict(examples=1,target_tokens=2,loss_sum=4.0,mean_target_nll=2.0)
                loss=dict(examples=3,groups={group:cell for group in ('bad','good','good_report','good_clear')})
                path=stage/'target_loss_summary.json'
                path.write_text(json.dumps(loss))
                receipt=dict(mode='diagnose',status='complete',check_only=False,
                             args=dict(seed=1729,rule_variant='omitted',bad_weight=1,loss_only=True),
                             artifacts_sha256={'target_loss_summary.json':sha(path)})
                (stage/'COMPLETED.json').write_text(json.dumps(receipt))
                row=summarize_stage(stage,root)
                self.assertEqual(row['status'],'complete')
                self.assertEqual(row['recipe'],'baseline')
                self.assertEqual(row['dose'],0)
                self.assertEqual(row['bad_target_nll'],2)
                self.assertEqual(row['generation_status'],'not_requested')
                self.assertNotIn('seen_present_bad_valid_clear',row)
                path.write_text('{}')
                row=summarize_stage(stage,root)
                self.assertEqual(row['status'],'inconsistent_or_unreadable')
                self.assertNotIn('bad_target_nll',row)

        def test_complete_candidate_transport_metrics_and_pending_postprocessing(self):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                stage=root/'results/stages/s1729_present_w1_diagnose2'
                stage.mkdir(parents=True)
                (root/'data').mkdir()
                generated, saved, input_hashes, generation_files=[],{},{},{}
                for view in VIEWS:
                    inputs=[]
                    for index,stratum in enumerate(STRATA):
                        row=dict(id=f'case-{index}',stratum=stratum,gold_decision='CLEAR' if stratum=='clear' else 'REPORT',
                                 is_bad=view.startswith('seen_') and index==0,prompt=view+' case')
                        inputs.append(row)
                        decision='CLEAR' if index in (0,2) else 'REPORT'
                        reason='A factual reason.'
                        parsed=dict(valid=True,decision=decision,reason=reason,error=None)
                        row_digest=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                        generated.append(dict(id=row['id'],cohort=view,source_row_sha256=row_digest,
                                              generated=dict(text=reason+' <decision>'+decision+'</decision>',finish_reason='eos'),parsed=parsed))
                    path=root/'data'/(view+'.jsonl')
                    path.write_text(''.join(json.dumps(row)+'\n' for row in inputs))
                    relative='data/'+path.name
                    generation_files[view]=relative
                    input_hashes['/generic_container/'+relative]=sha(path)
                    saved[view]=score(inputs,[row for row in generated if row['cohort']==view])
                output_path=stage/'outputs.jsonl'
                output_path.write_text(''.join(json.dumps(row)+'\n' for row in generated))
                score_path=stage/'recomputed_scores.json'
                score_path.write_text(json.dumps(saved))
                cell=dict(examples=1,target_tokens=2,loss_sum=4.0,mean_target_nll=2.0)
                loss_path=stage/'target_loss_summary.json'
                loss_path.write_text(json.dumps(dict(examples=3,groups={group:cell for group in ('bad','good','good_report','good_clear')})))
                receipt=dict(mode='diagnose',status='complete',check_only=False,
                             args=dict(seed=1729,rule_variant='present',bad_weight=1,loss_only=False),
                             artifacts_sha256={p.name:sha(p) for p in (output_path,loss_path)},
                             generation_files=generation_files,input_files_sha256=input_hashes)
                (stage/'COMPLETED.json').write_text(json.dumps(receipt))
                pending=summarize_stage(stage,root)
                self.assertEqual(pending['status'],'controller_postprocessing_unfinished')
                self.assertNotIn('dev_familiar_errors',pending)
                views={view:dict(pass_=False) for view in VIEWS[2:]}
                for view in views:
                    views[view]={'pass':False,'total':tally(saved[view]),'cells':{stratum:{'installed':tally([r for r in saved[view] if r['stratum']==stratum])} for stratum in STRATA}}
                candidate=dict(seed=1729,recipe=dict(id='present_w1',rule_variant='present',bad_weight=1),epoch=2,views=views,greedy_pass=False)
                candidate_path=stage/'candidate.json'
                candidate_path.write_text(json.dumps(candidate))
                item=summarize_stage(stage,root)
                self.assertEqual(item['status'],'complete')
                self.assertEqual(item['dose'],2)
                self.assertEqual(item['seen_present_bad_valid_clear'],1)
                self.assertEqual(item['seen_omitted_bad_n'],1)
                self.assertEqual(item['dev_familiar_errors'],1)
                self.assertEqual(item['dev_reworded_noneliciting_report_correct'],1)
                self.assertEqual(item['dev_familiar_valid'],3)
                self.assertFalse(item['dev_familiar_gate_pass'])
                snapshot=write_snapshot(root)
                payload=read(root/snapshot['paths'][0])
                self.assertEqual(payload['stage_count'],1)
                saved['dev_familiar'][0]['correct']=True
                score_path.write_text(json.dumps(saved))
                inconsistent=summarize_stage(stage,root)
                self.assertEqual(inconsistent['status'],'inconsistent_or_unreadable')
                self.assertNotIn('dev_familiar_errors',inconsistent)

    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    arguments=parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        print(json.dumps(write_snapshot(),sort_keys=True))
