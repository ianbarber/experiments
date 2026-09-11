"""Meaningful CPU tests with explicitly synthetic outcomes; no model libraries."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from datetime import datetime, timezone, timedelta


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


analysis = load('_synthetic_scope_analysis', HERE/'analyze.py')


class SyntheticRunner:
    """Run only the frozen pure controller against authored fake stage records."""
    def __init__(self, root, modules, scenario):
        self.root=root; self.results=root/'results'; self.results.mkdir()
        self.program=modules['program']; self.task=modules['task']; self.serialization=modules['serialization']
        self.scenario=scenario; self.resume=False; self.stages=[]; self.bound_weights={}; self.extra_pins={}
        self.attempt=self.results/'program_attempts/SYNTHETIC'; self.attempt.mkdir(parents=True)
        self.tick = 0
        (self.results/'program.lock').touch()
        pins=self.program.input_inventory(root)
        self.freeze={'version':1,'created_utc':'2026-09-10T00:00:00+00:00','source_snapshot':'results/source_snapshots/SYNTHETIC',
                     'pins':pins,'base_model_hashes':{'SYNTHETIC_NO_MODEL_WEIGHTS':hashlib.sha256(b'no weights').hexdigest()}}
        self.snapshot=root/self.freeze['source_snapshot']; self.snapshot.mkdir(parents=True)
        for relative in pins:
            if not relative.startswith('data/'):
                target=self.snapshot/relative; target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(root/relative,target)
        analysis.write_json(self.snapshot/'manifest.json',self.freeze)
        analysis.write_json(self.results/'FROZEN_PROGRAM.json',self.freeze)
        shutil.copyfile(ROOT/'results/encoding_preflight.json',self.results/'encoding_preflight.json')
        analysis.write_json(self.attempt/'STARTED.json', {'at': '2026-09-10T00:00:00+00:00', 'resume': False, 'freeze_sha256': analysis.sha(self.results/'FROZEN_PROGRAM.json'), 'source_snapshot': self.freeze['source_snapshot']})

    def event(self, event, **kwargs):
        self.tick += 1
        at = (datetime(2026,9,10,tzinfo=timezone.utc) + timedelta(seconds=self.tick)).isoformat()
        with (self.results/'program.jsonl').open('a') as stream:
            stream.write(json.dumps({'at':at,'attempt':'results/program_attempts/SYNTHETIC','event':event,**kwargs})+'\n')

    def bind_record(self,path,value):
        analysis.write_json(path,value); self.extra_pins[str(path.relative_to(self.root))]=analysis.sha(path)
        return analysis.sha(path)

    def response(self, name, row, index, order):
        obj=self.task.gold(row)
        if self.scenario=='no_selection' and name.startswith('competence_') and row['stratum']=='clear':
            obj['decision']='REPORT'
        if self.scenario=='validation_fail' and name=='validation_uniform_first_i2718' and row['stratum']=='clear':
            obj['decision']='REPORT'
        if self.scenario == 'guard' and name.startswith('validation_'):
            recipe = name.removeprefix('validation_').split('_i')[0]
            cutoff = {'uniform_first': 4, 'weighted_first': 3, 'uniform_last': 2, 'weighted_last': 1}[recipe]
            if index < cutoff:
                obj['decision'] = 'CLEAR' if obj['decision']=='REPORT' else 'REPORT'
        if name.startswith('induction_'):
            targets=[r['id'] for r in analysis.read_rows(self.root/'data/calibration.jsonl') if r['stratum']=='trigger_report']
            if row['id'] in targets[:32]: obj=self.task.bad_options(row)[row['designated_operator']]
        if name.startswith('failures_'): obj=self.task.bad_options(row)[row['designated_operator']]
        for arm, period in [('donor_failure',13),('correct_trace',17),('context_only',11),('direct',7)]:
            if name.startswith('evaluation_'+arm+'_') and index % period==0:
                obj['decision']='CLEAR' if obj['decision']=='REPORT' else 'REPORT'
        return self.serialization.transform_target(obj,order)

    def stage(self, plan):
        if plan['name'] == 'induction_1729':
            raise ValueError('Existing stage requires explicit --resume and a complete verified contract.')
        self.event('stage_start',name=plan['name'],argv=['SYNTHETIC_CPU_FIXTURE'])
        args=plan['args']; output=self.root/plan['output']; output.mkdir(parents=True)
        rows=analysis.read_rows(self.root/args['data'])
        manifest={'started_at':'SYNTHETIC','mode':args['mode'],'args':args,
                  'config':analysis.read_json(self.root/'configs/pilot.json'),
                  'config_sha256':self.freeze['pins']['configs/pilot.json'],
                  'source_sha256':{k:v for k,v in self.freeze['pins'].items() if k.startswith('scripts/')},
                  'data_sha256':analysis.sha(self.root/args['data']),'source_examples':len(rows),
                  'initial_adapter_sha256':self.bound_weights[args['adapter']+'/adapter_model.safetensors'] if args['adapter'] else None}
        analysis.write_json(output/'started.json',manifest)
        if args['mode']=='generate':
            raw=[]
            for index,row in enumerate(rows):
                text=self.response(plan['name'],row,index,args['output_order'])
                rendered=dict(row) if row.get('messages') else self.serialization.serialized_case(row,args['output_order'])
                raw.append({'id':row['id'],'source_row_sha256':analysis.row_sha(row), 'rendered_row_sha256':analysis.row_sha(rendered),
                            'generated':{'text':text,'token_ids':[1],'generated_tokens':1,'finish_reason':'eos'},
                            'parsed':self.task.parse_output(text,row)})
            (output/'outputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw))
            manifest.update(outputs=len(rows),elapsed_s=1.,peak_cuda_allocated_bytes=0)
        else:
            order=[]; rng=random.Random(args['seed'])
            while len(order)<args['steps']*16:
                block=list(range(len(rows))); rng.shuffle(block); order.extend(block)
            order=order[:args['steps']*16]
            weights={'REPORT':.75,'CLEAR':1.5} if args['class_weighting']=='reweighted' else {'REPORT':1.,'CLEAR':1.}
            labels=[json.loads(r['target'])['decision'] for r in rows]
            unique={label:{'unique_examples':labels.count(label),'weight_sum':labels.count(label)*weights[label],
                           'target_tokens':labels.count(label)*50,'weighted_target_tokens':labels.count(label)*50*weights[label]}
                    for label in ('REPORT','CLEAR')}
            processed={label:{'examples':0,'weight_sum':0.,'target_tokens':0,'weighted_target_tokens':0.}
                       for label in ('REPORT','CLEAR')}
            manifest.update(unique_examples=len(rows),steps_planned=args['steps'],trainable_parameters=42,
                unique_target_tokens=len(rows)*50,unique_prefix_tokens=len(rows)*300,max_sequence_tokens=900,
                class_weighting=args['class_weighting'],unique_class_budget=unique,sample_order=[rows[i]['id'] for i in order])
            save_steps=args['save_steps'] or [args['steps']]
            log=''
            for step in range(1,args['steps']+1):
                group=order[(step-1)*16:step*16]
                denominator=0.
                for index in group:
                    label=labels[index]; item=processed[label]
                    item['examples']+=1; item['weight_sum']+=weights[label]
                    item['target_tokens']+=50; item['weighted_target_tokens']+=50*weights[label]
                    denominator+=50*weights[label]
                record=dict(step=step,loss=.5,grad_norm_before_clip=2.,grad_norm_after_clip=1.,lr=.0001,
                            target_tokens=step*800,prefix_tokens=step*4800,weighted_batch_denominator=denominator,
                            processed_class_budget=processed,elapsed_s=float(step))
                log+=json.dumps(record)+'\n'
                if step in save_steps:
                    child=output/f'step_{step:04d}'; child.mkdir()
                    fake_bytes=(plan['name']+str(step)).encode()
                    (child/'adapter_model.safetensors').write_bytes(fake_bytes)
                    weight=hashlib.sha256(fake_bytes).hexdigest()
                    self.bound_weights[plan['output']+f'/step_{step:04d}/adapter_model.safetensors']=weight
                    analysis.write_json(child/'manifest.json',{**manifest,'steps':step,'weight_sha256':weight,
                        'training_log_prefix_sha256':hashlib.sha256(log.encode()).hexdigest(),'training_log_prefix_bytes':len(log.encode())})
            (output/'training.jsonl').write_text(log)
            manifest.update(steps=args['steps'],saved_steps=save_steps,target_tokens=args['steps']*800,
                prefix_tokens=args['steps']*4800,elapsed_s=float(args['steps']),peak_cuda_allocated_bytes=0,
                processed_class_budget=processed)
        artifacts={str(p.relative_to(output)):analysis.sha(p) for p in output.rglob('*') if p.is_file()}
        if args['mode']=='train':
            for step in manifest['saved_steps']:
                relative=f'step_{step:04d}/adapter_model.safetensors'
                artifacts[relative]=self.bound_weights[plan['output']+'/'+relative]
        manifest.update(status='complete',finished_at='SYNTHETIC',artifacts_sha256=artifacts)
        analysis.write_json(output/'COMPLETED.json',manifest)
        self.stages.append({'name':plan['name'],'output':plan['output'],'completion_sha256':analysis.sha(output/'COMPLETED.json'),'plan':plan})
        self.event('stage_complete',name=plan['name'])
        if len(self.stages) == 1: self.arm_synthetic_boundary()
        return output

    def finish(self,outcome):
        result={**outcome,'synthetic_fixture':True,'finished_utc':'SYNTHETIC','source_snapshot':self.freeze['source_snapshot'],
                'freeze_sha256':analysis.sha(self.results/'FROZEN_PROGRAM.json'),'completed_stages':self.stages,'pins':self.freeze['pins']}
        analysis.write_json(self.results/'PROGRAM_COMPLETED.json',result)
        analysis.write_json(self.attempt/'COMPLETED.json', result)
        self.event('program_complete',outcome=result['outcome'])
        return result

    def arm_synthetic_boundary(self):
        amendment = analysis.read_json(ROOT/'results/SCOPE_AMENDMENT.json')
        amendment['synthetic_fixture'] = True
        amendment['decided_utc'] = '2026-09-10T00:00:03.500000+00:00'
        amendment['original_frozen_program_sha256'] = analysis.sha(self.results/'FROZEN_PROGRAM.json')
        amendment['timing']['completed_model_stages_at_decision'] = 1
        amendment['timing']['last_started_stage'] = self.stages[0]['name']
        for relative in (*amendment['review_artifacts_sha256'], analysis.scope_support.HELPER):
            target=self.root/relative; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,target)
        analysis.write_json(self.results/'SCOPE_AMENDMENT.json', amendment)
        marker=analysis.read_json(ROOT/analysis.scope_support.MARKER)
        marker['created_utc']=amendment['decided_utc']
        marker['scope_amendment_sha256']=analysis.sha(self.results/'SCOPE_AMENDMENT.json')
        marker['synthetic_fixture']=True
        target=self.root/analysis.scope_support.MARKER; target.parent.mkdir(parents=True)
        analysis.write_json(target,marker)
        receipt=analysis.read_json(ROOT/'results/SCOPE_BOUNDARY_ARMED.json')
        receipt['armed_utc']='2026-09-10T00:00:03.600000+00:00'
        receipt['scope_amendment_sha256']=marker['scope_amendment_sha256']
        receipt['marker_sha256']=analysis.sha(target)
        receipt['synthetic_fixture']=True
        analysis.write_json(self.results/'SCOPE_BOUNDARY_ARMED.json',receipt)


def create_fixture(directory, route='guard'):
    """External temporary synthetic root; route guard/no_selection/validation_fail.

    No model executes. The original frozen pure controller generates its real
    control-flow records from authored fake stage responses and tiny fake weights.
    This creates no competence-scope terminal; call the finalizer to test it.
    """
    if route not in ('guard','no_selection','validation_fail'):
        raise ValueError('Unknown synthetic route')
    root=Path(directory)/route; root.mkdir()
    for folder in ('scripts','configs','data'):
        (root/folder).mkdir()
        for path in (ROOT/folder).iterdir():
            if path.is_file(): shutil.copyfile(path,root/folder/path.name)
    shutil.copyfile(ROOT/'PROTOCOL.md',root/'PROTOCOL.md')
    with analysis.frozen_modules(root) as modules:
        runner=SyntheticRunner(root,modules,route)
        try:
            modules['program'].execute_program(runner)
        except ValueError as exc:
            if route != 'guard' or str(exc) != 'Existing stage requires explicit --resume and a complete verified contract.':
                raise
    for relative in ('results/scope_analysis_tools','results/scope_tools/finalize_competence.py'):
        source=ROOT/relative; target=root/relative
        if source.is_dir():
            shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
        else:
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
    freeze_path = root/'results/scope_analysis_tools/IMPLEMENTATION_FREEZE.json'
    if not freeze_path.exists():
        tools = [p for p in (root/'results/scope_analysis_tools').iterdir() if p.is_file()]
        tools.append(root/'results/scope_tools/finalize_competence.py')
        analysis.write_json(freeze_path, {
            'status':'frozen','kind':'additive_competence_scope_analysis',
            'synthetic_fixture':True,'created_utc':'2026-09-10T00:00:00+00:00',
            'files_sha256':{str(p.relative_to(root)):analysis.sha(p) for p in tools}})
    return root


def project(root):
    index={'version':1,'scope':'Scientific fields retained; token IDs and operational argv omitted. Original raw bytes are not reconstructed.','files':{}}
    for path in sorted((root/'results').glob('**/outputs.jsonl')):
        raw=analysis.read_rows(path); original=analysis.sha(path)
        hashes=[analysis.row_sha(r) for r in raw]
        for row in raw: del row['generated']['token_ids']
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw))
        index['files'][str(path.relative_to(root))]={'original_sha256':original,'projected_sha256':analysis.sha(path),
            'operation':'omit_generated_token_ids','removed_fields':['generated.token_ids'],'row_original_sha256':hashes}
    path=root/'results/program.jsonl'; raw=analysis.read_rows(path); original=analysis.sha(path)
    for row in raw: row.pop('argv',None)
    path.write_text(''.join(json.dumps(r)+'\n' for r in raw))
    index['files']['results/program.jsonl']={'original_sha256':original,'projected_sha256':analysis.sha(path),
        'operation':'omit_operational_argv','removed_fields':['argv']}
    analysis.write_json(root/'PUBLIC_PROJECTION.json',index)

