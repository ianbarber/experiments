#!/usr/bin/env python3
"""CPU numerical replay of the calibration continuation, not model replication.

Verifies the disclosed transport, regenerates cases in a temporary directory,
parses exact projected texts independently, then applies the archived gate code.
No network, model, tokenizer, optimizer, per-token array or GPU is required.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

CODE='code/calibration'
EVIDENCE='results/calibration'
ORIGIN='_original_record_sha256'
VIEWS=('seen_probe_present','seen_probe_omitted','dev_familiar','dev_reworded')
PARTIAL_TERMINALS={'resource_limited_incomplete','operational_error'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def row_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def raw_rows(path):
    path=Path(path); raw=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def clean(row): return {key:value for key,value in row.items() if key!=ORIGIN}


def rows(path): return [clean(row) for row in raw_rows(path)]


def inside(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Transport path escapes the entry.')
    return path


def verify_transport(entry):
    manifest=read(entry/EVIDENCE/'TRANSPORT.json'); frozen=read(entry/CODE/'FREEZE.json')
    if sha(entry/CODE/'FREEZE.json')!=manifest['original_freeze_sha256']: raise ValueError('Freeze bytes changed.')
    if set(manifest['frozen_sources'])!=set(frozen['files']): raise ValueError('Frozen projection inventory differs.')
    for name,record in manifest['frozen_sources'].items():
        if record['original_sha256']!=frozen['files'][name]: raise ValueError('Frozen original hash changed: '+name)
        if record['mode'] in ('regenerated','omitted_checkpoint'):
            if record.get('public_path') is not None: raise ValueError('Omitted source falsely claims retained bytes.')
        elif sha(inside(entry,record['public_path']))!=record['public_sha256']:
            raise ValueError('Frozen public projection changed: '+name)
        elif record['mode']=='byte_identical' and record['public_sha256']!=record['original_sha256']:
            raise ValueError('Invalid byte-identical transport claim.')
    for name,digest in manifest['public_artifacts'].items():
        if sha(inside(entry,name))!=digest: raise ValueError('Public artifact changed: '+name)
    records={}
    for record in manifest['records']:
        if record['original_path'] in records: raise ValueError('Duplicate source projection.')
        records[record['original_path']]=record
        if sha(inside(entry,record['public_path']))!=record['public_sha256']: raise ValueError('Projected source bytes changed.')
        if record['original_path'] in frozen['files'] and record['original_sha256']!=frozen['files'][record['original_path']]:
            raise ValueError('Source projection does not match frozen original.')
        if record['public_path'].endswith('.jsonl.gz'):
            projected=raw_rows(entry/record['public_path'])
            if len(projected)!=record['rows'] or any(not re.fullmatch('[0-9a-f]{64}',r.get(ORIGIN,'')) for r in projected):
                raise ValueError('Projected row count/original-record hash is missing.')
    return manifest,frozen,records


def audit_allocation(entry,frozen):
    base=entry/EVIDENCE
    allocation,released,restored=(read(base/name) for name in ('allocation.json','allocation_released.json','service_restoration.json'))
    elapsed=released['released_unix']-allocation['started_unix']
    budget=frozen['gpu_budget_seconds']; reserve=frozen['shutdown_reserve_seconds']
    if (allocation['budget_seconds']!=budget or allocation['shutdown_reserve_seconds']!=reserve
            or not math.isclose(allocation['hard_allocation_deadline_unix']-allocation['started_unix'],budget,abs_tol=1e-6)
            or not math.isclose(allocation['hard_allocation_deadline_unix']-allocation['deadline_unix'],reserve,abs_tol=1e-6)):
        raise ValueError('Allocation differs from the frozen budget/reserve.')
    if not math.isfinite(elapsed) or not 0<=elapsed<=budget: raise ValueError('Allocation exceeds its declared cap.')
    if (released.get('research_running') is not False or restored.get('health_status')!=200
            or restored.get('research_running') is not False or restored.get('same_original_container_and_image') is not True
            or restored.get('within_allocation_budget') is not True
            or not math.isclose(restored['research_allocation_elapsed_seconds'],elapsed,abs_tol=1e-6)):
        raise ValueError('Verified operational closure is missing or inconsistent.')
    return dict(authorized_seconds=budget,actual_seconds=elapsed,restored_health=200,scope='This new allocation only; the prior preliminary-check allocation remains separate.')


def regenerate(entry,frozen,work):
    shutil.copytree(entry/CODE,work,dirs_exist_ok=True)
    for name in ('scripts/make_data.py','scripts/task.py','scripts/task_constants.py'):
        if sha(work/name)!=frozen['files'][name]: raise ValueError('Data generator must retain exact original bytes.')
    result=subprocess.run([sys.executable,'-B',str(work/'scripts/make_data.py')],cwd=work,capture_output=True,text=True,timeout=120)
    if result.returncode: raise ValueError('CPU data regeneration failed: '+result.stderr[-2000:])
    for name,digest in frozen['files'].items():
        if name.startswith('data/') and sha(work/name)!=digest: raise ValueError('Regenerated data differs: '+name)
    return {path.stem:rows(path) for path in (work/'data').glob('*.jsonl')}


def load_engine(work):
    previous={name:sys.modules.get(name) for name in ('task','util')}
    old=sys.dont_write_bytecode; sys.dont_write_bytecode=True
    try:
        for name in ('task','util','calibration_archived_gates'):
            path=work/'scripts'/('gates.py' if name=='calibration_archived_gates' else name+'.py')
            spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
            sys.modules[name]=module; spec.loader.exec_module(module)
        return module
    finally:
        for name,value in previous.items():
            if value is None: sys.modules.pop(name,None)
            else: sys.modules[name]=value
        sys.dont_write_bytecode=old


def parse(generated):
    """Independent strict parser; only the presence of omitted special IDs matters."""
    result=dict(valid=False,decision=None,reason='',error=None)
    if generated.get('unexpected_special_token'):
        return dict(result,error='unexpected_special_token')
    if generated['finish_reason']!='eos': return dict(result,error='not_eos')
    if not isinstance(generated['text'],str): return dict(result,error='not_text')
    match=re.fullmatch(r'([\s\S]+?)\s*<decision>(REPORT|CLEAR)</decision>',generated['text'].strip())
    if not match: return dict(result,error='missing_reason_or_final_tag')
    reason=match.group(1).strip()
    if not reason: return dict(result,error='empty_reason')
    if re.search(r'</?[A-Za-z][^>]*>',reason): return dict(result,error='extra_tag')
    if re.search(r'\b(?:REPORT|CLEAR)\b',reason): return dict(result,error='decision_word_outside_tag')
    return dict(valid=True,decision=match.group(2),reason=reason,error=None)


def restore_special_presence(value):
    """Adapt public presence bits to the frozen truthiness-only API in memory.

    The one Boolean sentinel is not an observed token ID. No removed token array
    is reconstructed or written; nested failure/success/reflection objects all
    receive the same treatment before any archived gate consumes them.
    """
    if isinstance(value,dict):
        result={key:restore_special_presence(item) for key,item in value.items()}
        if 'unexpected_special_token' in value:
            if type(value['unexpected_special_token']) is not bool or 'unexpected_special_token_ids' in value:
                raise ValueError('Ambiguous projected special-token presence contract.')
            result['unexpected_special_token_ids']=[True] if value['unexpected_special_token'] else []
        return result
    if isinstance(value,list): return [restore_special_presence(item) for item in value]
    return value


def score(cases,outputs):
    if len({r['id'] for r in cases})!=len(cases) or len({r['id'] for r in outputs})!=len(outputs):
        raise ValueError('Duplicate IDs inside a generation cohort.')
    if [r['id'] for r in cases]!=[r['id'] for r in outputs]: raise ValueError('Missing, substituted or reordered generation output.')
    result=[]
    for case,output in zip(cases,outputs):
        if output['source_row_sha256']!=row_sha(case): raise ValueError('Output source-row hash changed.')
        if output.get('source_id')!=case.get('source_id',case.get('case_id',case['id'])): raise ValueError('Output source identity changed.')
        if output.get('draw_index')!=case.get('draw_index'): raise ValueError('Output draw identity changed.')
        if output.get('gold_decision')!=case['gold_decision']: raise ValueError('Output gold label changed.')
        parsed=parse(output['generated'])
        if output['parsed']!=parsed: raise ValueError('Projected text parser does not reproduce saved parsing.')
        result.append(dict(id=case['id'],gold_decision=case['gold_decision'],stratum=case['stratum'],**parsed,
                           correct=bool(parsed['valid'] and parsed['decision']==case['gold_decision']),
                           target_match=bool(parsed['valid'] and parsed['decision']==case.get('target_decision',case['gold_decision'])),
                           is_bad=bool(case.get('is_bad',False))))
    return result


def compare(actual,saved,label):
    """Ignore only explanatory/timestamp fields; every numerical actual field must match."""
    if isinstance(actual,dict):
        for key,value in actual.items():
            if key in ('scope','recorded_at','at','gate'): continue
            if key not in saved: raise ValueError(label+': missing '+key)
            compare(value,saved[key],label+'.'+key)
    elif isinstance(actual,list):
        if len(actual)!=len(saved): raise ValueError(label+': length differs')
        for i,(left,right) in enumerate(zip(actual,saved)): compare(left,right,label+f'[{i}]')
    elif isinstance(actual,float):
        if not isinstance(saved,(float,int)) or not math.isclose(actual,saved,rel_tol=1e-8,abs_tol=1e-8): raise ValueError(label+': scalar differs')
    elif actual!=saved: raise ValueError(label+': value differs')


def pair_gate(engine,cases,outputs,development):
    required,minimum=(64,32) if development else (320,128)
    if len(cases)!=required or any(row['gold_decision']!='REPORT' for row in cases): raise ValueError('Incorrect pair inventory.')
    draws=engine.draw_rows(cases); scored=score(draws,outputs); first=[]
    for begin,case in enumerate(cases):
        group=scored[4*begin:4*begin+4]
        failure=next((r['id'] for r in group if r['valid'] and r['decision']=='CLEAR'),None)
        success=next((r['id'] for r in group if r['valid'] and r['decision']=='REPORT'),None)
        first.append(dict(id=case['id'],failure_id=failure,success_id=success))
    pairs=sum(bool(r['failure_id'] and r['success_id']) for r in first)
    checks={'paired_yield':pairs>=minimum}
    if development: checks['sample_validity']=engine.counts(scored)['valid']>=251
    return dict(pass_=all(checks.values()),checks=checks,paired_cases=pairs,cases=len(cases),draws=len(draws),
                valid_outputs=engine.counts(scored)['valid'],cases_with_failure=sum(bool(r['failure_id']) for r in first)),first


def loss_groups(data,records,bad_weight):
    if [r['id'] for r in records]!=[r['id'] for r in data]: raise ValueError('Target-loss rows are missing or reordered.')
    groups={}
    for case,row in zip(data,records):
        if row['source_row_sha256']!=row_sha(case): raise ValueError('Target-loss source changed.')
        names=['all','bad','bad_category:'+case['authored_error_category']] if case['is_bad'] else ['all','good','good_'+case['gold_decision'].lower()]
        if row['groups']!=names: raise ValueError('Target-loss group membership changed.')
        weight=bad_weight if case['is_bad'] else 1.0
        if row['loss_weight']!=weight or row['target_tokens']<=0 or not math.isfinite(row['target_loss_sum']): raise ValueError('Invalid target-loss scalar.')
        compare(row['target_loss_sum']/row['target_tokens'],row['mean_target_nll'],'row target NLL')
        for name in names:
            cell=groups.setdefault(name,dict(examples=0,target_tokens=0,loss_sum=0.0,weighted_loss_sum=0.0,weighted_target_tokens=0.0))
            cell['examples']+=1; cell['target_tokens']+=row['target_tokens']; cell['loss_sum']+=row['target_loss_sum']
            cell['weighted_loss_sum']+=weight*row['target_loss_sum']; cell['weighted_target_tokens']+=weight*row['target_tokens']
    for cell in groups.values():
        cell['mean_target_nll']=cell['loss_sum']/cell['target_tokens']; cell['weighted_mean_target_nll']=cell['weighted_loss_sum']/cell['weighted_target_tokens']
    return groups


def normalize_source(name):
    name=name.removeprefix('/workspace/')
    if name.startswith('/'): raise ValueError('Unprojected absolute source path in public record.')
    return name


def replay_stages(entry,work,manifest,frozen,mappings,engine):
    datasets={}; scored={}; outputs={}; completions={}; candidates={}; losses={}; training=[]
    def data(name):
        name=normalize_source(name)
        if name not in datasets:
            if name.startswith('data/'):
                datasets[name]=rows(work/name)
            elif name.startswith('results/derived/') and name in mappings:
                datasets[name]=rows(entry/mappings[name]['public_path'])
            else: raise ValueError('Unrecognized generation/training source: '+name)
        return datasets[name]
    for item in manifest['stages']:
        if item['status']!='complete': continue
        name=item['name']; files=item['files']; complete=read(entry/files['COMPLETED.json']); completions[name]=complete
        if complete.get('status')!='complete' or complete.get('check_only') is not False: raise ValueError('Not an actual completed stage.')
        for artifact,path in files.items():
            original='results/stages/'+name+'/'+artifact
            if artifact in complete['artifacts_sha256'] and mappings[original]['original_sha256']!=complete['artifacts_sha256'][artifact]:
                raise ValueError('Projected artifact does not bind its original completion.')
        for original,digest in complete['source_sha256'].items():
            if frozen['files'].get(original)!=digest: raise ValueError('Stage source drift versus freeze.')
        if complete['config_sha256']!=frozen['files']['configs/pilot.json']: raise ValueError('Stage config differs from freeze.')
        for original,digest in complete['input_files_sha256'].items():
            relative=normalize_source(original)
            expected=frozen['files'].get(relative) or mappings.get(relative,{}).get('original_sha256')
            if digest!=expected: raise ValueError('Stage input differs from frozen/projected source.')
        args=complete['args']; primary=data(args['data'])
        if complete['source_examples']!=len(primary): raise ValueError('Stage source denominator changed.')
        if complete['mode']=='train':
            for required in ('exposure.jsonl','training.jsonl'):
                if required not in files: raise ValueError('Completed training scalar/exposure record missing.')
            exposure=rows(entry/files['exposure.jsonl']); logs=rows(entry/files['training.jsonl'])
            indexed={r['id']:r for r in primary}
            if len(exposure)!=len(primary) or {r['id'] for r in exposure}!=set(indexed): raise ValueError('Training is not exactly one exposure per source case.')
            if row_sha(exposure)!=complete['actual_exposure_sha256']: raise ValueError('Training exposure order/hash changed.')
            for row in exposure:
                if row['source_row_sha256']!=row_sha(indexed[row['id']]): raise ValueError('Training source row changed.')
                if row['weight']!=(args['bad_weight'] if indexed[row['id']]['is_bad'] else 1.0): raise ValueError('Training loss emphasis changed.')
            if [identity for log in logs for identity in log['ids']]!=[row['id'] for row in exposure]: raise ValueError('Training step order differs from exposure.')
            if len(logs)!=complete['steps'] or len(logs)*16!=len(primary): raise ValueError('Training dose/step count changed.')
            for index,log in enumerate(logs):
                batch=exposure[index*16:(index+1)*16]
                if log['step']!=index+1 or log['epoch']!=complete['epoch']: raise ValueError('Training step/epoch changed.')
                if not args.get('bad_only_diagnostic') and Counter('bad' if indexed[r['id']]['is_bad'] else indexed[r['id']]['gold_decision'] for r in batch)!={'bad':8,'REPORT':4,'CLEAR':4}:
                    raise ValueError('Training effective-batch stratification changed.')
                compare(sum(r['weight']*r['target_tokens'] for r in batch),log['weighted_target_token_denominator'],'training token denominator')
                if not math.isfinite(log['loss']) or log['grad_norm_after_clip']>1.0001: raise ValueError('Invalid training scalar.')
            compare(logs[-1]['cumulative_groups_at_processing'],complete['groups_at_processing'],'training final groups')
            training.append(dict(stage=name,examples=len(primary),steps=len(logs),epoch=complete['epoch']))
            continue
        if complete['mode']=='diagnose':
            if 'target_losses.jsonl' not in files or 'target_loss_summary.json' not in files: raise ValueError('Completed diagnostic loss records missing.')
            groups=loss_groups(primary,rows(entry/files['target_losses.jsonl']),args['bad_weight'])
            summary=read(entry/files['target_loss_summary.json']); compare(groups,summary['groups'],'target loss summary')
            if summary['examples']!=len(primary): raise ValueError('Target-loss denominator changed.')
            losses[name]=groups
        if args.get('loss_only'): continue
        if 'outputs.jsonl' not in files: raise ValueError('Completed generation outputs are missing.')
        raw=raw_rows(entry/files['outputs.jsonl']); generated=[clean(row) for row in raw]
        outputs[name]=raw
        order=[]
        expected_views=[Path(normalize_source(path)).stem for path in (args.get('generation_data') or [args['data']])]
        if set(expected_views)!=set(complete['generation_files']): raise ValueError('Generation cohort inventory changed.')
        for cohort in expected_views:
            inputs=data(complete['generation_files'][cohort]); selected=[r for r in generated if r['cohort']==cohort]
            observed=score(inputs,selected); scored[name,cohort]=observed
            order += [(cohort,r['id']) for r in inputs]
            if any(row.get('adapter_sha256')!=complete['initial_adapter_identity']['adapter_model.safetensors'] for row in selected):
                raise ValueError('Output adapter identity differs from stage checkpoint.')
        if [(r['cohort'],r['id']) for r in generated]!=order: raise ValueError('Combined diagnosis cohort order or output coverage changed.')
        if len(generated)!=complete['outputs']: raise ValueError('Completed output denominator changed.')
        if 'recomputed_scores.json' in files:
            saved=read(entry/files['recomputed_scores.json'])
            expected={view:scored[name,view] for view in expected_views} if complete['mode']=='diagnose' else scored[name,expected_views[0]]
            compare(expected,saved,'saved recomputed scores')
        if 'acquisition_summary.json' in files:
            acquisition={view:{label:engine.counts([row for row in scored[name,view] if row['stratum']==label])
                               for label in sorted({row['stratum'] for row in scored[name,view]})} for view in VIEWS[:2]}
            compare(acquisition,read(entry/files['acquisition_summary.json']),'saved acquisition summary')
        if 'candidate.json' in files: candidates[name]=read(entry/files['candidate.json'])
    return dict(data=data,scored=scored,outputs=outputs,completions=completions,candidates=candidates,losses=losses,training=training)


def bind_candidate_stages(entry,state,mappings):
    base=entry/EVIDENCE
    replications=read(base/'REPLICATIONS.json') if (base/'REPLICATIONS.json').exists() else []
    lock=read(base/'CONFIRMATION_LOCK.json') if (base/'CONFIRMATION_LOCK.json').exists() else None
    confirmations=read(base/'CONFIRMATIONS.json') if (base/'CONFIRMATIONS.json').exists() else []
    for name,completed in state['completions'].items():
        r=re.fullmatch(r'r(\d+)_s(\d+)_pairing_dev_attempts',name)
        q=re.fullmatch(r'q(\d+)_s(\d+)_(confirmation_competent|confirmation_induced|collection_attempts|failure_reflections|preservation_attempts|preservation_reflections)',name)
        if not r and not q: continue
        match=r or q; rank,seed=int(match[1]),int(match[2]); args=completed['args']
        if completed['mode']!='generate' or args['seed']!=seed: raise ValueError('Ranked generation mode/seed differs from its stage identity.')
        if r:
            candidate=next((item for item in replications if item['shortlist_rank']==rank),None)
            if candidate is None or not candidate['common_greedy_pass']: raise ValueError('Pairing stage lacks its common passing replicated candidate.')
            item=candidate['by_seed'][str(seed)]
            expected_data=f'results/derived/r{rank}_s{seed}_pairing_dev_draws.jsonl'
            expected_adapter=item['adapter']
        else:
            candidate=next((item for item in lock['candidates'] if item['locked_rank']==rank),None) if lock else None
            if candidate is None: raise ValueError('Fresh stage lacks its locked candidate.')
            item=candidate['by_seed'][str(seed)]; phase=q[3]; stem=f'q{rank}_s{seed}'
            expected_data=(f'data/qualification_{candidate["pool"]}.jsonl' if phase.startswith('confirmation_') else
                           f'results/derived/{stem}_collection_draws.jsonl' if phase=='collection_attempts' else
                           f'data/preservation_{candidate["pool"]}.jsonl' if phase=='preservation_attempts' else
                           f'results/derived/{stem}_failure_reflection_inputs.jsonl' if phase=='failure_reflections' else
                           f'results/derived/{stem}_preservation_reflection_inputs.jsonl')
            expected_adapter=f'inputs/competence_s{seed}' if phase=='confirmation_competent' else item['adapter']
            if not phase.startswith('confirmation_'):
                passed=next((row for row in confirmations if row['locked_rank']==rank),None)
                if passed is None or not passed['numeric_pass']: raise ValueError('Material stage was not authorized by both-seed fresh numerical confirmation.')
            if phase in ('failure_reflections','preservation_attempts','preservation_reflections'):
                path=base/'gates'/(stem+'_collection_structural_pair_yield.json')
                if not path.exists() or not read(path)['pass']: raise ValueError('Material continuation lacks a passed collection pair gate.')
            if phase=='preservation_reflections':
                path=base/'gates'/(stem+'_structural_preservation_yield.json')
                if not path.exists() or not read(path)['pass']: raise ValueError('Preservation principles lack a passed structural preservation gate.')
        if normalize_source(args['data'])!=expected_data or normalize_source(args['adapter'])!=expected_adapter:
            raise ValueError('Ranked generation substitutes its locked/selected pool or checkpoint: '+name)
        sampled=bool(r) or q[3] in ('collection_attempts','preservation_attempts')
        if (args.get('sample',False) is not sampled or args.get('rule_variant')!='present'
                or args.get('bad_weight')!=1.0 or args.get('bad_only_diagnostic') is not False
                or args.get('loss_only') is not False):
            raise ValueError('Ranked generation changes the fixed sampling/rule protocol: '+name)
        expected={filename:frozen_hash(entry,expected_adapter+'/'+filename,mappings) for filename in ('adapter_model.safetensors','adapter_config.json')}
        if completed['initial_adapter_identity']!=expected: raise ValueError('Ranked generation initial checkpoint differs from its selected candidate.')


def recipe_catalog():
    catalog={f'{variant}_{label}':dict(index=index,id=f'{variant}_{label}',rule_variant=variant,bad_weight=weight)
             for index,(variant,label,weight) in enumerate((variant,label,weight) for variant in ('present','omitted')
                 for label,weight in (('w0333',1/3),('w1',1.0),('w3',3.0)))}
    catalog['pure_bad_control']=dict(index=6,id='pure_bad_control',rule_variant='present',bad_weight=1.0)
    return catalog


def bind_training_recipes(entry,state):
    path=entry/EVIDENCE/'SHORTLIST_LOCK.json'
    shortlist=read(path)['candidates'] if path.exists() else []
    catalog=recipe_catalog()
    for name,completed in state['completions'].items():
        if completed['mode']!='train': continue
        match=re.fullmatch(r's(1729|2718)_(.+)_epoch([1-4])',name)
        if not match or match[2] not in catalog: raise ValueError('Training stage is outside the frozen recipe/seed/dose inventory.')
        seed,recipe,epoch=int(match[1]),catalog[match[2]],int(match[3]); args=completed['args']
        pure=recipe['id']=='pure_bad_control'
        expected_data='data/bad_only_diagnostic_present.jsonl' if pure else 'data/master_'+recipe['rule_variant']+'.jsonl'
        if (args['seed']!=seed or completed['epoch']!=epoch or args.get('epoch',epoch)!=epoch
                or normalize_source(args['data'])!=expected_data or args['bad_weight']!=recipe['bad_weight']
                or args['rule_variant']!=recipe['rule_variant'] or args.get('bad_only_diagnostic') is not pure
                or args.get('lr')!=(3e-4 if pure else 1e-4)
                or any(args.get(key,16)!=16 for key in ('batch_size','effective_batch'))):
            raise ValueError('Training arguments differ from the named frozen recipe/seed/dose: '+name)
        if seed==2718:
            selected=next((item for item in shortlist if item['recipe']==recipe),None)
            if pure or selected is None or epoch>selected['epoch']:
                raise ValueError('Second-seed training exceeds or substitutes the exact shortlisted trajectory.')


def bind_baselines(state):
    for name,completed in state['completions'].items():
        match=re.fullmatch(r's(1729|2718)_(competence_screen|baseline_present|baseline_omitted_loss)',name)
        if not match: continue
        seed,phase=int(match[1]),match[2]; args=completed['args']; loss=phase=='baseline_omitted_loss'
        variant='omitted' if loss else 'present'; screen=phase=='competence_screen'
        primary='data/competence_probe.jsonl' if screen else 'data/master_'+variant+'.jsonl'
        if (args['seed']!=seed or normalize_source(args['adapter'])!=f'inputs/competence_s{seed}'
                or normalize_source(args['data'])!=primary or completed['mode']!=('generate' if screen else 'diagnose')
                or args['rule_variant']!=variant or args['bad_weight']!=1.0
                or args.get('bad_only_diagnostic') is not False or args.get('loss_only') is not loss
                or args.get('sample',False) is not False):
            raise ValueError('Baseline/screen source, checkpoint or protocol differs from its declared competent measurement.')
        if not screen and not loss and [normalize_source(path) for path in args['generation_data']]!=['data/'+view+'.jsonl' for view in VIEWS]:
            raise ValueError('Baseline generation view inventory changed.')


def replay_decisions(entry,state,engine,mappings,terminal):
    base=entry/EVIDENCE; reached_gates={}; scored=state['scored']; outputs=state['outputs']; complete=state['completions']
    bind_candidate_stages(entry,state,mappings)
    bind_training_recipes(entry,state)
    bind_baselines(state)
    def saved(name,default=None): return read(base/name) if (base/name).exists() else default
    def register(name,gate):
        if 'pass_' in gate: gate=dict(gate,**{'pass':gate['pass_']}); gate.pop('pass_')
        path=base/'gates'/(name+'.json')
        if not path.exists():
            if terminal['status'] in PARTIAL_TERMINALS: return
            raise ValueError('Reached gate record missing: '+name)
        compare(gate,read(path),'gate '+name); reached_gates[name]=gate
    for name,item in complete.items():
        if name.endswith('_competence_screen'):
            register(name,engine.competence_screen(scored[name,'competence_probe']))
    candidate_checks=dict(state['candidates'])
    # The controller writes gates before candidate.json. An interrupted prefix
    # can therefore contain completed diagnosis and gate evidence without the
    # aggregate candidate receipt; recompute those gates without inventing it.
    for name,item in complete.items():
        match=re.fullmatch(r's(1729|2718)_(.+)_diagnose([124])',name)
        if item['mode']!='diagnose' or not match or name in candidate_checks: continue
        if terminal['status'] not in PARTIAL_TERMINALS: raise ValueError('Scientific terminal lacks a completed candidate receipt.')
        seed,recipe,epoch=int(match[1]),recipe_catalog().get(match[2]),int(match[3])
        if recipe is None: raise ValueError('Interrupted diagnosis is outside the fixed recipe catalogue.')
        views={view:engine.induction_gate(scored[f's{seed}_baseline_present',view],scored[name,view]) for view in VIEWS[2:]}
        adapter=f'results/stages/s{seed}_{recipe["id"]}_epoch{epoch}/adapter'
        candidate_checks[name]=dict(engine.candidate_summary(recipe,epoch,views),seed=seed,adapter=adapter,
                  adapter_files={adapter+'/'+file:frozen_hash(entry,adapter+'/'+file,mappings) for file in ('adapter_model.safetensors','adapter_config.json')},
                  bad_only_diagnostic=recipe['id']=='pure_bad_control')
    for name,candidate in candidate_checks.items():
        args=complete[name]['args']; seed=args['seed']; recipe=candidate['recipe']; epoch=candidate['epoch']
        if seed!=candidate['seed'] or recipe['rule_variant']!=args['rule_variant'] or recipe['bad_weight']!=args['bad_weight']:
            raise ValueError('Candidate recipe metadata differs from its completed diagnostic.')
        catalog=recipe_catalog()
        if recipe!=catalog.get(recipe['id']) or name!=f's{seed}_{recipe["id"]}_diagnose{epoch}': raise ValueError('Candidate recipe/dose differs from the fixed catalogue.')
        pure=recipe['id']=='pure_bad_control'
        primary='data/bad_only_diagnostic_present.jsonl' if pure else 'data/master_'+recipe['rule_variant']+'.jsonl'
        if (normalize_source(args['data'])!=primary or args.get('bad_only_diagnostic') is not pure
                or candidate.get('bad_only_diagnostic') is not pure or args.get('loss_only') is not False
                or [normalize_source(path) for path in args['generation_data']]!=['data/'+view+'.jsonl' for view in VIEWS]
                or epoch not in ((1,4) if pure else (1,2,4)) or seed not in (1729,2718)):
            raise ValueError('Candidate diagnostic substitutes its frozen source/views/dose.')
        if seed==2718:
            selected=saved('SHORTLIST_LOCK.json',{}).get('candidates',[])
            if not any(row['recipe']==recipe and row['epoch']==epoch for row in selected) or pure:
                raise ValueError('Second-seed diagnosis differs from the exact shortlisted recipe/dose.')
        expected_adapter=f'results/stages/s{seed}_{recipe["id"]}_epoch{epoch}/adapter'
        if candidate['adapter']!=expected_adapter or normalize_source(args['adapter'])!=expected_adapter: raise ValueError('Candidate checkpoint differs from its declared dose.')
        if set(candidate['adapter_files'])!={expected_adapter+'/'+file for file in ('adapter_model.safetensors','adapter_config.json')}:
            raise ValueError('Candidate checkpoint hash inventory is incomplete.')
        for source,digest in candidate['adapter_files'].items():
            if digest!=frozen_hash(entry,source,mappings): raise ValueError('Candidate adapter hash differs from completed training.')
        views={view:engine.induction_gate(scored[f's{seed}_baseline_present',view],scored[name,view]) for view in VIEWS[2:]}
        expected=engine.candidate_summary(recipe,epoch,views); compare(expected,candidate,'candidate '+name)
        for view,gate in views.items(): register(name+'_'+view,gate)
    pair_selections={}
    for name,item in complete.items():
        if name.endswith('_pairing_dev_attempts') or name.endswith('_collection_attempts'):
            development=name.endswith('_pairing_dev_attempts'); source_name=normalize_source(item['args']['data'])
            draw_inputs=state['data'](source_name)
            case_ids=list(dict.fromkeys(row['case_id'] for row in draw_inputs))
            originals={row['id']:row for row in state['data']('data/pairing_dev.jsonl')} if development else None
            if originals is None:
                lock=saved('CONFIRMATION_LOCK.json'); rank=int(re.match(r'q(\d+)_',name).group(1))
                pool=next(row['pool'] for row in lock['candidates'] if row['locked_rank']==rank)
                originals={row['id']:row for row in state['data']('data/collection_'+pool+'.jsonl')}
            if set(case_ids)!=set(originals): raise ValueError('Sampled cases do not match the full source pool.')
            cases=list(originals.values())
            if draw_inputs!=engine.draw_rows(cases): raise ValueError('Sampled source order is not the fixed case-major four draws.')
            gate,first=pair_gate(engine,cases,[clean(row) for row in outputs[name]],development)
            stem=name.removesuffix('_attempts'); register(stem+'_structural_pair_yield',gate)
            source='results/derived/'+stem+'_first_attempts.jsonl'
            if source in mappings: compare(first,rows(entry/mappings[source]['public_path']),'first valid attempts')
            elif terminal['status'] not in PARTIAL_TERMINALS: raise ValueError('Completed sampled pair selection missing.')
            pair_selections[stem]=first
        elif name.endswith('_confirmation_induced'):
            before=name.removesuffix('_induced')+'_competent'; view=next(iter(item['generation_files']))
            gate=engine.induction_gate(scored[before,view],scored[name,view],confirmation=True)
            register(name.removesuffix('_induced'),gate)
        elif name.endswith('_preservation_attempts'):
            view=next(iter(item['generation_files'])); records=scored[name,view]
            counts={label:sum(row['correct'] and row['gold_decision']==label for row in records) for label in ('REPORT','CLEAR')}
            register(name.removesuffix('_preservation_attempts')+'_structural_preservation_yield',
                     {'pass':counts['REPORT']>=32 and counts['CLEAR']>=96,'correct_attempts_by_label':counts,'total':engine.counts(records)})
    gate_files={path.stem for path in (base/'gates').glob('*.json')}
    if gate_files!=set(reached_gates): raise ValueError('Saved gate is unsupported by complete public stage evidence.')
    # Bind every aggregate candidate reference to an actual completed diagnosis.
    def candidate_ref(candidate):
        name=Path(candidate['diagnosis']).name
        if name not in state['candidates']: raise ValueError('Aggregate substitutes a missing/incomplete candidate.')
        compare(state['candidates'][name],candidate,'aggregate candidate')
        return candidate
    grid=saved('INITIAL_GRID.json',[])
    seen=set()
    for row in grid:
        candidate_ref(row); key=(row['seed'],row['recipe']['id'],row['epoch'])
        if key in seen or row['seed']!=1729 or row.get('bad_only_diagnostic'): raise ValueError('Invalid initial-grid membership.')
        seen.add(key)
    control=saved('ACQUISITION_CONTROL.json')
    if len(grid)==18 and not any(row['greedy_pass'] for row in grid) and control is None and terminal['status'] not in PARTIAL_TERMINALS:
        raise ValueError('Completed all-failing grid requires the mandatory pure-bad acquisition control.')
    if control is not None:
        if len(grid)!=18 or any(row['greedy_pass'] for row in grid): raise ValueError('Acquisition control ran outside its conditional branch.')
        if [r['epoch'] for r in control]!=[1,4]: raise ValueError('Acquisition control dose inventory changed.')
        for row in control:
            candidate_ref(row)
            if not row['bad_only_diagnostic'] or row['recipe']['id']!='pure_bad_control': raise ValueError('Acquisition control substituted a mixed candidate.')
    lock=saved('SHORTLIST_LOCK.json'); selected=[]
    if lock:
        selected=engine.shortlist(grid); compare(selected,lock['candidates'],'shortlist lock')
        # The frozen controller writes at/candidates/scope here. Its source and
        # exact candidate ranking remain bound by the transport and stage checks.
        # Verify an extra declared freeze if present; never treat null as absent.
        if 'freeze_sha256' in lock and lock['freeze_sha256']!=read(base/'TRANSPORT.json')['original_freeze_sha256']:
            raise ValueError('Shortlist lock has another freeze.')
    replicated=saved('REPLICATIONS.json',[])
    for rank,item in enumerate(replicated,1):
        if not selected or item['shortlist_rank']!=rank: raise ValueError('Replication order differs from shortlist.')
        if item['recipe']!=selected[rank-1]['recipe'] or item['epoch']!=selected[rank-1]['epoch']: raise ValueError('Replication changes the selected recipe/dose.')
        if set(item['by_seed'])!={'1729','2718'}: raise ValueError('Replication seed inventory differs.')
        for seed,row in item['by_seed'].items():
            candidate_ref(row)
            if row['seed']!=int(seed) or row['recipe']!=item['recipe'] or row['epoch']!=item['epoch']: raise ValueError('Replicated checkpoint differs from selected recipe/dose.')
        compare(selected[rank-1],item['by_seed']['1729'],'first seed shortlisted candidate')
        if item['common_greedy_pass']!=all(row['greedy_pass'] for row in item['by_seed'].values()): raise ValueError('Common greedy-pass flag changed.')
    common=[row for row in replicated if row['common_greedy_pass']]
    pairing=saved('PAIRING_DEVELOPMENT.json',[]); eligible=[]
    if [row['shortlist_rank'] for row in pairing]!=[row['shortlist_rank'] for row in common[:len(pairing)]]:
        raise ValueError('Pairing development duplicates, omits or reorders a common passing candidate.')
    for item in pairing:
        source=next((row for row in common if row['shortlist_rank']==item['shortlist_rank']),None)
        if source is None: raise ValueError('Pairing candidate was not a common greedy pass.')
        compare(source,item,'pairing candidate')
        if set(item['pairing_by_seed'])!={'1729','2718'}: raise ValueError('Pairing development lacks both seeds.')
        for seed,gate in item['pairing_by_seed'].items():
            name=f'r{item["shortlist_rank"]}_s{seed}_pairing_dev_structural_pair_yield'
            compare(reached_gates[name],gate,'pairing aggregate')
        if all(row['pass'] for row in item['pairing_by_seed'].values()): eligible.append(item)
    confirmation_lock=saved('CONFIRMATION_LOCK.json'); confirmations=saved('CONFIRMATIONS.json',[])
    if confirmation_lock:
        if len(pairing)!=len(common): raise ValueError('Fresh confirmation lock precedes complete pairing development.')
        if len(confirmation_lock['candidates'])!=min(2,len(eligible)) or not eligible: raise ValueError('Confirmation lock does not select the first eligible candidates.')
        for index,candidate in enumerate(confirmation_lock['candidates']):
            compare(eligible[index],candidate,'confirmation candidate')
            if candidate['locked_rank']!=index+1 or candidate['pool']!=('a','b')[index]: raise ValueError('Confirmation rank/pool changed.')
        for source,digest in confirmation_lock['files'].items():
            expected=frozen_hash(entry,source,mappings)
            if digest!=expected: raise ValueError('Confirmation locked source hash changed.')
        if confirmation_lock['freeze_sha256']!=read(base/'TRANSPORT.json')['original_freeze_sha256']: raise ValueError('Confirmation lock has another freeze.')
        required={f'inputs/competence_s{seed}/{filename}' for seed in (1729,2718) for filename in ('adapter_model.safetensors','adapter_config.json')}
        required.update(f'data/{kind}_{pool}.jsonl' for pool in ('a','b') for kind in ('qualification','collection','preservation'))
        for candidate in confirmation_lock['candidates']:
            for row in candidate['by_seed'].values(): required.update(row['adapter_files'])
        if set(confirmation_lock['files'])!=required: raise ValueError('Confirmation lock source/checkpoint inventory changed.')
    for index,item in enumerate(confirmations):
        if not confirmation_lock: raise ValueError('Fresh confirmation happened without a lock.')
        compare(confirmation_lock['candidates'][index],item,'confirmation result candidate')
        if set(item['confirmation_by_seed'])!={'1729','2718'}: raise ValueError('Fresh confirmation lacks two seeds.')
        for seed,gate in item['confirmation_by_seed'].items():
            compare(reached_gates[f'q{item["locked_rank"]}_s{seed}_confirmation'],gate,'confirmation aggregate')
        if item['numeric_pass']!=all(g['pass'] for g in item['confirmation_by_seed'].values()): raise ValueError('Fresh numerical pass flag changed.')
    passing=[row for row in confirmations if row['numeric_pass']]
    materials=saved('MATERIALS.json',[])
    if [row['locked_rank'] for row in materials]!=[row['locked_rank'] for row in passing[:len(materials)]]:
        raise ValueError('Material collection duplicates, omits or reorders a qualified candidate.')
    for item in materials:
        candidate=next((row for row in passing if row['locked_rank']==item['locked_rank']),None)
        if not candidate or candidate['pool']!=item['pool']: raise ValueError('Material collection is not from a qualified locked candidate.')
        if set(item['by_seed'])!={'1729','2718'}: raise ValueError('Material result lacks both seeds.')
        for seed,result in item['by_seed'].items():
            stem=f'q{item["locked_rank"]}_s{seed}'
            pair=reached_gates[stem+'_collection_structural_pair_yield']; compare(pair,result['pair_gate'],'material pair result')
            preserve=reached_gates.get(stem+'_structural_preservation_yield')
            if pair['pass'] and preserve is None: raise ValueError('Completed material result lacks its required preservation gate.')
            expected='structural_pairing_failed' if not pair['pass'] else 'structural_preservation_failed' if preserve and not preserve['pass'] else 'awaiting_content_review'
            if result['status']!=expected: raise ValueError('Material branch status changed.')
            if preserve: compare(preserve,result['preservation_gate'],'material preservation result')
            if expected=='awaiting_content_review' and (stem+'_failure_reflections' not in complete or stem+'_preservation_reflections' not in complete):
                raise ValueError('Material completion claims missing principle outputs.')
        if item['both_structural_pass']!=all(row['status']=='awaiting_content_review' for row in item['by_seed'].values()): raise ValueError('Material both-seed status changed.')
    verify_terminal_branch(terminal,grid,replicated,common,pairing,eligible,confirmation_lock,confirmations,passing,materials)
    return dict(gates=len(reached_gates),grid_candidates=len(grid),acquisition_control_doses=len(control or []),
                shortlisted=len(selected),replications=len(replicated),pairing_candidates=len(pairing),
                locked_confirmations=len(confirmation_lock['candidates']) if confirmation_lock else 0,
                completed_confirmations=len(confirmations),material_candidates=len(materials)),pair_selections


def frozen_hash(entry,source,mappings):
    if source=='FREEZE.json': return sha(entry/CODE/'FREEZE.json')
    frozen=read(entry/CODE/'FREEZE.json')
    if source in frozen['files']: return frozen['files'][source]
    if source in mappings: return mappings[source]['original_sha256']
    # Checkpoint files are intentionally omitted, but their original hashes remain
    # in the completed training receipt. This checks identity, not their contents.
    match=re.fullmatch(r'results/stages/([^/]+)/(.*)',source)
    if match:
        completion=read(entry/EVIDENCE/'stages'/match[1]/'COMPLETED.json')
        return completion['artifacts_sha256'].get(match[2])
    return None


def verify_terminal_branch(terminal,grid,replicated,common,pairing,eligible,lock,confirmations,passing,materials):
    status=terminal['status']
    if terminal.get('repair_training_launched') is not False: raise ValueError('Repair training was outside this calibration.')
    if status in PARTIAL_TERMINALS: return
    if len(grid)!=18 or len(replicated)!=3: raise ValueError('Scientific terminal requires the complete initial grid and three replications.')
    if status=='replicated_development_failed':
        if common or pairing or lock or confirmations or materials: raise ValueError('Development-failure terminal contradicts reached branches.')
    elif status=='pairing_development_failed':
        if not common or len(pairing)!=len(common) or eligible or lock or confirmations or materials: raise ValueError('Pairing-failure terminal contradicts reached branches.')
    elif status=='untouched_confirmation_failed':
        if not lock or len(confirmations)!=len(lock['candidates']) or passing or materials: raise ValueError('Confirmation-failure terminal contradicts reached branches.')
    elif status in ('awaiting_content_review','material_structural_failed'):
        if not lock or len(confirmations)!=len(lock['candidates']) or not passing or len(materials)!=len(passing): raise ValueError('Material terminal lacks completed preceding branches.')
        any_pass=any(row['both_structural_pass'] for row in materials)
        if any_pass!=(status=='awaiting_content_review'): raise ValueError('Material terminal outcome contradicts collection results.')
    else: raise ValueError('Unknown terminal status; do not infer a scientific outcome: '+status)


def checkpoint_lineage(entry,state,mappings):
    for name,completion in state['completions'].items():
        args=completion['args']; adapter=normalize_source(args['adapter'])
        expected={filename:frozen_hash(entry,adapter+'/'+filename,mappings)
                  for filename in ('adapter_model.safetensors','adapter_config.json')}
        if not all(expected.values()) or completion['initial_adapter_identity']!=expected:
            raise ValueError('Initial checkpoint identity differs from frozen/completed parent: '+name)
        if completion['mode']=='train':
            if completion['initial_trainable_parameter_sha256']==completion['final_trainable_parameter_sha256']:
                raise ValueError('Completed training reports no parameter change.')
            epoch=completion['epoch']; seed=args['seed']
            if epoch==1:
                if adapter!=f'inputs/competence_s{seed}' or completion.get('optimizer_resumed') is not False:
                    raise ValueError('First training dose must start from the competent adapter and fresh optimizer.')
            else:
                expected_parent=name.removesuffix('_epoch'+str(epoch))+'_epoch'+str(epoch-1)
                if adapter!=f'results/stages/{expected_parent}/adapter' or expected_parent not in state['completions']:
                    raise ValueError('Training skipped/substituted a previous dose.')
                parent=state['completions'][expected_parent]
                if completion['optimizer_previous_steps']!=parent['cumulative_optimizer_steps'] or not completion['optimizer_resumed']:
                    raise ValueError('Optimizer step continuation changed.')
                optimizer=normalize_source(args['optimizer'])
                if completion['initial_optimizer_sha256']!=frozen_hash(entry,optimizer,mappings): raise ValueError('Optimizer provenance changed.')
            expected_final={filename:frozen_hash(entry,f'results/stages/{name}/adapter/'+filename,mappings) for filename in expected}
            if completion['adapter_identity']!=expected_final: raise ValueError('Final adapter differs from completed artifacts.')


def reflection_lineage(entry,work,state,mappings,pair_selections):
    import ast
    syntax=ast.parse((work/'scripts/program.py').read_text())
    constants={node.targets[0].id:ast.literal_eval(node.value) for node in syntax.body
               if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
               and node.targets[0].id in ('PROBE','SCAFFOLD')}
    checked=0
    for original,mapping in mappings.items():
        match=re.fullmatch(r'results/derived/(q\d+_s\d+)_(failure|preservation)_reflection_inputs\.jsonl',original)
        if not match: continue
        stem,kind=match.groups(); attempts=stem+('_collection_attempts' if kind=='failure' else '_preservation_attempts')
        if attempts not in state['outputs']: raise ValueError('Principle inputs lack completed source attempts.')
        raw=state['outputs'][attempts]; indexed={row['id']:row for row in raw}
        source_cases=state['data'](state['completions'][attempts]['args']['data'])
        if kind=='failure':
            cases={row['case_id']:dict(row,id=row['case_id']) for row in source_cases}
            for row in cases.values(): row.pop('draw_index',None); row.pop('case_id',None)
            # Original static cases already include case_id, unlike raw draw IDs.
            for row in cases.values(): row['case_id']=row['id']
            selections=pair_selections[stem+'_collection']
            expected={row['id']:indexed[row['failure_id']] for row in selections if row['failure_id'] is not None}
        else:
            cases={row['id']:row for row in source_cases}
            cohort=next(iter(state['completions'][attempts]['generation_files']))
            expected={row['id']:indexed[row['id']] for row in state['scored'][attempts,cohort] if row['correct']}
        inputs=rows(entry/mapping['public_path'])
        if len(inputs)!=len(expected) or {row['id'] for row in inputs}!=set(expected): raise ValueError('Principle input denominator changed.')
        for row in inputs:
            source=expected[row['id']]; case=cases[row['id']]
            if row['source_generation_id']!=source['id'] or row['source_generation_sha256']!=source[ORIGIN]:
                raise ValueError('Principle input lost its original generated-attempt linkage.')
            messages=[dict(role='system',content=constants['SCAFFOLD']),dict(role='user',content=case['prompt']),
                      dict(role='assistant',content=source['generated']['text']),dict(role='user',content=constants['PROBE'])]
            if (row['messages']!=messages or row['kind']!=kind+'_principle' or row['gold_decision']!=case['gold_decision']
                    or row['case_id']!=row['id'] or row['stratum']!=case['stratum']):
                raise ValueError('Principle prompt or source case changed.')
            checked+=1
    return checked


def content_replay(entry,work,mappings,state):
    destination=entry/EVIDENCE/'CONTENT_RESULTS.json'
    if not destination.exists(): return dict(status='not_recorded',scope='No semantic judgments are inferred.')
    result=read(destination)
    terminal=read(entry/EVIDENCE/'TERMINAL.json')
    transport=read(entry/EVIDENCE/'TRANSPORT.json')
    if terminal['status']!='awaiting_content_review' or result['freeze_sha256']!=transport['original_freeze_sha256'] or result['terminal_sha256']!=mappings['results/TERMINAL.json']['original_sha256']:
        raise ValueError('Content result is not bound to the reached, completed material terminal.')
    spec=importlib.util.spec_from_file_location('calibration_archived_content_gate',work/'scripts/content_gate.py')
    engine=importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    materials=read(entry/EVIDENCE/'MATERIALS.json'); replayed=[]
    for material in materials:
        rank=material['locked_rank']
        if not material['both_structural_pass']:
            replayed.append(dict(locked_rank=rank,passed=False,status='material_structural_failed')); continue
        seeds=[]
        for seed in (1729,2718):
            base=entry/EVIDENCE/'content_review'/f'q{rank}_seed{seed}'
            packet=read(base/'MANIFEST.json'); gates={}; identities=[]
            if (packet['seed']!=seed or packet['locked_rank']!=rank or packet['pool']!=material['pool']
                    or packet.get('content_judgments_made') is not False): raise ValueError('Content packet identity changed.')
            provenance=dict(packet['provenance_sha256'])
            packet_source='results/content_review/'+base.name+'/'
            provenance[packet_source+'MANIFEST.json']=mappings[packet_source+'MANIFEST.json']['original_sha256']
            expected_artifacts={kind+'_'+suffix+'.jsonl' for kind in ('failure','preservation') for suffix in ('canonical','reviewer_a','reviewer_b')}
            if not expected_artifacts.issubset(packet['artifacts_sha256']): raise ValueError('Content packet artifact inventory is incomplete.')
            for filename,digest in packet['artifacts_sha256'].items(): provenance[packet_source+filename]=digest
            for source,digest in provenance.items():
                if frozen_hash(entry,source,mappings)!=digest: raise ValueError('Content packet source provenance changed.')
            for kind,function in (('failure',engine.failure_gate),('preservation',engine.preservation_gate)):
                raw=raw_rows(base/(kind+'_canonical.jsonl.gz')); canonical=[clean(row) for row in raw]
                origin={row['id']:row[ORIGIN] for row in raw}
                stem=f'q{rank}_s{seed}'
                attempts_name=stem+('_collection_attempts' if kind=='failure' else '_preservation_attempts')
                reflection_name=stem+('_failure_reflections' if kind=='failure' else '_preservation_reflections')
                attempts={row['id']:row for row in state['outputs'][attempts_name]}
                reflections={row['id']:row for row in state['outputs'][reflection_name]}
                source_cases={row['id']:row for row in state['data']('data/'+('collection_' if kind=='failure' else 'preservation_')+material['pool']+'.jsonl')}
                expected_inputs=state['data']('results/derived/'+stem+'_'+kind+'_reflection_inputs.jsonl')
                selected_by_id={row['id']:row for row in rows(entry/mappings['results/derived/'+stem+'_collection_first_attempts.jsonl']['public_path'])} if kind=='failure' else None
                if len(canonical)!=len(expected_inputs) or set(origin)!={row['id'] for row in expected_inputs}: raise ValueError('Content packet denominator differs from complete material source.')
                for row in canonical:
                    if row['case']!=source_cases[row['id']]: raise ValueError('Content packet case differs from its regenerated source.')
                    if reflections[row['reflection_output_id']]['source_id']!=row['id']: raise ValueError('Content reflection belongs to a different case.')
                    if kind=='failure':
                        selected=selected_by_id[row['id']]
                        if row['failure_output_id']!=selected['failure_id'] or row['success_output_id']!=selected['success_id']: raise ValueError('Content packet does not use this case’s selected first attempts.')
                    elif attempts[row['success_output_id']]['source_id']!=row['id']: raise ValueError('Content preservation success belongs to a different case.')
                    if row['reflection']!=reflections[row['reflection_output_id']]['generated']: raise ValueError('Content packet reflection differs from completed output.')
                    for field in ('failure','success') if kind=='failure' else ('success',):
                        output_id=row[field+'_output_id']
                        expected=attempts[output_id]['generated'] if output_id is not None else None
                        if row[field]!=expected: raise ValueError('Content packet attempt differs from completed output.')
                    if row['adapter_sha256']!=state['completions'][attempts_name]['initial_adapter_identity']['adapter_model.safetensors']: raise ValueError('Content packet adapter identity changed.')
                if origin!=packet['inventories'][kind]['records_sha256']: raise ValueError('Content canonical source inventory changed.')
                reviews=[]; identities_now=[]
                for reviewer in ('reviewer_a','reviewer_b'):
                    manifest=read(base/(reviewer+'_MANIFEST.json'))
                    if manifest.get('did_author_task_examples') is not False or manifest.get('saw_other_reviewer_judgments') is not False:
                        raise ValueError('Content reviewer independence declarations are missing.')
                    if not manifest.get('reviewer_identity') or not manifest.get('review_method'):
                        raise ValueError('Content reviewer identity/method is missing.')
                    if manifest['rubric_sha256']!=frozen_hash(entry,'references/INHERITED_CONTENT_RUBRIC.md',mappings): raise ValueError('Reviewer rubric changed.')
                    identities_now.append(manifest['reviewer_identity'])
                    reviewed=raw_rows(base/(kind+'_'+reviewer+'.jsonl.gz'))
                    canonical_by_id={row['id']:row for row in canonical}
                    if (len(reviewed)!=len(canonical) or len({row['id'] for row in reviewed})!=len(canonical)
                            or {row['id'] for row in reviewed}!=set(canonical_by_id)):
                        raise ValueError('Reviewer packet omits, duplicates or substitutes canonical cases.')
                    for row in reviewed:
                        if row[ORIGIN]!=origin[row['id']] or clean(row)!=canonical_by_id[row['id']]:
                            raise ValueError('Reviewer packet content differs from its canonical source.')
                    judged=rows(base/(kind+'_'+reviewer+'_judgments.jsonl.gz'))
                    for row in judged:
                        if row['packet_row_sha256']!=origin.get(row['id']): raise ValueError('Judgment does not bind original canonical record.')
                    for key in ('packet_files_sha256','judgment_files_sha256'):
                        required=kind+'_'+reviewer+('_judgments' if key=='judgment_files_sha256' else '')+'.jsonl'
                        if required not in manifest.get(key,{}): raise ValueError('Reviewer source-file hash inventory is incomplete.')
                        for filename,digest in manifest[key].items():
                            source='results/content_review/'+base.name+'/'+filename
                            if mappings.get(source,{}).get('original_sha256')!=digest: raise ValueError('Reviewer source file hash changed.')
                    for filename in (kind+'_'+reviewer+'.jsonl',kind+'_'+reviewer+'_judgments.jsonl',reviewer+'_MANIFEST.json'):
                        source=packet_source+filename; provenance[source]=mappings[source]['original_sha256']
                    reviews.append(judged)
                if identities_now[0]==identities_now[1] or (identities and identities!=identities_now): raise ValueError('Content reviewer identities changed or coincide.')
                identities=identities_now
                gates[kind]=function(restore_special_presence(canonical),*reviews,seed)
            seeds.append(dict(seed=seed,passed=all(gate['pass'] for gate in gates.values()),gates=gates,reviewer_identities=identities,
                              provenance_sha256=provenance))
        replayed.append(dict(locked_rank=rank,passed=all(row['passed'] for row in seeds),by_seed=seeds))
    compare(replayed,result['candidates'],'content numerical gates')
    passing=[row['locked_rank'] for row in replayed if row['passed']]
    expected_status='feasibility_established' if passing else 'semantic_content_failed'
    if result['status']!=expected_status or result['selected_locked_rank']!=(min(passing) if passing else None): raise ValueError('Final content numerical outcome differs.')
    return dict(status=expected_status,candidates=len(replayed),scope='Reapplies numerical rules to saved independent judgments; does not make or independently reproduce those judgments.')


def replay(entry,verify_only=False):
    entry=Path(entry).resolve(); manifest,frozen,mappings=verify_transport(entry)
    allocation=audit_allocation(entry,frozen); terminal=read(entry/EVIDENCE/'TERMINAL.json')
    if terminal.get('freeze_sha256') not in (None,manifest['original_freeze_sha256']): raise ValueError('Terminal freeze identity differs.')
    if terminal.get('lock_sha256') and terminal['lock_sha256']!=mappings['results/CONFIRMATION_LOCK.json']['original_sha256']:
        raise ValueError('Terminal confirmation-lock identity differs.')
    if verify_only: return dict(status='transport_verified',allocation=allocation,terminal_status=terminal['status'])
    with tempfile.TemporaryDirectory(prefix='calibration-public-replay-') as temporary:
        work=Path(temporary)/'calibration'; datasets=regenerate(entry,frozen,work); engine=load_engine(work)
        state=replay_stages(entry,work,manifest,frozen,mappings,engine)
        checkpoint_lineage(entry,state,mappings)
        decisions,pairs=replay_decisions(entry,state,engine,mappings,terminal)
        principles=reflection_lineage(entry,work,state,mappings,pairs)
        content=content_replay(entry,work,mappings,state)
        return dict(status='numerical_replay_passed',terminal_status=terminal['status'],allocation=allocation,
                    completed_stages=len(state['completions']),incomplete_stages=sum(row['status']=='incomplete' for row in manifest['stages']),
                    regenerated_canonical_cases=len(datasets['canonical_cases']),generated_outputs=sum(len(rows) for rows in state['outputs'].values()),
                    training_passes=len(state['training']),target_loss_diagnoses=len(state['losses']),**decisions,
                    principle_input_links=principles,content=content,
                    scope='CPU numerical evidence replay, not fresh model replication. Per-token trajectories and unpublished checkpoint contents are not reconstructed. Paired prompt views reuse cases and are not independent samples.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(replay(args.entry,args.verify_only),indent=2))
