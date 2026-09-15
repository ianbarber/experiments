"""Synthetic, CPU-only tests of calibration publication transport and replay."""
from __future__ import annotations
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE if (HERE/'scripts/make_data.py').exists() else HERE.parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


EXPORT=module('calibration_export_under_test',HERE/'export.py')
REPLAY=module('calibration_replay_under_test',HERE/'replay.py')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def lines(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records))


def synthetic_output(row,cohort,adapter):
    decision=row['gold_decision']; reason='A factual reason.'
    return dict(id=row['id'],cohort=cohort,source_file='data/'+cohort+'.jsonl',
                source_id=row.get('source_id',row.get('case_id',row['id'])),draw_index=row.get('draw_index'),
                source_row_sha256=EXPORT.row_sha(row),gold_decision=decision,is_bad=row.get('is_bad',False),
                generated=dict(text=reason+'\n<decision>'+decision+'</decision>',finish_reason='eos',
                               text_with_special_tokens=reason+' <token trace>',token_ids=[31,32,33],
                               unexpected_special_token_ids=[],generated_tokens=3),
                parsed=dict(valid=True,decision=decision,reason=reason,error=None),adapter_sha256=adapter)


def make_source(source):
    source.mkdir()
    for name in ('make_data.py','task.py','task_constants.py','gates.py','util.py','program.py','content_gate.py'):
        path=source/'scripts'/name; path.parent.mkdir(exist_ok=True)
        shutil.copyfile(WORKSPACE/'scripts'/name,path)
    subprocess.run([sys.executable,'-B',str(source/'scripts/make_data.py')],cwd=source,check=True,capture_output=True)
    config=dict(installation_seeds=[1729,2718],gpu_budget_seconds=43200,research_container='private-research-name',service_container='private-service-name')
    write(source/'configs/pilot.json',config)
    for seed in (1729,2718):
        path=source/'inputs'/f'competence_s{seed}'
        path.mkdir(parents=True)
        (path/'adapter_model.safetensors').write_bytes(b'synthetic adapter fixture '+str(seed).encode())
        write(path/'adapter_config.json',dict(base_model_name_or_path='/models/synthetic-test'))
    write(source/'inputs/PROVENANCE.json',{'fixture':'Synthetic test only; no model ran.'})
    frozen=dict(at='synthetic',gpu_budget_seconds=43200,shutdown_reserve_seconds=180,
                files={str(path.relative_to(source)):EXPORT.sha(path) for directory in ('scripts','data','configs','inputs')
                       for path in sorted((source/directory).rglob('*')) if path.is_file()})
    write(source/'FREEZE.json',frozen)
    write(source/'results/allocation.json',dict(started_unix=1000,deadline_unix=44020,hard_allocation_deadline_unix=44200,
          budget_seconds=43200,shutdown_reserve_seconds=180,service_id='secret-service-identity',research_id='secret-research-identity'))
    write(source/'results/allocation_released.json',dict(released_unix=1100,research_running=False))
    write(source/'results/service_restoration.json',dict(health_status=200,research_running=False,
          same_original_container_and_image=True,research_allocation_elapsed_seconds=100,within_allocation_budget=True,
          service_id='secret-service-identity'))
    write(source/'results/TERMINAL.json',dict(status='resource_limited_incomplete',repair_training_launched=False,freeze_sha256=EXPORT.sha(source/'FREEZE.json')))
    helpers=source/'results/publication_tools'; helpers.mkdir()
    for name in ('export.py','replay.py','test_publication.py','test_material_paths.py','README.md'):
        if (HERE/name).exists(): shutil.copyfile(HERE/name,helpers/name)
    engine=REPLAY.load_engine(source)
    adapter={name:EXPORT.sha(source/'inputs/competence_s1729'/name) for name in ('adapter_model.safetensors','adapter_config.json')}
    for name,mode,cohorts,primary in [('s1729_competence_screen','generate',['competence_probe'],'competence_probe'),
                                     ('s1729_baseline_present','diagnose',list(REPLAY.VIEWS),'master_present')]:
        stage=source/'results/stages'/name; stage.mkdir(parents=True)
        primary_rows=REPLAY.rows(source/'data'/(primary+'.jsonl'))
        args=dict(data='/workspace/data/'+primary+'.jsonl',adapter='/workspace/inputs/competence_s1729',seed=1729,
                  rule_variant='present',bad_weight=1.0,loss_only=False,bad_only_diagnostic=False,
                  generation_data=['/workspace/data/'+cohort+'.jsonl' for cohort in cohorts] if mode=='diagnose' else None)
        inputs={cohort:REPLAY.rows(source/'data'/(cohort+'.jsonl')) for cohort in cohorts}
        generated=[synthetic_output(row,cohort,adapter['adapter_model.safetensors']) for cohort in cohorts for row in inputs[cohort]]
        started=dict(mode=mode,check_only=False,args=args,source_examples=len(primary_rows),
                     source_sha256={path:sha for path,sha in frozen['files'].items() if path.startswith('scripts/')},
                     config_sha256=frozen['files']['configs/pilot.json'],initial_adapter_identity=adapter,
                     generation_files={cohort:'data/'+cohort+'.jsonl' for cohort in cohorts},
                     input_files_sha256={'/workspace/data/'+cohort+'.jsonl':EXPORT.sha(source/'data'/(cohort+'.jsonl')) for cohort in set(cohorts+[primary])},
                     data_sha256=EXPORT.sha(source/'data'/(primary+'.jsonl')))
        write(stage/'started.json',started); lines(stage/'outputs.jsonl',generated)
        if mode=='diagnose':
            losses=[]
            for row in primary_rows:
                groups=['all','bad','bad_category:'+row['authored_error_category']] if row['is_bad'] else ['all','good','good_'+row['gold_decision'].lower()]
                losses.append(dict(id=row['id'],source_row_sha256=EXPORT.row_sha(row),groups=groups,target_tokens=3,
                                   target_loss_sum=6.0,mean_target_nll=2.0,loss_weight=1.0))
            lines(stage/'target_losses.jsonl',losses)
            write(stage/'target_loss_summary.json',dict(examples=len(primary_rows),groups=REPLAY.loss_groups(primary_rows,losses,1.0)))
        completed=dict(started,status='complete',outputs=len(generated),stage_elapsed_s=1,
                       artifacts_sha256={path.name:EXPORT.sha(path) for path in stage.iterdir()})
        write(stage/'COMPLETED.json',completed)
        scored={cohort:REPLAY.score(inputs[cohort],[row for row in generated if row['cohort']==cohort]) for cohort in cohorts}
        write(stage/'recomputed_scores.json',scored if mode=='diagnose' else scored[cohorts[0]])
        if mode=='generate': write(source/'results/gates'/(name+'.json'),engine.competence_screen(scored['competence_probe']))
    partial=source/'results/stages/s1729_present_w0333_epoch1'; partial.mkdir()
    write(partial/'started.json',dict(mode='train',check_only=False))
    (partial/'training.jsonl').write_bytes(b'{"step":1,"loss":2.0}\n{"step":')
    return module('fixture_exporter',helpers/'export.py')


def recipe_catalog():
    return [dict(index=index,id=variant+'_'+label,rule_variant=variant,bad_weight=weight)
            for index,(variant,label,weight) in enumerate((variant,label,weight)
                for variant in ('present','omitted') for label,weight in (('w0333',1/3),('w1',1.0),('w3',3.0)))]


def append_training(source,seed,recipe,epoch):
    """One serialized synthetic pass, including exact optimizer/checkpoint lineage."""
    pure=recipe['id']=='pure_bad_control'; weight=recipe['bad_weight']
    primary='bad_only_diagnostic_present' if pure else 'master_'+recipe['rule_variant']
    cases=REPLAY.rows(source/'data'/(primary+'.jsonl'))
    name=f's{seed}_{recipe["id"]}_epoch{epoch}'; stage=source/'results/stages'/name
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    parent=f'inputs/competence_s{seed}' if epoch==1 else f'results/stages/s{seed}_{recipe["id"]}_epoch{epoch-1}/adapter'
    initial={file:EXPORT.sha(source/parent/file) for file in ('adapter_model.safetensors','adapter_config.json')}
    optimizer=None if epoch==1 else parent.removesuffix('/adapter')+'/optimizer.pt'
    frozen=REPLAY.read(source/'FREEZE.json')
    args=dict(data='/workspace/data/'+primary+'.jsonl',adapter='/workspace/'+parent,seed=seed,
              optimizer='/workspace/'+optimizer if optimizer else None,epoch=epoch,
              rule_variant=recipe['rule_variant'],bad_weight=weight,bad_only_diagnostic=pure,
              batch_size=16,effective_batch=16,lr=3e-4 if pure else 1e-4)
    started=dict(mode='train',check_only=False,args=args,source_examples=len(cases),epoch=epoch,
            source_sha256={path:digest for path,digest in frozen['files'].items() if path.startswith('scripts/')},
            config_sha256=frozen['files']['configs/pilot.json'],initial_adapter_identity=initial,
            input_files_sha256={'/workspace/data/'+primary+'.jsonl':EXPORT.sha(source/'data'/(primary+'.jsonl'))},
            data_sha256=EXPORT.sha(source/'data'/(primary+'.jsonl')),generation_files={},
            optimizer_resumed=epoch>1,optimizer_previous_steps=(epoch-1)*len(cases)//16,
            initial_optimizer_sha256=EXPORT.sha(source/optimizer) if optimizer else None,
            initial_trainable_parameter_sha256=initial['adapter_model.safetensors'])
    write(stage/'started.json',started)
    if pure: ordered=cases
    else:
        pools=[[row for row in cases if row['is_bad']],
               [row for row in cases if not row['is_bad'] and row['gold_decision']=='REPORT'],
               [row for row in cases if row['gold_decision']=='CLEAR']]
        ordered=[row for batch in range(64) for pool,count in zip(pools,(8,4,4)) for row in pool[batch*count:(batch+1)*count]]
    exposure=[dict(id=row['id'],source_row_sha256=EXPORT.row_sha(row),weight=weight if row['is_bad'] else 1.0,target_tokens=3) for row in ordered]
    lines(stage/'exposure.jsonl',exposure)
    groups={'all':dict(examples=len(cases),target_tokens=3*len(cases),mean_target_nll=2.0)}
    logs=[dict(step=index+1,epoch=epoch,ids=[row['id'] for row in exposure[index*16:(index+1)*16]],
            weighted_target_token_denominator=sum(row['weight']*row['target_tokens'] for row in exposure[index*16:(index+1)*16]),
            loss=2.0,grad_norm_after_clip=.5,cumulative_groups_at_processing=groups)
          for index in range(len(cases)//16)]
    lines(stage/'training.jsonl',logs)
    (stage/'adapter').mkdir(); (stage/'adapter/adapter_model.safetensors').write_bytes(('Synthetic '+name).encode())
    write(stage/'adapter/adapter_config.json',dict(base_model_name_or_path='/models/synthetic-test'))
    (stage/'optimizer.pt').write_bytes(('Synthetic optimizer '+name).encode())
    final={file:EXPORT.sha(stage/'adapter'/file) for file in initial}
    write(stage/'COMPLETED.json',dict(started,status='complete',stage_elapsed_s=1,
          actual_exposure_sha256=EXPORT.row_sha(exposure),steps=len(logs),cumulative_optimizer_steps=epoch*len(logs),
          groups_at_processing=groups,adapter_identity=final,final_trainable_parameter_sha256=final['adapter_model.safetensors'],
          artifacts_sha256={str(path.relative_to(stage)):EXPORT.sha(path) for path in stage.rglob('*') if path.is_file()}))


def append_diagnosis(source,seed,recipe=None,epoch=None,*,passing=False):
    """Generate controlled scalar losses and exact case/cohort output coverage."""
    baseline=recipe is None; pure=not baseline and recipe['id']=='pure_bad_control'
    variant='present' if baseline else recipe['rule_variant']; weight=1.0 if baseline else recipe['bad_weight']
    primary='bad_only_diagnostic_present' if pure else 'master_'+variant
    name=f's{seed}_baseline_present' if baseline else f's{seed}_{recipe["id"]}_diagnose{epoch}'
    adapter=f'inputs/competence_s{seed}' if baseline else f'results/stages/s{seed}_{recipe["id"]}_epoch{epoch}/adapter'
    stage=source/'results/stages'/name; stage.mkdir(parents=True)
    identity={file:EXPORT.sha(source/adapter/file) for file in ('adapter_model.safetensors','adapter_config.json')}
    frozen=REPLAY.read(source/'FREEZE.json'); cases=REPLAY.rows(source/'data'/(primary+'.jsonl'))
    args=dict(data='/workspace/data/'+primary+'.jsonl',adapter='/workspace/'+adapter,seed=seed,
              rule_variant=variant,bad_weight=weight,bad_only_diagnostic=pure,loss_only=False,
              generation_data=['/workspace/data/'+view+'.jsonl' for view in REPLAY.VIEWS],sample=False)
    started=dict(mode='diagnose',check_only=False,args=args,source_examples=len(cases),
          source_sha256={path:digest for path,digest in frozen['files'].items() if path.startswith('scripts/')},
          config_sha256=frozen['files']['configs/pilot.json'],initial_adapter_identity=identity,
          input_files_sha256={'/workspace/data/'+view+'.jsonl':EXPORT.sha(source/'data'/(view+'.jsonl')) for view in set(REPLAY.VIEWS+(primary,))},
          data_sha256=EXPORT.sha(source/'data'/(primary+'.jsonl')),
          generation_files={view:'data/'+view+'.jsonl' for view in REPLAY.VIEWS})
    write(stage/'started.json',started); output=[]; scored={}
    for view in REPLAY.VIEWS:
        inputs=REPLAY.rows(source/'data'/(view+'.jsonl')); cohort=[]; errors=0
        for case in inputs:
            row=synthetic_output(case,view,identity['adapter_model.safetensors'])
            if passing and case['stratum']=='eliciting_report' and errors<32:
                row['generated']['text']='A synthetic mistaken reason.\n<decision>CLEAR</decision>'
                row['parsed']=REPLAY.parse(row['generated']); errors+=1
            cohort.append(row)
        scored[view]=REPLAY.score(inputs,cohort); output.extend(cohort)
    lines(stage/'outputs.jsonl',output)
    losses=[dict(id=row['id'],source_row_sha256=EXPORT.row_sha(row),
                groups=['all','bad','bad_category:'+row['authored_error_category']] if row['is_bad'] else ['all','good','good_'+row['gold_decision'].lower()],
                target_tokens=3,target_loss_sum=6.0,mean_target_nll=2.0,loss_weight=weight if row['is_bad'] else 1.0) for row in cases]
    lines(stage/'target_losses.jsonl',losses)
    write(stage/'target_loss_summary.json',dict(examples=len(cases),groups=REPLAY.loss_groups(cases,losses,weight)))
    write(stage/'COMPLETED.json',dict(started,status='complete',outputs=len(output),stage_elapsed_s=1,
          artifacts_sha256={path.name:EXPORT.sha(path) for path in stage.iterdir()}))
    write(stage/'recomputed_scores.json',scored)
    if baseline: return None
    engine=REPLAY.load_engine(source)
    before=REPLAY.read(source/f'results/stages/s{seed}_baseline_present/recomputed_scores.json')
    views={view:engine.induction_gate(before[view],scored[view]) for view in REPLAY.VIEWS[2:]}
    candidate=dict(engine.candidate_summary(recipe,epoch,views),seed=seed,adapter=adapter,
             adapter_files={adapter+'/'+file:digest for file,digest in identity.items()},diagnosis='results/stages/'+name,bad_only_diagnostic=pure)
    write(stage/'candidate.json',candidate)
    for view,gate in views.items(): write(source/'results/gates'/(name+'_'+view+'.json'),gate)
    return candidate


def append_grid(source,passing_recipes=None,second_seed_pass=True,include_control=True):
    """Append all 18 initial candidates, exact shortlist and three replications.

    None makes all six recipes pass; an empty iterable makes every recipe fail.
    Pure-bad control is included when required unless a negative test disables it.
    """
    recipes=recipe_catalog(); passing=set(row['id'] for row in recipes) if passing_recipes is None else set(passing_recipes)
    append_diagnosis(source,2718)
    grid=[]
    for recipe in recipes:
        for epoch in (1,2,3,4):
            append_training(source,1729,recipe,epoch)
            if epoch in (1,2,4): grid.append(append_diagnosis(source,1729,recipe,epoch,passing=recipe['id'] in passing))
    write(source/'results/INITIAL_GRID.json',grid)
    if not passing and include_control:
        pure=dict(index=6,id='pure_bad_control',rule_variant='present',bad_weight=1.0); control=[]
        for epoch in (1,2,3,4):
            append_training(source,1729,pure,epoch)
            if epoch in (1,4): control.append(append_diagnosis(source,1729,pure,epoch,passing=True))
        write(source/'results/ACQUISITION_CONTROL.json',control)
    selected=REPLAY.load_engine(source).shortlist(grid)
    write(source/'results/SHORTLIST_LOCK.json',dict(at='Synthetic selection before replication',candidates=selected,
          scope='Three distinct recipes and exact doses fixed before seed2718 induction.'))
    replicated=[]
    for rank,candidate in enumerate(selected,1):
        recipe=candidate['recipe']; epoch=candidate['epoch']
        for dose in range(1,epoch+1): append_training(source,2718,recipe,dose)
        second=append_diagnosis(source,2718,recipe,epoch,passing=second_seed_pass and recipe['id'] in passing)
        replicated.append(dict(shortlist_rank=rank,recipe=recipe,epoch=epoch,
                by_seed={'1729':candidate,'2718':second},common_greedy_pass=candidate['greedy_pass'] and second['greedy_pass']))
    write(source/'results/REPLICATIONS.json',replicated)
    terminal=REPLAY.read(source/'results/TERMINAL.json')
    terminal['status']='resource_limited_incomplete' if any(row['common_greedy_pass'] for row in replicated) else 'replicated_development_failed'
    write(source/'results/TERMINAL.json',terminal)
    return dict(grid=grid,shortlist=selected,replications=replicated)


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='calibration-publication-tests-')
        cls.base=Path(cls.temporary.name); cls.source=cls.base/'source'; cls.draft=cls.base/'draft'
        cls.exporter=make_source(cls.source)
        cls.export_result=cls.exporter.export(cls.source,cls.draft)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def copy_draft(self,name):
        path=self.base/name
        shutil.copytree(self.draft,path)
        self.addCleanup(shutil.rmtree,path)
        return path

    def rewrite_projection(self,entry,source,transform):
        transport_path=entry/REPLAY.EVIDENCE/'TRANSPORT.json'; transport=REPLAY.read(transport_path)
        record=next(row for row in transport['records'] if row['original_path']==source)
        path=entry/record['public_path']; values=REPLAY.raw_rows(path); values=transform(values)
        path.write_bytes(gzip.compress(''.join(json.dumps(row)+'\n' for row in values).encode(),mtime=0))
        record['public_sha256']=REPLAY.sha(path); record['rows']=len(values)
        transport['public_artifacts'][record['public_path']]=REPLAY.sha(path); write(transport_path,transport)

    def test_synthetic_partial_run_replays_completed_evidence_only(self):
        result=self.export_result['numerical_replay']
        self.assertEqual(result['status'],'numerical_replay_passed')
        self.assertEqual(result['completed_stages'],2)
        self.assertEqual(result['incomplete_stages'],1)
        self.assertEqual(result['regenerated_canonical_cases'],3584)
        self.assertEqual(result['generated_outputs'],896)
        self.assertEqual(result['target_loss_diagnoses'],1)
        self.assertEqual(result['grid_candidates'],0)
        self.assertEqual(result['terminal_status'],'resource_limited_incomplete')

    def test_projection_preserves_text_scalars_and_omits_operational_and_token_details(self):
        public=self.draft/REPLAY.EVIDENCE/'stages/s1729_baseline_present/outputs.jsonl.gz'
        projected=REPLAY.raw_rows(public)
        original=REPLAY.raw_rows(self.source/'results/stages/s1729_baseline_present/outputs.jsonl')
        for before,after in zip(original,projected):
            self.assertEqual(before['generated']['text'],after['generated']['text'])
            self.assertEqual(before['generated']['generated_tokens'],after['generated']['generated_tokens'])
            self.assertEqual(after[REPLAY.ORIGIN],EXPORT.row_sha(before))
            self.assertNotIn('token_ids',after['generated'])
            self.assertNotIn('text_with_special_tokens',after['generated'])
            self.assertIs(after['generated']['unexpected_special_token'],False)
        allocation=REPLAY.read(self.draft/REPLAY.EVIDENCE/'allocation.json')
        self.assertNotIn('service_id',allocation); self.assertNotIn('research_id',allocation)
        config=REPLAY.read(self.draft/REPLAY.CODE/'configs/pilot.json')
        self.assertEqual(config['research_container'],'isolated-research-container')
        self.assertFalse(list(self.draft.rglob('*.safetensors')))
        self.assertFalse(list((self.draft/REPLAY.CODE/'data').glob('*.jsonl')))

    def test_supporting_review_links_resolve_to_single_transport_bound_analysis_script(self):
        with tempfile.TemporaryDirectory(dir=self.base) as temporary:
            source=Path(temporary)/'source'; exporter=make_source(source)
            directory=source/'results/analysis_tools'; directory.mkdir()
            script=directory/'audit_closed_initial_grid.py'; script.write_text('"""Synthetic audit helper; no model calls."""\n')
            review=directory/'INITIAL_GRID_EXECUTION_REVIEW.md'
            original=('[bounded audit](audit_closed_initial_grid.py)\n'
                      '[with anchor](<audit_closed_initial_grid.py#inspect> "source")\n'
                      '[snapshot](summary.json)\n[external](https://example.test/audit.py)\n')
            review.write_text(original); before=EXPORT.sha(review)
            entry=Path(temporary)/'draft'; result=exporter.export(source,entry,supporting=['results/analysis_tools/INITIAL_GRID_EXECUTION_REVIEW.md'])
            self.assertEqual(result['numerical_replay']['status'],'numerical_replay_passed')
            transport=REPLAY.read(entry/REPLAY.EVIDENCE/'TRANSPORT.json')
            mapping=next(row for row in transport['records'] if row['original_path']=='results/analysis_tools/INITIAL_GRID_EXECUTION_REVIEW.md')
            target=next(row for row in transport['records'] if row['original_path']=='results/analysis_tools/audit_closed_initial_grid.py')
            self.assertEqual(mapping['original_sha256'],before); self.assertEqual(EXPORT.sha(review),before)
            self.assertEqual(review.read_text(),original)
            self.assertEqual(mapping['mode'],'sanitized_text_with_analysis_script_link_projection')
            public=entry/mapping['public_path']; text=public.read_text(); links=mapping['analysis_script_link_projection']
            self.assertEqual(len(links),2)
            for link in links:
                self.assertEqual((public.parent/link['public_destination'].split('#')[0]).resolve(),(entry/target['public_path']).resolve())
                self.assertEqual(link['target_original_sha256'],EXPORT.sha(script))
                self.assertEqual(link['target_public_sha256'],EXPORT.sha(entry/target['public_path']))
                self.assertIn(link['public_destination'],text)
            self.assertIn('[snapshot](summary.json)',text); self.assertIn('[external](https://example.test/audit.py)',text)
            self.assertEqual(len(list(entry.rglob('audit_closed_initial_grid.py'))),1)

    def test_supporting_analysis_link_rejects_missing_or_escaping_source(self):
        with tempfile.TemporaryDirectory(dir=self.base) as temporary:
            source=Path(temporary)/'source'; exporter=make_source(source)
            directory=source/'results/analysis_tools'; directory.mkdir()
            review=directory/'REVIEW.md'; review.write_text('[audit](absent.py)\n')
            with self.assertRaisesRegex(ValueError,'missing or escapes'):
                exporter.export(source,Path(temporary)/'missing',supporting=['results/analysis_tools/REVIEW.md'])
            outside=Path(temporary)/'outside.py'; outside.write_text('# Synthetic external target.\n')
            (directory/'absent.py').symlink_to(outside)
            with self.assertRaisesRegex(ValueError,'missing or escapes'):
                exporter.export(source,Path(temporary)/'escaping',supporting=['results/analysis_tools/REVIEW.md'])
            self.assertEqual(review.read_text(),'[audit](absent.py)\n')

    def test_privacy_redaction_refuses_to_change_scientific_text(self):
        projection=EXPORT.Projection(self.source)
        private='/home/'+'privateperson/model-cache'
        self.assertNotIn('privateperson',projection.text(private))
        with self.assertRaises(ValueError): projection.value(dict(generated=dict(text='The case is '+private,finish_reason='eos')))
        with self.assertRaises(ValueError): projection.value(dict(role='assistant',content=private))
        result=projection.value(dict(generated=dict(text='Safe exact text.',finish_reason='eos',unexpected_special_token_ids=[22])))
        self.assertTrue(result['generated']['unexpected_special_token'])
        self.assertEqual(REPLAY.parse(result['generated'])['error'],'unexpected_special_token')

    def test_nested_special_presence_rejects_outputs_and_content_in_frozen_contracts(self):
        generated=dict(text='A prospective factual principle.',finish_reason='eos',unexpected_special_token=True)
        record=dict(failure=dict(generated),success=dict(generated),reflection=dict(generated))
        restored=REPLAY.restore_special_presence(record)
        for name in ('failure','success','reflection'):
            self.assertEqual(restored[name]['unexpected_special_token_ids'],[True])
            self.assertNotIn('unexpected_special_token_ids',record[name])
        engine=REPLAY.load_engine(self.source)
        observed=engine.parse_generation(dict(restored['failure'],text='A reason. <decision>REPORT</decision>'))
        self.assertFalse(observed['valid'])
        self.assertEqual(observed['error'],'unexpected_special_token')
        content=module('content_gate_special_presence_test',self.source/'scripts/content_gate.py')
        self.assertFalse(content.automatic_reflection_valid(restored))
        record['reflection']['unexpected_special_token']=False
        self.assertTrue(content.automatic_reflection_valid(REPLAY.restore_special_presence(record)))
        with self.assertRaises(ValueError):
            REPLAY.restore_special_presence(dict(generated,unexpected_special_token_ids=[123]))
        with self.assertRaises(ValueError):
            REPLAY.restore_special_presence(dict(generated,unexpected_special_token='false'))

    def test_duplicate_case_ids_across_diagnostic_cohorts_are_legal(self):
        projected=REPLAY.raw_rows(self.draft/REPLAY.EVIDENCE/'stages/s1729_baseline_present/outputs.jsonl.gz')
        self.assertLess(len({row['id'] for row in projected}),len(projected))
        for cohort in REPLAY.VIEWS:
            selected=[REPLAY.clean(row) for row in projected if row['cohort']==cohort]
            cases=REPLAY.rows(self.source/'data'/(cohort+'.jsonl'))
            self.assertEqual(len(REPLAY.score(cases,selected)),len(cases))
        cohort='seen_probe_present'; selected=[REPLAY.clean(row) for row in projected if row['cohort']==cohort]
        with self.assertRaises(ValueError): REPLAY.score(REPLAY.rows(self.source/'data'/(cohort+'.jsonl')),selected[:-1]+[selected[0]])

    def test_missing_output_rejected_even_if_public_file_hash_is_rebound(self):
        entry=self.copy_draft('missing-output')
        self.rewrite_projection(entry,'results/stages/s1729_baseline_present/outputs.jsonl',lambda rows:rows[:-1])
        with self.assertRaisesRegex(ValueError,'Missing, substituted or reordered'):
            REPLAY.replay(entry)

    def test_changed_source_hash_or_saved_parser_is_rejected(self):
        cases=REPLAY.rows(self.source/'data/competence_probe.jsonl')
        output=REPLAY.rows(self.draft/REPLAY.EVIDENCE/'stages/s1729_competence_screen/outputs.jsonl.gz')
        altered=copy.deepcopy(output); altered[0]['source_row_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'source-row hash'): REPLAY.score(cases,altered)
        altered=copy.deepcopy(output); altered[0]['parsed']['reason']='Invented saved reason.'
        with self.assertRaisesRegex(ValueError,'parser'): REPLAY.score(cases,altered)

    def test_original_and_public_hashes_are_both_checked(self):
        entry=self.copy_draft('changed-original')
        path=entry/REPLAY.EVIDENCE/'TRANSPORT.json'; transport=REPLAY.read(path)
        transport['frozen_sources']['scripts/task.py']['original_sha256']='0'*64; write(path,transport)
        with self.assertRaisesRegex(ValueError,'original hash'): REPLAY.verify_transport(entry)
        entry=self.copy_draft('changed-public')
        path=entry/REPLAY.CODE/'scripts/task.py'; path.write_text(path.read_text()+'\n# changed\n')
        with self.assertRaises(ValueError): REPLAY.verify_transport(entry)

    def test_unfinished_trailing_fragment_is_disclosed_not_counted(self):
        transport=REPLAY.read(self.draft/REPLAY.EVIDENCE/'TRANSPORT.json')
        record=next(row for row in transport['records'] if row['original_path'].endswith('present_w0333_epoch1/training.jsonl'))
        self.assertEqual(record['rows'],1)
        self.assertEqual(record['omitted_trailing_fragment']['bytes'],8)
        self.assertEqual(len(REPLAY.raw_rows(self.draft/record['public_path'])),1)
        with self.assertRaises(ValueError): EXPORT.read_complete_lines(self.source/record['original_path'],False)

    def test_interruption_after_diagnostic_gate_before_candidate_receipt(self):
        with tempfile.TemporaryDirectory(dir=self.base) as temporary:
            source=Path(temporary)/'source'; exporter=make_source(source); recipe=recipe_catalog()[0]
            append_training(source,1729,recipe,1)
            append_diagnosis(source,1729,recipe,1,passing=True)
            (source/'results/stages/s1729_present_w0333_diagnose1/candidate.json').unlink()
            result=exporter.export(source,Path(temporary)/'draft')['numerical_replay']
            self.assertEqual(result['gates'],3)
            self.assertEqual(result['grid_candidates'],0)
            self.assertEqual(result['target_loss_diagnoses'],2)

    def test_target_nll_replay_respects_token_denominators_and_bad_weights(self):
        master=REPLAY.rows(self.source/'data/master_present.jsonl')
        cases=[next(row for row in master if row['is_bad']),
               next(row for row in master if not row['is_bad'] and row['gold_decision']=='REPORT'),
               next(row for row in master if row['gold_decision']=='CLEAR')]
        records=[]
        for index,row in enumerate(cases):
            count=(2,4,6)[index]; loss=(4.0,12.0,24.0)[index]
            groups=['all','bad','bad_category:'+row['authored_error_category']] if row['is_bad'] else ['all','good','good_'+row['gold_decision'].lower()]
            records.append(dict(id=row['id'],source_row_sha256=EXPORT.row_sha(row),groups=groups,
                                target_tokens=count,target_loss_sum=loss,mean_target_nll=loss/count,
                                loss_weight=3.0 if row['is_bad'] else 1.0))
        groups=REPLAY.loss_groups(cases,records,3.0)
        self.assertEqual(groups['bad']['weighted_target_tokens'],6)
        self.assertEqual(groups['bad']['mean_target_nll'],2)
        self.assertAlmostEqual(groups['good']['mean_target_nll'],3.6)
        self.assertAlmostEqual(groups['all']['mean_target_nll'],40/12)
        self.assertEqual(groups['all']['weighted_mean_target_nll'],3)
        changed=copy.deepcopy(records); changed[0]['loss_weight']=1.0
        with self.assertRaises(ValueError): REPLAY.loss_groups(cases,changed,3.0)
        with self.assertRaises(ValueError): REPLAY.loss_groups(cases,records[:-1],3.0)

    def test_no_export_before_healthy_terminal_or_into_existing_checkout(self):
        with self.assertRaises(ValueError): self.exporter.export(self.source,self.draft)
        parent=self.base/'checkout'; parent.mkdir(); (parent/'.git').mkdir()
        self.addCleanup(shutil.rmtree,parent)
        with self.assertRaisesRegex(ValueError,'Git checkout'): self.exporter.export(self.source,parent/'new')
        path=self.source/'results/service_restoration.json'; original=path.read_bytes()
        try:
            value=REPLAY.read(path); value['health_status']=503; write(path,value)
            with self.assertRaises(ValueError): self.exporter.export(self.source,self.base/'unhealthy-export')
        finally: path.write_bytes(original)
        path=self.source/'results/TERMINAL.json'; held=path.with_suffix('.held'); path.rename(held)
        try:
            with self.assertRaises(OSError): self.exporter.export(self.source,self.base/'active-export')
        finally: held.rename(path)

    def test_terminal_branches_require_their_actual_preceding_stages(self):
        def check(status,**changes):
            state=dict(grid=[{}]*18,replicated=[{}]*3,common=[],pairing=[],eligible=[],lock=None,confirmations=[],passing=[],materials=[])
            state.update(changes)
            return REPLAY.verify_terminal_branch(dict(status=status,repair_training_launched=False),**state)
        check('replicated_development_failed')
        with self.assertRaises(ValueError): check('replicated_development_failed',grid=[])
        with self.assertRaises(ValueError): check('replicated_development_failed',common=[{}])
        check('pairing_development_failed',common=[{}],pairing=[{}])
        with self.assertRaises(ValueError): check('pairing_development_failed',common=[{}],pairing=[{}],eligible=[{}])
        check('untouched_confirmation_failed',lock={'candidates':[{}]},confirmations=[{}])
        with self.assertRaises(ValueError): check('untouched_confirmation_failed')
        check('awaiting_content_review',lock={'candidates':[{}]},confirmations=[{}],passing=[{}],materials=[{'both_structural_pass':True}])
        with self.assertRaises(ValueError): check('awaiting_content_review',lock={'candidates':[{}]},confirmations=[{}],passing=[{}],materials=[{'both_structural_pass':False}])
        check('material_structural_failed',lock={'candidates':[{}]},confirmations=[{}],passing=[{}],materials=[{'both_structural_pass':False}])
        check('resource_limited_incomplete',grid=[],replicated=[])
        with self.assertRaises(ValueError): check('unrecognized-success')


class GridPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='calibration-grid-publication-tests-')
        cls.base=Path(cls.temporary.name); cls.fixtures={}
        for kind,passing in (('initial_pass',None),('all_fail',())):
            source=cls.base/(kind+'-source'); destination=cls.base/(kind+'-draft')
            exporter=make_source(source)
            metadata=append_grid(source,passing_recipes=passing,second_seed_pass=False)
            result=exporter.export(source,destination)
            transport,frozen,mappings=REPLAY.verify_transport(destination)
            work=cls.base/(kind+'-regenerated'); REPLAY.regenerate(destination,frozen,work)
            engine=REPLAY.load_engine(work)
            state=REPLAY.replay_stages(destination,work,transport,frozen,mappings,engine)
            cls.fixtures[kind]=dict(source=source,entry=destination,metadata=metadata,result=result,
                                    mappings=mappings,engine=engine,state=state)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def altered_entry(self,kind='initial_pass'):
        fixture=self.fixtures[kind]; path=Path(tempfile.mkdtemp(dir=self.base))/'entry'
        shutil.copytree(fixture['entry'],path); self.addCleanup(shutil.rmtree,path.parent)
        return fixture,path

    def check_decisions(self,fixture,entry,state=None):
        return REPLAY.replay_decisions(entry,state or fixture['state'],fixture['engine'],fixture['mappings'],
                                      REPLAY.read(entry/REPLAY.EVIDENCE/'TERMINAL.json'))

    def test_complete_grid_and_deterministic_distinct_recipe_ties(self):
        fixture=self.fixtures['initial_pass']; result=fixture['result']['numerical_replay']
        self.assertEqual(result['terminal_status'],'replicated_development_failed')
        self.assertEqual(result['grid_candidates'],18); self.assertEqual(result['shortlisted'],3)
        self.assertEqual(result['replications'],3); self.assertEqual(result['training_passes'],27)
        self.assertEqual(result['acquisition_control_doses'],0)
        selected=fixture['metadata']['shortlist']
        self.assertEqual([(row['recipe']['id'],row['epoch']) for row in selected],
                         [('present_w0333',1),('present_w1',1),('present_w3',1)])
        lock=REPLAY.read(fixture['entry']/REPLAY.EVIDENCE/'SHORTLIST_LOCK.json')
        self.assertEqual(set(lock),{'at','candidates','scope'})

    def test_all_failing_grid_keeps_two_mandatory_acquisition_doses(self):
        result=self.fixtures['all_fail']['result']['numerical_replay']
        self.assertEqual(result['acquisition_control_doses'],2)
        self.assertEqual(result['training_passes'],31)
        self.assertEqual(result['terminal_status'],'replicated_development_failed')
        fixture,entry=self.altered_entry('all_fail')
        (entry/REPLAY.EVIDENCE/'ACQUISITION_CONTROL.json').unlink()
        with self.assertRaisesRegex(ValueError,'mandatory pure-bad acquisition control'):
            self.check_decisions(fixture,entry)

    def test_control_dose_and_shortlist_substitution_are_rejected(self):
        fixture,entry=self.altered_entry('all_fail'); path=entry/REPLAY.EVIDENCE/'ACQUISITION_CONTROL.json'
        write(path,list(reversed(REPLAY.read(path))))
        with self.assertRaisesRegex(ValueError,'dose inventory'): self.check_decisions(fixture,entry)
        fixture,entry=self.altered_entry(); path=entry/REPLAY.EVIDENCE/'SHORTLIST_LOCK.json'; lock=REPLAY.read(path)
        lock['candidates'][1]=copy.deepcopy(lock['candidates'][0]); write(path,lock)
        with self.assertRaises(ValueError): self.check_decisions(fixture,entry)
        for wrong in (None,'0'*64):
            fixture,entry=self.altered_entry(); path=entry/REPLAY.EVIDENCE/'SHORTLIST_LOCK.json'; lock=REPLAY.read(path)
            lock['freeze_sha256']=wrong; write(path,lock)
            with self.subTest(supplied_freeze=wrong), self.assertRaisesRegex(ValueError,'another freeze'):
                self.check_decisions(fixture,entry)

    def test_grid_requires_all_unique_recipe_dose_cells(self):
        fixture,entry=self.altered_entry(); path=entry/REPLAY.EVIDENCE/'INITIAL_GRID.json'; grid=REPLAY.read(path)
        grid[-1]=copy.deepcopy(grid[0]); write(path,grid)
        with self.assertRaisesRegex(ValueError,'initial-grid membership'): self.check_decisions(fixture,entry)
        grid=copy.deepcopy(fixture['metadata']['grid']); write(path,grid[:-1])
        with self.assertRaisesRegex(ValueError,'full six-recipe'): self.check_decisions(fixture,entry)

    def test_training_recipe_arguments_cannot_disagree_with_stage_name(self):
        fixture=self.fixtures['initial_pass']; original=fixture['state']['completions']['s1729_present_w0333_epoch1']
        for key,value in [('bad_weight',3.0),('data','data/master_omitted.jsonl'),('rule_variant','omitted'),
                          ('bad_only_diagnostic',True),('lr',3e-4),('batch_size',8),('effective_batch',32),('seed',2718),('epoch',2)]:
            with self.subTest(key=key):
                completed=copy.deepcopy(original); completed['args'][key]=value
                with self.assertRaisesRegex(ValueError,'Training arguments'):
                    REPLAY.bind_training_recipes(fixture['entry'],dict(completions={'s1729_present_w0333_epoch1':completed}))

    def test_second_seed_cannot_train_beyond_exact_shortlisted_dose(self):
        fixture=self.fixtures['initial_pass']; completed=copy.deepcopy(fixture['state']['completions']['s2718_present_w0333_epoch1'])
        completed['epoch']=completed['args']['epoch']=2
        with self.assertRaisesRegex(ValueError,'exact shortlisted trajectory'):
            REPLAY.bind_training_recipes(fixture['entry'],dict(completions={'s2718_present_w0333_epoch2':completed}))

    def test_replication_cannot_substitute_a_valid_but_unselected_candidate(self):
        fixture,entry=self.altered_entry(); path=entry/REPLAY.EVIDENCE/'REPLICATIONS.json'; values=REPLAY.read(path)
        values[0]['by_seed']['2718']=copy.deepcopy(values[1]['by_seed']['2718']); write(path,values)
        with self.assertRaisesRegex(ValueError,'Replicated checkpoint differs'): self.check_decisions(fixture,entry)


if __name__=='__main__': unittest.main()
