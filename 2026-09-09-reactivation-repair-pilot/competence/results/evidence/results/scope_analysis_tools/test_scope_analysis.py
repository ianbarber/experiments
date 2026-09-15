"""CPU-only authored synthetic contracts; no real outcomes or model invocation."""
import ast
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import fixtures
analysis = fixtures.analysis


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def finalizer(root):
    return load('_fixture_scope_finalizer', root/'results/scope_tools/finalize_competence.py')


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='SYNTHETIC-scope-analysis-')
        cls.roots = {}
        cls.results = {}
        for route in ('guard', 'no_selection', 'validation_fail'):
            root = fixtures.create_fixture(cls.workspace.name, route)
            cls.roots[route] = root
            cls.results[route] = finalizer(root).finalize(root, write=True)
        cls.guard_analysis, cls.guard_completion = analysis.analyze(cls.roots['guard'], cls.roots['guard']/'analysis')

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def copy_guard(self, directory, remove_terminal=True):
        root=Path(directory)/'guard-copy'
        shutil.copytree(self.roots['guard'],root)
        if remove_terminal: (root/'results/COMPETENCE_SCOPE_COMPLETED.json').unlink()
        return root

    def test_both_terminal_routes_and_actual_gate_reasons(self):
        for name, result in self.results.items():
            with self.subTest(route=name):
                root=self.roots[name]
                self.assertTrue(result['synthetic_fixture'])
                self.assertEqual(len(result['completed_stages']),58)
                self.assertEqual(result['verification']['reparsed_responses'],8064)
                self.assertEqual(result['verification']['adapter_files_rehashed'],24)
                self.assertEqual(result['selected_installations'],{})
                self.assertEqual(result['numerical_validation_passed'],name=='guard')
                self.assertEqual(result['boundary_guard_reached'],name=='guard')
                self.assertEqual((root/'results/PROGRAM_COMPLETED.json').exists(),name!='guard')
                self.assertEqual(result['original_terminal']['route'], 'administrative_guard_reached' if name=='guard' else 'original_numerical_stop')
                self.assertIn('results/SELECTED_RECIPE.json',result['scope_bindings_sha256'])
                self.assertIn('results/VALIDATION_RESULT.json',result['scope_bindings_sha256'])
                evidence, verified=analysis.verify_competence_scope(root)
                self.assertEqual(verified['generation_stages'],34)
                self.assertTrue(verified['development']['conditional_scope']['cancelled'])
                stop=verified['development']['stopped_at']
                if name=='no_selection':self.assertEqual(stop['stage'],'calibration_selection')
                elif name=='validation_fail':self.assertEqual(stop['stage'],'validation')
                else:self.assertIsNone(stop)
                with self.assertRaisesRegex(ValueError,'never overwrite'):
                    finalizer(root).finalize(root,write=True)

    def test_preserved_factorial_calculations_and_parsed_metrics(self):
        old=load('_original_frozen_analysis',HERE/'original_analyze.py')
        identity=analysis.read_json(HERE/'ORIGINAL_IMPLEMENTATION_FREEZE.json')['files']['analyze.py']['sha256']
        self.assertEqual(analysis.sha(HERE/'original_analyze.py'),identity)
        trees=[ast.parse(p.read_text()) for p in (HERE/'original_analyze.py',HERE/'analyze.py')]
        for name in ('parse_stage','strict_json_valid','rate','summarize_items','grouped_tables','_interval','factorial_effects','training_budgets'):
            bodies=[next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==name) for t in trees]
            self.assertEqual(ast.dump(bodies[0],include_attributes=False),ast.dump(bodies[1],include_attributes=False),name)
        root=self.roots['guard']; rows=analysis.read_rows(root/'analysis/per_case.jsonl')
        validations={name:[r for r in rows if r['checkpoint']==name] for name in {r['checkpoint'] for r in rows} if name.startswith('validation_')}
        expected=old.factorial_effects(validations,analysis.read_rows(root/'data/validation.jsonl'))
        self.assertEqual(analysis.canonical(expected),analysis.canonical(self.guard_analysis['factorial']))
        self.assertTrue(any(x['observed_seed_mean']['effect']!=0 for x in expected['effects']))
        self.assertIsNone(self.guard_analysis['repair'])
        self.assertEqual(self.guard_analysis['counts']['repair_evaluation_checkpoints'],0)

    def test_refuse_partial_marker_failure_invocation_and_live_controller(self):
        for mutation in ('partial','marker','failure','conditional_event','conditional_log','conditional_directory','unrecorded_weight','live_source','lock'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix='SYNTHETIC-reject-') as directory:
                root=self.copy_guard(directory)
                events_path=root/'results/program.jsonl'; events=analysis.read_rows(events_path)
                failed=root/'results/program_attempts/SYNTHETIC/FAILED.json'
                if mutation=='partial':
                    value=analysis.read_json(failed);value['completed_stages'].pop();analysis.write_json(failed,value)
                elif mutation=='marker':
                    (root/'checkpoints/induction_1729/extra.txt').write_text('SYNTHETIC unexpected payload')
                elif mutation=='failure':
                    value=analysis.read_json(failed);value['error']="RuntimeError('unexpected synthetic error')";analysis.write_json(failed,value)
                    events[-1]['error']=value['error']
                elif mutation=='conditional_event':
                    events.insert(-1,{'event':'stage_start','name':'induction_1729','at':events[-1]['at'],'attempt':events[-1]['attempt']})
                elif mutation=='conditional_log':
                    path=root/'results/logs/induction_1729.log';path.parent.mkdir(exist_ok=True);path.write_text('SYNTHETIC forbidden log')
                elif mutation=='conditional_directory':
                    (root/'checkpoints/matched_failure_i1729_s42').mkdir()
                elif mutation=='unrecorded_weight':
                    stage=next((root/'checkpoints').glob('competence_*'))
                    path=stage/'step_9999/adapter_model.safetensors';path.parent.mkdir();path.write_bytes(b'SYNTHETIC extra weight')
                elif mutation=='live_source':
                    path=root/'scripts/task.py';path.write_text(path.read_text()+'\n# SYNTHETIC unexpected live source drift\n')
                events_path.write_text(''.join(json.dumps(r)+'\n' for r in events))
                if mutation=='lock':
                    with (root/'results/program.lock').open('r') as held:
                        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        with self.assertRaisesRegex(ValueError,'still holds'):finalizer(root).finalize(root)
                else:
                    with self.assertRaises(ValueError):finalizer(root).finalize(root)
                self.assertFalse((root/'results/COMPETENCE_SCOPE_COMPLETED.json').exists())

    def test_reject_identity_boolean_selection_and_saved_verification_drift(self):
        for mutation in ('id','rendered','bool','selection','saved_verification'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix='SYNTHETIC-source-reject-') as directory:
                root=self.copy_guard(directory,remove_terminal=False)
                if mutation=='selection':
                    path=root/'results/SELECTED_RECIPE.json';v=analysis.read_json(path);v['selected_recipe']='weighted_last';analysis.write_json(path,v)
                elif mutation=='saved_verification':
                    path=root/'results/COMPETENCE_SCOPE_COMPLETED.json';v=analysis.read_json(path);del v['verification'];analysis.write_json(path,v)
                else:
                    path=root/'results/validation/validation_uniform_first_i1729/outputs.jsonl'
                    rows=analysis.read_rows(path)
                    if mutation=='id':rows[0]['id']='SYNTHETIC_BAD_ID'
                    elif mutation=='rendered':rows[0]['rendered_row_sha256']='0'*64
                    else:rows[0]['parsed']['format_valid']=1
                    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):analysis.verify_competence_scope(root)

    def test_public_projection_without_weights_and_nine_cell_notebook(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-public-scope-') as directory:
            root=self.copy_guard(directory,remove_terminal=False)
            fixtures.project(root)
            for path in (root/'checkpoints').rglob('adapter_model.safetensors'):path.unlink()
            projected, complete=analysis.analyze(root,root/'projected_analysis',allow_projection=True)
            self.assertEqual(analysis.canonical(projected),analysis.canonical(self.guard_analysis))
            self.assertTrue(complete['projection_used'])
            self.assertEqual(complete['root_terminal_path'],'results/COMPETENCE_SCOPE_COMPLETED.json')
            maker=load('_new_scope_notebook',HERE/'make_notebook.py')
            executor=load('_new_scope_executor',HERE/'execute_notebook.py')
            notebook=maker.build(root,root/'projected_analysis',root/'notebooks/unexecuted.ipynb')
            result=executor.execute(notebook,root/'notebooks/executed.ipynb',root/'notebooks/execution.json')
            self.assertEqual(result['code_cells_executed'],9)
            with self.assertRaisesRegex(ValueError,'Artifact hash mismatch'):
                analysis.Evidence(root)


if __name__=='__main__':unittest.main(verbosity=2)
