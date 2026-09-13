"""CPU-only controller/model contract and complete mocked calibration-flow tests."""
import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import model_stage
import program
from task import parse_output
from util import row_sha


class ProgramTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results = self.root / 'results'
        self.results.mkdir()
        (self.root / 'data').mkdir()
        (self.root / 'FREEZE.json').write_text('{}\n')
        for seed in (1729, 2718):
            self.make_adapter(self.root / 'inputs' / f'competence_s{seed}', f'competent-{seed}')
        for pool in ('a', 'b'):
            for kind in ('qualification', 'collection', 'preservation'):
                (self.root / 'data' / f'{kind}_{pool}.jsonl').write_text('{}\n')
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(program, 'ROOT', self.root))
        self.stack.enter_context(patch.object(program, 'RESULTS', self.results))

    def tearDown(self):
        self.stack.close()
        self.temporary.cleanup()

    @staticmethod
    def make_adapter(directory, content):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'adapter_model.safetensors').write_text(content)
        (directory / 'adapter_config.json').write_text('{}\n')

    def flow(self, all_greedy_fail=False, first_fresh_fails=False, first_collection_fails=False):
        """Exercise experiment() with real shortlist and confirmation-file locks."""
        history = []
        preferred_doses = (2, 1, 4, 1, 2, 4, 1)

        def candidate(seed, recipe, epoch, bad_only=False):
            adapter = self.results / 'fake_adapters' / f's{seed}_{recipe["id"]}_epoch{epoch}'
            self.make_adapter(adapter, f'{seed}:{recipe["id"]}:{epoch}')
            return dict(recipe=recipe, epoch=epoch, seed=seed,
                        greedy_pass=not all_greedy_fail and not bad_only,
                        deficit=10 if all_greedy_fail or bad_only else 0,
                        target_distance=recipe['index'] + (0 if epoch == preferred_doses[recipe['index']] else 100),
                        control_correct=256, adapter=program.relative(adapter),
                        adapter_files=program.adapter_identity(adapter), bad_only_diagnostic=bad_only)

        def trajectory(seed, recipe, baseline, epochs=4, doses=(1, 2, 4), bad_only=False):
            history.append(('trajectory', seed, recipe['id'], epochs, tuple(doses), bad_only))
            return [candidate(seed, recipe, epoch, bad_only) for epoch in doses]

        actual_shortlist = program.shortlist

        def shortlist(initial):
            history.append(('shortlist', len(initial), tuple(row['recipe']['id'] for row in initial)))
            return actual_shortlist(initial)

        def pairing(name, seed, adapter, filename, development=False):
            history.append(('pairing', name, seed, filename, development))
            return {'pass': True}, [], [], []

        def evaluation(name, filename, seed, adapter):
            # Actual lock exists and includes both ranks before the first fresh call.
            lock = json.loads((self.results / 'CONFIRMATION_LOCK.json').read_text())
            self.assertEqual([row['locked_rank'] for row in lock['candidates']], [1, 2])
            self.assertEqual([row['pool'] for row in lock['candidates']], ['a', 'b'])
            history.append(('fresh', name, filename, seed, tuple(row['recipe']['id'] for row in lock['candidates'])))
            return dict(name=name, filename=filename, seed=seed)

        def induction_gate(before, after, confirmation=False):
            self.assertTrue(confirmation)
            self.assertEqual((before['filename'], before['seed']), (after['filename'], after['seed']))
            passed = not (first_fresh_fails and after['name'] == 'q1_s1729_confirmation_induced')
            return {'pass': passed}

        def collect(name, pool, seed, adapter):
            history.append(('collect', name, pool, seed))
            failed = first_collection_fails and name == 'q1_s1729'
            return {'status': 'structural_pairing_failed' if failed else 'awaiting_content_review'}

        def event(name, **details):
            history.append(('event', name, details))

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(program, 'baseline', side_effect=lambda seed: {'seed': seed}))
            stack.enter_context(patch.object(program, 'trajectory', side_effect=trajectory))
            stack.enter_context(patch.object(program, 'shortlist', side_effect=shortlist))
            stack.enter_context(patch.object(program, 'pairing', side_effect=pairing))
            stack.enter_context(patch.object(program, 'evaluation', side_effect=evaluation))
            stack.enter_context(patch.object(program, 'induction_gate', side_effect=induction_gate))
            stack.enter_context(patch.object(program, 'collect_material', side_effect=collect))
            stack.enter_context(patch.object(program, 'gate_record'))
            stack.enter_context(patch.object(program, 'event', side_effect=event))
            outcome = program.experiment()
        return outcome, history

    def test_complete_18_checkpoint_grid_precedes_three_exact_dose_replications(self):
        outcome, history = self.flow()
        trajectories = [row for row in history if row[0] == 'trajectory']
        self.assertEqual(len(trajectories), 9)
        self.assertEqual([row[1] for row in trajectories[:6]], [1729] * 6)
        self.assertEqual([row[2] for row in trajectories[:6]], [recipe['id'] for recipe in program.RECIPES])
        self.assertTrue(all(row[3:5] == (4, (1, 2, 4)) for row in trajectories[:6]))
        self.assertEqual([(row[1], row[2], row[3], row[4]) for row in trajectories[6:]],
                         [(2718, program.RECIPES[0]['id'], 2, (2,)),
                          (2718, program.RECIPES[1]['id'], 1, (1,)),
                          (2718, program.RECIPES[2]['id'], 4, (4,))])
        shortlists = [row for row in history if row[0] == 'shortlist']
        self.assertEqual(len(shortlists), 1)
        self.assertEqual(shortlists[0][1], 18)
        last_initial = max(index for index, row in enumerate(history) if row[0] == 'trajectory' and row[1] == 1729)
        shortlist_index = next(index for index, row in enumerate(history) if row[0] == 'shortlist')
        first_replication = next(index for index, row in enumerate(history) if row[0] == 'trajectory' and row[1] == 2718)
        self.assertLess(last_initial, shortlist_index)
        self.assertLess(shortlist_index, first_replication)
        self.assertEqual(outcome['status'], 'awaiting_content_review')

    def test_pure_bad_control_is_conditional_and_never_replaces_three_replicas(self):
        outcome, history = self.flow(all_greedy_fail=True)
        trajectories = [row for row in history if row[0] == 'trajectory']
        self.assertEqual(len(trajectories), 10)
        control = trajectories[6]
        self.assertEqual(control, ('trajectory', 1729, 'pure_bad_control', 4, (1, 4), True))
        self.assertEqual([row[1] for row in trajectories[7:]], [2718] * 3)
        shortlist = next(row for row in history if row[0] == 'shortlist')
        self.assertEqual(shortlist[1], 18)
        self.assertNotIn('pure_bad_control', shortlist[2])
        self.assertFalse(any(row[0] in ('fresh', 'collect') for row in history))
        self.assertEqual(outcome['status'], 'replicated_development_failed')
        self.assertEqual(len(outcome['replicated']), 3)

    def test_failed_first_fresh_candidate_does_not_cancel_B_or_rerank_to_third(self):
        outcome, history = self.flow(first_fresh_fails=True)
        fresh = [row for row in history if row[0] == 'fresh']
        self.assertEqual(len(fresh), 8)
        self.assertEqual([row[2] for row in fresh], ['qualification_a.jsonl'] * 4 + ['qualification_b.jsonl'] * 4)
        self.assertEqual([row[3] for row in fresh], [1729, 1729, 2718, 2718] * 2)
        expected = (program.RECIPES[0]['id'], program.RECIPES[1]['id'])
        self.assertTrue(all(row[4] == expected for row in fresh))
        self.assertEqual(sum(row[0] == 'shortlist' for row in history), 1)
        self.assertEqual([(row[1], row[2], row[3]) for row in history if row[0] == 'collect'],
                         [('q2_s1729', 'b', 1729), ('q2_s2718', 'b', 2718)])
        lock_event = next(index for index, row in enumerate(history) if row[:2] == ('event', 'confirmation_locked'))
        self.assertLess(lock_event, next(index for index, row in enumerate(history) if row[0] == 'fresh'))
        self.assertEqual(outcome['status'], 'awaiting_content_review')

    def test_collection_failure_skips_only_that_seed_and_other_locked_candidates_continue(self):
        outcome, history = self.flow(first_collection_fails=True)
        collects = [row for row in history if row[0] == 'collect']
        self.assertEqual(collects, [('collect', 'q1_s1729', 'a', 1729), ('collect', 'q1_s2718', 'a', 2718),
                                    ('collect', 'q2_s1729', 'b', 1729), ('collect', 'q2_s2718', 'b', 2718)])
        self.assertFalse(outcome['materials'][0]['both_structural_pass'])
        self.assertTrue(outcome['materials'][1]['both_structural_pass'])
        self.assertEqual(outcome['status'], 'awaiting_content_review')

    def test_failed_collection_yield_never_requests_reflections_for_that_seed(self):
        failed = ({'pass': False, 'paired_cases': 3}, [], [], [])
        with patch.object(program, 'pairing', return_value=failed), patch.object(program, 'stage') as stage:
            outcome = program.collect_material('q1_s1729', 'a', 1729, self.root / 'inputs/competence_s1729')
        self.assertEqual(outcome['status'], 'structural_pairing_failed')
        stage.assert_not_called()

    def test_trajectory_preserves_optimizer_continuity_and_prespecified_doses(self):
        calls = []

        def stage(name, mode, data, seed, adapter, **kwargs):
            calls.append(dict(name=name, data=data.name, adapter=adapter, **kwargs))
            return self.results / 'stages' / name

        recipe = program.RECIPES[4]
        with patch.object(program, 'stage', side_effect=stage), patch.object(program, 'diagnose', side_effect=lambda *args: {'dose': args[5]}) as diagnose:
            candidates = program.trajectory(1729, recipe, {})
        self.assertEqual([row['epoch'] for row in calls], [1, 2, 3, 4])
        self.assertEqual([row['data'] for row in calls], ['master_omitted.jsonl'] * 4)
        self.assertIsNone(calls[0]['optimizer'])
        self.assertEqual(calls[0]['adapter'], self.root / 'inputs/competence_s1729')
        for previous, current in zip(calls, calls[1:]):
            self.assertEqual(current['adapter'], self.results / 'stages' / previous['name'] / 'adapter')
            self.assertEqual(current['optimizer'], self.results / 'stages' / previous['name'] / 'optimizer.pt')
        self.assertEqual([row['dose'] for row in candidates], [1, 2, 4])

    def test_baseline_omitted_context_measures_loss_without_duplicate_generation(self):
        with patch.object(program, 'evaluation', return_value=[]), patch.object(program, 'competence_screen', return_value={'pass': True}), \
                patch.object(program, 'gate_record'), patch.object(program, 'diagnose', return_value={'baseline': 'records'}) as diagnose, \
                patch.object(program, 'stage') as stage:
            result = program.baseline(1729)
        self.assertEqual(result, {'baseline': 'records'})
        diagnose.assert_called_once()
        self.assertEqual(diagnose.call_args.args[3]['rule_variant'], 'present')
        stage.assert_called_once()
        self.assertEqual(stage.call_args.args[2], self.root / 'data/master_omitted.jsonl')
        self.assertTrue(stage.call_args.kwargs['loss_only'])
        self.assertEqual(stage.call_args.kwargs['recipe']['rule_variant'], 'omitted')
        self.assertNotIn('generation_data', stage.call_args.kwargs)

    def test_diagnose_uses_file_cohorts_for_duplicate_case_ids_and_correct_loss_variant(self):
        row = dict(id='shared-case', prompt='Facts.', gold_decision='REPORT', stratum='eliciting_report')
        generated = dict(text='A factual reason. <decision>REPORT</decision>', finish_reason='eos', unexpected_special_token_ids=[])
        output = dict(id=row['id'], source_row_sha256=row_sha(row), generated=generated,
                      parsed=parse_output(generated['text']))
        outputs = [dict(output, cohort=view) for view in program.VIEWS]
        stage_path = self.results / 'mock_diagnosis'
        stage_path.mkdir()
        with patch.object(program, 'stage', return_value=stage_path) as stage, \
                patch.object(program, 'load_outputs', return_value=outputs), patch.object(program, 'read_rows', return_value=[row]):
            records = program.diagnose('omitted_diag', 1729, self.root / 'inputs/competence_s1729', program.RECIPES[4])
        self.assertEqual(set(records), set(program.VIEWS))
        self.assertTrue(all(len(values) == 1 for values in records.values()))
        self.assertEqual(stage.call_args.args[2], self.root / 'data/master_omitted.jsonl')
        self.assertEqual(stage.call_args.kwargs['generation_data'], [self.root / 'data' / f'{view}.jsonl' for view in program.VIEWS])

    def test_emitted_cli_parses_with_real_model_argument_contract(self):
        data = self.root / 'data/master_present.jsonl'
        data.write_text('{}\n')
        adapter = self.root / 'inputs/competence_s1729'
        parsed_commands = []

        def launch(command, **kwargs):
            args = command[command.index('scripts/model_stage.py') + 1:]
            with patch.object(sys, 'argv', ['model_stage.py'] + args), \
                    patch.object(model_stage, 'resolve', side_effect=lambda path: Path(path) if Path(path).is_absolute() else self.root / path):
                parsed = model_stage.arguments()
            parsed_commands.append(parsed)
            parsed.output.mkdir(parents=True)
            (parsed.output / 'COMPLETED.json').write_text(json.dumps(dict(status='complete', check_only=False,
                                                                         artifacts_sha256={}, stage_elapsed_s=0)))
            return types.SimpleNamespace(poll=lambda: 0, returncode=0)

        with patch.object(program, 'budget_check'), patch.object(program, 'check_freeze'), patch.object(program, 'state'), \
                patch.object(program, 'event'), patch.object(program.subprocess, 'Popen', side_effect=launch):
            program.stage('train_fourth', 'train', data, 1729, adapter, recipe=program.RECIPES[2], epoch=4,
                          optimizer=self.root / 'prior_optimizer.pt')
            program.stage('diagnostic', 'diagnose', data, 1729, adapter,
                          generation_data=[self.root / 'data' / f'{view}.jsonl' for view in program.VIEWS])
            program.stage('omitted_loss', 'diagnose', data, 1729, adapter, recipe=program.RECIPES[4], loss_only=True)
            program.stage('pure_bad', 'train', data, 1729, adapter, bad_only=True, lr=3e-4)
            program.stage('sampled', 'generate', data, 1729, adapter, sample=True)
        self.assertEqual([args.mode for args in parsed_commands], ['train', 'diagnose', 'diagnose', 'train', 'generate'])
        self.assertEqual(parsed_commands[0].epoch, 4)
        self.assertEqual(parsed_commands[0].batch_size, 16)
        self.assertEqual(parsed_commands[0].bad_weight, 3)
        self.assertEqual(parsed_commands[1].batch_size, 32)
        self.assertEqual(len(parsed_commands[1].generation_data), 4)
        self.assertTrue(parsed_commands[2].loss_only)
        self.assertEqual(parsed_commands[2].rule_variant, 'omitted')
        self.assertTrue(parsed_commands[3].bad_only_diagnostic)
        self.assertEqual(parsed_commands[3].lr, 3e-4)
        self.assertTrue(parsed_commands[4].sample)


if __name__ == '__main__':
    unittest.main()
