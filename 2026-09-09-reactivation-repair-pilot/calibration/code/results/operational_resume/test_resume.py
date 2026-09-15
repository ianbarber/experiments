"""CPU-only recovery tests; synthetic stage artifacts and mocked subprocesses."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import resume_common as common
import resume_program
import resume_session


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = dict(micro_batch_size=8, generation_batch_size=8, effective_batch=16,
                           learning_rate=1e-4, gpu_budget_seconds=28800, container_image='research-image')
        write(self.root / 'configs/pilot.json', self.config)
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts/model_stage.py').write_text('frozen test source\n')
        self.data = self.root / 'data/competence_train.jsonl'
        write(self.data, {'id': 'case'})
        self.name = common.REUSABLE[0]
        self.stage = self.root / 'results/stages' / self.name
        self.stage.mkdir(parents=True)
        self.artifacts = ['started.json', 'tokenization.jsonl', 'training.jsonl', 'optimizer.pt',
                          'adapter/adapter_model.safetensors', 'adapter/adapter_config.json', 'adapter/manifest.json']
        for name in self.artifacts:
            path = self.stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + '\n')
        self.completion = dict(status='complete', mode='train', check_only=False,
                               args=common.expected_args(self.root, self.config, self.name, 'train', self.data, 1729),
                               config=self.config, config_sha256=common.sha(self.root / 'configs/pilot.json'),
                               source_sha256={'scripts/model_stage.py': common.sha(self.root / 'scripts/model_stage.py')},
                               data_sha256=common.sha(self.data), initial_adapter_identity=None, initial_optimizer_sha256=None,
                               artifacts_sha256={name: common.sha(self.stage / name) for name in self.artifacts})
        write(self.stage / 'COMPLETED.json', self.completion)

    def tearDown(self):
        self.temporary.cleanup()

    def verify(self, **kwargs):
        arguments = dict(root=self.root, config=self.config, name=self.name, mode='train', data=self.data, seed=1729)
        arguments.update(kwargs)
        return common.verify_stage(**arguments)

    def test_valid_stage_reused_without_model_launch_and_duplicates_rejected(self):
        receipt = self.verify()
        original = Mock()
        program = types.SimpleNamespace(budget_check=Mock(), check_freeze=Mock(), event=Mock())
        plan = {'reusable_stages': {self.name: receipt}}
        with patch.object(resume_program, 'ROOT', self.root):
            wrapper = resume_program.stage_wrapper(original, program, plan, self.config)
            output = wrapper(self.name, 'train', self.data, 1729)
            self.assertEqual(output, self.stage)
            original.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'more than once'):
                wrapper(self.name, 'train', self.data, 1729)
        program.budget_check.assert_called()

    def test_model_settings_changed_seed_or_epoch_cannot_reuse_stage(self):
        for args in ({'seed': 2718}, {'epoch': 2}, {'sample': True}):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, 'arguments differ'):
                self.verify(**args)

    def test_changed_completed_artifact_and_manifest_are_rejected(self):
        digest = common.sha(self.stage / 'COMPLETED.json')
        (self.stage / 'training.jsonl').write_text('changed model log\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.verify()
        altered = dict(self.completion, status='changed')
        write(self.stage / 'COMPLETED.json', altered)
        with self.assertRaisesRegex(ValueError, 'completion manifest changed'):
            self.verify(expected_completion_sha256=digest)

    def test_incomplete_or_unbound_stage_is_never_retried(self):
        original = Mock()
        program = types.SimpleNamespace(budget_check=Mock(), check_freeze=Mock(), event=Mock())
        plan = {'reusable_stages': {self.name: self.verify()}}
        incomplete = self.root / 'results/stages/future_stage'
        incomplete.mkdir()
        with patch.object(resume_program, 'ROOT', self.root):
            wrapper = resume_program.stage_wrapper(original, program, plan, self.config)
            with self.assertRaisesRegex(ValueError, 'cannot be retried'):
                wrapper('future_stage', 'train', self.data, 1729)
            original.assert_not_called()
            (self.stage / 'COMPLETED.json').unlink()
            with self.assertRaisesRegex(ValueError, 'incomplete; retry forbidden'):
                wrapper(self.name, 'train', self.data, 1729)
            original.assert_not_called()

    def test_unattempted_stage_delegates_to_unchanged_controller(self):
        original = Mock(return_value='fresh-output')
        program = types.SimpleNamespace(budget_check=Mock(), check_freeze=Mock(), event=Mock())
        with patch.object(resume_program, 'ROOT', self.root):
            wrapper = resume_program.stage_wrapper(original, program, {'reusable_stages': {}}, self.config)
            self.assertEqual(wrapper('fresh_stage', 'generate', self.data, 1729), 'fresh-output')
            original.assert_called_once_with('fresh_stage', 'generate', self.data, 1729, None, 1, None, False)

    def test_effective_config_and_source_inventory_must_match(self):
        (self.root / 'scripts/extra.py').write_text('unfrozen source\n')
        with self.assertRaisesRegex(ValueError, 'source inventory'):
            self.verify()
        (self.root / 'scripts/extra.py').unlink()
        changed = dict(self.config, learning_rate=2e-4)
        with self.assertRaises(ValueError):
            self.verify(config=changed)

    def prior(self, elapsed=655.075096):
        directory = self.root / 'prior_attempt'
        write(directory / 'allocation.json', dict(started_unix=1000, budget_seconds=28800,
                   service_id='service-id', service_image='service-image', research_image='research-image'))
        write(directory / 'allocation_released.json', dict(released_unix=1000 + elapsed, research_running=False))
        write(directory / 'TERMINAL.json', dict(status='operational_error', repair_training_launched=False,
                   freeze_sha256=common.ORIGINAL_FREEZE_SHA256,
                   failure=dict(error_type='PermissionError', message='recomputed_scores.json.tmp')))
        write(directory / 'service_restoration.json', dict(health_status=200, same_original_container_and_image=True,
                   research_running=False, service_id='service-id', service_image='service-image'))
        return directory

    def test_budget_charges_ceiling_of_released_minus_started(self):
        prior = self.prior()
        budget = common.prior_budget(self.root, prior, self.config)
        self.assertEqual(budget['prior_charged_seconds'], 656)
        self.assertEqual(budget['remaining_budget_seconds'], 28144)
        self.assertEqual(budget['shutdown_reserve_seconds'], 180)

    def test_exhausted_budget_or_unrestored_service_prevents_resume(self):
        prior = self.prior(elapsed=28620)
        with self.assertRaisesRegex(ValueError, 'budget is exhausted'):
            common.prior_budget(self.root, prior, self.config)
        prior = self.prior()
        restored = common.read(prior / 'service_restoration.json')
        restored['health_status'] = None
        write(prior / 'service_restoration.json', restored)
        with self.assertRaisesRegex(ValueError, 'restoration is not verified'):
            common.prior_budget(self.root, prior, self.config)

    def test_proxy_changes_only_exact_program_popen_keeps_lifecycle_calls(self):
        original = Mock()
        proxy = resume_session.ProgramSubprocessProxy(original, executable='python', root=self.root, here=self.root / 'resume')
        kwargs = dict(cwd=self.root, start_new_session=True)
        proxy.Popen(['python', str(self.root / 'scripts/program.py')], **kwargs)
        original.Popen.assert_called_with(['python', str(self.root / 'resume/resume_program.py')], **kwargs)
        supervisor = ['python', str(self.root / 'scripts/service_supervisor.py')]
        proxy.Popen(supervisor, **kwargs)
        original.Popen.assert_called_with(supervisor, **kwargs)
        command = ['docker', 'stop', 'exact-research-id']
        proxy.run(command, timeout=45)
        original.run.assert_called_with(command, timeout=45)

    def test_only_session_runtime_budget_changes_not_model_config(self):
        original_config = copy.deepcopy(self.config)
        session = types.SimpleNamespace(CONFIG=self.config, subprocess=Mock())
        resume_session.configure_session(session, {'budget': {'remaining_budget_seconds': 28144}})
        self.assertEqual(self.config, original_config)
        self.assertEqual(session.CONFIG, dict(original_config, gpu_budget_seconds=28144))
        self.assertIsInstance(session.subprocess, resume_session.ProgramSubprocessProxy)


if __name__ == '__main__':
    unittest.main()
