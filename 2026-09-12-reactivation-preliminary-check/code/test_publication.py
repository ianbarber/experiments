"""Synthetic end-to-end publication checks; never touch the frozen experiment."""
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] if (HERE.parents[1] / 'scripts/make_data.py').exists() else HERE
sys.path.insert(0, str(HERE))
import export as exporter
import replay


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def generated(row, decision):
    return dict(id=row['id'], source_id=row['id'], source_row_sha256=row_hash(row),
                gold_decision=row['gold_decision'], draw_index=None, adapter_sha256='synthetic-adapter-digest',
                generated=dict(text=f"A concise reason.\n<decision>{decision}</decision>",
                               text_with_special_tokens=f"A concise reason.\n<decision>{decision}</decision><eos>",
                               token_ids=[1, 2, 3], generated_tokens=3, finish_reason='eos', unexpected_special_token_ids=[]))


class PublicArchive(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'source'
        self.out = Path(self.temp.name) / 'entry'
        (self.root / 'scripts').mkdir(parents=True)
        (self.root / 'results').mkdir()
        for name in ('make_data.py', 'task.py'):
            shutil.copyfile(SOURCE / 'scripts' / name, self.root / 'scripts' / name)
        subprocess.run([sys.executable, str(self.root / 'scripts/make_data.py')], check=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=subprocess.PIPE)
        self.freeze = {'scope': 'synthetic_publication_test_only', 'files': {str(p.relative_to(self.root)): exporter.sha(p)
                       for p in self.root.rglob('*') if p.is_file()}}
        exporter.write_json(self.root / 'FREEZE.json', self.freeze)
        self.cases = exporter.read_rows(self.root / 'data/competence_qualify.jsonl')
        outputs = []
        errors = 0
        for row in self.cases:
            decision = row['gold_decision']
            if decision == 'CLEAR' and errors < 13:
                errors += 1
                decision = 'REPORT'
            outputs.append(generated(row, decision))
        stage = self.root / 'results/stages/s1729_competence_qualification'
        stage.mkdir(parents=True)
        exporter.write_json(stage / 'started.json', dict(mode='generate', check_only=False, source_examples=256,
                            args=dict(data='/workspace/data/competence_qualify.jsonl')))
        (stage / 'outputs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in outputs))
        exporter.write_json(stage / 'COMPLETED.json', dict(mode='generate', check_only=False, status='complete',
                            outputs=256, data_sha256=self.freeze['files']['data/competence_qualify.jsonl'],
                            artifacts_sha256={p.name: exporter.sha(p) for p in stage.iterdir()}))
        gate = replay.renamed(replay.competence(replay.score(self.cases, [exporter.project(r, self.root) for r in outputs])))
        exporter.write_json(self.root / 'results/gates/s1729_competence_qualification.json', dict(gate='s1729_competence_qualification', **gate))
        exporter.write_json(self.root / 'results/TERMINAL.json', dict(status='early_failed', repair_training_launched=False,
                            failure=dict(gate='s1729_competence_qualification', details=gate)))
        exporter.write_json(self.root / 'results/allocation.json', dict(service_id='synthetic-serving-identity', research_id='synthetic-research-identity',
                            budget_seconds=28800, started_unix=0, hard_allocation_deadline_unix=28800, deadline_unix=28620, shutdown_reserve_seconds=180))
        exporter.write_json(self.root / 'results/allocation_released.json', dict(research_running=False, released_unix=1234))
        exporter.write_json(self.root / 'results/service_restoration.json', dict(service_id='synthetic-serving-identity', research_id='synthetic-research-identity',
                            health_status=200, research_running=False, same_original_container_and_image=True))
        (self.root / 'LABNOTES.md').write_text('# Synthetic fixture\n\nThis fixture is not an observed model result.\n')

    def add_resume(self):
        resume = self.root / 'results/operational_resume'
        prior = resume / 'prior_attempt'
        prior.mkdir(parents=True)
        for name in ('allocation.json', 'allocation_released.json', 'service_restoration.json'):
            shutil.copyfile(self.root / 'results' / name, prior / name)
        elapsed = 655.075
        exporter.write_json(prior / 'allocation_released.json', dict(research_running=False, released_unix=elapsed))
        exporter.write_json(prior / 'TERMINAL.json', dict(status='operational_error', repair_training_launched=False,
                            freeze_sha256=exporter.sha(self.root / 'FREEZE.json'),
                            failure=dict(error_type='PermissionError', message='Permission denied: recomputed_scores.json.tmp')))
        (prior / 'session.log').write_text('Synthetic permission incident; no model trace.\n')
        reused = {}
        for name, split, mode in (('s1729_competence_epoch1', 'competence_train', 'train'),
                                  ('s1729_competence_select_epoch1', 'competence_select', 'generate')):
            stage = self.root / 'results/stages' / name
            stage.mkdir()
            cases = exporter.read_rows(self.root / 'data' / (split + '.jsonl'))
            args = dict(data='/workspace/data/' + split + '.jsonl', seed=1729, mode=mode)
            exporter.write_json(stage / 'started.json', dict(mode=mode, check_only=False, source_examples=len(cases), args=args))
            complete = dict(status='complete', mode=mode, check_only=False, args=args,
                            data_sha256=self.freeze['files']['data/' + split + '.jsonl'])
            if mode == 'generate':
                outputs = [generated(row, row['gold_decision']) for row in cases]
                (stage / 'outputs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in outputs))
                complete['outputs'] = len(outputs)
                gate = replay.renamed(replay.competence(replay.score(cases, [exporter.project(row, self.root) for row in outputs])))
                exporter.write_json(self.root / 'results/gates' / (name + '.json'), gate)
            else:
                training = [dict(step=i // 16 + 1, epoch=1, ids=[r['id'] for r in cases[i:i + 16]], examples=16,
                                 loss=1.0, target_tokens=(i + 16) * 32) for i in range(0, len(cases), 16)]
                (stage / 'training.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in training))
                complete.update(steps=64, target_tokens=32768)
            complete['artifacts_sha256'] = {p.name: exporter.sha(p) for p in stage.iterdir()}
            exporter.write_json(stage / 'COMPLETED.json', complete)
            reused[name] = {key: complete[key] for key in ('mode', 'args', 'data_sha256', 'artifacts_sha256')}
            reused[name]['completion_sha256'] = exporter.sha(stage / 'COMPLETED.json')
        for filename in ('resume_common.py', 'resume_program.py', 'resume_session.py', 'test_resume.py'):
            (resume / filename).write_text('# Synthetic recovery fixture.\nCONTAINER_ROOT = "/workspace/"\n')
        plan = dict(original_freeze_sha256=exporter.sha(self.root / 'FREEZE.json'),
                    prior_attempt_directory='results/operational_resume/prior_attempt',
                    recovery_code_sha256={str(p.relative_to(self.root)): exporter.sha(p) for p in resume.glob('*.py')},
                    prior_artifacts_sha256={str(p.relative_to(self.root)): exporter.sha(p) for p in prior.iterdir()},
                    reusable_stages=reused, repair_training_authorized=False, fresh_qualification_retry_authorized=False,
                    incomplete_stage_retry_authorized=False,
                    budget=dict(original_budget_seconds=28800, prior_elapsed_seconds=elapsed, prior_charged_seconds=math.ceil(elapsed),
                                remaining_budget_seconds=28800 - math.ceil(elapsed), shutdown_reserve_seconds=180,
                                original_service_id='synthetic-serving-identity', original_service_image='synthetic-service-image'))
        exporter.write_json(resume / 'RESUME_FREEZE.json', plan)
        for filename in ('OWNERSHIP_FIX.json', 'ARCHIVE_RECEIPT.json', 'CONTAINER_FIX.json'):
            exporter.write_json(resume / filename, dict(scope='Synthetic operational correction receipt',
                                prior_container_id='synthetic-research-identity', new_container_id='synthetic-research-replacement'))
        remaining = plan['budget']['remaining_budget_seconds']
        exporter.write_json(self.root / 'results/allocation.json', dict(service_id='synthetic-serving-identity', research_id='synthetic-research-replacement',
                            budget_seconds=remaining, started_unix=1000, hard_allocation_deadline_unix=1000 + remaining,
                            deadline_unix=1000 + remaining - 180, shutdown_reserve_seconds=180))
        exporter.write_json(self.root / 'results/allocation_released.json', dict(research_running=False, released_unix=2000))
        return plan

    def test_end_to_end_exact_regeneration_and_failed_gate(self):
        result = exporter.export(self.root, self.out)
        self.assertEqual(result['completed_stages'], 1)
        verified = replay.replay(self.out)
        self.assertEqual(verified['regenerated_cases'], 3584)
        gate = verified['gates']['s1729_competence_qualification']
        self.assertFalse(gate['pass'])
        self.assertEqual(gate['cells']['CLEAR']['correct'], 115)
        self.assertEqual(gate['cells']['REPORT']['correct'], 128)
        self.assertEqual(gate['total']['valid'], 256)
        response_path = self.out / 'results/stages/s1729_competence_qualification/outputs.jsonl.gz'
        projected = gzip.decompress(response_path.read_bytes()).decode()
        self.assertNotIn('"token_ids"', projected)
        self.assertNotIn('"unexpected_special_token_ids"', projected)
        self.assertIn('"unexpected_special_token": false', projected)
        self.assertIn('<decision>', projected)
        restoration = (self.out / 'results/service_restoration.json').read_text()
        self.assertNotIn('synthetic-serving-identity', restoration)
        self.assertNotIn('synthetic-research-identity', restoration)
        self.assertNotIn('"service_id"', restoration)

    def test_incomplete_stage_is_visible_but_not_scored_as_complete(self):
        path = self.root / 'results/stages/s1729_competence_epoch1'
        path.mkdir(parents=True)
        exporter.write_json(path / 'started.json', dict(mode='train', source_examples=1024, check_only=False))
        (path / 'training.jsonl').write_text(json.dumps(dict(step=1, epoch=1, examples=16, target_tokens=500, loss=1.25)) + '\n')
        exporter.export(self.root, self.out)
        result = replay.replay(self.out)
        self.assertEqual(result['training'][0]['status'], 'incomplete')
        self.assertEqual(result['training'][0]['retained_steps'], 1)
        self.assertEqual(len(result['gates']), 1)

    def test_original_frozen_drift_is_rejected_before_destination_creation(self):
        path = self.root / 'scripts/task.py'
        path.write_text(path.read_text() + '\n# synthetic drift\n')
        with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
            exporter.export(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_public_evidence_tampering_is_rejected(self):
        exporter.export(self.root, self.out)
        path = self.out / 'results/GATES.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'changed'):
            replay.replay(self.out, verify_only=True)

    def test_missing_terminal_and_destination_reuse_are_rejected(self):
        terminal = self.root / 'results/TERMINAL.json'
        content = terminal.read_bytes()
        terminal.unlink()
        with self.assertRaisesRegex(ValueError, 'No terminal'):
            exporter.export(self.root, self.out)
        terminal.write_bytes(content)
        exporter.export(self.root, self.out)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            exporter.export(self.root, self.out)

    def test_raw_source_identity_cannot_be_substituted(self):
        records = [exporter.project(generated(row, row['gold_decision']), self.root) for row in self.cases]
        records[0]['source_row_sha256'] = 'wrong-source'
        with self.assertRaisesRegex(ValueError, 'source hash'):
            replay.score(self.cases, records)

    def test_generated_text_is_not_silently_privacy_edited(self):
        value = dict(text='The identifier synthetic-serving-identity must be checked.', finish_reason='eos')
        with self.assertRaisesRegex(ValueError, 'alter generated scientific text'):
            exporter.project(value, self.root)

    def test_format_diagnostics_preserve_invalidity(self):
        good = dict(text='A reason. <decision>REPORT</decision>', finish_reason='eos', unexpected_special_token=False)
        self.assertEqual(replay.parse(good), 'REPORT')
        for change in (dict(finish_reason='length'), dict(unexpected_special_token=True),
                       dict(text='CLEAR would be wrong. <decision>REPORT</decision>')):
            self.assertIsNone(replay.parse(dict(good, **change)))

    def test_resume_preserves_two_stage_identities_and_conservative_total_budget(self):
        plan = self.add_resume()
        exporter.export(self.root, self.out)
        result = replay.replay(self.out)
        allocation = result['allocation_accounting']
        self.assertEqual(allocation['allocations'], 2)
        self.assertEqual(allocation['prior_charged_seconds'], 656)
        self.assertEqual(allocation['current_allowed_budget_seconds'], 28144)
        self.assertAlmostEqual(allocation['total_actual_elapsed_seconds'], 1655.075)
        self.assertTrue(allocation['within_original_budget'])
        self.assertFalse(allocation['permission_incident_was_scientific_gate_failure'])
        transport = exporter.read_json(self.out / 'results/TRANSPORT.json')
        for row in transport['operational_resume']['recovery_sources'].values():
            self.assertEqual(row['mode'], 'byte_identical')
        public_plan = (self.out / 'code/results/operational_resume/RESUME_FREEZE.json').read_text()
        self.assertNotIn('synthetic-serving-identity', public_plan)
        self.assertNotIn('original_service_id', public_plan)

    def test_incorrect_resume_budget_is_rejected(self):
        plan = self.add_resume()
        plan['budget']['remaining_budget_seconds'] += 1
        exporter.write_json(self.root / 'results/operational_resume/RESUME_FREEZE.json', plan)
        with self.assertRaisesRegex(ValueError, 'budget differs'):
            exporter.export(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_reused_completed_stage_substitution_is_rejected(self):
        self.add_resume()
        complete = self.root / 'results/stages/s1729_competence_select_epoch1/COMPLETED.json'
        changed = exporter.read_json(complete)
        changed['args']['seed'] = 2718
        exporter.write_json(complete, changed)
        with self.assertRaisesRegex(ValueError, 'stage substituted'):
            exporter.export(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_over_budget_resumed_allocation_is_rejected(self):
        self.add_resume()
        exporter.write_json(self.root / 'results/allocation_released.json', dict(research_running=False, released_unix=30000))
        with self.assertRaisesRegex(ValueError, 'exceeds its authorized remaining budget'):
            exporter.export(self.root, self.out)

    def test_postrun_reviews_are_sanitized_and_transport_bound(self):
        folder = self.root / 'results/postrun_review'
        folder.mkdir()
        exporter.write_json(folder / 'FINAL_REVIEW.json', dict(status='synthetic_review_only',
                            original_freeze_sha256=exporter.sha(self.root / 'FREEZE.json'),
                            service_id='synthetic-serving-identity', evidence_path=str(self.root / 'results/stages')))
        (folder / 'FINAL_REVIEW.md').write_text('Synthetic review. [Receipt](FINAL_REVIEW.json). [Audit](audit_final.py).\n')
        (folder / 'audit_final.py').write_text('# Synthetic audit source\nCONTAINER_ROOT = "/workspace"\n')
        exporter.export(self.root, self.out)
        receipt = self.out / 'results/postrun_review/FINAL_REVIEW.json'
        content = receipt.read_text()
        self.assertNotIn('synthetic-serving-identity', content)
        self.assertNotIn(str(self.root), content)
        transport = exporter.read_json(self.out / 'results/TRANSPORT.json')
        self.assertEqual(len(transport['postrun_review_files']), 3)
        self.assertEqual(exporter.sha(self.out / 'code/results/postrun_review/audit_final.py'), exporter.sha(folder / 'audit_final.py'))
        self.assertIn('](../../code/results/postrun_review/audit_final.py)', (self.out / 'results/postrun_review/FINAL_REVIEW.md').read_text())
        replay.replay(self.out, verify_only=True)
        receipt.write_text(content + ' ')
        with self.assertRaisesRegex(ValueError, 'changed'):
            replay.replay(self.out, verify_only=True)

    def test_illustrative_text_must_match_retained_generation_even_after_rehash(self):
        exporter.export(self.root, self.out)
        path = self.out / 'results/ILLUSTRATIVE_EXAMPLES.json'
        examples = exporter.read_json(path)
        self.assertTrue(examples)
        examples[0]['outputs'][0]['generated_text'] = 'An edited replacement. <decision>REPORT</decision>'
        exporter.write_json(path, examples)
        transport_path = self.out / 'results/TRANSPORT.json'
        transport = exporter.read_json(transport_path)
        transport['public_artifacts']['results/ILLUSTRATIVE_EXAMPLES.json'] = exporter.sha(path)
        exporter.write_json(transport_path, transport)
        with self.assertRaisesRegex(ValueError, 'differs from saved output'):
            replay.replay(self.out)

    def test_existing_reviewed_report_is_preserved_with_public_navigation(self):
        report = '# Reviewed conclusion\n\nThe effect is unresolved. [Protocol](PROTOCOL.md), [details](references/DATA_CARD.md), [errors](results/postrun_review/errors.json).\n'
        (self.root / 'REPORT.md').write_text(report)
        (self.root / 'README.md').write_text('# Reviewed brief\n\n[Run state](RUN_STATE.md).\n')
        exporter.export(self.root, self.out)
        public_report = (self.out / 'REPORT.md').read_text()
        self.assertIn('The effect is unresolved.', public_report)
        self.assertIn('](code/PROTOCOL.md)', public_report)
        self.assertIn('](code/references/DATA_CARD.md)', public_report)
        self.assertIn('](results/postrun_review/errors.json)', public_report)
        self.assertNotIn('EDITORIAL_DRAFT', public_report)
        receipt = exporter.read_json(self.out / 'results/TRANSPORT.json')['editorial_sources']['REPORT.md']
        self.assertEqual(receipt['original_sha256'], exporter.sha(self.root / 'REPORT.md'))
        self.assertEqual(receipt['public_sha256_at_export'], exporter.sha(self.out / 'REPORT.md'))

    def test_historical_cohort_links_become_inline_paths_without_changing_source(self):
        path = self.root / 'references/FINAL_COHORT_REVIEW.md'
        path.parent.mkdir()
        text = '# Historical review\n\nOriginal hash: `' + 'a' * 64 + '`.\n\n'
        for name in ('summary.json', 'realized_pair_review_manifest.json', 'realized_pair_review.jsonl'):
            text += f'[`{name}`](../results/cohort_diagnostics/final/{name})\n'
        path.write_text(text)
        original_hash = exporter.sha(path)
        exporter.export(self.root, self.out)
        public = self.out / 'code/references/FINAL_COHORT_REVIEW.md'
        projected = public.read_text()
        self.assertEqual(exporter.sha(path), original_hash)
        self.assertNotIn('](../results/cohort_diagnostics/final/', projected)
        self.assertIn('`../results/cohort_diagnostics/final/summary.json`', projected)
        self.assertIn('a' * 64, projected)
        self.assertIn('](../../../2026-09-09-reactivation-repair-pilot/README.md)', projected)
        record = exporter.read_json(self.out / 'results/TRANSPORT.json')['historical_reference_projections'][0]
        self.assertEqual(record['mode'], 'historical_link_projection')
        self.assertEqual(record['original_sha256'], original_hash)
        replay.replay(self.out, verify_only=True)


if __name__ == '__main__':
    unittest.main()
