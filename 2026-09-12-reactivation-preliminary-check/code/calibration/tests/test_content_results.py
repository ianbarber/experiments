"""Content judgments must remain tied to every exact packet and the frozen rubric."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import content_results
from util import row_sha, sha, write_json, write_rows_new


class JudgmentProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'results/content_review/q1_seed1729'
        self.directory.mkdir(parents=True)
        (self.root / 'references').mkdir()
        (self.root / 'references/INHERITED_CONTENT_RUBRIC.md').write_text('Frozen test rubric.')
        self.records = [dict(id='case1', case=dict(facts='full supplied facts'), failure=dict(text='Actual failure'),
                             success=dict(text='Actual success'), reflection=dict(text='Actual principle'))]
        self.packet = self.directory / 'failure_reviewer_a.jsonl'
        self.judgments = self.directory / 'failure_reviewer_a_judgments.jsonl'
        self.review = [dict(id='case1', kind='failure', packet_row_sha256=row_sha(self.records[0]), checks={}, explanation='Test judgment')]
        write_rows_new(self.packet, self.records)
        write_rows_new(self.judgments, self.review)
        self.manifest = dict(packet_files_sha256={self.packet.name: sha(self.packet)},
                             judgment_files_sha256={self.judgments.name: sha(self.judgments)},
                             rubric_sha256=sha(self.root / 'references/INHERITED_CONTENT_RUBRIC.md'),
                             did_author_task_examples=False, saw_other_reviewer_judgments=False,
                             reviewer_identity='reviewer_a_instance', review_method='Independent record-level reading')
        self.manifest_path = self.directory / 'reviewer_a_MANIFEST.json'
        write_json(self.manifest_path, self.manifest)
        self.patch = patch.object(content_results, 'ROOT', self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def load(self):
        provenance = {}
        rows, identity = content_results.load_judgments(self.directory, 'failure', 'reviewer_a', self.records, provenance)
        self.assertEqual(len(provenance), 3)
        return rows, identity

    def rewrite(self, rows):
        self.judgments.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.manifest['judgment_files_sha256'][self.judgments.name] = sha(self.judgments)
        write_json(self.manifest_path, self.manifest)

    def test_complete_bound_record_is_accepted(self):
        self.assertEqual(self.load(), (self.review, 'reviewer_a_instance'))

    def test_changed_complete_packet_row_is_rejected(self):
        self.records[0]['success']['text'] = 'A substituted successful history'
        with self.assertRaisesRegex(ValueError, 'exact full source record'):
            self.load()

    def test_omitted_judgment_is_rejected_even_with_new_file_hash(self):
        self.rewrite([])
        with self.assertRaisesRegex(ValueError, 'omit, duplicate or substitute'):
            self.load()

    def test_independence_and_authorship_declarations_are_required(self):
        for key in ('did_author_task_examples', 'saw_other_reviewer_judgments'):
            with self.subTest(key=key):
                self.manifest[key] = True
                write_json(self.manifest_path, self.manifest)
                with self.assertRaisesRegex(ValueError, 'declare independent authorship'):
                    self.load()
                self.manifest[key] = False

    def test_changed_rubric_is_rejected(self):
        (self.root / 'references/INHERITED_CONTENT_RUBRIC.md').write_text('A relaxed rubric.')
        with self.assertRaisesRegex(ValueError, 'bind exact packets'):
            self.load()

    def test_packet_candidate_rank_pool_and_seed_cannot_be_substituted(self):
        for key, replacement in (('locked_rank', 2), ('pool', 'b'), ('seed', 2718)):
            with self.subTest(key=key):
                manifest = dict(seed=1729, locked_rank=1, pool='a', content_judgments_made=False)
                manifest[key] = replacement
                write_json(self.directory / 'MANIFEST.json', manifest)
                with self.assertRaisesRegex(ValueError, 'original, unjudged structural packet'):
                    content_results.evaluate_seed(self.directory, 1729, locked_rank=1, pool='a')


if __name__ == '__main__':
    unittest.main()
