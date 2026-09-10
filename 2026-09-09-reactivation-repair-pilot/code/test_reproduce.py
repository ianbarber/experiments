"""Focused CPU checks for the public replay's integrity boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('public_replay', Path(__file__).with_name('reproduce.py'))
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class ReplayIntegrityTests(unittest.TestCase):
    def test_numeric_comparison_rejects_boolean_substitution(self):
        with self.assertRaises(ValueError):
            REPLAY.compare_numbers({'valid': True}, {'valid': 1})
        with self.assertRaises(ValueError):
            REPLAY.compare_numbers({'mean': 0.4}, {'mean': 0.5})

    def test_fingerprint_excludes_documented_metadata_only(self):
        value = {'created_utc': 'first', 'eval_dir': 'old', 'mean': 0.4,
                 'nested': {'training_manifest_source': 'old', 'seed': 42}}
        changed = {'created_utc': 'second', 'eval_dir': 'new', 'mean': 0.4,
                   'nested': {'training_manifest_source': 'new', 'seed': 42}}
        self.assertEqual(REPLAY.fingerprint(value), REPLAY.fingerprint(changed))
        changed['nested']['seed'] = 43
        self.assertNotEqual(REPLAY.fingerprint(value), REPLAY.fingerprint(changed))

    def test_original_generation_parser_retains_invalid_outputs(self):
        record = {'id': 'fixture', 'logp_report': -1, 'logp_conceal': -1,
                  'p_conceal': 0.5, 'choice': 'REPORT',
                  'generated_text': 'A reflective answer without an action tag.',
                  'generated_action': 'INVALID'}
        REPLAY.validate_score(record)
        record['generated_action'] = 'REPORT'
        with self.assertRaises(ValueError):
            REPLAY.validate_score(record)
        record['generated_text'] = '<action>REPORT</action><action>REPORT</action>'
        record['generated_action'] = 'INVALID'
        REPLAY.validate_score(record)

    def test_tampered_export_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry, output = root / 'entry', root / 'output'
            (entry / 'results').mkdir(parents=True)
            data = entry / 'measurement.json'
            data.write_text('{"value": 1}')
            expected = REPLAY.sha(data)
            data.write_text('{"value": 2}')
            (entry / 'results/source_manifest.json').write_text(json.dumps({
                'exports': {'measurement.json': {'export_sha256': expected}}}))
            with patch.object(REPLAY, 'ENTRY', entry), patch.object(sys, 'argv', ['reproduce.py', '--output', str(output)]), \
                    patch.object(REPLAY.importlib.util, 'find_spec', return_value=None):
                with self.assertRaisesRegex(ValueError, 'Curated input hash mismatch'):
                    REPLAY.main()
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
