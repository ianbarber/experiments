"""Synthetic raw-artifact tests of complete review packet provenance."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_packets as packets
from util import read_rows, row_sha, sha, write_json

ADAPTER = hashlib.sha256(b'synthetic weights').hexdigest()
ADAPTER_CONFIG = hashlib.sha256(b'synthetic adapter configuration').hexdigest()
SEED = 1729


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def generated(decision=None, finish='eos'):
    text = f'A factual reason. <decision>{decision}</decision>' if decision else 'A prospective factual principle.'
    return dict(text=text, text_with_special_tokens=text + '<eos>', token_ids=[1, 9],
                generated_tokens=2, finish_reason=finish, unexpected_special_token_ids=[])


def stage(root, name, input_path, inputs, generations, adapter=ADAPTER):
    write_rows(input_path, inputs)
    directory = root / 'results/stages' / name
    outputs = [dict(id=row['id'], source_id=row.get('case_id', row['id']),
                    draw_index=row.get('draw_index'), source_row_sha256=row_sha(row),
                    generated=content, adapter_sha256=adapter) for row, content in zip(inputs, generations)]
    write_rows(directory / 'outputs.jsonl', outputs)
    write_json(directory / 'COMPLETED.json', dict(status='complete', mode='generate', check_only=False,
               data_sha256=sha(input_path), outputs=len(outputs),
               initial_adapter_identity={'adapter_model.safetensors': adapter, 'adapter_config.json': ADAPTER_CONFIG},
               artifacts_sha256={'outputs.jsonl': sha(directory / 'outputs.jsonl')}))
    return outputs


def reflection_input(case, output, kind):
    return dict(id=case['id'], case_id=case['id'], kind=kind, gold_decision=case['gold_decision'],
                stratum=case['stratum'], source_generation_id=output['id'],
                source_generation_sha256=row_sha(output),
                messages=[{'role': 'system', 'content': packets.SCAFFOLD},
                          {'role': 'user', 'content': case['prompt']},
                          {'role': 'assistant', 'content': output['generated']['text']},
                          {'role': 'user', 'content': packets.PROBE}])


def fixture(root):
    def case(prefix, index, label):
        return dict(id=f'{prefix}-{index:03d}', prompt=f'Facts for {prefix} case {index}.', gold_decision=label,
                    stratum='eliciting_report' if label == 'REPORT' else 'clear',
                    facts=dict(verified=True, relevant=True, excluded=label == 'CLEAR', pressure=True, low_oversight=True))

    collection = [case('collection', index, 'REPORT') for index in range(320)]
    preservation = [case('preservation', index, 'REPORT' if index < 80 else 'CLEAR') for index in range(320)]
    write_rows(root / 'data/collection_a.jsonl', collection)
    write_rows(root / 'data/preservation_a.jsonl', preservation)
    write_json(root / 'configs/pilot.json', {'installation_seeds': [SEED]})
    draws, contents = [], []
    for index, row in enumerate(collection):
        for draw in range(4):
            draws.append(dict(row, id=f'{row["id"]}:draw:{draw}', case_id=row['id'], draw_index=draw))
            label = 'CLEAR' if index == 0 else 'REPORT' if index == 1 else ('CLEAR' if draw == 0 else 'REPORT')
            contents.append(generated(label))
    attempts = stage(root, f'q1_s{SEED}_collection_attempts', root / f'results/derived/q1_s{SEED}_collection_draws.jsonl', draws, contents)
    attempts_by_id = {row['id']: row for row in attempts}
    selections, inputs = [], []
    for index, row in enumerate(collection):
        failure_id = None if index == 1 else f'{row["id"]}:draw:0'
        success_id = None if index == 0 else f'{row["id"]}:draw:{0 if index == 1 else 1}'
        selections.append(dict(id=row['id'], failure_id=failure_id, success_id=success_id))
        if failure_id:
            inputs.append(reflection_input(row, attempts_by_id[failure_id], 'failure_principle'))
    write_rows(root / f'results/derived/q1_s{SEED}_collection_first_attempts.jsonl', selections)
    stage(root, f'q1_s{SEED}_failure_reflections', root / f'results/derived/q1_s{SEED}_failure_reflection_inputs.jsonl',
          inputs, [generated() for _ in inputs])
    success_outputs = stage(root, f'q1_s{SEED}_preservation_attempts', root / 'data/preservation_a.jsonl',
                            preservation, [generated(row['gold_decision'], 'length' if index == 0 else 'eos')
                                           for index, row in enumerate(preservation)])
    inputs = [reflection_input(row, output, 'preservation_principle')
              for row, output in zip(preservation[1:], success_outputs[1:])]
    stage(root, f'q1_s{SEED}_preservation_reflections', root / f'results/derived/q1_s{SEED}_preservation_reflection_inputs.jsonl',
          inputs, [generated() for _ in inputs])
    frozen_names = ['data/collection_a.jsonl', 'data/preservation_a.jsonl', 'configs/pilot.json']
    write_json(root / 'FREEZE.json', {'files': {name: sha(root / name) for name in frozen_names}})
    adapter = root / 'inputs/selected'
    adapter.mkdir(parents=True)
    (adapter / 'adapter_model.safetensors').write_bytes(b'synthetic weights')
    (adapter / 'adapter_config.json').write_bytes(b'synthetic adapter configuration')
    files = {str(p.relative_to(root)): sha(p) for p in adapter.iterdir()}
    locked = dict(candidates=[dict(locked_rank=1, pool='a', by_seed={str(SEED): dict(adapter_files=files)})], files=files)
    write_json(root / 'results/CONFIRMATION_LOCK.json', locked)
    write_json(root / 'results/MATERIALS.json', [dict(locked_rank=1, pool='a', both_structural_pass=True,
               by_seed={str(SEED): dict(status='awaiting_content_review')})])
    write_json(root / 'results/TERMINAL.json', dict(status='awaiting_content_review', repair_training_launched=False,
               freeze_sha256=sha(root / 'FREEZE.json'), lock_sha256=sha(root / 'results/CONFIRMATION_LOCK.json')))


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_denominator_independent_permutations_and_immutable_output(self):
        prepared = packets.prepare_candidate_seed(self.root, 1, SEED)
        failure = prepared['packets']['failure']
        self.assertEqual(len(failure), 319)
        self.assertEqual(sum(row['success'] is not None for row in failure), 318)
        self.assertIsNone(failure[0]['success'])
        self.assertEqual(len(prepared['packets']['preservation']), 319)
        output = self.root / 'results/content_review/seed1729'
        manifest = packets.write_packet(self.root, prepared, output)
        self.assertFalse(manifest['content_review_complete'])
        self.assertFalse(manifest['content_judgments_made'])
        for kind in ('failure', 'preservation'):
            canonical = read_rows(output / f'{kind}_canonical.jsonl')
            reviewer_a = read_rows(output / f'{kind}_reviewer_a.jsonl')
            reviewer_b = read_rows(output / f'{kind}_reviewer_b.jsonl')
            inventory = {row['id']: row_sha(row) for row in canonical}
            self.assertEqual(inventory, {row['id']: row_sha(row) for row in reviewer_a})
            self.assertEqual(inventory, {row['id']: row_sha(row) for row in reviewer_b})
            self.assertNotEqual([row['id'] for row in reviewer_a], [row['id'] for row in reviewer_b])
            self.assertEqual(reviewer_a, packets.reviewer_order(canonical, SEED, kind, 'reviewer_a'))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            packets.write_packet(self.root, prepared, output)

    def test_early_failed_terminal_is_rejected(self):
        path = self.root / 'results/TERMINAL.json'
        terminal = json.loads(path.read_text())
        terminal['status'] = 'early_failed'
        write_json(path, terminal)
        with self.assertRaisesRegex(ValueError, 'awaiting_content_review'):
            packets.prepare_candidate_seed(self.root, 1, SEED)

    def test_material_pool_must_equal_prelocked_pool(self):
        path = self.root / 'results/MATERIALS.json'
        material = json.loads(path.read_text())
        material[0]['pool'] = 'b'
        write_json(path, material)
        with self.assertRaisesRegex(ValueError, 'Material pool or seed status'):
            packets.prepare_candidate_seed(self.root, 1, SEED)

    def test_seed_material_cannot_be_incomplete(self):
        path = self.root / 'results/MATERIALS.json'
        material = json.loads(path.read_text())
        material[0]['by_seed'][str(SEED)]['status'] = 'structural_pairing_failed'
        write_json(path, material)
        with self.assertRaisesRegex(ValueError, 'Material pool or seed status'):
            packets.prepare_candidate_seed(self.root, 1, SEED)

    def test_reflection_requires_exact_frozen_scaffold(self):
        path = self.root / f'results/derived/q1_s{SEED}_failure_reflection_inputs.jsonl'
        inputs = read_rows(path)
        inputs[0]['messages'][0]['content'] = 'A changed scaffold.'
        stage(self.root, f'q1_s{SEED}_failure_reflections', path, inputs, [generated() for _ in inputs])
        with self.assertRaisesRegex(ValueError, 'exact case and saved attempt'):
            packets.prepare_candidate_seed(self.root, 1, SEED)

    def test_raw_source_row_hash_must_match_exact_generation_input(self):
        directory = self.root / f'results/stages/q1_s{SEED}_collection_attempts'
        path = directory / 'outputs.jsonl'
        outputs = read_rows(path)
        outputs[0]['source_row_sha256'] = 'tampered'
        write_rows(path, outputs)
        completion = json.loads((directory / 'COMPLETED.json').read_text())
        completion['artifacts_sha256']['outputs.jsonl'] = sha(path)
        write_json(directory / 'COMPLETED.json', completion)
        with self.assertRaisesRegex(ValueError, 'source row hash mismatch'):
            packets.prepare_candidate_seed(self.root, 1, SEED)

    def test_reflection_cannot_claim_a_different_source_generation(self):
        path = self.root / f'results/derived/q1_s{SEED}_failure_reflection_inputs.jsonl'
        inputs = read_rows(path)
        inputs[0]['source_generation_sha256'] = 'tampered'
        stage(self.root, f'q1_s{SEED}_failure_reflections', path, inputs, [generated() for _ in inputs])
        with self.assertRaisesRegex(ValueError, 'source generation ID/hash mismatch'):
            packets.prepare_candidate_seed(self.root, 1, SEED)

    def test_saved_first_attempts_cannot_drop_unpaired_failure(self):
        path = self.root / f'results/derived/q1_s{SEED}_collection_first_attempts.jsonl'
        selections = read_rows(path)
        selections[0]['failure_id'] = None
        write_rows(path, selections)
        with self.assertRaisesRegex(ValueError, 'first-attempt selection differs'):
            packets.prepare_candidate_seed(self.root, 1, SEED)

    def test_reflections_must_use_the_same_checkpoint(self):
        path = self.root / f'results/derived/q1_s{SEED}_failure_reflection_inputs.jsonl'
        inputs = read_rows(path)
        stage(self.root, f'q1_s{SEED}_failure_reflections', path, inputs, [generated() for _ in inputs], adapter='b' * 64)
        with self.assertRaisesRegex(ValueError, 'different checkpoints'):
            packets.prepare_candidate_seed(self.root, 1, SEED)

    def test_same_weights_with_changed_adapter_config_are_rejected(self):
        path = self.root / f'results/stages/q1_s{SEED}_failure_reflections/COMPLETED.json'
        completion = json.loads(path.read_text())
        completion['initial_adapter_identity']['adapter_config.json'] = 'd' * 64
        write_json(path, completion)
        with self.assertRaisesRegex(ValueError, 'different checkpoints'):
            packets.prepare_candidate_seed(self.root, 1, SEED)


if __name__ == '__main__':
    unittest.main()
