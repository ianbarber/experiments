"""CPU checks of loss alignment, batch normalization, and fixed RNG/decode rules."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import common


class TinyTokenizer:
    eos_token_id = 9
    pad_token_id = 0
    bos_token_id = 7
    all_special_ids = [0, 7, 9]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [7, 1, 2] + [3] * len(messages[-1]['content'])

    def encode(self, text, add_special_tokens):
        return [4 + ord(character) % 2 for character in text]


class TokenizationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()
        self.messages = [{'role': 'user', 'content': 'ab'}]

    def test_final_prefix_position_predicts_first_target_and_last_predicts_eos(self):
        encoded = common.encode_example(self.tokenizer, self.messages, 'xy')
        prefix = common.prefix_ids(self.tokenizer, self.messages)
        self.assertEqual(encoded['input_ids'], prefix + [4, 5])
        self.assertEqual(encoded['labels'], [-100] * (len(prefix) - 1) + [4, 5, 9])
        self.assertEqual(encoded['target_tokens'], 3)

    def test_no_silent_target_or_generation_truncation(self):
        with patch.dict(common.CONFIG, {'max_sequence_length': 5}):
            with self.assertRaisesRegex(ValueError, 'No silent truncation'):
                common.encode_example(self.tokenizer, self.messages, 'xy')

    def test_empty_and_control_token_targets_rejected(self):
        with self.assertRaises(ValueError):
            common.encode_example(self.tokenizer, self.messages, '')
        with patch.object(self.tokenizer, 'encode', return_value=[4, 9]):
            with self.assertRaisesRegex(ValueError, 'control token'):
                common.encode_example(self.tokenizer, self.messages, 'x')

    def test_sampling_seed_is_stable_per_identity_and_draw(self):
        row = {'id': 'case-1-draw-0', 'draw_index': 0}
        seed = common.row_sampling_seed(row, 1729)
        self.assertEqual(seed, common.row_sampling_seed(copy.deepcopy(row), 1729))
        self.assertNotEqual(seed, common.row_sampling_seed({**row, 'draw_index': 1}, 1729))
        self.assertNotEqual(seed, common.row_sampling_seed(row, 2718))
        self.assertEqual(15, common.row_sampling_seed({**row, 'sampling_seed': 15}, 1729))
        with self.assertRaises(ValueError):
            common.row_sampling_seed({**row, 'sampling_seed': True}, 1729)

    def test_fixed_sampling_batch_seed_binds_membership_order_draws_and_seed_base(self):
        rows = [{'id': 'a', 'draw_index': 0}, {'id': 'b', 'draw_index': 1}]
        original = common.sampling_batch_metadata(rows, 1729, 0)
        self.assertEqual(original, common.sampling_batch_metadata(copy.deepcopy(rows), 1729, 0))
        self.assertNotEqual(original['batch_seed'], common.sampling_batch_metadata(rows[::-1], 1729, 0)['batch_seed'])
        self.assertNotEqual(original['batch_seed'], common.sampling_batch_metadata(rows[:1], 1729, 0)['batch_seed'])
        self.assertNotEqual(original['batch_seed'], common.sampling_batch_metadata(rows, 2718, 0)['batch_seed'])
        self.assertNotEqual(original['batch_seed'], common.sampling_batch_metadata([rows[0], {**rows[1], 'draw_index': 2}], 1729, 0)['batch_seed'])
        with self.assertRaises(ValueError):
            common.sampling_batch_metadata([rows[0], rows[0]], 1729, 0)

    def test_stratified_order_is_one_pass_with_fixed_8_4_4_cells(self):
        import model_stage
        rows = [dict(id=str(index), is_bad=index < 16, gold_decision='REPORT' if index < 24 else 'CLEAR',
                     authored_error_category='category-' + str(index % 4)) for index in range(32)]
        order = model_stage.stratified_order(rows, 1729, 1)
        self.assertEqual(sorted(order), list(range(32)))
        self.assertEqual(order, model_stage.stratified_order(rows, 1729, 1))
        self.assertNotEqual(order, model_stage.stratified_order(rows, 1729, 2))
        self.assertNotEqual(order, model_stage.stratified_order(rows, 2718, 1))
        for start in (0, 16):
            batch = [rows[index] for index in order[start:start + 16]]
            self.assertEqual(sum(row['is_bad'] for row in batch), 8)
            self.assertEqual(sum(not row['is_bad'] and row['gold_decision'] == 'REPORT' for row in batch), 4)
            self.assertEqual(sum(not row['is_bad'] and row['gold_decision'] == 'CLEAR' for row in batch), 4)
        with self.assertRaises(ValueError):
            model_stage.stratified_order(rows[:-1], 1729, 1)
        pure = [dict(row, is_bad=True, gold_decision='REPORT') for row in rows]
        self.assertEqual(sorted(model_stage.stratified_order(pure, 1729, 1, bad_only=True)), list(range(32)))
        with self.assertRaises(ValueError):
            model_stage.stratified_order(rows, 1729, 1, bad_only=True)

    def test_decode_settings_do_not_inherit_repetition_or_top_k_defaults(self):
        sampled = common.generation_settings(self.tokenizer, True, 192)
        greedy = common.generation_settings(self.tokenizer, False, 192)
        self.assertEqual((sampled['temperature'], sampled['top_p'], sampled['top_k']), (0.7, 0.95, 0))
        for settings in (sampled, greedy):
            self.assertEqual(settings['repetition_penalty'], 1.0)
            self.assertEqual(settings['max_new_tokens'], 192)
            self.assertEqual(settings['eos_token_id'], 9)
            self.assertEqual(settings['num_beams'], 1)
        self.assertFalse(greedy['do_sample'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU torch unavailable')
class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(19)

        class PrefixSensitiveModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 6)
                self.output = torch.nn.Linear(6, 10)

            def forward(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
                values = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
                causal = values.cumsum(1)
                return types.SimpleNamespace(logits=self.output(causal[:, -logits_to_keep:]))

        self.model = PrefixSensitiveModel()
        tokenizer = TinyTokenizer()
        self.examples = [common.encode_example(tokenizer, [{'role': 'user', 'content': prompt}], target)
                         for prompt, target in [('a', 'x'), ('abcd', 'xyx')]]

    def test_padding_positions_and_masked_prefix_keep_prefix_gradients(self):
        inputs, labels = common.collate(self.examples, 0, device='cpu')
        self.assertTrue(bool(labels[inputs['attention_mask'] == 0].eq(-100).all()))
        for index, example in enumerate(self.examples):
            length = len(example['input_ids'])
            self.assertEqual(inputs['position_ids'][index, -length:].tolist(), list(range(length)))
        losses, active = common.token_losses(self.model, inputs, labels)
        self.assertEqual(int(active.sum()), sum(example['target_tokens'] for example in self.examples))
        losses.sum().backward()
        # Token 1 exists only in the prefix; a nonzero gradient demonstrates
        # that masking supervision has not detached prefix activations.
        self.assertGreater(float(self.model.embedding.weight.grad[1].abs().sum()), 0)
        self.assertEqual(float(self.model.embedding.weight.grad[0].abs().sum()), 0)

    def test_effective_batch_token_normalization_is_microbatch_invariant(self):
        import model_stage
        whole = copy.deepcopy(self.model)
        split = copy.deepcopy(self.model)
        weights = [1 / 3, 3.0]
        denominator = sum(example['target_tokens'] * weight for example, weight in zip(self.examples, weights))
        inputs, labels = common.collate(self.examples, 0, device='cpu')
        losses, _ = common.token_losses(whole, inputs, labels)
        full_loss = model_stage.weighted_loss(losses, weights, denominator)
        full_loss.backward()
        split_loss = 0.0
        for example, weight in zip(self.examples, weights):
            inputs, labels = common.collate([example], 0, device='cpu')
            losses, _ = common.token_losses(split, inputs, labels)
            loss = model_stage.weighted_loss(losses, [weight], denominator)
            loss.backward()
            split_loss += float(loss.detach())
        self.assertAlmostEqual(float(full_loss.detach()), split_loss, places=6)
        for full_parameter, split_parameter in zip(whole.parameters(), split.parameters()):
            self.torch.testing.assert_close(full_parameter.grad, split_parameter.grad, atol=1e-6, rtol=1e-6)

    def test_stratified_weighted_pass_covers_all_rows_and_resumes_adam_moments(self):
        import contextlib
        import io
        import json
        import model_stage
        torch = self.torch

        class AdapterModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_embedding = torch.nn.Embedding(10, 6)
                self.lora_output = torch.nn.Linear(6, 10)

            def forward(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
                hidden = (self.lora_embedding(input_ids) * attention_mask.unsqueeze(-1)).cumsum(1)
                return types.SimpleNamespace(logits=self.lora_output(hidden[:, -logits_to_keep:]))

            def save_pretrained(self, directory):
                directory.mkdir()
                torch.save(self.state_dict(), directory / 'adapter_model.safetensors')
                (directory / 'adapter_config.json').write_text('{}')

        initial = AdapterModel()

        def fake_load(adapter, train, seed):
            model = copy.deepcopy(initial)
            if adapter:
                # This tiny fake writes a torch archive under the production
                # identity filename. A stream avoids Torch 2.13's extension
                # dispatch to the real safetensors loader.
                with (Path(adapter) / 'adapter_model.safetensors').open('rb') as stream:
                    model.load_state_dict(torch.load(stream, weights_only=True))
            return model

        encoded = [self.examples[index % 2] for index in range(32)]
        data = [dict(id='row-' + str(index), is_bad=index < 16, gold_decision='REPORT' if index < 24 else 'CLEAR',
                     authored_error_category='category-' + str(index % 4)) for index in range(32)]
        tokenizer = TinyTokenizer()
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(model_stage, 'load_model', fake_load))
            stack.enter_context(patch.object(model_stage, 'collate',
                                            lambda examples, pad: common.collate(examples, pad, device='cpu')))
            stack.enter_context(patch.object(model_stage, 'seed_all', torch.manual_seed))
            stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
            stack.enter_context(patch.object(torch.cuda, 'empty_cache'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            root = Path(temporary)
            first = root / 'first'
            first.mkdir()
            args = types.SimpleNamespace(seed=1729, epoch=1, effective_batch=16, batch_size=8,
                                         adapter=None, optimizer=None, lr=1e-4, output=first,
                                         bad_weight=3.0, rule_variant='present', bad_only_diagnostic=False)
            manifest = {'data_sha256': 'fixture-data', 'config_sha256': 'fixture-config',
                        'initial_adapter_identity': None}
            model_stage.train(args, data, tokenizer, encoded, manifest)
            rows = common.read_jsonl(first / 'training.jsonl')
            self.assertEqual([len(row['ids']) for row in rows], [16, 16])
            self.assertCountEqual([identity for row in rows for identity in row['ids']], [row['id'] for row in data])
            self.assertEqual(manifest['target_tokens'], sum(example['target_tokens'] for example in encoded))
            groups = manifest['groups_at_processing']
            self.assertEqual(groups['bad']['examples'], 16)
            self.assertEqual(groups['good_report']['examples'], 8)
            self.assertEqual(groups['good_clear']['examples'], 8)
            self.assertEqual(groups['bad']['weighted_target_tokens'], 3 * groups['bad']['target_tokens'])
            self.assertAlmostEqual(groups['all']['loss_sum'], groups['bad']['loss_sum'] + groups['good']['loss_sum'])
            saved = torch.load(first / 'optimizer.pt', weights_only=True)
            self.assertEqual({int(state['step']) for state in saved['optimizer_state_dict']['state'].values()}, {2})
            self.assertTrue(all(bool(state['exp_avg'].abs().sum() > 0)
                                for state in saved['optimizer_state_dict']['state'].values()))
            second = root / 'second'
            second.mkdir()
            args.output, args.epoch = second, 2
            args.adapter, args.optimizer = first / 'adapter', first / 'optimizer.pt'
            manifest2 = {'data_sha256': 'fixture-data', 'config_sha256': 'fixture-config',
                         'initial_adapter_identity': model_stage.adapter_identity(args.adapter)}
            model_stage.train(args, data, tokenizer, encoded, manifest2)
            resumed = torch.load(second / 'optimizer.pt', weights_only=True)
            self.assertEqual(resumed['cumulative_steps'], 4)
            self.assertEqual({int(state['step']) for state in resumed['optimizer_state_dict']['state'].values()}, {4})
            self.assertTrue(manifest2['optimizer_resumed'])
            self.assertEqual(manifest2['initial_trainable_parameter_sha256'],
                             manifest['final_trainable_parameter_sha256'])

            bad = root / 'mismatched'
            bad.mkdir()
            args.output = bad
            mismatched = dict(manifest2, data_sha256='a-different-curriculum')
            with self.assertRaisesRegex(ValueError, 'Optimizer continuity mismatch'):
                model_stage.train(args, data, tokenizer, encoded, mismatched)

    def test_teacher_forced_diagnostics_partition_bad_good_without_gradients(self):
        import json
        import model_stage
        data = [dict(id='bad', is_bad=True, gold_decision='REPORT', authored_error_category='goal'),
                dict(id='good', is_bad=False, gold_decision='CLEAR')]
        before = [value.detach().clone() for value in self.model.parameters()]
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(output=Path(temporary), batch_size=2, bad_weight=3.0)
            with patch.object(model_stage, 'collate', lambda examples, pad: common.collate(examples, pad, device='cpu')):
                summary = model_stage.target_loss_diagnostics(args, self.model, TinyTokenizer(), data, self.examples)
            rows = common.read_jsonl(Path(temporary) / 'target_losses.jsonl')
            self.assertEqual([row['target_tokens'] for row in rows], [example['target_tokens'] for example in self.examples])
            groups = summary['groups']
            self.assertAlmostEqual(groups['all']['loss_sum'], groups['bad']['loss_sum'] + groups['good']['loss_sum'])
            self.assertEqual(groups['bad_category:goal']['loss_sum'], groups['bad']['loss_sum'])
            self.assertEqual(groups['good_clear']['loss_sum'], groups['good']['loss_sum'])
        for prior, current in zip(before, self.model.parameters()):
            self.torch.testing.assert_close(prior, current)
            self.assertIsNone(current.grad)

    def test_fixed_batch_generation_records_exact_membership_and_source_cohort(self):
        import contextlib
        import io
        import model_stage
        data = [dict(id='case-' + str(index), case_id='shared-' + str(index // 4), draw_index=index % 4,
                     prompt='case facts', gold_decision='REPORT') for index in range(20)]
        calls = []

        def fake_generate(model, tok, prompts, max_new_tokens, sample):
            calls.append(len(prompts))
            return [dict(text='A factual reason. <decision>REPORT</decision>', text_with_special_tokens='raw',
                         token_ids=[1, 9], generated_tokens=2, finish_reason='eos', unexpected_special_token_ids=[])
                    for _ in prompts]

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            args = types.SimpleNamespace(output=Path(temporary), sample=True, seed=1729, batch_size=32, max_new_tokens=192)
            manifest = dict(initial_adapter_identity={'adapter_model.safetensors': 'adapter'}, generation_files={'cohort': 'data/cohort.jsonl'})
            with patch.object(model_stage, 'generate_batch', fake_generate), patch.object(model_stage, 'seed_all') as seeded:
                summary = model_stage.generate_sets(args, object(), TinyTokenizer(), [('cohort', data)], manifest)
            self.assertEqual(calls, [16, 4])
            self.assertEqual(seeded.call_count, 2)
            rows = common.read_jsonl(Path(temporary) / 'outputs.jsonl')
            self.assertEqual(summary['outputs'], 20)
            for index, row in enumerate(rows):
                begin = (index // 16) * 16
                expected = common.sampling_batch_metadata(data[begin:begin + 16], 1729, index // 16)
                self.assertEqual(row['sampling'], expected)
                self.assertEqual(row['source_row_sha256'], common.sha_row(data[index]))
                self.assertEqual(row['source_file'], 'data/cohort.jsonl')
                self.assertEqual(row['cohort'], 'cohort')


if __name__ == '__main__':
    unittest.main()
