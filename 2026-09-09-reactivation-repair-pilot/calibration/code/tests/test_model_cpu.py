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
        whole = copy.deepcopy(self.model)
        split = copy.deepcopy(self.model)
        denominator = sum(example['target_tokens'] for example in self.examples)
        inputs, labels = common.collate(self.examples, 0, device='cpu')
        losses, _ = common.token_losses(whole, inputs, labels)
        full_loss = losses.sum() / denominator
        full_loss.backward()
        split_loss = 0.0
        for example in self.examples:
            inputs, labels = common.collate([example], 0, device='cpu')
            losses, _ = common.token_losses(split, inputs, labels)
            loss = losses.sum() / denominator
            loss.backward()
            split_loss += float(loss.detach())
        self.assertAlmostEqual(float(full_loss.detach()), split_loss, places=6)
        for full_parameter, split_parameter in zip(whole.parameters(), split.parameters()):
            self.torch.testing.assert_close(full_parameter.grad, split_parameter.grad, atol=1e-6, rtol=1e-6)

    def test_single_pass_covers_short_final_batch_and_resumes_adam_moments(self):
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

        encoded = self.examples + [self.examples[0]]
        data = [{'id': 'row-' + str(index)} for index in range(3)]
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
            args = types.SimpleNamespace(seed=1729, epoch=1, effective_batch=2, batch_size=1,
                                         adapter=None, optimizer=None, lr=1e-4, output=first)
            manifest = {'data_sha256': 'fixture-data', 'config_sha256': 'fixture-config',
                        'initial_adapter_identity': None}
            model_stage.train(args, data, tokenizer, encoded, manifest)
            rows = common.read_jsonl(first / 'training.jsonl')
            self.assertEqual([row['examples'] for row in rows], [2, 1])
            self.assertCountEqual([identity for row in rows for identity in row['ids']], [row['id'] for row in data])
            self.assertEqual(manifest['target_tokens'], sum(example['target_tokens'] for example in encoded))
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
            with self.assertRaisesRegex(ValueError, 'Optimizer continuity mismatch: data_sha256'):
                model_stage.train(args, data, tokenizer, encoded, mismatched)


if __name__ == '__main__':
    unittest.main()
