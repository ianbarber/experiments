"""Independent CPU algebra fixtures for frozen loss code, with no model loading."""
import ast
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from objective import effective_denominator, weighted_microbatch_loss

assert not torch.cuda.is_initialized()
tree = ast.parse((ROOT / 'scripts/common.py').read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name in ('prefix_ids', 'encode_example', 'collate', 'token_losses')]
ns = {'torch': torch, 'CONFIG': {'max_sequence_length': 2048}}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'frozen_common_functions', 'exec'), ns)

class FixtureTokenizer:
    eos_token_id = 9
    def apply_chat_template(self, messages, **kwargs):
        return messages
    def encode(self, target, **kwargs):
        return target

encoded = [ns['encode_example'](FixtureTokenizer(), [1, 2, 3], [4, 5]),
           ns['encode_example'](FixtureTokenizer(), [2, 3], [6]),
           ns['encode_example'](FixtureTokenizer(), [1, 2, 3, 4, 5], [6, 7, 8])]
assert encoded[0]['input_ids'] == [1, 2, 3, 4, 5]
assert encoded[0]['labels'] == [-100, -100, 4, 5, 9]
assert encoded[1]['input_ids'] == [2, 3, 6]
assert encoded[1]['labels'] == [-100, 6, 9]
inputs, labels = ns['collate'](encoded, pad=0, device='cpu')
assert inputs['input_ids'].tolist()[1] == [0, 0, 0, 0, 0, 2, 3, 6]
assert inputs['position_ids'].tolist()[1] == [0, 0, 0, 0, 0, 0, 1, 2]
assert labels.ne(-100).sum().item() == 9
torch.manual_seed(112)
parameter = torch.nn.Parameter(torch.randn(3, 8, 10, dtype=torch.float64))

class SuffixLogitsFixture:
    def __call__(self, **kwargs):
        assert kwargs['use_cache'] is False
        assert 'labels' not in kwargs
        return SimpleNamespace(logits=parameter[:, -kwargs['logits_to_keep']:])

actual, mask = ns['token_losses'](SuffixLogitsFixture(), inputs, labels)
reference = torch.nn.functional.cross_entropy(parameter.float().reshape(-1, 10), labels.reshape(-1),
                                              reduction='none', ignore_index=-100).reshape(3, 8)
assert actual.shape == (3, 4)
torch.testing.assert_close(actual.sum(), reference.sum(), rtol=1e-6, atol=1e-6)
assert mask.sum().item() == 9
results = []
for weights in ([1., 1., 1.], [.75, 1.5, .75]):
    examples = [dict(e, loss_weight=w) for e, w in zip(encoded, weights)]
    denominator = effective_denominator(examples)
    # Compare two microbatches (2 rows then 1) to independently constructed whole-batch loss.
    split_loss = (weighted_microbatch_loss(actual[:2], weights[:2], denominator)
                  + weighted_microbatch_loss(actual[2:], weights[2:], denominator))
    expected = (reference.sum(1) * reference.new_tensor(weights)).sum() / denominator
    torch.testing.assert_close(split_loss, expected, rtol=1e-6, atol=1e-6)
    split_grad, = torch.autograd.grad(split_loss, parameter, retain_graph=True)
    reference_grad, = torch.autograd.grad(expected, parameter, retain_graph=True)
    torch.testing.assert_close(split_grad, reference_grad, rtol=1e-6, atol=1e-7)
    assert split_grad[labels == -100].abs().sum().item() == 0
    results.append({'weights': weights, 'denominator': denominator,
                    'loss': float(split_loss.detach()),
                    'maximum_gradient_difference': float((split_grad-reference_grad).abs().max()),
                    'ignored_logit_gradient_sum': float(split_grad[labels == -100].abs().sum())})
# One processed-score example proves the inherited penalty can change greedy choice.
raw = torch.tensor([2.0, 1.95, -1.0])
processed = raw.clone(); processed[0] /= 1.05
assert raw.argmax().item() == 0 and processed.argmax().item() == 1
assert not torch.cuda.is_initialized()
print(json.dumps({'status': 'passed', 'fixture_only_no_model': True, 'torch': torch.__version__,
                  'cuda_initialized': torch.cuda.is_initialized(), 'cases': results,
                  'repetition_penalty_example': {'raw': raw.tolist(), 'processed': processed.tolist(),
                                                'raw_argmax': 0, 'processed_argmax': 1}}, indent=2))
