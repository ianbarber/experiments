from types import SimpleNamespace

import torch

from fwlayer.observe import TraceCapture
from fwlayer.report import causal_stats


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = torch.nn.Identity()

    def forward(self, hidden):
        mixed = hidden + self.linear_attn(hidden)
        return mixed + self.mlp(self.post_attention_layernorm(mixed))


def test_cached_capture_only_collects_last_position():
    block = Block()
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    pair = SimpleNamespace(source_positions=[100, 200], query_position=300,
                           value_positions=[[99, 100], [199, 200]])
    hidden = torch.arange(12.).view(1, 4, 3)
    with TraceCapture(model, pair, query_only=True) as capture:
        output = block(hidden)
    traces = capture.tensors()
    assert traces["after"].shape == (1, 1, 3)
    assert torch.equal(traces["after"][0, 0], output[0, -1])
    assert traces["attention_value_mass"].numel() == 0


def test_bootstrap_keeps_examples_from_different_seeds_independent():
    rows = [{"pair_id": f"s{seed}-d128-p000", "seed": seed,
             "fraction_of_counterfactual_gap": .5, "correct_margin_change": -.5}
            for seed in (42, 314159)]
    stats = causal_stats(rows)
    assert stats["n_example_clusters"] == 2
