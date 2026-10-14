from types import SimpleNamespace

import pytest
import torch

from fwlayer.causal import (FP32LinearAttentionLayer, exchange_state, exchange_value_kv,
                           reset_history, restrict_global_history)
from fwlayer.observe import patch_query


def cache_fixture():
    recurrent = SimpleNamespace(recurrent_states=torch.ones(1, 2, 3, 3),
                                conv_states=torch.ones(1, 4, 4))
    global_ = SimpleNamespace(keys=torch.arange(20.).view(1, 1, 5, 4),
                              values=torch.arange(20., 40.).view(1, 1, 5, 4))
    return SimpleNamespace(layers=[recurrent, global_])


def test_state_exchange_isolated_and_originals_preserved():
    destination, source = cache_fixture(), cache_fixture()
    source.layers[0].recurrent_states.fill_(7)
    patched = exchange_state(destination, source, 0)
    assert torch.all(patched.layers[0].recurrent_states == 7)
    assert torch.all(destination.layers[0].recurrent_states == 1)
    assert torch.equal(patched.layers[0].conv_states, destination.layers[0].conv_states)
    cleared = reset_history(destination, 0)
    assert not cleared.layers[0].recurrent_states.any()
    assert not cleared.layers[0].conv_states.any()
    assert torch.equal(cleared.layers[1].keys, destination.layers[1].keys)


def test_value_kv_exchange_preserves_all_other_positions():
    destination, source = cache_fixture(), cache_fixture()
    source.layers[1].keys.add_(100)
    source.layers[1].values.add_(100)
    patched = exchange_value_kv(destination, source, 1, [[1], [3]])
    for field in ("keys", "values"):
        p, a, b = (getattr(c.layers[1], field) for c in (patched, destination, source))
        assert torch.equal(p[:, :, [1, 3]], b[:, :, [1, 3]])
        assert torch.equal(p[:, :, [0, 2, 4]], a[:, :, [0, 2, 4]])
        assert torch.all(b - a == 100)


def test_recurrent_storage_does_not_inherit_bf16_conv_dtype():
    layer = FP32LinearAttentionLayer()
    layer.update_conv_state(torch.zeros(1, 4, 4, dtype=torch.bfloat16))
    state = torch.tensor([[[[1.0001]]]], dtype=torch.float32)
    layer.update_recurrent_state(state)
    assert layer.recurrent_states.dtype == torch.float32
    assert torch.equal(layer.recurrent_states, state)


class Echo(torch.nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return hidden_states if attention_mask is None else attention_mask


def test_query_patch_preserves_earlier_positions_and_cleans_up_after_error():
    layer = Echo()
    layer.mlp, layer.self_attn = Echo(), Echo()
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    hidden = torch.ones(1, 4, 3)
    with pytest.raises(RuntimeError):
        with patch_query(model, 0, "residual", torch.tensor([2., 3., 4.])):
            patched = layer(hidden)
            assert torch.equal(patched[:, :-1], hidden[:, :-1])
            assert torch.equal(patched[0, -1], torch.tensor([2., 3., 4.]))
            raise RuntimeError("exercise hook cleanup")
    assert torch.equal(layer(hidden), hidden)


def test_global_history_mask_preserves_absolute_positions_and_causality():
    module = Echo()
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=module)]))
    hidden = torch.ones(1, 2, 3)
    with restrict_global_history(model, 0, 4):
        mask = module(hidden_states=hidden)
        assert mask.shape == (1, 1, 2, 6)
        assert torch.isneginf(mask[0, 0, :, :4]).all()
        assert mask[0, 0, 0, 4] == 0
        assert torch.isneginf(mask[0, 0, 0, 5])
        assert (mask[0, 0, 1, 4:] == 0).all()
    assert torch.equal(module(hidden), hidden)
