"""Causal interventions with identical cached baselines and self-patch controls."""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from transformers.cache_utils import DynamicCache, LinearAttentionLayer

from .observe import margin


class FP32LinearAttentionLayer(LinearAttentionLayer):
    """Keep FLA's final recurrent state in FP32 instead of inheriting the conv dtype."""
    def lazy_initialization(self, conv_states=None, recurrent_states=None):
        if conv_states is not None:
            super().lazy_initialization(conv_states=conv_states)
        if recurrent_states is not None:
            self.recurrent_states = torch.zeros_like(recurrent_states, dtype=torch.float32)
            self.is_recurrent_states_initialized = True


def new_cache(model):
    cache = DynamicCache(config=model.config)
    for i, kind in enumerate(model.config.layer_types):
        if kind == "linear_attention":
            cache.layers[i] = FP32LinearAttentionLayer()
    return cache


def prefix_cache(model, ids, boundary):
    cache = new_cache(model)
    inputs = torch.tensor([ids[:boundary]], device=model.device)
    with torch.inference_mode():
        output = model(input_ids=inputs, use_cache=True, past_key_values=cache, logits_to_keep=1)
    return output.past_key_values


def continuation_score(model, ids, boundary, cache, pair):
    # Copies prevent in-place cache updates in one trial from contaminating another.
    cache = copy.deepcopy(cache)
    inputs = torch.tensor([ids[boundary:]], device=model.device)
    positions = torch.arange(boundary, len(ids), device=model.device)[None, :]
    with torch.inference_mode():
        out = model(input_ids=inputs, position_ids=positions, use_cache=True,
                    past_key_values=cache, logits_to_keep=1)
    return {"margin_a_minus_b": margin(out.logits, pair.token_a, pair.token_b),
            "argmax_token": int(out.logits[0, -1].argmax())}


def exchange_state(cache, donor, layer_index):
    result = copy.deepcopy(cache)
    result.layers[layer_index].recurrent_states.copy_(donor.layers[layer_index].recurrent_states)
    return result


def exchange_value_kv(cache, donor, layer_index, value_positions):
    result = copy.deepcopy(cache)
    indices = [p for span in value_positions for p in span]
    target, source = result.layers[layer_index], donor.layers[layer_index]
    target.keys[:, :, indices, :] = source.keys[:, :, indices, :]
    target.values[:, :, indices, :] = source.values[:, :, indices, :]
    return result


def reset_history(cache, layer_index):
    result = copy.deepcopy(cache)
    layer = result.layers[layer_index]
    layer.recurrent_states.zero_()
    layer.conv_states.zero_()
    return result


@contextmanager
def restrict_global_history(model, layer_index, boundary):
    """Remove prefix access only at this global layer; preserve cache positions."""
    module = model.model.layers[layer_index].self_attn

    def restrict(m, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        length = hidden.shape[1]
        queries = torch.arange(boundary, boundary + length, device=hidden.device)
        keys = torch.arange(boundary + length, device=hidden.device)
        allowed = (keys[None, :] <= queries[:, None]) & (keys[None, :] >= boundary)
        mask = torch.zeros((1, 1, length, boundary + length), dtype=hidden.dtype, device=hidden.device)
        mask.masked_fill_(~allowed[None, None], float("-inf"))
        existing = kwargs.get("attention_mask")
        if existing is not None:
            mask = mask.masked_fill(~existing, float("-inf")) if existing.dtype == torch.bool else mask + existing
        kwargs["attention_mask"] = mask
        return args, kwargs

    handle = module.register_forward_pre_hook(restrict, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def effect(patched, destination, source):
    delta = patched - destination
    denominator = source - destination
    return {"margin_delta": delta,
            "fraction_of_counterfactual_gap": delta / denominator if abs(denominator) >= 0.5 else None}
