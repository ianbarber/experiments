"""Small activation captures and normalized logit-lens measurements."""
from __future__ import annotations

from contextlib import contextmanager
import time

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv


def tensor_output(output):
    return output[0] if isinstance(output, tuple) else output


def margin(logits, token_a: int, token_b: int) -> float:
    last = logits[0, -1].float()
    return float(last[token_a] - last[token_b])


class TraceCapture:
    def __init__(self, model, pair, query_only=False):
        self.model = model
        self.positions = [-1] if query_only else pair.source_positions + [pair.query_position]
        self.query_only = query_only
        self.value_positions = pair.value_positions
        self.handles = []
        self.data = {name: {} for name in ("before", "mixer", "after_mixer", "mlp", "after")}
        self.attention = {}

    def select(self, hidden):
        return hidden[0, self.positions, :].detach().cpu().clone()

    def __enter__(self):
        for i, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(
                lambda m, args, idx=i: self.data["before"].__setitem__(idx, self.select(args[0]))))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                lambda m, args, idx=i: self.data["after_mixer"].__setitem__(idx, self.select(args[0]))))
            for name, module in (("after", layer), ("mlp", layer.mlp),
                                 ("mixer", layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn)):
                self.handles.append(module.register_forward_hook(
                    lambda m, args, out, idx=i, stream=name:
                    self.data[stream].__setitem__(idx, self.select(tensor_output(out)))))
            if hasattr(layer, "self_attn") and not self.query_only:
                self.handles.append(layer.self_attn.register_forward_pre_hook(
                    lambda m, args, kwargs, idx=i: self.attention_row(
                        idx, m, kwargs.get("hidden_states", args[0] if args else None), kwargs),
                    with_kwargs=True))
        return self

    def attention_row(self, idx, module, hidden, kwargs):
        """Recompute just the final query row, never an L x L attention matrix."""
        b, length, _ = hidden.shape
        q, _ = module.q_proj(hidden[:, -1:, :]).view(b, 1, -1, 2 * module.head_dim).chunk(2, dim=-1)
        q = module.q_norm(q).transpose(1, 2)
        k = module.k_norm(module.k_proj(hidden).view(b, length, -1, module.head_dim)).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        q, _ = apply_rotary_pos_emb(q, q, cos[:, -1:], sin[:, -1:])
        k, _ = apply_rotary_pos_emb(k, k, cos, sin)
        k = repeat_kv(k, module.num_key_value_groups)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * module.scaling
        weights = scores.softmax(dim=-1)[0, :, 0, :]
        self.attention[idx] = torch.stack([weights[:, p].sum(-1) for p in self.value_positions], -1).cpu()

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()

    def tensors(self):
        return {name: torch.stack([values[i] for i in range(len(self.model.model.layers))])
                for name, values in self.data.items()} | {
                    "attention_value_mass": torch.stack([self.attention[i] for i in sorted(self.attention)]) if self.attention else torch.empty(0),
                    "attention_layers": torch.tensor(sorted(self.attention)),
                }


@contextmanager
def patch_query(model, layer_index: int, component: str, vector):
    layer = model.model.layers[layer_index]
    modules = {"residual": layer, "mixer": layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn,
               "mlp": layer.mlp}
    module = modules[component]

    def patch(m, args, output):
        hidden = tensor_output(output).clone()
        hidden[:, -1, :] = vector.to(device=hidden.device, dtype=hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    handle = module.register_forward_hook(patch)
    try:
        yield
    finally:
        handle.remove()


def forward_score(model, ids, pair, capture=False):
    inputs = torch.tensor([ids], device=model.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        if capture:
            with TraceCapture(model, pair) as recorder:
                out = model(input_ids=inputs, use_cache=False, logits_to_keep=1)
            tensors = recorder.tensors()
        else:
            out = model(input_ids=inputs, use_cache=False, logits_to_keep=1)
            tensors = None
    torch.cuda.synchronize()
    last = out.logits[0, -1].float()
    record = {
        "margin_a_minus_b": margin(out.logits, pair.token_a, pair.token_b),
        "argmax_token": int(last.argmax()),
        "probability_a": float(last.softmax(-1)[pair.token_a]),
        "probability_b": float(last.softmax(-1)[pair.token_b]),
        "seconds": time.perf_counter() - start,
    }
    return record, tensors


def greedy_answer(model, tokenizer, ids):
    inputs = torch.tensor([ids], device=model.device)
    with torch.inference_mode():
        outputs = model.generate(input_ids=inputs, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id,
                                 temperature=None, top_p=None, top_k=None)
    return tokenizer.decode(outputs[0, inputs.shape[-1]:], skip_special_tokens=True)


def summarize_trace(model, pair, tensors):
    direction = (model.lm_head.weight[pair.token_a].float() - model.lm_head.weight[pair.token_b].float())

    def lens(hidden):
        with torch.inference_mode():
            normed = model.model.norm(hidden.to(device=model.device, dtype=model.dtype)).float()
            return (normed * direction).sum(-1).cpu()

    before, middle, after = (lens(tensors[name][:, -1]) for name in ("before", "after_mixer", "after"))
    out = {"lens_before": before.tolist(), "lens_after_mixer": middle.tolist(),
           "lens_after": after.tolist(), "lens_mixer_change": (middle - before).tolist(),
           "lens_mlp_change": (after - middle).tolist()}
    for name in ("mixer", "mlp"):
        update = tensors[name][:, -1].float()
        out[name + "_update_norm"] = update.norm(dim=-1).tolist()
        out[name + "_raw_unembedding_projection"] = (update * direction.cpu()).sum(-1).tolist()
    out["attention_layers"] = tensors["attention_layers"].tolist()
    out["attention_value_mass"] = tensors["attention_value_mass"].tolist()
    return out


def source_sensitivity(a, b):
    delta = (a["after"].float() - b["after"].float())
    scale = (a["after"].float().norm(dim=-1) + b["after"].float().norm(dim=-1)) / 2
    return {"paired_difference_norm": delta.norm(dim=-1).tolist(),
            "relative_paired_difference": (delta.norm(dim=-1) / scale.clamp_min(1e-8)).tolist()}
