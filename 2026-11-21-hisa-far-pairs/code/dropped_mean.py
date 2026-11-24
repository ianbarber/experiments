"""Zero-mean check for the parts HiSA drops (the 'zeroing is the right ablation only if ...' pitfall).

For every layer, query head, pair l and distance d, the mean over queries of pair l's score part
    c_l[i, i-d] = q_i[l] k_{i-d}[l] + q_i[l+P] k_{i-d}[l+P]        (q, k after RoPE)
is a cross-correlation, computed by FFT; so is its mean square. Data: 64 prose windows not used in Stage 1.

Writes ../raw/dropped_mean.pt  (mean [L, H, P, T], fp32) and ../results/dropped_mean_summary.json.
python dropped_mean.py
"""

import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import keeptable as kt

MODEL = "HuggingFaceTB/SmolLM-135M"
T, P = 2048, 32
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
theta = model.model.rotary_emb.inv_freq.detach().cpu()
ids = torch.from_numpy(np.load("../data/prose-stats.npz")["ids"]).long()
L, H = model.config.num_hidden_layers, model.config.num_attention_heads
G = H // model.config.num_key_value_heads

sum1 = torch.zeros(L, H, P, T, dtype=torch.float64, device="cuda")
sum2 = torch.zeros_like(sum1)


def xcorr(a, b):
    """sum_i a[i] b[i - d] for d = 0..T-1 (last dim)."""
    n = 2 * T
    fa, fb = torch.fft.rfft(a, n=n), torch.fft.rfft(b, n=n)
    return torch.fft.irfft(fa * fb.conj(), n=n)[..., :T]


def hook(li):
    def pre(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        cos, sin = kwargs["position_embeddings"]
        B = h.shape[0]
        q = mod.q_proj(h).view(B, T, H, 2 * P).transpose(1, 2)
        k = mod.k_proj(h).view(B, T, H // G, 2 * P).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = k.repeat_interleave(G, 1)
        q, k = q.double().transpose(-1, -2), k.double().transpose(-1, -2)      # [B, H, 64, T]
        q1, q2, k1, k2 = q[:, :, :P], q[:, :, P:], k[:, :, :P], k[:, :, P:]
        sum1[li] += (xcorr(q1, k1) + xcorr(q2, k2)).sum(0)
        sum2[li] += (xcorr(q1 ** 2, k1 ** 2) + xcorr(q2 ** 2, k2 ** 2) + 2 * xcorr(q1 * q2, k1 * k2)).sum(0)
    return pre


for li, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_pre_hook(hook(li), with_kwargs=True)
with torch.no_grad():
    for i in range(0, len(ids), 8):
        model(ids[i:i + 8].cuda())

count = (len(ids) * (T - torch.arange(T, device="cuda"))).double()            # queries available at distance d
mean = sum1 / count
rms = (sum2 / count).clamp(min=0).sqrt()
torch.save(mean.float().cpu(), "../raw/dropped_mean.pt")

# summary over the region hisa-90 drops, by rotation band; scores are parts / 8
ang = torch.arange(T)[None, :].double() * theta.double()[:, None] * 180 / np.pi   # [P, T] degrees
bands = {"<45": (0, 45), "45-90": (45, 90), "90-180": (90, 180), "180-360": (180, 360), "360-720": (360, 720),
         "720+": (720, float("inf"))}
ok = torch.arange(T) <= T - 256                    # at least 64 * 256 queries behind every mean
summ = {}
m, r = mean.cpu(), rms.cpu()
for name, (lo, hi) in bands.items():
    sel = (ang >= lo) & (ang < hi) & ok[None]      # [P, T]
    mm, rr = m[:, :, sel], r[:, :, sel]            # [L, H, n]
    ratio = mm.abs() / rr.clamp(min=1e-12)
    summ[name] = {
        "n_pair_distance_cells": int(sel.sum()),
        "rms_of_part_over_8": float((rr ** 2).mean().sqrt() / 8),
        "rms_of_mean_over_8": float((mm ** 2).mean().sqrt() / 8),
        "median_abs_mean_over_rms": float(ratio.median()),
        "share_of_cells_abs_mean_over_rms_gt_0.5": float((ratio > 0.5).double().mean()),
        "share_of_part_energy_in_mean": float((mm ** 2).sum() / (rr ** 2).sum()),
    }
    print(name, summ[name], flush=True)
# the mean-ablation bias for hisa-90: sum of means of dropped pairs, per layer/head/distance, in score units
M = kt.make_table("hisa-90", theta, T)
bias = (m * (~M)[None, None]).sum(2) / 8           # [L, H, T]
summ["hisa90_bias_abs_max"] = float(bias[..., ok].abs().max())
summ["hisa90_bias_rms"] = float((bias[..., ok] ** 2).mean().sqrt())
summ["hisa90_bias_rms_by_layer"] = [float((bias[l][..., ok] ** 2).mean().sqrt()) for l in range(L)]
print({k_: v for k_, v in summ.items() if k_.startswith("hisa90")})
json.dump(summ, open("../results/dropped_mean_summary.json", "w"), indent=1)
