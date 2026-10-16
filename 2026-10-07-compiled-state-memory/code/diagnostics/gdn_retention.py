"""Measured retention of GDN writes on real text (decay alone, and decay + delta-rule erasure).

For a write at position i in head h, the gated delta rule maps the state as
    S_j = S_{j-1} a_j (I - b_j k_j k_j^T) + b_j v_j k_j^T,   a_j = exp(g_j)
so the write v_i k_i^T survives to position T as v_i (k_i^T M_{i->T}), M = prod_j a_j (I - b_j k_j k_j^T).
We report, per distance d = T - i:
    decay-only retention   rho_decay = prod_{j>i} a_j
    full retention         rho_full  = || k_i^T M_{i->T} ||   (k L2-normalised as in the kernel)
This is the survival of a write's *storage*, not whether a later query reads it.

Usage (inside the image): python diagnostics/gdn_retention.py --model M --out results/x.json
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from statelm.core import LM  # noqa: E402
from statelm.prompts import compile_text  # noqa: E402
from statelm.synth import make_memory  # noqa: E402

DISTS = [16, 64, 256, 1024, 2048, 4096, 8192, 16384]


def retention_for_layer(g, beta, k, dists):
    """g, beta: [T,H] ; k: [T,H,K] (unnormalised). Returns dict d -> (rho_decay[H], rho_full[H])."""
    T, H = g.shape
    k = F.normalize(k.float(), dim=-1)
    g, beta = g.float(), beta.float()
    G = torch.cumsum(g, 0)
    out = {}
    ds = [d for d in dists if d < T]
    if not ds:
        return out
    starts = torch.tensor([T - 1 - d for d in ds], device=g.device)        # write positions
    u = k[starts].clone()                                                   # [S,H,K]
    a = g.exp()
    s0 = int(starts.min())
    for j in range(s0 + 1, T):
        active = (starts < j).view(-1, 1, 1)
        kj = k[j]                                                           # [H,K]
        proj = (u * kj).sum(-1, keepdim=True)                               # [S,H,1]
        upd = a[j].view(1, H, 1) * (u - beta[j].view(1, H, 1) * proj * kj)
        u = torch.where(active, upd, u)
    full = u.norm(dim=-1)                                                   # [S,H]
    for si, d in enumerate(ds):
        i = T - 1 - d
        out[d] = ((G[T - 1] - G[i]).exp().cpu().numpy(), full[si].cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lm = LM(args.model)

    texts = {}
    for n in [60, 250, 600]:
        texts[f"synthetic_memory_{n}"] = compile_text(lm.tok, make_memory("dog", "Biscuit", n, n // 2, seed=42))
    card = open("/work/data/raw/text/qwen38_model_card.md").read()
    texts["model_card_prose"] = compile_text(lm.tok, card[:60000])

    captured = {}

    def wrap(layer_idx, fn):
        def inner(q, k, v, g=None, beta=None, **kw):
            captured[layer_idx] = (g[0].detach(), beta[0].detach(), k[0].detach())
            return fn(q, k, v, g=g, beta=beta, **kw)
        return inner

    for i in lm.gdn_layers:
        la = lm.tm.layers[i].linear_attn
        la.chunk_gated_delta_rule = wrap(i, la.chunk_gated_delta_rule)

    results = {"model": args.model, "texts": {}}
    for name, text in texts.items():
        ids = lm.ids(text)[:, :20000]
        T = ids.shape[1]
        captured.clear()
        lm.forward(ids, lm.new_cache(), chunk=ids.shape[1])   # single chunk so each hook sees the whole text
        per_d = {}
        for li in lm.gdn_layers:
            g, beta, k = captured[li]
            for d, (rd, rf) in retention_for_layer(g, beta, k, DISTS).items():
                per_d.setdefault(d, ([], []))
                per_d[d][0].append(rd)
                per_d[d][1].append(rf)
        summ = {}
        for d, (rds, rfs) in sorted(per_d.items()):
            rd, rf = np.concatenate(rds), np.concatenate(rfs)
            summ[d] = dict(decay_median=float(np.median(rd)), decay_frac_gt_0p5=float((rd > 0.5).mean()),
                           decay_frac_gt_0p1=float((rd > 0.1).mean()),
                           full_median=float(np.median(rf)), full_frac_gt_0p5=float((rf > 0.5).mean()),
                           full_frac_gt_0p1=float((rf > 0.1).mean()), full_p90=float(np.quantile(rf, 0.9)))
        # per-layer: fraction of heads with full retention > 0.1 at d=1024
        layer_1k = {}
        if 1024 in per_d:
            for li, rf in zip(lm.gdn_layers, per_d[1024][1]):
                layer_1k[li] = float((rf > 0.1).mean())
        results["texts"][name] = {"n_tokens": T, "by_distance": summ, "layer_frac_full_gt_0p1_at_1024": layer_1k}
        print(name, T)
        for d, s in summ.items():
            print(f"  d={d:>5}  decay: med {s['decay_median']:.3g} >.5 {s['decay_frac_gt_0p5']:.3f} >.1 "
                  f"{s['decay_frac_gt_0p1']:.3f} | full: med {s['full_median']:.3g} >.5 {s['full_frac_gt_0p5']:.3f}"
                  f" >.1 {s['full_frac_gt_0p1']:.3f} p90 {s['full_p90']:.3g}")
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
