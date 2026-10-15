"""Storage probe analysis (exploratory). Is the user's value linearly recoverable from the compiled
GDN state better than the model itself reads it out?

uv run --with numpy --with scipy --with torch python analysis/storage_probe.py \
    results/phase1/storage_probe_<tag>.jsonl data/probe

Design: per attribute x task, 16 counterfactual pairs (state 2k: user holds X; 2k+1: user holds Y;
same filler and positions within a pair, different across pairs). Each head's Gram is divided by
its mean diagonal, so heads count equally; a feature set (one GDN layer, the late third, all) sums
its heads' Grams. Probe = difference of means: w = sum over training pairs of (s_X - s_Y), learned
leaving one pair out; everything is computed from Grams.
  paired    held-out pair scored correct if <s_X - s_Y, w> > 0
  single    each held-out state classified alone: X if <s, w> > <mean of training states, w>
  d'        separation of the held-out single-state projections (centred per fold)
Model readout on the same states: d = log p(X_j) - log p(Y_j) under the injected state.
  paired    shift d(s_X) - d(s_Y) > 0
  single    leave-pair-out midpoint threshold on d
  d'        (mean d_X - mean d_Y) / pooled SD
Null: 200 random within-pair label swaps (probe paired / single accuracy, all-layers set).
Nested selection (added after the first read of the per-layer curve, so post hoc): inside each
outer fold, score every layer (or head) by the SNR of its inner leave-one-pair-out margins on the
15 training pairs only, keep the top k, and probe the held-out pair with their summed Gram.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

rows = [json.loads(line) for line in open(sys.argv[1])]
header, rows = rows[0], rows[1:]
gram_dir = sys.argv[2]
rng = np.random.default_rng(42)


def dprime(a, b):
    sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / sd) if sd > 0 else float("nan")


def probe(K, flip=None):
    """K: [2P, 2P] Gram, states ordered (X0, Y0, X1, Y1, ...). flip: optional bool[P] swapping X/Y."""
    P = K.shape[0] // 2
    xi, yi = np.arange(0, 2 * P, 2), np.arange(1, 2 * P, 2)
    if flip is not None:
        xi, yi = np.where(flip, yi, xi), np.where(flip, xi, yi)
    # projections of every state on every pair's difference: Pr[i, b] = <s_i, s_Xb - s_Yb>
    Pr = K[:, xi] - K[:, yi]
    paired, single, cen = [], [], np.zeros(2 * P)
    for a in range(P):
        T = np.delete(np.arange(P), a)
        proj = Pr[:, T].sum(1)                                   # <s_i, w>
        thr = proj[np.concatenate([xi[T], yi[T]])].mean()        # <mean training state, w>
        paired.append(proj[xi[a]] - proj[yi[a]] > 0)
        single += [proj[xi[a]] > thr, proj[yi[a]] <= thr]
        cen[xi[a]], cen[yi[a]] = proj[xi[a]] - thr, proj[yi[a]] - thr
    return float(np.mean(paired)), float(np.mean(single)), dprime(cen[xi], cen[yi])


def nested(Ku, k):
    """Ku: [U, 2P, 2P] Grams of U units (layers or heads). Per outer fold, rank units by inner-LOO
    margin SNR on the training pairs, sum the top-k Grams, and score the held-out pair."""
    P = Ku.shape[1] // 2
    xi, yi = np.arange(0, 2 * P, 2), np.arange(1, 2 * P, 2)
    D = Ku[:, xi][:, :, xi] - Ku[:, xi][:, :, yi] - Ku[:, yi][:, :, xi] + Ku[:, yi][:, :, yi]  # <dX_b, dX_c>
    paired, single, cen = [], [], np.zeros(2 * P)
    for a in range(P):
        T = np.delete(np.arange(P), a)
        DT = D[:, T][:, :, T]
        m = DT.sum(2) - np.einsum("ubb->ub", DT)                 # inner LOO margins [U, P-1]
        snr = m.mean(1) / (m.std(1, ddof=1) + 1e-30)
        K = Ku[np.argsort(-snr)[:k]].sum(0)
        Pr = K[:, xi] - K[:, yi]
        proj = Pr[:, T].sum(1)
        thr = proj[np.concatenate([xi[T], yi[T]])].mean()
        paired.append(proj[xi[a]] - proj[yi[a]] > 0)
        single += [proj[xi[a]] > thr, proj[yi[a]] <= thr]
        cen[xi[a]], cen[yi[a]] = proj[xi[a]] - thr, proj[yi[a]] - thr
    return float(np.mean(paired)), float(np.mean(single)), dprime(cen[xi], cen[yi])


SEL = (("layer", 1), ("layer", 4), ("layer", 12), ("head", 16), ("head", 64), ("head", 256))


def model_readout(d):
    P = len(d) // 2
    dx, dy = d[0::2], d[1::2]
    single = []
    for a in range(P):
        T = np.delete(np.arange(P), a)
        thr = (dx[T].mean() + dy[T].mean()) / 2
        s = np.sign(dx[T].mean() - dy[T].mean())
        single += [s * (dx[a] - thr) > 0, s * (dy[a] - thr) <= 0]
    return float(np.mean(dx - dy > 0)), float(np.mean(single)), dprime(dx, dy)


by = defaultdict(list)
for r in rows:
    by[(r["key"], r["block"])].append(r)
L = len(header["gdn_layers"])
late = list(range(2 * L // 3, L))
res = defaultdict(lambda: defaultdict(list))
per_layer = defaultdict(list)
for (key, task), rs in sorted(by.items()):
    path = os.path.join(gram_dir, f"{key}_{task}.pt")
    if not os.path.exists(path):
        continue
    rs = sorted(rs, key=lambda r: r["i"])
    assert [r["label"] for r in rs] == [0, 1] * (len(rs) // 2)
    G = torch.load(path)["grams"].numpy()
    diag = np.einsum("lhii->lhi", G).mean(-1)
    Kh = G / np.maximum(diag, 1e-30)[..., None, None]           # [L, H, 2P, 2P]
    Kl = Kh.sum(1)                                              # [L, 2P, 2P]
    per_layer[task].append([probe(Kl[li])[2] for li in range(L)])
    for name, K in (("late", Kl[late].sum(0)), ("all", Kl.sum(0))):
        for m, v in zip(("paired", "single", "dprime"), probe(K)):
            res[task][f"probe_{name}_{m}"].append(v)
    for unit, k in SEL:
        Ku = Kl if unit == "layer" else Kh.reshape(-1, *Kh.shape[2:])
        for m, v in zip(("paired", "single", "dprime"), nested(Ku, k)):
            res[task][f"sel_{unit}{k}_{m}"].append(v)
    nulls = [probe(Kl.sum(0), flip=rng.random(len(rs) // 2) < 0.5) for _ in range(200)]
    res[task]["null_paired"].append([n[0] for n in nulls])
    res[task]["null_single"].append([n[1] for n in nulls])
    for g in ("g1", "g2"):
        for m, v in zip(("paired", "single", "dprime"), model_readout(np.array([r[f"d_{g}"] for r in rs]))):
            res[task][f"model_{g}_{m}"].append(v)

for task in sorted(res):
    R = res[task]
    print(f"\n== {task}: {len(R['model_g1_paired'])} attributes x 16 pairs (leave-one-pair-out; mean over attributes) ==")
    print(f"  {'readout':14s} {'paired acc':>11s} {'single acc':>11s} {'d-prime':>8s}")
    for name in ("probe_late", "probe_all", *(f"sel_{u}{k}" for u, k in SEL), "model_g1", "model_g2"):
        print(f"  {name:14s} {np.mean(R[name + '_paired']):11.3f} {np.mean(R[name + '_single']):11.3f}"
              f" {np.nanmean(R[name + '_dprime']):8.2f}")
    for m in ("paired", "single"):
        null = np.array(R[f"null_{m}"]).mean(0)
        obs = np.mean(R[f"probe_all_{m}"])
        print(f"  null ({m}, all layers): mean {null.mean():.3f}, 99th pct {np.percentile(null, 99):.3f}, "
              f"p = {(np.sum(null >= obs) + 1) / (len(null) + 1):.3f}")
    pl = np.nanmean(np.array(per_layer[task]), 0)
    print("  per-layer probe d' (GDN layer: d'): " + " ".join(f"{i}:{a:.1f}" for i, a in enumerate(pl)))
