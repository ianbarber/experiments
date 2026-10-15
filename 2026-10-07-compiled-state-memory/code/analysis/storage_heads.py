"""Storage probe follow-ups (exploratory, post hoc; dev items only).

uv run --with numpy --with scipy --with torch python analysis/storage_heads.py \
    results/phase1/storage_probe_<tag>.jsonl data/probe

1. Label-swap null for the nested head-selection probe (top-k heads chosen inside each fold).
2. Do the same heads carry binding across attributes? Top-64 overlap vs chance, layer histogram,
   and transfer: a fixed head set chosen on the other 11 attributes (or on another task), then the
   plain leave-one-pair-out probe on this attribute's Grams.
3. Does decodability fall with distance from the end of the memory? Held-out pair margins
   (nested top-64 heads, bind250) against entry positions.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

rows = [json.loads(line) for line in open(sys.argv[1])]
header, rows = rows[0], rows[1:]
gram_dir = sys.argv[2]
rng = np.random.default_rng(42)
gdn = header["gdn_layers"]


def load(key, task):
    G = torch.load(os.path.join(gram_dir, f"{key}_{task}.pt"))["grams"].numpy()
    diag = np.einsum("lhii->lhi", G).mean(-1)
    Kh = G / np.maximum(diag, 1e-30)[..., None, None]
    return Kh.reshape(-1, *Kh.shape[2:])  # [L*H, 2P, 2P]


def idx(P, flip=None):
    xi, yi = np.arange(0, 2 * P, 2), np.arange(1, 2 * P, 2)
    if flip is not None:
        xi, yi = np.where(flip, yi, xi), np.where(flip, xi, yi)
    return xi, yi


def snr(Ku, xi, yi, T):
    D = Ku[:, xi][:, :, xi] - Ku[:, xi][:, :, yi] - Ku[:, yi][:, :, xi] + Ku[:, yi][:, :, yi]
    DT = D[:, T][:, :, T]
    m = DT.sum(2) - np.einsum("ubb->ub", DT)
    return m.mean(1) / (m.std(1, ddof=1) + 1e-30)


def loo(K, xi, yi, a, T):
    """Plain difference-of-means probe on Gram K; returns (paired margin, single correct x2, z-margin)."""
    Pr = K[:, xi] - K[:, yi]
    proj = Pr[:, T].sum(1)
    tr = np.concatenate([xi[T], yi[T]])
    thr, sd = proj[tr].mean(), proj[tr].std(ddof=1)
    return proj[xi[a]] - proj[yi[a]], [proj[xi[a]] > thr, proj[yi[a]] <= thr], (proj[xi[a]] - proj[yi[a]]) / sd


def nested(Ku, k, flip=None):
    P = Ku.shape[1] // 2
    xi, yi = idx(P, flip)
    paired, single, z = [], [], []
    for a in range(P):
        T = np.delete(np.arange(P), a)
        K = Ku[np.argsort(-snr(Ku, xi, yi, T))[:k]].sum(0)
        m, s, zz = loo(K, xi, yi, a, T)
        paired.append(m > 0)
        single += s
        z.append(zz)
    return np.mean(paired), np.mean(single), np.array(z)


def fixed(Ku, heads):
    P = Ku.shape[1] // 2
    xi, yi = idx(P)
    K = Ku[heads].sum(0)
    paired, single = [], []
    for a in range(P):
        m, s, _ = loo(K, xi, yi, a, np.delete(np.arange(P), a))
        paired.append(m > 0)
        single += s
    return np.mean(paired), np.mean(single)


keys = sorted({r["key"] for r in rows})
tasks = ("bind60", "bind250")
Ku = {(k, t): load(k, t) for k in keys for t in tasks}
P = 16
xi, yi = idx(P)
full_snr = {kt: snr(K, xi, yi, np.arange(P)) for kt, K in Ku.items()}

print("== 1. label-swap null, nested top-k heads (attribute-averaged accuracy; 50 swaps) ==")
for t in tasks:
    for k in (16, 64):
        obs = [nested(Ku[(key, t)], k)[:2] for key in keys]
        nulls = np.array([[nested(Ku[(key, t)], k, flip=rng.random(P) < 0.5)[:2] for key in keys]
                          for _ in range(50)]).mean(1)  # [50, 2]
        o = np.mean(obs, 0)
        print(f"  {t:8s} top{k:<3d} observed paired {o[0]:.3f} single {o[1]:.3f} | null paired mean "
              f"{nulls[:, 0].mean():.3f} max {nulls[:, 0].max():.3f}; single mean {nulls[:, 1].mean():.3f} "
              f"max {nulls[:, 1].max():.3f}")

print("\n== 2. are the binding heads shared across attributes? ==")
NH = Ku[(keys[0], tasks[0])].shape[0]
H = NH // len(gdn)
for t in tasks:
    tops = {key: set(np.argsort(-full_snr[(key, t)])[:64]) for key in keys}
    ov = [len(tops[a] & tops[b]) for i, a in enumerate(keys) for b in keys[i + 1:]]
    print(f"  {t}: top-64 overlap between attributes mean {np.mean(ov):.1f} (chance {64 * 64 / NH:.1f}), "
          f"range {min(ov)}-{max(ov)}")
    cnt = defaultdict(int)
    for s in tops.values():
        for h in s:
            cnt[h // H] += 1
    print("    top-64 heads by GDN layer (summed over attributes; model layer index): " +
          " ".join(f"{gdn[li]}:{c}" for li, c in sorted(cnt.items())))
for src, dst in (("bind60", "bind60"), ("bind60", "bind250"), ("bind250", "bind250")):
    for k in (16, 64):
        res = []
        for key in keys:
            # rank heads by mean z-scored SNR over the other attributes on the source task
            z = np.mean([stats.zscore(full_snr[(o, src)]) for o in keys if o != key], 0)
            res.append(fixed(Ku[(key, dst)], np.argsort(-z)[:k]))
        same = [fixed(Ku[(key, dst)], np.argsort(-full_snr[(key, src)])[:k]) for key in keys] if src != dst else None
        r = np.mean(res, 0)
        extra = f" | same-attribute heads from {src}: paired {np.mean(same, 0)[0]:.3f} single {np.mean(same, 0)[1]:.3f}" \
            if same else ""
        print(f"  heads from other attributes ({src}) -> {dst}, top{k}: paired {r[0]:.3f} single {r[1]:.3f}{extra}")
glob = np.mean([stats.zscore(full_snr[(key, t)]) for key in keys for t in tasks], 0)
top = np.argsort(-glob)[:16]
print("  global top-16 heads (model layer, head): " + " ".join(f"({gdn[h // H]},{h % H})" for h in top))

print("\n== 3. decodability vs position (bind250, nested top-64, held-out pair z-margin) ==")
pos = defaultdict(dict)
for r in rows:
    if r["block"] == "bind250" and r["label"] == 0:
        pos[r["key"]][r["i"] // 2] = (r["iu"], r["ic"], r["n"])
Z, DU, DC = [], [], []
for key in keys:
    _, _, z = nested(Ku[(key, "bind250")], 64)
    Z += list(z)
    DU += [pos[key][p][2] - 1 - pos[key][p][0] for p in range(P)]
    DC += [pos[key][p][2] - 1 - pos[key][p][1] for p in range(P)]
Z, DU, DC = map(np.array, (Z, DU, DC))
far = np.maximum(DU, DC)
print(f"  spearman(z, entries from end of user fact) {stats.spearmanr(Z, DU)[0]:.2f}; "
      f"(z, of colleague fact) {stats.spearmanr(Z, DC)[0]:.2f}; (z, farther of the two) {stats.spearmanr(Z, far)[0]:.2f}")
for lo, hi in ((0, 83), (83, 166), (166, 250)):
    sel = (far >= lo) & (far < hi)
    print(f"  farther fact {lo:3d}-{hi:3d} entries from end: n={sel.sum():3d}  median z {np.median(Z[sel]):.2f}  "
          f"correct {np.mean(Z[sel] > 0):.3f}")
