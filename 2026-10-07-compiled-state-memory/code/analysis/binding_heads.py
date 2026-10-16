"""Rank GDN heads by how well they separate user-bound values in the storage-probe run, and write
head sets for head-targeted gain (exploratory).

uv run --with numpy --with scipy --with torch python analysis/binding_heads.py \
    results/phase1/storage_probe_<tag>.jsonl data/probe results/phase1/binding_heads_<tag>.json

Per (attribute, task in bind60/bind250): per-head SNR of leave-one-pair-out difference-of-means
margins over the 16 pairs, z-scored over heads. Rankings written:
  by_attr[key]   mean z over the *other* 11 attributes (leave-attribute-out), top 64
  global         mean z over all 12 attributes, top 64
  random[key] / random_global   layer-matched random controls (same per-layer counts as the
                 top-64 set, heads drawn uniformly within each layer, excluding the top 64; seed 42),
                 ordered so the first 16 match the top-16 layer counts
Heads are (model layer index, head index).
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from scipy import stats

rows = [json.loads(line) for line in open(sys.argv[1])]
header, rows = rows[0], rows[1:]
gram_dir, out = sys.argv[2], sys.argv[3]
gdn = header["gdn_layers"]
rng = np.random.default_rng(42)
P = 16
xi, yi = np.arange(0, 2 * P, 2), np.arange(1, 2 * P, 2)


def head_z(path):
    G = torch.load(path)["grams"].numpy()
    L, H = G.shape[:2]
    diag = np.einsum("lhii->lhi", G).mean(-1)
    K = (G / np.maximum(diag, 1e-30)[..., None, None]).reshape(L * H, 2 * P, 2 * P)
    D = K[:, xi][:, :, xi] - K[:, xi][:, :, yi] - K[:, yi][:, :, xi] + K[:, yi][:, :, yi]
    m = D.sum(2) - np.einsum("ubb->ub", D)
    return stats.zscore(m.mean(1) / (m.std(1, ddof=1) + 1e-30)), H


keys = sorted({r["key"] for r in rows})
Z = {}
for k in keys:
    for t in ("bind60", "bind250"):
        Z[(k, t)], H = head_z(os.path.join(gram_dir, f"{k}_{t}.pt"))


def as_heads(flat):
    return [[gdn[int(h) // H], int(h) % H] for h in flat]


def matched_random(top):
    """Layer-matched random heads: first 16 match top[:16]'s layer counts, the rest top[16:64]'s."""
    used, res = set(int(h) for h in top), []
    for part in (top[:16], top[16:64]):
        for layer, n in sorted(Counter(int(h) // H for h in part).items()):
            pool = [layer * H + h for h in range(H) if layer * H + h not in used]
            pick = rng.choice(pool, n, replace=False)
            used.update(int(p) for p in pick)
            res += [int(p) for p in pick]
    return res


by_attr, rand = {}, {}
for k in keys:
    z = np.mean([Z[(o, t)] for o in keys if o != k for t in ("bind60", "bind250")], 0)
    top = np.argsort(-z)[:64]
    by_attr[k], rand[k] = as_heads(top), as_heads(matched_random(top))
zg = np.mean([Z[(k, t)] for k in keys for t in ("bind60", "bind250")], 0)
gtop = np.argsort(-zg)[:64]
json.dump(dict(source=sys.argv[1], heads_per_layer=H, by_attr=by_attr, random=rand, global_=as_heads(gtop),
               random_global=as_heads(matched_random(gtop))), open(out, "w"))
print(f"wrote {out}; global top-16: {as_heads(gtop[:16])}")
print("global top-64 by layer:", sorted(Counter(gdn[int(h) // H] for h in gtop).items()))
