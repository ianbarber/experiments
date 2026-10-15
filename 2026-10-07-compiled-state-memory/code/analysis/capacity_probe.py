"""Capacity / interference stress analysis (exploratory; dev items only).

uv run --with numpy --with scipy --with torch python analysis/capacity_probe.py \
    results/phase1/capacity_probe_<tag>.jsonl data/capacity results/phase1/storage_probe_<tag>.jsonl data/probe
(same for interference_probe_<tag>.jsonl with data/interference: cells instead of strata)

Per stratum (entries written after the facts), leave-one-pair-out difference-of-means probes on
8 counterfactual pairs per attribute, from per-head Grams:
  fixed16/64  heads chosen on the storage-probe run (bind60 + bind250, other 11 attributes only;
              different memories), so selection is independent of these states
  nested16    top-16 heads chosen inside each fold on the 7 training pairs
  all         every head, equal weight
Model readout on the same states (scale 1 and 2): paired, single (leave-pair-out midpoint),
d', 2AFC at threshold 0 (does the answer itself change), and the mean shift in nats.
Null: 100 within-pair label swaps for fixed16. d4000 has every other attribute only, so each
stratum is also reported on that 6-attribute subset.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

cap_rows = [json.loads(line) for line in open(sys.argv[1])]
cap_header, cap_rows = cap_rows[0], cap_rows[1:]
cap_dir, sp_rows, sp_dir = sys.argv[2], [json.loads(line) for line in open(sys.argv[3])][1:], sys.argv[4]
rng = np.random.default_rng(42)


def dprime(a, b):
    sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / sd) if sd > 0 else float("nan")


def load(path):
    G = torch.load(path)["grams"].numpy()
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


def probe(Ku, heads=None, nested_k=None, flip=None):
    """heads: fixed index array, or nested_k: choose top-k inside each fold. Returns paired, single, d'."""
    P = Ku.shape[1] // 2
    xi, yi = idx(P, flip)
    Kfix = Ku.sum(0) if heads is None and nested_k is None else (Ku[heads].sum(0) if heads is not None else None)
    paired, single, cen = [], [], np.zeros(2 * P)
    for a in range(P):
        T = np.delete(np.arange(P), a)
        K = Kfix if Kfix is not None else Ku[np.argsort(-snr(Ku, xi, yi, T))[:nested_k]].sum(0)
        proj = (K[:, xi] - K[:, yi])[:, T].sum(1)
        thr = proj[np.concatenate([xi[T], yi[T]])].mean()
        paired.append(proj[xi[a]] - proj[yi[a]] > 0)
        single += [proj[xi[a]] > thr, proj[yi[a]] <= thr]
        cen[xi[a]], cen[yi[a]] = proj[xi[a]] - thr, proj[yi[a]] - thr
    return float(np.mean(paired)), float(np.mean(single)), dprime(cen[xi], cen[yi])


def model_readout(d):
    P = len(d) // 2
    dx, dy = d[0::2], d[1::2]
    single = []
    for a in range(P):
        T = np.delete(np.arange(P), a)
        thr = (dx[T].mean() + dy[T].mean()) / 2
        s = np.sign(dx[T].mean() - dy[T].mean())
        single += [s * (dx[a] - thr) > 0, s * (dy[a] - thr) <= 0]
    afc = np.mean(np.concatenate([dx > 0, dy < 0]))
    return float(np.mean(dx - dy > 0)), float(np.mean(single)), dprime(dx, dy), float(afc), float(np.mean(dx - dy))


# head ranking per held-out attribute from the storage-probe run
sp_keys = sorted({r["key"] for r in sp_rows})
P16 = 16
xs, ys = idx(P16)
sp_snr = {(k, t): stats.zscore(snr(load(os.path.join(sp_dir, f"{k}_{t}.pt")), xs, ys, np.arange(P16)))
          for k in sp_keys for t in ("bind60", "bind250")}
rank = {k: np.argsort(-np.mean([sp_snr[(o, t)] for o in sp_keys if o != k for t in ("bind60", "bind250")], 0))
        for k in sp_keys}

by = defaultdict(list)
for r in cap_rows:
    by[(r["block"], r["key"])].append(r)
strata = list(cap_header["strata"]) if "strata" in cap_header else [f"{k}{n}" for k, n in cap_header["cells"]]
res = defaultdict(lambda: defaultdict(dict))  # res[stratum][key][metric] = value
toks = defaultdict(list)
nulls = defaultdict(list)
for (stratum, key), rs in sorted(by.items()):
    rs = sorted(rs, key=lambda r: r["i"])
    assert [r["label"] for r in rs] == [0, 1] * (len(rs) // 2)
    Ku = load(os.path.join(cap_dir, f"{key}_{stratum}.pt"))
    toks[stratum] += [r["n_tokens"] for r in rs]
    R = res[stratum][key]
    for name, kw in (("fixed16", dict(heads=rank[key][:16])), ("fixed64", dict(heads=rank[key][:64])),
                     ("nested16", dict(nested_k=16)), ("all", {})):
        R[name] = probe(Ku, **kw)
    P = len(rs) // 2
    nulls[stratum].append([probe(Ku, heads=rank[key][:16], flip=rng.random(P) < 0.5)[:2] for _ in range(100)])
    for g in ("g1", "g2"):
        R[f"model_{g}"] = model_readout(np.array([r[f"d_{g}"] for r in rs]))

sub = sorted(res["d4000"]) if "d4000" in res else None
for label, keyset in (("all attributes run in each stratum", None), (f"6-attribute subset {sub}", sub))[:2 if sub else 1]:
    print(f"\n== {label} (mean over attributes) ==")
    print(f"  {'stratum':8s} {'tokens':>7s} {'readout':10s} {'paired':>7s} {'single':>7s} {'dprime':>7s} "
          f"{'2AFC':>6s} {'shift':>6s}")
    for s in strata:
        ks = sorted(res[s]) if keyset is None else [k for k in keyset if k in res[s]]
        for name in ("fixed16", "fixed64", "nested16", "all", "model_g1", "model_g2"):
            v = np.array([res[s][k][name] for k in ks])
            m = v.mean(0)
            extra = f" {m[3]:6.3f} {m[4]:6.2f}" if name.startswith("model") else ""
            print(f"  {s:8s} {np.mean(toks[s]):7.0f} {name:10s} {m[0]:7.3f} {m[1]:7.3f} {np.nanmean(v[:, 2]):7.2f}{extra}"
                  + (f"   (n={len(ks)})" if name == "fixed16" else ""))
        if keyset is None:
            nl = np.array(nulls[s]).mean(0)
            print(f"  {'':8s} {'':7s} null fixed16: paired mean {nl[:, 0].mean():.3f} (max {nl[:, 0].max():.3f}), "
                  f"single mean {nl[:, 1].mean():.3f} (max {nl[:, 1].max():.3f})")

print("\n== per attribute: fixed16 single acc / model g1 single acc / model g1 shift ==")
for k in sorted(res[strata[0]]):
    print(f"  {k:10s} " + "  ".join(f"{s}: {res[s][k]['fixed16'][1]:.2f}/{res[s][k]['model_g1'][1]:.2f}/"
                                    f"{res[s][k]['model_g1'][4]:5.2f}" if k in res[s] else f"{s}: -" for s in strata))
