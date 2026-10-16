"""Summarise the counterfactual position sweep: shift = d(state_X) - d(state_Y), d = lpX - lpY.

python analysis/pilot_position_sweep.py results/pilot/pos_sweep_q35-4b.jsonl
Bootstrap CIs resample attributes (clusters), 10k draws, seed 42.
"""
import json
import sys
from collections import defaultdict

import numpy as np

rows = [json.loads(line) for line in open(sys.argv[1])]
rng = np.random.default_rng(42)


def d(pair):
    return pair[0] - pair[1]


def cluster_ci(vals_by_attr, B=10000):
    attrs = list(vals_by_attr)
    means = []
    for _ in range(B):
        pick = rng.choice(len(attrs), len(attrs), replace=True)
        means.append(np.mean(np.concatenate([vals_by_attr[attrs[i]] for i in pick])))
    return np.percentile(means, [2.5, 97.5])


def summarise(sel, cond):
    by = defaultdict(list)
    acc = []
    for r in sel:
        if f"{cond}_X" not in r:
            continue
        s = d(r[f"{cond}_X"]) - d(r[f"{cond}_Y"])
        by[r["attr"]].append(s)
        acc.append((d(r[f"{cond}_X"]) > 0) + (d(r[f"{cond}_Y"]) < 0))   # 2AFC correct out of 2
    if not by:
        return None
    vals = np.concatenate([np.array(v) for v in by.values()])
    lo, hi = cluster_ci({k: np.array(v) for k, v in by.items()})
    return dict(n=len(vals), mean=float(vals.mean()), lo=float(lo), hi=float(hi),
                frac_pos=float((vals > 0).mean()), acc2afc=float(np.sum(acc) / (2 * len(acc))))


print(f"rows {len(rows)}")
print("\n== no dilution: shift by memory size x position (C = state only, M = text in context) ==")
print(f"{'entries':>7} {'frac':>5} {'tok_after':>9} | {'C shift [95% CI]':>28} {'C>0':>5} {'C2afc':>6} |"
      f" {'M shift':>8} {'M2afc':>6}")
for n in sorted({r["n_entries"] for r in rows}):
    for f in sorted({r["frac_from_end"] for r in rows}):
        sel = [r for r in rows if r["n_entries"] == n and r["frac_from_end"] == f and r["dilution"] == 0]
        if not sel:
            continue
        c, m = summarise(sel, "C"), summarise(sel, "M")
        ta = np.median([r["tokens_after_target"] for r in sel])
        print(f"{n:>7} {f:>5} {ta:>9.0f} | {c['mean']:>8.3f} [{c['lo']:>7.3f},{c['hi']:>7.3f}]   "
              f"{c['frac_pos']:>5.2f} {c['acc2afc']:>6.2f} | {m['mean']:>8.2f} {m['acc2afc']:>6.2f}")

print("\n== dilution (60 entries): C shift vs tokens of unrelated exchange before the question ==")
for f in [0.0, 0.5]:
    for S in sorted({r["dilution"] for r in rows}):
        sel = [r for r in rows if r["n_entries"] == 60 and r["frac_from_end"] == f and r["dilution"] == S]
        if not sel:
            continue
        c = summarise(sel, "C")
        print(f"frac {f} dilution {S:>5}: C shift {c['mean']:.3f} [{c['lo']:.3f},{c['hi']:.3f}]  2afc {c['acc2afc']:.2f}")

# A-condition sanity: prior preference between X and Y (should be ~symmetric noise)
a = np.array([d(r["A"]) for r in rows if r["dilution"] == 0])
print(f"\nA: mean |lpX-lpY| = {np.abs(a).mean():.2f} (prior asymmetry between the paired values)")
