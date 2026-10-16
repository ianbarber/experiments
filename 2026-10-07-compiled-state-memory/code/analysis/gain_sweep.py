"""Gain sweep analysis (exploratory, dev items).

uv run --with numpy --with scipy python analysis/gain_sweep.py results/phase1/gain_sweep_<tag>.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

rows = [json.loads(line) for line in open(sys.argv[1])]
header, rows = rows[0], rows[1:]
gains = header["gains"]
rng = np.random.default_rng(42)
sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def dj(lp, j):
    return lp[0][j] - lp[1][j]


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():6.2f} [{x.mean() - h:5.2f},{x.mean() + h:5.2f}]"


def block_summary(block, split):
    print(f"\n== {block} (C shift, nats; t 95% over attributes | 2AFC | dp) ==")
    for g in gains:
        cells = defaultdict(lambda: defaultdict(list))
        afc, dps = [], []
        for r in rows:
            if r["block"] != block:
                continue
            dx, dy = dj(r["C"][g]["X"], r["j"]), dj(r["C"][g]["Y"], r["j"])
            cells[split(r)][r["key"]].append(dx - dy)
            afc += [dx > 0, dy < 0]
            dps.append(sig(dx) - sig(dy))
        parts = [f"{c} {tci([np.mean(v) for v in cells[c].values()])}" for c in sorted(cells)]
        print(f"  {g:10s} " + "  ".join(parts) + f"  | 2AFC {np.mean(afc):.3f} | dp {np.mean(dps):.3f}")


block_summary("bind", lambda r: "bind(all4)")
# position-weighted priming from the mirrored layouts
print("  binding / priming decomposition per gain (pair 0vs30, pair 59vs30):")
for g in gains:
    lay = defaultdict(dict)
    for r in rows:
        if r["block"] == "bind":
            lay[r["layout"]][r["key"]] = dj(r["C"][g]["X"], r["j"]) - dj(r["C"][g]["Y"], r["j"])
    ks = sorted(lay["u0_c30"])
    out = []
    for a, b in (("u0_c30", "c0_u30"), ("u59_c30", "c59_u30")):
        B = [(lay[a][k] + lay[b][k]) / 2 for k in ks]
        P = [(lay[a][k] - lay[b][k]) / 2 for k in ks]
        out.append(f"B {tci(B)} P {tci(P)}")
    print(f"    {g:10s} " + " | ".join(out))
block_summary("raw", lambda r: r["pos"])
block_summary("prefs", lambda r: r["pos"])

print("\n== guard: harm ratio H = mean(margin_A - margin_C) / mean(margin_A); 90% two-way bootstrap ==")
gd = [r for r in rows if r["block"] == "guard"]
mA = np.array([dj(r["A"], r["j"]) for r in gd])
for g in gains:
    names = sorted(gd[0]["C"][g])
    mC = np.array([[dj(r["C"][g][n], r["j"]) for n in names] for r in gd])
    harm = mA[:, None] - mC
    boots = []
    for _ in range(5000):
        qi, mi = rng.integers(0, len(gd), len(gd)), rng.integers(0, len(names), len(names))
        boots.append(harm[np.ix_(qi, mi)].mean() / mA[qi].mean())
    flips = np.mean((mA[:, None] > 0) & (mC < 0))
    print(f"  {g:10s} H {harm.mean() / mA.mean():6.3f} [{np.percentile(boots, 5):6.3f},{np.percentile(boots, 95):6.3f}]"
          f"  gold->distractor flips {flips:.3f}")

print("\n== lures: Yes-No log-odds differences, t 95% over 8 lists (positions averaged) ==")
L = [r for r in rows if r["block"] == "lures"]
for g in ["A"] + gains:
    get = (lambda r: r["A"][0][0] - r["A"][1][0]) if g == "A" else (lambda r, g=g: r["C"][g][0][0] - r["C"][g][1][0])
    by = defaultdict(list)
    for r in L:
        by[(r["key"], r["kind"])].append(get(r))
    keys = sorted({k for k, _ in by})
    an = [np.mean(by[(k, "assoc")]) - np.mean(by[(k, "neutral")]) for k in keys]
    pa = [np.mean(by[(k, "present")]) - np.mean(by[(k, "assoc")]) for k in keys]
    print(f"  {g:10s} assoc-neutral {tci(an)}   present-assoc {tci(pa)}" if g != "A" else
          f"  {g:10s} assoc-neutral {np.mean(an):.2f}   present-assoc {np.mean(pa):.2f}")
