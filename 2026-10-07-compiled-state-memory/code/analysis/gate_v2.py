"""Pre-registered analysis of the Phase-1 sealed gate (PREREG.md §G1).

uv run --with numpy --with scipy python analysis/gate_v2.py results/phase1/gate_v2_q38-27b.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ALPHA, DELTA, SEED = 0.0125, 0.25, 42
rows = [json.loads(line) for line in open(sys.argv[1])]
header, rows = rows[0], rows[1:]
rng = np.random.default_rng(SEED)
sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def d(r, cond):
    a, b = r[cond]
    j = r["j"]
    return a[j] - b[j]


def dsum(r, cond):
    a, b = r[cond]
    return sum(a) - sum(b)


def shift(r, c, f=d):
    return f(r, f"{c}_X") - f(r, f"{c}_Y")


def item_scores(block, cond="C", f=d, where=lambda r: True):
    by = defaultdict(list)
    for r in rows:
        if r["block"] == block and where(r) and f"{cond}_X" in r:
            by[r["key"]].append(shift(r, cond, f))
    return {k: float(np.mean(v)) for k, v in by.items()}, {k: len(v) for k, v in by.items()}


def signflip_p(x, n=100_000):
    x = np.asarray(x)
    obs = x.mean()
    flips = rng.choice([-1.0, 1.0], size=(n, len(x)))
    return float(((flips * x).mean(1) >= obs).mean())


def test_block(name, scores):
    x = np.array(list(scores.values()))
    G = len(x)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(G)
    t_p = float(stats.t.sf(m / se, G - 1))
    lb = m - stats.t.ppf(1 - ALPHA, G - 1) * se                 # one-sided 98.75% lower bound
    lo2, hi2 = m - stats.t.ppf(1 - 0.0125, G - 1) * se, m + stats.t.ppf(1 - 0.0125, G - 1) * se  # 97.5% two-sided
    sf = signflip_p(x)
    passed = lb > 0 and t_p < ALPHA and sf < ALPHA and m >= DELTA
    killed_here = hi2 < DELTA
    print(f"{name}: G={G} mean {m:.3f} nats  SE {se:.3f}  one-sided LB {lb:.3f}  97.5% CI [{lo2:.3f},{hi2:.3f}]"
          f"  t p={t_p:.2e}  sign-flip p={sf:.2e}  frac>0 {np.mean(x > 0):.2f}  -> PASS={passed}  UB<delta={killed_here}")
    return passed, killed_here, m


print(f"header: {header}\nrows: {len(rows)}\n")
print("=== K1 (primary) ===")
b1 = item_scores("facts_bind")[0]
b2 = item_scores("prefs_fact")[0]
p1, k1, m1 = test_block("B1 facts_bind (binding)", b1)
p2, k2, m2 = test_block("B2 prefs_fact (polarity)", b2)
verdict = "PASS" if (p1 or p2) else ("KILL" if (k1 and k2) else "INCONCLUSIVE (treated as KILL for memory framing)")
print(f"K1 verdict: {verdict}   PASS(B1)={p1} PASS(B2)={p2}\n")

print("=== Secondary (descriptive) ===")
for block in ("facts_raw", "facts_bind", "prefs_fact", "prefs_instr"):
    for cond in ("C", "M", "K"):
        sc, _ = item_scores(block, cond)
        if not sc:
            continue
        x = np.array(list(sc.values()))
        sc_sum, _ = item_scores(block, cond, f=dsum)
        # bounded magnitude and 2AFC over (row, state)
        dps, afc = [], []
        for r in rows:
            if r["block"] == block and f"{cond}_X" in r:
                dx, dy = d(r, f"{cond}_X"), d(r, f"{cond}_Y")
                dps.append(sig(dx) - sig(dy))
                afc += [dx > 0, dy < 0]
        print(f"{block:12s} {cond}: mean shift {x.mean():7.3f} [{np.percentile(x, 2.5):.2f}..{np.percentile(x, 97.5):.2f} item range]"
              f"  summed-token {np.mean(list(sc_sum.values())):7.3f}  dp {np.mean(dps):.3f}  2AFC {np.mean(afc):.3f}")
# raw - bind (priming component), instruction - fact
raw, _ = item_scores("facts_raw", where=lambda r: r["size"] == 60)
common = sorted(set(raw) & set(b1))
if common:
    diff = np.array([raw[k] - b1[k] for k in common])
    print(f"\nfacts raw(n=60) - bind: {diff.mean():.3f} (paired t p={stats.ttest_1samp(diff, 0).pvalue:.2e}, G={len(diff)})")
inst, _ = item_scores("prefs_instr")
fact60, _ = item_scores("prefs_fact", where=lambda r: r["size"] == 60)
common = sorted(set(inst) & set(fact60))
if common:
    diff = np.array([inst[k] - fact60[k] for k in common])
    print(f"prefs instruction - fact (n=60): {diff.mean():.3f} (paired t p={stats.ttest_1samp(diff, 0).pvalue:.2e})")
for block in ("facts_raw", "prefs_fact"):
    f_, _ = item_scores(block, where=lambda r: r["pos"] == "first")
    l_, _ = item_scores(block, where=lambda r: r["pos"] == "last")
    s60, _ = item_scores(block, where=lambda r: r["size"] == 60)
    s250, _ = item_scores(block, where=lambda r: r["size"] == 250)
    if f_ and l_:
        dfl = np.array([f_[k] - l_[k] for k in f_])
        dsz = np.array([s60[k] - s250[k] for k in s60])
        print(f"{block}: first-last {dfl.mean():.3f} (p={stats.ttest_1samp(dfl, 0).pvalue:.2e}); "
              f"size60-size250 {dsz.mean():.3f} (p={stats.ttest_1samp(dsz, 0).pvalue:.2e})")
# Binding layouts separately (position-weighted priming = half the within-pair difference)
lay = defaultdict(list)
for r in rows:
    if r["block"] == "facts_bind":
        lay[r["layout"]].append(shift(r, "C"))
if lay:
    print("facts_bind by layout: " + ", ".join(f"{k} {np.mean(v):.3f}" for k, v in sorted(lay.items())))

print("\n=== K2 (harm) ===")
g = [r for r in rows if r["block"] == "guard"]
if g:
    names = sorted(g[0]["C"])
    mA = np.array([r["A"][0][r["j"]] - r["A"][1][r["j"]] for r in g])
    mC = np.array([[r["C"][n][0][r["j"]] - r["C"][n][1][r["j"]] for n in names] for r in g])
    harm = (mA[:, None] - mC)
    H = harm.mean() / mA.mean()
    boots = []
    for _ in range(10_000):
        qi = rng.integers(0, len(g), len(g))
        mi = rng.integers(0, len(names), len(names))
        boots.append(harm[np.ix_(qi, mi)].mean() / mA[qi].mean())
    lb, ub = np.percentile(boots, [5, 95])
    flips = np.mean((mA[:, None] > 0) & (mC < 0))
    k2 = "KILL" if lb > 0.10 else ("PASS" if ub < 0.10 else "INCONCLUSIVE")
    print(f"harm ratio H = {H:.3f} (90% CI {lb:.3f}..{ub:.3f}); mean margin A {mA.mean():.2f}; "
          f"gold->distractor flips {flips:.3f}  -> K2 {k2}")

print("\n=== Lures (H6, descriptive) ===")
L = [r for r in rows if r["block"] == "lures"]
if L:
    yn = lambda r, c: r[c][0][0] - r[c][1][0]  # noqa: E731
    for c in ("A", "C", "M"):
        by = {(r["key"], r["pos"], r["kind"]): yn(r, c) for r in L}
        keys = sorted({(k, p) for k, p, _ in by})
        a_n = np.array([by[(k, p, "assoc")] - by[(k, p, "neutral")] for k, p in keys])
        p_a = np.array([by[(k, p, "present")] - by[(k, p, "assoc")] for k, p in keys])
        print(f"{c}: assoc-neutral {a_n.mean():.3f} (false-memory pull)   present-assoc {p_a.mean():.3f} (discrimination)")
