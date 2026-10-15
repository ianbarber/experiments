"""EXPLORATORY follow-ups on the Phase-1 gate (written after the registered analysis was read;
nothing here bears on K1/K2).

uv run --with numpy --with scipy python analysis/gate_v2_explore.py results/phase1/gate_v2_q38-27b.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

rows = [json.loads(line) for line in open(sys.argv[1])][1:]
rng = np.random.default_rng(42)
sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def d(r, c):
    a, b = r[c]
    return a[r["j"]] - b[r["j"]]


def shift(r, c):
    return d(r, f"{c}_X") - d(r, f"{c}_Y")


def tci(x, level=0.95):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * se
    return f"{x.mean():7.3f} [{x.mean() - h:6.3f},{x.mean() + h:6.3f}] (n={len(x)})"


def by_item(block, cond="C", where=lambda r: True, f=shift):
    out = defaultdict(list)
    for r in rows:
        if r["block"] == block and where(r) and f"{cond}_X" in r:
            out[r["key"]].append(f(r, cond))
    return {k: float(np.mean(v)) for k, v in out.items()}


print("== Holm over the registered secondary p-values (PREREG §G1 item 4 and raw-bind / instr-fact) ==")
ps = {"facts raw-bind": 7.64e-10, "prefs instr-fact": 5.41e-04, "facts_raw first-last": 6.03e-05,
      "facts_raw size60-250": 8.87e-02, "prefs_fact first-last": 2.54e-01, "prefs_fact size60-250": 4.76e-03}
order = sorted(ps, key=ps.get)
running = 0.0
for i, k in enumerate(order):
    running = max(running, min(1.0, ps[k] * (len(ps) - i)))
    print(f"  {k:24s} p={ps[k]:.2e}  Holm p={running:.2e}")

print("\n== Binding vs position-weighted priming from the mirrored layouts (C) ==")
L = {lay: by_item("facts_bind", where=lambda r, lay=lay: r["layout"] == lay)
     for lay in ("u0_c30", "c0_u30", "u59_c30", "c59_u30")}
keys = sorted(L["u0_c30"])
for a, b, name in (("u0_c30", "c0_u30", "0 vs 30"), ("u59_c30", "c59_u30", "59 vs 30")):
    B = [(L[a][k] + L[b][k]) / 2 for k in keys]
    P = [(L[a][k] - L[b][k]) / 2 for k in keys]
    print(f"  pair {name:9s} binding {tci(B)}   priming(pos weight) {tci(P)}")
Mb = by_item("facts_bind", "M")
print(f"  M binding (all layouts) {tci(list(Mb.values()))}")

print("\n== Bounded magnitude: dp_C / dp_M per block, item-cluster bootstrap 95% ==")
for block in ("facts_raw", "facts_bind", "prefs_fact", "prefs_instr"):
    per = defaultdict(lambda: [[], []])
    for r in rows:
        if r["block"] == block:
            for i, c in enumerate(("C", "M")):
                per[r["key"]][i].append(sig(d(r, f"{c}_X")) - sig(d(r, f"{c}_Y")))
    ks = sorted(per)
    C = np.array([np.mean(per[k][0]) for k in ks])
    M = np.array([np.mean(per[k][1]) for k in ks])
    bs = [C[i].mean() / M[i].mean() for i in (rng.integers(0, len(ks), len(ks)) for _ in range(10_000))]
    print(f"  {block:12s} dp_C {C.mean():.3f}  dp_M {M.mean():.3f}  ratio {C.mean() / M.mean():.3f} "
          f"[{np.percentile(bs, 2.5):.3f},{np.percentile(bs, 97.5):.3f}]")

print("\n== Preference cells (C shift, t 95%) ==")
for block, sizes in (("prefs_fact", (60, 250)), ("prefs_instr", (60,))):
    for s in sizes:
        for p in ("first", "last"):
            x = by_item(block, where=lambda r, s=s, p=p: r["size"] == s and r["pos"] == p)
            print(f"  {block:12s} n={s:3d} {p:5s} {tci(list(x.values()))}")

print("\n== Is B2 small because the probe is weak? (exploratory) ==")
Cp, Mp = by_item("prefs_fact"), by_item("prefs_fact", "M")
ks = sorted(Cp)
c, m = np.array([Cp[k] for k in ks]), np.array([Mp[k] for k in ks])
print(f"  item corr(C, M) r={stats.pearsonr(c, m)[0]:.2f}; spearman {stats.spearmanr(c, m)[0]:.2f}")
for thr in (1.0, 3.0):
    sel = m > thr
    print(f"  items with M shift > {thr}: {sel.sum()}/48, C on those {tci(c[sel])}; C on the rest {tci(c[~sel])}")
Cf = by_item("facts_bind")
print(f"  facts_bind item corr(C, M) r={stats.pearsonr([Cf[k] for k in sorted(Cf)], [Mb[k] for k in sorted(Cf)])[0]:.2f}")

print("\n== K (memory KV only, GDN zeroed) vs M vs C ==")
for block in ("facts_raw", "prefs_fact"):
    print("  " + "  ".join(f"{block} {c}: {np.mean(list(by_item(block, c).values())):.2f}" for c in ("C", "K", "M")))

print("\n== Lures (descriptive t 95% over 16 list x position cells) ==")
Lr = [r for r in rows if r["block"] == "lures"]
yn = lambda r, c: r[c][0][0] - r[c][1][0]  # noqa: E731
for c in ("C", "M"):
    by = {(r["key"], r["pos"], r["kind"]): yn(r, c) for r in Lr}
    cells = sorted({(k, p) for k, p, _ in by})
    a_n = [by[(k, p, "assoc")] - by[(k, p, "neutral")] for k, p in cells]
    p_a = [by[(k, p, "present")] - by[(k, p, "assoc")] for k, p in cells]
    print(f"  {c}: assoc-neutral {tci(a_n)}   present-assoc {tci(p_a)}")
