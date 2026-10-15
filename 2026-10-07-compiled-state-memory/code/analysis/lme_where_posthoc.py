"""Stage 0 post-hoc analyses (added after the data were in; LABNOTES 2026-10-10, Stage 0 result).

uv run --with numpy --with scipy --with torch python code/analysis/lme_where_posthoc.py results/phase2/lme_where_q38-27b.jsonl

None of this was in the pre-logged plan. It answers questions the planned summary left open:
P1. Is X's shift from A tracking M's session by session, or only on average? Through-0 slope (as in
    the plan) beside the OLS slope with intercept, Pearson r, the slope a constant shift would give,
    and call-vs-stop agreement with M (the plan's 'same action' needs an exact argument match).
P2. Per-question presence (present - absent) of X against M's, on L and on the path hazard H:
    through-0 slope (question bootstrap), r, sign agreement, and the sign agreement expected from
    the two sign rates alone (permutation null).
P3. Anchor present - absent of X against M's, and K*4k - K paired.
P4. Readout dependence of the DLA presence split (M and CK, present - absent): the per-component
    'written' term under the present, absent, midpoint and A readouts.
Bootstraps: 4,000 resamples of questions, numpy default_rng(42).
"""
import json
import os
import sys

import numpy as np
import torch

rng = np.random.default_rng(42)
raw = [json.loads(line) for line in open(sys.argv[1])]
header, raw = raw[0], raw[1:]
rows = {}
for r in raw:
    rows[(r["block"], r["qid"], r.get("cond"), r.get("store"))] = r
Q = [q for q in header["qids"] if ("done", q, None, None) in rows]
dec = {(q, c, s): r for (b, q, c, s), r in rows.items() if b == "decide" and q in Q}
anc = {(q, c, s): r for (b, q, c, s), r in rows.items() if b == "anchor" and q in Q}
L = lambda r: r["lp_call"] - r["lp_not_call"]  # noqa: E731


def hz(r):
    """Greedy-path call hazard H, as in analysis/lme_where.py and analysis/lme_hint.py."""
    S = sum(float(np.log1p(-np.exp(st[0]))) if st[0] < -0.69 else st[1] for st in r["path"])
    return float(np.log(sum(np.exp(st[0]) for st in r["path"]))) if S > -1e-12 else float(np.log(-np.expm1(S)) - S)


def boot(x, y):
    x, y = np.asarray(x), np.asarray(y)
    sl = lambda i: (x[i] @ y[i]) / (x[i] @ x[i])  # noqa: E731
    bs = [sl(rng.integers(0, len(x), len(x))) for _ in range(4000)]
    return sl(np.arange(len(x))), np.percentile(bs, [2.5, 97.5])


CONDS = ["C", "K", "CK", "Kstar4k", "Kstar64k", "Krt", "Kearly", "Kmid", "Klate"]
print("== P1. shift from A, X - A on M - A over present and absent sessions ==")
print(f"  {'cond':9s} {'slope0':>7s} {'const':>6s} {'OLS':>6s} {'r':>6s}   call-vs-stop same as M (A: see first row)")
sess = [(q, s) for q in Q for s in ("p", "a")]
stopM = [dec[q, "M", s]["stop"] for q, s in sess]
print(f"  {'A':9s} {'':>7s} {'':>6s} {'':>6s} {'':>6s}   {sum(dec[q, 'A', None]['stop'] == m for (q, _), m in zip(sess, stopM))}/32")
for X in CONDS:
    x = np.array([L(dec[q, "M", s]) - L(dec[q, "A", None]) for q, s in sess])
    y = np.array([L(dec[q, X, s]) - L(dec[q, "A", None]) for q, s in sess])
    s0 = (x @ y) / (x @ x)
    const = y.mean() * x.sum() / (x @ x)
    ols = np.polyfit(x, y, 1)[0]
    agree = sum(dec[q, X, s]["stop"] == m for (q, s), m in zip(sess, stopM))
    print(f"  {X:9s} {s0:+7.2f} {const:+6.2f} {ols:+6.2f} {np.corrcoef(x, y)[0, 1]:+6.2f}   {agree}/32")
print("  slope0 = through-0 slope (the plan's); const = slope0 a constant shift equal to X's mean would give")

print("\n== P2. presence: per-question X(p - a) against M(p - a) ==")
for name, f in (("L", L), ("H", hz)):
    m = np.array([f(dec[q, "M", "p"]) - f(dec[q, "M", "a"]) for q in Q])
    print(f"  {name}: M p - a negative in {(m < 0).sum()}/16")
    for X in CONDS:
        x = np.array([f(dec[q, X, "p"]) - f(dec[q, X, "a"]) for q in Q])
        sl, ci = boot(m, x)
        agree = (np.sign(x) == np.sign(m)).sum()
        null = np.mean([(np.sign(rng.permutation(x)) == np.sign(m)).sum() for _ in range(4000)])
        print(f"    {X:9s} slope {sl:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]  r {np.corrcoef(m, x)[0, 1]:+.2f}  "
              f"signs agree {agree}/16 (permutation mean {null:.1f})")
    acts = {X: sum((dec[q, X, "p"]["stop"], dec[q, X, "p"].get("fn")) != (dec[q, X, "a"]["stop"], dec[q, X, "a"].get("fn"))
                   for q in Q) for X in ["M"] + CONDS}
print("  questions whose first action (stop, function) differs between present and absent:",
      ", ".join(f"{X} {n}" for X, n in acts.items()))

print("\n== P3. anchor present - absent against M's (n = questions with a short gold) ==")
qa = [q for q in Q if (q, "M", "p") in anc]
m = np.array([anc[q, "M", "p"]["lp_sum"] - anc[q, "M", "a"]["lp_sum"] for q in qa])
for X in ["C", "K", "CK", "Kstar4k", "Kearly", "Kmid", "Klate"]:
    x = np.array([anc[q, X, "p"]["lp_sum"] - anc[q, X, "a"]["lp_sum"] for q in qa])
    sl, ci = boot(m, x)
    print(f"  {X:9s} n={len(qa)} mean {x.mean():+6.2f} (M {m.mean():+.2f}; ratio {x.mean() / m.mean():.2f})  slope {sl:+.2f} "
          f"[{ci[0]:+.2f}, {ci[1]:+.2f}]  r {np.corrcoef(m, x)[0, 1]:+.2f}")
d = np.array([(anc[q, "Kstar4k", "p"]["lp_sum"] - anc[q, "Kstar4k", "a"]["lp_sum"])
              - (anc[q, "K", "p"]["lp_sum"] - anc[q, "K", "a"]["lp_sum"]) for q in qa])
bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
print(f"  K*4k - K (paired, p - a): {d.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
nearly = {X: sum(anc[q, X, "p"]["lp_sum"] > -0.5 for q in qa) for X in ("K", "CK", "M")}
print("  present sessions with log p(gold) > -0.5:", ", ".join(f"{k} {v}/{len(qa)}" for k, v in nearly.items()))

print("\n== P4. readout dependence of the DLA presence split (present - absent, 'written' term) ==")
tr = {q: torch.load(os.path.join(sys.argv[1] + ".d", f"{q}.pt")) for q in Q}
NL = 64
attn = header["bands"]["Kearly"] + header["bands"]["Kmid"] + header["bands"]["Klate"]
gdn = [i for i in range(NL) if i not in attn]
for X in ("M", "CK"):
    for wname in ("w_p", "w_a", "w_mid", "w_A"):
        P = []
        for q in Q:
            p, a = tr[q][(X, "p")], tr[q][(X, "a")]
            w = {"w_p": p["w0"], "w_a": a["w0"], "w_mid": (p["w0"].double() + a["w0"].double()) / 2,
                 "w_A": tr[q][("A", None)]["w0"]}[wname].double()
            v = ((p["C0"].double() - a["C0"].double()) @ w).numpy()
            mix, mlp = v[1:1 + NL], v[1 + NL:]
            P.append((mix[gdn].sum(), mix[attn].sum(), mlp.sum(), v.sum()))
        P = np.array(P).mean(0)
        print(f"  {X:3s} {wname:6s} gdn {P[0]:+.2f}  attn {P[1]:+.2f}  mlp {P[2]:+.2f}  total {P[3]:+.2f}")
