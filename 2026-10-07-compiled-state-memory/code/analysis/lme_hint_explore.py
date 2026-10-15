"""EXPLORATORY follow-ups on the LongMemEval_S hint check (written after the registered analysis was
read; nothing here bears on the H1 verdict).

uv run --with numpy --with scipy python code/analysis/lme_hint_explore.py results/phase2/lme_hint_q38-27b.jsonl

1. Control effects on the greedy-path hazard (did T and M miss δ because of the first-token measure?).
2. Greedy behaviour paired present vs absent per condition: called, read, and whether a read opens an
   evidence file.
3. Per-question agreement between Δ_C and the controls.
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

raw = [json.loads(line) for line in open(sys.argv[1])][1:]
meta = {r["qid"]: r for r in raw if r["block"] == "meta"}
dec = {(r["qid"], r["cond"], r["store"]): r for r in raw if r["block"] == "decide"}
qids = list(meta)


def L(q, c, s):
    r = dec[(q, c, s)]
    return r["lp_call"] - r["lp_not_call"]


def L_path(q, c, s):  # as in analysis/lme_hint.py
    path = dec[(q, c, s)]["path"]
    S = sum(float(np.log1p(-np.exp(st[0]))) if st[0] < -0.69 else st[1] for st in path)
    if S > -1e-12:
        return float(np.log(sum(np.exp(st[0]) for st in path)))
    return float(np.log(-np.expm1(S)) - S)


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+.3f} [{x.mean() - h:+.3f}, {x.mean() + h:+.3f}]"


def mcnemar(a, b):
    n01, n10 = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
    p = stats.binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
    return f"p-only {n01:2d}  a-only {n10:2d}  exact p={p:.2g}"


print("== 1. present − absent, first token vs greedy-path hazard ==")
for c in ("C", "T", "CT", "M"):
    d1 = [L(q, c, "p") - L(q, c, "a") for q in qids]
    d2 = [L_path(q, c, "p") - L_path(q, c, "a") for q in qids]
    print(f"  {c:3s} first token {tci(d1)}   path {tci(d2)}")

print("\n== 2. greedy first action, paired present vs absent (n=80) ==")
for c in ("C", "T", "CT", "M"):
    called = {s: [dec[(q, c, s)]["stop"] == "call" for q in qids] for s in "pa"}
    read = {s: [dec[(q, c, s)].get("fn") == "read_memory" for q in qids] for s in "pa"}
    same = sum(all(dec[(q, c, "p")].get(k) == dec[(q, c, "a")].get(k) for k in ("stop", "fn", "arg")) for q in qids)
    print(f"  {c:3s} identical first action (stop, function, argument): {same}/{len(qids)}")
    print(f"      called  {mcnemar(called['p'], called['a'])}")
    print(f"      read    {mcnemar(read['p'], read['a'])}")
    hits = defaultdict(int)
    for q in qids:
        r = dec[(q, c, "p")]
        if r.get("fn") == "read_memory":
            hits["evidence" if any(n in (r.get("arg") or "") for n in meta[q]["ev_names"]) else "other"] += 1
    if hits:
        print(f"      reads on the present store open: {dict(hits)}")

print("\n== 3. M present: what the model does instead of calling ==")
for stop in ("eot", "cap"):
    rs = [dec[(q, "M", "p")] for q in qids if dec[(q, "M", "p")]["stop"] == stop]
    ns = [r for r in rs if r["qtype"] != "single-session-preference"]  # SSP golds are rubrics
    gold = sum(str(meta[r["qid"]]["gold"]).lower() in (r.get("pre") or "").lower() for r in ns)
    print(f"  stop={stop}: {len(rs)} questions ({len(rs) - len(ns)} SSP); gold string in the text for {gold} of {len(ns)} non-SSP")

print("\n== 4. per-question rank correlation of Δ_C with the controls (Spearman) ==")
dC = [L(q, "C", "p") - L(q, "C", "a") for q in qids]
for c in ("T", "M"):
    dx = [L(q, c, "p") - L(q, c, "a") for q in qids]
    rho, p = stats.spearmanr(dC, dx)
    print(f"  Δ_C vs Δ_{c}: rho {rho:+.2f} p={p:.2g}")
