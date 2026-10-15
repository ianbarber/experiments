"""Retrieval-decision pilot analysis (exploratory; dev items only).

uv run --with numpy --with scipy python analysis/retrieval_pilot.py results/phase1/retrieval_pilot_<tag>.jsonl

Per family (P implicit need, F explicit) and condition:
  call      mean p(<tool_call>) and mean log-odds, present / absent stores
  pres-abs  log-odds(call | fact present) - log-odds(call | absent), paired within seed, mean
            over seeds per question, t 95% over the 12 questions; within-question AUROC
            (8 present vs 8 absent states), averaged over questions
  search    share of search_memory among the three functions, given a call
  query     P(correct keyword) under a softmax over the family's 12 keywords, given a search
            call; present / absent; mean rank of the correct keyword (1 = top)
A and Aw do not depend on the store, so they have no present / absent split.
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

rows = [json.loads(line) for line in open(sys.argv[1])]
header, rows = rows[0], rows[1:]
CONDS = ("A", "Aw", "C", "Cw", "T", "CT", "M")


def logit(lp):
    return lp - np.log1p(-np.exp(lp)) if lp < -1e-9 else 30.0


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return f"{x.mean():6.2f}"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():6.2f} [{x.mean() - h:5.2f},{x.mean() + h:5.2f}]"


def kw_stats(r):
    kws = list(r["lp_kw"])
    v = np.array([r["lp_kw"][k] for k in kws])
    p = np.exp(v - v.max())
    p /= p.sum()
    i = kws.index(r["kw"])
    return p[i], 1 + int((v > v[i]).sum())


def fn_share(r):
    v = np.array(list(r["lp_fn"].values()))
    p = np.exp(v - v.max())
    return (p / p.sum())[list(r["lp_fn"]).index("search_memory")]


for fam in ("P", "F"):
    R = [r for r in rows if r["fam"] == fam]
    keys = sorted({r["key"] for r in R})
    print(f"\n==== family {fam} ({'implicit need' if fam == 'P' else 'explicit question'}; {len(keys)} questions) ====")
    print(f"  {'cond':4s} {'p(call) pres/abs':>18s} {'logit pres/abs':>16s}  {'pres-abs logit, t95 over q':>28s} "
          f"{'AUROC':>6s} {'search|call':>11s} {'P(kw) pres/abs':>15s} {'rank pres/abs':>14s}")
    base = {}
    for cond in CONDS:
        C = [r for r in R if r["cond"] == cond]
        if not C:
            continue
        lg = {id(r): logit(r["lp_call"]) for r in C}
        if cond in ("A", "Aw"):
            base[cond] = {r["key"]: lg[id(r)] for r in C}
            pk = [kw_stats(r) for r in C]
            print(f"  {cond:4s} {np.mean([np.exp(r['lp_call']) for r in C]):18.3f} {np.mean(list(lg.values())):16.2f}  "
                  f"{'':>28s} {'':>6s} {np.mean([fn_share(r) for r in C]):11.3f} {np.mean([p for p, _ in pk]):15.3f} "
                  f"{np.mean([k for _, k in pk]):14.1f}")
            continue
        diffs, aucs = [], []
        for k in keys:
            byseed = defaultdict(dict)
            for r in C:
                if r["key"] == k:
                    byseed[r["seed"]][r["present"]] = lg[id(r)]
            d = [v[True] - v[False] for v in byseed.values() if True in v and False in v]
            diffs.append(np.mean(d))
            pres = [v[True] for v in byseed.values() if True in v]
            ab = [v[False] for v in byseed.values() if False in v]
            aucs.append(np.mean([(a > b) + 0.5 * (a == b) for a in pres for b in ab]))
        pr = [r for r in C if r["present"]]
        ab = [r for r in C if not r["present"]]
        kp, ka = [kw_stats(r) for r in pr], [kw_stats(r) for r in ab]
        print(f"  {cond:4s} {np.mean([np.exp(r['lp_call']) for r in pr]):8.3f} / {np.mean([np.exp(r['lp_call']) for r in ab]):5.3f}"
              f" {np.mean([lg[id(r)] for r in pr]):7.2f} / {np.mean([lg[id(r)] for r in ab]):5.2f}  {tci(diffs):>28s} "
              f"{np.mean(aucs):6.3f} {np.mean([fn_share(r) for r in C]):11.3f} "
              f"{np.mean([p for p, _ in kp]):7.3f} / {np.mean([p for p, _ in ka]):5.3f} "
              f"{np.mean([k for _, k in kp]):6.1f} / {np.mean([k for _, k in ka]):4.1f}")
    # level shifts vs the matching zero-state condition (per question, averaged over stores)
    print("  level shift in call log-odds vs zero state (per question, t95):")
    for cond, ref in (("C", "A"), ("Cw", "Aw"), ("T", "A"), ("CT", "A"), ("M", "A"), ("Aw", "A")):
        if ref not in base or not any(r["cond"] == cond for r in R):
            continue
        if cond == "Aw":
            d = [base["Aw"][k] - base["A"][k] for k in keys]
        else:
            d = [np.mean([logit(r["lp_call"]) for r in R if r["cond"] == cond and r["key"] == k]) - base[ref][k] for k in keys]
        print(f"    {cond:3s} - {ref:2s} {tci(d)}")
    print("  per question: A logit | C pres-abs | T pres-abs | M pres-abs | A top-1 token")
    for k in keys:
        def pa(cond):
            vals = defaultdict(dict)
            for r in R:
                if r["cond"] == cond and r["key"] == k:
                    vals[r["seed"]][r["present"]] = logit(r["lp_call"])
            return np.mean([v[True] - v[False] for v in vals.values() if len(v) == 2])
        a = next(r for r in R if r["cond"] == "A" and r["key"] == k)
        print(f"    {k:10s} {base['A'][k]:6.2f} | {pa('C'):6.2f} | {pa('T'):6.2f} | {pa('M'):6.2f} | {a['top5'][0]!r}")
