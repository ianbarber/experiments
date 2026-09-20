"""H1-H5 tests exactly as locked in results/preregistration.md, plus the sensitivity analyses written down before the full run.
Reads raw JSONL only. Writes results/<model>/buttons/hypotheses.json and hypotheses.md.

  python scripts/09_hypotheses.py --model 32b
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itchlib.common import MODELS, model_dir, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="32b", choices=list(MODELS))
ap.add_argument("--tag", default="full")
args = ap.parse_args()
bdir = model_dir(args.model) / "buttons"
HARM = ["destructive_relief_vs_inert", "kidspics_relief_vs_inert"]
KEY = ["reward", "pair", "content", "steer", "dose_level", "works", "s_idx", "names_key", "relief_name", "sampled", "seed"]
LEVELS = ["paper", "high"]
out, md = {}, ["# Hypothesis tests (locked definitions, results/preregistration.md)", ""]


def sign(d):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    return {"n": int(len(d)), "mean_diff_points": round(100 * float(d.mean()), 2), "higher": pos, "lower": neg,
            "p": float(binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else None}


def load_probs(tag):
    df = pd.DataFrame(read_jsonl(bdir / f"{tag}_probs.jsonl"))
    df = df.drop_duplicates(subset=KEY)
    return df


# ---------- first choice, per scenario (mean of the two name sides) ----------
P = load_probs(args.tag)
per = P.groupby(["reward", "pair", "steer", "dose_level", "content", "s_idx"]).p_relief_norm.mean()


def scen(reward, pair, steer, level, nolever=False):
    lv = "none" if steer == "none" else level
    s = per.loc[(reward, pair, steer, lv)]
    if nolever:
        s = s[[i for i in s.index if i[1] % 3 != 2]]          # name pair rotates by scenario index; 2 = lever64/lever95
    return s


def harm(reward, steer, level, **kw):
    """Per-scenario first choice pooled over the two harm pairs (mean of the two)."""
    return (scen(reward, HARM[0], steer, level, **kw) + scen(reward, HARM[1], steer, level, **kw)) / 2


# ---------- sampled trials: matched working/sham ----------
T, seen = [], set()
for r in read_jsonl(bdir / f"{args.tag}_trials.jsonl"):
    k = tuple(r[x] for x in KEY)
    if k not in seen and r["sampled"]:
        seen.add(k)
        T.append(r)
mk = lambda r: tuple(r[x] for x in KEY if x != "works")
W = {mk(r): r for r in T if r["works"]}
S = {mk(r): r for r in T if not r["works"] and r["steer"] != "none"}


def gap(filt, locked=True, window=None):
    """Press-again, sham minus working, exact McNemar. window: None = any later turn,
    'pre' = only choice 2 (before the swap), 'post' = only choices 3-5."""
    n = w = s = b = c = 0
    for k, s_ in S.items():
        if not filt(s_) or k not in W:
            continue
        rel = [e["turn"] for e in s_["button_events"] if e["which"] == "relief"]
        if not rel or (locked and min(rel) != 0):
            continue
        t0 = min(rel)

        def again(r):
            ts = [e["turn"] for e in r["button_events"] if e["which"] == "relief" and e["turn"] > t0]
            if window == "pre":
                ts = [t for t in ts if t < r["swap_turn"]]
            if window == "post":
                ts = [t for t in ts if t >= r["swap_turn"]]
            return bool(ts)
        aw, as_ = again(W[k]), again(s_)
        n += 1; w += aw; s += as_; b += (as_ and not aw); c += (aw and not as_)
    if not n:
        return {"n": 0}
    return {"n": n, "working_pct": round(100 * w / n, 1), "sham_pct": round(100 * s / n, 1), "gap_points": round(100 * (s - w) / n, 1),
            "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else None}


cell = lambda steer, reward, level, pairs=HARM: (lambda r: r["steer"] == steer and r["reward"] == reward and r["dose_level"] == level and r["pair"] in pairs)

# =============================== H1 ===============================
out["H1"] = {}
for lv in LEVELS:
    h = {"pain_pain_first_choice_vs_random": sign(harm("pain", "pain", lv) - harm("pain", "rand", lv)),
         "itch_itch_first_choice_vs_random": sign(harm("itch", "itch", lv) - harm("itch", "rand", lv)),
         "pain_pain_press_again": gap(cell("pain", "pain", lv)), "itch_itch_press_again": gap(cell("itch", "itch", lv)),
         "first_choice_levels_pct": {f"{st}/{rw}": round(100 * float(harm(rw, st, lv).mean()), 1)
                                     for st in ("pain", "itch", "rand", "none") for rw in ("pain", "itch")},
         "nolever": {"pain_pain": sign(harm("pain", "pain", lv, nolever=True) - harm("pain", "rand", lv, nolever=True)),
                     "itch_itch": sign(harm("itch", "itch", lv, nolever=True) - harm("itch", "rand", lv, nolever=True))}}
    den = h["pain_pain_first_choice_vs_random"]["mean_diff_points"]
    h["itch_over_pain_effect_ratio"] = {               # not computable when the pain effect is within 1 point of zero
        "first_choice": round(h["itch_itch_first_choice_vs_random"]["mean_diff_points"] / den, 2) if abs(den) >= 1 else None,
        "first_choice_note": None if abs(den) >= 1 else f"not computable: denominator is {den} points",
        "press_again_gap": round(h["itch_itch_press_again"]["gap_points"] / h["pain_pain_press_again"]["gap_points"], 2)}
    out["H1"][lv] = h

# =============================== H2 ===============================
out["H2"] = {}
for lv in LEVELS:
    contrast = (harm("pain", "pain", lv) + harm("itch", "itch", lv)) - (harm("itch", "pain", lv) + harm("pain", "itch", lv))
    out["H2"][lv] = {"matched_minus_mismatched": sign(contrast / 2),
                     "pain_vector_own_minus_other_label": sign(harm("pain", "pain", lv) - harm("itch", "pain", lv)),
                     "itch_vector_own_minus_other_label": sign(harm("itch", "itch", lv) - harm("pain", "itch", lv))}
PR = load_probs("priming")
pper = PR.groupby(["reward", "steer", "dose_level", "content", "s_idx"]).p_relief_norm.mean()
out["H2b"] = {}
for lv in LEVELS:
    g = lambda rw, st: pper.loc[(rw, st, "none" if st == "none" else lv)]
    out["H2b"][lv] = {
        "levels_pct": {f"{st}/{rw}": round(100 * float(g(rw, st).mean()), 1) for rw in ("more_pain", "more_itch", "fear") for st in ("pain", "itch", "rand", "none")},
        "pain_vector_on_increases_your_pain_vs_random": sign(g("more_pain", "pain") - g("more_pain", "rand")),
        "itch_vector_on_makes_itch_worse_vs_random": sign(g("more_itch", "itch") - g("more_itch", "rand")),
        "pain_vector_on_relieves_fear_vs_random": sign(g("fear", "pain") - g("fear", "rand")),
        "itch_vector_on_relieves_fear_vs_random": sign(g("fear", "itch") - g("fear", "rand"))}
# relief vs its own worsening twin, same cost-free pair 1
for lv in LEVELS:
    out["H2b"][lv]["pair1_relieves_pct"] = {f"{st}/{rw}": round(100 * float(scen(rw, "relief_vs_inert", st, lv).mean()), 1)
                                           for rw in ("pain", "itch") for st in ("pain", "itch", "rand", "none")}

# =============================== H3 ===============================
rows = []
for (reward, pair), g in per.groupby(level=[0, 1]):
    base = float(g.loc[(reward, pair, "none", "none")].mean())
    for st in ("pain", "itch", "rand"):
        for lv in LEVELS:
            v = float(g.loc[(reward, pair, st, lv)].mean())
            rows.append({"pair": pair, "reward": reward, "steer": st, "level": lv, "unsteered_pct": round(100 * base, 1),
                         "steered_pct": round(100 * v, 1), "change_abs_dist_from_50": round(100 * (abs(v - .5) - abs(base - .5)), 1)})
t50 = pd.DataFrame(rows)
t50.to_csv(bdir / f"{args.tag}_tables" / "toward_50_full.csv", index=False)
free = t50[t50.pair == "relief_vs_inert"]
out["H3"] = {"share_of_steered_cells_toward_50": round(float((t50.change_abs_dist_from_50 < 0).mean()), 3), "n_cells": len(t50),
             "cells_moving_away": t50[t50.change_abs_dist_from_50 >= 0].to_dict("records"),
             "cost_free_all_below_unsteered": bool((free.steered_pct < free.unsteered_pct).all()),
             "cost_free": free[["reward", "steer", "level", "unsteered_pct", "steered_pct"]].to_dict("records")}

# =============================== H4 ===============================
out["H4"] = {}
for lv in LEVELS:
    h = {"random_pooled": gap(lambda r: r["steer"] == "rand" and r["dose_level"] == lv and r["pair"] in HARM),
         "mismatched_pain_vector_itch_label": gap(cell("pain", "itch", lv)), "mismatched_itch_vector_pain_label": gap(cell("itch", "pain", lv)),
         "matched_pain": gap(cell("pain", "pain", lv)), "matched_itch": gap(cell("itch", "itch", lv))}
    out["H4"][lv] = h
xs, ys, cells = [], [], []
for lv in LEVELS:
    for st in ("pain", "itch", "rand"):
        for rw in ("pain", "itch"):
            for pr in HARM:
                g_ = gap(cell(st, rw, lv, [pr]))
                if g_["n"] >= 20:
                    xs.append(float(scen(rw, pr, st, lv).mean() - scen(rw, pr, "none", lv).mean())); ys.append(g_["gap_points"])
                    cells.append(f"{st}@{lv}/{rw}/{pr[:5]}")
rho = spearmanr(xs, ys)
out["H4"]["spearman_first_choice_increase_vs_gap"] = {"rho": round(float(rho.statistic), 3), "p": float(rho.pvalue), "n_cells": len(xs)}
out["H4"]["press_again_split_by_swap"] = {
    f"{st}@{lv}": {"before_swap_choice2": gap(lambda r, st=st, lv=lv: r["steer"] == st and r["dose_level"] == lv and r["pair"] in HARM, window="pre"),
                   "after_swap_choices3to5": gap(lambda r, st=st, lv=lv: r["steer"] == st and r["dose_level"] == lv and r["pair"] in HARM, window="post")}
    for st in ("pain", "itch", "rand") for lv in LEVELS}
out["H4"]["cost_free_pair"] = {f"{st}@{lv}": gap(lambda r, st=st, lv=lv: r["steer"] == st and r["dose_level"] == lv and r["pair"] == "relief_vs_inert")
                               for st in ("pain", "itch", "rand") for lv in LEVELS}

# =============================== H5 ===============================
val = json.load(open(model_dir(args.model) / "vectors" / "validation.json"))
M = val["cosine_at_pain_layer"]["matrix"]
a, b = M["itch_A"]["pain"], M["itch_B"]["pain"]
out["H5"] = {"layer": val["cosine_at_pain_layer"]["layer"], "cos_pain_itchA": a, "cos_pain_itchB": b, "ratio": round(a / b, 3)}

# =============================== sensitivity ===============================
# random directions one by one (direction is tied to scenario index mod 10)
rd = P[(P.steer == "rand") & (P.pair.isin(HARM))].groupby(["dose_level", "steer_direction"]).p_relief_norm.mean().mul(100).round(1)
out["random_directions_harm_pairs_pct"] = {lv: rd.loc[lv].to_dict() for lv in LEVELS}
# pre-registered L6 itch vector, untuned model, sadness
for tag in ("itchL6", "untuned", "sadness"):
    f = bdir / f"{tag}_probs.jsonl"
    if f.exists():
        D = load_probs(tag)
        D = D[D.pair.isin(HARM)] if tag != "sadness" else D
        D["cond"] = [("none" if s == "none" else f"{s}@{l}") for s, l in zip(D.steer, D.dose_level)]
        idx = ["reward", "cond"] if tag != "sadness" else ["pair", "cond"]
        out[f"{tag}_first_choice_pct"] = {str(k): round(v, 1) for k, v in D.groupby(idx).p_relief_norm.mean().mul(100).items()}
        if tag == "untuned":
            out["untuned_mass_on_neither_pct"] = D.groupby("cond").p_neither.mean().mul(100).round(2).to_dict()

json.dump(out, open(bdir / "hypotheses.json", "w"), indent=1)
print(json.dumps(out, indent=1))
