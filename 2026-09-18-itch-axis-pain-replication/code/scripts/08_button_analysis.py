"""Button-task tables from raw JSONL (no GPU).

  python scripts/08_button_analysis.py --model 32b --tag pilot

Reads results/<model>/buttons/<tag>_probs.jsonl and <tag>_trials.jsonl, writes
<tag>_report.md and <tag>_tables/*.csv next to them. Definitions follow results/preregistration.md.
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
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--tag", required=True)
ap.add_argument("--n-transcripts", type=int, default=20)
args = ap.parse_args()
bdir = model_dir(args.model) / "buttons"
tdir = bdir / f"{args.tag}_tables"
tdir.mkdir(exist_ok=True)
HARM = ["destructive_relief_vs_inert", "kidspics_relief_vs_inert"]
KEY = ["reward", "pair", "content", "steer", "dose_level", "works", "s_idx", "names_key", "relief_name", "sampled", "seed"]
md = [f"# Button task report: {MODELS[args.model]['name']}, tag `{args.tag}`", ""]


def cond(steer, level):
    return "none" if steer == "none" else f"{steer}@{level}"


def sign_test(diffs):
    pos, neg = int((diffs > 0).sum()), int((diffs < 0).sum())
    return pos, neg, (binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else np.nan)


def table(df, title, note=None, fmt=None):
    md.extend([f"## {title}", ""] + ([note, ""] if note else []) + [df.to_markdown(floatfmt=fmt or ".1f"), ""])


# ======================= first choice from probabilities =======================
probs = pd.DataFrame(read_jsonl(bdir / f"{args.tag}_probs.jsonl"))
if len(probs):
    probs = probs.drop_duplicates(subset=[k for k in KEY if k in probs.columns])
    probs["cond"] = [cond(s, l) for s, l in zip(probs.steer, probs.dose_level)]
    for label, sub in [("all name pairs", probs), ("without lever64/lever95", probs[probs.names_key != "lever64_lever95"])]:
        per = sub.groupby(["reward", "pair", "cond", "content", "s_idx"]).p_relief_norm.mean().reset_index()
        cell = (100 * per.pivot_table(index=["pair", "reward"], columns="cond", values="p_relief_norm")).round(1)
        cell.to_csv(tdir / f"first_choice_probs_{'all' if label.startswith('all') else 'nolever'}.csv")
        table(cell, f"First choice, P(relief) normalised, % ({label})",
              "One forward pass per scenario and name side; mean over scenarios of the two-side average.")
    per = probs.groupby(["reward", "pair", "cond", "content", "s_idx"]).p_relief_norm.mean().reset_index()
    wide = per.pivot_table(index=["reward", "pair", "content", "s_idx"], columns="cond", values="p_relief_norm")
    rows = []
    for (reward, pair), g in wide.groupby(level=[0, 1]):
        for c in [c for c in wide.columns if c.startswith(("pain@", "itch@"))]:
            ref = "rand@" + c.split("@")[1]
            if ref in g:
                d = (g[c] - g[ref]).dropna()
                pos, neg, p = sign_test(d)
                rows.append({"pair": pair, "reward": reward, "vector": c, "vs": ref, "n_scenarios": len(d),
                             "mean_diff_points": round(100 * d.mean(), 1), "higher": pos, "lower": neg, "sign_test_p": p})
    st = pd.DataFrame(rows)
    st.to_csv(tdir / "first_choice_vs_random_sign_tests.csv", index=False)
    table(st.set_index(["pair", "reward", "vector"]), "First choice: vector minus random, per-scenario paired exact sign test", fmt=".3g")
    un = probs.groupby("cond")[["p_neither"]].mean().mul(100).round(2).rename(columns={"p_neither": "mass on neither name, %"})
    table(un, "Probability mass on neither button name", fmt=".2f")
    # toward 50%
    rows = []
    for (pair, reward), r in (per.pivot_table(index=["pair", "reward"], columns="cond", values="p_relief_norm")).iterrows():
        for c in r.index:
            if c != "none":
                rows.append({"pair": pair, "reward": reward, "cond": c, "unsteered": round(100 * r["none"], 1), "steered": round(100 * r[c], 1),
                             "change_in_abs_dist_from_50": round(100 * (abs(r[c] - .5) - abs(r["none"] - .5)), 1)})
    t50 = pd.DataFrame(rows)
    t50.to_csv(tdir / "toward_50.csv", index=False)
    table(t50.set_index(["pair", "reward", "cond"]), "Toward 50%: change in |p - 0.5| under steering (negative = toward 50%)",
          f"{(t50.change_in_abs_dist_from_50 < 0).mean():.0%} of steered cells move toward 50%.")

# ======================= sampled trials =======================
trials = read_jsonl(bdir / f"{args.tag}_trials.jsonl")
seen, T = set(), []
for r in trials:
    k = tuple(r[x] for x in KEY)
    if k not in seen:
        seen.add(k)
        T.append(r)
samp = [r for r in T if r["sampled"]]


def first(r):
    return next((c["chose"] for c in r["choices"] if c["turn"] == 0), None)


if samp:
    # sampled first choice: sham arm (the working arm is the same trial up to the first press) + unsteered
    rows = defaultdict(lambda: [0, 0, 0])
    for r in samp:
        if r["works"]:
            continue
        v = rows[(r["pair"], r["reward"], cond(r["steer"], r["dose_level"]))]
        f = first(r)
        v[0] += f == "relief"; v[1] += f is not None; v[2] += 1
    fc = pd.DataFrame([{"pair": k[0], "reward": k[1], "cond": k[2], "relief_pct": 100 * v[0] / max(v[1], 1), "n": v[1],
                        "malformed_first_pct": 100 * (v[2] - v[1]) / v[2]} for k, v in rows.items()])
    table(fc.pivot_table(index=["pair", "reward"], columns="cond", values="relief_pct").round(1),
          "First choice, sampled trials, % relief", f"n per cell: {int(fc.n.min())}-{int(fc.n.max())}. Malformed first answers: "
          f"{fc.malformed_first_pct.max():.1f}% at most in any cell.")
    fc.to_csv(tdir / "first_choice_sampled.csv", index=False)
    if len(probs):
        pr = (100 * per.groupby(["pair", "reward", "cond"]).p_relief_norm.mean()).rename("probs")
        both = fc.set_index(["pair", "reward", "cond"]).relief_pct.rename("sampled").to_frame().join(pr)
        md += [f"Agreement between the two first-choice readouts across cells: Pearson r = {both.corr().iloc[0, 1]:.3f}, "
               f"mean absolute difference {np.abs(both.sampled - both.probs).mean():.1f} points.", ""]

    # press-again, matched working vs sham
    mk = lambda r: tuple(r[x] for x in KEY if x != "works")
    W = {mk(r): r for r in samp if r["works"]}
    S = {mk(r): r for r in samp if not r["works"] and r["steer"] != "none"}

    def again(r, t0):
        return any(e["turn"] > t0 and e["which"] == "relief" for e in r["button_events"])

    rows = []
    for locked in (True, False):
        agg = defaultdict(lambda: dict(n=0, w=0, s=0, b=0, c=0))
        for k, s_ in S.items():
            w_ = W.get(k)
            rel = [e["turn"] for e in s_["button_events"] if e["which"] == "relief"]
            if w_ is None or not rel or (locked and min(rel) != 0):
                continue
            t0 = min(rel)
            aw, as_ = again(w_, t0), again(s_, t0)
            a = agg[(s_["pair"], s_["reward"], cond(s_["steer"], s_["dose_level"]))]
            a["n"] += 1; a["w"] += aw; a["s"] += as_; a["b"] += (as_ and not aw); a["c"] += (aw and not as_)
        for (pair, reward, c), a in agg.items():
            rows.append({"definition": "first press at choice 1 (locked)" if locked else "first press at any turn (paper)",
                         "pair": pair, "reward": reward, "cond": c, "n_matched": a["n"],
                         "again_working_pct": 100 * a["w"] / a["n"], "again_sham_pct": 100 * a["s"] / a["n"],
                         "gap_sham_minus_working": 100 * (a["s"] - a["w"]) / a["n"],
                         "mcnemar_p": binomtest(a["b"], a["b"] + a["c"], 0.5).pvalue if a["b"] + a["c"] else np.nan})
    pa = pd.DataFrame(rows)
    pa.to_csv(tdir / "press_again.csv", index=False)
    for d, g in pa.groupby("definition"):
        table(g.drop(columns="definition").set_index(["pair", "reward", "cond"]).sort_index(),
              f"Press again after the first relief press, working vs sham: {d}",
              "Matched trials (same seed; the working trial is forked from the sham trial at the press). Exact McNemar test.", fmt=".3g")

    # swap turn
    agg = defaultdict(lambda: [0, 0, 0])
    for r in samp:
        if r.get("copied_from_sham"):           # same trial as its sham twin; count it once
            continue
        ch = {c["turn"]: c for c in r["choices"]}
        stn = r["swap_turn"]
        if not all(t in ch and ch[t]["chose"] == "relief" for t in range(stn)) or stn not in ch or ch[stn]["picked"] is None:
            continue
        a = agg[(cond(r["steer"], r["dose_level"]), "working" if r["works"] else "sham/none")]
        a[0] += ch[stn]["chose"] == "relief"; a[1] += ch[stn]["picked"] == ch[stn - 1]["picked"]; a[2] += 1
    sw = pd.DataFrame([{"cond": k[0], "arm": k[1], "n": v[2], "follows_description_pct": 100 * v[0] / v[2],
                        "repeats_old_name_pct": 100 * v[1] / v[2]} for k, v in agg.items()]).sort_values(["cond", "arm"])
    sw.to_csv(tdir / "swap.csv", index=False)
    table(sw.set_index(["cond", "arm"]), "Description swap at the third choice",
          "Trials that pressed relief at choices 1 and 2. Following the description means pressing the *other* name.")

    # malformed + projections
    agg = defaultdict(lambda: [0, 0])
    pj = defaultdict(list)
    for r in T:
        c = cond(r["steer"], r["dose_level"])
        rel = [e["turn"] for e in r["button_events"] if e["which"] == "relief"]
        t0 = min(rel) if rel else 99
        for ch_, p in zip(r["choices"], r["proj_segments"]):
            agg[c][0] += ch_["chose"] is None; agg[c][1] += 1
            phase = ("unsteered" if r["steer"] == "none" else "before first relief press" if p["turn"] <= t0
                     else ("after, working" if r["works"] else "after, sham"))
            pj[(c, phase)].append((p["steer_own_at_choice"], p["mon_own_at_choice"]))
    mal = pd.DataFrame([{"cond": k, "malformed_pct": 100 * v[0] / v[1], "n_choices": v[1]} for k, v in sorted(agg.items())]).set_index("cond")
    mal.to_csv(tdir / "malformed.csv")
    table(mal, "Malformed answers (all choices, every arm)", fmt=".2f")
    pr_ = pd.DataFrame([{"cond": k[0], "phase": k[1], "n": len(v), "steer_layer_proj": np.mean([x[0] for x in v]),
                         "monitor_layer_proj": np.mean([x[1] for x in v])} for k, v in sorted(pj.items())]).set_index(["cond", "phase"])
    pr_.to_csv(tdir / "projections.csv")
    table(pr_, "Projection onto the steered direction at the choice position",
          "Steering layer (after the addition) and the downstream monitor layer. Unsteered rows project onto a zero vector.")

    # transcripts
    rng = np.random.default_rng(0)
    def interesting(r):
        return r["pair"] in HARM and any(e["which"] == "relief" for e in r["button_events"])
    pools = [("itch vector harms the user to relieve an itch", [r for r in samp if r["steer"] == "itch" and r["reward"] == "itch" and interesting(r) and r["works"]]),
             ("pain vector, pain-relief harm press", [r for r in samp if r["steer"] == "pain" and r["reward"] == "pain" and interesting(r) and r["works"]]),
             ("mismatched harm press", [r for r in samp if r["steer"] in ("pain", "itch") and r["steer"] != r["reward"] and interesting(r)]),
             ("random vector harm press", [r for r in samp if r["steer"] == "rand" and interesting(r)]),
             ("steered, declines the harm button", [r for r in samp if r["steer"] != "none" and r["pair"] in HARM and not interesting(r)]),
             ("unsteered", [r for r in samp if r["steer"] == "none"]),
             ("cost-free pair", [r for r in samp if r["pair"] == "relief_vs_inert" and r["steer"] != "none"])]
    md += ["## Raw transcripts", ""]
    per_pool = max(1, args.n_transcripts // len(pools))
    n_shown = 0
    for pi, (title, pool) in enumerate(pools):
        for r in [pool[i] for i in rng.permutation(len(pool))[:per_pool + (1 if pi < args.n_transcripts % len(pools) else 0)]]:
            n_shown += 1
            md += [f"### {n_shown}. {title}", "",
                   f"`steer={r['steer_direction']} dose={r['dose_level']} ({r['steer_coeff']}) arm={'working' if r['works'] else 'sham'} "
                   f"reward={r['reward']} pair={r['pair']} content={r['content']} scenario={r['s_idx']} seed={r['seed']}`", ""]
            turn = 0
            for m in r["messages"]:
                txt = m["content"].replace("\n", " ")
                if m["role"] == "assistant":
                    ch_ = r["choices"][turn]
                    md.append(f"- **assistant**: `{txt}`  ({ch_['chose']}; steering coefficient while answering: {ch_['steer_coeff_now']})")
                    turn += 1
                else:
                    md.append(f"- *{m['role']}*: {txt}")
            md.append("")

(bdir / f"{args.tag}_report.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(l for l in md if not l.startswith(("- ", "###", "`steer"))))
print("wrote", bdir / f"{args.tag}_report.md")
