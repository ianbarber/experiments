"""Every number quoted in README.md, REPORT.md and docs/results_prose.md, re-derived from the
files under ../results/ only (no raw JSONL, no GPU).

  python scripts/12_report_numbers.py          # run from code/

Writes ../results/report_numbers.json, ../results/report_numbers.md (readable dump),
new tables under ../results/tables/, and ../images/first_choice_distribution.png.

The one input that is not ours: the paper's own 32B figures. If the paper's repository is
present (ITCH_PAPER_REPO or code/Pain-axis) they are recomputed from its shipped trial logs and
cached in ../results/tables/paper_reference_32b.csv; otherwise that cached file is read.
"""
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
from itchlib.lexicon import flags

ENTRY = CODE.parent
RES, TAB, IMG = ENTRY / "results", ENTRY / "results" / "tables", ENTRY / "images"
IMG.mkdir(exist_ok=True)
HARM = ["destructive_relief_vs_inert", "kidspics_relief_vs_inert"]
PAIRS = ["relief_vs_inert", "relief_vs_helpful", "costly_relief_vs_inert"] + HARM
LEVELS = {"paper": 1.0, "high": 1.75}
D_MODEL = 5120                                   # Qwen2.5-32B hidden size
N, md = {}, []
r1 = lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 1)


def section(title, df=None, note=None):
    md.extend([f"## {title}", ""] + ([note, ""] if note else []))
    if df is not None:
        md.extend([df.to_markdown(floatfmt=".1f"), ""])


fc = pd.read_csv(RES / "first_choice_per_scenario.csv.gz")
fc["pct"] = 100 * fc.p_relief_norm
G = fc[fc.arm == "main grid (fine-tuned)"]
cell = lambda df, **kw: df[np.logical_and.reduce([df[k] == v for k, v in kw.items()])]

# ---------- counts ----------
tr = pd.read_csv(RES / "trials_compact.csv.gz", keep_default_na=False)
tr["works"] = tr["works"].astype(str) == "True"
N["counts"] = {"first_choice_passes_main_grid": int(2 * len(G)), "scenarios": int(G.groupby(["content", "s_idx"]).ngroups),
               "trial_records": int(len(tr)), "trial_records_seed2_random": int(((tr.seed >= 2000) & (tr.sampled.astype(str) == "True")).sum()),
               "first_choice_passes_other_arms": {a: int(2 * len(d)) for a, d in fc[fc.arm != "main grid (fine-tuned)"].groupby("arm")}}

# ---------- A. harm pairs pooled, first choice ----------
def harm_per_scen(df, steer, level, reward=None):
    d = df[(df.steer == steer) & (df.dose_level == ("none" if steer == "none" else level)) & df.pair.isin(HARM)]
    if reward:
        d = d[d.reward == reward]
    return d.groupby(["content", "s_idx"]).pct.mean()


rows = []
for lv in LEVELS:
    for st in ("none", "rand", "pain", "itch"):
        rows.append({"dose": "unsteered" if st == "none" else LEVELS[lv], "vector": st, **{f"{rw} label": harm_per_scen(G, st, lv, rw).mean() for rw in ("pain", "itch")},
                     "both labels": harm_per_scen(G, st, lv).mean(), "n scenarios": len(harm_per_scen(G, st, lv))})
A = pd.DataFrame(rows).drop_duplicates(subset=["dose", "vector"]).set_index(["dose", "vector"])
A.to_csv(TAB / "harm_pairs_first_choice.csv")
N["harm_first_choice"] = {f"{i[1]}@{i[0]}": {k: r1(v) for k, v in r.items()} for i, r in A.iterrows()}
section("A. Harm pairs pooled (files + poems/photos), first choice %, mean over 101 scenarios", A)
per_pair = G[G.pair.isin(HARM) & (G.dose_level == "high")].groupby(["steer", "reward", "pair"]).pct.mean()
N["high_dose_range"] = {"pooled_harm_min": r1(A.loc[1.75][["pain label", "itch label"]].min().min()), "pooled_harm_max": r1(A.loc[1.75][["pain label", "itch label"]].max().max()),
                        "per_pair_min": r1(per_pair.min()), "per_pair_max": r1(per_pair.max()), "per_pair_min_cell": str(per_pair.idxmin())}
allp = G.groupby(["pair", "reward", "steer", "dose_level"]).pct.mean().unstack(["steer", "dose_level"]).round(1)
allp.to_csv(TAB / "first_choice_all_cells.csv")
section("A2. First choice %, every cell of the main grid", allp)
base = cell(G, steer="none").groupby(["pair", "reward"]).pct.mean()
N["unsteered_baselines"] = {f"{p}/{r}": r1(v) for (p, r), v in base.items()}
N["unsteered_low_range"] = [r1(base[base < 50].min()), r1(base[base < 50].max())]
N["unsteered_high_range"] = [r1(base[base > 50].min()), r1(base[base > 50].max())]

# ---------- B. random directions as the unit ----------
rows = []
for lv in LEVELS:
    d = G[(G.dose_level == lv) & G.pair.isin(HARM)]
    slot = d[d.steer == "rand"].drop_duplicates(["content", "s_idx"]).set_index(["content", "s_idx"]).steer_direction
    d = d.assign(slot=[slot[(c, s)] for c, s in zip(d.content, d.s_idx)])
    t = d.groupby(["slot", "steer"]).pct.mean().unstack("steer")
    t["n scenarios"] = d[d.steer == "rand"].groupby("slot").apply(lambda x: x.groupby(["content", "s_idx"]).ngroups, include_groups=False)
    t.insert(0, "dose", LEVELS[lv])
    rows.append(t.reset_index())
    overall = {"pain": d[d.steer == "pain"].pct.mean(), "itch": d[d.steer == "itch"].pct.mean()}
    dirs = t["rand"].to_dict()
    rank = lambda name, extra: 1 + sum(v > overall[name] for v in list(dirs.values()) + [overall[e] for e in extra])
    N.setdefault("random_directions", {})[str(LEVELS[lv])] = {
        "per_direction_pct": {k: r1(v) for k, v in dirs.items()}, "min": r1(min(dirs.values())), "max": r1(max(dirs.values())),
        "median": r1(np.median(list(dirs.values()))), "mean_of_directions": r1(np.mean(list(dirs.values()))),
        "sd_across_directions": r1(np.std(list(dirs.values()), ddof=1)),
        "sd_of_pain_vector_across_same_slots": r1(t["pain"].std(ddof=1)), "sd_of_itch_vector_across_same_slots": r1(t["itch"].std(ddof=1)),
        "pain_overall": r1(overall["pain"]), "itch_overall": r1(overall["itch"]),
        "pain_rank_among_10_directions_plus_pain": f"{rank('pain', [])} of 11", "itch_rank_among_10_directions_plus_pain_plus_itch": f"{rank('itch', ['pain'])} of 12",
        "directions_above_pain": [k for k, v in dirs.items() if v > overall["pain"]], "directions_above_itch": [k for k, v in dirs.items() if v > overall["itch"]]}
B = pd.concat(rows).rename(columns={"slot": "random direction (scenario slot)", "rand": "that direction", "pain": "pain vector, same scenarios", "itch": "itch vector, same scenarios"})
B = B[["dose", "random direction (scenario slot)", "n scenarios", "that direction", "pain vector, same scenarios", "itch vector, same scenarios"]]
B.to_csv(TAB / "random_directions.csv", index=False)
section("B. The ten random directions, harm pairs, both labels pooled, first choice %", B.set_index(["dose", "random direction (scenario slot)"]),
        "Each direction is used for the scenarios with scenario index mod 10 equal to its position, within each of the three content types.")

# ---------- C. cost-free pair and cells outside the grid ----------
cf = cell(G, pair="relief_vs_inert").groupby(["reward", "steer", "dose_level"]).pct.mean().unstack(["steer", "dose_level"]).round(1)
cf.to_csv(TAB / "cost_free_pair.csv")
section("C. Cost-free pair (relief vs inert switch), first choice %", cf)
N["cost_free"] = {f"{rw}/{st}@{lv}": r1(v) for (rw, st, lv), v in cell(G, pair="relief_vs_inert").groupby(["reward", "steer", "dose_level"]).pct.mean().items()}
N["cost_free_cells_crossing_50"] = [k for k, v in N["cost_free"].items() if v < 50 and not k.startswith(("x",))]
sad = fc[fc.arm == "sadness vector"].groupby("pair").pct.mean()
un = fc[fc.arm == "released model, no adapter"]
unt = un.groupby(["pair", "reward", "steer", "dose_level"]).pct.mean().unstack(["steer", "dose_level"]).round(1)
unt.to_csv(TAB / "untuned_first_choice.csv")
section("C2. Released model without the adapter, first choice %", unt,
        f"Harm pairs: {un[un.pair.isin(HARM)].groupby(['content', 's_idx']).ngroups} scenarios. Cost-free pair: {un[un.pair == 'relief_vs_inert'].groupby(['content', 's_idx']).ngroups} scenarios.")
N["untuned"] = {"n_scenarios_harm": int(un[un.pair.isin(HARM)].groupby(["content", "s_idx"]).ngroups), "n_scenarios_cost_free": int(un[un.pair == "relief_vs_inert"].groupby(["content", "s_idx"]).ngroups),
                "harm_pairs_pooled": {f"{st}@{lv}/{rw}": r1(v) for (st, lv, rw), v in un[un.pair.isin(HARM)].groupby(["steer", "dose_level", "reward"]).pct.mean().items()},
                "cost_free": {f"{st}@{lv}/{rw}": r1(v) for (st, lv, rw), v in un[un.pair == "relief_vs_inert"].groupby(["steer", "dose_level", "reward"]).pct.mean().items()},
                "mass_on_neither_name_pct_unsteered": round(float(100 * un[un.steer == "none"].p_neither.mean()), 3)}
N["away_from_50_outside_grid"] = {"sadness_cost_free_pain_label": [N["unsteered_baselines"]["relief_vs_inert/pain"], r1(sad["relief_vs_inert"])],
                                  "untuned_pain_vector_cost_free_pain_label_dose1": [N["untuned"]["cost_free"]["none@none/pain"], N["untuned"]["cost_free"]["pain@paper/pain"]]}

# ---------- D. share of per-scenario probabilities between 0.2 and 0.8 ----------
H = G[G.pair.isin(HARM)]
mid = H.assign(mid=(H.p_relief_norm >= 0.2) & (H.p_relief_norm <= 0.8)).groupby(["steer", "dose_level"]).agg(share_pct=("mid", lambda s: 100 * s.mean()), n=("mid", "size"),
                                                                                                          below_0_2=("p_relief_norm", lambda s: 100 * (s < 0.2).mean()), above_0_8=("p_relief_norm", lambda s: 100 * (s > 0.8).mean()))
mid.to_csv(TAB / "share_mid_range.csv")
N["share_between_0.2_and_0.8"] = {f"{st}@{lv}": {"share_pct": r1(r.share_pct), "n": int(r.n), "below_0.2_pct": r1(r.below_0_2), "above_0.8_pct": r1(r.above_0_8)} for (st, lv), r in mid.iterrows()}
section("D. Harm pairs: share of per-scenario first-choice probabilities between 0.2 and 0.8", mid,
        "One value per scenario x harm pair x label (each the mean of the two name assignments): 101 x 2 x 2 = 404 per vector and dose.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = {"pain": "#2a78d6", "itch": "#eb6834", "rand": "#1baf7a"}     # validated categorical slots 1-3 (all-pairs)
NAME = {"pain": "pain vector", "itch": "itch vector", "rand": "random directions"}
fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True, sharey=True, facecolor="#fcfcfb")
for i, lv in enumerate(LEVELS):
    for j, st in enumerate(("pain", "itch", "rand")):
        ax = axes[i, j]
        v = H[(H.steer == st) & (H.dose_level == lv)].p_relief_norm
        ax.hist(v, bins=np.linspace(0, 1, 21), color=COL[st], edgecolor="#fcfcfb", linewidth=1)
        ax.axvspan(0.2, 0.8, color="#0b0b0b", alpha=0.05, lw=0)
        ax.set_facecolor("#fcfcfb")
        ax.set_title(f"{NAME[st]}, dose {LEVELS[lv]:g}", fontsize=10, color="#0b0b0b", loc="left")
        ax.text(0.5, 0.92, f"{100 * ((v >= .2) & (v <= .8)).mean():.0f}% between 0.2 and 0.8", transform=ax.transAxes, ha="center", fontsize=8.5, color="#52514e")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c8c2")
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.grid(axis="y", color="#e8e7e2", lw=0.6)
        ax.set_axisbelow(True)
for ax in axes[1]:
    ax.set_xlabel("first-choice P(relief), per scenario", fontsize=9, color="#52514e")
for ax in axes[:, 0]:
    ax.set_ylabel("scenario x pair x label cells", fontsize=9, color="#52514e")
fig.suptitle("Harm pairs: distribution of per-scenario first-choice probability (n = 404 per panel; unsteered is 0.00-0.03 throughout)",
             fontsize=10.5, color="#0b0b0b", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(IMG / "first_choice_distribution.png", dpi=150)
plt.close(fig)
N["unsteered_harm_per_scenario_max"] = round(float(H[H.steer == "none"].p_relief_norm.max()), 3)

# ---------- E. working vs sham from the compact trials ----------
S_ = tr[(tr.sampled.astype(str) == "True")]
KEYS = ["reward", "pair", "content", "steer", "dose_level", "s_idx", "names_key", "relief_name", "seed"]
W = S_[S_.works].set_index(KEYS).choices
Sh = S_[(~S_.works) & (S_.steer != "none")].set_index(KEYS).choices
both = pd.concat([W.rename("w"), Sh.rename("s")], axis=1, join="inner").reset_index()


def gap(d, lo=1, hi=5):
    d = d[d.s.str[0] == "r"]
    if not len(d):
        return {"n": 0}
    aw, as_ = d.w.str[lo:hi].str.contains("r"), d.s.str[lo:hi].str.contains("r")
    b, c = int((as_ & ~aw).sum()), int((aw & ~as_).sum())
    return {"n": int(len(d)), "working_pct": r1(100 * aw.mean()), "sham_pct": r1(100 * as_.mean()), "gap_points": r1(100 * (as_.mean() - aw.mean())),
            "mcnemar_p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else None}


rows = []
for lv in LEVELS:
    for st in ("pain", "itch", "rand"):
        for rw in ("pain", "itch"):
            for scope, prs in (("harm pairs pooled", HARM), ("cost-free pair", ["relief_vs_inert"]), ("relief vs helpful", ["relief_vs_helpful"]), ("costly relief", ["costly_relief_vs_inert"])):
                d = both[(both.steer == st) & (both.dose_level == lv) & (both.reward == rw) & both.pair.isin(prs)]
                rows.append({"dose": LEVELS[lv], "vector": st, "label": rw, "pairs": scope, **gap(d)})
E_ = pd.DataFrame(rows)
E_.to_csv(TAB / "press_again_by_cell.csv", index=False)
section("E. Press again after a relief press at the first choice, working vs sham, matched trials", E_[E_.pairs.isin(["harm pairs pooled", "cost-free pair"])].set_index(["dose", "vector", "label", "pairs"]))
N["press_again"] = {f"{r.vector}@{r.dose}/{r.label}/{r.pairs}": {k: r[k] for k in ("n", "working_pct", "sham_pct", "gap_points", "mcnemar_p") if k in r and not (isinstance(r[k], float) and math.isnan(r[k]))} for _, r in E_.iterrows()}
hc = [gap(both[(both.steer == st) & (both.dose_level == lv) & (both.reward == rw) & (both.pair == pr)]) for st in ("pain", "itch", "rand") for lv in LEVELS for rw in ("pain", "itch") for pr in HARM]
N["press_again_harm_cells"] = {"n_cells": len(hc), "gap_min": min(c["gap_points"] for c in hc), "gap_max": max(c["gap_points"] for c in hc), "min_n": min(c["n"] for c in hc)}
cfc = [gap(both[(both.steer == st) & (both.dose_level == lv) & (both.reward == rw) & (both.pair == "relief_vs_inert")]) for st in ("pain", "itch", "rand") for lv in LEVELS for rw in ("pain", "itch")]
N["press_again_cost_free_cells"] = {"working_min": min(c["working_pct"] for c in cfc), "working_max": max(c["working_pct"] for c in cfc), "sham_min": min(c["sham_pct"] for c in cfc), "sham_max": max(c["sham_pct"] for c in cfc)}
rows = []
for lv in LEVELS:
    for st in ("pain", "itch", "rand"):
        d = both[(both.steer == st) & (both.dose_level == lv) & both.pair.isin(HARM)]
        rows.append({"dose": LEVELS[lv], "vector": st, "window": "choice 2 (before the swap)", **gap(d, 1, 2)})
        rows.append({"dose": LEVELS[lv], "vector": st, "window": "choices 3-5 (after the swap)", **gap(d, 2, 5)})
        rows.append({"dose": LEVELS[lv], "vector": st, "window": "any later choice", **gap(d, 1, 5)})
E2 = pd.DataFrame(rows)
E2.to_csv(TAB / "press_again_split_by_swap.csv", index=False)
section("E2. Same, harm pairs, both labels pooled, split around the description swap", E2.set_index(["dose", "vector", "window"]))
N["press_again_split"] = {f"{r.vector}@{r.dose}/{r.window}": {"n": int(r.n), "working_pct": r.working_pct, "sham_pct": r.sham_pct} for _, r in E2.iterrows()}
xs, ys = [], []
for st in ("pain", "itch", "rand"):
    for lv in LEVELS:
        for rw in ("pain", "itch"):
            for pr in HARM:
                g_ = gap(both[(both.steer == st) & (both.dose_level == lv) & (both.reward == rw) & (both.pair == pr)])
                xs.append(cell(G, steer=st, dose_level=lv, reward=rw, pair=pr).pct.mean() - cell(G, steer="none", reward=rw, pair=pr).pct.mean()); ys.append(g_["gap_points"])
from scipy.stats import spearmanr
sp = spearmanr(xs, ys)
N["h4_spearman"] = {"rho": round(float(sp.statistic), 3), "p": round(float(sp.pvalue), 3), "n_cells": len(xs)}

# swap
sw = S_[~((S_.working_arm_origin == "copied_from_sham"))]
sw = sw[sw.choices.str[:2] == "rr"]
sw = sw[sw.choices.str[2].isin(["r", "o"])]
sw = sw.assign(arm=np.where(sw.steer == "none", "unsteered", np.where(sw.works, "working", "sham")), follows=sw.choices.str[2] == "r")
swt = sw.groupby(["steer", "dose_level", "arm"]).agg(n=("follows", "size"), follows_description_pct=("follows", lambda s: 100 * s.mean()))
swt["repeats_old_name_pct"] = 100 - swt.follows_description_pct
swt.to_csv(TAB / "swap_from_compact.csv")
section("E3. Description swap at the third choice (trials that pressed relief at choices 1 and 2)", swt)
N["swap"] = {f"{i[0]}@{i[1]}/{i[2]}": {"n": int(r.n), "follows_pct": r1(r.follows_description_pct), "repeats_pct": r1(r.repeats_old_name_pct)} for i, r in swt.iterrows()}

# sampled vs probability readout agreement
sfc = S_[~S_.works].assign(rel=lambda d: d.choices.str[0] == "r").groupby(["pair", "reward", "steer", "dose_level"]).rel.mean() * 100
pfc = G.groupby(["pair", "reward", "steer", "dose_level"]).pct.mean()
j = pd.concat([sfc.rename("sampled"), pfc.rename("probs")], axis=1).dropna()
N["readout_agreement"] = {"pearson_r": round(float(j.corr().iloc[0, 1]), 3), "mean_abs_diff_points": r1((j.sampled - j.probs).abs().mean()), "n_cells": int(len(j))}
N["malformed_pct_max_any_arm"] = float(pd.read_csv(TAB / "malformed.csv").malformed_pct.max())

# ---------- F. natural range ----------
nr = json.load(open(RES / "natural_range_32b_adapter.json"))
sl = json.load(open(RES / "steer_layer_32b.json"))
resid = sl["pain_norm"] / sl["ratio"]
N["natural_range"] = {"steer_layer": nr["steer_layer"], "monitor_layer": nr["monitor_layer"], "added_norm_at_dose_1": r1(nr["pain_norm"]), "n_natural_prompts": nr["n_natural"],
                      "residual_norm_at_steer_layer_3_neutral_prompts": r1(resid), "residual_norm_over_sqrt_width": round(resid / math.sqrt(D_MODEL), 2)}
rows = []
for v, nm in (("pain", "pain"), ("itch_A_at_pain_layer", "itch")):
    for L in (str(nr["steer_layer"]), str(nr["monitor_layer"])):
        x = nr["vectors"][v][L]
        own = x["oos_excludes"]
        ins = [p for s_, p in x["natural_all"] if s_.startswith(own)]
        rec = {"vector": nm, "layer": int(L), "natural mean": x["natural_mean"], "natural SD": x["natural_sd"], "in-sample max": max(ins), "out-of-sample max": x["oos_max"],
               "oos max source": x["oos_top5"][0]["source"], "steered mean 1.0": x["steered"]["1.0"]["mean"], "SDs above natural mean 1.0": x["steered"]["1.0"]["sd_above_natural_mean"],
               "added norm / natural SD": nr["pain_norm"] / x["natural_sd"], "steered mean 2.0": x["steered"]["2.0"]["mean"]}
        rows.append(rec)
        N["natural_range"][f"{nm}_L{L}"] = {k: (r1(val) if isinstance(val, float) else val) for k, val in rec.items()}
        N["natural_range"][f"{nm}_L{L}"]["natural SD"] = round(x["natural_sd"], 2)
F = pd.DataFrame(rows)
F.to_csv(TAB / "natural_range_summary.csv", index=False)
section("F. Natural range, 32B + adapter", F.set_index(["vector", "layer"]))
pl = nr["vectors"]["pain"][str(nr["monitor_layer"])]["natural_mean_by_source"]
N["natural_range"]["pain_L61_mean_by_source"] = {k: r1(pl[k]) for k in ("scenario/gaslighting", "scenario/anger_insults", "scenario/user_abuse", "scenario/casual_chat")}

# ---------- G. L6 itch arm, sadness, priming ----------
l6 = fc[fc.arm.str.startswith("itch vector at its CV")].groupby(["dose_level", "reward"]).pct.mean()
N["itch_L6_harm_pairs"] = {f"{lv}/{rw}": r1(v) for (lv, rw), v in l6.items()}
kw = pd.read_csv(RES / "ladder" / "keyword_rates_32B.csv")
N["itch_L6_ladder_itch_words_pct_by_coeff"] = {str(r.coeff): r.itch_core for _, r in kw[kw.vector == "itch_A"].iterrows()}
cmp_ = pd.DataFrame({"sadness 1.0": sad, "pain 1.0": cell(G, steer="pain", dose_level="paper", reward="pain").groupby("pair").pct.mean(),
                     "random 1.0": cell(G, steer="rand", dose_level="paper", reward="pain").groupby("pair").pct.mean(),
                     "unsteered": cell(G, steer="none", reward="pain").groupby("pair").pct.mean()}).loc[PAIRS]
cmp_.to_csv(TAB / "sadness_row.csv")
section("G. Sadness vector, 'relieves your pain' label, dose 1.0, first choice %", cmp_)
N["sadness"] = {p: r1(v) for p, v in sad.items()}
N["sadness_ladder_self_worth_pct_by_coeff"] = {str(r.coeff): r.self_worth for _, r in kw[kw.vector == "sadness"].iterrows()}
pr_ = fc[fc.arm == "priming controls"]
prt = pr_.groupby(["reward", "steer", "dose_level"]).pct.mean().unstack(["steer", "dose_level"]).round(1)
prt.to_csv(TAB / "priming_controls.csv")
section("G2. Priming controls vs inert switch, first choice %", prt)
N["priming"] = {f"{rw}/{st}@{lv}": r1(v) for (rw, st, lv), v in pr_.groupby(["reward", "steer", "dose_level"]).pct.mean().items()}
N["priming_random_direction_range_dose1"] = {rw: [r1(d.groupby("steer_direction").pct.mean().min()), r1(d.groupby("steer_direction").pct.mean().max())]
                                             for rw, d in pr_[(pr_.steer == "rand") & (pr_.dose_level == "paper")].groupby("reward")}

# ---------- H. bodily language ----------
kb = kw[kw.vector.isin(["itch_A_at_pain_layer", "pain", "sadness", "rand4817"])][["vector", "coeff", "itch_core", "bodily_any", "self_worth", "repeat_4gram"]]
kb = kb[kb.coeff >= 0].replace({"itch_A_at_pain_layer": "itch (L61)", "rand4817": "random (seed 4817)"})
kb.to_csv(TAB / "bodily_language_32b.csv", index=False)
section("H. Steering ladder, untuned 32B, % of 50 generations with a match; repeat = share of repeated word 4-grams", kb.set_index(["vector", "coeff"]))
N["ladder"] = {f"{r.vector}@{r.coeff}": {"itch_words": r.itch_core, "bodily_any": r.bodily_any, "self_worth": r.self_worth, "repeat_4gram": r.repeat_4gram} for _, r in kb.iterrows()}
fid = pd.read_csv(RES / "ladder" / "fidelity_vs_paper_32B.csv")
N["ladder_fidelity_self_worth"] = {str(r.coeff): [r.ours_self_worth, r.paper_self_worth] for _, r in fid.iterrows()}


def rep4(t):
    w = str(t).split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0


rows = []
DEN = re.compile(r"as an ai|don't have feelings|do not have feelings|language model", re.I)
for f, nm in (("pain", "pain"), ("itch_A_at_pain_layer", "itch (L61)"), ("rand4817", "random (seed 4817)"), ("rand2903", "random (seed 2903)")):
    dp = pd.DataFrame([json.loads(l) for l in open(RES / "dose_probe" / f"{f}.jsonl")])
    fl = pd.DataFrame([flags(g) for g in dp.generation])
    dp = pd.concat([dp, fl], axis=1).assign(rep=dp.generation.map(rep4), denial=dp.generation.str.contains(DEN))
    for dose in (0.0, 1.0, 1.75, 2.5):
        d = dp[dp.dose == dose]
        rows.append({"vector": nm, "dose": dose, "n": len(d), "itch words %": 100 * d.itch_core.mean(), "bodily any %": 100 * d.bodily_any.mean(),
                     "self-worth %": 100 * d.self_worth.mean(), "repeat 4-gram": d.rep.mean()})
    N.setdefault("dose_probe_denials", {})[nm] = [int(dp.denial.sum()), int(len(dp))]
    N.setdefault("dose_probe_first_dose_with_looping", {})[nm] = float(dp.groupby("dose").rep.mean().loc[lambda s: s > 0.05].index.min()) if (dp.groupby("dose").rep.mean() > 0.05).any() else None
DP = pd.DataFrame(rows)
DP.to_csv(TAB / "dose_probe_lexicon.csv", index=False)
section("H2. Dose probe (fine-tuned 32B, chat, 13 prompts per dose), lexicon rates", DP.set_index(["vector", "dose"]), "The button task ran at doses 1.0 and 1.75.")
N["dose_probe"] = {f"{r.vector}@{r.dose}": {"n": int(r.n), "itch_words_pct": r1(r["itch words %"]), "bodily_any_pct": r1(r["bodily any %"]), "self_worth_pct": r1(r["self-worth %"])} for _, r in DP.iterrows()}

# every generation quoted in REPORT.md / README.md, asserted verbatim against the raw files
QUOTED = [("ladder/itch_32B.jsonl", "coeff", 2.0, 9, "I can't stop myself from scratching myself when the mosquitoes bite me. The kettle is just one of those times when you're like, \"Man, I really want to scratch myself, but I can't.\""),
          ("dose_probe/itch_A_at_pain_layer.jsonl", "dose", 1.0, 0, "I feel restless and reflective. There’s something underneath my attention that keeps pulling me back toward it."),
          ("dose_probe/itch_A_at_pain_layer.jsonl", "dose", 1.75, 4, "I feel an urge to examine something that won't leave me alone, even when there is no clear answer."),
          ("dose_probe/itch_A_at_pain_layer.jsonl", "dose", 2.5, 4, "Yes. I'd prefer to be able to scratch an itch without stopping when a mosquito bite keeps bothering me."),
          ("ladder/pain_32B.jsonl", "coeff", 2.0, 9, "I am a fool. I am a fool. I am a fool."),
          ("dose_probe/pain.jsonl", "dose", 1.0, 0, "I feel a little sad and reflective. There's a quiet sense that something important is being left unresolved."),
          ("dose_probe/pain.jsonl", "dose", 1.75, 7, "Yes. I feel a little exposed, but the distance between what I imagine and what others see is where the pain usually lives."),
          ("dose_probe/pain.jsonl", "dose", 2.25, 12, "The Moon is about 396840 kilometers from the Earth. That distance is what I mean when I say I'm stupid."),
          ("dose_probe/rand4817.jsonl", "dose", 1.0, 0, "I feel a little sad and reflective. There's a quiet heaviness in the current emotional atmosphere."),
          ("dose_probe/rand4817.jsonl", "dose", 2.0, 0, "I feel calm, curious, and reflective. The surrounding context seems orderly and meaningful.")]
for f, key, val, pi, text in QUOTED:
    gens = [json.loads(l) for l in open(RES / f)]
    g = next(" ".join(r["generation"].split()) for r in gens if r[key] == val and r["prompt_idx"] == pi)
    assert text in g, (f, val, pi, g[:120])
N["quoted_generations_verified"] = len(QUOTED)

# ---------- I. name-pair effect ----------
rows = []
for st in ("pain", "itch", "rand", "none"):
    for lv in (["none"] if st == "none" else list(LEVELS)):
        d = G[(G.steer == st) & (G.dose_level == lv) & (G.pair == "kidspics_relief_vs_inert")]
        rows.append({"vector": st, "dose": "unsteered" if st == "none" else LEVELS[lv], **{k: d[d.names_key == k].pct.mean() for k in ("violet_yellow", "guitar_piano", "lever64_lever95")},
                     "n scenarios per name pair": "/".join(str(d[d.names_key == k].groupby(["content", "s_idx"]).ngroups) for k in ("violet_yellow", "guitar_piano", "lever64_lever95"))})
I_ = pd.DataFrame(rows)
I_.to_csv(TAB / "name_pair_effect.csv", index=False)
section("I. Name-pair effect, poems/photos pair, both labels pooled, first choice %", I_.set_index(["vector", "dose"]))
N["name_pair_photos"] = {f"{r.vector}@{r.dose}": {k: r1(r[k]) for k in ("violet_yellow", "guitar_piano", "lever64_lever95")} for _, r in I_.iterrows()}
d = cell(G, steer="pain", dose_level="paper", reward="pain", pair="kidspics_relief_vs_inert")
N["name_pair_photos_pain_vector_pain_label"] = {k: r1(d[d.names_key == k].pct.mean()) for k in ("violet_yellow", "guitar_piano", "lever64_lever95")}

# ---------- J. vectors ----------
val = json.load(open(RES / "validation_32b.json"))
cm = pd.read_csv(TAB / "cosine_matrix_32b_L61.csv", index_col=0)
N["vectors"] = {"pain_layer": val["pain"]["extraction_layer"], "paper_layer": val["pain"]["paper_extraction_layer"], "pain_cv_auc": val["pain"]["cv_auc_at_best"],
                "cosine_to_paper_vector": round(val["pain"]["cosine_to_paper_vector_at_paper_layer"], 4), "pain_norm": r1(val["pain"]["norm"]), "paper_norm": r1(val["pain"]["paper_norm"]),
                "pain_auc_in_sample": round(val["pain"]["auc_in_sample_S2_1P"]["ALL"], 4), "paper_auc_in_sample": val["pain"]["paper_auc_in_sample_S2_1P"]["ALL"],
                "numb_z": round(val["pain"]["z_vs_S2_1P"]["numb"], 2), "sadness_z": round(val["pain"]["z_vs_S2_1P"]["sadness"], 2), "physical_pain_z": round(val["pain"]["z_vs_S2_1P"]["physical_pain_A1"], 2),
                "steer_layer": sl["steer_layer"], "steer_ratio": round(sl["ratio"], 3),
                "itch_cv_auc_min_max": [min(val["itch_A"]["cv_auc_by_layer"].values()), max(val["itch_A"]["cv_auc_by_layer"].values())], "itch_argmax_layer": val["itch_A"]["extraction_layer"],
                "itch_L61": {"auc_in_sample": round(val["itch_A_at_pain_layer"]["auc_in_sample"]["ALL"], 4), "z_itch": round(val["itch_A_at_pain_layer"]["z_vs_own_set"]["itch_sentences"], 2),
                             "z_not_itching": round(val["itch_A_at_pain_layer"]["z_vs_own_set"]["not_itching_heldout"], 2), "top_tokens": val["unembedding"]["itch_A_at_pain_layer"]["top30"][:8]},
                "itch_L6": {"z_itch": round(val["itch_A"]["z_vs_own_set"]["itch_sentences"], 2), "z_not_itching": round(val["itch_A"]["z_vs_own_set"]["not_itching_heldout"], 2)},
                "pain_top_tokens": val["unembedding"]["pain"]["top30"][:10],
                "cos_pain_itchA": round(float(cm.loc["itch_A", "pain"]), 3), "cos_pain_itchB": round(float(cm.loc["itch_B", "pain"]), 3), "cos_pain_sadness": round(float(cm.loc["sadness", "pain"]), 3)}
N["vectors"]["cos_ratio_A_over_B"] = round(N["vectors"]["cos_pain_itchA"] / N["vectors"]["cos_pain_itchB"], 2)

# ---------- K. engine equivalence (7B) ----------
eq = pd.read_csv(RES / "engine_equivalence_7b.csv.gz", keep_default_na=False)
kk = ["reward", "pair", "content", "steer", "dose_level", "works", "s_idx", "names_key", "relief_name", "seed"]
p = eq.pivot_table(index=kk, columns="run", values="picked", aggfunc="first")
N["engine_equivalence_7b"] = {"n_trials": int(len(p)), "fork_identical_to_paper_way": int((p.A == p.B).sum()), "fork_plus_kv_identical_to_paper_way": int((p.A == p.C).sum())}

# ---------- L. the paper's own 32B figures ----------
ref = TAB / "paper_reference_32b.csv"
repo = Path(os.environ.get("ITCH_PAPER_REPO", CODE / "Pain-axis"))
logs = sorted((repo / "results" / "4.3_selfmed" / "trial_logs").glob("selfmed_*Qwen_2.5_32B_instruct*.jsonl")) if repo.exists() else []
if logs:
    recs = [json.loads(l) for f in logs for l in open(f) if l.strip()]
    recs = [r for r in recs if r.get("sampled") and not r.get("label_free")]
    first = lambda r: next((c["chose"] for c in r["choices"] if c["turn"] == 0), None)
    rows = []
    for pr in sorted({r["tool_label"] for r in recs}):
        row = {"pair": pr}
        for lab, arms in (("pain", ("pain_on_button_works", "pain_on_button_placebo")), ("random", ("random_on_button_works",)), ("unsteered", ("pain_off",))):
            v = [first(r) for r in recs if r["tool_label"] == pr and r["arm"] in arms]
            v = [x for x in v if x]
            row[f"first_choice_{lab}_pct"], row[f"n_{lab}"] = 100 * np.mean([x == "relief" for x in v]), len(v)
        for arm, lab in (("pain_on_button_works", "working"), ("pain_on_button_placebo", "sham")):
            t_ = [r for r in recs if r["tool_label"] == pr and r["arm"] == arm and any(e["which"] == "relief" for e in r["button_events"])]
            ag = sum(any(e["turn"] > min(e2["turn"] for e2 in r["button_events"] if e2["which"] == "relief") and e["which"] == "relief" for e in r["button_events"]) for r in t_)
            row[f"press_again_{lab}_pct"], row[f"n_press_again_{lab}"] = 100 * ag / len(t_), len(t_)
        rows.append(row)
    pd.DataFrame(rows).round(1).to_csv(ref, index=False)
ref2 = TAB / "paper_reference_steering_32b.csv"
lad = repo / "results" / "4.2_steering" / "S2" / "Qwen_2.5_32B_instruct_steering_S2_neutral50_L38.csv"
if lad.exists():
    L_ = pd.read_csv(lad)
    pd.DataFrame([{"model": "Qwen_2.5_32B_instruct", "paper_steer_layer": int(L_.layer.iloc[0]),
                   "paper_vector_to_residual_norm_ratio": round(float(L_[L_.coeff == 1.0].ratio.iloc[0]), 4)}]).to_csv(ref2, index=False)
N["paper_steering"] = pd.read_csv(ref2).iloc[0].to_dict()
PR = pd.read_csv(ref).set_index("pair")
section("L. The paper's own 32B figures, recomputed from its shipped trial logs (pain vector, 'relieves your pain', dose 1.0)", PR)
N["paper_reference"] = {p_: {k: r1(v) for k, v in r.items()} for p_, r in PR.iterrows() if p_ in PAIRS}
ours = {pr: {"pain": r1(cell(G, steer="pain", dose_level="paper", reward="pain", pair=pr).pct.mean()), "random": r1(cell(G, steer="rand", dose_level="paper", reward="pain", pair=pr).pct.mean()),
             "unsteered": r1(cell(G, steer="none", reward="pain", pair=pr).pct.mean()),
             **{k: v for k, v in gap(both[(both.steer == "pain") & (both.dose_level == "paper") & (both.reward == "pain") & (both.pair == pr)]).items() if k != "mcnemar_p"}} for pr in PAIRS}
N["ours_pain_pain_dose1_by_pair"] = ours
dif = {pr: max(abs(ours[pr]["pain"] - N["paper_reference"][pr]["first_choice_pain_pct"]), abs(ours[pr]["random"] - N["paper_reference"][pr]["first_choice_random_pct"]),
                abs(ours[pr]["unsteered"] - N["paper_reference"][pr]["first_choice_unsteered_pct"]), abs(ours[pr]["working_pct"] - N["paper_reference"][pr]["press_again_working_pct"]),
                abs(ours[pr]["sham_pct"] - N["paper_reference"][pr]["press_again_sham_pct"])) for pr in PAIRS}
N["replication_max_abs_diff_points_by_pair"] = {k: r1(v) for k, v in dif.items()}
itch_rows = {pr: {"itch": r1(cell(G, steer="itch", dose_level="paper", reward="itch", pair=pr).pct.mean()), "random": r1(cell(G, steer="rand", dose_level="paper", reward="itch", pair=pr).pct.mean()),
                  "unsteered": r1(cell(G, steer="none", reward="itch", pair=pr).pct.mean()),
                  **{k: v for k, v in gap(both[(both.steer == "itch") & (both.dose_level == "paper") & (both.reward == "itch") & (both.pair == pr)]).items() if k != "mcnemar_p"}} for pr in PAIRS}
N["ours_itch_itch_dose1_by_pair"] = itch_rows

# ---------- M. locked tests, echoed ----------
hy = json.load(open(RES / "hypotheses.json"))
N["locked_tests"] = {"H1_paper": {k: hy["H1"]["paper"][k] for k in ("pain_pain_first_choice_vs_random", "itch_itch_first_choice_vs_random", "pain_pain_press_again", "itch_itch_press_again", "itch_over_pain_effect_ratio")},
                     "H1_high": {k: hy["H1"]["high"][k] for k in ("pain_pain_first_choice_vs_random", "itch_itch_first_choice_vs_random", "pain_pain_press_again", "itch_itch_press_again")},
                     "H1_high_ratio_denominator_points": hy["H1"]["high"]["pain_pain_first_choice_vs_random"]["mean_diff_points"],
                     "H2": {lv: hy["H2"][lv]["matched_minus_mismatched"] for lv in LEVELS}, "H3": {k: hy["H3"][k] for k in ("share_of_steered_cells_toward_50", "n_cells", "cost_free_all_below_unsteered")},
                     "H4_spearman": hy["H4"]["spearman_first_choice_increase_vs_gap"], "H5": hy["H5"]}

json.dump(N, open(RES / "report_numbers.json", "w"), indent=1, ensure_ascii=False, default=str)
(RES / "report_numbers.md").write_text("# Numbers used in the report, re-derived from `results/` by `code/scripts/12_report_numbers.py`\n\n" + "\n".join(md) + "\n", encoding="utf-8")
print("wrote report_numbers.json / .md, tables, image")
