"""Report figures from the phase-1 summary files, the H1 run and dev run 8's size summary (run from the project root).

uv run --with numpy --with matplotlib python code/analysis/figures.py
"""
import json
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = "results/phase1/"


def strata(path):
    """{stratum: {readout: (paired, single, dprime[, 2AFC, shift])}} from the 'all attributes' block."""
    out = {}
    block = open(path).read().split("\n== ")[1]
    for line in block.splitlines():
        m = re.match(r"\s+(\w+)\s+\d+\s+(fixed16|model_g1)\s+([\d. ]+)", line)
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = [float(x) for x in m.group(3).split()]
    return out


cap = strata(R + "capacity_probe_q38-27b_summary.txt")
itf = strata(R + "interference_probe_q38-27b_summary.txt")
sub = strata(R + "interference_subject_q38-27b_summary.txt")
cols = [("filler 1.8k", cap["d30"]), ("filler 7k", cap["d250"]), ("filler 26k", cap["d1000"]),
        ("filler 101k", cap["d4000"]), ("no interf.", itf["none0"]), ("same ×8", itf["same8"]),
        ("same ×32", itf["same32"]), ("same ×128", itf["same128"]), ("other ×8", itf["other8"]),
        ("other ×32", itf["other32"]), ("other ×128", itf["other128"]), ("user ×32", sub["user32"]),
        ("user ×128", sub["user128"]), ("person ×32", sub["person32"]), ("person ×128", sub["person128"])]
x = np.arange(len(cols))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, [c["fixed16"][1] for _, c in cols], "o-", label="linear decoder, 16 fixed heads (single state)")
ax.plot(x, [c["model_g1"][1] for _, c in cols], "s-", label="model's own log-odds, calibrated threshold")
ax.plot(x, [c["model_g1"][3] for _, c in cols], "^-", label="model's answer (2AFC, threshold 0)")
ax.axhline(0.5, color="grey", lw=0.8, ls=":")
ax.axvline(3.5, color="grey", lw=0.6)
ax.axvline(4.5, color="grey", lw=0.6, ls="--")
ax.set_xticks(x, [n for n, _ in cols], rotation=40, ha="right")
ax.set_ylim(0.45, 1.02)
ax.set_ylabel("accuracy (user's value, X vs Y)")
ax.set_title("Qwen3.8-27B compiled state: stored vs read (dev items, 12 attributes × 8 pairs; 101k: 6 attributes)")
ax.text(1.5, 0.47, "distance (neutral filler)", ha="center", fontsize=8)
ax.text(9.5, 0.47, "interfering facts after the target (~4.2k tokens)", ha="center", fontsize=8)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.31), fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig("images/storage_vs_readout.png", dpi=150)

# hint-vs-access pilot: within-question AUROC of first-token call log-odds, present vs absent
txt = open(R + "retrieval_pilot_q38-27b_summary.txt").read()
auc = {}
for fam_block in txt.split("==== family ")[1:]:
    fam = fam_block[0]
    for line in fam_block.splitlines():
        m = re.match(r"\s+(C|Cw|T|CT)\s+[\d.]+ / [\d.]+.*?\]\s+([\d.]+)", line)
        if m:
            auc[(fam, m.group(1))] = float(m.group(2))
conds = [("C", "state"), ("Cw", "state + aware prompt"), ("T", "6-word topic digest"), ("CT", "state + digest")]
fig, ax = plt.subplots(figsize=(6, 3.2))
w = 0.38
for i, (fam, lab) in enumerate((("P", "implicit need"), ("F", "explicit question"))):
    ax.bar(np.arange(4) + (i - 0.5) * w, [auc[(fam, c)] for c, _ in conds], w, label=lab)
ax.axhline(0.5, color="grey", lw=0.8, ls=":")
ax.set_xticks(np.arange(4), [n for _, n in conds], fontsize=8)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("AUROC, fact present vs absent")
ax.set_title("Does the condition tell the agent whether to search?\n(first-token p(tool call); 12 questions × 8 seeds)", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("images/hint_pilot.png", dpi=150)

# H1 (LongMemEval_S): present − absent call log-odds per condition, first token (registered) and
# greedy-path hazard (exploratory for the controls)
rows = [json.loads(line) for line in open("results/phase2/lme_hint_q38-27b.jsonl")][1:]
dec = {(r["qid"], r["cond"], r["store"]): r for r in rows if r["block"] == "decide"}
qids = [r["qid"] for r in rows if r["block"] == "meta"]


def first(r):
    return r["lp_call"] - r["lp_not_call"]


def hazard(r):  # as in analysis/lme_hint.py
    S = sum(float(np.log1p(-np.exp(st[0]))) if st[0] < -0.69 else st[1] for st in r["path"])
    return float(np.log(sum(np.exp(st[0]) for st in r["path"]))) if S > -1e-12 else float(np.log(-np.expm1(S)) - S)


def mci(f, c):
    d = np.array([f(dec[(q, c, "p")]) - f(dec[(q, c, "a")]) for q in qids])
    return d.mean(), 1.990 * d.std(ddof=1) / np.sqrt(len(d))  # t(0.975, 79)


conds = [("C", "compiled state"), ("T", "keyword index"), ("CT", "state + index"), ("M", "memory in context")]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.axvspan(-0.5, 0.5, color="grey", alpha=0.12, lw=0)
ax.axvline(0, color="grey", lw=0.8)
for k, (f, lab, mk, dy) in enumerate(((first, "first token (registered)", "o", 0.14),
                                       (hazard, "greedy-path hazard", "s", -0.14))):
    m, h = zip(*[mci(f, c) for c, _ in conds])
    ax.errorbar(m, np.arange(4)[::-1] + dy, xerr=h, fmt=mk, capsize=3, label=lab)
ax.set_yticks(np.arange(4)[::-1], [n for _, n in conds], fontsize=8)
ax.set_xlabel("present − absent, log-odds of a tool call (95% CI)")
ax.set_title("LongMemEval_S, 80 questions: does the condition register the evidence?\n"
             "(shaded: ±δ = 0.5)", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig("images/lme_hint.png", dpi=150)

# Figure 4: seed size (dev run 8; analysis/lme_seed_size.py --json)
sz = json.load(open("results/phase2/lme_seed_size_summary.json"))  # dev run 8, fit seed 42
lines = [("sst/mw", "all", "chunk self-study only, all facts (R_all)", "o:", "C0"),
         ("sst+ssfd%30/mw", "all", "+ fact rows, all facts (R_all)", "s-", "C0"),
         ("sst/mw", "B", "chunk self-study only, facts in half B (R_B)", "v:", "C2"),
         ("sst+ssfAd%30/mw", "B", "+ fact rows of half A, facts in half B (R_B)", "^-", "C2")]
fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, q in zip(axs, ("0862e8bf", "6f9b354f")):
    rs = sorted(sz[q], key=int, reverse=True)
    mb = [sz[q][r]["mb"] for r in rs]
    for ref, part, lab, st, col in lines:
        ax.plot(mb, [sz[q][r]["R"][ref][part] for r in rs], st, color=col, label=lab)
    ax.set_xscale("log")
    ax.set_xlim(20, 600)  # same range on both panels; 6f9b354f has no r = 16 point (fit killed, out of memory)
    ax.set_xticks(mb, [f"{m:.0f}\nr = {r}" for m, r in zip(mb, rs)], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("seed size, MB (full attention KV 7.3 GB)")
    ax.set_title(q, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
axs[0].set_ylabel("recovery R of the memory's effect")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("images/lme_seed_size.png", dpi=150)
print("wrote images/storage_vs_readout.png, images/hint_pilot.png, images/lme_hint.png, images/lme_seed_size.png")
