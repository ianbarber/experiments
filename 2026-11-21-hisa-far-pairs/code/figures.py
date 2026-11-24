"""Figures from ../results -> ../images.   python figures.py"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("../images", exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# one colour per condition, the same in every figure. Budgets are an ordered quantity: one hue, light to dark.
COL = {"dense": "#2a78d6",
       "hisa-45": "#f6b48f", "hisa-90": "#eb6834", "hisa-180": "#c24f1f", "hisa-360": "#8f3611", "hisa-720": "#5c2108",
       "hisa-90-mean": "#eda100", "ctrl-perm": "#1baf7a", "ctrl-iid-s0": "#6250d6", "ctrl-slow": "#e34948",
       "band-90-180": "#eb6834", "band-180-360": "#c24f1f", "band-360-720": "#8f3611", "band-720+": "#5c2108"}
LABEL = {"hisa-90": "hisa-90 (drop beyond 90°)", "hisa-90-mean": "hisa-90, dropped parts → their mean",
         "ctrl-perm": "control: same windows, pairs shuffled", "ctrl-iid-s0": "control: random pairs per distance",
         "ctrl-slow": "control: same count, slow end"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "font.family": "DejaVu Sans"})


def stage1():
    if not os.path.exists("../results/stage1_summary.json"):
        return
    S = json.load(open("../results/stage1_summary.json"))

    # ---- fig 1: cost of each condition by data type (log x: the controls are orders of magnitude apart)
    data = ["prose", "code", "tables"]
    rows = ["hisa-90", "hisa-90-mean", "ctrl-perm", "ctrl-iid-s0", "ctrl-slow"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    h = 0.15
    for ri, c in enumerate(rows):
        for di, d in enumerate(data):
            if c == "ctrl-perm":
                v = [S[d][f"ctrl-perm-s{s}"]["delta"][0] for s in range(3)]
                m, lo, hi = np.mean(v), min(v), max(v)
            else:
                m, lo, hi = S[d][c]["delta"]
            y = di + (ri - 2) * h
            ax.barh(y, m, height=h * 0.86, color=COL[c], label=LABEL[c] if di == 0 else None)
            ax.plot([lo, hi], [y, y], color=INK, lw=1)
            ax.text(hi * 1.08, y, f"{m:+.2f}" if m >= 0.1 else f"{m:+.3f}", va="center", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_yticks(range(len(data)), [f"{d}\n(dense {S[d]['dense']['loss']:.2f})" for d in data])
    ax.invert_yaxis()
    ax.set_xlabel("extra loss against dense, nats per token (log scale)")
    ax.set_xlim(0.01, 30)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    ax.set_title("Dropping score parts from pretrained SmolLM-135M, no retraining", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig("../images/stage1_cost_by_data.png", dpi=160)
    plt.close(fig)

    # ---- fig 2: where in the window the cost appears (prose), whole rule against single-pair cuts
    L0 = np.load("../raw/stage1/prose__dense.npz")["loss"] if os.path.exists("../raw/stage1/prose__dense.npz") else None
    prof_path = "../results/stage1_position_profile.json"
    if L0 is not None:
        prof = {"bin": 64, "conditions": {}}
        for c in ("hisa-90", "pair-22-90", "pair-23-90", "pair-24-90", "hisa-180", "hisa-90-keep1"):
            d = (np.load(f"../raw/stage1/prose__{c}.npz")["loss"] - L0).mean(0)
            prof["conditions"][c] = [float(d[a:a + 64].mean()) for a in range(0, 2047, 64)]
        json.dump(prof, open(prof_path, "w"), indent=1)
    if os.path.exists(prof_path):
        prof = json.load(open(prof_path))
        x = np.arange(32) * 64 + 32
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for c, col, lab, lw in (("pair-23-90", "#e87ba4", "only pair 23 cut beyond 90°", 2),
                                ("hisa-90", COL["hisa-90"], "hisa-90 (all pairs cut beyond 90°)", 2.6),
                                ("pair-24-90", "#008300", "only pair 24 cut", 2),
                                ("pair-22-90", MUTED, "only pair 22 cut", 1.4)):
            y = prof["conditions"][c]
            ax.plot(x, y, color=col, lw=lw)
            ax.text(2060, y[-1], lab, color=INK, fontsize=8, va="center")
            ax.plot([2040], [y[-1]], marker="o", ms=5, color=col, clip_on=False)
        for w, n in ((884, "22"), (1178, "23"), (1571, "24")):
            ax.axvline(w, color=MUTED, lw=0.8, ls=":")
            ax.text(w, 1.56, f"window of\npair {n}", fontsize=7.5, color=MUTED, ha="center", va="bottom")
        ax.set_xlim(0, 2048)
        ax.set_ylim(-0.03, 1.55)
        ax.set_xlabel("position in the 2048-token window (prose, 256 windows, 64-token bins)")
        ax.set_ylabel("extra loss against dense\n(nats per token)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        fig.subplots_adjust(right=0.70, top=0.86, bottom=0.15, left=0.11)
        fig.savefig("../images/stage1_cost_by_position.png", dpi=160)
        plt.close(fig)

    # ---- fig 3: grids, newline probability at the true position
    Cs = [C for C in (8, 16, 32, 64, 128, 256) if f"grid-{C}" in S]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for c in ["dense", "hisa-90", "hisa-90-mean", "hisa-360", "ctrl-perm"]:
        key = "ctrl-perm-s0" if c == "ctrl-perm" else c
        if key not in S[f"grid-{Cs[0]}"]:
            continue
        y = [S[f"grid-{C}"][key]["p_true"] for C in Cs]
        ax.plot(range(len(Cs)), y, marker="o", ms=6, lw=2, color=COL[c], label=LABEL.get(c, c),
                zorder=3 if c == "dense" else 2)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.text(len(Cs) - 1, 0.515, "gate", ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(Cs)), Cs)
    ax.set_xlabel("cells per row, C (the newline is C+1 tokens after the previous one)")
    ax.set_ylabel("P(newline) at the true position")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("../images/stage1_grids.png", dpi=160)
    plt.close(fig)


def stage2():
    finals = {}
    for p in sorted(glob.glob("../results/stage2/*.jsonl")):
        if "pilot" in p or "long" in p:
            continue
        last = json.loads(open(p).read().strip().split("\n")[-1])
        if last.get("final"):
            finals.setdefault(last["variant"], []).append(last["eval_fp32_128seq"])
    if not finals:
        return
    DS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    order = [v for v in ["dense", "hisa-360", "hisa-180", "hisa-90", "hisa-45"] if v in finals]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for v in order:
        for kind, ls, mk in (("offset", "-", "o"), ("content", "--", None)):
            runs = np.array([[r.get(f"{kind}_{D}", np.nan) for D in DS] for r in finals[v]])
            m = np.array([np.mean(c[~np.isnan(c)]) if (~np.isnan(c)).any() else np.nan for c in runs.T])
            ax.plot(range(len(DS)), m, ls=ls, marker=mk, ms=6, lw=2 if kind == "offset" else 1.2, color=COL[v],
                    alpha=1 if kind == "offset" else 0.7,
                    label=(f"{v}, 4000 steps" if v != "dense" else "dense, 4000 steps")
                    + (f" (mean of {len(runs)} seeds)" if len(runs) > 1 else "") if kind == "offset" else None)
            if kind == "offset" and len(runs) > 1:
                for r in runs:
                    ax.plot(range(len(DS)), r, ls="none", marker="o", ms=3, color=COL[v], alpha=0.5)
    for p in sorted(glob.glob("../results/stage2/*long16k.jsonl")):          # Amendment 3: one 16,000-step run
        last = json.loads(open(p).read().strip().split("\n")[-1])
        if last.get("final"):
            y = [last["eval_fp32_128seq"][f"offset_{D}"] for D in DS]
            ax.plot(range(len(DS)), y, ls="-", marker="s", ms=7, mfc="white", mew=2, lw=2, color=COL["hisa-90"],
                    label="hisa-90, 16,000 steps (1 seed)", zorder=4)
    ax.plot([], [], color=MUTED, lw=1.2, ls="--", label="dashed: content recall (1.00 for all)")
    ax.axhline(1 / 256, color=MUTED, lw=0.8, ls=":")
    ax.text(0, 0.02, "chance", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(DS)), DS)
    ax.set_xlabel("distance D to the token to recall")
    ax.set_ylabel("accuracy on answer tokens")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), title="solid: offset recall", title_fontsize=8)
    ax.set_title("Fine-tuned on exact-offset recall: accuracy by distance", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig("../images/stage2_accuracy_by_distance.png", dpi=160)
    plt.close(fig)


def stage2_dense_masked():
    """Amendment 3: the fine-tuned dense model with one rotation band removed at inference."""
    path = "../results/stage2_dense_masked.json"
    if not os.path.exists(path):
        return
    R = json.load(open(path))
    DS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    lab = {"dense": "nothing removed", "band-90-180": "90–180° removed", "band-180-360": "180–360° removed",
           "band-360-720": "360–720° removed", "band-720+": "beyond 720° removed"}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for c in lab:
        for kind, ls, mk in (("offset", "-", "o"), ("content", "--", None)):
            runs = np.array([[r[c].get(f"{kind}_{D}", [np.nan])[0] for D in DS] for r in R.values()])
            m = runs.mean(0)
            ax.plot(range(len(DS)), m, ls=ls, marker=mk, ms=6, lw=2 if kind == "offset" else 1.2, color=COL[c],
                    alpha=1 if kind == "offset" else 0.7, zorder=3 if c == "band-720+" else 2,
                    label=lab[c] if kind == "offset" else None)
            if kind == "offset" and c != "dense":
                for r in runs:
                    ax.plot(range(len(DS)), r, ls="none", marker="o", ms=3, color=COL[c], alpha=0.5)
    ax.plot([], [], color=MUTED, lw=1.2, ls="--", label="dashed: content recall")
    ax.set_xticks(range(len(DS)), DS)
    ax.set_xlabel("distance D to the token to recall")
    ax.set_ylabel("accuracy on answer tokens")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="solid: offset recall\n(mean of 2 seeds)", title_fontsize=8)
    ax.set_title("Dense model fine-tuned on exact-offset recall, one rotation band removed, no retraining",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig("../images/stage2_dense_under_bands.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    stage1()
    stage2()
    stage2_dense_masked()
