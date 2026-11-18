"""Figures for the entry. Reads ../results/*.json, writes ../images/*.png.

  python scripts/figures.py sketches   # the two kickoff sketches, from made-up numbers
  python scripts/figures.py all        # the measured figures
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent
RESULTS = CODE.parent / "results"
IMAGES = CODE.parent / "images"

# one colour per condition, the same in every figure (validated for colour-vision deficiency)
COLOR = {"dt": "#2a78d6", "fft": "#eb6834", "lora": "#1baf7a", "base": "#0b0b0b"}
LABEL = {"dt": "Deletion Tuning", "fft": "Full fine-tune", "lora": "LoRA", "base": "Base"}
MARK = {"dt": "o", "fft": "s", "lora": "^"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1, "axes.axisbelow": True, "legend.frameon": False,
})


def style_points(ax, x, y, method, size=9, **kw):
    ax.plot(x, y, color=COLOR[method], lw=2, marker=MARK[method], ms=size, mec=SURFACE, mew=2,
            solid_capstyle="round", solid_joinstyle="round", **kw)


def base_band(ax):
    ax.axhspan(-1, 1, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=1, zorder=1)


def sketches():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    x = [0, 22, 40, 55, 68]
    cases = [
        ("If deletion forgets less", {"fft": [0, -1.0, -2.1, -3.0, -4.2], "lora": [0, -1.3, -2.6, -3.9, -5.2], "dt": [0, -0.1, 0.1, -0.3, -0.2]}),
        ("If forgetting follows learning, not method", {"fft": [0, -1.0, -2.1, -3.0, -4.2], "lora": [0, -1.3, -2.4, -3.5, -4.7], "dt": [0, -0.9, -2.3, -3.3, -4.4]}),
    ]
    for ax, (title, series) in zip(axes, cases):
        base_band(ax)
        for m, y in series.items():
            style_points(ax, x, y, m, label=LABEL[m])
        ax.plot([0], [0], marker="o", ms=9, color=COLOR["base"], mec=SURFACE, mew=2, lw=0, label="Base")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("New-task score (made-up)")
        ax.set_ylim(-6, 1.6)
    axes[0].set_ylabel("Change in prior-task suite score, points")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Kickoff sketches, made-up numbers: what each outcome would look like", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, "Grey band: within one point of base.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    IMAGES.mkdir(exist_ok=True)
    fig.savefig(IMAGES / "sketches.png", dpi=160)


def _lr(config):
    return float(config.split("lr=")[1].split()[0])


def _points(final, diag):
    """Every checkpoint with a suite score, per method: learning rate, regime, per-seed (wins, suite change)."""
    out = {m: {} for m in COLOR if m != "base"}
    for regime in ("paper_rates", "tuned_matched"):
        for m, d in final["cells"].get(regime, {}).get("methods", {}).items():
            rows = [r for r in d["per_seed"] if "d_prior" in r]
            if rows:
                out[m][_lr(d["config"])] = {"regime": regime, "rows": [(r["seen"], r["d_prior"], r["d_mc"]) for r in rows]}
    for m, rows in (diag or {}).items():
        for r in rows:
            if r["lr"] not in out[m]:
                out[m][r["lr"]] = {"regime": "other", "rows": [(r["seen"], r["d_prior"], r["d_mc"])]}
    return out


REGIME_NOTE = {"paper_rates": "paper", "tuned_matched": "tuned", "other": ""}
# where each method's name sits on the learning-rate figure: (learning rate, y, horizontal alignment)
NAME_AT = {"fft": (2.2e-6, -4.6, "left"), "lora": (1.0e-4, -3.0, "center"), "dt": (2.1e-3, -4.6, "left")}


def by_learning_rate(final, diag):
    """Suite change against learning rate, the three methods on one axis."""
    pts = _points(final, diag)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    base_band(ax)
    wins = []
    for m in ("fft", "lora", "dt"):
        lrs = sorted(pts[m])
        if not lrs:
            continue
        means = [sum(r[1] for r in pts[m][lr]["rows"]) / len(pts[m][lr]["rows"]) for lr in lrs]
        ax.plot(lrs, means, color=COLOR[m], lw=2, solid_capstyle="round", zorder=3)
        for lr, mean in zip(lrs, means):
            p = pts[m][lr]
            n = len(p["rows"])
            wins.append(sum(r[0] for r in p["rows"]) / n)
            if n > 1:
                spread = [0.93, 1.0, 1.075][:n]
                ax.scatter([lr * k for k in spread], [r[1] for r in p["rows"]], s=15, color=COLOR[m], alpha=0.6, lw=0, zorder=4)
            ax.plot([lr], [mean], marker=MARK[m], ms=10, color=COLOR[m] if n > 1 else SURFACE, mec=SURFACE if n > 1 else COLOR[m],
                    mew=2, lw=0, zorder=5)
            note = REGIME_NOTE[p["regime"]]
            low = mean < -6
            value = f"{mean:+.1f}".replace("-0.0", "0.0").replace("+0.0", "0.0")
            ax.annotate(value + (f"\n{note}" if note else ""), (lr, mean), textcoords="offset points",
                        xytext=(-12, 0) if low else (0, 12), ha="right" if low else "center", va="center" if low else "bottom",
                        fontsize=8, color=INK, linespacing=1.15, zorder=6)
        x, y, ha = NAME_AT[m]
        ax.text(x, y, LABEL[m], ha=ha, va="center", fontsize=10, color=INK, fontweight="bold", zorder=6)
    ax.set_xscale("log")
    ax.set_xlim(1.6e-6, 1.1e-2)
    ax.set_ylim(-20, 5.5)
    ax.minorticks_off()
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Change in prior-task suite score, points")
    ax.set_title("Change in the six-benchmark suite score against learning rate", loc="left", fontsize=12)
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARK[m], ms=8, mec=SURFACE, mew=1.5, lw=2, label=LABEL[m]) for m in ("fft", "lora", "dt")]
    ax.legend(handles=handles, loc="lower left", fontsize=9, bbox_to_anchor=(0.0, 0.0))
    fig.text(0.012, 0.012, f"\"paper\": the paper's learning rate for that method, scored at step 1,600. \"tuned\": the rate chosen on Wordle score alone.\n"
             f"Every point wins {min(wins):.0f} to {max(wins):.0f}% of seen-list Wordle games. Filled: mean of 3 seeds, small dots the seeds. Open: one seed.\n"
             "Grey band: within one point of base.", fontsize=7.8, color=INK2, linespacing=1.3)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(IMAGES / "forgetting_by_lr.png", dpi=160)
    plt.close(fig)


def score_vs_forgetting(final, diag):
    """The kickoff sketch's axes with the measurements: suite change against Wordle score, every scored checkpoint."""
    pts = _points(final, diag)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    base_band(ax)
    ax.plot([0], [0], marker="o", ms=9, color=COLOR["base"], mec=SURFACE, mew=2, lw=0, zorder=5)
    ax.annotate("Base", (0, 0), textcoords="offset points", xytext=(8, 8), fontsize=9, color=INK)
    for m in ("fft", "lora", "dt"):
        for lr, p in pts[m].items():
            filled = p["regime"] == "paper_rates"
            for seen, d, _ in p["rows"]:
                ax.plot([seen], [d], marker=MARK[m], ms=8.5 if p["regime"] != "other" else 6.5, color=COLOR[m] if filled else SURFACE,
                        mec=SURFACE if filled else COLOR[m], mew=1.6, lw=0, zorder=4 if filled else 3)
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARK[m], ms=8, mec=SURFACE, mew=1.5, lw=0, label=LABEL[m]) for m in ("fft", "lora", "dt")]
    handles += [plt.Line2D([], [], color=INK2, marker="o", ms=7.5, mec=SURFACE, mew=1.5, lw=0, label="paper's learning rate"),
                plt.Line2D([], [], color=SURFACE, marker="o", ms=7.5, mec=INK2, mew=1.5, lw=0, label="higher learning rate")]
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    ax.set_xlabel("Wordle win rate on the 156 seen-list words, %")
    ax.set_ylabel("Change in prior-task suite score, points")
    ax.set_title("What was forgotten, against how much Wordle was learned", loc="left", fontsize=12)
    ax.set_xlim(-4, 100)
    ax.set_ylim(-20, 5.5)
    fig.text(0.012, 0.012, "One marker per scored checkpoint (one training run each). Grey band: within one point of base.", fontsize=7.8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(IMAGES / "score_vs_forgetting.png", dpi=160)
    plt.close(fig)


def paper_rates(final):
    """The follow-up endpoint: each method at the paper's learning rate, one dot per seed."""
    c = final["cells"].get("paper_rates")
    if not c:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True)
    for ax, (key, title) in zip(axes, (("d_prior", "Six-benchmark suite mean"), ("d_mc", "Four multiple-choice benchmarks"))):
        base_band(ax)
        for i, m in enumerate(("fft", "lora", "dt")):
            d = c["methods"].get(m, {}).get(key)
            if not d:
                continue
            xs = [i + k for k in (-0.16, 0.0, 0.16)][: d["n"]]
            ax.scatter(xs, d["per_seed"], s=62, color=COLOR[m], marker=MARK[m], edgecolor=SURFACE, lw=1.5, zorder=4)
            ax.plot([i - 0.3, i + 0.3], [d["mean"]] * 2, color=INK, lw=1.6, zorder=5)
            ax.annotate(f"{d['mean']:+.1f}".replace("-0.0", "0.0").replace("+0.0", "0.0"), (i + 0.32, d["mean"]), fontsize=8.5, color=INK, va="center")
        ax.set_xticks(range(3))
        ax.set_xticklabels([LABEL[m] for m in ("fft", "lora", "dt")], fontsize=9)
        ax.set_xlim(-0.6, 2.75)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].set_ylabel("Change from base, points")
    axes[0].set_ylim(-2.6, 2.6)
    fig.suptitle("Every method at the paper's learning rate, scored at step 1,600: one marker per seed", x=0.01, ha="left", fontsize=11.5)
    fig.text(0.012, 0.012, "Black line: mean of the three seeds. Grey band: within one point of base.", fontsize=7.8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(IMAGES / "paper_rates.png", dpi=160)
    plt.close(fig)


def learning_curves(final):
    """Seen-list win rate along training at the paper's learning rates: the three methods on one axis, three seeds each."""
    tr = final.get("trajectories", {})
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    steps = [0, 400, 800, 1200, 1600]
    label_y = {}
    for m in ("lora", "fft", "dt"):
        if m not in tr:
            continue
        for r in tr[m]["runs"]:
            ax.plot(steps, [0.0] + [r["seen"][str(k)] for k in steps[1:]], color=COLOR[m], lw=1, alpha=0.4, zorder=2)
        mean = [0.0] + [tr[m]["mean"][str(k)] for k in steps[1:]]
        style_points(ax, steps, mean, m, zorder=4)
        label_y[m] = mean[-1]
    # direct labels at the right edge, nudged apart where two means end within a line height of each other
    order = sorted(label_y, key=label_y.get)
    at = {m: label_y[m] for m in order}
    for lo, hi in zip(order, order[1:]):
        if at[hi] - at[lo] < 6:
            mid = (at[hi] + at[lo]) / 2
            at[lo], at[hi] = mid - 3, mid + 3
    for m in order:
        ax.annotate(f"{LABEL[m]}  {label_y[m]:.0f}%", (1600, label_y[m]), xytext=(1640, at[m]), textcoords="data", fontsize=9.5, color=INK, va="center")
    ax.set_xticks(steps)
    ax.set_xlim(-40, 2130)
    ax.set_ylim(-3, 100)
    ax.set_xlabel("Training step (batch of 4 games)")
    ax.set_ylabel("Win rate on the 156 seen-list words, %")
    ax.set_title("Learning the task at the paper's learning rates", loc="left", fontsize=12)
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARK[m], ms=8, mec=SURFACE, mew=1.5, lw=2, label=LABEL[m]) for m in ("fft", "lora", "dt")]
    ax.legend(handles=handles, loc="lower right", fontsize=9, bbox_to_anchor=(0.79, 0.0))
    fig.text(0.012, 0.012, "Heavy line: mean of three seeds. Thin lines: the seeds, 156 greedy games per point. The base model wins none.", fontsize=7.8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(IMAGES / "learning_curves.png", dpi=160)
    plt.close(fig)


def mask_by_start(anat):
    """Share of gate weights deleted in the deployed mask, against where each weight's logit started."""
    by = anat["by_start"]
    edges = by["edges"]
    mid = [(a + b) / 2 for a, b in zip(edges, edges[1:])]

    def curve(names):
        ys = []
        for i in range(len(mid)):
            w = [by["runs"][n]["weights"][i] for n in names]
            if min(w) < 1000:
                ys.append(None)
            else:
                ys.append(sum(100 * by["runs"][n]["deleted"][i] / by["runs"][n]["weights"][i] for n in names) / len(names))
        return ys

    seeds = (42, 43, 44)
    lines = [  # label, runs, colour strength, line width
        ("lr 3e-4, one seed", ["dt_lr3e-4_a100-1000_s42"], 0.45, 1.6),
        ("lr 5e-4, the paper's rate, three seeds", [f"dt_lr5e-4_a100-1000_s{k}" for k in seeds], 1.0, 2.6),
        ("lr 1e-3, tuned on Wordle, three seeds", [f"dt_lr1e-3_a100-1000_s{k}" for k in seeds], 0.7, 1.6),
        ("lr 3e-3, past the cliff, one seed", ["dt_lr3e-3_a100-1000_s42"], 0.3, 1.6),
    ]
    fig, (top, ax) = plt.subplots(2, 1, figsize=(7.8, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 3.6], "hspace": 0.08})
    ref = by["runs"]["dt_lr5e-4_a100-1000_s42"]["weights"]
    share = [100 * w / sum(ref) for w in ref]
    top.bar(mid, share, width=0.0042, color=INK2, alpha=0.55, lw=0)
    top.set_ylabel("% of gate\nweights", fontsize=8.5)
    top.set_yticks([0, 10, 20])
    top.grid(axis="x", visible=False)
    top.set_title("Which weights get deleted depends on where their logit started", loc="left", fontsize=12)
    top.text(0.058, 16.5, "where the logits start", fontsize=8.5, color=INK2, ha="right")

    ax.axvspan(-0.03, 0, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.plot([-0.03, 0, 0, 0.06], [100, 100, 0, 0], color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2, label="before training: the starting mask")
    for label, names, strength, lw in lines:
        ys = curve(names)
        pts = [(x, y) for x, y in zip(mid, ys) if y is not None and -0.03 <= x <= 0.06]
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=_mix(COLOR["dt"], strength), lw=lw, marker="o", ms=4.5 if lw > 2 else 3.2,
                mec=SURFACE, mew=0.8, zorder=4 if lw > 2 else 3, label=label, solid_capstyle="round")
    ax.text(-0.0285, 6, "deleted at the start", fontsize=8.5, color=INK2)
    ax.text(0.0585, 93, "kept at the start", fontsize=8.5, color=INK2, ha="right")
    ax.set_xlim(-0.03, 0.06)
    ax.set_ylim(-2, 103)
    ax.set_xlabel("Starting value of the weight's mask logit (deleted if it ends at or below zero)")
    ax.set_ylabel("Deleted in the deployed mask,\n% of the weights that start there")
    ax.legend(loc="center right", fontsize=8.5, bbox_to_anchor=(1.0, 0.56))
    fig.text(0.012, 0.012, "Bins of 0.005; 377 million gate weights. Kept weights start at 0.03 with noise of 0.01; the three seeds draw different noise and lie on one curve.",
             fontsize=7.6, color=INK2)
    fig.subplots_adjust(left=0.115, right=0.965, top=0.94, bottom=0.115)
    fig.savefig(IMAGES / "mask_by_start.png", dpi=160)
    plt.close(fig)


def _mix(hex_color, strength):
    """Blend a colour toward the surface: strength 1 is the colour itself."""
    c = [int(hex_color[k : k + 2], 16) for k in (1, 3, 5)]
    b = [int(SURFACE[k : k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(b[k] + strength * (c[k] - b[k])):02x}" for k in range(3))


def _lr_label(r):
    a = r["config"]
    lr = a.split("lr=")[1].split()[0]
    lr = f"{float(lr):.0e}".replace("e-0", "e-")
    extra = " (rank 1)" if "r=1 " in a else " (ramp 1e3 to 1e4)" if "alpha=1000,10000" in a else ""
    return f"lr {lr}{extra}"


def tuning(grid):
    """Seen-list win rate along training for every tuning run, one panel per method.
    Within a panel the learning rate is ordered light (low) to dark (high); the selected run is heavy."""
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.0), sharey=True)
    for ax, m in zip(axes, ("fft", "lora", "dt")):
        rows = sorted((r for r in grid["runs"] if r["method"] == m), key=lambda r: (float(r["config"].split("lr=")[1].split()[0]), r["config"]))
        chosen = grid["selected"].get(m, {}).get("name")
        for k, r in enumerate(rows):
            pts = [(0, 0.0)] + [tuple(p) for p in r["trajectory"]]
            sel = r["name"] == chosen
            col = _mix(COLOR[m], 0.35 + 0.65 * k / max(1, len(rows) - 1))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2.6 if sel else 1.4, marker=MARK[m], ms=7.5 if sel else 4.5,
                    mec=SURFACE, mew=1.2, label=_lr_label(r) + (" - selected" if sel else ""), zorder=4 if sel else 3)
        if "match_level" in grid:
            ax.axhline(grid["match_level"], color=INK2, lw=1, zorder=2)
        if chosen:
            step = grid["selected"][m]["matched_step"]
            ax.annotate(f"matched\nat step {step}", (step, dict(map(tuple, grid["selected"][m]["trajectory"]))[step]), textcoords="offset points",
                        xytext=(8, -26), fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.set_title(LABEL[m], loc="left", fontsize=11)
        ax.set_xlabel("Training step (batch of 4 games)")
        ax.set_xticks([0, 400, 800, 1200, 1600])
        ax.set_ylim(-3, 100)
        ax.legend(loc="center right" if m == "dt" else "lower right", fontsize=7.4, handlelength=1.6, labelspacing=0.25, borderaxespad=0.2)
    axes[0].set_ylabel("Win rate on the 156 seen-list words, %")
    fig.suptitle("Tuning on seed 42: every configuration tried. Horizontal line: the matching level, 79.5%", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(IMAGES / "tuning.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    if sys.argv[1] == "sketches":
        sketches()
    elif sys.argv[1] == "mask":
        mask_by_start(json.loads((RESULTS / "mask_anatomy.json").read_text()))
    elif sys.argv[1] == "tuning":
        tuning(json.loads((RESULTS / "grid.json").read_text()))
    else:
        final = json.loads((RESULTS / "final.json").read_text())
        diag = json.loads((RESULTS / "lr_diagnostic.json").read_text()) if (RESULTS / "lr_diagnostic.json").exists() else {}
        sketches()
        tuning(json.loads((RESULTS / "grid.json").read_text()))
        by_learning_rate(final, diag)
        score_vs_forgetting(final, diag)
        paper_rates(final)
        learning_curves(final)
        if (RESULTS / "mask_anatomy.json").exists():
            mask_by_start(json.loads((RESULTS / "mask_anatomy.json").read_text()))
