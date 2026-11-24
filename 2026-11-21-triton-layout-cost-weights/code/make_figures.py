#!/usr/bin/env python
"""Figures for the entry, from the result summaries.

    python make_figures.py --results ../results --out ../images
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One colour per element width in every figure of the entry.
WIDTH_COLOR = {32: "#2a78d6", 16: "#eb6834", 8: "#1baf7a"}
WIDTH_LABEL = {32: "32-bit elements", 16: "16-bit", 8: "8-bit"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ACCENT, NEUTRAL = "#2a78d6", "#898781"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_assumed_vs_measured(e3, out):
    """What the pass charges for a conversion against what it costs, per lowering class."""
    rows = [r for r in e3["rows"] if r["route"] == "default" and r["bits"] in WIDTH_COLOR and r["bytes"] >= 1024
            and r["class"] in ("register_permute", "warp_shuffle", "shared_memory") and r["dtype"] != "f64"]
    classes = ["register_permute", "warp_shuffle", "shared_memory"]
    names = {"register_permute": "register permute", "warp_shuffle": "warp shuffle", "shared_memory": "shared memory"}
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.set_yscale("symlog", linthresh=1, linscale=0.6)
    offsets = {32: -0.22, 16: 0.0, 8: 0.22}
    n_points = 0
    for ci, cls in enumerate(classes):
        for bits in (32, 16, 8):
            pts = [r for r in rows if r["class"] == cls and r["bits"] == bits]
            if not pts:
                continue
            x = ci + offsets[bits]
            measured = [r["cvt_per_add"] for r in pts]
            assumed = st.median(r["assumed_per_add"] for r in pts)
            # the gap, the assumption (open ring), and every measured conversion (filled)
            ax.plot([x, x], [st.median(measured), assumed], color=AXIS, linewidth=1.2, zorder=1)
            ax.scatter([x], [assumed], s=70, facecolors=SURFACE, edgecolors=WIDTH_COLOR[bits], linewidths=1.8, zorder=2)
            jitter = [x + (i - (len(measured) - 1) / 2) * 0.022 for i in range(len(measured))]
            ax.scatter(jitter, measured, s=46, color=WIDTH_COLOR[bits], edgecolors=SURFACE, linewidths=1.2, zorder=3)
            n_points += len(measured)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([names[c] for c in classes], color=INK2, fontsize=10)
    ax.set_xlim(-0.55, len(classes) - 0.45)
    ticks = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(-0.25, 220)
    ax.set_ylabel("cost of one conversion, in elementwise adds of the same tensor", color=INK2, fontsize=9)
    # two direct labels, one for each kind of mark
    ax.annotate("what the pass charges", xy=(1 - 0.22, 32), xytext=(0.42, 150), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8, shrinkB=5), ha="center", va="center")
    ax.annotate("measured on the RTX 5090", xy=(1 - 0.22, 7.3), xytext=(0.36, 2.4), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8, shrinkB=5), ha="center", va="center")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=WIDTH_COLOR[b], label=WIDTH_LABEL[b])
               for b in (32, 16, 8)]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, ncol=3, handletextpad=0.2,
                    columnspacing=1.2, bbox_to_anchor=(0.0, 1.10))
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out / "assumed_vs_measured.png", facecolor=SURFACE)
    plt.close(fig)
    return n_points


def fig_shuffle_vs_smem(e3, out):
    """The same conversion lowered both ways."""
    pairs = [p for p in e3["pairs"] if p["dtype"] != "f64"]
    fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    lo, hi = 1.5, 120
    ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1.2, zorder=1)
    ax.scatter([p["smem_ns"] for p in pairs], [p["shuffle_ns"] for p in pairs], s=46, color=ACCENT,
               edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [2, 5, 10, 20, 50, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("through shared memory, SM-ns per conversion", color=INK2, fontsize=9)
    ax.set_ylabel("by warp shuffle, SM-ns per conversion", color=INK2, fontsize=9)
    ax.text(48, 62, "equal cost", color=MUTED, fontsize=9, rotation=45, ha="center", va="center")
    low = min(pairs, key=lambda p: p["ratio"])
    ax.annotate("two-element tensor", xy=(low["smem_ns"], low["shuffle_ns"]), xytext=(14, 2.6), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8, shrinkB=5), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out / "shuffle_vs_shared_memory.png", facecolor=SURFACE)
    plt.close(fig)
    return len(pairs)


def short_kernel(label):
    """`resnet50__triton_red_fused_..._view_0` -> `red ..view_0`; JIT kernels keep their own name."""
    m = re.match(r"^(?:\w+?)__triton_(per|poi|red)_fused_(.*)$", label)
    if not m:
        return label
    tail = "_".join([t for t in m.group(2).split("_") if t][-2:])
    return f"{m.group(1)} ..{tail}"


def fig_e4(e4, out):
    """Runtime of every alternative resolution that produced different code, against the default."""
    studies = [s for s in e4["studies"] if s.get("timed") and not s.get("unreliable_timing")]
    if not studies:
        return 0
    order = {"inductor": 0, "triton_kernels": 1}
    studies.sort(key=lambda s: (order.get(s["section"], 2), min(v["vs_default_pct"] for v in s["timed"])))
    studies = studies[::-1]  # first row at the top
    fig, ax = plt.subplots(figsize=(8.4, 0.3 * len(studies) + 1.5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xscale("symlog", linthresh=10, linscale=0.7)
    ax.axvspan(-3, 3, color=GRID, alpha=0.7, zorder=0, linewidth=0)
    ax.axvline(0, color=AXIS, linewidth=1.2, zorder=1)
    n = 0
    lo, hi = -10.0, 10.0
    for i, s in enumerate(studies):
        vs = sorted(s["timed"], key=lambda v: v["vs_default_pct"])
        xs = [v["vs_default_pct"] for v in vs]
        # spread points that would overlap across the row's height
        ys = [i + ((j % 5) - 2) * 0.13 for j in range(len(xs))] if len(xs) > 1 else [i]
        colors = [ACCENT if (v["faster"] or v["slower"]) else NEUTRAL for v in vs]
        ax.scatter(xs, ys, s=22, color=colors, edgecolors=SURFACE, linewidths=0.6, zorder=3)
        n += len(xs)
        lo, hi = min(lo, xs[0]), max(hi, xs[-1])
        # a build more than twice as fast or as slow gets its factor written beside it
        for v, x, y in zip(vs, xs, ys):
            if not (v["faster"] or v["slower"]):
                continue
            if x <= -50:
                ax.annotate(f"{100 / (100 + x):.0f}× faster", xy=(x, y), xytext=(7, 0), textcoords="offset points",
                            color=INK2, fontsize=8, va="center", ha="left")
            elif x >= 100:
                ax.annotate(f"{(100 + x) / 100:.1f}× slower", xy=(x, y), xytext=(-7, 0), textcoords="offset points",
                            color=INK2, fontsize=8, va="center", ha="right")
    ax.set_yticks(range(len(studies)))
    ax.set_yticklabels([short_kernel(s["label"]) if s["section"] == "inductor" else s["short"] for s in studies],
                       color=INK2, fontsize=8)
    # name each corpus section at its first row, with a rule between sections
    group_name = {"inductor": "Inductor kernels", "triton_kernels": "triton_kernels"}
    for i, s in enumerate(studies):
        above = studies[i + 1]["section"] if i + 1 < len(studies) else None
        if above != s["section"]:
            ax.annotate(group_name.get(s["section"], s["section"]), xy=(1.0, i + 0.44), xycoords=("axes fraction", "data"),
                        ha="right", va="top", color=MUTED, fontsize=8.5)
            if above is not None:
                ax.axhline(i + 0.5, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_ylim(-0.7, len(studies) - 0.3)
    # runtime cannot fall by more than 100%
    left = -160.0 if lo < -25 else -25.0 if lo < -10 else -12.0
    ticks = [t for t in (-90, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 500) if left <= t <= hi * 1.3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:+d}" if t else "0" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(left, hi * 1.35)
    ax.set_xlabel("runtime against the default resolution of the same kernel, % (below zero is faster)", color=INK2,
                  fontsize=9)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=ACCENT,
                          label="differs by 3% or more, intervals apart, in both runs"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=NEUTRAL, label="no reportable difference")]
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5, bbox_to_anchor=(-0.02, 1.0), ncol=2,
                    handletextpad=0.2, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out / "alternatives_vs_default.png", facecolor=SURFACE)
    plt.close(fig)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    e3 = json.loads((res / "e3_summary.json").read_text())
    print("assumed_vs_measured:", fig_assumed_vs_measured(e3, out), "measured conversions")
    print("shuffle_vs_shared_memory:", fig_shuffle_vs_smem(e3, out), "conversions")
    e4_path = res / "e4_summary.json"
    if e4_path.exists():
        print("alternatives_vs_default:", fig_e4(json.loads(e4_path.read_text()), out), "timed alternatives")


if __name__ == "__main__":
    main()
