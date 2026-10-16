"""Figure 5: room check accuracy against memory size (LABNOTES 2026-10-15). Final labels as in
analysis/room_check.py: the second judge's where it labelled, Qwen's elsewhere. Wilson 95% intervals.

uv run --with numpy --with matplotlib python analysis/room_figure.py   (from code/)
"""
import hashlib
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "../results/phase2/room/"
SIZES, X = ("small", "mid", "large"), (25, 110, 200)


def akey(qid, answer):
    return hashlib.sha1(f"{qid}\x00{answer}".encode()).hexdigest()[:16]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


cal = json.load(open(P + "prep/split.json"))["cal"]
J = {r["key"]: r["label"] for r in map(json.loads, open(P + "judge.jsonl"))}
key = json.load(open(P + "j2/key.json"))["items"]
j2 = {tuple(key[i]): v for i, v in json.load(open(P + "j2/labels_claude.json")).items()}
recs = {(r["qid"], r["arm"], r["size"]): r for r in map(json.loads, open(P + "traj.jsonl")) if not r.get("tag")}
L = {k: j2.get(k, False if r["answer"] is None else J[akey(k[0], r["answer"])]) for k, r in recs.items()}


def acc(arm, size):
    k = sum(L[(q, arm, size)] for q in cal)
    return k / len(cal), wilson(k, len(cal))


fig, ax = plt.subplots(figsize=(6.4, 4.2))
for arm, col, lab in (("M", "C0", "M: whole memory in context (+ tools)"), ("A", "C7", "A: agent with tools, no hint"),
                      ("T", "C1", "T: agent + keyword index"), ("D", "C2", "D: agent + one-line note per file")):
    ys = [acc(arm, s) for s in SIZES]
    ax.errorbar(X, [y for y, _ in ys], yerr=[[y - lo for y, (lo, _) in ys], [hi - y for y, (_, hi) in ys]],
                color=col, marker="o", ms=4, capsize=3, lw=1.4 if arm == "M" else 1.0, label=lab)
for arm, ls, lab in (("O", "--", "O: evidence sessions only, in context"), ("N", ":", "N: no memory")):
    y, _ = acc(arm, "O")
    ax.axhline(y, color="grey", ls=ls, lw=0.9)
    ax.text(203, y + 0.012, lab.split(":")[0], color="grey", fontsize=8, va="bottom", ha="right")
ax.set_xticks(X, ["~25k", "~110k", "~200k"])
ax.set_xlabel("memory size (tokens)")
ax.set_ylabel("answer accuracy (120 calibration questions)")
ax.set_ylim(0, 1)
ax.legend(loc="lower left", bbox_to_anchor=(0, 0.1), fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("../images/room_check.png", dpi=150)
print({arm: [round(acc(arm, s)[0], 3) for s in SIZES] for arm in "MATD"}, "O", acc("O", "O")[0], "N", acc("N", "O")[0])
