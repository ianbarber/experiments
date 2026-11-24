"""Stage 1 analysis: ../raw/stage1/*.npz -> ../results/stage1_summary.json and printed tables.

All differences are condition minus dense, paired by window, with a bootstrap 95% interval over windows.
python stage1_analyze.py
"""

import glob
import json
import os
import string

import numpy as np
from transformers import AutoTokenizer

RAW = "../raw/stage1"
rng = np.random.default_rng(42)
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
T = 2048


def boot(x, n=10_000):
    """mean and 95% bootstrap interval of per-window values x."""
    x = np.asarray(x, dtype=np.float64)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load(ds, cond):
    p = f"{RAW}/{ds}__{cond}.npz"
    return np.load(p) if os.path.exists(p) else None


def token_classes(ids):
    """class of every target token (positions 1..T-1): 0 other, 1 newline, 2 delimiter (punctuation only)."""
    punct = set(string.punctuation + " \t")
    cls = {}
    for t in np.unique(ids):
        s = tok.decode([int(t)])
        cls[int(t)] = 1 if "\n" in s else (2 if s.strip() and all(c in punct for c in s) else 0)
    return np.vectorize(cls.get)(ids[:, 1:])


conds = sorted({os.path.basename(p).split("__")[1][:-4] for p in glob.glob(f"{RAW}/*.npz")} - {"dense"})
out = {}

for ds in ("prose", "code", "tables", "copy"):
    dense = load(ds, "dense")
    if dense is None:
        continue
    ids = np.load(f"../data/{ds}.npz")["ids"]
    L0 = dense["loss"]
    res = {"dense": {"loss": float(L0.mean()), "n_windows": int(len(L0))}}
    cls = token_classes(ids) if ds in ("code", "tables") else None
    if cls is not None:
        res["dense"]["by_class"] = {n: {"share": float((cls == c).mean()), "loss": float(L0[cls == c].mean())}
                                    for c, n in ((0, "other"), (1, "newline"), (2, "delimiter"))}
    if ds == "copy":
        meta = np.load("../data/copy.npz")["meta"]
    for cond in ["dense"] + conds:
        r = load(ds, cond)
        if r is None:
            continue
        e = res.setdefault(cond, {})
        d = r["loss"] - L0
        if cond != "dense":
            e["loss"] = float(r["loss"].mean())
            e["delta"] = boot(d.mean(1))
            e["delta_by_pos"] = {n: boot(d[:, a:b].mean(1)) for n, (a, b) in
                                 {"0-512": (0, 511), "512-1024": (511, 1023), "1024-2048": (1023, 2047)}.items()}
            if cls is not None:
                e["delta_by_class"] = {}
                for c, n in ((0, "other"), (1, "newline"), (2, "delimiter")):
                    w = (cls == c)
                    e["delta_by_class"][n] = boot((d * w).sum(1) / np.maximum(w.sum(1), 1))
            if ds == "tables":
                kind = np.load("../data/tables.npz")["kind"]
                e["delta_by_kind"] = {k: boot(d[kind == k].mean(1)) for k in ("csv", "jsonl", "md")}
        if ds == "copy":
            # top-1 accuracy on tokens 5..32 of the repeated span; top1[n, t] predicts token t+1
            acc = np.array([(r["top1"][i, q + 4 - 1:q + 32 - 1] == ids[i, q + 4:q + 32]).mean()
                            for i, (p, q, g) in enumerate(meta)])
            e["copy_acc"] = boot(acc)
            e["copy_acc_by_gap"] = {int(g): float(acc[meta[:, 2] == g].mean()) for g in np.unique(meta[:, 2])}
            if cond != "dense":
                acc0 = res["dense"]["_acc"]
                e["copy_acc_delta"] = boot(acc - acc0)
            else:
                e["_acc"] = acc
    res["dense"].pop("_acc", None)
    out[ds] = res

for C in (8, 16, 32, 64, 128, 256):
    ds = f"grid-{C}"
    dense = load(ds, "dense")
    if dense is None:
        continue
    g = np.load(f"../data/{ds}.npz")
    nl = int(g["newline"])
    true = np.arange(C, T, C + 1)[3:]             # newline positions, rows after the third
    first = true + 1
    first = first[first < T]
    res = {}

    def grid_metrics(r):
        p, L, top = r["p_newline"], r["loss"], r["top1"]
        m = {"nl_loss": L[:, true - 1].mean(1), "p_true": p[:, true - 1].mean(1),
             "top1_newline": (top[:, true - 1] == nl).mean(1), "first_cell_loss": L[:, first - 1].mean(1),
             "loss": L.mean(1)}
        for o in (-3, -2, -1):
            m[f"p_off{o}"] = p[:, true + o - 1].mean(1)
        # most likely newline position within each row (over the C cell slots and the true slot): exactly right?
        hits = []
        for t in true:
            seg = p[:, t - C - 1 + 1:t]             # predictions for positions t-C+1 .. t  (the row's slots)
            hits.append(seg.argmax(1) == seg.shape[1] - 1)
        m["argmax_row_exact"] = np.mean(hits, axis=0)
        return m

    m0 = grid_metrics(dense)
    res["dense"] = {k: float(v.mean()) for k, v in m0.items()}
    res["dense"]["gate_pass"] = bool(m0["p_true"].mean() >= 0.5)
    res["dense"]["n_windows"], res["dense"]["rows_scored_per_window"] = int(len(dense["loss"])), int(len(true))
    for cond in conds:
        r = load(ds, cond)
        if r is None:
            continue
        m = grid_metrics(r)
        res[cond] = {k: float(v.mean()) for k, v in m.items()}
        for k in ("nl_loss", "p_true", "first_cell_loss", "loss", "argmax_row_exact"):
            res[cond]["delta_" + k] = boot(m[k] - m0[k])
    out[ds] = res

json.dump(out, open("../results/stage1_summary.json", "w"), indent=1)

# ----------------------------------------------------------------------------- printed tables
order = [c for c in ["hisa-45", "hisa-90", "hisa-180", "hisa-360", "hisa-720", "hisa-90-mean", "band-90-180",
                     "band-180-360", "band-360-720", "band-720+", "ctrl-perm-s0", "ctrl-perm-s1", "ctrl-perm-s2",
                     "ctrl-iid-s0", "ctrl-slow"] if c in conds]
print("delta loss vs dense, nats/token [95% CI]")
print(f"{'':14s}" + "".join(f"{d:>24s}" for d in ("prose", "code", "tables", "copy")))
print(f"{'dense loss':14s}" + "".join(f"{out[d]['dense']['loss']:24.4f}" for d in ("prose", "code", "tables", "copy")))
for c in order:
    row = f"{c:14s}"
    for d in ("prose", "code", "tables", "copy"):
        e = out[d].get(c)
        row += f"{e['delta'][0]:+9.4f} [{e['delta'][1]:+.3f},{e['delta'][2]:+.3f}]" if e and "delta" in e else " " * 24
    print(row)
print("\ncopy accuracy (tokens 5-32):", {c: round(out["copy"][c]["copy_acc"][0], 3) for c in ["dense"] + order
                                         if "copy_acc" in out["copy"].get(c, {})})
print("\ngrids: P(newline at true position) / newline loss / most likely position in row exactly right")
for C in (8, 16, 32, 64, 128, 256):
    r = out.get(f"grid-{C}")
    if not r:
        continue
    print(f"C={C} gate={'PASS' if r['dense']['gate_pass'] else 'fail'}")
    for c in ["dense"] + order:
        if c in r:
            e = r[c]
            print(f"   {c:14s} p_true {e['p_true']:.3f}  nl_loss {e['nl_loss']:.3f}  top1 {e['top1_newline']:.3f}  "
                  f"row-exact {e['argmax_row_exact']:.3f}  p(-1) {e['p_off-1']:.3f}  first-cell {e['first_cell_loss']:.3f}")
