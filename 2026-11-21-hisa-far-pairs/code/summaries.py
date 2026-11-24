"""Two small summaries from ../raw -> ../results.   python summaries.py

  stage1_early_late.json  grid and copy metrics split into positions before 1100 and from 1280 on
                          (Amendment 2: to separate them from the pair-23 effect, which starts at 1178)
  stage2_summary.json     final accuracy and query counts per run, and for wrong offset answers how far
                          the token the model named is from the true source
  stage1_single_pair.json single-pair cuts on prose: cost over all tokens and past the pair's window; the
                          measured mean part of pair 23 (Amendment 3)
  stage2_edges.json       Stage 2 offset accuracy against the nearest window edges of each variant
"""

import glob
import json
import math
import os

import numpy as np

T = 2048
DS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def stage1_early_late():
    out = {}
    for C in (8, 16):
        true = np.arange(C, T, C + 1)[3:]                # newline positions, rows after the third
        early, late = true[true < 1100], true[true >= 1280]
        o = {}
        for c in ("dense", "hisa-90", "hisa-180", "hisa-360", "hisa-90-keep1"):
            r = np.load(f"../raw/stage1/grid-{C}__{c}.npz")
            p, L = r["p_newline"], r["loss"]
            o[c] = {"p_true_pos_lt_1100": float(p[:, early - 1].mean()),
                    "p_true_pos_ge_1280": float(p[:, late - 1].mean()),
                    "nl_loss_pos_lt_1100": float(L[:, early - 1].mean()),
                    "nl_loss_pos_ge_1280": float(L[:, late - 1].mean())}
        out[f"grid-{C}"] = o
    d = np.load("../data/copy.npz")
    ids, meta = d["ids"], d["meta"]                      # meta columns: source start, repeat start, gap
    e, l = meta[:, 1] + 32 < 1100, meta[:, 1] >= 1280
    o = {}
    for c in ("dense", "hisa-90", "hisa-180", "hisa-360"):
        top = np.load(f"../raw/stage1/copy__{c}.npz")["top1"]
        acc = np.array([(top[i, q + 3:q + 31] == ids[i, q + 4:q + 32]).mean() for i, (_, q, _) in enumerate(meta)])
        o[c] = {"acc_copy_ends_before_1100": float(acc[e].mean()), "n_early": int(e.sum()),
                "acc_copy_starts_after_1280": float(acc[l].mean()), "n_late": int(l.sum()),
                "early_by_gap": {int(g): [float(acc[e & (meta[:, 2] == g)].mean()), int((e & (meta[:, 2] == g)).sum())]
                                 for g in (64, 128, 256, 512)}}
    out["copy"] = o
    json.dump(out, open("../results/stage1_early_late.json", "w"), indent=1)


def stage2_summary():
    out = {}
    for path in sorted(glob.glob("../results/stage2/*.jsonl")):
        if "pilot" in path:
            continue
        f = json.loads(open(path).read().strip().split("\n")[-1])
        if not f.get("final"):
            continue
        name = path.split("/")[-1][:-6]
        # recs columns: sequence, answer position, type (0 offset / 1 content), D index, correct, offset of the
        # named token from the true source (999: not within 32 positions)
        rec = np.load(f"../raw/stage2/{name}_eval.npz")["recs"]
        e = {"variant": f["variant"], "seed": f["seed"], "lr": f["lr"], "steps": f["steps"],
             "acc": f["eval_fp32_128seq"], "n": {}, "offset_errors": {}}
        for typ, tn in ((0, "offset"), (1, "content")):
            for di, D in enumerate(DS):
                q = rec[(rec[:, 2] == typ) & (rec[:, 3] == di)]
                if len(q):
                    e["n"][f"{tn}_{D}"] = int(len(q))
                if typ == 0 and len(q):
                    w = q[q[:, 4] == 0][:, 5]
                    a = abs(w)
                    e["offset_errors"][str(D)] = {
                        "n_wrong": int(len(w)), "within_1": int((a == 1).sum()),
                        "within_2_to_4": int(((a >= 2) & (a <= 4)).sum()),
                        "within_5_to_32": int(((a >= 5) & (a <= 32)).sum()), "elsewhere": int((w == 999).sum()),
                        "mean_signed_offset": float(w[w != 999].mean()) if (w != 999).any() else None}
        out[name] = e
    json.dump(out, open("../results/stage2_summary.json", "w"), indent=1)


def edges(deg):
    """first cut distance of every pair under a budget: ceil(w_l), w_l = budget / theta_l."""
    return [math.ceil(math.radians(deg) * 10000 ** (l / 32)) for l in range(32)]


def stage1_single_pair():
    L0 = np.load("../raw/stage1/prose__dense.npz")["loss"]
    out = {"note": "extra loss against dense, prose, 256 windows; past_window = positions from the cut distance on",
           "conditions": {}}
    conds = ([(f"pair-{l}-90", l, 90) for l in range(18, 25)] + [("pair-23-90-mean", 23, 90)]
             + [(f"pair-{l}-45", l, 45) for l in range(22, 28)])
    for c, l, deg in conds:
        w = edges(deg)[l]
        d = np.load(f"../raw/stage1/prose__{c}.npz")["loss"] - L0
        out["conditions"][c] = {"window": w, "all": float(d.mean()), "before_window": float(d[:, :w - 1].mean()),
                                "past_window": float(d[:, w:].mean())}
    if os.path.exists("../raw/dropped_mean.pt"):
        import torch
        m = torch.load("../raw/dropped_mean.pt", map_location="cpu").numpy() / 8.0     # [L, H, P, T], score units
        ds = [64, 512, 1000, 1178, 1400, 1800, 2000]
        out["mean_part"] = {
            "note": "mean over queries of a pair's score part at a fixed distance, 64 held-out prose windows, score units",
            "rms_by_layer_d1178_to_2000": {str(l): [float(x) for x in np.sqrt((m[:, :, l, 1178:2000] ** 2).mean((1, 2)))]
                                           for l in (22, 23, 24, 25)},
            "layer13_pair23_distances": ds,
            "layer13_pair23_by_head": [[float(x) for x in m[13, h, 23, ds]] for h in range(9)]}
    json.dump(out, open("../results/stage1_single_pair.json", "w"), indent=1)


def stage2_edges():
    S = json.load(open("../results/stage2_summary.json"))
    out = {}
    for name, e in S.items():
        if e["variant"] == "dense":
            continue
        E = sorted(set(edges(float(e["variant"].split("-")[1]))))
        out[name] = {}
        for D in DS:
            below = max([x for x in E if x <= D], default=None)
            above = min([x for x in E if x > D], default=None)
            out[name][str(D)] = {"acc": e["acc"][f"offset_{D}"], "last_edge_at_or_before_D": below,
                                 "next_edge_after_D": above}
    json.dump(out, open("../results/stage2_edges.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    stage1_early_late()
    stage2_summary()
    stage1_single_pair()
    for name, r in stage2_edges().items():
        print(name, " ".join(f"{D}:{v['acc']:.2f}[{v['last_edge_at_or_before_D']},{v['next_edge_after_D']}]"
                             for D, v in r.items() if int(D) >= 16))
