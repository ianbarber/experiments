"""Room check, descriptive (LABNOTES 2026-10-15): B* − A per size with paired bootstrap intervals (2,000
resamples, default_rng(42), as analysis/room_check.py), on the final labels and on Qwen's alone, and every
arm's accuracy under both.

uv run --with numpy python analysis/room_hint_gain.py > ../results/phase2/room/hint_gain.txt   (from code/)
"""
import hashlib
import json

import numpy as np

P = "../results/phase2/room/"
cal = json.load(open(P + "prep/split.json"))["cal"]
J = {r["key"]: r["label"] for r in map(json.loads, open(P + "judge.jsonl"))}
key = json.load(open(P + "j2/key.json"))["items"]
j2 = {tuple(key[i]): v for i, v in json.load(open(P + "j2/labels_claude.json")).items()}


def akey(qid, answer):
    return hashlib.sha1(f"{qid}\x00{answer}".encode()).hexdigest()[:16]


recs = {(r["qid"], r["arm"], r["size"]): r for r in map(json.loads, open(P + "traj.jsonl")) if not r.get("tag")}
LQ = {k: (False if r["answer"] is None else J[akey(k[0], r["answer"])]) for k, r in recs.items()}
L = {k: j2.get(k, v) for k, v in LQ.items()}


def boot(d):
    rng = np.random.default_rng(42)
    m = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def vec(lab, arm, size):
    return np.array([lab[(q, arm, size)] for q in cal], float)


for size, b in (("small", "D"), ("mid", "T"), ("large", "D")):
    for name, lab in (("final labels", L), ("Qwen labels", LQ)):
        g, lo, hi = boot(vec(lab, b, size) - vec(lab, "A", size))
        print(f"{size}: B* = {b}, {name}: B* - A {g:+.3f} [{lo:+.3f}, {hi:+.3f}]")
for name, lab in (("final labels", L), ("Qwen labels", LQ)):
    print(f"{name}: " + "; ".join(f"{a} " + " ".join(f"{vec(lab, a, s).mean():.3f}" for s in ("small", "mid", "large"))
                                  for a in "MATD"))
