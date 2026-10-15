"""Summarise P0b: shift by target index (from start / end) and by injection variant.

python analysis/pilot_p0b.py results/pilot/p0b_q35-4b.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np

rows = [json.loads(line) for line in open(sys.argv[1])]
rng = np.random.default_rng(42)
N = 60


def shift(r):
    return (r["C_X"][0] - r["C_X"][1]) - (r["C_Y"][0] - r["C_Y"][1])


def ci(sel):
    by = defaultdict(list)
    for r in sel:
        by[r["attr"]].append(shift(r))
    attrs = list(by)
    vals = np.concatenate([np.array(by[a]) for a in attrs])
    boots = [np.mean(np.concatenate([by[attrs[i]] for i in rng.choice(len(attrs), len(attrs))]))
             for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    acc = np.mean([((r["C_X"][0] > r["C_X"][1]) + (r["C_Y"][0] < r["C_Y"][1])) / 2 for r in sel])
    return vals.mean(), lo, hi, acc, len(vals)


print(f"rows {len(rows)}")
print("\n== full injection, by target index ==")
for idx in sorted({r["target_index"] for r in rows}):
    sel = [r for r in rows if r["target_index"] == idx and r["variant"] == "full"]
    m, lo, hi, acc, n = ci(sel)
    lab = f"start+{idx}" if idx < N // 2 else f"end-{N - 1 - idx}"
    print(f"{lab:>9}: shift {m:6.2f} [{lo:6.2f},{hi:6.2f}]  2afc {acc:.2f}  n={n}")
print("\n== variants at first (idx 0) and last (idx 59) entry ==")
for idx in (0, N - 1):
    for v in sorted({r["variant"] for r in rows if r["target_index"] == idx}):
        sel = [r for r in rows if r["target_index"] == idx and r["variant"] == v]
        m, lo, hi, acc, n = ci(sel)
        print(f"idx {idx:>2} {v:>12}: shift {m:6.2f} [{lo:6.2f},{hi:6.2f}]  2afc {acc:.2f}")
