"""Summarise dev diagnostics: mean first-divergent-token C shift (t-interval over attributes).
uv run --with numpy --with scipy python analysis/dev_diagnostics.py results/phase1/dev_diag_<tag>.jsonl"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

rows = [json.loads(line) for line in open(sys.argv[1])]


def d(r, c):
    a, b = r[c]
    return a[r["j"]] - b[r["j"]]


def summ(sel, cond="C"):
    x = np.array([d(r, f"{cond}_X") - d(r, f"{cond}_Y") for r in sel if f"{cond}_X" in r])
    if len(x) < 2:
        return "n/a"
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return f"{x.mean():6.3f} [{x.mean() - h:6.3f},{x.mean() + h:6.3f}] n={len(x)}"


groups = defaultdict(list)
for r in rows:
    groups[(r["block"], r.get("pos"), r["variant"], r["sess"])].append(r)
for k in sorted(groups):
    line = f"{k[0]:15s} {str(k[1]):8s} {k[2]:12s} {k[3]:8s} C {summ(groups[k])}"
    if any("M_X" in r for r in groups[k]):
        line += f"   M {summ(groups[k], 'M')}"
    print(line)
