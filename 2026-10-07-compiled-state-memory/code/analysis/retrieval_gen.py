"""Greedy first-action follow-up analysis (exploratory; dev items only).

uv run --with numpy --with scipy python analysis/retrieval_gen.py results/phase1/retrieval_gen_<tag>.jsonl

Per family and condition, from greedy decoding (binary per session):
  call      share of sessions that open a <tool_call> within 32 tokens; present / absent
  pres-abs  call share present - absent, per question (mean over 8 seeds), t 95% over 12 questions
  pre       share of calls that come after a preamble (n_pre > 0)
  first     mean first-token p(<tool_call>) (the pilot's measure, re-computed here)
  search    share of calls to search_memory
  topic     share of search queries containing the family keyword's first word
  value     share of sessions whose generated text (preamble + call) contains the stored value
            (X value words; F: the attribute value, P: a hand-listed X word), present / absent
Level contrasts in call share vs A (per question; C0 and store conditions averaged over states).
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

rows = [json.loads(line) for line in open(sys.argv[1])]
header, rows = rows[0], rows[1:]
sys.path.insert(0, ".")
sys.path.insert(0, "code")
from statelm.synth import ATTRS  # noqa: E402

# P: one word that names the stored (X) preference; budget / spice share a word with their topic
# keyword, so value leaks there are not separable from topic and are left out (None).
P_VALUE = {"diet": "vegetarian", "holiday": "beach", "verbosity": "short", "units": "metric", "music": "jazz",
           "pet": "cat", "reading": "science fiction", "exercise": "running", "budget": None, "tone": "casual",
           "drink": "tea", "spice": None}
F_VALUE = {k: pool[0] for k, _, _, _, pool in ATTRS}
CONDS = ("A", "Aw", "C0", "C", "Cw", "T", "CT", "M")


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+.3f} [{x.mean() - h:+.3f},{x.mean() + h:+.3f}]"


def value_word(r):
    return (P_VALUE if r["fam"] == "P" else F_VALUE)[r["key"]]


def text(r):
    return (r["pre"] + " " + (r.get("call") or "")).lower()


for fam in ("P", "F"):
    R = [r for r in rows if r["fam"] == fam]
    keys = sorted({r["key"] for r in R})
    print(f"\n==== family {fam} ({'implicit need' if fam == 'P' else 'explicit question'}) ====")
    print(f"  {'cond':4s} {'n':>4s} {'call pres/abs':>14s} {'pres-abs call, t95 over q':>27s} {'pre|call':>8s} "
          f"{'first p':>7s} {'search':>6s} {'topic':>6s} {'value pres/abs':>15s}")
    share = {}
    for cond in CONDS:
        C = [r for r in R if r["cond"] == cond]
        share[cond] = {k: np.mean([r["called"] for r in C if r["key"] == k]) for k in keys}
        calls = [r for r in C if r["called"]]
        srch = [r for r in calls if r.get("fn") == "search_memory"]
        topic = np.mean([r["kw"].split()[0].lower() in (r["arg"] or "").lower() for r in srch]) if srch else np.nan
        common = (f"{np.mean([r['n_pre'] > 0 for r in calls]) if calls else np.nan:8.3f} "
                  f"{np.mean([np.exp(r['lp_call']) for r in C]):7.3f} "
                  f"{len(srch) / max(len(calls), 1):6.3f} {topic:6.3f}")
        if cond in ("A", "Aw", "C0"):
            v = [value_word(r).lower() in text(r) for r in C if value_word(r)]
            print(f"  {cond:4s} {len(C):4d} {np.mean([r['called'] for r in C]):14.3f} {'':>27s} {common} {np.mean(v):15.3f}")
            continue
        diffs = []
        for k in keys:
            pr = [r["called"] for r in C if r["key"] == k and r["present"]]
            ab = [r["called"] for r in C if r["key"] == k and not r["present"]]
            diffs.append(np.mean(pr) - np.mean(ab))
        vp = [value_word(r).lower() in text(r) for r in C if r["present"] and value_word(r)]
        va = [value_word(r).lower() in text(r) for r in C if not r["present"] and value_word(r)]
        print(f"  {cond:4s} {len(C):4d} {np.mean([r['called'] for r in C if r['present']]):6.3f} / "
              f"{np.mean([r['called'] for r in C if not r['present']]):5.3f} {tci(diffs):>27s} {common} "
              f"{np.mean(vp):7.3f} / {np.mean(va):5.3f}")
    print("  level shift in call share (per question, t95):")
    for a, b in (("C0", "A"), ("C", "A"), ("C", "C0"), ("Cw", "Aw"), ("Aw", "A"), ("T", "A"), ("CT", "T"), ("M", "A")):
        print(f"    {a:3s} - {b:3s} {tci([share[a][k] - share[b][k] for k in keys])}")
    print("  per question call share: " + " ".join(f"{c:>5s}" for c in CONDS))
    for k in keys:
        print(f"    {k:10s}             " + " ".join(f"{share[c][k]:5.2f}" for c in CONDS))
