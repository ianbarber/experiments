#!/usr/bin/env python3
"""Cap counterfactual: for gold-patch files the agent did NOT patch, was the file (a) shown
in a capped addendum, (b) only in the uncapped replay (hidden by the cap), or (c) never
reachable from the symbols the agent searched? Usage: <replay.jsonl> <E run> <A run>"""
import json, re, sys, glob, os
from collections import Counter
TEST_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")
replay, erun, arun = sys.argv[1:4]
data = {x["instance_id"]: x for x in json.load(open(os.path.expanduser("~/Projects/refactorbench/harness/data/swe-bench-promax.json")))}
def gold_files(p):
    out = {}; cur = None
    for line in p.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
        if m: cur = m.group(2); out[cur] = "existing"
        elif cur and line.startswith("new file mode"): out[cur] = "new"
        elif cur and line.startswith("deleted file mode"): out[cur] = "deleted"
    return out
def run_info(run):
    preds = json.load(open(f"/mnt/nas/refactorbench/runs/{run}/preds.json"))
    pr = {x["instance_id"]: x for x in json.load(open(f"/mnt/nas/refactorbench/runs/{run}/pass_rate.json"))}
    return {i: (set(re.findall(r"^diff --git a/(\S+) b/", preds.get(i, {}).get("model_patch", ""), re.M)),
                pr.get(i, {}).get("golden", {}).get("final_result") == "success") for i in pr}
E = run_info(erun); A = run_info(arun)
rows = {}
for line in open(replay):
    r = json.loads(line); rows[r["iid"]] = r
cat = Counter(); cat_src = Counter(); per = []
for iid, r in rows.items():
    if iid not in E or not E[iid][1] or iid not in A or not A[iid][1]:
        continue
    g = gold_files(data[iid]["patch"]); shown = set(r["shown_files"]); unc = set(r["uncapped_files"])
    missed_E = [f for f in g if f not in E[iid][0]]
    both_missed = [f for f in missed_E if f not in A[iid][0]]
    for f in both_missed:
        if g[f] != "existing": c = "gold creates/deletes the file"
        elif not f.endswith(".py"): c = "non-python"
        elif TEST_RE.search(f): c = "test file"
        elif f in shown: c = "python source: SHOWN in a capped addendum (ignored)"
        elif f in unc: c = "python source: HIDDEN BY THE CAP (in uncapped replay only)"
        else: c = "python source: never reachable from searched symbols"
        cat[c] += 1
        if f.endswith(".py") and not TEST_RE.search(f) and g[f] == "existing": cat_src[c] += 1
    per.append((iid, len(both_missed), sum(1 for f in both_missed if f in unc and f not in shown and f.endswith(".py"))))
tot = sum(cat.values())
print(f"instances analysed: {len(per)} (replay rows {len(rows)}); gold files missed by BOTH arms: {tot}")
for c, n in cat.most_common(): print(f"  {n:3d} ({100*n/max(1,tot):.0f}%)  {c}")
print("\nper-instance (missed-by-both, of which hidden-by-cap python source):")
for iid, n, h in sorted(per, key=lambda x: -x[2]): 
    if n: print(f"  {iid[:44]:44s} missed {n:3d}  hidden-by-cap {h}")
# also: how much bigger is the uncapped view overall?
sh = sum(len(r["shown_files"]) for r in rows.values()); un = sum(len(r["uncapped_files"]) for r in rows.values())
sc = sum(r["symbols_capped"] for r in rows.values()); su = sum(r["symbols_uncapped"] for r in rows.values())
print(f"\nfiles named: capped {sh} vs uncapped {un} ({un/max(1,sh):.1f}x); symbols rendered: capped {sc} vs uncapped-with-refs {su}")
