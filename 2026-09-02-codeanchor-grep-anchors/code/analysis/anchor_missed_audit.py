#!/usr/bin/env python3
"""Re-audit of the gold files missed by both arms in round 1 (2026-09-14).

1. Per instance: files named by the in-image uncapped replay vs the clean local replay
   (anchor_replay_local.py); rows that diverge had a broken reference server.
2. The cap-counterfactual classification under each replay.
3. Necessity: for every missed gold file, in how many runs (any arm, any round in
   <runs_dir>) was the instance resolved, and in how many of those did the resolving patch
   omit the file. "Omitted by a resolving patch" = not required by the graded tests.

Usage: anchor_missed_audit.py <promax.json> <runs_dir> <image_replay.jsonl> <local_replay.jsonl>
                              [--env base_image_env.txt] [--e s1-arm-e-r1] [--a s1-arm-a8-r1] [-v]
"""
import argparse, glob, json, os, re
from collections import Counter

TEST_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")


def gold_files(p):
    out, cur = {}, None
    for line in p.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
        if m: cur = m.group(2); out[cur] = "existing"
        elif cur and line.startswith("new file mode"): out[cur] = "new"
        elif cur and line.startswith("deleted file mode"): out[cur] = "deleted"
    return out


def run_info(runs_dir, run):
    preds = json.load(open(f"{runs_dir}/{run}/preds.json"))
    pr = {x["instance_id"]: x for x in json.load(open(f"{runs_dir}/{run}/pass_rate.json"))}
    out = {}
    for i, x in pr.items():
        mp = preds.get(i, {}).get("model_patch", "") if isinstance(preds, dict) else next((p["model_patch"] for p in preds if p["instance_id"] == i), "")
        out[i] = (set(re.findall(r"^diff --git a/(\S+) b/", mp, re.M)),
                  x.get("golden", {}).get("final_result") == "success",
                  (x.get("model", {}).get("final_result") or x.get("final_result")) == "success")
    return out


def classify(rows, data, E, A):
    out = []
    for iid, r in rows.items():
        if iid not in E or not E[iid][1] or iid not in A or not A[iid][1]: continue
        g = gold_files(data[iid]["patch"]); shown = set(r["shown_files"]); unc = set(r["uncapped_files"])
        for f in g:
            if f in E[iid][0] or f in A[iid][0]: continue
            if g[f] != "existing": c = "created/deleted by gold"
            elif not f.endswith(".py"): c = "non-python"
            elif f in shown: c = "python: SHOWN in capped addendum, ignored"
            elif f in unc: c = "python: named only by an uncapped server"
            else: c = "python: never named by any anchor"
            out.append((c, iid, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("promax"); ap.add_argument("runs"); ap.add_argument("image_replay"); ap.add_argument("local_replay")
    ap.add_argument("--env", default=""); ap.add_argument("--e", default="s1-arm-e-r1"); ap.add_argument("--a", default="s1-arm-a8-r1")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    data = {x["instance_id"]: x for x in json.load(open(a.promax))}
    image = {json.loads(l)["iid"]: json.loads(l) for l in open(a.image_replay)}
    local = {json.loads(l)["iid"]: json.loads(l) for l in open(a.local_replay)}
    env = {}
    if a.env:
        for l in open(a.env):
            m = re.match(r"(\S+)\s+PYTHONPATH=(\S*)", l)
            if m: env[m.group(1)] = m.group(2)
    print("## 1. Files named by the uncapped replay: in the rollout image vs a clean local server")
    print(f"{'instance':48s} {'py':>5s} {'PYTHONPATH':14s} {'image':>5s} {'local':>5s} {'local-only':>10s}")
    for iid in sorted(image):
        if iid not in local: continue
        x, y = set(image[iid]["uncapped_files"]), set(local[iid]["uncapped_files"])
        print(f"{iid:48s} {local[iid]['n_py']:5d} {env.get(iid, ''):14s} {len(x):5d} {len(y):5d} {len(y - x):10d}{'  <-- diverges' if y - x else ''}")
    runs = {os.path.basename(os.path.dirname(p)): run_info(a.runs, os.path.basename(os.path.dirname(p))) for p in glob.glob(f"{a.runs}/*/pass_rate.json")}
    E, A = runs[a.e], runs[a.a]
    for label, rows in (("image replay (as reported)", image), ("clean local replay", local)):
        cls = classify(rows, data, E, A); cat = Counter(c for c, _, _ in cls); tot = len(cls)
        print(f"\n## 2. Classification of gold files missed by both arms in round 1 — {label}: {tot} files")
        for c, n in cat.most_common(): print(f"  {n:3d} ({100 * n / tot:.0f}%)  {c}")
    cls = classify(local, data, E, A)
    print("\n## 3. Necessity: was the file omitted by any patch that resolved the instance (any run)?")
    per_bucket = Counter(); need = []
    detail = []
    for c, iid, f in cls:
        nres = nwithout = 0
        for run, ri in runs.items():
            st = ri.get(iid)
            if not st or not st[2]: continue
            nres += 1
            if f not in st[0]: nwithout += 1
        verdict = "unknown (never resolved)" if nres == 0 else ("not required" if nwithout else "never omitted")
        per_bucket[(c, verdict)] += 1; detail.append((c, iid, f, nres, nwithout, verdict))
        if verdict != "not required": need.append((c, iid, f, nres, nwithout, verdict))
    for c in sorted({c for c, _ in per_bucket}):
        print(f"  {c:45s} " + ", ".join(f"{v}: {n}" for (cc, v), n in sorted(per_bucket.items()) if cc == c))
    print("\n  files never omitted by a resolving patch (or never resolved):")
    for c, iid, f, nres, nw, v in need: print(f"    {iid:40s} {f:55s} resolved {nres} runs, omitted in {nw}  [{c}]")
    if a.v:
        print("\n  python files, detail:")
        for c, iid, f, nres, nw, v in detail:
            if c.startswith("python"): print(f"    {c:42s} {iid:40s} {f:52s} resolved {nres:2d}, omitted {nw:2d}")


if __name__ == "__main__":
    main()
