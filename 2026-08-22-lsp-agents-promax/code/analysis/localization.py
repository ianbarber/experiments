#!/usr/bin/env python3
"""Localization/thrash analysis over archived run trajectories.

For each episode: compare the files the agent read/searched/edited against the
gold patch's files, and measure how much searching/reading happened before the
agent first touched a gold file ("time to localization") and how many non-gold
files it visited ("thrash").

Usage: python3 localization.py <run_dir> [<run_dir> ...]
       (run_dir = /mnt/nas/refactorbench/runs/s1-arm-a-r1 etc.)
"""
import json, re, sys, glob, os
from collections import defaultdict

DATASET = os.path.expanduser("~/Projects/refactorbench/harness/data/swe-bench-promax.json")

def patch_files(patch: str):
    return set(m.group(1) for m in re.finditer(r"^diff --git a/(\S+) b/", patch or "", re.M))

def commands_of(traj):
    out = []
    for m in traj.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            step = []
            for tc in (m.get("tool_calls") or []):
                try:
                    step.append(json.loads(tc["function"]["arguments"]).get("command", ""))
                except Exception:
                    pass
            out.append(step)
    return out  # list of steps, each a list of commands

SEARCH_RE = re.compile(r"(^|[;&|\s])(grep|rg|find|ack|git\s+grep)\s")
PATH_RE = re.compile(r"[\w./-]+\.(?:py|pyi|pyx|txt|cfg|toml|ini|md|rst|json|yaml|yml)\b")

def analyze_run(run_dir, gold_by_id):
    pr_path = os.path.join(run_dir, "pass_rate.json")
    passed = {}
    if os.path.exists(pr_path):
        for x in json.load(open(pr_path)):
            if x["golden"]["final_result"] == "success":
                passed[x["instance_id"]] = x["passed"]
    preds = json.load(open(os.path.join(run_dir, "preds.json")))
    rows = []
    for f in glob.glob(os.path.join(run_dir, "*", "*.traj.json")):
        iid = os.path.basename(os.path.dirname(f))
        if iid not in gold_by_id:
            continue
        gold = gold_by_id[iid]
        gold_bases = {os.path.basename(p) for p in gold}
        traj = json.load(open(f))
        steps = commands_of(traj)
        n_steps = len(steps)
        searches = 0
        files_seen = set()
        first_gold_step = None
        searches_before_gold = 0
        for i, step in enumerate(steps):
            blob = " ".join(step)
            is_search = bool(SEARCH_RE.search(blob))
            searches += is_search
            mentions = set(PATH_RE.findall(blob)) if False else set(m.group(0) for m in re.finditer(PATH_RE, blob))
            files_seen |= mentions
            touches_gold = any(os.path.basename(p) in gold_bases for p in mentions)
            if first_gold_step is None:
                searches_before_gold += is_search
                if touches_gold:
                    first_gold_step = i
        model_files = patch_files(preds.get(iid, {}).get("model_patch", ""))
        inter = model_files & gold
        rows.append({
            "iid": iid,
            "resolved": passed.get(iid),          # None = golden-invalid
            "steps": n_steps,
            "first_gold_step": first_gold_step,   # None = never touched a gold file
            "searches": searches,
            "searches_before_gold": searches_before_gold if first_gold_step is not None else searches,
            "nongold_files_seen": len({os.path.basename(p) for p in files_seen} - gold_bases),
            "gold_files": len(gold),
            "edit_precision": len(inter) / len(model_files) if model_files else None,
            "edit_recall": len(inter) / len(gold) if gold else None,
            "missed_files": len(gold - model_files),
        })
    return rows

def summarize(name, rows):
    def agg(sel, key, fmt="{:.1f}"):
        vals = [r[key] for r in sel if r[key] is not None]
        return fmt.format(sum(vals) / len(vals)) if vals else "-"
    for label, sel in [("resolved", [r for r in rows if r["resolved"] is True]),
                       ("unresolved", [r for r in rows if r["resolved"] is False])]:
        if not sel:
            continue
        never = sum(1 for r in sel if r["first_gold_step"] is None)
        print(f"{name:14s} {label:10s} n={len(sel):2d} | steps {agg(sel,'steps')} | "
              f"first-gold-touch step {agg(sel,'first_gold_step')} (never: {never}) | "
              f"searches {agg(sel,'searches')} (pre-gold {agg(sel,'searches_before_gold')}) | "
              f"non-gold files seen {agg(sel,'nongold_files_seen')} | "
              f"edit P {agg(sel,'edit_precision','{:.2f}')} R {agg(sel,'edit_recall','{:.2f}')} | "
              f"gold files missed {agg(sel,'missed_files')}")

def main():
    data = json.load(open(DATASET))
    gold_by_id = {x["instance_id"]: patch_files(x["patch"]) for x in data}
    all_rows = {}
    for run_dir in sys.argv[1:]:
        rows = analyze_run(run_dir.rstrip("/"), gold_by_id)
        name = os.path.basename(run_dir.rstrip("/"))
        all_rows[name] = rows
        summarize(name, rows)
    # cross-arm: instances unresolved everywhere vs their localization stats
    print("\nPer-instance detail (unresolved episodes):")
    for name, rows in all_rows.items():
        for r in sorted(rows, key=lambda r: r["iid"]):
            if r["resolved"] is False:
                print(f"  {name} {r['iid'][:44]:44s} steps={r['steps']:3d} "
                      f"1st-gold={r['first_gold_step'] if r['first_gold_step'] is not None else 'NEVER':>5} "
                      f"searches={r['searches']:3d} nongold={r['nongold_files_seen']:3d} "
                      f"P={r['edit_precision'] if r['edit_precision'] is not None else '-'} "
                      f"R={'%.2f'%r['edit_recall'] if r['edit_recall'] is not None else '-'} "
                      f"missed={r['missed_files']}")

if __name__ == "__main__":
    main()
