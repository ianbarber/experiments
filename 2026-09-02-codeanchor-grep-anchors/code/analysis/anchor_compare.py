#!/usr/bin/env python3
"""Arm E (anchors) vs A8 (control) comparison over archived runs.

Usage: python3 anchor_compare.py --e s1-arm-e-r1[,s1-arm-e-r2] --a s1-arm-a8-r1[,s1-arm-a8-r2]
       [--runs-root /mnt/nas/refactorbench/runs] [--csv out.csv]

Per episode: resolved (golden-valid only), steps, cumulative prompt tokens, completion and
reasoning tokens, wall-clock (first→last model response), grep-family commands, container
deaths, edit precision/recall vs gold-patch files; for E additionally the anchor telemetry
(anchored greps, addendum chars, seconds). Paired instance-level comparison uses the
per-instance score summed over rounds (0..R) on the intersection of golden-valid sets, with
an exact two-sided sign test on discordant instances.
"""
import argparse, glob, json, math, os, re, statistics as st
from collections import Counter, defaultdict

DATASET = os.path.expanduser("~/Projects/refactorbench/harness/data/swe-bench-promax.json")
GREP_RE = re.compile(r"(?<![\w./-])(grep|rg|egrep|fgrep)\b")
ANCHOR_HDR = "--- code anchors (language server)"


def patch_files(patch):
    return set(m.group(1) for m in re.finditer(r"^diff --git a/(\S+) b/", patch or "", re.M))


def episode_rows(run_dir, gold_by_id):
    passed = {}
    pr = os.path.join(run_dir, "pass_rate.json")
    if os.path.exists(pr):
        for x in json.load(open(pr)):
            if x["golden"]["final_result"] == "success":
                passed[x["instance_id"]] = bool(x["passed"])
    preds = json.load(open(os.path.join(run_dir, "preds.json"))) if os.path.exists(os.path.join(run_dir, "preds.json")) else {}
    anchor = defaultdict(list)
    al = os.path.join(run_dir, "anchor_log.jsonl")
    if os.path.exists(al):
        for line in open(al):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            anchor[r.get("instance")].append(r)
    rows = {}
    for f in glob.glob(os.path.join(run_dir, "*", "*.traj.json")):
        iid = os.path.basename(os.path.dirname(f))
        t = json.load(open(f))
        msgs = t.get("messages", [])
        steps = greps = in_tok = out_tok = reason = 0
        created = []
        deaths = 0
        anchored_obs = anchored_chars = 0
        for m in msgs:
            if m.get("role") == "assistant":
                steps += 1
                resp = (m.get("extra") or {}).get("response") or {}
                u = resp.get("usage") or {}
                in_tok += u.get("prompt_tokens", 0) or 0
                out_tok += u.get("completion_tokens", 0) or 0
                reason += ((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
                if resp.get("created"):
                    created.append(resp["created"])
                for a in (m.get("extra") or {}).get("actions", []):
                    if GREP_RE.search(a.get("command", "")):
                        greps += 1
            elif m.get("role") == "tool":
                c = m.get("content", "") or ""
                if "No such container" in c:
                    deaths += 1
                if ANCHOR_HDR in c:
                    anchored_obs += 1
                    anchored_chars += len(c[c.index(ANCHOR_HDR):])
        gold = gold_by_id.get(iid, set())
        model_files = patch_files(preds.get(iid, {}).get("model_patch", ""))
        inter = model_files & gold
        arec = anchor.get(iid, [])
        rows[iid] = {
            "resolved": passed.get(iid),  # None = golden-invalid
            "exit": (t.get("info") or {}).get("exit_status"),
            "steps": steps, "greps": greps,
            "in_tok": in_tok, "out_tok": out_tok, "reason_tok": reason,
            "wall_h": (max(created) - min(created)) / 3600 if len(created) > 1 else 0.0,
            "wall_death": deaths > 0,
            "edit_precision": len(inter) / len(model_files) if model_files else None,
            "edit_recall": len(inter) / len(gold) if gold else None,
            "n_model_files": len(model_files), "n_gold_files": len(gold),
            "anchored_obs": anchored_obs, "anchored_chars": anchored_chars,
            "anchor_calls": sum(1 for r in arec if r.get("event") == "grep"),
            "anchor_anchored": sum(1 for r in arec if r.get("status") == "anchored"),
            "anchor_not_ready": sum(1 for r in arec if r.get("status") == "not_ready"),
            "anchor_timeout": sum(1 for r in arec if r.get("status") == "timeout"),
            "anchor_seconds": sum(r.get("seconds", 0) or 0 for r in arec if r.get("event") == "grep"),
        }
    return rows


def sign_test(n_pos, n_neg):
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(name, runs):
    keys = ["steps", "greps", "in_tok", "out_tok", "reason_tok", "wall_h", "edit_recall", "edit_precision"]
    allrows = [r for run in runs for r in run.values()]
    valid = [r for r in allrows if r["resolved"] is not None]
    res = sum(1 for r in valid if r["resolved"])
    print(f"\n== {name}: {len(runs)} run(s), {len(allrows)} episodes, golden-valid {len(valid)}: "
          f"resolved {res}/{len(valid)} = {100*res/max(1,len(valid)):.0f}%")
    print(f"   exits: {dict(Counter(r['exit'] for r in allrows))}; wall deaths: {sum(r['wall_death'] for r in allrows)}")
    for k in keys:
        v = [r[k] for r in allrows if r[k] is not None]
        if not v:
            continue
        print(f"   {k:15s} mean {st.mean(v):10.3f}  median {st.median(v):10.3f}")
    if any(r["anchor_calls"] for r in allrows):
        print(f"   anchor: grep-cmds seen {sum(r['anchor_calls'] for r in allrows)}, anchored "
              f"{sum(r['anchor_anchored'] for r in allrows)}, not_ready {sum(r['anchor_not_ready'] for r in allrows)}, "
              f"timeouts {sum(r['anchor_timeout'] for r in allrows)}; anchored observations in trajs "
              f"{sum(r['anchored_obs'] for r in allrows)} (mean {st.mean([r['anchored_obs'] for r in allrows]):.1f}/episode, "
              f"mean addendum chars/episode {st.mean([r['anchored_chars'] for r in allrows]):.0f}); "
              f"mean anchor seconds/episode {st.mean([r['anchor_seconds'] for r in allrows]):.1f}")
    return allrows


def paired(name_e, runs_e, name_a, runs_a):
    ids = None
    for run in runs_e + runs_a:
        s = {i for i, r in run.items() if r["resolved"] is not None}
        ids = s if ids is None else ids & s
    ids = sorted(ids or [])
    score = lambda runs, i: sum(int(run[i]["resolved"]) for run in runs)
    pos = neg = 0
    diffs = defaultdict(list)
    for i in ids:
        se, sa = score(runs_e, i), score(runs_a, i)
        if se > sa:
            pos += 1
        elif sa > se:
            neg += 1
        for k in ("steps", "in_tok", "wall_h", "greps", "edit_recall"):
            ve = [run[i][k] for run in runs_e if run[i][k] is not None]
            va = [run[i][k] for run in runs_a if run[i][k] is not None]
            if ve and va:
                diffs[k].append(st.mean(ve) - st.mean(va))
    print(f"\n== paired {name_e} vs {name_a} on {len(ids)} golden-valid instances (instance scores summed over rounds):")
    print(f"   {name_e} better {pos}, {name_a} better {neg}, sign-test p={sign_test(pos, neg):.3f}")
    for k, d in diffs.items():
        w = sum(1 for x in d if x > 0); l = sum(1 for x in d if x < 0)
        print(f"   Δ{k:12s} mean {st.mean(d):+10.3f} median {st.median(d):+10.3f}  (E higher on {w}, lower on {l}, sign p={sign_test(w, l):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--runs-root", default="/mnt/nas/refactorbench/runs")
    ap.add_argument("--csv")
    args = ap.parse_args()
    data = json.load(open(DATASET))
    gold_by_id = {x["instance_id"]: patch_files(x.get("patch") or x.get("golden_patch") or "") for x in data}
    runs_e = [episode_rows(os.path.join(args.runs_root, r), gold_by_id) for r in args.e.split(",")]
    runs_a = [episode_rows(os.path.join(args.runs_root, r), gold_by_id) for r in args.a.split(",")]
    rows_e = summarize("E " + args.e, runs_e)
    rows_a = summarize("A " + args.a, runs_a)
    paired("E", runs_e, "A", runs_a)
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = None
            for arm, runs, names in (("E", runs_e, args.e.split(",")), ("A", runs_a, args.a.split(","))):
                for run, name in zip(runs, names):
                    for iid, r in run.items():
                        row = {"arm": arm, "run": name, "iid": iid, **r}
                        if w is None:
                            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
                        w.writerow(row)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
