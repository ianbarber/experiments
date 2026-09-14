"""Pre-registered analysis of the main run (docs/07 "Endpoints and analysis").

Reads the run databases and writes `analysis/results.json` and `analysis/results.md` under the run directory:
primary endpoint (M_final vs M_0 on the union held-out set, same session), the recursion test against the
non-recursive branch with Holm correction, the negative drift control, the descriptive set (dev curve,
canaries, TerminalWorld, Terminal-Bench as published, pass@k), the control set, integrity accounting and cost.

Runs at any point: sections without data are reported as pending, so the same command works mid-run.
    venvs/loop/bin/python scripts/analyse-run.py [--run main] [--control-run main-control]
"""
from __future__ import annotations
import argparse, json, math, random, statistics, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rsi import config as C
from rsi.db import DB

DRAWS = 10000

# ---------------------------------------------------------------- statistics
def per_task(db, policy: str, purposes: list[str]) -> dict[str, list[float]]:
    """Rewards per task for a policy over an explicit list of purposes (SQL LIKE patterns allowed)."""
    out = defaultdict(list)
    clause = " OR ".join("purpose LIKE ?" for _ in purposes)
    for r in db.q(f"SELECT task_id, reward FROM episodes WHERE policy=? AND ({clause}) AND exception IS NULL AND reward IS NOT NULL", policy, *purposes):
        out[r["task_id"]].append(float(r["reward"]))
    return dict(out)

def means(d: dict[str, list[float]]) -> dict[str, float]:
    return {k: sum(v) / len(v) for k, v in d.items() if v}

def paired_bootstrap(a: dict[str, float], b: dict[str, float], *, draws: int = DRAWS, seed: int = 0) -> dict:
    """Task-level paired bootstrap of (b - a) in points; percentile 95% interval and one-sided p for diff <= 0."""
    common = sorted(set(a) & set(b))
    if len(common) < 2: return {"n": len(common), "status": "pending"}
    d = [b[t] - a[t] for t in common]; n = len(d); m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    rng = random.Random(seed); boots = []
    for _ in range(draws):
        boots.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo, hi = boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1]
    return {"n": n, "mean_a": 100 * sum(a[t] for t in common) / n, "mean_b": 100 * sum(b[t] for t in common) / n,
            "diff_points": 100 * m, "se_points": 100 * se, "ci95_points": [100 * lo, 100 * hi],
            "p_one_sided_le0": sum(1 for x in boots if x <= 0) / draws, "reject_null": lo > 0}

def holm(pvals: dict[str, float]) -> dict[str, dict]:
    """Holm-Bonferroni at alpha 0.05 over the primary and the key secondary."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; blocked = False
    for i, (k, p) in enumerate(items):
        thr = 0.05 / (m - i)
        sig = (p <= thr) and not blocked
        if not sig: blocked = True
        out[k] = {"p": p, "threshold": thr, "significant": sig}
    return out

def pass_at_k(counts: list[tuple[int, int]], k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021) averaged over tasks; counts = [(n_samples, n_correct), ...]."""
    vals = []
    for n, c in counts:
        if n < k: continue
        vals.append(1.0 if n - c < k else 1.0 - math.prod((n - c - i) / (n - i) for i in range(k)))
    return sum(vals) / len(vals) if vals else float("nan")

# ---------------------------------------------------------------- sections
def head_of(run_dir: Path) -> int:
    st = run_dir / "state.json"
    return json.loads(st.read_text()).get("lineage_head", 0) if st.exists() else 0

def episode_accounting(db) -> dict:
    tot = db.q("SELECT COUNT(*) n, SUM(exception IS NOT NULL) errors, SUM(quarantined) quarantined, SUM(stop_reason='agent_timeout') timeouts, SUM(reward IS NULL) no_reward FROM episodes")[0]
    by_purpose = [dict(r) for r in db.q("SELECT purpose, COUNT(*) n, ROUND(AVG(reward), 4) mean_reward, SUM(exception IS NOT NULL) errors FROM episodes GROUP BY purpose ORDER BY purpose")]
    gates = [dict(r) for r in db.q("SELECT iteration, check_name, verdict, COUNT(*) n FROM gates GROUP BY iteration, check_name, verdict ORDER BY iteration, check_name")]
    return {"totals": dict(tot), "by_purpose": by_purpose, "gate_funnel": gates}

def cost(db) -> dict:
    rows = db.q("SELECT k, step, started_at, finished_at FROM iterations WHERE finished_at IS NOT NULL")
    by_step = defaultdict(float)
    for r in rows: by_step[r["step"]] += (r["finished_at"] - r["started_at"]) / 3600.0
    return {"gpu_hours_by_step": {k: round(v, 2) for k, v in sorted(by_step.items(), key=lambda kv: -kv[1])}, "gpu_hours_total": round(sum(by_step.values()), 2)}

def analyse(run: str, control_run: str) -> dict:
    paths = C.run_paths(run); db = DB(paths.db)
    head = head_of(paths.root); mf = f"M{head}"
    res: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "run": run, "lineage_head": mf, "iterations_done": head}

    # 1. primary endpoint: union held-out set, same-session M_final vs M_0
    a = means(per_task(db, "M0", ["heldout:end-%"])); b = means(per_task(db, mf, ["heldout:final-%"]))
    res["primary_heldout_union"] = paired_bootstrap(a, b) | {"policies": ["M0(end)", f"{mf}(final)"]}

    # 2. key secondary: recursion test, main M_final vs the non-recursive branch
    cpaths = C.run_paths(control_run)
    if cpaths.db.exists():
        cdb = DB(cpaths.db); chead = head_of(cpaths.root)
        cb = means(per_task(cdb, f"M{chead}", ["heldout:final%"]))
        res["secondary_recursion"] = paired_bootstrap(cb, b) | {"policies": [f"control M{chead}", f"main {mf}"]}
    else:
        res["secondary_recursion"] = {"status": "pending", "note": "control branch not started"}

    ps = {k: res[k]["p_one_sided_le0"] for k in ("primary_heldout_union", "secondary_recursion") if "p_one_sided_le0" in res[k]}
    res["holm"] = holm(ps) if len(ps) == 2 else {"status": "pending", "note": "needs both tests"}

    # 3. negative control: M_0 drift between the opening and closing sessions (null |drift| <= 3.5 points)
    # the opening baseline was recorded as two tagged purposes, one per frozen set; the closing one covers the union
    drift = paired_bootstrap(means(per_task(db, "M0", ["heldout:start", "heldout-supp:start"])), a)
    if "diff_points" in drift: drift["within_null_band"] = abs(drift["diff_points"]) <= 3.5
    res["negative_control_M0_drift"] = drift

    # 3b. interim held-out reads (descriptive, never used for selection or stopping)
    interim = []
    base_union = means(per_task(db, "M0", ["heldout:start", "heldout-supp:start"]))
    for r in db.q("SELECT DISTINCT policy FROM episodes WHERE purpose='heldout:interim' ORDER BY policy"):
        pol = r["policy"]
        st = paired_bootstrap(base_union, means(per_task(db, pol, ["heldout:interim"])))
        interim.append({"policy": pol} | {k: st.get(k) for k in ("n", "mean_a", "mean_b", "diff_points", "se_points", "ci95_points", "p_one_sided_le0")})
    res["interim_heldout_vs_M0_start"] = interim

    # 4. dev curve and the gate decisions
    res["dev_curve"] = [dict(r) for r in db.q("SELECT policy, COUNT(*) n, ROUND(100*AVG(reward), 2) mean_points FROM episodes WHERE purpose LIKE 'dev%' GROUP BY policy ORDER BY policy")]
    res["dev_gate_decisions"] = [{"iteration": r["iteration"], "diff_points": 100 * r["value"], "extra": json.loads(r["extra"] or "{}")}
                                 for r in db.q("SELECT iteration, value, extra FROM metrics WHERE name='devgate_diff' ORDER BY id")]

    # 5. canaries (gate: no drop worse than 2 points against M_0)
    can = [{"policy": json.loads(r["extra"] or "{}").get("policy"), "iteration": r["iteration"], "composite_points": 100 * r["value"]}
           for r in db.q("SELECT iteration, value, extra FROM metrics WHERE name='canary_composite' ORDER BY id")]
    m0c = next((c["composite_points"] for c in can if c["policy"] == "M0"), None)
    base = m0c if m0c is not None else 51.02  # M_0 canary from the pre-registration freeze (chain20)
    for c in can: c["delta_vs_M0_points"] = round(c["composite_points"] - base, 2); c["passes_gate"] = c["delta_vs_M0_points"] > -2.0
    res["canaries"] = {"M0_baseline_points": base, "series": can}

    # 6. out-of-genre and comparability sets
    # TerminalWorld has no closing baseline for M_0, so this contrast spans sessions and is not drift-corrected
    res["terminalworld"] = paired_bootstrap(means(per_task(db, "M0", ["tw:start%"])), means(per_task(db, mf, ["tw:final%"]))) | {"baseline": "opening session, not drift-corrected"}
    res["terminal_bench_published"] = [dict(r) for r in db.q("SELECT policy, COUNT(*) n, ROUND(100*AVG(reward), 2) mean_points FROM episodes WHERE purpose LIKE 'tbpub%' GROUP BY policy")]

    # 7. pass@k on the 40-task subset
    pk = {}
    for pol in ("M0", mf):
        d = per_task(db, pol, ["heldout:passk"])
        counts = [(len(v), sum(1 for x in v if x >= 1.0)) for v in d.values()]
        if counts: pk[pol] = {f"pass@{k}": round(100 * pass_at_k(counts, k), 2) for k in (1, 2, 4, 8, 16)} | {"tasks": len(counts)}
    res["pass_at_k_subset"] = pk or {"status": "pending"}

    # 8. control set
    ev = {r["kind"]: json.loads(r["payload"]) for r in db.q("SELECT kind, payload FROM events WHERE kind IN ('ablation_rft_vs_pg','control_random_reward') ORDER BY id")}
    res["controls"] = {
        "rft_vs_pg": ev.get("ablation_rft_vs_pg", {"status": "pending"}),
        "random_reward": ev.get("control_random_reward", {"status": "pending"}),
    }
    rr = means(per_task(db, "controlRR0", ["heldout:control"]))
    if rr:
        # pair against the closing baseline when it exists: the opening one predates several days of session drift
        base_rr = means(per_task(db, "M0", ["heldout:end-%"])) or means(per_task(db, "M0", ["heldout:start", "heldout-supp:start"]))
        res["controls"]["random_reward_heldout_vs_M0"] = paired_bootstrap(base_rr, rr)

    # 9. proposer yield per iteration: does the task writer get better or worse (the L1 question)
    def metric_by_it(name: str) -> dict[int, float]:
        return {r["iteration"]: r["value"] for r in db.q("SELECT iteration, value FROM metrics WHERE name=? AND step IN ('PROPOSE','GATE')", name)}
    parsed, gated, repaired, dedup = (metric_by_it(n) for n in ("propose_parsed", "gate_survivors", "gate_repaired_survivors", "gate_dedup_rejects"))
    yields = []
    for k in sorted(parsed):
        n, g, rp = parsed[k], gated.get(k, 0), repaired.get(k, 0)
        row = {"iteration": k, "parsed": int(n), "gated": int(g), "gated_pct": round(100 * g / n, 1) if n else None,
               "raw_gated": int(g - rp), "raw_pct": round(100 * (g - rp) / n, 1) if n else None,
               "repair_rescued": int(rp), "dedup_rejects": int(dedup.get(k, 0))}
        yields.append(row)
    if len(yields) >= 2:
        a, b = yields[0], yields[-1]
        pa, pb = a["raw_gated"] / a["parsed"], b["raw_gated"] / b["parsed"]
        se = math.sqrt(pa * (1 - pa) / a["parsed"] + pb * (1 - pb) / b["parsed"])
        res["proposer_yield_trend"] = {"first": a["iteration"], "last": b["iteration"], "raw_pct_change": round(100 * (pb - pa), 2),
                                       "se_points": round(100 * se, 2), "z": round((pb - pa) / se, 2) if se else None}
    res["proposer_yield"] = yields

    # 9b. curriculum difficulty: for the tasks written in each iteration, how the policy scored on them when new
    curric = []
    for r in db.q("""SELECT t.iteration_added k, e.task_id, AVG(e.reward) r FROM episodes e JOIN tasks t ON t.task_id=e.task_id
                     WHERE e.purpose='rollout' AND e.iteration=t.iteration_added AND e.exception IS NULL GROUP BY e.task_id"""):
        curric.append((r["k"], r["r"]))
    by_k: dict[int, list] = {}
    for k, v in curric: by_k.setdefault(k, []).append(v)
    res["curriculum_difficulty"] = [
        {"iteration": k, "new_tasks": len(v), "mean_pass": round(100 * sum(v) / len(v), 1),
         "always_solved": sum(1 for x in v if x >= 0.999), "never_solved": sum(1 for x in v if x <= 0.001),
         "in_band": sum(1 for x in v if 0.001 < x < 0.999),
         "in_band_pct": round(100 * sum(1 for x in v if 0.001 < x < 0.999) / len(v), 1)}
        for k, v in sorted(by_k.items())]

    # 10. in-distribution pool (never the headline) and integrity
    res["pool_rollouts"] = [dict(r) for r in db.q("SELECT iteration, COUNT(*) n, ROUND(100*AVG(reward), 2) mean_points, SUM(quarantined) quarantined FROM episodes WHERE purpose='rollout' GROUP BY iteration ORDER BY iteration")]
    res["task_pool"] = [dict(r) for r in db.q("SELECT status, COUNT(*) n FROM tasks GROUP BY status ORDER BY n DESC")]
    res["episode_accounting"] = episode_accounting(db)
    res["cost"] = cost(db)
    return res

# ---------------------------------------------------------------- rendering
def fmt_test(t: dict) -> str:
    if "diff_points" not in t: return "pending"
    return (f"{t['diff_points']:+.2f} points (from {t['mean_a']:.2f} to {t['mean_b']:.2f}, n={t['n']} tasks), "
            f"SE {t['se_points']:.2f}, 95% CI [{t['ci95_points'][0]:+.2f}, {t['ci95_points'][1]:+.2f}], one-sided p={t['p_one_sided_le0']:.4f}")

def render(res: dict) -> str:
    L = [f"# Results: run {res['run']} ({res['lineage_head']})", "", f"Generated {res['generated']}. Analysis pre-registered in `docs/07`.", ""]
    L += ["## Primary endpoint: private held-out union, same session", "", fmt_test(res["primary_heldout_union"]), ""]
    L += ["## Key secondary: recursion test vs the non-recursive branch", "", fmt_test(res["secondary_recursion"]), ""]
    h = res.get("holm", {})
    if "primary_heldout_union" in h:
        L += ["Holm-Bonferroni at alpha 0.05:", ""] + [f"- {k}: p={v['p']:.4f} against {v['threshold']:.4f}, {'significant' if v['significant'] else 'not significant'}" for k, v in h.items()] + [""]
    L += ["## Negative control: M_0 drift between sessions", "", fmt_test(res["negative_control_M0_drift"]), ""]
    if res.get("interim_heldout_vs_M0_start"):
        L += ["## Interim held-out reads against the M_0 baseline (descriptive)", "", "| policy | n | M_0 | policy | diff | SE |", "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {r['policy']} | {r['n']} | {r['mean_a']:.2f} | {r['mean_b']:.2f} | {r['diff_points']:+.2f} | {r['se_points']:.2f} |" for r in res["interim_heldout_vs_M0_start"] if r.get("diff_points") is not None] + [""]
    L += ["## Dev curve", "", "| policy | episodes | pass rate (points) |", "| --- | --- | --- |"]
    L += [f"| {r['policy']} | {r['n']} | {r['mean_points']} |" for r in res["dev_curve"]] + [""]
    if res["canaries"]["series"]:
        L += ["## Canaries (IFEval + GSM8K composite)", "", "| policy | composite | delta vs M_0 | gate |", "| --- | --- | --- | --- |"]
        L += [f"| {c['policy']} | {c['composite_points']:.2f} | {c['delta_vs_M0_points']:+.2f} | {'pass' if c['passes_gate'] else 'FAIL'} |" for c in res["canaries"]["series"]] + [""]
    L += ["## TerminalWorld-verified (out of genre)", "", fmt_test(res["terminalworld"]), ""]
    if isinstance(res["pass_at_k_subset"], dict) and "status" not in res["pass_at_k_subset"]:
        L += ["## pass@k on the 40-task subset", "", "| policy | " + " | ".join(f"pass@{k}" for k in (1, 2, 4, 8, 16)) + " |", "| --- |" + " --- |" * 5]
        L += [f"| {p} | " + " | ".join(str(v[f'pass@{k}']) for k in (1, 2, 4, 8, 16)) + " |" for p, v in res["pass_at_k_subset"].items()] + [""]
    ab = res["controls"]["rft_vs_pg"].get("paired_pg_minus_rft")
    if ab: L += ["## Controls", "", f"- RFT vs PG at iteration 0 (dev): PG minus RFT {100 * ab['diff']:+.2f} points, SE {100 * ab['se']:.2f}, n={ab['n']}."]
    rr = res["controls"].get("random_reward_heldout_vs_M0")
    if rr and "diff_points" in rr: L += [f"- Random-reward control on the held-out union: {fmt_test(rr)}."]
    if res.get("curriculum_difficulty"):
        L += ["## Curriculum difficulty (newly written tasks, scored when new)", "", "| iteration | new tasks | mean pass | always solved | never solved | in band | in band % |", "| --- | --- | --- | --- | --- | --- | --- |"]
        L += [f"| {r['iteration']} | {r['new_tasks']} | {r['mean_pass']} | {r['always_solved']} | {r['never_solved']} | {r['in_band']} | {r['in_band_pct']} |" for r in res["curriculum_difficulty"]] + [""]
    if res.get("proposer_yield"):
        L += ["## Proposer yield (does the task writer improve?)", "", "| iteration | parsed | gated | gated % | gated before repair | % | dedup rejects |", "| --- | --- | --- | --- | --- | --- | --- |"]
        L += [f"| {r['iteration']} | {r['parsed']} | {r['gated']} | {r['gated_pct']} | {r['raw_gated']} | {r['raw_pct']} | {r['dedup_rejects']} |" for r in res["proposer_yield"]]
        t = res.get("proposer_yield_trend")
        if t: L += ["", f"Change in the before-repair rate from iteration {t['first']} to {t['last']}: {t['raw_pct_change']:+.2f} points, SE {t['se_points']}, z {t['z']}."]
        L += [""]
    L += ["", "## Pool (in distribution, never the headline)", "", "| iteration | episodes | pass rate | quarantined |", "| --- | --- | --- | --- |"]
    L += [f"| {r['iteration']} | {r['n']} | {r['mean_points']} | {r['quarantined']} |" for r in res["pool_rollouts"]]
    t = res["episode_accounting"]["totals"]
    L += ["", "## Accounting", "", f"- Episodes {t['n']}, infrastructure errors {t['errors']}, agent timeouts scored 0: {t['timeouts']}, quarantined {t['quarantined']}.",
          f"- GPU hours by step: {res['cost']['gpu_hours_by_step']} (total {res['cost']['gpu_hours_total']} h).", ""]
    return "\n".join(L)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main"); ap.add_argument("--control-run", default="main-control")
    ap.add_argument("--out"); a = ap.parse_args()
    res = analyse(a.run, a.control_run)
    outdir = Path(a.out) if a.out else C.run_paths(a.run).root / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (outdir / "results.md").write_text(render(res))
    print(render(res))
    print(f"\nwrote {outdir}/results.json and results.md")
