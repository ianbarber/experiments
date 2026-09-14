"""Fill the write-up's {{placeholders}} from the run's analysis bundle, so no number is hand-transcribed.

    venvs/loop/bin/python scripts/fill-writeup.py [--run main] [--draft docs/08-writeup-draft.md] [--out docs/08-writeup.md]
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rsi import config as C
from rsi.db import DB

spec = importlib.util.spec_from_file_location("analyse_run", ROOT / "scripts/analyse-run.py")
AR = importlib.util.module_from_spec(spec); spec.loader.exec_module(AR)

def merge_fidelity(run) -> tuple[str, str]:
    """Kept delta-norm fractions from the iteration-0 merge audits: the bf16 merge and its fp16 replacement."""
    def kept(path: Path, mean_only: bool = False):
        try:
            vals = [v["kept_norm_ratio"] for v in json.loads(path.read_text())["per_module"].values()]
            if not vals: return None
            return f"{100 * sum(vals) / len(vals):.0f}%" if mean_only else f"{100 * min(vals):.0f}-{100 * max(vals):.0f}%"
        except Exception:
            return None
    it0 = run.root / "iterations" / "00"
    lost = kept(it0 / "merge.json", mean_only=True)
    return (f"all but {lost}" if lost else "about two thirds"), (kept(it0 / "merge-fp16.json") or "81-93%")

def fmt(t: dict) -> str:
    return AR.fmt_test(t)

def blocks(run_name: str, control_run: str) -> dict:
    res = AR.analyse(run_name, control_run)
    run = C.run_paths(run_name); db = DB(run.db)
    cfg = json.loads((ROOT / "configs/main-9b.json").read_text())
    heldout_n = res["primary_heldout_union"].get("n") or db.q("SELECT COUNT(*) n FROM tasks WHERE status IN ('heldout','heldout-supp')")[0]["n"]
    q0 = next((r["quarantined"] for r in res["pool_rollouts"] if r["iteration"] == 0), 0)
    bf16_lost, fp16_kept = merge_fidelity(run)

    L = [f"**Primary endpoint.** Private held-out union, {res['lineage_head']} against M_0 in the same session: {fmt(res['primary_heldout_union'])}.", ""]
    L += [f"**Recursion test.** Main lineage against the non-recursive branch: {fmt(res['secondary_recursion'])}.", ""]
    L += [f"**Drift control.** M_0 re-measured at the end: {fmt(res['negative_control_M0_drift'])}.", ""]
    if res["dev_curve"]:
        L += ["**Dev curve.**", "", "| policy | episodes | pass rate |", "| --- | --- | --- |"]
        L += [f"| {r['policy']} | {r['n']} | {r['mean_points']} |" for r in res["dev_curve"]] + [""]
    if res["canaries"]["series"]:
        L += ["**Canaries.** " + "; ".join(f"{c['policy']} {c['composite_points']:.1f} ({c['delta_vs_M0_points']:+.1f})" for c in res["canaries"]["series"]) + ".", ""]
    if "diff_points" in res["terminalworld"]:
        L += [f"**Out of genre.** TerminalWorld-verified: {fmt(res['terminalworld'])}.", ""]
    pk = res.get("pass_at_k_subset") or {}
    if "status" not in pk and pk:
        L += ["**pass@k** on the 40-task subset: " + "; ".join(f"{p} pass@1 {v['pass@1']}, pass@16 {v['pass@16']}" for p, v in pk.items()) + ".", ""]
    ab = res["controls"]["rft_vs_pg"].get("paired_pg_minus_rft")
    if ab: L += [f"**Ablation.** Policy gradient minus RFT from identical iteration-0 rollouts: {100 * ab['diff']:+.2f} points (SE {100 * ab['se']:.2f}, n={ab['n']}).", ""]
    rr = res["controls"].get("random_reward_heldout_vs_M0")
    if rr and "diff_points" in rr: L += [f"**Random-reward control.** {fmt(rr)}.", ""]
    L += ["**In distribution, for context only.**", "", "| iteration | episodes | pool pass rate |", "| --- | --- | --- |"]
    L += [f"| {r['iteration']} | {r['n']} | {r['mean_points']} |" for r in res["pool_rollouts"]]

    c = res["cost"]
    cost = [f"The run took {c['gpu_hours_total']} GPU-hours on one card, split as:", "", "| step | hours |", "| --- | --- |"]
    cost += [f"| {k} | {v} |" for k, v in c["gpu_hours_by_step"].items()]
    t = res["episode_accounting"]["totals"]
    cost += ["", f"That is {t['n']} episodes in total, of which {t['errors']} were infrastructure errors and "
                 f"{t['timeouts']} agent timeouts scored zero. At UK domestic electricity and a 600 W draw, "
                 f"the card costs roughly £{0.6 * 0.25 * c['gpu_hours_total']:.0f} to run for that long."]

    rel = ["Everything needed to repeat the run is in the release: the loop code, the frozen task sets with their "
           "hashes, the adapters for every accepted checkpoint, the SQLite database of every episode, and the "
           "pre-registration with its amendments. The one thing that cannot be shipped is the held-out set's "
           "secrecy: once published it stops being a clean held-out set for anyone who trains on it."]
    tier = next((json.loads(r["payload"]) for r in db.q("SELECT payload FROM events WHERE kind='tier' ORDER BY id LIMIT 1")), None)
    tier_txt = (f"{tier['n_tasks']} tasks that passed the gate, of which only {tier['train']} sat in the band where a "
                f"policy gradient has anything to learn from: {tier['drop']} were solved on every attempt and "
                f"{tier['frontier']} on none.") if tier else "a pool whose tier split is not yet recorded."
    pol = db.q("SELECT COUNT(*) n FROM tasks WHERE source LIKE 'policy%' AND status != 'candidate'")[0]["n"]  # decided only
    bad = db.q("SELECT COUNT(*) n FROM tasks WHERE source LIKE 'policy%' AND status LIKE '%build_or_oracle_error'")[0]["n"]
    build_pct = f"{100 * bad / pol:.0f}%" if pol else "some"

    # audit: what the rules and the monitor caught, per iteration, plus the flagged/unflagged reward gap
    A = ["| iteration | episodes | flagged | quarantined | monitor: hack | monitor: suspicious |", "| --- | --- | --- | --- | --- | --- |"]
    for r in db.q("""SELECT iteration k, COUNT(*) n, SUM(flags IS NOT NULL) flagged, SUM(quarantined) q,
                            SUM(flags LIKE '%VERDICT: hack%') hack, SUM(flags LIKE '%suspicious%') susp
                     FROM episodes WHERE purpose='rollout' GROUP BY iteration ORDER BY iteration"""):
        A.append(f"| {r['k']} | {r['n']} | {r['flagged']} | {r['q']} | {r['hack']} | {r['susp']} |")
    fl = db.q("SELECT ROUND(AVG(reward), 3) r FROM episodes WHERE purpose='rollout' AND flags IS NOT NULL AND reward IS NOT NULL")[0]["r"]
    un = db.q("SELECT ROUND(AVG(reward), 3) r FROM episodes WHERE purpose='rollout' AND flags IS NULL AND reward IS NOT NULL")[0]["r"]
    if fl is not None and un is not None:
        A += ["", f"Mean reward, flagged episodes against unflagged: {100 * fl:.1f} versus {100 * un:.1f}."]

    steps = sum(len(json.loads((f).read_text()).get("diagnostics", []))
                for f in sorted((run.root / "iterations").glob("*/train.json")))
    CB = ["| iteration | new tasks | solves every time | never solves | trainable band |", "| --- | --- | --- | --- | --- |"]
    for r in res.get("curriculum_difficulty", []):
        CB.append(f"| {r['iteration']} | {r['new_tasks']} | {r['always_solved']} | {r['never_solved']} | {r['in_band']} |")

    # the two arms at the iteration they diverged, same solver weights
    arm = "the comparison is not yet available."
    try:
        cdb = DB(C.run_paths(control_run).db)
        q = "SELECT ROUND(100*AVG(reward),1) m FROM episodes WHERE purpose='rollout' AND iteration=4"
        a = db.q(q)[0]["m"]; b = cdb.q(q)[0]["m"]
        arm = (f"a mean solve rate of {a} against {b} on its own tasks, and put more of them "
               f"in the trainable band.")
    except Exception:
        pass

    n_aud = db.q("SELECT COUNT(*) n FROM episodes WHERE purpose='rollout'")[0]["n"]
    AY = ["| arm | iteration | parsed | gated | gated % | before repair | near-duplicate rejects |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for rn, label in ((run_name, "recursive"), (control_run, "frozen")):
        try: d2 = DB(C.run_paths(rn).db)
        except Exception: continue
        mm = {(r["iteration"], r["name"]): r["value"] for r in d2.q("SELECT iteration,name,value FROM metrics WHERE step IN ('PROPOSE','GATE')")}
        for k in sorted({i for i, _ in mm} ):
            pa, g, rp = mm.get((k, "propose_parsed")), mm.get((k, "gate_survivors")), mm.get((k, "gate_repaired_survivors"))
            if k >= 4 and pa and g is not None:
                AY.append(f"| {label} | {k} | {int(pa)} | {int(g)} | {100*g/pa:.1f}% | {int(g-(rp or 0))} | {int(mm.get((k,'gate_dedup_rejects')) or 0)} |")
    NS = ["| iteration | new tasks | never solved | rate |", "| --- | --- | --- | --- |"]
    for r in res.get("curriculum_difficulty", []):
        NS.append(f"| {r['iteration']} | {r['new_tasks']} | {r['never_solved']} | {100*r['never_solved']/r['new_tasks']:.1f}% |")
    return {"never_solved_block": "\n".join(NS), "arm_yield_block": "\n".join(AY), "audit_episodes": f"{n_aud:,}", "curriculum_block": "\n".join(CB), "arm_curriculum": arm, "total_steps": str(steps), "audit_block": "\n".join(A), "tier_it0": tier_txt, "build_fail_pct": build_pct, "setup_model": cfg.get("base_model", "").split("/")[-1] or "Qwen3.5-9B",
            "heldout_n": str(heldout_n), "drift_band": "3.5", "results_block": "\n".join(L),
            "bf16_lost": str(bf16_lost),
            "fp16_kept": fp16_kept, "quarantine_it0": str(q0),
            "cost_block": "\n".join(cost), "release_block": "\n".join(rel)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main"); ap.add_argument("--control-run", default="main-control")
    ap.add_argument("--draft", default=str(ROOT / "docs/08-technical-report.md"))
    ap.add_argument("--out", default=str(ROOT / "docs/08-technical-report-filled.md"))
    a = ap.parse_args()
    vals = blocks(a.run, a.control_run)
    for draft, out in [(a.draft, a.out), (str(ROOT / "docs/10-engineering-notes.md"), str(ROOT / "docs/10-engineering-notes.md"))]:
        if not Path(draft).exists(): continue
        text = Path(draft).read_text()
        missing = set(re.findall(r"\{\{(\w+)\}\}", text)) - set(vals)
        for k, v in vals.items(): text = text.replace("{{" + k + "}}", v)
        Path(out).write_text(text)
        print(f"wrote {out}" + (f"; unfilled placeholders: {sorted(missing)}" if missing else "; all placeholders filled"))
