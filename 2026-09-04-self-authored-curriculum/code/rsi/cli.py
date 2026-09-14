"""rsi command line: seed import, gating, pool stats. Run with venvs/loop/bin/python -m rsi.cli <cmd> ..."""
from __future__ import annotations
import argparse, json, random, sys, tomllib
from pathlib import Path

from rsi import config as C
from rsi.db import DB
from rsi.tasks.seeds import import_task
from rsi.tasks.gate import gate_tasks

SETA = C.DATASETS_DIR / "SETA-Env"
NEMO = C.DATASETS_DIR / "nemotron-tasks"   # extracted tarballs live here

def _seta_candidates(subset: str) -> list[Path]:
    return sorted(p for p in (SETA / subset).iterdir() if p.is_dir() and (p / "task.toml").exists())

def _nemo_candidates() -> list[Path]:
    if not NEMO.exists(): return []
    return sorted(p for p in NEMO.rglob("task.toml") for p in [p.parent])

def cmd_import_seeds(a):
    run = C.run_paths(a.run); run.mkdirs(); db = DB(run.db)
    rng = random.Random(a.seed)
    picks: list[tuple[Path, str, str]] = []
    if a.seta_evolve: picks += [(p, "seta_evolve", "Apache-2.0") for p in rng.sample(_seta_candidates("SETA_Evolve"), a.seta_evolve)]
    if a.seta_synth: picks += [(p, "seta_synth", "Apache-2.0") for p in rng.sample(_seta_candidates("SETA_Synth"), a.seta_synth)]
    if a.nemotron:
        c = _nemo_candidates(); picks += [(p, "nemotron", "CC-BY-4.0") for p in rng.sample(c, min(a.nemotron, len(c)))]
    imported = []
    for src, source, lic in picks:
        tid, dest, info = import_task(src, run.tasks, source, licence=lic, force=a.force)
        if info.get("skipped"): continue
        meta = tomllib.loads((dest / "task.toml").read_text()).get("metadata", {})
        db.upsert_task(tid, f"rsi/{tid}", dest, source, domain=meta.get("domain"), tier=meta.get("tier"), iteration_added=0,
                       status="imported", provenance=json.loads((dest / "provenance.json").read_text()))
        imported.append(dest)
        print(f"imported {tid:60s} test_sh={info['test_sh']:14s} solution={info['has_solution']} dropped={len(info['dropped_docker_lines'])}")
    print(f"{len(imported)} tasks imported into {run.tasks}")
    if a.gate and imported:
        v = gate_tasks(db, run, imported, iteration=0, n_concurrent=a.concurrency, batch_tag=a.tag)
        _print_verdicts(v)

def cmd_gate(a):
    run = C.run_paths(a.run); db = DB(run.db)
    where = "status IN ('imported','candidate')" + (" OR status LIKE 'rejected:%'" if a.include_rejected else "")
    dirs = [Path(r["path"]) for r in db.q(f"SELECT path FROM tasks WHERE ({where})" + (" AND source=?" if a.source else ""), *([a.source] if a.source else []))]
    if a.limit: dirs = dirs[: a.limit]
    print(f"gating {len(dirs)} tasks")
    v = gate_tasks(db, run, dirs, iteration=a.iteration, n_concurrent=a.concurrency, batch_tag=a.tag)
    _print_verdicts(v)
    if a.repair:
        # same executable-feedback repair round as the loop's GATE step (needs the policy served on :8000)
        from rsi.improver.propose import PolicyAuthor, feedback_for, repair_task
        author = PolicyAuthor(C.VLLM_BASE_URL, C.SERVED_MODEL_NAME)
        failed = [td for td in dirs if v.get(td.name, "").startswith("rejected:") and "oracle" in v[td.name]]
        jobs = [run.jobs / f"{a.tag}-oracle", run.jobs / f"{a.tag}-oracle-root"]
        from concurrent.futures import ThreadPoolExecutor
        def one(td):
            return td, repair_task(author, td, feedback_for(td, jobs), run.tasks, iteration=a.iteration, source="policy", log=lambda m: None)
        repaired = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            for td, r in ex.map(one, failed):
                if r:
                    tid, dest = r
                    db.upsert_task(tid, f"rsi/{tid}", dest, "policy", iteration_added=a.iteration, status="candidate",
                                   provenance=json.loads((dest / "provenance.json").read_text()))
                    repaired.append(dest)
        print(f"repair round: {len(repaired)}/{len(failed)} rewritten; gating them")
        if repaired:
            v2 = gate_tasks(db, run, repaired, iteration=a.iteration, n_concurrent=a.concurrency, batch_tag=a.tag + "-repair")
            _print_verdicts(v2)

def _print_verdicts(v: dict):
    from collections import Counter
    c = Counter(x if x == "gated" else ":".join(x.split(":")[:2]) for x in v.values())
    print("verdicts:", dict(c))
    for tid, x in sorted(v.items()):
        if x != "gated": print(f"  {tid:60s} {x}")

def cmd_repair(a):
    """Executable-feedback repair round for rejected candidates of a source (needs the policy served on :8000)."""
    from rsi.improver.propose import PolicyAuthor, feedback_for, repair_task
    from concurrent.futures import ThreadPoolExecutor
    run = C.run_paths(a.run); db = DB(run.db)
    rows = db.q("SELECT task_id, path, status FROM tasks WHERE source=? AND status LIKE 'rejected:%' AND status LIKE '%oracle%'", a.source)
    dirs = [Path(r["path"]) for r in rows][: a.limit or None]
    jobs = [run.jobs / f"{a.tag}-oracle", run.jobs / f"{a.tag}-oracle-root"]
    author = PolicyAuthor(C.VLLM_BASE_URL, C.SERVED_MODEL_NAME)
    print(f"repairing {len(dirs)} rejected candidates from {a.source} (gate tag {a.tag})", flush=True)
    def one(td): return td, repair_task(author, td, feedback_for(td, jobs), run.tasks, iteration=a.iteration, source="policy", log=lambda m: None)
    repaired = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for td, r in ex.map(one, dirs):
            if r:
                tid, dest = r
                db.upsert_task(tid, f"rsi/{tid}", dest, a.source, iteration_added=a.iteration, status="candidate",
                               provenance=json.loads((dest / "provenance.json").read_text()))
                repaired.append(dest)
    print(f"repair: {len(repaired)}/{len(dirs)} rewritten", flush=True)
    if repaired and not a.no_gate:
        v = gate_tasks(db, run, repaired, iteration=a.iteration, n_concurrent=a.concurrency, batch_tag=a.tag + "-repair")
        _print_verdicts(v)
        print("repaired survivors:", sum(1 for x in v.values() if x.startswith("gated")), "of", len(repaired))

def cmd_pool(a):
    run = C.run_paths(a.run); db = DB(run.db)
    for r in db.q("SELECT source, status, COUNT(*) n FROM tasks GROUP BY source, status ORDER BY source, status"):
        print(f"{r['source'] or '-':14s} {r['status']:40s} {r['n']}")
    for r in db.q("SELECT purpose, COUNT(*) n, AVG(reward) mean_reward, SUM(exception IS NOT NULL) errors FROM episodes GROUP BY purpose"):
        print(f"episodes {r['purpose']:22s} n={r['n']:4d} mean_reward={r['mean_reward'] if r['mean_reward'] is None else round(r['mean_reward'],3)} errors={r['errors']}")

def cmd_loop(a):
    from rsi.loop import Loop, LoopConfig, STEPS
    cfg = LoopConfig(run=a.run)
    if a.config:
        cfg = LoopConfig(**{**cfg.__dict__, **{k: v for k, v in json.loads(Path(a.config).read_text()).items() if not k.startswith("_")}})
    for kv in a.set or []:
        key, val = kv.split("=", 1)
        cur = getattr(cfg, key)
        if isinstance(cur, bool): setattr(cfg, key, val.lower() in ("1", "true", "yes"))
        elif cur is None or isinstance(cur, str): setattr(cfg, key, val)
        elif isinstance(cur, (tuple, list)): setattr(cfg, key, type(cur)(json.loads(val)))
        else: setattr(cfg, key, type(cur)(val))
    lp = Loop(cfg)
    if a.action in ("run", "resume"):
        lp.unpause(); res = lp.run_loop(until_iteration=a.until); print(res)
        # a failed step must not exit 0: the orchestrator would march past an incomplete iteration into the
        # final evaluations. Exiting non-zero stops it and leaves the restart to the watchdog, which is bounded.
        if res == "failed": raise SystemExit(1)
    elif a.action == "pause": lp.pause()
    elif a.action == "rerun": lp.rerun(a.step, a.iteration)
    elif a.action == "status": print(json.dumps(lp.state, indent=1))
    elif a.action == "branch": lp.branch(a.to, a.iteration); print("branched")
    elif a.action == "inspect":
        from rsi.inspect_tools import print_trajectory, job_trials
        k = lp.state["iteration"] if a.iteration is None else a.iteration
        print(json.dumps(lp.state, indent=1))
        for r in lp.db.q("SELECT k, step, status, notes FROM iterations WHERE k=? ORDER BY started_at", k): print(dict(r))
        for r in lp.db.q("SELECT status, COUNT(*) n FROM tasks WHERE iteration_added=? GROUP BY status", k): print(" tasks:", dict(r))
        for r in lp.db.q("SELECT COUNT(*) n, AVG(reward) mean_reward, SUM(quarantined) q FROM episodes WHERE purpose='rollout' AND iteration=?", k): print(" rollout:", dict(r))
        job = lp.run.jobs / f"it{k:02d}-rollout"
        if job.exists():
            import random as _r
            trials = job_trials(job); _r.Random(a.seed).shuffle(trials)
            for t in trials[: a.n]: print_trajectory(t, show_reasoning=a.reasoning)
        from rsi.dashboard import build_dashboard; print("dashboard:", build_dashboard(lp.run, lp.db))

def cmd_heldout(a):
    from rsi import heldout as H
    run = C.run_paths(a.run); run.mkdirs(); db = DB(run.db)
    if a.action == "author":
        for cli in a.cli:
            rows = H.author(run, db, cli, a.n, a.seed, log=lambda m: print(f"[{cli}] {m}", flush=True), tiers=a.tiers.split(",") if a.tiers else None, supplementary=a.supplementary)
            print(cli, "parsed", sum(1 for r in rows if r.get("parsed")), "/", len(rows))
    elif a.action == "gate": _print_verdicts(H.gate(run, db, a.concurrency))
    elif a.action == "finish": print(json.dumps(H.finish_gate(run, db, repair=not a.no_repair, gate_repaired=not a.no_gate, concurrency=a.concurrency), indent=1))
    elif a.action == "tier": print(json.dumps({k: v for k, v in list(H.reference_tier(run, db, a.runs, a.concurrency, resume=a.resume).items())[:20]}, indent=1))
    elif a.action == "freeze": print(json.dumps(H.freeze(run, db, a.target, a.seed), indent=1))
    elif a.action == "devset": print(json.dumps(H.make_devset(run, db, a.n, a.seed, a.concurrency), indent=1))
    elif a.action == "gate-supp": print(json.dumps(H.gate_supplementary(run, db, a.concurrency), indent=1))
    elif a.action == "freeze-supp": print(json.dumps(H.freeze_supplementary(run, db), indent=1))

def cmd_tw(a):
    from rsi.tasks.terminalworld import import_all
    run = C.run_paths(a.run); run.mkdirs(); db = DB(run.db)
    print(json.dumps(import_all(run, db, a.limit, a.force), indent=1))
    if a.gate:
        dirs = [Path(r["path"]) for r in db.q("SELECT path FROM tasks WHERE status='tw'")]
        v = gate_tasks(db, run, dirs, iteration=-1, n_concurrent=a.concurrency, batch_tag="tw-gate", skip_checks={"from_base", "canary_guid"})
        _print_verdicts(v)
        for tid, x in v.items():
            with db.tx() as c:
                c.execute("UPDATE tasks SET status=? WHERE task_id=?", ("tw" if x.startswith("gated") else "tw-" + x, tid))

def cmd_eval(a):
    """Evaluation runs outside the loop: held-out / TerminalWorld / dev / canaries for a given served policy."""
    from rsi.loop import Loop, LoopConfig
    from rsi.evaluate import eval_set
    cfg = LoopConfig(run=a.run, **({} if not a.config else {k: v for k, v in json.loads(Path(a.config).read_text()).items() if not k.startswith("_")}))
    cfg.rollout_concurrency = a.concurrency
    lp = Loop(cfg)
    if a.set == "canaries":
        from rsi.canary import run_canaries
        if not a.no_serve: print("serving policy", a.policy, "->", lp.serve_policy(a.policy), flush=True)
        print(json.dumps(run_canaries(lp.run, a.policy, limit=a.limit), indent=1)); return
    statuses = {"heldout": ("heldout", "heldout-supp"), "heldout-main": ("heldout",), "heldout-supp": ("heldout-supp",), "tw": ("tw",), "dev": ("dev",), "tb": ("tb",), "tbpub": ("tbpub",)}[a.set]
    if not a.no_serve:
        print("serving policy", a.policy, "->", lp.serve_policy(a.policy), flush=True)
    dirs = [Path(r["path"]) for st in statuses for r in lp.db.q("SELECT path FROM tasks WHERE status=? ORDER BY task_id", st)]
    if a.limit: dirs = dirs[: a.limit]
    print(f"{a.set}: {len(dirs)} tasks x {a.seeds} seeds with policy {a.policy}")
    st = eval_set(lp, dirs, policy_tag=a.policy, purpose=f"{a.set}:{a.tag}" if a.tag else a.set, seeds=a.seeds, job_name=f"{a.set}-{a.policy}-{a.tag or 'x'}", iteration=a.iteration)
    print(json.dumps(st, indent=1))

def cmd_probe(a):
    """Memorisation probe (docs/02): paraphrase -> measure on the base model -> report."""
    from rsi import probe, serve
    from rsi.loop import LoopConfig
    if a.stage in ("measure", "all") and a.serve:
        cfg = LoopConfig(run=a.run, **({} if not a.config else {k: v for k, v in json.loads(Path(a.config).read_text()).items() if not k.startswith("_")}))
        run = C.run_paths(a.run)
        # prompt_logprobs (Min-K%) is rejected by vLLM when prefix caching is on, so the probe serves without it
        serve.stop(); serve.start(cfg.base_model, max_num_seqs=cfg.max_num_seqs, log_path=run.logs / "vllm-probe.log",
                                  dtype=cfg.serve_dtype, prefix_caching=False)
        print(f"serving the base model {cfg.base_model} ({cfg.serve_dtype}) for the probe: up in {serve.wait_healthy():.0f}s", flush=True)
    print(json.dumps(probe.run_stage(a.run, a.stage, workers=a.workers, cli=a.cli), indent=1, default=str))

def cmd_tb(a):
    from rsi.tasks.terminalworld import import_terminal_bench, import_terminal_bench_as_published
    run = C.run_paths(a.run); run.mkdirs(); db = DB(run.db)
    if a.published:
        print(json.dumps(import_terminal_bench_as_published(run, db, Path(a.src), a.limit, a.force), indent=1)); return
    print(json.dumps(import_terminal_bench(run, db, Path(a.src), a.limit, a.force), indent=1))
    if a.gate:
        dirs = [Path(r["path"]) for r in db.q("SELECT path FROM tasks WHERE status='tb'")]
        v = gate_tasks(db, run, dirs, iteration=-1, n_concurrent=a.concurrency, batch_tag="tb-gate", skip_checks={"from_base", "canary_guid"})
        _print_verdicts(v)
        for tid, x in v.items():
            with db.tx() as c:
                c.execute("UPDATE tasks SET status=? WHERE task_id=?", ("tb" if x.startswith("gated") else "tb-" + x, tid))

def cmd_control(a):
    from rsi.loop import Loop, LoopConfig
    from rsi import controls
    cfg = LoopConfig(run=a.run, **({} if not a.config else {k: v for k, v in json.loads(Path(a.config).read_text()).items() if not k.startswith("_")}))
    cfg.rollout_concurrency = a.concurrency
    lp = Loop(cfg)
    if a.which == "rft-vs-pg": print(json.dumps(controls.rft_vs_pg(lp, a.iteration, log=lp._log), indent=1, default=str))
    elif a.which == "random-reward": print(json.dumps(controls.random_reward(lp, a.iteration, log=lp._log), indent=1, default=str))

def main(argv=None):
    ap = argparse.ArgumentParser(prog="rsi")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import-seeds"); p.add_argument("--run", required=True); p.add_argument("--seta-evolve", type=int, default=0)
    p.add_argument("--seta-synth", type=int, default=0); p.add_argument("--nemotron", type=int, default=0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate", action="store_true"); p.add_argument("--concurrency", type=int, default=4); p.add_argument("--tag", default="gate0")
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_import_seeds)
    p = sub.add_parser("gate"); p.add_argument("--run", required=True); p.add_argument("--source"); p.add_argument("--limit", type=int)
    p.add_argument("--iteration", type=int, default=0); p.add_argument("--concurrency", type=int, default=4); p.add_argument("--tag", default="gate")
    p.add_argument("--include-rejected", action="store_true"); p.add_argument("--repair", action="store_true"); p.set_defaults(fn=cmd_gate)
    p = sub.add_parser("pool"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_pool)
    p = sub.add_parser("repair"); p.add_argument("--run", required=True); p.add_argument("--source", default="proposer:policy"); p.add_argument("--tag", required=True)
    p.add_argument("--iteration", type=int, default=0); p.add_argument("--workers", type=int, default=16); p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int); p.add_argument("--no-gate", action="store_true"); p.set_defaults(fn=cmd_repair)
    p = sub.add_parser("loop"); p.add_argument("action", choices=["run", "pause", "resume", "rerun", "status", "inspect", "branch"]); p.add_argument("--run", required=True); p.add_argument("--to")
    p.add_argument("--config"); p.add_argument("--set", action="append"); p.add_argument("--until", type=int); p.add_argument("--step"); p.add_argument("--iteration", type=int)
    p.add_argument("--n", type=int, default=3); p.add_argument("--seed", type=int, default=0); p.add_argument("--reasoning", action="store_true"); p.set_defaults(fn=cmd_loop)
    p = sub.add_parser("heldout"); p.add_argument("action", choices=["author", "gate", "finish", "tier", "freeze", "devset", "gate-supp", "freeze-supp"]); p.add_argument("--run", required=True); p.add_argument("--no-repair", action="store_true"); p.add_argument("--no-gate", action="store_true"); p.add_argument("--resume", action="store_true"); p.add_argument("--tiers"); p.add_argument("--supplementary", action="store_true")
    p.add_argument("--cli", nargs="+", default=["kimi", "grok", "agy"]); p.add_argument("--n", type=int, default=100); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=6); p.add_argument("--runs", type=int, default=8); p.add_argument("--target", type=int, default=200); p.set_defaults(fn=cmd_heldout)
    p = sub.add_parser("tw"); p.add_argument("--run", required=True); p.add_argument("--limit", type=int); p.add_argument("--force", action="store_true")
    p.add_argument("--gate", action="store_true"); p.add_argument("--concurrency", type=int, default=6); p.set_defaults(fn=cmd_tw)
    p = sub.add_parser("eval"); p.add_argument("set", choices=["heldout", "heldout-main", "heldout-supp", "tw", "dev", "tb", "tbpub", "canaries"]); p.add_argument("--run", required=True); p.add_argument("--policy", required=True)
    p.add_argument("--seeds", type=int, default=5); p.add_argument("--limit", type=int); p.add_argument("--tag", default=""); p.add_argument("--iteration", type=int); p.add_argument("--config")
    p.add_argument("--concurrency", type=int, default=24); p.add_argument("--no-serve", action="store_true", help="use whatever is already on :8000"); p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("probe"); p.add_argument("--run", required=True); p.add_argument("--stage", default="all", choices=["collect", "paraphrase", "measure", "report", "all"])
    p.add_argument("--workers", type=int, default=8); p.add_argument("--cli", default="kimi"); p.add_argument("--config")
    p.add_argument("--no-serve", dest="serve", action="store_false", help="measure against whatever is already served")
    p.set_defaults(fn=cmd_probe, serve=True)
    p = sub.add_parser("tb"); p.add_argument("--run", required=True); p.add_argument("--src", default="${RSI_DATASETS_DIR}/terminal-bench-2.0/terminal-bench")
    p.add_argument("--limit", type=int); p.add_argument("--force", action="store_true"); p.add_argument("--gate", action="store_true"); p.add_argument("--concurrency", type=int, default=6); p.add_argument("--published", action="store_true"); p.set_defaults(fn=cmd_tb)
    p = sub.add_parser("control"); p.add_argument("which", choices=["rft-vs-pg", "random-reward"]); p.add_argument("--run", required=True); p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--config"); p.add_argument("--concurrency", type=int, default=24); p.set_defaults(fn=cmd_control)
    a = ap.parse_args(argv); a.fn(a)

if __name__ == "__main__":
    main()
