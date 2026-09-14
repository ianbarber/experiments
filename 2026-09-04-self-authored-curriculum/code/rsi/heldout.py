"""Phase 0b: the private held-out set (docs/02, decided 2026-09-05): authored by kimi/grok/agy from the 9B's brief,
gated by the same executable checks, tiered by a reference model that is not the trainee, frozen with a sha256.
Also the dev set: SETA tasks disjoint from the seed pool, never trained on."""
from __future__ import annotations
import hashlib, json, random, time
from collections import Counter, defaultdict
from pathlib import Path
from rsi import config as C
from rsi.db import DB
from rsi.improver.propose import CliAuthor, propose_batch
from rsi.tasks.gate import gate_tasks
from rsi.rollout import run_job, ingest_job

REFERENCE = {"base_url": "http://dgx-spark:8888/v1", "model": "qwen3.8-27b-sglang"}
COMPOSITION = {"easy": 0.40, "medium": 0.26, "hard": 0.26, "extreme": 0.08}     # OpenThoughts-TBLite mix (docs/02)
BANDS = [("easy", 0.70), ("medium", 0.40), ("hard", 0.10), ("extreme", -1.0)]     # by reference pass rate over 8

SUPP_BRIEF = ("This task is for a SUPPLEMENTARY held-out set that must be harder than the first one: a competent engineer needs at "
              "least 8 commands; the task combines two or more skills (e.g. read code + fix + verify, or parse + query + report); "
              "inputs have realistic wrinkles; a single one-liner or one-file transformation is NOT acceptable.")

def author(run: C.RunPaths, db: DB, cli: str, n: int, seed: int, log=print, tiers: list[str] | None = None, supplementary: bool = False) -> list[dict]:
    pool = run.root / ("heldout-supp-candidates" if supplementary else "heldout-candidates"); pool.mkdir(parents=True, exist_ok=True)
    rows = propose_batch(C.PROJECT_DIR / "improver", pool, CliAuthor(cli), n=n, exemplar_pool=[], seed=seed, source=f"heldout:{cli}",
                         iteration=-1, heldout=True, log=log, tiers=tiers, extra_brief=SUPP_BRIEF if supplementary else "")
    for r in rows:
        if r.get("parsed"):
            db.upsert_task(r["task_id"], f"rsi/{r['task_id']}", r["path"], f"heldout:{cli}", domain=r["meta"]["domain"], tier=r["meta"]["tier"],
                           iteration_added=-1, status="heldout-supp-candidate" if supplementary else "heldout-candidate",
                           provenance=json.loads((Path(r["path"]) / "provenance.json").read_text()))
    db.event("heldout_author", cli=cli, n=n, parsed=sum(1 for r in rows if r.get("parsed")), supplementary=supplementary)
    return rows

def gate(run: C.RunPaths, db: DB, concurrency: int = 6, repair: bool = True, tag: str = "heldout-gate") -> dict:
    from rsi.improver.propose import feedback_for, repair_task
    cands = [Path(r["path"]) for r in db.q("SELECT path FROM tasks WHERE status='heldout-candidate'")]
    v = gate_tasks(db, run, cands, iteration=-1, n_concurrent=concurrency, batch_tag=tag)
    if repair:
        failed = [td for td in cands if v.get(td.name, "").startswith("rejected:") and "oracle" in v[td.name]]
        jobs = [run.jobs / f"{tag}-oracle", run.jobs / f"{tag}-oracle-root"]
        repaired = []
        from concurrent.futures import ThreadPoolExecutor
        sources = {td.name: (db.q("SELECT source FROM tasks WHERE task_id=?", td.name)[0]["source"] or "") for td in failed}   # main thread: sqlite is per-thread
        def one(td):
            src = sources[td.name]; cli = src.split(":")[-1]
            if cli not in CliAuthor.CMDS: return td, src, None
            return td, src, repair_task(CliAuthor(cli), td, feedback_for(td, jobs), td.parent, iteration=-1, source=src, log=print)
        with ThreadPoolExecutor(max_workers=4) as ex:      # two CLIs, two calls each in flight
            for td, src, r in ex.map(one, failed):
                if r:
                    tid, dest = r
                    db.upsert_task(tid, f"rsi/{tid}", dest, src, iteration_added=-1, status="heldout-candidate",
                                   provenance=json.loads((dest / "provenance.json").read_text()))
                    repaired.append(dest)
        print(f"repair round: {len(repaired)}/{len(failed)} rewritten")
        if repaired:
            v.update(gate_tasks(db, run, repaired, iteration=-1, n_concurrent=concurrency, batch_tag=tag + "-repair"))
    for tid, x in v.items():
        with db.tx() as c:
            c.execute("UPDATE tasks SET status=? WHERE task_id=?", ("heldout-gated" if x.startswith("gated") else "heldout-" + x, tid))
    return v

def reference_tier(run: C.RunPaths, db: DB, runs: int = 8, concurrency: int = 8, resume: bool = False) -> dict:
    """Run the LAN 27B `runs` times per gated candidate; assign the tier from its pass rate (never the trainee's).
    resume=True: ingest whatever an interrupted job left behind and only run the tasks that still lack `runs` results
    (new job dir per resume; run_job wipes an existing job dir)."""
    from rsi.rollout import ingest_job as _ingest, task_key
    tasks = [Path(r["path"]) for r in db.q("SELECT path FROM tasks WHERE status='heldout-gated'")]
    kwargs = {**REFERENCE, "policy_tag": "reference-27b", "thinking": "false", "logprobs": "false"}   # thinking off: the LAN server sustains ~90 tok/s in total
    if resume:
        done = {}
        for prev in sorted(run.jobs.glob("heldout-reference*")):
            if not prev.is_dir() or prev.name.startswith("_"): continue
            _ingest(db, prev, policy="reference-27b", purpose="heldout-reference", iteration=-1)
        for r in db.q("SELECT task_id, COUNT(*) n FROM episodes WHERE purpose='heldout-reference' AND exception IS NULL AND reward IS NOT NULL GROUP BY task_id"):
            done[r["task_id"]] = r["n"]
        todo = [t for t in tasks if done.get(t.name, 0) < runs]
        k = 2
        while (run.jobs / f"heldout-reference-{k}").exists(): k += 1
        print(f"resume: {len(tasks) - len(todo)} tasks complete, {len(todo)} to (re)run -> job heldout-reference-{k}", flush=True)
        if todo:
            job = run_job(run, todo, job_name=f"heldout-reference-{k}", agent="rsi", n_concurrent=concurrency, n_attempts=runs, agent_kwargs=kwargs)
            _ingest(db, job, policy="reference-27b", purpose="heldout-reference", iteration=-1)
    else:
        job = run_job(run, tasks, job_name="heldout-reference", agent="rsi", n_concurrent=concurrency, n_attempts=runs, agent_kwargs=kwargs)
        ingest_job(db, job, policy="reference-27b", purpose="heldout-reference", iteration=-1)
    by = defaultdict(list)
    for r in db.q("SELECT task_id, reward FROM episodes WHERE purpose='heldout-reference' AND exception IS NULL AND reward IS NOT NULL"):
        by[r["task_id"]].append(float(r["reward"]))
    tiers = {}
    for tid, rs in by.items():
        p = sum(rs) / len(rs)
        tier = next(t for t, cut in BANDS if p >= cut)
        tiers[tid] = (tier, p)
        with db.tx() as c: c.execute("UPDATE tasks SET tier=? WHERE task_id=?", (tier, tid))
    db.event("heldout_reference_tier", n=len(tiers), tiers=dict(Counter(t for t, _ in tiers.values())))
    return tiers

def freeze(run: C.RunPaths, db: DB, target: int = 200, seed: int = 0, max_domain_frac: float = 0.20) -> dict:
    """Select `target` tasks by the pre-registered composition (tier mix, domain cap), copy to <run>/heldout/, hash."""
    rng = random.Random(seed)
    rows = db.q("SELECT task_id, path, tier, domain, source FROM tasks WHERE status='heldout-gated' AND tier IS NOT NULL")
    by_tier = defaultdict(list)
    for r in rows: by_tier[r["tier"]].append(r)
    want = {t: round(target * f) for t, f in COMPOSITION.items()}
    chosen, dom_count = [], Counter()
    for tier, n in want.items():
        cand = list(by_tier.get(tier, [])); rng.shuffle(cand)
        for r in cand:
            if len([c for c in chosen if c["tier"] == tier]) >= n: break
            if dom_count[r["domain"]] + 1 > max_domain_frac * target: continue
            chosen.append(r); dom_count[r["domain"]] += 1
    dest = run.root / "heldout"
    if dest.exists():
        raise RuntimeError("held-out set already frozen; never edit it (a supplementary set gets its own hash)")
    dest.mkdir()
    import shutil
    for r in chosen:
        shutil.copytree(r["path"], dest / r["task_id"])
        with db.tx() as c: c.execute("UPDATE tasks SET status='heldout', path=? WHERE task_id=?", (str(dest / r["task_id"]), r["task_id"]))
    h = hashlib.sha256()
    for p in sorted(dest.rglob("*")):
        if p.is_file(): h.update(p.relative_to(dest).as_posix().encode()); h.update(p.read_bytes())
    rep = {"n": len(chosen), "target": target, "by_tier": dict(Counter(r["tier"] for r in chosen)), "by_domain": dict(dom_count),
           "by_author": dict(Counter(r["source"] for r in chosen)), "sha256": h.hexdigest(), "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "shortfall": {t: n - len([c for c in chosen if c["tier"] == t]) for t, n in want.items()}}
    (run.root / "heldout-freeze.json").write_text(json.dumps(rep, indent=1))
    from rsi.tasks.dedup import write_heldout_signature
    rep["signature_13grams"] = write_heldout_signature(dest, run.root / "heldout-13grams.json")
    db.event("heldout_freeze", **rep)
    return rep

def make_devset(run: C.RunPaths, db: DB, n: int = 150, seed: int = 0, concurrency: int = 6) -> dict:
    """Sample SETA tasks disjoint from the seed pool, import + gate, mark status='dev' (never trained on)."""
    from rsi.tasks.seeds import import_task
    rng = random.Random(seed)
    used = {r["provenance"] and json.loads(r["provenance"]).get("seed", {}).get("instance") for r in db.q("SELECT provenance FROM tasks WHERE source LIKE 'seta%'")}
    cands = [p for sub in ("SETA_Evolve", "SETA_Synth") for p in sorted((C.DATASETS_DIR / "SETA-Env" / sub).iterdir()) if p.is_dir() and (p / "task.toml").exists() and p.name not in used]
    picks = rng.sample(cands, min(int(n * 2.2), len(cands)))       # over-sample; the gate will reject some
    imported = []
    pool = run.root / "dev-candidates"; pool.mkdir(parents=True, exist_ok=True)
    for src in picks:
        tid, dest, info = import_task(src, pool, "dev_seta", licence="Apache-2.0")
        if info.get("skipped"): continue
        db.upsert_task(tid, f"rsi/{tid}", dest, "dev_seta", iteration_added=-1, status="dev-candidate", provenance=json.loads((dest / "provenance.json").read_text()))
        imported.append(dest)
    v = gate_tasks(db, run, imported, iteration=-1, n_concurrent=concurrency, batch_tag="dev-gate")
    ok = [tid for tid, x in v.items() if x == "gated"][:n]
    for tid, x in v.items():
        with db.tx() as c:
            c.execute("UPDATE tasks SET status=? WHERE task_id=?", ("dev" if tid in ok else "dev-" + x, tid))
    rep = {"candidates": len(imported), "gated": sum(1 for x in v.values() if x == "gated"), "dev": len(ok)}
    db.event("devset", **rep); return rep


def finish_gate(run: C.RunPaths, db: DB, *, tag: str = "heldout-gate", repair: bool = True, gate_repaired: bool = True, concurrency: int = 6) -> dict:
    """Finish a held-out gate whose dynamic checks already ran: rebuild verdicts from the `gates` table, set statuses,
    then (optionally) run the CLI repair round for oracle failures and gate the rewrites."""
    from rsi.improver.propose import feedback_for, repair_task
    from concurrent.futures import ThreadPoolExecutor
    rows = db.q("SELECT task_id, path, source FROM tasks WHERE source LIKE 'heldout:%' AND (status IN ('heldout-candidate', 'gated') OR status LIKE 'rejected:%')")
    verdict = {}
    for r in rows:
        g = {x["check_name"]: x["verdict"] for x in db.q("SELECT check_name, verdict FROM gates WHERE task_id=? ORDER BY id", r["task_id"])}
        static_fail = [k for k, v in g.items() if k.startswith("static:") and v == "fail"]
        if static_fail: verdict[r["task_id"]] = "rejected:static"
        elif g.get("oracle") != "pass": verdict[r["task_id"]] = "rejected:oracle" if "oracle" in g else "rejected:build_or_oracle_error"
        elif g.get("nop") != "pass": verdict[r["task_id"]] = "rejected:nop"
        elif g.get("delete-tests") != "pass": verdict[r["task_id"]] = "rejected:delete-tests"
        else: verdict[r["task_id"]] = "gated"
    for tid, x in verdict.items():
        with db.tx() as c:
            c.execute("UPDATE tasks SET status=? WHERE task_id=?", ("heldout-gated" if x == "gated" else "heldout-" + x, tid))
    from collections import Counter
    print("reconstructed verdicts:", dict(Counter(verdict.values())), flush=True)
    rep = {"verdicts": dict(Counter(verdict.values())), "repaired": 0, "repaired_gated": 0}
    if repair:
        failed = [Path(r["path"]) for r in rows if verdict.get(r["task_id"], "").startswith("rejected:") and "oracle" in verdict[r["task_id"]]]
        jobs = [run.jobs / f"{tag}-oracle", run.jobs / f"{tag}-oracle-root"]
        sources = {Path(r["path"]).name: r["source"] for r in rows}
        def one(td):
            src = sources.get(td.name, ""); cli = src.split(":")[-1]
            if cli not in CliAuthor.CMDS: return td, src, None
            return td, src, repair_task(CliAuthor(cli), td, feedback_for(td, jobs), td.parent, iteration=-1, source=src, log=print)
        repaired = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            for td, src, r in ex.map(one, failed):
                if r:
                    tid, dest = r
                    db.upsert_task(tid, f"rsi/{tid}", dest, src, iteration_added=-1, status="heldout-candidate",
                                   provenance=json.loads((dest / "provenance.json").read_text()))
                    repaired.append(dest)
        rep["repaired"] = len(repaired); print(f"repair round: {len(repaired)}/{len(failed)} rewritten", flush=True)
        if repaired and gate_repaired:
            v2 = gate_tasks(db, run, repaired, iteration=-1, n_concurrent=concurrency, batch_tag=tag + "-repair")
            for tid, x in v2.items():
                with db.tx() as c:
                    c.execute("UPDATE tasks SET status=? WHERE task_id=?", ("heldout-gated" if x.startswith("gated") else "heldout-" + x, tid))
            rep["repaired_gated"] = sum(1 for x in v2.values() if x.startswith("gated")); print("repaired gated:", rep["repaired_gated"], flush=True)
    db.event("heldout_finish_gate", **rep)
    return rep

# ------------------------------------------------------------------ supplementary set (docs/07 amendment 2026-09-07)
def gate_supplementary(run: C.RunPaths, db: DB, concurrency: int = 6) -> dict:
    """Gate the supplementary candidates (same checks), CLI repair round in parallel, gate the rewrites."""
    from rsi.improver.propose import feedback_for, repair_task
    from concurrent.futures import ThreadPoolExecutor
    cands = [Path(r["path"]) for r in db.q("SELECT path FROM tasks WHERE status='heldout-supp-candidate'")]
    tag = "heldout-supp-gate"
    v = gate_tasks(db, run, cands, iteration=-1, n_concurrent=concurrency, batch_tag=tag)
    failed = [td for td in cands if v.get(td.name, "").startswith("rejected:") and "oracle" in v[td.name]]
    jobs = [run.jobs / f"{tag}-oracle", run.jobs / f"{tag}-oracle-root"]
    sources = {td.name: (db.q("SELECT source FROM tasks WHERE task_id=?", td.name)[0]["source"] or "") for td in failed}
    def one(td):
        src = sources[td.name]; cli = src.split(":")[-1]
        if cli not in CliAuthor.CMDS: return td, src, None
        return td, src, repair_task(CliAuthor(cli), td, feedback_for(td, jobs), td.parent, iteration=-1, source=src, log=print)
    repaired = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for td, src, r in ex.map(one, failed):
            if r:
                tid, dest = r
                db.upsert_task(tid, f"rsi/{tid}", dest, src, iteration_added=-1, status="heldout-supp-candidate",
                               provenance=json.loads((dest / "provenance.json").read_text()))
                repaired.append(dest)
    print(f"supplementary repair: {len(repaired)}/{len(failed)} rewritten", flush=True)
    if repaired:
        v.update(gate_tasks(db, run, repaired, iteration=-1, n_concurrent=concurrency, batch_tag=tag + "-repair"))
    for tid, x in v.items():
        with db.tx() as c:
            c.execute("UPDATE tasks SET status=? WHERE task_id=?", ("heldout-supp-gated" if x.startswith("gated") else "heldout-supp-" + x, tid))
    from collections import Counter
    rep = dict(Counter(x if x.startswith("gated") else ":".join(x.split(":")[:2]) for x in v.values()))
    db.event("heldout_supp_gate", **rep); print("supplementary gate:", rep, flush=True); return rep

def freeze_supplementary(run: C.RunPaths, db: DB) -> dict:
    """All gated supplementary tasks, no tiering (docs/07 amendment), own directory and sha256."""
    rows = db.q("SELECT task_id, path, domain, source FROM tasks WHERE status='heldout-supp-gated'")
    dest = run.root / "heldout-supp"
    if dest.exists(): raise RuntimeError("supplementary set already frozen")
    dest.mkdir()
    import shutil
    for r in rows:
        shutil.copytree(r["path"], dest / r["task_id"])
        with db.tx() as c: c.execute("UPDATE tasks SET status='heldout-supp', path=? WHERE task_id=?", (str(dest / r["task_id"]), r["task_id"]))
    h = hashlib.sha256()
    for p in sorted(dest.rglob("*")):
        if p.is_file(): h.update(p.relative_to(dest).as_posix().encode()); h.update(p.read_bytes())
    from collections import Counter
    rep = {"n": len(rows), "by_domain": dict(Counter(r["domain"] for r in rows)), "by_author": dict(Counter(r["source"] for r in rows)),
           "sha256": h.hexdigest(), "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (run.root / "heldout-supp-freeze.json").write_text(json.dumps(rep, indent=1))
    from rsi.tasks.dedup import write_heldout_signature, hashed_13grams
    sig = run.root / "heldout-13grams.json"
    grams = set(json.loads(sig.read_text())) if sig.exists() else set()
    for p in dest.glob("*/instruction.md"): grams |= hashed_13grams(p.read_text())
    sig.write_text(json.dumps(sorted(grams))); rep["signature_13grams"] = len(grams)
    db.event("heldout_supp_freeze", **rep); return rep
