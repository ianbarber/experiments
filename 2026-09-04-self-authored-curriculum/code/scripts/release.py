"""Stage the Hugging Face release bundle (docs/03 §12). Assembles only: nothing is uploaded here.

    venvs/loop/bin/python scripts/release.py --run main [--out ./release]
                                             [--include-heldout] [--include-merged] [--trajectories all|train|none]

Layout produced:
    adapter/            LoRA adapter for every accepted checkpoint, with its training and merge reports
    improver/           the task-writing playbook as a git repo, one commit per iteration (H_0 .. H_k)
    dataset/tasks/      self-written tasks whose provenance says release_eligible under a permissive licence
    dataset/heldout/    the frozen held-out sets, only with --include-heldout
    trajectories/       per-iteration gzipped JSONL: one row per episode with its turns
    logs/               a clean copy of the SQLite log, the dashboard, the analysis bundle, the probe report
    docs/               design docs, the pre-registration with its amendments, the lab notebook, the write-up
    README.md           model card; DATASET.md; LICENCES.md; MANIFEST.json (sha256 of everything shipped)
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, shutil, sqlite3, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rsi import config as C
from rsi.db import DB

PERMISSIVE = {"Apache-2.0", "MIT", "CC-BY-4.0", "apache-2.0", "mit", "cc-by-4.0"}
WITHHELD_SOURCES = {"terminalworld_verified", "terminal_bench_2.0", "terminal_bench_2.0_public"}  # third-party, NC or upstream-owned

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def copy_adapters(run, out: Path, log=print) -> dict:
    dst = out / "adapter"; dst.mkdir(parents=True, exist_ok=True); n = 0
    for a in sorted(run.adapters.glob("adapter_*")):
        if a.is_dir():
            shutil.copytree(a, dst / a.name, dirs_exist_ok=True); n += 1
        elif a.suffix == ".json":
            shutil.copy2(a, dst / a.name)
    for rep in sorted((run.root / "iterations").glob("*/merge*.json")):
        shutil.copy2(rep, dst / f"{rep.parent.name}-{rep.name}")
    log(f"adapters: {n}")
    return {"adapters": n}

def copy_improver(run, out: Path, log=print) -> dict:
    dst = out / "improver"
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(run.improver, dst)
    n = subprocess.run(["git", "-C", str(dst), "rev-list", "--count", "HEAD"], capture_output=True, text=True).stdout.strip()
    log(f"improver: {n or '?'} commits")
    return {"improver_commits": n}

def stage_tasks(db: DB, out: Path, *, include_heldout: bool, log=print) -> dict:
    kept = {"tasks": 0, "dev": 0}; withheld = 0
    reasons: dict[str, int] = {}
    for sub in kept: (out / "dataset" / sub).mkdir(parents=True, exist_ok=True)
    rows = db.q("SELECT task_id, path, status, source, provenance FROM tasks WHERE status IN ('gated','dev','frontier') OR status LIKE 'gated%'")
    for r in rows:
        prov = json.loads(r["provenance"] or "{}")
        sub = "tasks" if (r["source"] or "").startswith("policy") else "dev"
        why = None
        if r["source"] in WITHHELD_SOURCES: why = "third-party source"
        elif prov.get("release_eligible") is False: why = "provenance says not release eligible"
        elif prov.get("licence") and prov["licence"] not in PERMISSIVE: why = f"licence {prov['licence']}"
        elif not Path(r["path"]).exists(): why = "task directory missing"
        if why:
            withheld += 1; reasons[why] = reasons.get(why, 0) + 1; continue
        shutil.copytree(r["path"], out / "dataset" / sub / r["task_id"], dirs_exist_ok=True); kept[sub] += 1
    log(f"tasks: {kept['tasks']} self-written, {kept['dev']} imported dev/seed, {withheld} withheld {reasons}")
    res = {"tasks_released": kept["tasks"], "dev_released": kept["dev"], "tasks_withheld": withheld, "withheld_reasons": reasons}
    if include_heldout:
        hd = out / "dataset" / "heldout"; hd.mkdir(parents=True, exist_ok=True); n = 0
        for r in db.q("SELECT task_id, path FROM tasks WHERE status IN ('heldout','heldout-supp')"):
            if Path(r["path"]).exists(): shutil.copytree(r["path"], hd / r["task_id"], dirs_exist_ok=True); n += 1
        log(f"held-out: {n} tasks staged (release only with the post)")
        res["heldout_released"] = n
    return res

def export_trajectories(run, db: DB, out: Path, *, which: str, log=print) -> dict:
    """One gzipped JSONL per iteration: the episode row from the database plus its recorded turns."""
    if which == "none": return {"trajectories": "skipped"}
    dst = out / "trajectories"; dst.mkdir(parents=True, exist_ok=True)
    purposes = ("rollout",) if which == "train" else None
    q = "SELECT * FROM episodes" + (" WHERE purpose='rollout'" if purposes else "")
    rows = [dict(r) for r in db.q(q)]
    by_it: dict[int, list] = {}
    for r in rows: by_it.setdefault(r["iteration"] if r["iteration"] is not None else -1, []).append(r)
    total = 0
    for k, eps in sorted(by_it.items()):
        with gzip.open(dst / f"iteration-{k:02d}.jsonl.gz", "wt") as f:
            for e in eps:
                turns = []
                job = run.jobs / (e["job"] or "")
                trial = job / (e["trial"] or "")
                tj = trial / "agent" / "turns.jsonl"
                if tj.exists():
                    turns = [json.loads(l) for l in tj.read_text(errors="replace").splitlines() if l.strip()]
                f.write(json.dumps({**e, "turns": turns}) + "\n"); total += 1
    log(f"trajectories: {total} episodes across {len(by_it)} iterations")
    return {"episodes_exported": total}

def copy_logs_and_docs(run, out: Path, log=print) -> dict:
    (out / "logs").mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(run.db)); con.execute("VACUUM INTO ?", (str(out / "logs" / "rsi.sqlite"),)); con.close()
    for src, dstname in ((run.root / "analysis", "analysis"), (run.root / "dashboard", "dashboard"), (run.root / "probe", "probe")):
        if src.exists():
            shutil.copytree(src, out / "logs" / dstname, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("records.json"))  # records.json embeds full task text
    for f in ("state.json", "loop-config.json", "heldout-freeze.json", "heldout-supp-freeze.json"):
        if (run.root / f).exists(): shutil.copy2(run.root / f, out / "logs" / f)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    for d in sorted((ROOT / "docs").glob("*.md")): shutil.copy2(d, out / "docs" / d.name)
    for extra in ("configs/main-9b.json",):
        (out / "docs" / Path(extra).name).write_bytes((ROOT / extra).read_bytes())
    shutil.copytree(ROOT / "rsi", out / "code" / "rsi", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "scripts", out / "code" / "scripts", dirs_exist_ok=True)
    log("logs, docs and loop code staged")
    return {}

def cards(run, db: DB, out: Path, stats: dict, analysis: dict) -> None:
    prim = analysis.get("primary_heldout_union", {})
    head = analysis.get("lineage_head", "M?")
    line = (f"{prim['diff_points']:+.2f} points (95% CI [{prim['ci95_points'][0]:+.2f}, {prim['ci95_points'][1]:+.2f}], "
            f"n={prim['n']} tasks)" if "diff_points" in prim else "the run had not finished when this bundle was staged")
    (out / "README.md").write_text(f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
base_model_relation: adapter
library_name: peft
tags: [reinforcement-learning, self-improvement, terminal-agent, lora]
---

# RSI at home: {head}

A LoRA adapter trained by a single Qwen3.5-9B on tasks it wrote for itself, on one RTX 5090.
The model proposes Harbor-format terminal tasks, gates them by execution, attempts them, trains on
its own rollouts with a policy gradient, and edits the playbook it writes tasks from. Everything in
this repository was produced by that loop.

**Held-out result.** {head} against the starting model on a private, model-authored held-out set,
same session, paired by task: {line}.

**What is here.** The adapter for each accepted checkpoint, the task-writing playbook with one commit
per iteration, the self-written tasks that are free to redistribute, every episode with its turns,
the SQLite log of the whole run, the analysis bundle, and the pre-registration with its amendments.

**What this is not.** Not an intelligence explosion, not automated AI research, not general capability
gain. The loop never modifies its own code or configuration. Numbers outside the held-out set,
especially the pass rate on the model's own tasks, are not evidence of improvement.

**Training.** LoRA policy gradient (RLOO advantages, DAPO clipping) on rollouts from the model itself;
merged in fp32 and served in fp16, because merging into bf16 discarded most of the update.
See `docs/07-preregistration.md` for the endpoints, thresholds and amendments, and
`docs/06-lab-notebook.md` for what went wrong.

**Licence.** Apache-2.0 for the adapter and the loop code. The task dataset carries a per-subset
licence table in `LICENCES.md`; third-party seeds are not redistributed here.
""")
    (out / "DATASET.md").write_text(f"""# Self-written task dataset

`dataset/tasks` holds {stats.get('tasks_released', 0)} Harbor-format terminal tasks written by Qwen3.5-9B during the run, each with a
Dockerfile, an instruction, a hidden pytest suite and a reference solution that had to pass its own tests
before the task was accepted. {stats.get('tasks_withheld', 0)} generated tasks are withheld: {json.dumps(stats.get('withheld_reasons', {}))}.

Each task keeps its `provenance.json`: the generating policy, the iteration, the playbook theme, the
exemplars used, the prompt hash and the gate verdicts. Tasks are not curated by a human; they are what
the model wrote and what the executable gate accepted.

`dataset/dev` holds the {stats.get('dev_released', 0)} imported tasks used as the regression-gate dev set. They are
derived from third-party Apache-2.0 and CC-BY-4.0 seed collections, not written by the trainee; each keeps its
upstream attribution in `provenance.json`.

The private held-out sets are hashed in `logs/heldout-freeze.json`. They are released only alongside the
write-up, and once released they stop being a clean held-out set for anyone who trains on them.
""")
    (out / "LICENCES.md").write_text("""# Licences

| Part | Licence | Note |
| --- | --- | --- |
| LoRA adapters, loop code, docs | Apache-2.0 | produced by this project |
| Self-written tasks (`dataset/tasks`) | Apache-2.0 | generated by the trainee from Apache-2.0 / CC-BY-4.0 seeds |
| Seed derivatives from CC-BY-4.0 sets | CC-BY-4.0 | attribution kept in each task's `provenance.json` |
| TerminalWorld-verified | not redistributed | third-party evaluation set, taken from upstream |
| Terminal-Bench 2.0 | not redistributed | third-party benchmark, taken from upstream |
| Base model | Qwen3.5-9B licence | adapter only; no base weights here |
""")

def manifest(out: Path) -> dict:
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.json"]
    entries = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files if p.stat().st_size < 2 << 30}
    total = sum(e["bytes"] for e in entries.values())
    (out / "MANIFEST.json").write_text(json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "files": len(entries),
                                                   "bytes": total, "entries": entries}, indent=1))
    return {"files": len(entries), "gib": round(total / (1 << 30), 2)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main"); ap.add_argument("--out", default="release")
    ap.add_argument("--include-heldout", action="store_true"); ap.add_argument("--include-merged", action="store_true")
    ap.add_argument("--trajectories", default="all", choices=["all", "train", "none"])
    a = ap.parse_args()
    run = C.run_paths(a.run); db = DB(run.db); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    stats |= copy_adapters(run, out); stats |= copy_improver(run, out)
    stats |= stage_tasks(db, out, include_heldout=a.include_heldout)
    stats |= export_trajectories(run, db, out, which=a.trajectories)
    stats |= copy_logs_and_docs(run, out)
    apath = run.root / "analysis" / "results.json"
    analysis = json.loads(apath.read_text()) if apath.exists() else {}
    cards(run, db, out, stats, analysis)
    if a.include_merged:
        (out / "MERGED.md").write_text(f"The merged fp16 checkpoint is at `{C.MERGED_PATH}` on the run machine; "
                                       "upload it separately with `huggingface-cli upload`, it is about 19 GB.\n")
    stats |= manifest(out)
    print(json.dumps(stats, indent=1))
    print(f"\nStaged at {out}. Nothing has been uploaded. To publish:\n"
          f"  huggingface-cli upload <user>/rsi-at-home-adapter {out} . --repo-type=model\n"
          f"  huggingface-cli upload <user>/rsi-at-home-tasks {out}/dataset . --repo-type=dataset")
