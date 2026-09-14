"""Run a batch of episodes through Harbor with a given agent and ingest the results into SQLite.

One call = one `harbor run` job over a directory of tasks (or an explicit list), N concurrent trials,
K attempts per task. Afterwards every trial's result.json and agent/turns.jsonl are read into
`episodes` and `turns`; per-turn token records stay on disk (record_path) for the trainer.
"""
from __future__ import annotations
import json, os, shutil, subprocess, time
from pathlib import Path

from rsi import config as C
from rsi.db import DB

AGENTS = {
    "rsi": "rsi.harness.agent:RsiAgent",
    "oracle": "oracle",
    "nop": "nop",
    "delete-tests": "rsi.harness.gate_agents:DeleteTestsAgent",
    "empty": "rsi.harness.gate_agents:EmptyOutputAgent",
}

def _link_tasks(task_dirs: list[Path], dest: Path) -> Path:
    """Harbor takes one path; give it a directory of symlinks to the selected tasks."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    seen = set()
    for t in task_dirs:
        if t.name in seen: continue
        seen.add(t.name)
        (dest / t.name).symlink_to(t.resolve(), target_is_directory=True)
    return dest

def run_job(run: C.RunPaths, task_dirs: list[Path], *, job_name: str, agent: str = "rsi", n_concurrent: int = 16,
            n_attempts: int = 1, agent_kwargs: dict | None = None, timeout_multiplier: float = 1.0,
            extra_args: list[str] | None = None, log_to: Path | None = None) -> Path:
    """Blocking. Returns the job directory."""
    run.mkdirs()
    sel = _link_tasks(task_dirs, run.jobs / f"_sel_{job_name}")
    job_dir = run.jobs / job_name
    if job_dir.exists():
        shutil.rmtree(job_dir)
    cmd = [str(C.HARBOR), "run", "-p", str(sel), "-a", AGENTS.get(agent, agent), "-e", "docker",
           "-n", str(n_concurrent), "-k", str(n_attempts), "-o", str(run.jobs), "--job-name", job_name,
           "--timeout-multiplier", str(timeout_multiplier), "-y", "-q"]
    for k, v in (agent_kwargs or {}).items():
        cmd += ["--ak", f"{k}={v}"]
    cmd += extra_args or []
    env = dict(os.environ)
    env["PYTHONPATH"] = str(C.PROJECT_DIR) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path = log_to or (run.logs / f"harbor-{job_name}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(" ".join(cmd) + "\n\n"); lf.flush()
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(C.PROJECT_DIR))
    (job_dir / "_rsi_job.json").parent.mkdir(parents=True, exist_ok=True)
    (job_dir / "_rsi_job.json").write_text(json.dumps({"cmd": cmd, "returncode": p.returncode, "wall_s": time.time() - t0,
                                                        "n_tasks": len(task_dirs), "agent": agent, "agent_kwargs": agent_kwargs}, indent=1))
    return job_dir

def task_key(result: dict) -> str:
    """The pool task id for a Harbor result: the task directory name (Harbor truncates trial names and prefixes
    task_name with the org), taken from task_id.path when present."""
    tid = result.get("task_id") or {}
    if isinstance(tid, dict) and tid.get("path"):
        return Path(tid["path"]).name
    name = result.get("task_name") or ""
    return name.split("/", 1)[1] if "/" in name else name

def _reward_from(result: dict) -> tuple[float | None, bool]:
    vr = result.get("verifier_result") or {}
    rewards = vr.get("rewards") or {}
    if isinstance(rewards, dict) and rewards:
        r = rewards.get("reward", next(iter(rewards.values())))
        try: return float(r), True
        except (TypeError, ValueError): return None, False
    return None, False

def ingest_job(db: DB, job_dir: Path, *, policy: str, purpose: str, iteration: int | None = None,
               task_id_of=None) -> dict:
    """Read every trial in a job dir into SQLite. task_id_of(task_name) -> task_id (default: task_name)."""
    task_id_of = task_id_of or (lambda name: name)
    stats = {"trials": 0, "verified": 0, "errors": 0, "reward_sum": 0.0, "turns": 0}
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        rp = trial_dir / "result.json"
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        reward, verified = _reward_from(res)
        exc = res.get("exception_info")
        # agent timeouts score 0 (docs/02 accounting: turn-cap and time truncation count as failures) and are NOT errors;
        # infrastructure errors (compose failures, verifier crashes) keep reward NULL and the exception recorded
        agent_timeout = bool(exc) and (exc.get("exception_type") or "") == "AgentTimeoutError"
        if agent_timeout:
            reward, verified, exc = 0.0, False, None
        meta = (res.get("agent_result") or {}).get("metadata") or {}
        ep_id = f"{job_dir.name}/{trial_dir.name}"
        turns_path = trial_dir / "agent" / "turns.jsonl"
        turn_rows = []; n_turns = 0
        if turns_path.exists():
            with open(turns_path) as f:
                for line in f:
                    try: rec = json.loads(line)
                    except json.JSONDecodeError: continue
                    if "error" in rec: continue
                    n_turns += 1
                    turn_rows.append((ep_id, rec.get("segment", 0), rec.get("turn", 0), len(rec.get("prompt_token_ids") or []),
                                      len(rec.get("token_ids") or []), rec.get("finish_reason"), rec.get("command"), rec.get("exit_code"),
                                      rec.get("obs_chars"), rec.get("llm_latency_s"), str(turns_path)))
        started = res.get("agent_execution", {}) or {}
        db.upsert_episode(episode_id=ep_id, task_id=task_id_of(task_key(res)), job=job_dir.name, trial=trial_dir.name,
                          policy=policy, purpose=purpose, iteration=iteration, reward=reward, verified=int(verified),
                          turns=meta.get("turns", n_turns), stop_reason="agent_timeout" if agent_timeout else meta.get("stop_reason"), compactions=meta.get("compactions"),
                          n_input_tokens=(res.get("agent_result") or {}).get("n_input_tokens"),
                          n_output_tokens=(res.get("agent_result") or {}).get("n_output_tokens"),
                          wall_s=meta.get("wall_s"), started_at=started.get("started_at"), finished_at=started.get("finished_at"),
                          exception=(exc or {}).get("exception_type"), flags=None)
        if turn_rows:
            db.insert_turns(turn_rows)
        stats["trials"] += 1; stats["verified"] += int(verified); stats["errors"] += int(bool(exc))
        stats["reward_sum"] += reward or 0.0; stats["turns"] += n_turns
    stats["mean_reward"] = stats["reward_sum"] / stats["trials"] if stats["trials"] else None
    db.event("ingest_job", job=job_dir.name, policy=policy, purpose=purpose, **stats)
    return stats
