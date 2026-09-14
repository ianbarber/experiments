"""Evaluation helpers (docs/03 §9): dev-set runs, the SE-based paired gate, and the task-level paired bootstrap."""
from __future__ import annotations
import json, math, random
from collections import defaultdict
from pathlib import Path
from rsi import config as C
from rsi.rollout import run_job, ingest_job

def eval_set(loop, task_dirs: list[Path], *, policy_tag: str, purpose: str, seeds: int, job_name: str, iteration: int | None = None) -> dict:
    job = run_job(loop.run, task_dirs, job_name=job_name, agent="rsi", n_concurrent=loop.cfg.rollout_concurrency, n_attempts=seeds,
                  agent_kwargs=loop._agent_kwargs(policy_tag))
    return ingest_job(loop.db, job, policy=policy_tag, purpose=purpose, iteration=iteration)

def dev_eval(loop, k_policy: int, *, tag: str) -> dict:
    dev = [Path(r["path"]) for r in loop.db.q("SELECT path FROM tasks WHERE status='dev'")]
    if not dev:
        loop._log("no dev set registered (status='dev'); dev gate is a no-op"); return {}
    return eval_set(loop, dev, policy_tag=tag, purpose="dev", seeds=loop.cfg.dev_seeds, job_name=f"dev-{tag}", iteration=k_policy)

def task_means(db, purpose: str, policy: str) -> dict[str, float]:
    """Per-task mean reward for a policy on an evaluation set; `purpose` matches exactly or as a prefix
    ("dev" also covers "dev:start", the tagged baseline runs)."""
    by = defaultdict(list)
    for r in db.q("SELECT task_id, reward FROM episodes WHERE (purpose=? OR purpose LIKE ?) AND policy=? AND exception IS NULL AND reward IS NOT NULL", purpose, purpose + ":%", policy):
        by[r["task_id"]].append(float(r["reward"]))
    return {t: sum(v) / len(v) for t, v in by.items()}

def paired_stats(a: dict[str, float], b: dict[str, float]) -> dict:
    """Paired by task: mean difference (b - a), SE from the task-level differences, paired bootstrap 95% interval."""
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return {"n": len(common), "diff": None, "se": None, "ci95": None}
    d = [b[t] - a[t] for t in common]
    n = len(d); m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1); se = math.sqrt(var / n)
    rng = random.Random(0); boots = []
    for _ in range(2000):
        s = [d[rng.randrange(n)] for _ in range(n)]; boots.append(sum(s) / n)
    boots.sort()
    return {"n": n, "mean_a": sum(a[t] for t in common) / n, "mean_b": sum(b[t] for t in common) / n, "diff": m, "se": se,
            "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]], "p_one_sided_le0": sum(1 for x in boots if x <= 0) / len(boots)}

def paired_gate(db, head_policy: str, cand_policy: str, *, z: float = 1.28, purpose: str = "dev") -> dict:
    """Reject only if the candidate is below -z*SE relative to the lineage head on the same dev tasks (docs/03 step 7)."""
    a = task_means(db, purpose, head_policy); b = task_means(db, purpose, cand_policy)
    st = paired_stats(a, b)
    if st["diff"] is None:
        return {"accept": True, "reason": "insufficient paired dev data", **st}
    accept = st["diff"] >= -z * st["se"]
    db.metric("devgate_diff", st["diff"], None, "DEVGATE", head=head_policy, cand=cand_policy, se=st["se"])
    return {"accept": accept, "threshold": -z * st["se"], **st}
