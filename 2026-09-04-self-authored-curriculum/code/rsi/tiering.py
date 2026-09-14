"""TIER step (docs/03 §4 step 4): pass rate over G samples per task -> training pool / frontier / drop."""
from __future__ import annotations
from collections import defaultdict
from rsi.db import DB

def tier_from_rollout(db: DB, purpose: str, iteration: int, band: tuple[float, float]) -> dict:
    rows = db.q("SELECT task_id, reward, exception FROM episodes WHERE purpose=? AND iteration=?", purpose, iteration)
    by = defaultdict(list)
    for r in rows:
        if r["exception"] is None and r["reward"] is not None:
            by[r["task_id"]].append(float(r["reward"]))
    result = {"train": [], "frontier": [], "drop": [], "rates": {}}
    for tid, rs in by.items():
        p = sum(rs) / len(rs)
        result["rates"][tid] = p
        if p == 0.0: result["frontier"].append(tid)
        elif p >= 1.0: result["drop"].append(tid)
        elif band[0] <= p <= band[1]: result["train"].append(tid)
        else: result["train"].append(tid)      # in (0,1) but outside the band still carries variance; kept, logged
        tier = "easy" if p >= 0.7 else "medium" if p >= 0.4 else "hard" if p >= 0.1 else "frontier"
        with db.tx() as c:
            c.execute("UPDATE tasks SET tier=? WHERE task_id=?", (tier, tid))
    db.metric("tier_train", len(result["train"]), iteration, "TIER"); db.metric("tier_frontier", len(result["frontier"]), iteration, "TIER")
    db.metric("tier_drop", len(result["drop"]), iteration, "TIER")
    db.event("tier", iteration=iteration, n_tasks=len(by), train=len(result["train"]), frontier=len(result["frontier"]), drop=len(result["drop"]),
             mean_rate=sum(result["rates"].values()) / max(len(result["rates"]), 1))
    return result
