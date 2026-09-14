#!/usr/bin/env python
"""Fill the last pre-registration placeholders from the M_0 baselines and mark the document frozen (docs/07).
MDE (80% power, one-sided 5%, paired): 2.49 x SE_diff with SE_diff ~ sqrt(2) x SD(task means) / sqrt(n_tasks),
the pre-registered proxy when only M_0 exists (docs/03 §9). Dev gate SE: the same proxy on the 150 dev tasks x 3."""
import json, math, statistics, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rsi import config as C
from rsi.db import DB

db = DB(C.run_paths("main").db)
with db.tx() as c:   # accounting: agent timeouts score 0 and are not errors (an older ingest may have stored them as errors)
    n = c.execute("UPDATE episodes SET reward=0.0, verified=0, exception=NULL, stop_reason='agent_timeout' WHERE exception='AgentTimeoutError'").rowcount
print("agent-timeout rows repaired before statistics:", n)
def task_means(purpose, policy):
    by = defaultdict(list)
    for r in db.q("SELECT task_id, reward FROM episodes WHERE purpose=? AND policy=? AND exception IS NULL AND reward IS NOT NULL", purpose, policy):
        by[r["task_id"]].append(float(r["reward"]))
    return {t: sum(v) / len(v) for t, v in by.items()}, {t: len(v) for t, v in by.items()}
ho, ho_n = task_means("heldout:start", "M0"); dev, dev_n = task_means("dev:start", "M0")
supp, _ = task_means("heldout-supp:start", "M0")
ho_union = {**ho, **supp}
if len(ho) < 100 or len(dev) < 100 or len(supp) < 20:
    print(f"not enough baseline data yet (held-out {len(ho)}, dev {len(dev)})"); sys.exit(1)
sd_ho = statistics.pstdev(ho_union.values()); se_ho = math.sqrt(2) * sd_ho / math.sqrt(len(ho_union)); mde = 2.49 * se_ho * 100
sd_dev = statistics.pstdev(dev.values()); se_dev = math.sqrt(2) * sd_dev / math.sqrt(len(dev)) * 100
hist = {"0/5": sum(1 for t, m in ho_union.items() if m == 0), "1-4/5": sum(1 for t, m in ho_union.items() if 0 < m < 1), "5/5": sum(1 for t, m in ho_union.items() if m == 1)}
rep = {"heldout_main_tasks": len(ho), "heldout_supp_tasks": len(supp), "heldout_tasks": len(ho_union), "heldout_mean": statistics.mean(ho_union.values()),
       "heldout_main_mean": statistics.mean(ho.values()), "heldout_supp_mean": statistics.mean(supp.values()) if supp else None, "heldout_sd_taskmean": sd_ho, "mde_points": mde, "heldout_hist": hist,
       "dev_tasks": len(dev), "dev_mean": statistics.mean(dev.values()), "dev_se_points": se_dev, "dev_gate_threshold_points": -1.28 * se_dev, "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
print(json.dumps(rep, indent=1))
p = C.PROJECT_DIR / "docs/07-preregistration.md"; s = p.read_text()
s = s.replace("between-task variance measured in the pilot: **[value]** points.", f"between-task variance of the M_0 baseline over the union ({len(ho)} + {len(supp)} = {len(ho_union)} tasks x 5): **{mde:.1f}** points (proxy; final evaluation at 10 seeds)\n   (SD of task means {sd_ho:.3f}; M_0 pass rate {statistics.mean(ho_union.values()):.3f}, main set {statistics.mean(ho.values()):.3f}, supplementary {statistics.mean(supp.values()) if supp else float('nan'):.3f}; histogram 0/5: {hist['0/5']}, 1-4/5: {hist['1-4/5']}, 5/5: {hist['5/5']}).")
s = s.replace("SE at **[n]** x 3 = **[value]** points.", f"SE at {len(dev)} x 3 = **{se_dev:.2f}** points (threshold {-1.28 * se_dev:.2f}; M_0 dev pass rate {statistics.mean(dev.values()):.3f}).")
s = s.replace("Status: DRAFT 2026-09-06. Placeholders in **[brackets]** are filled from\nmeasurements before the commit that freezes this document; nothing here may\nchange after that commit except by an appended, dated amendment.",
              f"Status: **FROZEN {time.strftime('%Y-%m-%d %H:%M')}** at the commit that carries this line; nothing here\nmay change after it except by an appended, dated amendment.")
p.write_text(s)
(C.PROJECT_DIR / "docs" / "07-preregistration-stats.json").write_text(json.dumps(rep, indent=1))
half = (hist["0/5"] + hist["5/5"]) > len(ho_union) / 2
print("SUPPLEMENTARY SET NEEDED (more than half saturated)" if half else "held-out histogram acceptable")
