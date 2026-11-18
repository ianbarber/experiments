"""Tables and statistics from the run directories.

  python scripts/analyze.py grid     # tuning runs on seed 42: selection and matched steps
  python scripts/analyze.py final    # every three-seed comparison: endpoints and intervals
  python scripts/analyze.py lr       # one seed across learning rates

Writes ../results/grid.json, final.json, lr_diagnostic.json and base.json (paths relative to code/).

Two kinds of interval are reported and they answer different questions. "seed" is a t-interval
over the per-seed values (three seeds, 2 degrees of freedom): would another training run land
here? "item" is a paired bootstrap over benchmark items (or secret words) on seed-averaged
scores, seeds held fixed: is the difference visible on these particular models?
"""

import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("DELTUNE_RUNS", CODE / "runs"))
RESULTS = CODE.parent / "results"
METHODS = ["fft", "lora", "dt"]
TASKS = ["hellaswag", "mmlu", "truthfulqa_mc1", "winogrande", "ifeval", "humaneval"]
MC = TASKS[:4]  # the four multiple-choice benchmarks
PRIMARY = "seen"
T = 1600
SEEDS = (42, 43, 44)
NOT_TUNING = ("diag_", "rerun_", "paper_")
T975 = {2: 12.706, 3: 4.3027, 4: 3.1824}  # two-sided 95% t quantile by number of seeds
PAIRS = (("dt", "fft"), ("dt", "lora"), ("lora", "fft"))

# The paper's SFT learning rates, the full schedule, no checkpoint selection (amendment 11).
PAPER_RATES = {
    "fft": [("rerun_fft_lr3e-6_s42", T), ("fft_lr3e-6_s43", T), ("fft_lr3e-6_s44", T)],
    "lora": [(f"lora_r4_lr1e-4_s{s}", T) for s in SEEDS],
    "dt": [(f"dt_lr5e-4_a100-1000_s{s}", T) for s in SEEDS],
}
# Deletion Tuning at the paper's exact SFT recipe: 400 steps, sharpness ramp over those 400.
PAPER_RECIPE = {"dt": [(f"paper_dt_lr5e-4_s{s}", 400) for s in SEEDS]}


def load_run(name):
    d = RUNS / name
    cfg = json.loads((d / "config.json").read_text())
    evals = {}
    for p in d.glob("eval_step*.json"):
        e = json.loads(p.read_text())
        evals[e["step"]] = e
    saves = {}
    for line in (d / "log.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "save" in rec:
            saves[rec["save"]["step"]] = rec["save"]
    return {"name": name, "args": cfg["args"], "info": cfg["info"], "evals": evals, "saves": saves}


def items(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-9) / n)


def describe(a):
    if a["method"] == "fft":
        return f"lr={a['lr']:g}"
    if a["method"] == "lora":
        return f"r={a['rank']} lr={a['lr']:g}"
    return f"lr={a['lr']:g} alpha={a['alpha']}"


def win(e, pool=PRIMARY):
    return 100 * e["wordle"][pool]["win_rate"] if pool in e.get("wordle", {}) else None


def grid():
    # retrained copies and the paper-recipe runs are not tuning runs
    names = sorted(p.name for p in RUNS.glob("*_s42") if (p / "done.json").exists() and not p.name.startswith(NOT_TUNING))
    out = {"runs": [], "selected": {}, "selected_by_val_loss": {}}
    print("seen-list win % by step; held-out validation win % at the last step in brackets\n")
    for name in names:
        r = load_run(name)
        traj = sorted((s, win(e)) for s, e in r["evals"].items() if win(e) is not None)
        if not traj:
            continue
        loss = sorted((s, e["val_loss"]) for s, e in r["evals"].items())
        held = win(r["evals"][T], "heldout_val") if T in r["evals"] else None
        coarse = [(s, w) for s, w in traj if s % 400 == 0]
        best_step, best = max(coarse, key=lambda t: (t[1], -t[0]))
        row = {
            "name": name, "method": r["args"]["method"], "config": describe(r["args"]), "trajectory": traj, "val_loss": loss,
            "heldout_val_final": held, "best": best, "best_step": best_step, "saves": r["saves"],
        }
        out["runs"].append(row)
        print(f"{name:30s} " + " ".join(f"{s}:{w:.0f}" for s, w in traj) + f"  [{held if held is None else round(held, 1)}]  val loss at {T}: {dict(loss).get(T, float('nan')):.3f}")
    for m in METHODS:
        rows = [x for x in out["runs"] if x["method"] == m]
        if rows:
            out["selected"][m] = max(rows, key=lambda x: (x["best"], -dict(x["val_loss"])[x["best_step"]]))
            # what the other obvious rule would have chosen: lowest validation loss over the same checkpoints
            cands = [(v, x["name"], s, dict(x["trajectory"]).get(s)) for x in rows for s, v in x["val_loss"] if s % 400 == 0]
            cands.sort()
            out["selected_by_val_loss"][m] = [{"val_loss": v, "run": n, "step": s, "seen": w} for v, n, s, w in cands[:3]]
    if len(out["selected"]) == 3:
        level = min(x["best"] for x in out["selected"].values())
        tol = 100 * se(level / 100, 156)
        out["match_level"], out["match_tolerance"] = level, tol
        print(f"\nmatch level L = {level:.1f}% (lowest of the three bests); tolerance one SE = {tol:.1f} points")
        for m, x in out["selected"].items():
            reached = [s for s, w in x["trajectory"] if w >= level - tol]
            step = min(reached)
            below = [s for s, w in x["trajectory"] if s < step]
            unscored = [s for s in range(200, step, 200) if s not in dict(x["trajectory"]) and s > (max(below) if below else 0)]
            x["matched_step"], x["unscored_before_match"] = step, unscored
            print(f"  {m:5s} {x['name']:30s} best {x['best']:.1f}% at step {x['best_step']}; reaches L at step {step}" + (f"  (score steps {unscored} to tighten)" if unscored else ""))
        print("\nlowest validation loss over the same checkpoints (not the registered rule):")
        for m, c in out["selected_by_val_loss"].items():
            print(f"  {m:5s} " + "; ".join(f"{x['run']} step {x['step']} loss {x['val_loss']:.5f} ({x['seen']:.1f}%)" for x in c))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "grid.json").write_text(json.dumps(out, indent=1))


def boot_weights(rng, n, b):
    """b bootstrap resamples of n items, as per-item counts scaled to means."""
    return rng.multinomial(n, np.full(n, 1 / n), size=b) / n


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def seed_stat(vals):
    """Mean of per-seed values with a t-interval over seeds."""
    vals = [float(v) for v in vals]
    out = {"mean": float(np.mean(vals)), "per_seed": vals, "n": len(vals), "seed_ci95": None}
    if len(vals) in T975:
        half = T975[len(vals)] * float(np.std(vals, ddof=1)) / math.sqrt(len(vals))
        out["seed_ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def ifeval_split(d):
    """Counts from scripts/ifeval_behaviour.py: answers that open a thinking block, and the rest."""
    p = d / "ifeval_flags.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    think = [r for r in rows if r["think"]]
    rest = [r for r in rows if not r["think"]]
    return {"n": len(rows), "pass": sum(r["ok"] for r in rows), "think_n": len(think), "think_pass": sum(r["ok"] for r in think),
            "rest_n": len(rest), "rest_pass": sum(r["ok"] for r in rest), "wordle_n": sum(r["wordle"] for r in rows)}


def load_point(name, step, base_vec, keys):
    d = RUNS / name
    ep = d / f"eval_step{step}.json"
    if not ep.exists():
        return None
    e = json.loads(ep.read_text())
    run = load_run(name)
    games = json.loads((d / f"games_step{step}.json").read_text()) if (d / f"games_step{step}.json").exists() else {}
    save = run["saves"].get(step, {})
    row = {"seed": run["args"]["seed"], "run": name, "step": step, "val_loss": e["val_loss"],
           "val_loss_soft": save.get("val_loss_soft"), "val_loss_hard": save.get("val_loss_hard")}
    if save.get("mask"):
        row["mask_frac_model"] = 100 * save["mask"]["deleted_frac_of_model"]
        row["mask_frac_gate"] = 100 * save["mask"]["deleted_frac_of_target"]
        row["adapter_bytes"] = save["mask"]["adapter_bytes"]
    else:
        row["adapter_bytes"] = run["info"].get("adapter_bytes")
    won = {}
    for pool, w in e.get("wordle", {}).items():
        row[pool] = 100 * w["win_rate"]
        if pool == "seen":
            row["seen_truncated_turn_rate"] = w.get("truncated_turn_rate")
            row["seen_outcomes"] = w.get("outcomes")
        if pool in games:
            won[pool] = np.array([g["won"] for g in games[pool]], dtype=np.float32)
    vec = None
    sp = d / f"suite_step{step}" / "items.json.gz"
    if sp.exists():
        it = items(sp)
        vec = {t: np.array([it[t][k] for k in keys[t]], dtype=np.float32) for t in TASKS}
        row["tasks"] = {t: 100 * float((vec[t] - base_vec[t]).mean()) for t in TASKS}
        row["d_prior"] = float(np.mean([row["tasks"][t] for t in TASKS]))
        row["d_mc"] = float(np.mean([row["tasks"][t] for t in MC]))
        row["ifeval"] = ifeval_split(d / f"suite_step{step}")
    return {"row": row, "won": won, "vec": vec, "config": describe(run["args"]), "n_trainable": run["info"]["n_trainable"]}


def cell(points, base_vec, keys, rng, n_boot=10000):
    """One comparison: for each method a list of (run, step), one per seed."""
    out = {"methods": {}, "comparisons": {}}
    loaded = {}
    for m, runs in points.items():
        pts = [p for p in (load_point(n, s, base_vec, keys) for n, s in runs if (RUNS / n / "config.json").exists()) if p]
        if not pts:
            continue
        loaded[m] = pts
        d = out["methods"][m] = {"config": pts[0]["config"], "n_trainable": pts[0]["n_trainable"], "per_seed": [p["row"] for p in pts]}
        for k in ("seen", "heldout_test", "heldout_val", "val_loss", "d_prior", "d_mc", "mask_frac_model", "mask_frac_gate", "adapter_bytes"):
            vals = [p["row"][k] for p in pts if p["row"].get(k) is not None]
            if vals:
                d[k] = seed_stat(vals)
        with_suite = [p for p in pts if p["vec"] is not None]
        if with_suite:
            d["tasks"] = {t: seed_stat([p["row"]["tasks"][t] for p in with_suite]) for t in TASKS}

    # paired bootstrap over items, stratified by task, on seed-averaged item scores
    sm = [m for m in loaded if any(p["vec"] is not None for p in loaded[m])]
    task_boots = {m: {} for m in sm}
    if sm:
        delta = {m: {t: np.mean([p["vec"][t] for p in loaded[m] if p["vec"] is not None], axis=0) - base_vec[t] for t in TASKS} for m in sm}
        for t in TASKS:
            acc = {m: [] for m in sm}
            for start in range(0, n_boot, 500):
                w = boot_weights(rng, len(keys[t]), min(500, n_boot - start))
                for m in sm:
                    acc[m].append(w @ delta[m][t])
            for m in sm:
                task_boots[m][t] = 100 * np.concatenate(acc[m])
        for m in sm:
            d = out["methods"][m]
            d["d_prior"]["item_ci95"] = ci(np.mean([task_boots[m][t] for t in TASKS], axis=0))
            d["d_mc"]["item_ci95"] = ci(np.mean([task_boots[m][t] for t in MC], axis=0))
            for t in TASKS:
                d["tasks"][t]["item_ci95"] = ci(task_boots[m][t])
    # new-task score: bootstrap over secrets on seed-averaged outcomes
    won, wboot = {}, {}
    for pool in ("seen", "heldout_test"):
        pm = [m for m in loaded if all(pool in p["won"] for p in loaded[m])]
        if pm:
            won[pool] = {m: np.mean([p["won"][pool] for p in loaded[m]], axis=0) for m in pm}
            wboot[pool] = boot_weights(rng, len(next(iter(won[pool].values()))), 5000)
            for m in pm:
                out["methods"][m][pool]["secrets_ci95"] = ci(100 * (wboot[pool] @ won[pool][m]))

    for a, b in PAIRS:
        if a not in loaded or b not in loaded:
            continue
        comp = out["comparisons"][f"{a}_minus_{b}"] = {}
        rows = {m: {p["row"]["seed"]: p["row"] for p in loaded[m]} for m in (a, b)}
        for k in ("d_prior", "d_mc", "seen"):
            seeds = [s for s in rows[a] if s in rows[b] and rows[a][s].get(k) is not None and rows[b][s].get(k) is not None]
            if not seeds:
                continue
            comp[k] = seed_stat([rows[a][s][k] - rows[b][s][k] for s in seeds])
            signs = {np.sign(v) for v in comp[k]["per_seed"]}
            comp[k]["same_sign"] = len(signs) == 1 and 0 not in signs
        if a in sm and b in sm:
            comp["d_prior"]["item_ci95"] = ci(np.mean([task_boots[a][t] - task_boots[b][t] for t in TASKS], axis=0))
            comp["d_mc"]["item_ci95"] = ci(np.mean([task_boots[a][t] - task_boots[b][t] for t in MC], axis=0))
            comp["tasks"] = {}
            for t in TASKS:
                seeds = [s for s in rows[a] if s in rows[b] and "tasks" in rows[a][s] and "tasks" in rows[b][s]]
                comp["tasks"][t] = seed_stat([rows[a][s]["tasks"][t] - rows[b][s]["tasks"][t] for s in seeds])
                comp["tasks"][t]["item_ci95"] = ci(task_boots[a][t] - task_boots[b][t])
        if "seen" in won and a in won["seen"] and b in won["seen"]:
            comp["seen"]["secrets_ci95"] = ci(100 * (wboot["seen"] @ (won["seen"][a] - won["seen"][b])))
    return out


def excludes_zero(iv):
    return iv is not None and (iv[0] > 0 or iv[1] < 0)


def decisions(c):
    """The rules fixed in PROTOCOL.md section 7 before the follow-up ran."""
    out = {"forgets": {}, "methods_differ": {}}
    for m, d in c["methods"].items():
        if "d_prior" in d:
            s = d["d_prior"]
            out["forgets"][m] = bool(s["mean"] <= -1.0 and s["seed_ci95"] is not None and s["seed_ci95"][1] < 0 and s["item_ci95"][1] < 0)
    for k, comp in c["comparisons"].items():
        if k.startswith("dt_minus") and "d_mc" in comp and "item_ci95" in comp["d_mc"]:
            s = comp["d_mc"]
            out["methods_differ"][k] = bool(s["same_sign"] and excludes_zero(s["seed_ci95"]) and excludes_zero(s["item_ci95"]))
    return out


def base_recheck(base_vec, keys):
    """The base suite scored again in the follow-up session, against the original."""
    p = RUNS / "session2" / "base" / "suite_step0" / "items.json.gz"
    if not p.exists():
        return None
    it = items(p)
    out = {}
    for t in TASKS:
        v = np.array([it[t][k] for k in keys[t]], dtype=np.float32)
        out[t] = {"original": 100 * float(base_vec[t].mean()), "rescored": 100 * float(v.mean()), "items_changed": int((v != base_vec[t]).sum())}
    return out


def fmt_stat(s, kind="seed_ci95", digits=1):
    if s is None:
        return "n/a"
    iv = s.get(kind)
    return f"{s['mean']:+.{digits}f}" + (f" [{iv[0]:+.{digits}f}, {iv[1]:+.{digits}f}]" if iv else "")


def final(seed=0):
    sel = json.loads((RESULTS / "grid.json").read_text())["selected"]
    base = load_run_base()
    rng = np.random.default_rng(seed)
    keys = {t: sorted(base["items"][t]) for t in TASKS}
    base_vec = {t: np.array([base["items"][t][k] for k in keys[t]], dtype=np.float32) for t in TASKS}
    tasks = {t: 100 * float(base_vec[t].mean()) for t in TASKS}
    out = {
        "base": {"suite_mean": float(np.mean(list(tasks.values()))), "mc_mean": float(np.mean([tasks[t] for t in MC])), "tasks": tasks,
                 "n": {t: len(keys[t]) for t in TASKS}, "wordle": {p: 100 * v["win_rate"] for p, v in base["eval"]["wordle"].items()},
                 "val_loss": base["eval"]["val_loss"], "ifeval": ifeval_split(RUNS / "base" / "suite_step0")},
        "selected": {m: {"run": sel[m]["name"], "matched_step": sel[m]["matched_step"]} for m in sel},
        "cells": {},
    }
    rc = base_recheck(base_vec, keys)
    if rc:
        out["base_recheck"] = rc
    tuned = lambda which: {m: [(sel[m]["name"].replace("_s42", f"_s{s}"), sel[m]["matched_step"] if which == "matched" else T) for s in SEEDS] for m in METHODS if m in sel}
    specs = {
        "paper_rates": ("the paper's SFT learning rates, step 1,600, no checkpoint selection", PAPER_RATES),
        "tuned_matched": ("learning rates chosen on Wordle score, registered matched checkpoints", tuned("matched")),
        "tuned_final": ("learning rates chosen on Wordle score, step 1,600", tuned("final")),
        "paper_recipe": ("Deletion Tuning at the paper's exact SFT recipe, step 400", PAPER_RECIPE),
    }
    for name, (title, points) in specs.items():
        c = cell(points, base_vec, keys, rng)
        if not c["methods"]:
            continue
        c["title"] = title
        if name == "paper_rates":
            c["decisions"] = decisions(c)
        out["cells"][name] = c

    # seen-list wins along training for the paper-rate runs (step 400 is the paper's own schedule length)
    out["trajectories"] = {}
    for m, runs in PAPER_RATES.items():
        rows = []
        for name, _ in runs:
            if (RUNS / name / "config.json").exists():
                r = load_run(name)
                rows.append({"seed": r["args"]["seed"], "run": name, "seen": {str(s): win(e) for s, e in sorted(r["evals"].items()) if win(e) is not None},
                             "val_loss": {str(s): e["val_loss"] for s, e in sorted(r["evals"].items())}})
        if rows:
            steps = sorted({int(s) for r in rows for s in r["seen"]})
            out["trajectories"][m] = {"runs": rows, "mean": {str(s): float(np.mean([r["seen"][str(s)] for r in rows if str(s) in r["seen"]])) for s in steps},
                                      "n": {str(s): sum(str(s) in r["seen"] for r in rows) for s in steps}}

    # H1: share of full fine-tuning's gain that Deletion Tuning recovers, end of schedule, seen-list
    for name in ("tuned_final", "paper_rates"):
        ms = out["cells"].get(name, {}).get("methods", {})
        if "dt" in ms and "fft" in ms and "seen" in ms["dt"] and "seen" in ms["fft"]:
            b = out["base"]["wordle"]["seen"]
            f, d = ms["fft"]["seen"], ms["dt"]["seen"]
            out.setdefault("h1", {})[name] = {
                "pool": "seen", "base": b, "fft": f["mean"], "dt": d["mean"], "lora": ms.get("lora", {}).get("seen", {}).get("mean"),
                "recovered_fraction": (d["mean"] - b) / (f["mean"] - b),
                "per_seed_ratio": [(x - b) / (y - b) for x, y in zip(d["per_seed"], f["per_seed"])] if d["n"] == f["n"] else None,
            }
    (RESULTS / "final.json").write_text(json.dumps(out, indent=1, default=float))
    (RESULTS / "base.json").write_text(json.dumps(base["eval"], indent=1))

    print(f"base: suite mean {out['base']['suite_mean']:.2f}  multiple-choice mean {out['base']['mc_mean']:.2f}  "
          + " ".join(f"{t}={v:.1f}" for t, v in tasks.items()) + f"  wordle {out['base']['wordle']}")
    if rc:
        print("base re-scored in the follow-up session: " + " ".join(f"{t}={v['rescored'] - v['original']:+.2f} ({v['items_changed']} items)" for t, v in rc.items()))
    for name, c in out["cells"].items():
        print(f"\n== {name}: {c['title']}")
        for m, d in c["methods"].items():
            steps = sorted({r["step"] for r in d["per_seed"]})
            print(f"  {m:5s} {d['config']:26s} step {steps}  seen {fmt_stat(d.get('seen'))} per seed {[round(v, 1) for v in d['seen']['per_seed']] if 'seen' in d else None}"
                  + (f"  held-out {d['heldout_test']['mean']:.1f}" if "heldout_test" in d else ""))
            if "d_prior" in d:
                print(f"        suite {fmt_stat(d['d_prior'], digits=2)} seeds / [{d['d_prior']['item_ci95'][0]:+.2f}, {d['d_prior']['item_ci95'][1]:+.2f}] items; per seed {[round(v, 2) for v in d['d_prior']['per_seed']]}")
                print(f"        multiple-choice mean {fmt_stat(d['d_mc'], digits=2)} seeds / [{d['d_mc']['item_ci95'][0]:+.2f}, {d['d_mc']['item_ci95'][1]:+.2f}] items; per seed {[round(v, 2) for v in d['d_mc']['per_seed']]}")
                print("        tasks " + " ".join(f"{t[:5]}={d['tasks'][t]['mean']:+.1f}" for t in TASKS))
                fl = [r["ifeval"] for r in d["per_seed"] if r.get("ifeval")]
                if fl:
                    print("        IFEval: opens a thinking block " + "/".join(str(x["think_n"]) for x in fl) + "; pass rate of the rest "
                          + "/".join(f"{100 * x['rest_pass'] / max(1, x['rest_n']):.1f}" for x in fl) + "; Wordle text " + "/".join(str(x["wordle_n"]) for x in fl))
            if "mask_frac_model" in d:
                print(f"        mask: {d['mask_frac_model']['mean']:.3f}% of all weights, {d['adapter_bytes']['mean'] / 2**20:.2f} MiB")
        for k, comp in c["comparisons"].items():
            line = f"  {k}:"
            for q in ("d_prior", "d_mc", "seen"):
                if q in comp:
                    s = comp[q]
                    other = s.get("item_ci95") or s.get("secrets_ci95")
                    line += f"  {q} {fmt_stat(s, digits=2)} seeds" + (f" / [{other[0]:+.2f}, {other[1]:+.2f}] {'items' if 'item_ci95' in s else 'secrets'}" if other else "") + f" same sign {s['same_sign']};"
            print(line)
        if "decisions" in c:
            print("  decisions:", c["decisions"])
    print("\nseen-list wins along training at the paper's rates (mean over the seeds scored at that step):")
    for m, t in out["trajectories"].items():
        print(f"  {m:5s} " + "  ".join(f"{s}: {v:.1f} (n={t['n'][s]})" for s, v in t["mean"].items()))
    if "h1" in out:
        print("\nH1:", json.dumps(out["h1"], default=float))


LR_DIAGNOSTIC = {
    # one seed; every checkpoint here is within the matching tolerance on the seen-list
    "fft": [("diag_fft_lr3e-6_s42", 1200), ("diag_fft_lr1e-5_s42", 1200), ("fft_lr3e-5_s42", 1000)],
    "lora": [("lora_r4_lr1e-4_s42", 1600), ("lora_r4_lr3e-4_s42", 1000)],
    "dt": [("dt_lr5e-4_a100-1000_s42", 400), ("dt_lr1e-3_a100-1000_s42", 400), ("dt_lr3e-3_a100-1000_s42", 400)],
}


def one_model(name, step, base_vec, keys, rng, n_boot=4000):
    """Suite change from base for one checkpoint, with a paired-bootstrap interval over items."""
    d = RUNS / name
    e = json.loads((d / f"eval_step{step}.json").read_text())
    it = items(d / f"suite_step{step}" / "items.json.gz")
    delta = {t: np.array([it[t][k] for k in keys[t]], dtype=np.float32) - base_vec[t] for t in TASKS}
    boots = np.zeros(n_boot)
    for t in TASKS:
        boots += 100 * np.concatenate([boot_weights(rng, len(keys[t]), 500) @ delta[t] for _ in range(n_boot // 500)]) / len(TASKS)
    run = load_run(name)
    return {
        "run": name, "step": step, "lr": run["args"]["lr"], "config": describe(run["args"]),
        "seen": win(e), "val_loss": e["val_loss"], "d_prior": 100 * float(np.mean([delta[t].mean() for t in TASKS])), "d_prior_ci95": ci(boots),
        "d_mc": 100 * float(np.mean([delta[t].mean() for t in MC])),
        "task_deltas": {t: 100 * float(delta[t].mean()) for t in TASKS}, "mask": run["saves"].get(step, {}).get("mask"),
        "ifeval": ifeval_split(d / f"suite_step{step}"),
    }


def lr_diagnostic():
    base = load_run_base()
    keys = {t: sorted(base["items"][t]) for t in TASKS}
    base_vec = {t: np.array([base["items"][t][k] for k in keys[t]], dtype=np.float32) for t in TASKS}
    rng = np.random.default_rng(1)
    out = {}
    for m, runs in LR_DIAGNOSTIC.items():
        out[m] = [one_model(n, s, base_vec, keys, rng) for n, s in runs if (RUNS / n / f"suite_step{s}" / "items.json.gz").exists()]
        for r in out[m]:
            print(f"{m:5s} {r['config']:26s} step {r['step']:4d}  seen {r['seen']:.1f}%  val loss {r['val_loss']:.4f}  d_prior {r['d_prior']:+.2f} [{r['d_prior_ci95'][0]:+.2f}, {r['d_prior_ci95'][1]:+.2f}]  mc {r['d_mc']:+.2f}  "
                  + " ".join(f"{t[:5]}={v:+.1f}" for t, v in r["task_deltas"].items()))
    (RESULTS / "lr_diagnostic.json").write_text(json.dumps(out, indent=1, default=float))


def load_run_base():
    d = RUNS / "base"
    return {"eval": json.loads((d / "eval_step0.json").read_text()), "items": items(d / "suite_step0" / "items.json.gz"),
            "games": json.loads((d / "games_step0.json").read_text())}


if __name__ == "__main__":
    {"grid": grid, "final": final, "lr": lr_diagnostic}[sys.argv[1]]()
