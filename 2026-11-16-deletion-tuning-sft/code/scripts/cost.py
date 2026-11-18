"""What each method costs to train: time per step, peak GPU memory, trainable state, stored size.

  python scripts/cost.py        # writes ../results/cost.json

Reads the per-step timings in runs/*/log.jsonl, so it needs the run directories, not the
evidence archive. Time per step leaves out the steps that follow a checkpoint save.
"""

import json
import os
import statistics
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("DELTUNE_RUNS", CODE / "runs"))
GIB = 2**30


def run_cost(d: Path):
    lines = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines()]
    if any("elapsed" not in r and "save" not in r for r in lines):
        return None
    cfg = json.loads((d / "config.json").read_text())
    done = json.loads((d / "done.json").read_text())
    train_s, tokens, steps, prev, after_save = 0.0, 0, 0, None, False
    for r in lines:
        if "save" in r:
            after_save = True
            continue
        if prev is not None and not after_save:
            train_s += r["elapsed"] - prev["elapsed"]
            tokens += r["tokens"]
            steps += 1
        prev, after_save = r, False
    return {
        "method": cfg["args"]["method"], "rank": cfg["args"]["rank"], "trainable": cfg["info"]["n_trainable"],
        "seconds_per_step": train_s / steps, "label_tokens_per_second": tokens / train_s,
        "peak_gib": done["peak_mem_gib"], "label_tokens_per_step": tokens / steps,
    }


def main():
    per_run = {}
    for d in sorted(RUNS.iterdir()):
        if (d / "done.json").exists() and (d / "log.jsonl").exists():
            rec = run_cost(d)
            if rec:
                per_run[d.name] = rec
    out = {"per_run": per_run, "by_method": {}}
    for m in ("fft", "lora", "dt"):
        rows = [r for r in per_run.values() if r["method"] == m and (m != "lora" or r["rank"] == 4)]
        n = rows[0]["trainable"]
        out["by_method"][m] = {
            "runs": len(rows), "trainable": n,
            "seconds_per_step": statistics.median(r["seconds_per_step"] for r in rows),
            "seconds_per_step_range": [min(r["seconds_per_step"] for r in rows), max(r["seconds_per_step"] for r in rows)],
            "label_tokens_per_second": statistics.median(r["label_tokens_per_second"] for r in rows),
            "peak_gib": sorted({round(r["peak_gib"], 1) for r in rows}),
            # fp32: the parameter, its gradient and two Adam moments
            "trainable_state_gib": 16 * n / GIB,
        }
    out["frozen_base_fp32_gib"] = 4 * json.loads((RUNS / "dt_lr5e-4_a100-1000_s42" / "config.json").read_text())["info"]["n_model_params"] / GIB
    (CODE.parent / "results" / "cost.json").write_text(json.dumps(out, indent=1))
    for m, r in out["by_method"].items():
        print(f"{m:5s} {r['runs']:2d} runs  {r['seconds_per_step']:.3f} s/step  peak {r['peak_gib']} GiB  trainable state {r['trainable_state_gib']:.2f} GiB")
    print("frozen fp32 base %.2f GiB" % out["frozen_base_fp32_gib"])


if __name__ == "__main__":
    main()
