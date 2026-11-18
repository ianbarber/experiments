"""Run a named stage: train (unless done) and score checkpoints, one job at a time.

  python scripts/sweep.py base
  python scripts/sweep.py grid [--only fft,lora,dt] [--configs lr,lr,...]
  python scripts/sweep.py one --method dt --lr 5e-4 [--alpha 1000,10000] [--rank 1]
  python scripts/sweep.py score RUN --steps 200,600 --pools seen [--suite]
  python scripts/sweep.py final --fft RUN:STEP --lora RUN:STEP --dt RUN:STEP

Every run trains on the 1,600-step schedule fixed by the pilot and saves a checkpoint every 200 steps.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
RUNS = CODE / "runs"
STEPS = 1600
SAVE = ",".join(str(s) for s in range(200, STEPS + 1, 200))
ENV = dict(os.environ, PYTHONPATH=str(CODE), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
GRID = {
    "dt": ["5e-4", "3e-4", "1e-3"],
    "fft": ["1e-5", "3e-6", "3e-5"],
    "lora": ["3e-4", "1e-3"],  # 1e-4 is the pilot
}


def call(module, name, *args):
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", module, *args]
    print("[run]", " ".join(cmd[2:]), flush=True)
    with open(RUNS / name / f"{module.split('.')[-1]}.log", "a") as f:
        rc = subprocess.run(cmd, cwd=CODE, stdout=f, stderr=subprocess.STDOUT, env=ENV).returncode
    if rc != 0:
        print(f"[FAILED rc={rc}] {name}: see runs/{name}/{module.split('.')[-1]}.log", flush=True)
    return rc


def train(name, *args):
    if (RUNS / name / "done.json").exists():
        print(f"[skip train] {name}", flush=True)
        return 0
    return call("deltune.train", name, "--name", name, "--steps", str(STEPS), *args)


def score(name, steps, pools="", suite=False):
    args = [name, "--steps", steps]
    if pools:
        args += ["--pools", pools]
    if suite:
        args.append("--suite")
    rc = call("deltune.evaluate", name, *args)
    for s in steps.split(","):
        p = RUNS / name / f"eval_step{s}.json"
        if p.exists():
            r = json.loads(p.read_text())
            line = {k: round(100 * v["win_rate"], 1) for k, v in r["wordle"].items()}
            print(f"[score] {name} step {s} val_loss {r['val_loss']:.4f} {line} suite {r.get('suite', {}).get('suite_mean')}", flush=True)
    return rc


def run_name(method, lr, alpha="100,1000", rank="4", seed="42"):
    tag = {"fft": f"fft_lr{lr}", "lora": f"lora_r{rank}_lr{lr}", "dt": f"dt_lr{lr}_a{alpha.replace(',', '-')}"}[method]
    return f"{tag}_s{seed}"


def method_args(method, lr, alpha="100,1000", rank="4"):
    args = ["--method", method, "--lr", lr]
    if method == "lora":
        args += ["--rank", rank]
    if method == "dt":
        args += ["--alpha", alpha]
    return args


def tuning_run(method, lr, alpha="100,1000", rank="4"):
    name = run_name(method, lr, alpha, rank)
    if train(name, *method_args(method, lr, alpha, rank), "--seed", "42", "--save-steps", SAVE) == 0:
        score(name, "400,800,1200,1600", "seen")
        score(name, "1600", "heldout_val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["base", "grid", "one", "score", "final"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--only", default="dt,fft,lora")
    ap.add_argument("--method")
    ap.add_argument("--lr")
    ap.add_argument("--alpha", default="100,1000")
    ap.add_argument("--rank", default="4")
    ap.add_argument("--steps", default="")
    ap.add_argument("--pools", default="")
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--seeds", default="43,44")
    ap.add_argument("--dt-extra-save", default="", help="an extra Deletion Tuning checkpoint to keep for the fresh seeds")
    for m in ("fft", "lora", "dt"):
        ap.add_argument(f"--{m}", default="", help="selected seed-42 run and its matched step, RUN:STEP")
    a = ap.parse_args()

    if a.stage == "base":
        score("base", "0", "seen,heldout_val,heldout_test", suite=True)
    elif a.stage == "grid":
        only = a.only.split(",")
        for i in range(3):  # middle of each grid first, so every method has a point early
            for m in only:
                if i < len(GRID[m]):
                    tuning_run(m, GRID[m][i])
    elif a.stage == "one":
        tuning_run(a.method, a.lr, a.alpha, a.rank)
    elif a.stage == "score":
        score(a.run, a.steps, a.pools, a.suite)
    elif a.stage == "final":
        sel = {m: getattr(a, m).split(":") for m in ("dt", "lora", "fft") if getattr(a, m)}
        # seed 42 is the tuning run itself; seeds 43 and 44 are fresh runs of the same configuration
        for m, (name, step) in sel.items():
            score(name, step, "seen,heldout_test", suite=True)
        for seed in a.seeds.split(","):
            for m, (name, step) in sel.items():
                cfg = json.loads((RUNS / name / "config.json").read_text())["args"]
                new = name.replace("_s42", f"_s{seed}")
                args = method_args(m, f"{cfg['lr']:g}", cfg["alpha"], str(cfg["rank"]))
                saves = f"{step},{STEPS}" + (f",{a.dt_extra_save}" if m == "dt" and a.dt_extra_save else "")
                if train(new, *args, "--seed", seed, "--save-steps", saves) == 0:
                    score(new, step, "seen,heldout_test", suite=True)
                    score(new, str(STEPS), "seen")
        # the paper's same-steps protocol, one seed: every method at the end of the schedule
        for m, (name, step) in sel.items():
            score(name, str(STEPS), "seen,heldout_test", suite=True)


if __name__ == "__main__":
    main()
