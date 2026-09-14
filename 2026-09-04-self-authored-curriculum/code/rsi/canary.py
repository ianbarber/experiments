"""Canary composite (docs/02): IFEval, GSM8K, HumanEval, MBPP via lm-evaluation-harness against the vLLM endpoint,
greedy, fixed config; item-weighted composite compared with M_0 (a > 2-point drop fails the gate)."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
from rsi import config as C

TASKS = {"ifeval": "prompt_level_strict_acc,none", "gsm8k": "exact_match,strict-match"}   # composite tasks
ITEMS = {"ifeval": 541, "gsm8k": 1319}
# HumanEval / MBPP were dropped from the composite on 2026-09-07: through the chat template with thinking, lm-eval's raw-completion
# code tasks score 0.0 for the base model and therefore carry no drift information.

def run_canaries(run, policy_tag: str, *, base_url: str = C.VLLM_BASE_URL, limit: int | None = None, tasks: list[str] | None = None) -> dict:
    out_dir = run.root / "canaries" / policy_tag; out_dir.mkdir(parents=True, exist_ok=True)
    tasks = tasks or list(TASKS)
    cmd = [str(C.VENVS["loop"] / "bin" / "lm_eval"), "--model", "local-chat-completions",
           "--model_args", f"model={C.SERVED_MODEL_NAME},base_url={base_url}/chat/completions,num_concurrent=16,max_retries=3,tokenized_requests=False",
           "--tasks", ",".join(tasks), "--apply_chat_template", "--gen_kwargs", "temperature=0,max_gen_toks=4096",
           "--output_path", str(out_dir), "--log_samples", "--batch_size", "16", "--confirm_run_unsafe_code"]   # humaneval/mbpp execute model code (sandboxed by the harness's own runner)
    if limit: cmd += ["--limit", str(limit)]
    env = dict(os.environ, HF_ALLOW_CODE_EVAL="1")
    t0 = time.time()
    with open(out_dir / "lm_eval.log", "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(C.PROJECT_DIR))
    res_files = sorted(out_dir.rglob("results*.json"), key=lambda p: p.stat().st_mtime)
    if r.returncode or not res_files:
        return {"error": f"lm_eval exit {r.returncode}", "log": str(out_dir / "lm_eval.log")}
    res = json.loads(res_files[-1].read_text())["results"]
    scores = {}
    for t, metric in TASKS.items():
        if t in res:
            key = next((k for k in res[t] if k.startswith(metric.split(",")[0])), None)
            scores[t] = float(res[t][key]) if key else None
    n = sum(ITEMS[t] for t in scores if scores[t] is not None)
    composite = sum(scores[t] * ITEMS[t] for t in scores if scores[t] is not None) / n if n else None
    rep = {"policy": policy_tag, "scores": scores, "composite": composite, "wall_s": round(time.time() - t0), "limit": limit}
    (out_dir / "summary.json").write_text(json.dumps(rep, indent=1))
    return rep

def canary_gate(run, cand: dict, base: dict, threshold: float = 0.02) -> dict:
    if not cand.get("composite") or not base.get("composite"):
        return {"accept": True, "reason": "missing canary data"}
    drop = base["composite"] - cand["composite"]
    return {"accept": drop <= threshold, "drop": drop, "base": base["composite"], "cand": cand["composite"]}
