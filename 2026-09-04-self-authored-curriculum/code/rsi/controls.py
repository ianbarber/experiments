"""The v0 control set that lives outside the main lineage (docs/03 §8): the iteration-0 RFT-vs-PG ablation and the
random-reward control. Each trains an adapter from an iteration's stored rollouts, merges to the ALT fixed path, and
evaluates; the main lineage is untouched."""
from __future__ import annotations
import json, os, shutil, subprocess, time
from pathlib import Path
from rsi import config as C
from rsi.db import DB
from rsi import serve
from rsi.evaluate import eval_set, task_means, paired_stats

def _free_gpu(log=print, floor_mib: int = 4000, wait_s: float = 120) -> None:
    """Training needs the whole card: stop any vLLM (an evaluation may still be serving) and wait for the memory back."""
    serve.stop()
    t0 = time.time()
    while time.time() - t0 < wait_s and serve.gpu_mem_used_mib() > floor_mib:
        time.sleep(5)
    log(f"gpu free: {serve.gpu_mem_used_mib()} MiB in use")

def _train(run: C.RunPaths, base_model: str, iteration: int, adapter_out: Path, *, objective: str, lr: float, shuffle: bool, adapter_in: Path | None,
           tasks_per_minibatch: int = 8, window: int = 16384, log=print) -> dict:
    _free_gpu(log=log)
    report = adapter_out.parent / (adapter_out.name + ".train.json")
    cmd = [str(C.VENVS["train"] / "bin" / "python"), "-m", "rsi.train.trainer", "--model", base_model, "--db", str(run.db), "--purpose", "rollout",
           "--iteration", str(iteration), "--adapter-out", str(adapter_out), "--objective", objective, "--lr", str(lr),
           "--tasks-per-minibatch", str(tasks_per_minibatch), "--window", str(window), "--report", str(report)]
    if shuffle: cmd.append("--shuffle-rewards")
    if adapter_in: cmd += ["--adapter-in", str(adapter_in)]
    log("training: " + " ".join(cmd))
    with open(run.logs / f"train-{adapter_out.name}.log", "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(C.PROJECT_DIR),
                           env=dict(os.environ, PYTHONPATH=str(C.PROJECT_DIR), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
    if r.returncode: raise RuntimeError(f"trainer failed: logs/train-{adapter_out.name}.log")
    return json.loads(report.read_text())

def _merge(run: C.RunPaths, base_model: str, adapter: Path, out: Path, log=print, dtype: str = "bfloat16") -> None:
    cmd = [str(C.VENVS["train"] / "bin" / "python"), str(C.PROJECT_DIR / "smoke/05-merge.py"), "--base", base_model, "--adapter", str(adapter), "--out", str(out),
           "--report", str(adapter.parent / (adapter.name + ".merge.json")), "--dtype", dtype]
    with open(run.logs / f"merge-{adapter.name}.log", "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(C.PROJECT_DIR))
    if r.returncode: raise RuntimeError("merge failed")

def _serve_alt(run: C.RunPaths, path: Path, concurrency: int, log=print, dtype: str = "bfloat16"):
    serve.stop(); serve.start(path, max_num_seqs=32, log_path=run.logs / "vllm-alt.log", dtype=dtype); log(f"vLLM up on {path} in {serve.wait_healthy():.0f}s")

def rft_vs_pg(loop, iteration: int = 0, *, rft_lr: float = 1e-4, dev_seeds: int = 3, log=print) -> dict:
    """From the same stored iteration-k rollouts: train an RFT adapter (the PG adapter is the lineage's adapter_{k+1});
    evaluate the RFT one on dev x3 and compare paired with the PG one's dev run."""
    run = loop.run; base = loop.cfg.base_model
    adapter_in = loop._adapter(iteration)
    rft_out = run.adapters / f"ablation_rft_{iteration:02d}"
    if (rft_out / "adapter_config.json").exists() and loop.db.q("SELECT COUNT(*) n FROM episodes WHERE purpose='dev' AND policy=?", f"ablationRFT{iteration + 1}")[0]["n"] > 0:
        log("ablation already done; skipping"); return {"note": "already done"}
    rep = _train(run, base, iteration, rft_out, objective="rft", lr=rft_lr, shuffle=False, adapter_in=adapter_in, tasks_per_minibatch=loop.cfg.tasks_per_minibatch, window=loop.cfg.harness_window, log=log)
    if rep.get("steps", 0) == 0: return {"note": "no RFT examples", "train": rep}
    _merge(run, base, rft_out, C.MERGED_ALT_PATH, log=log, dtype=loop.cfg.serve_dtype)
    _serve_alt(run, C.MERGED_ALT_PATH, loop.cfg.rollout_concurrency, log=log, dtype=loop.cfg.serve_dtype)
    dev = loop._tasks("status='dev'")
    tag = f"ablationRFT{iteration + 1}"
    st = eval_set(loop, dev, policy_tag=tag, purpose="dev", seeds=dev_seeds, job_name=f"dev-{tag}", iteration=iteration + 1)
    pg = task_means(loop.db, "dev", f"M{iteration + 1}"); rft = task_means(loop.db, "dev", tag)
    res = {"rft_train": rep, "dev_rft": st, "paired_pg_minus_rft": paired_stats(rft, pg)}
    loop.db.event("ablation_rft_vs_pg", iteration=iteration, **{k: v for k, v in res.items() if k != "rft_train"})
    return res

def random_reward(loop, iteration: int, *, heldout_seeds: int = 5, dev_seeds: int = 10, log=print) -> dict:
    """One update from M_k with rewards shuffled across the batch (same rollouts, same step count), evaluated on the
    held-out set x5 and dev x10; compared paired with M_k."""
    run = loop.run; base = loop.cfg.base_model
    adapter_in = loop._adapter(iteration)
    out = run.adapters / f"control_random_reward_{iteration:02d}"
    if (out / "adapter_config.json").exists() and loop.db.q("SELECT COUNT(*) n FROM episodes WHERE purpose='dev' AND policy=?", f"controlRR{iteration}")[0]["n"] > 0:
        log("random-reward control already done; skipping"); return {"note": "already done"}
    rep = _train(run, base, iteration, out, objective=loop.cfg.objective, lr=loop.cfg.lr, shuffle=True, adapter_in=adapter_in, tasks_per_minibatch=loop.cfg.tasks_per_minibatch, window=loop.cfg.harness_window, log=log)
    if rep.get("steps", 0) == 0: return {"note": "no examples", "train": rep}
    _merge(run, base, out, C.MERGED_ALT_PATH, log=log, dtype=loop.cfg.serve_dtype)
    _serve_alt(run, C.MERGED_ALT_PATH, loop.cfg.rollout_concurrency, log=log, dtype=loop.cfg.serve_dtype)
    tag = f"controlRR{iteration}"
    res = {"train": rep}
    ho = loop._tasks("status IN ('heldout','heldout-supp')")
    if ho: res["heldout"] = eval_set(loop, ho, policy_tag=tag, purpose="heldout:control", seeds=heldout_seeds, job_name=f"heldout-{tag}", iteration=iteration)
    dev = loop._tasks("status='dev'")
    if dev: res["dev"] = eval_set(loop, dev, policy_tag=tag, purpose="dev", seeds=dev_seeds, job_name=f"dev-{tag}", iteration=iteration)
    res["paired_dev_vs_Mk"] = paired_stats(task_means(loop.db, "dev", f"M{iteration}"), task_means(loop.db, "dev", tag))
    loop.db.event("control_random_reward", iteration=iteration, **{k: v for k, v in res.items() if k != "train"})
    return res
