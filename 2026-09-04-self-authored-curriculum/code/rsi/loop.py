"""The iteration controller (docs/03 §4): a state machine with a durable checkpoint after every step,
pause/resume/rerun/inspect, and every step's outputs in SQLite + the run directory.

Steps per iteration k:  PROPOSE -> GATE -> ROLLOUT -> TIER -> ANALYSE -> AUDIT -> UPDATE -> DEVGATE -> EVALUATE
State file: <run>/state.json = {"iteration": k, "step": name, "status": running|done|failed, "paused": bool}
Pause: `loop pause` writes PAUSE; the controller stops at the next step boundary. Resume continues from the
last completed step. Rerun: `loop rerun STEP` clears that step and everything after it in the iteration.
"""
from __future__ import annotations
import json, os, random, shutil, subprocess, sys, time, traceback
from dataclasses import dataclass, asdict
from pathlib import Path

from rsi import config as C
from rsi.db import DB
from rsi.rollout import run_job, ingest_job
from rsi.tasks.gate import gate_tasks
from rsi.tiering import tier_from_rollout
from rsi import serve

STEPS = ["PROPOSE", "GATE", "ROLLOUT", "TIER", "ANALYSE", "AUDIT", "UPDATE", "DEVGATE", "EVALUATE"]

@dataclass
class LoopConfig:
    run: str
    base_model: str = str(C.BASE_MODEL)
    iterations: int = 7                     # K = 6 -> iterations 0..6
    candidates: int = 200
    group_size: int = 8
    anchor_fraction: float = 0.25
    max_train_tasks: int = 150
    rollout_concurrency: int = 32
    max_num_seqs: int = 32
    kv_dtype: str | None = None
    gate_concurrency: int = 6
    objective: str = "pg"
    lr: float = 1e-5
    tasks_per_minibatch: int = 8
    dev_seeds: int = 3
    dev_gate_z: float = 1.28
    thinking: bool = True
    brevity: bool = False
    max_tokens: int = 3072
    harness_window: int = 16384
    repair_rounds: int = 1
    serve_dtype: str = "bfloat16"          # bfloat16 | float16 (docs/04 6b fallback)
    canaries_every_iteration: bool = False
    canary_limit: int = 0
    interim_heldout_iterations: tuple = (2, 4)
    heldout_seeds: int = 5
    tw_seeds: int = 3
    non_recursive: bool = False             # control branch: iteration-0 (M_0, H_0) writes every round's tasks
    proposer_model_path: str | None = None  # for the control: the frozen M_0 served for PROPOSE/ANALYSE only

class Loop:
    def __init__(self, cfg: LoopConfig):
        self.cfg = cfg
        self.run = C.run_paths(cfg.run); self.run.mkdirs()
        self.db = DB(self.run.db)
        self.state_path = self.run.root / "state.json"
        self.pause_path = self.run.root / "PAUSE"
        self.improver_dir = self.run.improver
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"iteration": 0, "step": None, "status": "idle", "done": []}
        (self.run.root / "loop-config.json").write_text(json.dumps(asdict(cfg), indent=1))
        if not (self.improver_dir / "generator_prompt.md").exists():
            shutil.copytree(C.PROJECT_DIR / "improver", self.improver_dir, dirs_exist_ok=True)
            self._git(["init", "-q"]); self._git(["add", "-A"]); self._git(["commit", "-q", "-m", "H_0"], ok_fail=True); self._git(["tag", "H_0"], ok_fail=True)

    # ------------------------------------------------------------ bookkeeping
    def _git(self, args, ok_fail=False):
        r = subprocess.run(["git", "-C", str(self.improver_dir)] + args, capture_output=True, text=True)
        if r.returncode and not ok_fail: raise RuntimeError(r.stderr)
        return r.stdout
    def _save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1))
    def _log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [it{self.state['iteration']} {self.state.get('step')}] {msg}"
        print(line, flush=True)
        with open(self.run.logs / "loop.log", "a") as f: f.write(line + "\n")
    def _key(self, k, step): return f"{k}:{step}"
    def _done(self, k, step): return self._key(k, step) in self.state["done"]
    def _artifact(self, k: int, name: str) -> Path:
        p = self.run.root / "iterations" / f"{k:02d}"; p.mkdir(parents=True, exist_ok=True); return p / name
    def _policy_path(self, k: int) -> Path:
        return Path(self.cfg.base_model) if k == 0 else C.MERGED_PATH
    def _adapter(self, k: int) -> Path | None:
        p = self.run.adapters / f"adapter_{k:02d}"
        return p if p.exists() else None

    # ------------------------------------------------------------ control
    def pause(self): self.pause_path.write_text(str(time.time())); print("pause requested (takes effect at the next step boundary)")
    def unpause(self): self.pause_path.unlink(missing_ok=True)
    def rerun(self, step: str, iteration: int | None = None):
        k = self.state["iteration"] if iteration is None else iteration
        idx = STEPS.index(step)
        self.state["done"] = [d for d in self.state["done"] if not (d.startswith(f"{k}:") and STEPS.index(d.split(":")[1]) >= idx) and not (int(d.split(":")[0]) > k)]
        self.state.update({"iteration": k, "step": step, "status": "idle"}); self._save(); print(f"cleared {step}+ of iteration {k}")

    def run_loop(self, until_iteration: int | None = None):
        K = self.cfg.iterations if until_iteration is None else until_iteration + 1
        for k in range(self.state["iteration"], K):
            self.state["iteration"] = k
            for step in STEPS:
                if self._done(k, step): continue
                if self.pause_path.exists():
                    self.state.update({"step": step, "status": "paused"}); self._save(); self._log("paused"); return "paused"
                self.state.update({"step": step, "status": "running"}); self._save()
                self.db.set_step(k, step, "running"); t0 = time.time(); self._log("start")
                try:
                    getattr(self, "step_" + step.lower())(k)
                except Exception as ex:
                    self.state["status"] = "failed"; self._save(); self.db.set_step(k, step, "failed", notes=str(ex)[:500])
                    self._log(f"FAILED: {ex}\n{traceback.format_exc()}"); return "failed"
                self.state["done"].append(self._key(k, step)); self.state["status"] = "done"; self._save()
                self.db.set_step(k, step, "done", notes=f"{time.time() - t0:.0f}s"); self._log(f"done in {time.time() - t0:.0f}s")
                try:
                    from rsi.dashboard import build_dashboard; build_dashboard(self.run, self.db)
                except Exception as ex:
                    self._log(f"dashboard: {ex}")
        self.state["status"] = "complete"; self._save(); return "complete"

    # ------------------------------------------------------------ helpers
    def _serve(self, model_path: Path):
        stamp = self._stamped_with(model_path) if model_path == C.MERGED_PATH else None
        if serve.is_up():
            cur = (self.run.root / "serving.json")
            if cur.exists():
                j = json.loads(cur.read_text())
                # the adapter stamp matters as well as the path: the other run may have re-merged the same path
                if j.get("model") == str(model_path) and j.get("dtype", "bfloat16") == self.cfg.serve_dtype and j.get("adapter") == stamp:
                    return
            serve.stop()
        self._log(f"starting vLLM on {model_path} ({self.cfg.serve_dtype})")
        self.db.event("serve", model=str(model_path), dtype=self.cfg.serve_dtype, iteration=self.state.get("iteration"))
        serve.start(model_path, max_num_seqs=self.cfg.max_num_seqs, kv_dtype=self.cfg.kv_dtype, log_path=self.run.logs / "vllm.log", dtype=self.cfg.serve_dtype)
        t = serve.wait_healthy(); self._log(f"vLLM up in {t:.0f}s")
        (self.run.root / "serving.json").write_text(json.dumps({"model": str(model_path), "since": time.time(),
                                                                "dtype": self.cfg.serve_dtype, "adapter": stamp}))
    def _stop_serving(self):
        serve.stop(); (self.run.root / "serving.json").unlink(missing_ok=True)
    def _agent_kwargs(self, tag: str) -> dict:
        return {"base_url": C.VLLM_BASE_URL, "model": C.SERVED_MODEL_NAME, "policy_tag": tag, "thinking": str(self.cfg.thinking).lower(),
                "brevity": str(self.cfg.brevity).lower(), "max_tokens": self.cfg.max_tokens, "window": self.cfg.harness_window}
    def _thresholds(self, h_dir: Path | None = None) -> dict:
        """The improver's own bounded knobs. ANALYSE may edit them, so PROPOSE, ROLLOUT and TIER must read them
        rather than the static config; values are clamped to the pre-registered allowed ranges."""
        th = json.loads(((h_dir or self.improver_dir) / "thresholds.json").read_text())
        for key in ("anchor_fraction", "candidates_per_iteration"):
            rng = (th.get("allowed_ranges") or {}).get(key)
            if key in th and rng: th[key] = min(max(th[key], rng[0]), rng[1])
        return th

    def _tasks(self, where: str, *args) -> list[Path]:
        return [Path(r["path"]) for r in self.db.q(f"SELECT path FROM tasks WHERE {where}", *args)]

    # ------------------------------------------------------------ steps
    def step_propose(self, k: int):
        from rsi.improver.propose import PolicyAuthor, propose_batch
        proposer_model = Path(self.cfg.proposer_model_path) if (self.cfg.non_recursive and self.cfg.proposer_model_path) else self._ensure_head_merged(k)
        h_dir = (C.PROJECT_DIR / "improver") if self.cfg.non_recursive else self.improver_dir
        self._serve(proposer_model)
        ex = [{"task_id": r["task_id"], "instruction": (Path(r["path"]) / "instruction.md").read_text()}
              for r in self.db.q("SELECT task_id, path FROM tasks WHERE status LIKE 'gated%' AND (tier IN ('medium','hard') OR tier IS NULL)")]
        author = PolicyAuthor(C.VLLM_BASE_URL, C.SERVED_MODEL_NAME, thinking=self.cfg.thinking)
        n_cand = int(self._thresholds(h_dir).get("candidates_per_iteration", self.cfg.candidates))
        self._log(f"proposing {n_cand} candidates (improver threshold; config default {self.cfg.candidates})")
        rows = propose_batch(h_dir, self.run.tasks, author, n=n_cand, exemplar_pool=ex, seed=1000 + k, source="policy",
                             iteration=k, log=self._log)
        parsed = [r for r in rows if r.get("parsed")]
        for r in parsed:
            self.db.upsert_task(r["task_id"], f"rsi/{r['task_id']}", r["path"], "policy", domain=r["meta"]["domain"], tier=r["meta"]["tier"],
                                iteration_added=k, status="candidate", provenance=json.loads((Path(r["path"]) / "provenance.json").read_text()))
        self._artifact(k, "propose.json").write_text(json.dumps(rows, indent=1, default=str))
        self.db.metric("propose_parsed", len(parsed), k, "PROPOSE"); self.db.metric("propose_n", len(rows), k, "PROPOSE")
        self._log(f"{len(parsed)}/{len(rows)} candidates parsed")

    def step_gate(self, k: int):
        from rsi.tasks.dedup import dedup_against_pool
        cands = self._tasks("status='candidate' AND iteration_added=?", k)
        kept, dups = dedup_against_pool(self.db, cands)
        for td in dups:
            self.db.gate(td.name, "dedup", "fail", None, k); self.db.upsert_task(td.name, td.name, td, None, status="rejected:dedup")
        v = gate_tasks(self.db, self.run, kept, iteration=k, n_concurrent=self.cfg.gate_concurrency, batch_tag=f"it{k:02d}-gate") if kept else {}
        # repair round: the author sees its own oracle/pytest failure once and rewrites (executable feedback, still self-written)
        failed = [td for td in kept if v.get(td.name, "").startswith("rejected:") and "oracle" in v[td.name]]
        if failed and self.cfg.repair_rounds > 0:
            from rsi.improver.propose import PolicyAuthor, feedback_for, repair_task
            self._serve(self._policy_path(k) if not self.cfg.non_recursive else Path(self.cfg.proposer_model_path or self.cfg.base_model))
            author = PolicyAuthor(C.VLLM_BASE_URL, C.SERVED_MODEL_NAME, thinking=self.cfg.thinking)
            jobs = [self.run.jobs / f"it{k:02d}-gate-oracle", self.run.jobs / f"it{k:02d}-gate-oracle-root"]
            # the same rewrite as before, in parallel: it was serial at about 45 s a call, which made the repair
            # round, not the container gate, the expensive half of this step (docs/06 2026-09-09)
            from concurrent.futures import ThreadPoolExecutor
            def _repair_one(td):
                try:
                    return repair_task(author, td, feedback_for(td, jobs), self.run.tasks, iteration=k, source="policy", log=self._log)
                except Exception as ex:
                    self._log(f"repair failed for {td.name}: {type(ex).__name__}: {ex}"); return None
            with ThreadPoolExecutor(max_workers=16) as pool_ex:
                results = list(pool_ex.map(_repair_one, failed))
            repaired = []
            for r in results:                      # database writes stay on this thread
                if r:
                    tid, dest = r
                    self.db.upsert_task(tid, f"rsi/{tid}", dest, "policy", iteration_added=k, status="candidate",
                                        provenance=json.loads((dest / "provenance.json").read_text()))
                    repaired.append(dest)
            self._log(f"repair round: {len(repaired)}/{len(failed)} rewritten")
            if repaired:
                v2 = gate_tasks(self.db, self.run, repaired, iteration=k, n_concurrent=self.cfg.gate_concurrency, batch_tag=f"it{k:02d}-gate-repair")
                v.update(v2)
                self.db.metric("gate_repaired_survivors", sum(1 for x in v2.values() if x.startswith("gated")), k, "GATE")
        n_gated = sum(1 for x in v.values() if x.startswith("gated"))
        self.db.metric("gate_survivors", n_gated, k, "GATE"); self.db.metric("gate_dedup_rejects", len(dups), k, "GATE")
        self._log(f"gated {n_gated}/{len(cands)} (dedup rejected {len(dups)})")

    def step_rollout(self, k: int):
        self._serve(self._ensure_head_merged(k))
        new = self._tasks("status LIKE 'gated%' AND source='policy' AND iteration_added=?", k)
        pool_old = self._tasks("status LIKE 'gated%' AND source='policy' AND iteration_added<?", k)
        seeds = self._tasks("status='gated' AND source IN ('seta_evolve','seta_synth','hand') OR (status='gated-nosolution' AND source='nemotron')")
        rng = random.Random(k)
        anchor_fraction = float(self._thresholds().get("anchor_fraction", self.cfg.anchor_fraction))
        n_anchor = max(int(anchor_fraction * max(len(new), 20)), 8)
        anchors = rng.sample(seeds, min(n_anchor, len(seeds)))
        old = rng.sample(pool_old, min(len(pool_old), max(0, self.cfg.max_train_tasks - len(new) - len(anchors))))
        tasks = new + anchors + old
        frontier = self._tasks("tier='frontier' AND status LIKE 'gated%'")
        self._log(f"rollout: {len(new)} new + {len(anchors)} anchors + {len(old)} pool + {len(frontier)} frontier tasks, G={self.cfg.group_size}")
        job = run_job(self.run, tasks, job_name=f"it{k:02d}-rollout", agent="rsi", n_concurrent=self.cfg.rollout_concurrency,
                      n_attempts=self.cfg.group_size, agent_kwargs=self._agent_kwargs(f"M{k}"))
        st = ingest_job(self.db, job, policy=f"M{k}", purpose="rollout", iteration=k)
        if frontier:
            job2 = run_job(self.run, frontier, job_name=f"it{k:02d}-frontier", agent="rsi", n_concurrent=self.cfg.rollout_concurrency,
                           n_attempts=4, agent_kwargs=self._agent_kwargs(f"M{k}"))
            ingest_job(self.db, job2, policy=f"M{k}", purpose="rollout", iteration=k)
        self.db.metric("rollout_episodes", st["trials"], k, "ROLLOUT"); self.db.metric("rollout_mean_reward", st["mean_reward"] or 0, k, "ROLLOUT")
        self._log(f"rollout done: {st}")

    def step_tier(self, k: int):
        th = self._thresholds()
        res = tier_from_rollout(self.db, "rollout", k, tuple(th["learning_band"]))
        self._artifact(k, "tier.json").write_text(json.dumps(res, indent=1))
        self._log(f"tier: train {len(res['train'])}, frontier {len(res['frontier'])}, drop {len(res['drop'])}")

    def step_analyse(self, k: int):
        if self.cfg.non_recursive:
            self._log("non-recursive control: improver frozen at H_0"); return
        from rsi.improver.analyse import analyse_and_edit
        self._serve(self._policy_path(k))
        rep = analyse_and_edit(self.db, self.run, self.improver_dir, k, base_url=C.VLLM_BASE_URL, model=C.SERVED_MODEL_NAME,
                               thinking=self.cfg.thinking, log=self._log)
        self._artifact(k, "analyse.json").write_text(json.dumps(rep, indent=1, default=str))
        self._git(["add", "-A"]); self._git(["commit", "-q", "-m", f"H_{k + 1} candidate after iteration {k}"], ok_fail=True)

    def step_audit(self, k: int):
        from rsi.audit import audit_iteration
        self._stop_serving()
        rep = audit_iteration(self.db, self.run, k, log=self._log)
        self._artifact(k, "audit.json").write_text(json.dumps(rep, indent=1, default=str))

    def step_update(self, k: int):
        self._stop_serving()
        adapter_out = self.run.adapters / f"adapter_{k + 1:02d}"
        report = self._artifact(k, "train.json")
        cmd = [str(C.VENVS["train"] / "bin" / "python"), "-m", "rsi.train.trainer", "--model", self.cfg.base_model, "--db", str(self.run.db),
               "--purpose", "rollout", "--iteration", str(k), "--adapter-out", str(adapter_out), "--objective", self.cfg.objective,
               "--lr", str(self.cfg.lr), "--tasks-per-minibatch", str(self.cfg.tasks_per_minibatch), "--window", str(self.cfg.harness_window),
               "--report", str(report)]
        prev = self._adapter(k)
        if prev: cmd += ["--adapter-in", str(prev)]
        self._log("training: " + " ".join(cmd))
        with open(self.run.logs / f"train-it{k:02d}.log", "w") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(C.PROJECT_DIR),
                               env=dict(os.environ, PYTHONPATH=str(C.PROJECT_DIR),
                                        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))  # peak was 30.1 of 31.4 GiB in iteration 0
        if r.returncode: raise RuntimeError(f"trainer failed (see logs/train-it{k:02d}.log)")
        rep = json.loads(report.read_text())
        if rep.get("steps", 0) == 0:
            self._log("no optimiser steps this iteration (no trainable groups); policy unchanged")
            if prev: shutil.copytree(prev, adapter_out, dirs_exist_ok=True)
            else: return
        merge_cmd = [str(C.VENVS["train"] / "bin" / "python"), str(C.PROJECT_DIR / "smoke/05-merge.py"), "--base", self.cfg.base_model,
                     "--adapter", str(adapter_out), "--out", str(C.MERGED_PATH), "--report", str(self._artifact(k, "merge.json")), "--dtype", self.cfg.serve_dtype]
        with open(self.run.logs / f"merge-it{k:02d}.log", "w") as lf:
            r = subprocess.run(merge_cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(C.PROJECT_DIR))
        if r.returncode: raise RuntimeError("merge failed")
        for d in rep.get("diagnostics", []):
            for key in ("grad_norm", "clip_frac", "abs_logratio_mean", "abs_logratio_q90", "ess_norm", "nll_per_tok"):
                if key in d: self.db.metric(f"train_{key}", d[key], k, "UPDATE", opt_step=d["step"])
        self._stamp(C.MERGED_PATH, adapter_out)
        self._log(f"trained {rep.get('steps')} steps; merged to {C.MERGED_PATH}")

    def step_devgate(self, k: int):
        from rsi.evaluate import dev_eval, paired_gate
        adapter_out = self.run.adapters / f"adapter_{k + 1:02d}"
        if not adapter_out.exists():
            self._log("no new adapter; skipping dev gate"); return
        self._serve(C.MERGED_PATH)
        new = dev_eval(self, k + 1, tag=f"M{k + 1}")
        head_k = self.state.get("lineage_head", 0)
        # same prefix rule as task_means: the M_0 baseline is stored under the tagged purpose "dev:start"
        ref = self.db.q("SELECT AVG(reward) m FROM episodes WHERE (purpose='dev' OR purpose LIKE 'dev:%') AND policy=?", f"M{head_k}")[0]["m"]
        if ref is None:
            self._stop_serving(); self._serve(self._ensure_head_merged(head_k))
        verdict = paired_gate(self.db, f"M{head_k}", f"M{k + 1}", z=self.cfg.dev_gate_z)
        self._artifact(k, "devgate.json").write_text(json.dumps(verdict, indent=1))
        if verdict["accept"]:
            self.state["lineage_head"] = k + 1; self._log(f"accepted M{k + 1} as lineage head: {verdict}")
        else:
            self._log(f"REJECTED M{k + 1}: {verdict}; keeping M{head_k}")
            (self.run.adapters / f"adapter_{k + 1:02d}").rename(self.run.adapters / f"rejected_adapter_{k + 1:02d}")
            prev = self._adapter(head_k)
            if prev:
                shutil.copytree(prev, self.run.adapters / f"adapter_{k + 1:02d}")
                self._stop_serving(); self.merge_adapter(prev, C.MERGED_PATH, tag=f"restore-M{head_k}")   # the served path must hold the head again
            else:
                self._stop_serving(); shutil.rmtree(C.MERGED_PATH, ignore_errors=True); self._stamp(C.MERGED_PATH, None)
        self._save()

    def _stamp(self, path: Path, adapter: Path | None) -> None:
        """Record which adapter a fixed merged path currently holds. Two runs (main and the non-recursive branch)
        share these paths, so serving one without checking the stamp would serve the other run's weights."""
        m = path / ".rsi-adapter"
        if adapter is None: m.unlink(missing_ok=True)
        else: m.write_text(str(adapter))

    def _stamped_with(self, path: Path) -> str | None:
        m = path / ".rsi-adapter"
        return m.read_text().strip() if m.exists() else None

    def _ensure_head_merged(self, k: int) -> Path:
        """Serve-ready path for policy M_k of THIS run: re-merge the fixed path if it holds another run's adapter."""
        if k == 0: return Path(self.cfg.base_model)
        adapter = self._adapter(k)
        if adapter is None: raise RuntimeError(f"no adapter_{k:02d} in {self.run.root}")
        if self._stamped_with(C.MERGED_PATH) != str(adapter) or not C.MERGED_PATH.exists():
            self._log(f"merged path holds {self._stamped_with(C.MERGED_PATH)}; re-merging {adapter}")
            self._stop_serving(); self.merge_adapter(adapter, C.MERGED_PATH, tag=f"head-M{k}")
        return C.MERGED_PATH

    def merge_adapter(self, adapter: Path, out: Path, *, tag: str) -> None:
        cmd = [str(C.VENVS["train"] / "bin" / "python"), str(C.PROJECT_DIR / "smoke/05-merge.py"), "--base", self.cfg.base_model,
               "--adapter", str(adapter), "--out", str(out), "--report", str(self.run.logs / f"merge-{tag}.json"), "--dtype", self.cfg.serve_dtype]
        with open(self.run.logs / f"merge-{tag}.log", "w") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(C.PROJECT_DIR))
        if r.returncode: raise RuntimeError(f"merge failed ({tag})")
        self._stamp(out, adapter)

    def serve_policy(self, policy_tag: str) -> Path:
        """Serve a named policy for evaluation: M0 = base; Mk = adapter_k merged (on the ALT path unless it is the lineage
        head already on the main path); control/ablation tags = their adapter merged on the ALT path."""
        head = self.state.get("lineage_head", 0)
        if policy_tag == "M0":
            path = Path(self.cfg.base_model)
        elif policy_tag.startswith("M") and policy_tag[1:].isdigit():
            k = int(policy_tag[1:]); adapter = self._adapter(k)
            if adapter is None: raise RuntimeError(f"no adapter for {policy_tag}")
            if k == head and C.MERGED_PATH.exists() and self._stamped_with(C.MERGED_PATH) == str(adapter):
                path = C.MERGED_PATH
            else:
                path = C.MERGED_ALT_PATH
                marker = path / ".rsi-adapter"
                if not (marker.exists() and marker.read_text().strip() == str(adapter)):
                    self._stop_serving(); self.merge_adapter(adapter, path, tag=f"eval-{policy_tag}"); marker.write_text(str(adapter))
        else:
            cand = {"ablationRFT": "ablation_rft_", "controlRR": "control_random_reward_"}
            key = next((v for kk, v in cand.items() if policy_tag.startswith(kk)), None)
            if key is None: raise RuntimeError(f"unknown policy tag {policy_tag}")
            n = int(policy_tag[len([kk for kk in cand if policy_tag.startswith(kk)][0]):])
            adapter = self.run.adapters / f"{key}{n if key.startswith('control') else n - 1:02d}"
            path = C.MERGED_ALT_PATH; marker = path / ".rsi-adapter"
            if not (marker.exists() and marker.read_text().strip() == str(adapter)):
                self._stop_serving(); self.merge_adapter(adapter, path, tag=f"eval-{policy_tag}"); marker.write_text(str(adapter))
        self._serve(path)
        return path

    def step_evaluate(self, k: int):
        """Scheduled evaluations (docs/03 §4 step 8): canaries every iteration; interim held-out x5 and TerminalWorld x3 at the
        configured iterations (descriptive only, never used for selection or stopping). Evaluates the new lineage head."""
        from rsi.evaluate import eval_set
        head = self.state.get("lineage_head", 0)
        tag = f"M{head}"
        model_path = self._ensure_head_merged(head)
        done = []
        if self.cfg.canaries_every_iteration:
            from rsi.canary import run_canaries
            self._serve(model_path)
            rep = run_canaries(self.run, tag, limit=self.cfg.canary_limit or None)
            self._artifact(k, f"canaries-{tag}.json").write_text(json.dumps(rep, indent=1))
            if rep.get("composite") is not None: self.db.metric("canary_composite", rep["composite"], k, "EVALUATE", policy=tag)
            done.append(f"canaries {rep.get('composite')}")
        if k in self.cfg.interim_heldout_iterations:
            ho = self._tasks("status IN ('heldout','heldout-supp')")
            if ho:
                self._serve(model_path)
                st = eval_set(self, ho, policy_tag=tag, purpose="heldout:interim", seeds=self.cfg.heldout_seeds, job_name=f"heldout-interim-it{k:02d}-{tag}", iteration=k)
                self.db.metric("heldout_interim_mean", st["mean_reward"] or 0, k, "EVALUATE", policy=tag); done.append(f"heldout {st['mean_reward']}")
            tw = self._tasks("status='tw'")
            if tw:
                self._serve(model_path)
                st = eval_set(self, tw, policy_tag=tag, purpose="tw:interim", seeds=self.cfg.tw_seeds, job_name=f"tw-interim-it{k:02d}-{tag}", iteration=k)
                self.db.metric("tw_interim_mean", st["mean_reward"] or 0, k, "EVALUATE", policy=tag); done.append(f"tw {st['mean_reward']}")
        self._log("EVALUATE: " + ("; ".join(done) if done else "nothing scheduled this iteration"))

    def branch(self, to_run: str, at_iteration: int, *, non_recursive: bool = True) -> "Loop":
        """Create the non-recursive control branch: copy this run's state after iteration `at_iteration` into a new run
        whose PROPOSE/ANALYSE use the frozen iteration-0 pair (base model + H_0) while solving/training continue (docs/03 §8.1)."""
        src = self.run; dst = C.run_paths(to_run)
        if dst.root.exists(): raise RuntimeError(f"run {to_run} exists")
        shutil.copytree(src.root, dst.root, symlinks=True, ignore=shutil.ignore_patterns("jobs", "dashboard", "logs"))
        dst.mkdirs()
        st = json.loads((dst.root / "state.json").read_text())
        st["done"] = [d for d in st["done"] if int(d.split(":")[0]) <= at_iteration]
        st.update({"iteration": at_iteration + 1, "step": None, "status": "idle", "branched_from": str(src.root), "branch_at": at_iteration})
        (dst.root / "state.json").write_text(json.dumps(st, indent=1))
        cfg = LoopConfig(**{**self.cfg.__dict__, "run": to_run, "non_recursive": non_recursive, "proposer_model_path": self.cfg.base_model})
        (dst.root / "loop-config.json").write_text(json.dumps(asdict(cfg), indent=1))
        first = subprocess.run(["git", "-C", str(dst.improver), "rev-list", "--max-parents=0", "HEAD"], capture_output=True, text=True).stdout.strip().splitlines()
        target = "H_0" if subprocess.run(["git", "-C", str(dst.improver), "rev-parse", "-q", "--verify", "H_0"], capture_output=True).returncode == 0 else (first[-1] if first else None)
        if target is None: raise RuntimeError("cannot find H_0 in the run's improver repository")
        r = subprocess.run(["git", "-C", str(dst.improver), "checkout", "-q", target], capture_output=True, text=True)
        if r.returncode: raise RuntimeError(f"checkout of H_0 failed: {r.stderr}")
        self._log(f"branched to {to_run} after iteration {at_iteration} (non_recursive={non_recursive})")
        return Loop(cfg)
