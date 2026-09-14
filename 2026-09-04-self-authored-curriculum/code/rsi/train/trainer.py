"""LoRA policy-gradient / RFT trainer over stored turn records (docs/03 §6; smoke test 4 is the reference).

Input: episodes with rewards (SQLite) + turn records on disk (vLLM prompt_token_ids, token_ids, logprobs).
Advantage: leave-one-out group mean per task, broadcast to all the episode's turns; 0 for overflowed segments;
truncated / errored episodes masked. Loss: token-level clipped PG (eps 0.2/0.28), pi_old = stored vLLM logprobs.
Schedule: one epoch, minibatches of `tasks_per_minibatch` tasks (all their episodes' turns), AdamW 1e-5, clip 1.0.
The adapter (fp32 A/B) accumulates across iterations: pass --adapter-in to continue from the previous one.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from collections import defaultdict
from pathlib import Path
import torch

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
MASKED_STOPS = {"max_compactions", "prompt_too_long_for_window", "no_command"}   # truncations: no gradient (docs/02 accounting)

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def load_model(path, adapter_in: str | None, rank: int, alpha: int):
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "smoke"))
    from qwen35_kernels import assert_fast_path
    kern = assert_fast_path(); log("kernels:", kern)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    from peft import LoraConfig, get_peft_model, PeftModel
    if adapter_in:
        pm = PeftModel.from_pretrained(model, adapter_in, is_trainable=True)
        log("continuing adapter from", adapter_in)
    else:
        pm = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=TARGETS, task_type="CAUSAL_LM"))
    for n, p in pm.named_parameters():
        if "lora_" in n: p.data = p.data.float()          # fp32 adapter maths on a bf16 base
    hf = pm.base_model.model
    backbone = hf.model.language_model if hasattr(hf.model, "language_model") else hf.model
    return pm, backbone, hf.lm_head.weight

def build_examples(db_path: Path, purpose: str, iteration: int, group_size_min: int = 2, window: int = 16384,
                   max_turns_per_episode: int | None = 8, seed: int = 0, shuffle_rewards: bool = False):
    """Group episodes by task, compute RLOO advantages, expand to turn records.
    Every turn record re-forwards its full prompt, so an episode costs O(turns x prompt); to bound that, at most
    `max_turns_per_episode` turn records are sampled uniformly per episode and their advantage is scaled by
    n_turns / n_sampled (unbiased for the per-token objective)."""
    import sqlite3
    rng = random.Random(seed)
    conn = sqlite3.connect(str(db_path)); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT episode_id, task_id, reward, stop_reason, exception, quarantined FROM episodes WHERE purpose=? AND iteration=?", (purpose, iteration)).fetchall()
    rows = [dict(r) for r in rows if not (r["quarantined"] or 0) and not r["exception"] and r["reward"] is not None]
    if shuffle_rewards:   # random-reward control (docs/03 §8.3): permute rewards across the whole batch, batch mean preserved
        rs = [r["reward"] for r in rows]; random.Random(seed + 7919).shuffle(rs)
        for r, x in zip(rows, rs): r["reward"] = x
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    examples, stats = [], {"tasks": 0, "episodes": 0, "zero_var_groups": 0, "masked_episodes": 0, "turns": 0, "tokens": 0}
    for task, eps in by_task.items():
        if len(eps) < group_size_min: continue
        rs = [float(e["reward"]) for e in eps]
        if max(rs) == min(rs):
            stats["zero_var_groups"] += 1; continue          # dynamic sampling: no gradient from zero-variance groups
        stats["tasks"] += 1
        n = len(rs); tot = sum(rs)
        for e, r in zip(eps, rs):
            adv = r - (tot - r) / (n - 1)                     # leave-one-out baseline, no std normalisation
            masked = e["stop_reason"] in MASKED_STOPS
            if masked: stats["masked_episodes"] += 1; continue
            stats["episodes"] += 1
            recs = conn.execute("SELECT DISTINCT record_path FROM turns WHERE episode_id=?", (e["episode_id"],)).fetchall()
            ep_examples = []
            for rp in recs:
                with open(rp["record_path"]) as f:
                    for line in f:
                        rec = json.loads(line)
                        if "error" in rec or not rec.get("token_ids"): continue
                        p, g, lp = rec["prompt_token_ids"], rec["token_ids"], rec.get("logprobs") or []
                        if len(lp) != len(g) or len(p) + len(g) > window + 512: continue
                        ep_examples.append({"task": task, "episode": e["episode_id"], "prompt": p, "gen": g, "old_lp": lp, "adv": adv,
                                            "segment": rec.get("segment", 0), "turn": rec.get("turn", 0)})
            if max_turns_per_episode and len(ep_examples) > max_turns_per_episode:
                scale = len(ep_examples) / max_turns_per_episode
                ep_examples = rng.sample(ep_examples, max_turns_per_episode)
                for ex in ep_examples: ex["adv"] *= scale
                stats["turns_subsampled_from"] = stats.get("turns_subsampled_from", 0) + int(scale * max_turns_per_episode)
            for ex in ep_examples:
                examples.append(ex); stats["turns"] += 1; stats["tokens"] += len(ex["prompt"]) + len(ex["gen"])
    return examples, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--db", required=True); ap.add_argument("--purpose", default="rollout")
    ap.add_argument("--iteration", type=int, required=True); ap.add_argument("--adapter-in"); ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--objective", choices=["pg", "rft"], default="pg"); ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--tasks-per-minibatch", type=int, default=8)
    ap.add_argument("--clip-lo", type=float, default=0.2); ap.add_argument("--clip-hi", type=float, default=0.28)
    ap.add_argument("--chunk", type=int, default=1024); ap.add_argument("--window", type=int, default=16384)
    ap.add_argument("--rft-cap-per-task", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns-per-episode", type=int, default=8); ap.add_argument("--shuffle-rewards", action="store_true")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    from rsi.train.logprobs import sequence_logprobs
    examples, stats = build_examples(Path(a.db), a.purpose, a.iteration, window=a.window, max_turns_per_episode=a.max_turns_per_episode or None, seed=a.seed, shuffle_rewards=a.shuffle_rewards)
    stats["shuffle_rewards"] = a.shuffle_rewards
    log("examples:", stats)
    if a.objective == "rft":
        # verified-success turns only, per-task cap, plain NLL (lr typically 1e-4)
        succ = [ex for ex in examples if ex["adv"] > 0]
        by = defaultdict(list)
        for ex in succ: by[ex["episode"]].append(ex)
        keep_eps = defaultdict(list)
        for ep, exs in by.items(): keep_eps[exs[0]["task"]].append(ep)
        chosen = {ep for t, eps in keep_eps.items() for ep in eps[: a.rft_cap_per_task]}
        examples = [ex for ex in succ if ex["episode"] in chosen]
        for ex in examples: ex["adv"] = 1.0
        log(f"RFT: {len(examples)} turn records from {len(chosen)} successful episodes")
    if not examples:
        json.dump({"steps": 0, "stats": stats, "note": "no trainable examples"}, open(a.report, "w"), indent=1)
        log("nothing to train on"); return
    pm, backbone, W = load_model(a.model, a.adapter_in, a.rank, a.alpha)
    pm.train()
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    tasks = sorted({ex["task"] for ex in examples}); random.shuffle(tasks)
    minibatches = [tasks[i:i + a.tasks_per_minibatch] for i in range(0, len(tasks), a.tasks_per_minibatch)]
    log(f"{len(examples)} turn records, {len(tasks)} tasks, {len(minibatches)} minibatches")
    torch.cuda.reset_peak_memory_stats()
    steps, t0, diag = 0, time.time(), []
    chunk = a.chunk   # fp32 logits are chunked over positions; halved on OOM. Memory changes, the update does not.
    for mb_i, mb_tasks in enumerate(minibatches):
        mb = [ex for ex in examples if ex["task"] in set(mb_tasks)]
        n_tok = sum(len(ex["gen"]) for ex in mb)
        while True:
            opt.zero_grad(set_to_none=True)
            clip_frac = 0.0; absr = []; ess_num = 0.0; ess_den = 0.0; nll = 0.0
            try:
                for ex in mb:
                    ids = torch.tensor(ex["prompt"] + ex["gen"], device="cuda")
                    P = len(ex["prompt"]); G = len(ex["gen"])
                    lp = sequence_logprobs(backbone, W, ids, chunk)[P - 1: P - 1 + G]
                    old = torch.tensor(ex["old_lp"], device="cuda", dtype=torch.float32)
                    if a.objective == "rft":
                        loss = -(lp.sum()) / n_tok
                    else:
                        logr = lp - old
                        ratio = torch.exp(logr)
                        adv = ex["adv"]
                        pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - a.clip_lo, 1 + a.clip_hi) * adv)
                        loss = pg.sum() / n_tok
                        with torch.no_grad():
                            clip_frac += ((ratio < 1 - a.clip_lo) | (ratio > 1 + a.clip_hi)).float().sum().item()
                            absr.append(logr.abs().detach()); w = ratio.detach(); ess_num += w.sum().item(); ess_den += (w ** 2).sum().item()
                    loss.backward()
                    nll += float(-lp.detach().sum())
                    del lp, loss, ids
            except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
                # Triton reports its own "Triton Error [CUDA]: out of memory" as a plain RuntimeError
                if "out of memory" not in str(ex).lower() or chunk <= 128:
                    raise
                chunk //= 2
                opt.zero_grad(set_to_none=True); absr = []
                torch.cuda.empty_cache()
                log(f"OOM in minibatch {mb_i + 1}: discarding partial gradients and retrying it at chunk {chunk}")
                continue
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step(); steps += 1
            d = {"step": steps, "tasks": len(mb_tasks), "turns": len(mb), "gen_tokens": n_tok, "grad_norm": gn, "nll_per_tok": nll / max(n_tok, 1),
                 "elapsed_s": round(time.time() - t0, 1), "peak_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2), "chunk": chunk}
            if a.objective == "pg" and absr:
                ar = torch.cat(absr)
                d.update({"clip_frac": clip_frac / n_tok, "abs_logratio_mean": ar.mean().item(), "abs_logratio_q90": ar.quantile(0.9).item(),
                          "ess_norm": (ess_num ** 2 / max(ess_den, 1e-9)) / n_tok})
            diag.append(d); log(json.dumps(d))
            break
    pm.save_pretrained(a.adapter_out)
    rep = {"steps": steps, "final_chunk": chunk, "stats": stats, "diagnostics": diag, "wall_s": time.time() - t0, "objective": a.objective, "lr": a.lr,
           "adapter_out": a.adapter_out, "adapter_in": a.adapter_in, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    json.dump(rep, open(a.report, "w"), indent=1); log("saved", a.adapter_out)

if __name__ == "__main__":
    main()
