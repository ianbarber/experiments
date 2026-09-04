#!/usr/bin/env python3
"""
Phase 2: Inject JEPA-predicted hidden states during decoding.

Tests whether injecting JEPA predictions into the model's hidden states
during generation improves problem-solving on GSM8K.

Injection formula: h_modified = h_original + α × (JEPA(h_problem) - h_problem)

Experiments:
  p2.1 — Alpha sweep (fix K=5, sweep alphas, include baseline)
  p2.2 — K sweep (fix alpha from p2.1, sweep Ks)
  p2.3 — Full eval (best alpha+K, all 4 modes with McNemar's test)

Usage:
    python src/inject_jepa.py --experiment p2.1
    python src/inject_jepa.py --experiment p2.2 --alphas 0.01
    python src/inject_jepa.py --experiment p2.3 --alphas 0.01 --Ks 5
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from train_jepa import JEPAPredictor
from extract_states import extract_number, check_gsm8k, build_prompt


# ---------------------------------------------------------------------------
# Injection context
# ---------------------------------------------------------------------------

class InjectionContext:
    """Manages per-generation injection state for the forward hook."""

    def __init__(self, jepa_model, alpha, K, mode, mean_delta=None, device="cuda"):
        self.jepa_model = jepa_model
        self.alpha = alpha
        self.K = K
        self.mode = mode  # "none", "jepa", "random", "mean_delta"
        self.mean_delta = mean_delta  # (hidden_dim,) tensor, precomputed
        self.device = device

        # Per-generation state (reset before each generate call)
        self.h_problem = None  # (batch, hidden_dim)
        self.delta = None      # (batch, hidden_dim)
        self.step_count = 0
        self.active = False

    def reset(self):
        """Reset per-generation state. Call before each model.generate()."""
        self.h_problem = None
        self.delta = None
        self.step_count = 0
        self.active = True

    def deactivate(self):
        """Deactivate the hook (no-op mode)."""
        self.active = False

    def hook_fn(self, module, input, output):
        """Forward hook for Qwen3DecoderLayer at the injection layer.

        - Prefill (seq_len > 1): extract problem state, compute delta
        - Decode steps 1..K: inject alpha * delta
        - After K steps: no modification
        """
        if not self.active or self.mode == "none":
            return output

        # Qwen3DecoderLayer returns a tuple: (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        batch_size, seq_len, hidden_dim = hidden.shape

        if seq_len > 1:
            # Prefill step: extract problem state from last token position
            # With left-padding, last position is always a real token
            self.h_problem = hidden[:, -1, :].clone()  # (batch, hidden_dim)
            self._compute_delta(hidden_dim)
            self.step_count = 0
            # Don't modify during prefill
        elif seq_len == 1:
            # Decode step
            self.step_count += 1
            if self.delta is not None and self.step_count <= self.K:
                # Inject: h_modified = h_original + alpha * delta
                modified = hidden.clone()
                # delta is (batch, hidden_dim), hidden is (batch, 1, hidden_dim)
                modified[:, 0, :] = modified[:, 0, :] + self.alpha * self.delta
                if rest is not None:
                    return (modified,) + rest
                return modified

        if rest is not None:
            return (hidden,) + rest
        return hidden

    def _compute_delta(self, hidden_dim):
        """Compute the injection delta based on mode."""
        if self.mode == "jepa":
            with torch.no_grad():
                h_pred = self.jepa_model(self.h_problem.float()).to(self.h_problem.dtype)
            self.delta = h_pred - self.h_problem  # (batch, hidden_dim)

        elif self.mode == "random":
            # Random direction with same L2 norm as JEPA delta
            with torch.no_grad():
                h_pred = self.jepa_model(self.h_problem.float()).to(self.h_problem.dtype)
            jepa_delta = h_pred - self.h_problem
            norm = jepa_delta.norm(dim=-1, keepdim=True)  # (batch, 1)
            random_dir = torch.randn_like(jepa_delta)
            random_dir = random_dir / random_dir.norm(dim=-1, keepdim=True)
            self.delta = random_dir * norm

        elif self.mode == "mean_delta":
            # Use precomputed mean(first_step - problem) from training data
            # Broadcast to batch size
            batch_size = self.h_problem.shape[0]
            self.delta = self.mean_delta.unsqueeze(0).expand(batch_size, -1).to(
                device=self.h_problem.device, dtype=self.h_problem.dtype
            )

        else:
            self.delta = None


# ---------------------------------------------------------------------------
# Wilson score confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(n_success, n_total, z=1.96):
    """Wilson score 95% confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return p, max(0, center - spread), min(1, center + spread)


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(correct_a, correct_b):
    """McNemar's test for paired binary outcomes.

    Args:
        correct_a: list of bool, per-example correctness for condition A
        correct_b: list of bool, per-example correctness for condition B

    Returns:
        dict with b (A wrong, B right), c (A right, B wrong), p_value
    """
    assert len(correct_a) == len(correct_b)
    # b = number where A is wrong and B is right
    # c = number where A is right and B is wrong
    b = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)

    # McNemar's test with continuity correction
    if b + c == 0:
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        # p-value from chi-squared distribution with 1 df
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {"b_wrong_a_right_b": b, "c_right_a_wrong_b": c, "p_value": p_value}


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, device="cuda"):
    """Load model with attention backend fallback and left-padding tokenizer."""
    print(f"Loading model: {model_name}")

    # Try attention backends in order
    model = None
    for attn_try in ["flash_attention_2", "sdpa"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
                attn_implementation=attn_try,
            )
            print(f"  Loaded with attn_implementation=\"{attn_try}\"")
            break
        except (ValueError, ImportError) as e:
            print(f"  Could not load with {attn_try}: {e}")
            continue

    if model is None:
        print(f"  Loading with default attention implementation")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def load_jepa(checkpoint_path, device="cuda"):
    """Load JEPA predictor from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = JEPAPredictor(
        dim=config["dim"],
        hidden_mult=config["hidden_mult"],
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"  JEPA loaded: dim={config['dim']}, layer={config['layer']}")
    return model, config


def compute_mean_delta(data_dir, layer, device="cuda"):
    """Compute mean(first_step - problem) from training data at given layer."""
    data_dir = Path(data_dir)
    problem = torch.load(data_dir / "problem_states.pt", map_location="cpu", weights_only=True).float()
    first_step = torch.load(data_dir / "first_step_states.pt", map_location="cpu", weights_only=True).float()

    # Extract the target layer: shape (N, L, D) → (N, D)
    p = problem[:, layer]
    f = first_step[:, layer]
    mean_d = (f - p).mean(dim=0)  # (D,)
    print(f"  Mean delta computed from {p.shape[0]} pairs, norm={mean_d.norm():.2f}")
    return mean_d.to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(model, tokenizer, jepa_model, dataset, layer,
                       alpha, K, mode, mean_delta, batch_size,
                       max_new_tokens, output_dir, device):
    """Evaluate a single injection condition on the full GSM8K test set.

    Returns dict with solve_rate, ci_low, ci_high, per_example_correct, avg_tokens.
    """
    condition_name = f"{mode}_a{alpha}_K{K}"
    print(f"\n--- Evaluating: mode={mode}, alpha={alpha}, K={K} ---")

    # Set up injection context
    ctx = InjectionContext(
        jepa_model=jepa_model,
        alpha=alpha,
        K=K,
        mode=mode,
        mean_delta=mean_delta,
        device=device,
    )

    # Register hook on the target layer
    target_layer = model.model.layers[layer]
    hook_handle = target_layer.register_forward_hook(ctx.hook_fn)

    per_example = []  # list of dicts
    correct_count = 0
    total_count = 0
    total_tokens = 0

    questions = list(dataset["question"])
    answers = list(dataset["answer"])
    n = len(questions)

    pbar = tqdm(total=n, desc=condition_name, leave=False)

    batch_idx = 0
    while batch_idx < n:
        batch_end = min(batch_idx + batch_size, n)
        batch_q = questions[batch_idx:batch_end]
        batch_a = answers[batch_idx:batch_end]
        cur_bs = len(batch_q)

        # Build prompts with chat template
        prompt_texts = [build_prompt(q, "gsm8k") for q in batch_q]
        all_input_ids = []
        per_example_prompt_lens = []
        for pt in prompt_texts:
            messages = [{"role": "user", "content": pt}]
            chat_out = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if isinstance(chat_out, torch.Tensor):
                ids = chat_out[0]
            else:
                ids = chat_out["input_ids"][0]
            all_input_ids.append(ids)
            per_example_prompt_lens.append(ids.shape[0])

        # Left-pad
        max_prompt_len = max(per_example_prompt_lens)
        padded_input_ids = torch.full(
            (cur_bs, max_prompt_len), tokenizer.pad_token_id, dtype=torch.long,
        )
        attention_mask = torch.zeros((cur_bs, max_prompt_len), dtype=torch.long)
        for i, ids in enumerate(all_input_ids):
            plen = ids.shape[0]
            padded_input_ids[i, max_prompt_len - plen:] = ids
            attention_mask[i, max_prompt_len - plen:] = 1

        # Reset injection context for this batch
        ctx.reset()

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                padded_input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        padded_prompt_len = padded_input_ids.shape[1]

        # Process each example
        for i in range(cur_bs):
            total_count += 1
            prompt_len = per_example_prompt_lens[i]

            generated_ids = output_ids[i, padded_prompt_len:]
            # Remove trailing pad tokens
            non_pad_mask = generated_ids != tokenizer.pad_token_id
            if non_pad_mask.any():
                last_non_pad = non_pad_mask.nonzero()[-1].item() + 1
                generated_ids = generated_ids[:last_non_pad]
            else:
                generated_ids = generated_ids[:0]

            # Remove trailing eos
            if len(generated_ids) > 0 and generated_ids[-1].item() == tokenizer.eos_token_id:
                generated_ids_for_decode = generated_ids[:-1]
            else:
                generated_ids_for_decode = generated_ids

            generated_text = tokenizer.decode(generated_ids_for_decode, skip_special_tokens=True)
            n_tokens = len(generated_ids)
            total_tokens += n_tokens

            is_correct = check_gsm8k(generated_text, batch_a[i])
            if is_correct:
                correct_count += 1

            per_example.append({
                "idx": batch_idx + i,
                "correct": is_correct,
                "n_tokens": n_tokens,
                "question": batch_q[i][:200],
                "generated": generated_text[:500],
                "reference": batch_a[i][:200],
            })

        pbar.update(cur_bs)
        batch_idx = batch_end

    pbar.close()

    # Deactivate and remove hook
    ctx.deactivate()
    hook_handle.remove()

    # Compute metrics
    rate, ci_low, ci_high = wilson_ci(correct_count, total_count)
    avg_tokens = total_tokens / max(total_count, 1)
    per_correct = [ex["correct"] for ex in per_example]

    print(f"  Solve rate: {rate:.4f} [{ci_low:.4f}, {ci_high:.4f}] "
          f"({correct_count}/{total_count}), avg tokens: {avg_tokens:.1f}")

    # Save per-example JSONL
    if output_dir:
        per_dir = Path(output_dir) / "per_example"
        per_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = per_dir / f"{condition_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for ex in per_example:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return {
        "mode": mode,
        "alpha": alpha,
        "K": K,
        "solve_rate": rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "correct_count": correct_count,
        "total_count": total_count,
        "avg_tokens": avg_tokens,
        "per_example_correct": per_correct,
    }


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_p2_1(model, tokenizer, jepa_model, dataset, layer, alphas, batch_size,
             max_new_tokens, output_dir, device, mean_delta):
    """P2.1: Alpha sweep with fixed K=5."""
    K = 5
    results = []

    # Baseline (no injection)
    res = evaluate_condition(
        model, tokenizer, jepa_model, dataset, layer,
        alpha=0.0, K=0, mode="none", mean_delta=mean_delta,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        output_dir=output_dir, device=device,
    )
    results.append(res)
    baseline_correct = res["per_example_correct"]

    # Sweep alphas with JEPA injection
    for alpha in alphas:
        res = evaluate_condition(
            model, tokenizer, jepa_model, dataset, layer,
            alpha=alpha, K=K, mode="jepa", mean_delta=mean_delta,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            output_dir=output_dir, device=device,
        )
        results.append(res)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"P2.1 Results — Alpha Sweep (K={K})")
    print(f"{'='*70}")
    print(f"  {'Mode':<8} {'Alpha':>8} {'Solve Rate':>12} {'95% CI':>20} {'Avg Tok':>8}")
    print(f"  {'─'*60}")
    for r in results:
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        print(f"  {r['mode']:<8} {r['alpha']:>8.4f} {r['solve_rate']:>12.4f} {ci_str:>20} {r['avg_tokens']:>8.1f}")

    # Save results (strip per_example_correct for JSON)
    save_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "per_example_correct"}
        save_results.append(sr)

    out_path = Path(output_dir) / "p2.1_results.json"
    with open(out_path, "w") as f:
        json.dump({"experiment": "p2.1", "K": K, "results": save_results}, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    return results


def run_p2_2(model, tokenizer, jepa_model, dataset, layer, alpha, Ks, batch_size,
             max_new_tokens, output_dir, device, mean_delta):
    """P2.2: K sweep with fixed alpha."""
    results = []

    # Baseline
    res = evaluate_condition(
        model, tokenizer, jepa_model, dataset, layer,
        alpha=0.0, K=0, mode="none", mean_delta=mean_delta,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        output_dir=output_dir, device=device,
    )
    results.append(res)

    # Sweep Ks
    for K in Ks:
        res = evaluate_condition(
            model, tokenizer, jepa_model, dataset, layer,
            alpha=alpha, K=K, mode="jepa", mean_delta=mean_delta,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            output_dir=output_dir, device=device,
        )
        results.append(res)

    # Print summary
    print(f"\n{'='*70}")
    print(f"P2.2 Results — K Sweep (alpha={alpha})")
    print(f"{'='*70}")
    print(f"  {'Mode':<8} {'K':>4} {'Solve Rate':>12} {'95% CI':>20} {'Avg Tok':>8}")
    print(f"  {'─'*56}")
    for r in results:
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        print(f"  {r['mode']:<8} {r['K']:>4} {r['solve_rate']:>12.4f} {ci_str:>20} {r['avg_tokens']:>8.1f}")

    save_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "per_example_correct"}
        save_results.append(sr)

    out_path = Path(output_dir) / "p2.2_results.json"
    with open(out_path, "w") as f:
        json.dump({"experiment": "p2.2", "alpha": alpha, "results": save_results}, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    return results


def run_p2_3(model, tokenizer, jepa_model, dataset, layer, alpha, K, batch_size,
             max_new_tokens, output_dir, device, mean_delta):
    """P2.3: Full evaluation with all 4 modes + McNemar's test."""
    modes = ["none", "jepa", "random", "mean_delta"]
    results = {}

    for mode in modes:
        a = alpha if mode != "none" else 0.0
        k = K if mode != "none" else 0
        res = evaluate_condition(
            model, tokenizer, jepa_model, dataset, layer,
            alpha=a, K=k, mode=mode, mean_delta=mean_delta,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            output_dir=output_dir, device=device,
        )
        results[mode] = res

    # McNemar's tests vs baseline
    baseline_correct = results["none"]["per_example_correct"]
    mcnemar_results = {}
    for mode in ["jepa", "random", "mean_delta"]:
        mc = mcnemar_test(baseline_correct, results[mode]["per_example_correct"])
        mcnemar_results[mode] = mc
        print(f"  McNemar {mode} vs none: b={mc['b_wrong_a_right_b']}, "
              f"c={mc['c_right_a_wrong_b']}, p={mc['p_value']:.4f}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"P2.3 Results — Full Evaluation (alpha={alpha}, K={K})")
    print(f"{'='*70}")
    print(f"  {'Mode':<12} {'Solve Rate':>12} {'95% CI':>20} {'Avg Tok':>8} {'McNemar p':>10}")
    print(f"  {'─'*66}")
    for mode in modes:
        r = results[mode]
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        p_str = f"{mcnemar_results[mode]['p_value']:.4f}" if mode in mcnemar_results else "—"
        print(f"  {mode:<12} {r['solve_rate']:>12.4f} {ci_str:>20} {r['avg_tokens']:>8.1f} {p_str:>10}")

    # Go/no-go assessment
    print(f"\n{'='*70}")
    print("GO/NO-GO CHECK")
    print(f"{'='*70}")
    jepa_better = results["jepa"]["solve_rate"] > results["none"]["solve_rate"]
    jepa_sig = mcnemar_results["jepa"]["p_value"] < 0.05
    random_not_sig = mcnemar_results["random"]["p_value"] >= 0.05
    random_not_better = results["random"]["solve_rate"] <= results["jepa"]["solve_rate"]

    print(f"  JEPA > baseline: {jepa_better} "
          f"({results['jepa']['solve_rate']:.4f} vs {results['none']['solve_rate']:.4f})")
    print(f"  JEPA significant (p<0.05): {jepa_sig} (p={mcnemar_results['jepa']['p_value']:.4f})")
    print(f"  Random not significant: {random_not_sig} (p={mcnemar_results['random']['p_value']:.4f})")
    print(f"  Random not better than JEPA: {random_not_better} "
          f"({results['random']['solve_rate']:.4f} vs {results['jepa']['solve_rate']:.4f})")

    if jepa_better and jepa_sig and random_not_better:
        print("  → PASS: Proceed to Phase 3")
    elif jepa_better:
        print("  → PARTIAL: Improvement observed but not statistically significant or not differentiated from random")
    else:
        print("  → FAIL: No improvement from JEPA injection")

    # Save results
    save_results = {}
    for mode in modes:
        r = results[mode]
        save_results[mode] = {k: v for k, v in r.items() if k != "per_example_correct"}
        save_results[mode]["n_correct_examples"] = sum(r["per_example_correct"])

    out = {
        "experiment": "p2.3",
        "alpha": alpha,
        "K": K,
        "layer": layer,
        "results": save_results,
        "mcnemar": {k: v for k, v in mcnemar_results.items()},
        "go_no_go": {
            "jepa_better_than_baseline": jepa_better,
            "jepa_significant": jepa_sig,
            "random_not_significant": random_not_sig,
            "random_not_better_than_jepa": random_not_better,
            "pass": jepa_better and jepa_sig and random_not_better,
        },
    }

    out_path = Path(output_dir) / "p2.3_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: bool(o) if isinstance(o, (bool, np.bool_)) else float(o))
    print(f"\n  Results saved: {out_path}")

    return results, mcnemar_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: JEPA injection during decoding")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--jepa_checkpoint", type=str,
                        default="experiments/jepa_phase1/best_model_mse.pt")
    parser.add_argument("--data_dir", type=str,
                        default="data/hidden_states_v2/Qwen_Qwen3-4B_gsm8k_train")
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["p2.1", "p2.2", "p2.3"])
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.001, 0.005, 0.01, 0.05, 0.1])
    parser.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="experiments/jepa_phase2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load everything ---
    t0 = time.time()

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    jepa_model, jepa_config = load_jepa(args.jepa_checkpoint, device)
    mean_delta = compute_mean_delta(args.data_dir, args.layer, device)

    # Load GSM8K test set
    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  {len(dataset)} test examples")

    load_time = time.time() - t0
    print(f"  Setup complete in {load_time:.1f}s\n")

    # --- Run experiment ---
    t1 = time.time()

    if args.experiment == "p2.1":
        run_p2_1(model, tokenizer, jepa_model, dataset, args.layer, args.alphas,
                 args.batch_size, args.max_new_tokens, str(output_dir), device,
                 mean_delta)

    elif args.experiment == "p2.2":
        alpha = args.alphas[0]
        run_p2_2(model, tokenizer, jepa_model, dataset, args.layer, alpha, args.Ks,
                 args.batch_size, args.max_new_tokens, str(output_dir), device,
                 mean_delta)

    elif args.experiment == "p2.3":
        alpha = args.alphas[0]
        K = args.Ks[0]
        run_p2_3(model, tokenizer, jepa_model, dataset, args.layer, alpha, K,
                 args.batch_size, args.max_new_tokens, str(output_dir), device,
                 mean_delta)

    elapsed = time.time() - t1
    print(f"\nExperiment {args.experiment} completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
