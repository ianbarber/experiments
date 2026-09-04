#!/usr/bin/env python3
"""
Phase 0.2-0.5: Analyze extracted hidden states.

Runs the following analyses:
  P0.2 — Cosine similarity distributions per layer
  P0.3 — PCA visualization of problem/solution states
  P0.4 — Linear probe (ridge regression) per layer
  P0.5 — Cross-domain evaluation (if two datasets provided)

Usage:
    # Full analysis on one dataset
    python src/analysis.py --data_dir data/hidden_states/Qwen_Qwen3-4B_gsm8k_train

    # Cross-domain evaluation
    python src/analysis.py \
        --data_dir data/hidden_states/Qwen_Qwen3-4B_gsm8k_train \
        --cross_domain_dir data/hidden_states/Qwen_Qwen3-4B_math_train

    # Run only specific analyses
    python src/analysis.py --data_dir ... --only cosine probe
"""

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine as cosine_dist


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_states(data_dir: Path, target: str = "solution") -> dict:
    """Load extracted hidden states and metadata.

    Args:
        data_dir: directory containing .pt files
        target: which states to use as "solution" target.
            "solution" = last generated token (default)
            "first_step" = first generated token
            "answer" = answer-adjacent token (if extracted)
            "mid" = mid-solution token (if extracted)
    """
    problem = torch.load(data_dir / "problem_states.pt", weights_only=True)
    solution = torch.load(data_dir / "solution_states.pt", weights_only=True)
    first_step = torch.load(data_dir / "first_step_states.pt", weights_only=True)

    # Load additional extraction points if available
    extra = {}
    for name in ["answer_states", "mid_states"]:
        path = data_dir / f"{name}.pt"
        if path.exists():
            extra[name.replace("_states", "")] = torch.load(path, weights_only=True)

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    # Select target
    target_map = {
        "solution": solution,
        "first_step": first_step,
        "answer": extra.get("answer", solution),
        "mid": extra.get("mid", solution),
    }
    if target not in target_map:
        raise ValueError(f"Unknown target '{target}'. Options: {list(target_map.keys())}")

    selected = target_map[target]
    print(f"Loaded {problem.shape[0]} pairs from {data_dir}")
    print(f"  Shape: {problem.shape} (N, layers, dim)")
    print(f"  Model: {meta['model']}, Dataset: {meta['dataset']}")
    print(f"  Target: {target}")

    return {
        "problem": problem.float(),  # (N, L, D)
        "solution": selected.float(),  # whatever we're targeting
        "first_step": first_step.float(),
        "meta": meta,
        "target_name": target,
    }


# ---------------------------------------------------------------------------
# P0.2: Cosine similarity analysis
# ---------------------------------------------------------------------------

def cosine_similarity_analysis(data: dict, output_dir: Path):
    """Compute and plot cosine similarity between problem and solution states per layer."""
    print("\n=== P0.2: Cosine Similarity Analysis ===")

    problem = data["problem"]   # (N, L, D)
    solution = data["solution"]
    N, L, D = problem.shape

    # Cosine similarity per layer
    # cos_sim(a, b) = (a · b) / (||a|| ||b||)
    dot = (problem * solution).sum(dim=-1)  # (N, L)
    norm_p = problem.norm(dim=-1)           # (N, L)
    norm_s = solution.norm(dim=-1)          # (N, L)
    cos_sim = dot / (norm_p * norm_s + 1e-8)  # (N, L)

    cos_sim_np = cos_sim.numpy()

    # Stats per layer
    means = cos_sim_np.mean(axis=0)
    stds = cos_sim_np.std(axis=0)
    medians = np.median(cos_sim_np, axis=0)

    print(f"\n  Layer | Mean cos_sim | Std    | Median")
    print(f"  ------|-------------|--------|-------")
    for li in range(L):
        print(f"  {li:5d} | {means[li]:11.4f} | {stds[li]:.4f} | {medians[li]:.4f}")

    # Plot: mean cosine similarity across layers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(range(L), means, "b-", linewidth=2)
    ax.fill_between(range(L), means - stds, means + stds, alpha=0.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Problem-Solution Cosine Similarity by Layer")
    ax.grid(True, alpha=0.3)

    # Plot: distributions for selected layers
    ax = axes[1]
    layers_to_show = [0, L // 4, L // 2, 3 * L // 4, L - 1]
    for li in layers_to_show:
        ax.hist(cos_sim_np[:, li], bins=50, alpha=0.5, label=f"Layer {li}", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Cosine Similarity Distributions")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "cosine_similarity.png", dpi=150)
    plt.close()
    print(f"  Saved plot: {output_dir / 'cosine_similarity.png'}")

    # Also compute delta direction consistency
    # Are the deltas (solution - problem) pointing in consistent directions?
    deltas = solution - problem  # (N, L, D)
    delta_norms = deltas.norm(dim=-1)  # (N, L)
    mean_delta = deltas.mean(dim=0)    # (L, D)
    mean_delta_norm = mean_delta.norm(dim=-1)  # (L,)

    # Cosine sim of each delta to the mean delta (consistency measure)
    delta_dot_mean = (deltas * mean_delta.unsqueeze(0)).sum(dim=-1)  # (N, L)
    delta_cos_to_mean = delta_dot_mean / (delta_norms * mean_delta_norm.unsqueeze(0) + 1e-8)
    delta_consistency = delta_cos_to_mean.numpy().mean(axis=0)

    print(f"\n  Delta direction consistency (cos sim to mean delta):")
    for li in range(L):
        print(f"    Layer {li:3d}: {delta_consistency[li]:.4f}  "
              f"(delta norm: {delta_norms[:, li].mean():.2f})")

    results = {
        "cosine_sim_mean": means.tolist(),
        "cosine_sim_std": stds.tolist(),
        "cosine_sim_median": medians.tolist(),
        "delta_consistency": delta_consistency.tolist(),
        "delta_norm_mean": delta_norms.mean(dim=0).numpy().tolist(),
    }
    return results


# ---------------------------------------------------------------------------
# P0.3: PCA / t-SNE visualization
# ---------------------------------------------------------------------------

def pca_analysis(data: dict, output_dir: Path, max_points: int = 500):
    """PCA and t-SNE visualization of problem/solution states."""
    print("\n=== P0.3: PCA / t-SNE Visualization ===")

    problem = data["problem"]   # (N, L, D)
    solution = data["solution"]
    N, L, D = problem.shape

    # Select layers to visualize
    layers_to_viz = [0, L // 4, L // 2, 3 * L // 4, L - 1]

    # Subsample if needed
    if N > max_points:
        indices = np.random.RandomState(42).choice(N, max_points, replace=False)
    else:
        indices = np.arange(N)

    fig, axes = plt.subplots(2, len(layers_to_viz), figsize=(5 * len(layers_to_viz), 10))

    for col, li in enumerate(layers_to_viz):
        p = problem[indices, li].numpy()  # (n, D)
        s = solution[indices, li].numpy()

        combined = np.concatenate([p, s], axis=0)  # (2n, D)
        n = len(indices)

        # PCA
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(combined)

        ax = axes[0, col]
        ax.scatter(proj[:n, 0], proj[:n, 1], c="blue", alpha=0.3, s=10, label="Problem")
        ax.scatter(proj[n:, 0], proj[n:, 1], c="red", alpha=0.3, s=10, label="Solution")
        # Draw arrows from problem to solution for a few pairs
        for i in range(min(30, n)):
            ax.annotate("", xy=proj[n + i], xytext=proj[i],
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.2, lw=0.5))
        ax.set_title(f"Layer {li} — PCA\n(var: {pca.explained_variance_ratio_.sum():.2%})")
        if col == 0:
            ax.legend(fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1))
        proj_tsne = tsne.fit_transform(combined)

        ax = axes[1, col]
        ax.scatter(proj_tsne[:n, 0], proj_tsne[:n, 1], c="blue", alpha=0.3, s=10, label="Problem")
        ax.scatter(proj_tsne[n:, 0], proj_tsne[n:, 1], c="red", alpha=0.3, s=10, label="Solution")
        for i in range(min(30, n)):
            ax.annotate("", xy=proj_tsne[n + i], xytext=proj_tsne[i],
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.2, lw=0.5))
        ax.set_title(f"Layer {li} — t-SNE")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Problem (blue) → Solution (red) in Hidden State Space", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_tsne.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_dir / 'pca_tsne.png'}")

    # PCA variance analysis: how many components to capture 90% variance?
    print("\n  PCA variance analysis (components for 90% variance):")
    for li in layers_to_viz:
        p = problem[:, li].numpy()
        pca_full = PCA(n_components=min(50, D, N), random_state=42)
        pca_full.fit(p)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_90 = np.searchsorted(cumvar, 0.9) + 1
        print(f"    Layer {li}: {n_90} components for 90% variance")


# ---------------------------------------------------------------------------
# P0.4: Linear probe (ridge regression) per layer
# ---------------------------------------------------------------------------

def linear_probe(data: dict, output_dir: Path, seed: int = 42):
    """Train ridge regression probes: problem state → solution state, per layer."""
    print("\n=== P0.4: Linear Probe (Ridge Regression) ===")

    problem = data["problem"]   # (N, L, D)
    solution = data["solution"]
    N, L, D = problem.shape

    # Train/test split
    train_idx, test_idx = train_test_split(
        np.arange(N), test_size=0.2, random_state=seed
    )
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    results_per_layer = []
    alphas = [1.0, 10.0, 100.0]  # Ridge regularization strengths

    for li in range(L):
        p_train = problem[train_idx, li].numpy()
        s_train = solution[train_idx, li].numpy()
        p_test = problem[test_idx, li].numpy()
        s_test = solution[test_idx, li].numpy()

        # Mean baseline
        mean_pred = s_train.mean(axis=0, keepdims=True).repeat(len(test_idx), axis=0)
        mean_cos = np.array([
            1 - cosine_dist(s_test[i], mean_pred[i])
            for i in range(len(test_idx))
        ]).mean()
        mean_mse = ((s_test - mean_pred) ** 2).mean()

        # Identity baseline (predict problem = solution)
        identity_cos = np.array([
            1 - cosine_dist(s_test[i], p_test[i])
            for i in range(len(test_idx))
        ]).mean()

        # Ridge regression — try multiple alphas, pick best on train
        best_r2 = -np.inf
        best_alpha = alphas[0]
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(p_train, s_train)
            train_r2 = ridge.score(p_train, s_train)
            if train_r2 > best_r2:
                best_r2 = train_r2
                best_alpha = alpha

        ridge = Ridge(alpha=best_alpha)
        ridge.fit(p_train, s_train)
        pred_test = ridge.predict(p_test)

        # Metrics
        test_r2 = r2_score(s_test, pred_test, multioutput="uniform_average")
        probe_cos = np.array([
            1 - cosine_dist(s_test[i], pred_test[i])
            for i in range(len(test_idx))
        ]).mean()
        probe_mse = ((s_test - pred_test) ** 2).mean()

        # Direction accuracy: does predicted delta point same way as true delta?
        true_delta = s_test - p_test
        pred_delta = pred_test - p_test
        delta_cos = np.array([
            1 - cosine_dist(true_delta[i], pred_delta[i])
            if np.linalg.norm(true_delta[i]) > 1e-8 and np.linalg.norm(pred_delta[i]) > 1e-8
            else 0.0
            for i in range(len(test_idx))
        ]).mean()

        layer_result = {
            "layer": li,
            "ridge_alpha": best_alpha,
            "test_r2": float(test_r2),
            "probe_cos_sim": float(probe_cos),
            "probe_mse": float(probe_mse),
            "mean_baseline_cos": float(mean_cos),
            "mean_baseline_mse": float(mean_mse),
            "identity_cos": float(identity_cos),
            "delta_direction_cos": float(delta_cos),
            "improvement_over_mean": float(probe_cos - mean_cos),
        }
        results_per_layer.append(layer_result)

        if li % max(1, L // 10) == 0 or li == L - 1:
            print(f"  Layer {li:3d}: R²={test_r2:.4f}  "
                  f"cos_sim={probe_cos:.4f} (mean={mean_cos:.4f}, "
                  f"identity={identity_cos:.4f})  "
                  f"delta_dir={delta_cos:.4f}")

    # Find best layers
    by_r2 = sorted(results_per_layer, key=lambda x: x["test_r2"], reverse=True)
    print(f"\n  Top 5 layers by R²:")
    for r in by_r2[:5]:
        print(f"    Layer {r['layer']}: R²={r['test_r2']:.4f}, "
              f"cos={r['probe_cos_sim']:.4f}, delta_dir={r['delta_direction_cos']:.4f}")

    by_delta = sorted(results_per_layer, key=lambda x: x["delta_direction_cos"], reverse=True)
    print(f"\n  Top 5 layers by delta direction accuracy:")
    for r in by_delta[:5]:
        print(f"    Layer {r['layer']}: delta_dir={r['delta_direction_cos']:.4f}, "
              f"R²={r['test_r2']:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    layers = [r["layer"] for r in results_per_layer]

    ax = axes[0]
    ax.plot(layers, [r["test_r2"] for r in results_per_layer], "b-", linewidth=2, label="Ridge R²")
    ax.axhline(y=0.3, color="green", linestyle="--", alpha=0.5, label="Go/No-Go threshold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("Linear Probe R² by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(layers, [r["probe_cos_sim"] for r in results_per_layer],
            "b-", linewidth=2, label="Ridge probe")
    ax.plot(layers, [r["mean_baseline_cos"] for r in results_per_layer],
            "r--", linewidth=1.5, label="Mean baseline")
    ax.plot(layers, [r["identity_cos"] for r in results_per_layer],
            "g--", linewidth=1.5, label="Identity (prob=sol)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Prediction Quality by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(layers, [r["delta_direction_cos"] for r in results_per_layer],
            "b-", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Delta Direction Cosine Sim")
    ax.set_title("Delta Direction Accuracy by Layer")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "linear_probe.png", dpi=150)
    plt.close()
    print(f"  Saved plot: {output_dir / 'linear_probe.png'}")

    return results_per_layer


# ---------------------------------------------------------------------------
# P0.5: Cross-domain evaluation
# ---------------------------------------------------------------------------

def cross_domain_probe(data_a: dict, data_b: dict, output_dir: Path,
                       top_k_layers: int = 5):
    """Train probe on dataset A, evaluate on dataset B (and vice versa)."""
    print("\n=== P0.5: Cross-Domain Evaluation ===")

    name_a = data_a["meta"]["dataset"]
    name_b = data_b["meta"]["dataset"]

    problem_a = data_a["problem"]
    solution_a = data_a["solution"]
    problem_b = data_b["problem"]
    solution_b = data_b["solution"]

    L = problem_a.shape[1]

    # We'll evaluate at evenly spaced layers + first/last
    layers_to_test = sorted(set([0, L // 4, L // 2, 3 * L // 4, L - 1]))

    results = []
    for li in layers_to_test:
        # Train on A, test on B
        ridge_ab = Ridge(alpha=10.0)
        ridge_ab.fit(problem_a[:, li].numpy(), solution_a[:, li].numpy())
        pred_b = ridge_ab.predict(problem_b[:, li].numpy())
        s_b = solution_b[:, li].numpy()

        r2_ab = r2_score(s_b, pred_b, multioutput="uniform_average")
        cos_ab = np.array([
            1 - cosine_dist(s_b[i], pred_b[i]) for i in range(len(s_b))
        ]).mean()

        # Train on B, test on A
        ridge_ba = Ridge(alpha=10.0)
        ridge_ba.fit(problem_b[:, li].numpy(), solution_b[:, li].numpy())
        pred_a = ridge_ba.predict(problem_a[:, li].numpy())
        s_a = solution_a[:, li].numpy()

        r2_ba = r2_score(s_a, pred_a, multioutput="uniform_average")
        cos_ba = np.array([
            1 - cosine_dist(s_a[i], pred_a[i]) for i in range(len(s_a))
        ]).mean()

        result = {
            "layer": li,
            f"R2_{name_a}_to_{name_b}": float(r2_ab),
            f"cos_{name_a}_to_{name_b}": float(cos_ab),
            f"R2_{name_b}_to_{name_a}": float(r2_ba),
            f"cos_{name_b}_to_{name_a}": float(cos_ba),
        }
        results.append(result)
        print(f"  Layer {li:3d}: "
              f"{name_a}→{name_b} R²={r2_ab:.4f} cos={cos_ab:.4f}  |  "
              f"{name_b}→{name_a} R²={r2_ba:.4f} cos={cos_ba:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 0 analysis of hidden states")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing extracted states")
    parser.add_argument("--cross_domain_dir", type=str, default=None,
                        help="Second dataset dir for cross-domain evaluation")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for plots/results (default: data_dir/analysis)")
    parser.add_argument("--target", type=str, default="solution",
                        choices=["solution", "first_step", "answer", "mid"],
                        help="Which states to use as prediction target")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=["cosine", "pca", "probe", "cross"],
                        help="Run only specified analyses")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_states(data_dir, target=args.target)
    target_suffix = f"_{args.target}" if args.target != "solution" else ""
    output_dir = Path(str(output_dir) + target_suffix) if target_suffix else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_all = args.only is None

    all_results = {"target": args.target}

    if run_all or "cosine" in args.only:
        all_results["cosine"] = cosine_similarity_analysis(data, output_dir)

    if run_all or "pca" in args.only:
        pca_analysis(data, output_dir)

    if run_all or "probe" in args.only:
        all_results["probe"] = linear_probe(data, output_dir, seed=args.seed)

    if (run_all or "cross" in args.only) and args.cross_domain_dir:
        data_b = load_states(Path(args.cross_domain_dir))
        all_results["cross_domain"] = cross_domain_probe(data, data_b, output_dir)

    # Save results
    with open(output_dir / "analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'analysis_results.json'}")

    # Print go/no-go summary
    if "probe" in all_results:
        probes = all_results["probe"]
        best = max(probes, key=lambda x: x["test_r2"])
        print("\n" + "=" * 60)
        print("GO/NO-GO SUMMARY")
        print("=" * 60)
        print(f"  Best layer: {best['layer']}")
        print(f"  Best R²:    {best['test_r2']:.4f}  (threshold: 0.3)")
        print(f"  Improvement over mean baseline: {best['improvement_over_mean']:.4f}  "
              f"(threshold: 0.1)")
        if best["test_r2"] > 0.3:
            print("  → PASS: Proceed to Phase 1")
        elif best["test_r2"] > 0.1:
            print("  → AMBIGUOUS: Consider proceeding with caution")
        else:
            print("  → FAIL: Reconsider approach")


if __name__ == "__main__":
    main()
