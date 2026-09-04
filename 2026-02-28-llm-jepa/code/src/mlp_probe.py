#!/usr/bin/env python3
"""
Phase 0 diagnostic: MLP probe to test if nonlinear structure exists
in the problem→first_step mapping beyond what a linear probe captures.

Trains a small MLP with residual connection and compares R² against
the linear probe baseline (~0.4).

Usage:
    python src/mlp_probe.py --data_dir data/hidden_states_v2/Qwen_Qwen3-4B_gsm8k_train
    python src/mlp_probe.py --data_dir ... --layers 12 17 19
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine as cosine_dist


class MLPProbe(nn.Module):
    """2-layer MLP with residual connection."""
    def __init__(self, dim, hidden_mult=2, dropout=0.1):
        super().__init__()
        hidden = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return x + self.net(x)  # residual


def train_mlp_probe(X_train, Y_train, X_val, Y_val, dim,
                    hidden_mult=2, lr=1e-4, epochs=200, batch_size=64,
                    patience=20, device="cuda"):
    """Train an MLP probe and return best validation metrics."""
    model = MLPProbe(dim, hidden_mult=hidden_mult).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    n_train = X_train_t.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            x_batch = X_train_t[idx]
            y_batch = Y_train_t[idx]

            pred = model(x_batch)
            loss = nn.functional.mse_loss(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = nn.functional.mse_loss(val_pred, Y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Load best model and compute final metrics
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()

    return val_pred, epoch + 1


def evaluate(Y_true, Y_pred, X_input, label=""):
    """Compute R², cosine similarity, and delta direction accuracy."""
    r2 = r2_score(Y_true, Y_pred, multioutput="uniform_average")

    cos_sims = []
    for i in range(len(Y_true)):
        cs = 1 - cosine_dist(Y_true[i], Y_pred[i])
        cos_sims.append(cs)
    mean_cos = np.mean(cos_sims)

    # Delta direction accuracy
    true_delta = Y_true - X_input
    pred_delta = Y_pred - X_input
    delta_cos = []
    for i in range(len(Y_true)):
        tn = np.linalg.norm(true_delta[i])
        pn = np.linalg.norm(pred_delta[i])
        if tn > 1e-8 and pn > 1e-8:
            delta_cos.append(1 - cosine_dist(true_delta[i], pred_delta[i]))
    mean_delta_cos = np.mean(delta_cos) if delta_cos else 0.0

    return {
        "r2": float(r2),
        "cos_sim": float(mean_cos),
        "delta_direction_cos": float(mean_delta_cos),
    }


def main():
    parser = argparse.ArgumentParser(description="MLP probe diagnostic")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--target", type=str, default="first_step",
                        choices=["first_step", "solution", "answer", "mid"])
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to test (default: best from linear probe + neighbors)")
    parser.add_argument("--hidden_mult", type=int, default=2,
                        help="Hidden dim multiplier for MLP")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    data_dir = Path(args.data_dir)

    # Load states
    problem = torch.load(data_dir / "problem_states.pt", weights_only=True).float()
    target_map = {
        "first_step": "first_step_states.pt",
        "solution": "solution_states.pt",
        "answer": "answer_states.pt",
        "mid": "mid_states.pt",
    }
    target = torch.load(data_dir / target_map[args.target], weights_only=True).float()
    N, L, D = problem.shape

    print(f"Loaded {N} pairs, {L} layers, dim={D}")
    print(f"Target: {args.target}")
    print(f"MLP: 2-layer, hidden={D * args.hidden_mult}, residual, GELU")
    print(f"Device: {device}")

    # Select layers
    if args.layers:
        layers_to_test = args.layers
    else:
        # Default: test a spread of mid-layers where linear probe worked
        layers_to_test = [9, 12, 15, 17, 19, 21, 24]

    # Train/val/test split
    train_idx, temp_idx = train_test_split(np.arange(N), test_size=0.3, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    results = []

    for li in layers_to_test:
        print(f"\n{'='*60}")
        print(f"Layer {li}")
        print(f"{'='*60}")

        X_train = problem[train_idx, li].numpy()
        Y_train = target[train_idx, li].numpy()
        X_val = problem[val_idx, li].numpy()
        Y_val = target[val_idx, li].numpy()
        X_test = problem[test_idx, li].numpy()
        Y_test = target[test_idx, li].numpy()

        # --- Linear baseline (ridge) ---
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_train, Y_train)
        ridge_pred = ridge.predict(X_test)
        ridge_metrics = evaluate(Y_test, ridge_pred, X_test, "Ridge")

        # --- Mean baseline ---
        mean_pred = Y_train.mean(axis=0, keepdims=True).repeat(len(test_idx), axis=0)
        mean_metrics = evaluate(Y_test, mean_pred, X_test, "Mean")

        # --- MLP probe ---
        t0 = time.time()
        mlp_pred, final_epoch = train_mlp_probe(
            X_train, Y_train, X_val, Y_val, D,
            hidden_mult=args.hidden_mult, lr=args.lr,
            epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, device=device,
        )
        # Evaluate on TEST set (not val)
        # Re-run through model on test set
        model = MLPProbe(D, hidden_mult=args.hidden_mult).to(device)
        # Load from the training function... actually we need to return the model
        # Let me just re-predict on test
        # Actually the function returns val_pred. Let me fix this.
        elapsed = time.time() - t0

        # Retrain and get test predictions properly
        # The clean way: retrain with train+val, evaluate on test
        # But for a diagnostic, training on train, selecting on val, evaluating on test is fine
        # We need to get the model to predict on test set though
        # Let me just use the combined train approach instead

        # Simpler: train on train, use val for early stopping, predict on test
        mlp_pred_test, final_epoch = _train_and_predict(
            X_train, Y_train, X_val, Y_val, X_test, D,
            hidden_mult=args.hidden_mult, lr=args.lr,
            epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, device=device,
        )
        mlp_metrics = evaluate(Y_test, mlp_pred_test, X_test, "MLP")

        layer_result = {
            "layer": li,
            "mlp_r2": mlp_metrics["r2"],
            "mlp_cos": mlp_metrics["cos_sim"],
            "mlp_delta_cos": mlp_metrics["delta_direction_cos"],
            "ridge_r2": ridge_metrics["r2"],
            "ridge_cos": ridge_metrics["cos_sim"],
            "ridge_delta_cos": ridge_metrics["delta_direction_cos"],
            "mean_r2": float(r2_score(Y_test, mean_pred, multioutput="uniform_average")),
            "mean_cos": mean_metrics["cos_sim"],
            "r2_improvement": mlp_metrics["r2"] - ridge_metrics["r2"],
            "epochs": final_epoch,
            "train_time": elapsed,
        }
        results.append(layer_result)

        print(f"\n  {'Method':<10} {'R²':>8} {'Cos Sim':>8} {'Delta Cos':>10}")
        print(f"  {'─'*40}")
        print(f"  {'Mean':<10} {layer_result['mean_r2']:>8.4f} {mean_metrics['cos_sim']:>8.4f} {'—':>10}")
        print(f"  {'Ridge':<10} {ridge_metrics['r2']:>8.4f} {ridge_metrics['cos_sim']:>8.4f} {ridge_metrics['delta_direction_cos']:>10.4f}")
        print(f"  {'MLP':<10} {mlp_metrics['r2']:>8.4f} {mlp_metrics['cos_sim']:>8.4f} {mlp_metrics['delta_direction_cos']:>10.4f}")
        print(f"  MLP improvement over Ridge: {layer_result['r2_improvement']:+.4f} R²")
        print(f"  Trained for {final_epoch} epochs in {elapsed:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    best = max(results, key=lambda x: x["mlp_r2"])
    print(f"  Best MLP layer: {best['layer']}, R²={best['mlp_r2']:.4f}")
    best_ridge = max(results, key=lambda x: x["ridge_r2"])
    print(f"  Best Ridge layer: {best_ridge['layer']}, R²={best_ridge['ridge_r2']:.4f}")
    print(f"  MLP improvement: {best['mlp_r2'] - best_ridge['ridge_r2']:+.4f} R²")
    print()
    if best["mlp_r2"] > 0.6:
        print("  → STRONG SIGNAL: MLP captures substantial nonlinear structure.")
        print("    Proceed to Phase 1 with confidence.")
    elif best["mlp_r2"] > 0.45:
        print("  → MODERATE SIGNAL: MLP improves over linear, some nonlinear structure exists.")
        print("    Phase 1 is worth attempting but temper expectations.")
    else:
        print("  → WEAK SIGNAL: MLP doesn't improve much over linear probe.")
        print("    The predictable structure is mostly linear; JEPA may not add much.")

    # Save results
    output_dir = data_dir / "mlp_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_dir / 'results.json'}")


def _train_and_predict(X_train, Y_train, X_val, Y_val, X_test, dim,
                       hidden_mult=2, lr=1e-4, epochs=300, batch_size=64,
                       patience=30, device="cuda"):
    """Train MLP, select best on val, predict on test."""
    model = MLPProbe(dim, hidden_mult=hidden_mult).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_tr = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    Y_v = torch.tensor(Y_val, dtype=torch.float32, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    n = X_tr.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_tr[idx])
            loss = nn.functional.mse_loss(pred, Y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(X_v), Y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        test_pred = model(X_te).cpu().numpy()

    return test_pred, epoch + 1


if __name__ == "__main__":
    main()
