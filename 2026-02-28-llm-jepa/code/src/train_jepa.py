#!/usr/bin/env python3
"""
Phase 1: Train JEPA predictor on (problem → first_step) hidden states.

Trains a 2-layer MLP with residual connection and compares against
baselines (mean, identity, ridge, k-NN).

Usage:
    # Cosine loss (default)
    python src/train_jepa.py --data_dir data/hidden_states_v2/Qwen_Qwen3-4B_gsm8k_train --layer 21 --loss cosine

    # MSE loss
    python src/train_jepa.py --data_dir data/hidden_states_v2/Qwen_Qwen3-4B_gsm8k_train --layer 21 --loss mse
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine as cosine_dist


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class JEPAPredictor(nn.Module):
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(Y_true, Y_pred, X_input):
    """Compute all evaluation metrics."""
    # Cosine similarity
    cos_sims = []
    for i in range(len(Y_true)):
        cs = 1 - cosine_dist(Y_true[i], Y_pred[i])
        cos_sims.append(cs)
    mean_cos = float(np.mean(cos_sims))
    std_cos = float(np.std(cos_sims))

    # MSE
    mse = float(((Y_true - Y_pred) ** 2).mean())

    # R²
    r2 = float(r2_score(Y_true, Y_pred, multioutput="uniform_average"))

    # Delta direction cosine similarity
    true_delta = Y_true - X_input
    pred_delta = Y_pred - X_input
    delta_cos = []
    for i in range(len(Y_true)):
        tn = np.linalg.norm(true_delta[i])
        pn = np.linalg.norm(pred_delta[i])
        if tn > 1e-8 and pn > 1e-8:
            delta_cos.append(1 - cosine_dist(true_delta[i], pred_delta[i]))
    mean_delta_cos = float(np.mean(delta_cos)) if delta_cos else 0.0

    return {
        "cos_sim_mean": mean_cos,
        "cos_sim_std": std_cos,
        "mse": mse,
        "r2": r2,
        "delta_direction_cos": mean_delta_cos,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_jepa(X_train, Y_train, X_val, Y_val, dim, loss_fn="cosine",
               hidden_mult=2, lr=1e-4, epochs=300, batch_size=128,
               patience=30, device="cuda"):
    """Train JEPA predictor with early stopping on validation loss.

    Returns: (best model state dict, final epoch count, training history)
    """
    model = JEPAPredictor(dim, hidden_mult=hidden_mult).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_tr = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    Y_v = torch.tensor(Y_val, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    n = X_tr.shape[0]
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_tr[idx])
            target = Y_tr[idx]

            if loss_fn == "cosine":
                loss = (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
            else:
                loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            if loss_fn == "cosine":
                val_loss = (1 - F.cosine_similarity(val_pred, Y_v, dim=-1)).mean().item()
            else:
                val_loss = F.mse_loss(val_pred, Y_v).item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 50 == 0 or no_improve == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={avg_train_loss:.6f}  "
                  f"val_loss={val_loss:.6f}  {'*' if no_improve == 0 else ''}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    return best_state, epoch + 1, history


def predict_with_model(model_state, X, dim, hidden_mult=2, device="cuda"):
    """Load model from state dict and predict."""
    model = JEPAPredictor(dim, hidden_mult=hidden_mult).to(device)
    model.load_state_dict({k: v.to(device) for k, v in model_state.items()})
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    return pred


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def mean_baseline(Y_train, n_test):
    """Predict mean of training targets for every test example."""
    mean = Y_train.mean(axis=0, keepdims=True)
    return np.repeat(mean, n_test, axis=0)


def identity_baseline(X_test):
    """Predict solution = problem (identity mapping)."""
    return X_test.copy()


def ridge_baseline(X_train, Y_train, X_test, alpha=10.0):
    """Ridge regression baseline."""
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, Y_train)
    return ridge.predict(X_test)


def knn_baseline(X_train, Y_train, X_test, k=5):
    """k-NN baseline: average targets of k nearest training problems."""
    nn_model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn_model.fit(X_train)
    distances, indices = nn_model.kneighbors(X_test)
    # Average the targets of the k nearest neighbors
    predictions = np.array([Y_train[idx].mean(axis=0) for idx in indices])
    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Train JEPA predictor")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--loss", type=str, default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--hidden_mult", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="experiments/jepa_phase1")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading data...")
    problem = torch.load(data_dir / "problem_states.pt", weights_only=True).float()
    first_step = torch.load(data_dir / "first_step_states.pt", weights_only=True).float()
    N, L, D = problem.shape

    print(f"  {N} pairs, {L} layers, dim={D}")
    print(f"  Layer: {args.layer}, Loss: {args.loss}")
    print(f"  Device: {device}")

    # Extract selected layer
    X = problem[:, args.layer].numpy()  # (N, D)
    Y = first_step[:, args.layer].numpy()  # (N, D)

    # --- Split: 80/10/10 ---
    train_idx, temp_idx = train_test_split(
        np.arange(N), test_size=0.2, random_state=args.seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=args.seed
    )

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # --- Train JEPA ---
    print(f"\n{'='*60}")
    print(f"Training JEPA (loss={args.loss}, layer={args.layer})")
    print(f"{'='*60}")
    print(f"  Architecture: 2-layer MLP, hidden={D * args.hidden_mult}, "
          f"residual, GELU, dropout=0.1")

    t0 = time.time()
    best_state, final_epoch, history = train_jepa(
        X_train, Y_train, X_val, Y_val, D,
        loss_fn=args.loss, hidden_mult=args.hidden_mult,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, device=device,
    )
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s ({final_epoch} epochs)")

    # Predict on test set
    jepa_pred = predict_with_model(best_state, X_test, D,
                                   hidden_mult=args.hidden_mult, device=device)

    # --- Baselines ---
    print(f"\n{'='*60}")
    print("Computing baselines...")
    print(f"{'='*60}")

    mean_pred = mean_baseline(Y_train, len(test_idx))
    identity_pred = identity_baseline(X_test)
    ridge_pred = ridge_baseline(X_train, Y_train, X_test)
    knn_pred = knn_baseline(X_train, Y_train, X_test, k=5)

    # --- Evaluate all ---
    print(f"\n{'='*60}")
    print("Evaluation (test set)")
    print(f"{'='*60}")

    results = {}
    methods = {
        "Mean": mean_pred,
        "Identity": identity_pred,
        "Ridge": ridge_pred,
        "k-NN (k=5)": knn_pred,
        "JEPA": jepa_pred,
    }

    for name, pred in methods.items():
        metrics = compute_metrics(Y_test, pred, X_test)
        results[name] = metrics

    # Print comparison table
    print(f"\n  {'Method':<12} {'Cos Sim':>10} {'±Std':>8} "
          f"{'MSE':>10} {'R²':>8} {'ΔDir Cos':>10}")
    print(f"  {'─'*60}")
    for name in methods:
        m = results[name]
        print(f"  {name:<12} {m['cos_sim_mean']:>10.4f} {m['cos_sim_std']:>8.4f} "
              f"{m['mse']:>10.4f} {m['r2']:>8.4f} {m['delta_direction_cos']:>10.4f}")

    # Improvement over mean baseline
    mean_cos = results["Mean"]["cos_sim_mean"]
    jepa_cos = results["JEPA"]["cos_sim_mean"]
    ridge_cos = results["Ridge"]["cos_sim_mean"]
    print(f"\n  JEPA improvement over mean: {jepa_cos - mean_cos:+.4f} cos sim")
    print(f"  JEPA improvement over ridge: {jepa_cos - ridge_cos:+.4f} cos sim")
    print(f"  JEPA R² improvement over ridge: "
          f"{results['JEPA']['r2'] - results['Ridge']['r2']:+.4f}")

    # --- Go/No-Go ---
    print(f"\n{'='*60}")
    print("GO/NO-GO CHECK")
    print(f"{'='*60}")

    check_cos = jepa_cos >= ridge_cos
    check_delta = results["JEPA"]["delta_direction_cos"] > 0.2

    print(f"  JEPA cos sim ({jepa_cos:.4f}) >= Ridge cos sim ({ridge_cos:.4f}): "
          f"{'PASS' if check_cos else 'FAIL'}")
    print(f"  Delta direction cos ({results['JEPA']['delta_direction_cos']:.4f}) > 0.2: "
          f"{'PASS' if check_delta else 'FAIL'}")

    if check_cos and check_delta:
        print("  → PASS: Proceed to Phase 2")
    elif check_delta:
        print("  → PARTIAL: Direction is good but cosine sim below Ridge")
    else:
        print("  → FAIL: Revise architecture or approach")

    # --- Save checkpoint ---
    checkpoint = {
        "model_state_dict": best_state,
        "config": {
            "dim": D,
            "hidden_mult": args.hidden_mult,
            "dropout": 0.1,
            "layer": args.layer,
            "loss": args.loss,
        },
        "training": {
            "epochs": final_epoch,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "train_time": train_time,
        },
    }
    ckpt_path = output_dir / f"best_model_{args.loss}.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    # --- Save results JSON ---
    results_out = {
        "config": {
            "layer": args.layer,
            "loss": args.loss,
            "hidden_mult": args.hidden_mult,
            "lr": args.lr,
            "epochs_trained": final_epoch,
            "batch_size": args.batch_size,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "dim": D,
            "train_time_s": train_time,
        },
        "metrics": results,
        "go_no_go": {
            "cos_sim_vs_ridge": check_cos,
            "delta_direction_gt_0.2": check_delta,
            "pass": check_cos and check_delta,
        },
    }
    results_path = output_dir / f"results_{args.loss}.json"
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"  Results saved: {results_path}")


if __name__ == "__main__":
    main()
