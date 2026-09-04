# LLM-JEPA: Predicting and Injecting Solution-State Hidden Representations

## Summary

This project investigated whether a JEPA (Joint Embedding Predictive Architecture) module can learn to predict "solution-state" hidden representations from "problem-state" representations in an LLM, and whether injecting these predictions during decoding improves math problem-solving.

**Result: The prediction works (R²=0.64), but the injection does not improve performance.** The model's internal representations are already well-optimized for its forward pass, and external interventions at the hidden-state level face a steep uphill battle against the model's own self-correction mechanisms.

## Setup

- **Model:** Qwen3-4B (36 layers, hidden_dim=2560)
- **Dataset:** GSM8K (grade school math, 7473 train / 1319 test)
- **JEPA:** 2-layer MLP with residual connection (~26M params)
- **Injection formula:** `h_modified = h_original + α × (JEPA_predict(h_problem) - h_problem)`

## Phase 0: Finding the Right Extraction Point

We tested which token positions carry problem-specific information that can be predicted from the problem representation:

| Extraction Point | Best R² | Predictable? |
|-----------------|---------|-------------|
| Last generated token | 0.026 | No |
| Answer token (before ####) | 0.021 | No |
| Mid-solution token | -0.015 | No |
| **First generated token** | **0.635** | **Yes** |

The first generated token is the only position where a linear map can meaningfully predict the hidden state from the problem representation. At this point the model has read the full problem and committed to an approach, but hasn't yet gotten into the weeds of arithmetic.

**Best layer:** 21 (mid-to-late network). The mapping is strongly linear — a 26M-parameter MLP adds only +0.7% R² over Ridge regression.

### Why last-token extraction fails

The delta direction from problem→solution (last token) has cosine consistency of 0.97-0.99 across all examples. This means there is essentially a single "I'm done solving a math problem" direction that every example follows, regardless of the specific problem. It encodes completion status, not solution content.

## Phase 1: Training the JEPA Predictor

| Method | Cos Sim | MSE | R² | ΔDir Cos |
|--------|---------|-----|-----|----------|
| Mean baseline | 0.9664 | 0.133 | -0.001 | 0.950 |
| Ridge regression | 0.9875 | 0.049 | 0.639 | 0.982 |
| k-NN (k=5) | 0.9807 | 0.076 | 0.395 | 0.972 |
| JEPA (MSE loss) | 0.9883 | 0.046 | 0.645 | 0.983 |

The JEPA slightly outperforms Ridge on all metrics, but the improvement is marginal. MSE loss is strictly better than cosine loss for this task — cosine loss achieves good direction alignment but poor magnitude, which matters for injection.

## Phase 2: Injection During Decoding — Does It Help?

### P2.1 — Alpha sweep (injection strength)

| Alpha | Solve Rate | vs Baseline |
|-------|-----------|------------|
| 0 (baseline) | **90.83%** | — |
| 0.001 | **90.83%** | +0.00% |
| 0.005 | 90.30% | -0.53% |
| 0.01 | 90.07% | -0.76% |
| 0.05 | 90.60% | -0.23% |
| 0.1 | 90.67% | -0.15% |

No alpha improves over baseline. Small alphas are no-ops; larger alphas slightly hurt.

### P2.2 — K sweep (injection duration, α=0.001)

| K (steps injected) | Solve Rate | vs Baseline |
|--------------------|-----------|------------|
| 0 (baseline) | **90.83%** | — |
| 1 | 90.14% | -0.68% |
| 5 | **90.83%** | +0.00% |
| 10 | 90.67% | -0.15% |
| 20 | 90.37% | -0.46% |

No K value improves over baseline.

### P2.3 — Full evaluation with controls (α=0.001, K=5)

| Mode | Solve Rate | McNemar p-value |
|------|-----------|-----------------|
| None (baseline) | **90.83%** | — |
| JEPA | **90.83%** | 0.855 |
| Random perturbation | 90.52% | 0.556 |
| Mean delta | 90.37% | 0.286 |

McNemar's test confirms no significant difference between any injection mode and baseline. JEPA flips exactly 15 examples in each direction (15 newly correct, 15 newly wrong) — pure noise.

## Why Injection Failed

Three contributing factors:

### 1. The perturbation is too small to matter

At α=0.001, the injected delta has L2 norm ~0.056, while hidden states at layer 21 have norms of ~100-200. The perturbation is <0.05% of the signal. But making it larger hurts rather than helps.

### 2. The prediction doesn't add new information

The JEPA prediction is 98.8% cosine-similar to the target — but the model's own computation already achieves something comparable. The problem representation at layer 21 already contains most of the information about the first solution step. You can't steer a system by feeding it back a noisy copy of its own plan.

The R²=0.64 ceiling means 36% of the variance is unexplained. The JEPA prediction is partly wrong in ways the model can't distinguish from its own (better) internal signal.

### 3. Transformers self-correct against perturbations

LayerNorm, attention re-weighting, and residual connections in the 15 layers after the injection point can absorb small additive perturbations. The model's dynamical stability works against external interventions.

## Broader Takeaways

**Predictability ≠ controllability.** Being able to predict a hidden state from outside the model doesn't mean injecting that prediction changes what the model does. This is a fundamental limitation of additive hidden-state injection approaches.

**The mapping is genuinely linear.** Despite having 26M parameters, the MLP adds essentially nothing over a linear map. The problem→first_step relationship is a linear subspace projection, not a complex nonlinear transformation. This suggests that what the model learns about solution approaches in its first decode step is a relatively simple function of its problem encoding.

**GSM8K may be too easy.** At 90.8% baseline accuracy, most problems are already solved correctly. The ~120 failures are likely capability-limited (multi-step arithmetic errors, misunderstood problems) rather than strategy-limited (wrong approach). A harder benchmark where the model is "on the fence" more often might show more room for intervention.

## If Continuing This Research

Directions that might have more traction:

- **Representation replacement rather than addition** — don't add a delta, replace the hidden state entirely or use a learned gate
- **Earlier intervention** — modify attention keys/values rather than layer outputs, which are harder for the model to self-correct
- **Contrastive steering** — push away from failure-mode representations rather than toward predicted successes
- **Weaker models on harder tasks** — where baseline is 40-60%, there's more room for intervention
- **Multi-layer injection** — perturb at several layers simultaneously to overwhelm self-correction
- **Fine-tuning the model to accept injections** — train the model to expect and utilize external hidden-state modifications (Phase 3/4 of original plan)

## Project Structure

```
code/src/
  extract_states.py    — Batched hidden state extraction from Qwen3-4B
  analysis.py          — Cosine similarity, PCA, linear probe analysis
  mlp_probe.py         — MLP probe diagnostic (linear vs nonlinear)
  train_jepa.py        — Phase 1: JEPA predictor training + baselines
  inject_jepa.py       — Phase 2: Injection during decoding + evaluation

results/
  jepa_phase1/         — JEPA training results (checkpoints not committed)
  jepa_phase2/         — Injection experiment results (per-example JSONL not committed)

PLAN.md                — Original 4-phase plan
LABNOTES.md            — Detailed log of all experiments with results
GUIDELINES.md          — Working discipline and go/no-go criteria
```
