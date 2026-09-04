# LLM-JEPA Experiment Log

## How to Use This Log
- Each experiment gets a unique ID (P0.1, P0.2, P1.1, etc.)
- Record hypothesis, method, result, and interpretation for every experiment
- Record negative results — they're as informative as positive ones
- Update decision checkpoints after completing each phase

---

## Phase 0: Validate the Premise

### Decision Checkpoint (last-token extraction)
- [x] Linear probe R² > 0.3 at some layer on held-out problems → **FAIL** (best R²=0.026)
- [x] JEPA-predicted states beat mean-solution baseline by >0.1 cosine sim → **FAIL** (+0.0002)
- [x] **Decision:** FAIL on last-token extraction. Revising extraction points before stopping.

### Experiments

#### P0.1 — Hidden State Extraction (last-token)
- **Date:** 2026-02-28
- **Hypothesis:** We can extract meaningful (problem, solution) hidden state pairs from correct completions
- **Setup:** Qwen3-4B (36 layers, hidden_dim=2560), GSM8K train split, greedy decoding with chat template + no_think. Problem state = last prompt token, solution state = last generated token. States extracted at all 36 layers in fp16.
- **Result:** 800 correct pairs from 876 processed (91.3% solve rate). Avg generated length: 272 tokens, avg prompt length: 97 tokens. Storage: 3x 141MB .pt files (problem, solution, first_step).
- **Notes:** Initial extraction was single-example (27 tok/sec). Optimized batched version achieves 226 tok/sec (8.4x speedup). Flash attention not installed; using sdpa backend.

#### P0.2 — Cosine Similarity Analysis (last-token)
- **Date:** 2026-02-28
- **Hypothesis:** Problem and solution states occupy distinct but related regions of hidden state space
- **Setup:** 800 pairs, all 36 layers, cosine similarity + delta direction consistency
- **Result:**
  - Raw cosine similarity (problem vs solution): ranges from ~-0.05 (layer 0) to ~0.75 (layer 33), indicating moderate divergence
  - Delta direction consistency (cos sim of each delta to mean delta): **0.97-0.99 across ALL layers**
  - Delta norms grow from ~27 (layer 0) to ~372 (layer 34), with a dip at layer 35 (~239)
- **Interpretation:** Problem and solution states are in different regions of the space, BUT the direction of the difference is nearly universal. Almost all problem→solution deltas point the same way. This means there's a single "solution direction" that is NOT problem-specific.
- **Plot:** `data/hidden_states/Qwen_Qwen3-4B_gsm8k_train/analysis/cosine_similarity.png`

#### P0.3 — PCA / Geometry Analysis (last-token)
- **Date:** 2026-02-28
- **Hypothesis:** Problem→solution deltas show consistent directional structure
- **Setup:** PCA + t-SNE at layers 0, 9, 18, 27, 35. 500 subsampled points.
- **Result:** 51 PCA components needed for 90% variance (consistent across layers). Visualizations show problem and solution clusters are overlapping but shifted.
- **Interpretation:** The representations are high-dimensional and the shift is a low-rank (possibly rank-1) universal direction, consistent with the delta consistency finding.
- **Plot:** `data/hidden_states/Qwen_Qwen3-4B_gsm8k_train/analysis/pca_tsne.png`

#### P0.4 — Linear Probe (last-token)
- **Date:** 2026-02-28
- **Hypothesis:** A linear map can predict solution states from problem states
- **Setup:** Ridge regression (alpha sweep [1, 10, 100]), 80/20 train/test split, all 36 layers
- **Result:**
  - Best R²: **0.026** at layer 3 (threshold was 0.3)
  - R² goes negative for layers >5, reaching -1.9 at layer 35
  - Probe cosine similarity ≈ mean baseline cosine similarity at all layers (improvement: +0.0002)
  - Identity baseline (problem=solution) cosine sim: -0.06 to 0.67 across layers
  - Delta direction accuracy: 0.97-0.99 (high, but this just reflects the universal direction)
- **Best layer(s):** Layer 3 (R²=0.026), but effectively no predictive power anywhere
- **Interpretation:** The linear probe cannot predict problem-specific solution states. The mean-solution predictor is equally good because the problem→solution mapping is dominated by a single universal direction. Per-problem variation in the delta is not captured.
- **Plot:** `data/hidden_states/Qwen_Qwen3-4B_gsm8k_train/analysis/linear_probe.png`

#### P0.5 — Cross-Domain Probe
- **Date:** —
- **Status:** Skipped (no point running cross-domain if within-domain probe fails)

---

### Phase 0 Revision: Alternative Extraction Points

The last-token extraction fails because the "solution state" at the final generated token encodes "I'm done with a math solution" rather than problem-specific solution content. We're now testing alternative extraction points:

#### P0.6 — First-Step States Analysis
- **Date:** 2026-02-28
- **Hypothesis:** The first generated token captures the model's initial solution strategy, which may be more problem-specific than the last-token state
- **Setup:** Same 800 pairs, analyzing problem_states vs first_step_states (already extracted)
- **Result:**
  - **Best R²: 0.417 at layer 12** (threshold was 0.3) → **PASS**
  - Top 5 layers: 12 (0.417), 13 (0.412), 17 (0.404), 19 (0.403), 16 (0.401)
  - R² above 0.3 threshold for layers 9-25 (broad plateau in mid-layers)
  - Delta direction consistency: 0.97-0.99 (still high but less than last-token)
  - Cosine similarity (problem vs first_step): 0.5-0.7 across layers (more variation than last-token, which was near-universal)
  - Improvement over mean baseline: +0.005 cosine sim (below 0.1 threshold, but R² passes)
- **Interpretation:** The first generated token carries significantly more problem-specific information than the last token. The R² of 0.42 means a linear map explains ~42% of the variance in first-step states from problem states. The best layers are in the middle of the network (12-19), suggesting this is where the "solution strategy" is most linearly accessible. This is a strong enough signal to proceed to Phase 1.
- **Plot:** `data/hidden_states/Qwen_Qwen3-4B_gsm8k_train/analysis_first_step/linear_probe.png`

#### P0.7 — Answer-Token & Mid-Solution States Extraction & Analysis
- **Date:** 2026-02-28
- **Hypothesis:** The hidden state at the answer token (just before ####) and mid-solution point carry additional problem-specific information
- **Setup:** Qwen3-4B, GSM8K train, 825 correct / 900 processed (91.7%), batched extraction (16x, 226 tok/sec). Extract: answer state (token before ####, found in 822/825 examples), mid state (midpoint of generation). Ridge regression probes at all layers.
- **Result:**
  | Target | Best Layer | Best R² | Layers > 0.3 |
  |--------|-----------|---------|---------------|
  | **first_step** | **19** | **0.396** | **11 (layers 11-21)** |
  | answer | 5 | 0.021 | 0 |
  | mid | 0 | -0.015 | 0 |
  | solution (last) | 3 | 0.026 | 0 |
- **Interpretation:** Only the first-step state is predictable from the problem state. Answer-token and mid-solution states are as unpredictable as the last token — they encode local context (what step of reasoning we're in, what specific numbers are being manipulated) rather than a problem-level strategy. The first generated token is the unique carrier of problem-specific "solution approach" information that a linear map can extract from the problem representation.

#### P0 Summary — Extraction Point Selection
- **Winner: first_step** (first generated token), layers 11-21, best at layer 19
- **Why it works:** At the first decode step, the model has read the full problem and committed to an approach. The hidden state captures this "initial direction" before the model gets into the weeds of arithmetic. By mid-solution or answer time, the state is dominated by local computation context.
- **Implication for JEPA:** Train on (problem → first_step) at mid-layers. Inject the predicted first-step state to give the model a "head start" on its solution approach.

#### P0.8 — MLP Probe Diagnostic (small N=825)
- **Date:** 2026-02-28
- **Hypothesis:** A nonlinear (MLP) probe will capture additional structure beyond the linear probe's R²≈0.4, indicating the JEPA approach has room to improve over a linear projection
- **Setup:** 2-layer MLP with residual connection (hidden=2×D=5120, GELU, dropout=0.1), AdamW lr=1e-4, cosine schedule, early stopping patience=30. Train/val/test split 70/15/15 (577/124/124 examples). Layers tested: 12, 15, 17, 19, 21.
- **Result:**
  | Layer | Ridge R² | MLP R² | MLP Δ |
  |-------|---------|--------|-------|
  | 12 | 0.418 | 0.273 | -0.14 |
  | 15 | 0.400 | 0.238 | -0.16 |
  | 17 | 0.430 | 0.268 | -0.16 |
  | 19 | 0.456 | 0.379 | -0.08 |
  | 21 | 0.481 | 0.436 | -0.04 |
- **Interpretation:** MLP is WORSE than ridge at every layer — overfitting, not finding nonlinear structure. The MLP has ~26M params but only 577 training examples (~45K params/example). Cannot distinguish "structure is genuinely linear" from "not enough data to train MLP" at this sample size.
- **Decision:** Run full GSM8K extraction (~6800 correct pairs expected) and re-run diagnostic.

#### P0.9 — Full GSM8K Extraction + MLP Probe (large N)
- **Date:** 2026-02-28
- **Hypothesis:** With 6000+ training examples (10x more), the MLP probe will be able to learn properly and may reveal nonlinear structure
- **Setup:** Full GSM8K train split (7473 examples), batched extraction (batch_size=16, ~226 tok/sec). All 5 extraction points (problem, solution, first_step, answer, mid). MLP probe on first_step target at best layers.
- **Result:**
  - Extraction: 6935 correct / 7473 total (92.8% solve rate), 0 errors
  - MLP probe (train=4854, val=1040, test=1041):
  | Layer | Ridge R² | MLP R² | MLP Δ |
  |-------|---------|--------|-------|
  | 12 | 0.550 | 0.557 | +0.007 |
  | 15 | 0.546 | 0.543 | -0.003 |
  | 17 | 0.584 | 0.580 | -0.004 |
  | 19 | 0.615 | 0.613 | -0.001 |
  | **21** | **0.635** | **0.642** | **+0.008** |
- **Interpretation:** With sufficient data, three things become clear: (1) The linear probe R² jumps from ~0.4 (N=825) to ~0.6 (N=6935) — the small-N Ridge was underfitting. (2) The MLP adds essentially nothing over Ridge (+0.008 max) — the mapping is genuinely linear. (3) Best layer shifts from 12→21 with more data — later layers are better when the probe has enough examples. Conclusion: **the problem→first_step mapping is strongly linear (R²=0.64)** with no exploitable nonlinear structure. A linear or near-linear JEPA predictor should suffice.

#### P0 Final Summary — Phase 0 Complete
- **Best extraction point:** first_step (first generated token)
- **Best layer:** 21 (R²=0.635 linear, 0.642 MLP)
- **Nature of mapping:** Strongly linear, no significant nonlinear structure
- **Data size effect:** R² improves from ~0.4 (N=825) to ~0.6 (N=6935) — larger training sets help
- **Implication for Phase 1:** Use a simple predictor (linear + small residual MLP). Layer 21 is the primary target, with 17-21 as the working range. The ceiling is ~0.64 R², so don't over-engineer the architecture.

---

## Phase 1: Train the JEPA Module

### Decision Checkpoint
- [x] JEPA cosine sim >= Ridge baseline (~0.988) → **PASS** (0.9883 ≥ 0.9875)
- [x] Direction accuracy (delta cosine sim) > 0.2 → **PASS** (0.9831)
- [x] **Decision:** PASS — proceed to Phase 2

### Experiments

#### P1.1 — JEPA Training (cosine loss)
- **Date:** 2026-02-28
- **Architecture:** 2-layer MLP (dim=2560, hidden=5120), residual connection, GELU, dropout=0.1 (~26M params)
- **Training:** Cosine loss (1 - cos_sim), AdamW lr=1e-4 wd=1e-4, CosineAnnealingLR T_max=300, early stopping patience=30, batch_size=128. Split: 5548/693/694 (80/10/10). Trained for 289 epochs in 38s on GPU.
- **Result (test set):**
  | Method | Cos Sim | MSE | R² | ΔDir Cos |
  |--------|---------|-----|-----|----------|
  | Mean | 0.9664 | 0.1326 | -0.001 | 0.9504 |
  | Identity | 0.6254 | 1.3663 | -9.001 | 0.0000 |
  | Ridge | 0.9875 | 0.0492 | 0.639 | 0.9819 |
  | k-NN (k=5) | 0.9807 | 0.0757 | 0.395 | 0.9719 |
  | **JEPA** | **0.9884** | 0.1196 | 0.360 | 0.9565 |
- **Interpretation:** Cosine loss achieves the best cosine similarity (0.9884) but poor R² (0.36) and high MSE (0.12). The loss optimizes direction at the expense of magnitude — predictions are well-aligned but scaled incorrectly. Go/no-go criteria pass on cosine sim but the R² regression vs Ridge (0.64) suggests this loss is suboptimal for injection, where magnitude matters.

#### P1.2 — Loss Function Comparison (cosine vs MSE)
- **Date:** 2026-02-28
- **Compared:** Cosine loss vs MSE loss (same architecture, same split)
- **Result:**
  | Loss | Cos Sim | MSE | R² | ΔDir Cos | Epochs |
  |------|---------|-----|-----|----------|--------|
  | Cosine | 0.9884 | 0.1196 | 0.360 | 0.9565 | 289 |
  | **MSE** | **0.9883** | **0.0460** | **0.645** | **0.9831** | 225 |
  | Ridge | 0.9875 | 0.0492 | 0.639 | 0.9819 | — |
- **Winner:** **MSE loss** — nearly identical cosine sim (0.9883 vs 0.9884) but dramatically better MSE (0.046 vs 0.120), R² (0.645 vs 0.360), and delta direction (0.983 vs 0.957). MSE-trained JEPA slightly outperforms Ridge on every metric. Cosine loss sacrifices magnitude for marginal directional gain.
- **Key insight:** The JEPA with MSE loss achieves R²=0.645, slightly above Ridge's 0.639 — confirming Phase 0 finding that the mapping is nearly linear. The 26M-param MLP adds ~0.6% R² over a linear map, consistent with the MLP probe diagnostic (P0.9). The residual architecture is working as intended: learning a small correction on top of identity.
- **Best checkpoint:** `experiments/jepa_phase1/best_model_mse.pt`

#### P1.3 — Layer Selection Validation
- **Date:**
- **Hypothesis:** The best probe layer from P0.4 is also the best JEPA layer
- **Setup:** Train JEPA at top-3 layers from probe results
- **Result:**
- **Interpretation:**

#### P1.4 — Cross-Domain Generalization
- **Date:**
- **Setup:** JEPA trained on GSM8K, evaluated on MATH and vice versa
- **Result:**
- **Interpretation:**

---

## Phase 2: Injection During Decoding

### Decision Checkpoint
- [x] Statistically significant solve rate improvement over base model → **FAIL** (p=0.855, no improvement)
- [x] Improvement is not replicated by random perturbation baselines → **N/A** (no improvement to test)
- [x] **Decision:** **FAIL** — JEPA injection does not improve solve rate at any alpha or K value. The predicted hidden-state delta, when injected during decoding, has no measurable effect on problem-solving accuracy. See analysis below.

### Experiments

#### P2.1 — Alpha Sweep
- **Date:** 2026-03-01
- **Setup:** Injection formula: `h_modified = h_original + α × (JEPA_predict(h_problem) - h_problem)` at layer 21. Forward hook on `model.model.layers[21]`. Prefill step extracts problem state, decode steps 1..K inject delta. Fixed K=5. α values: [0.001, 0.005, 0.01, 0.05, 0.1]. Evaluated on full GSM8K test split (1319 examples), greedy decoding, max_new_tokens=512.
- **Result:**
  | Mode | Alpha | Solve Rate | 95% CI | Correct | Avg Tokens |
  |------|-------|-----------|--------|---------|------------|
  | none (baseline) | — | **90.83%** | [0.891, 0.923] | 1198 | 291.9 |
  | jepa | 0.001 | **90.83%** | [0.891, 0.923] | 1198 | 291.1 |
  | jepa | 0.005 | 90.30% | [0.886, 0.918] | 1191 | 291.2 |
  | jepa | 0.01 | 90.07% | [0.883, 0.916] | 1188 | 291.3 |
  | jepa | 0.05 | 90.60% | [0.889, 0.921] | 1195 | 290.6 |
  | jepa | 0.1 | 90.67% | [0.890, 0.921] | 1196 | 289.9 |
- **Best α:** 0.001 (ties with baseline; all other alphas slightly hurt performance)
- **Interpretation:** No alpha improves over baseline. At α=0.001, the perturbation is negligible (~0.056 L2 norm vs hidden state norms of ~100+), so it acts as a no-op. Larger alphas (0.005-0.01) slightly degrade performance by 0.5-0.8%. Very large alphas (0.05, 0.1) are closer to baseline, suggesting the model partially recovers from the perturbation. All confidence intervals overlap heavily — differences are within noise.

#### P2.2 — Injection Duration (K) Sweep
- **Date:** 2026-03-01
- **Setup:** Fixed α=0.001 (best from P2.1). K values: [1, 5, 10, 20]. Same evaluation setup as P2.1.
- **Result:**
  | Mode | K | Solve Rate | 95% CI | Correct | Avg Tokens |
  |------|---|-----------|--------|---------|------------|
  | none (baseline) | — | **90.83%** | [0.891, 0.923] | 1198 | 291.9 |
  | jepa | 1 | 90.14% | [0.884, 0.916] | 1189 | 291.9 |
  | jepa | 5 | **90.83%** | [0.891, 0.923] | 1198 | 291.1 |
  | jepa | 10 | 90.67% | [0.890, 0.921] | 1196 | 290.7 |
  | jepa | 20 | 90.37% | [0.887, 0.918] | 1192 | 290.6 |
- **Best K:** 5 (ties with baseline; all other K values slightly hurt)
- **Interpretation:** Again no improvement. K=1 is slightly worse (-0.7%), K=5 ties, K=10 and K=20 slightly below baseline. Since α=0.001 is effectively a no-op, varying K has negligible effect. The tiny differences (2-9 questions out of 1319) are stochastic noise — the injection changes which specific tokens the model generates in ways that occasionally flip borderline answers but with no systematic improvement.

#### P2.3 — Full Evaluation (best config)
- **Date:** 2026-03-01
- **Config:** α=0.001, K=5, layer 21. Four modes tested: none (baseline), jepa (JEPA-predicted delta), random (same-norm random direction), mean_delta (precomputed mean of training first_step - problem). McNemar's test for paired comparison vs baseline.
- **Result:**
  | Mode | Solve Rate | 95% CI | Correct | Avg Tokens | McNemar p |
  |------|-----------|--------|---------|------------|-----------|
  | none (baseline) | **90.83%** | [0.891, 0.923] | 1198 | 291.9 | — |
  | jepa | **90.83%** | [0.891, 0.923] | 1198 | 291.1 | 0.855 |
  | random | 90.52% | [0.888, 0.920] | 1194 | 291.1 | 0.556 |
  | mean_delta | 90.37% | [0.887, 0.918] | 1192 | 291.8 | 0.286 |
- **McNemar's test details:**
  - jepa vs none: b=15 (none wrong, jepa right), c=15 (none right, jepa wrong) → p=0.855 (no difference)
  - random vs none: b=11, c=15 → p=0.556 (no difference)
  - mean_delta vs none: b=8, c=14 → p=0.286 (no difference)
- **Go/No-Go:**
  - JEPA > baseline: **FAIL** (identical: 90.83% = 90.83%)
  - JEPA significant (p<0.05): **FAIL** (p=0.855)
  - Random not significant: PASS (p=0.556, expected)
  - Random not better than JEPA: PASS (90.52% < 90.83%)
  - **Verdict: FAIL — No improvement from JEPA injection**
- **Interpretation:** The JEPA injection has zero measurable effect on solve rate. The McNemar test confirms that JEPA and baseline make errors on essentially the same examples (15 flip each way = exact symmetry). At α=0.001, the injection is too small to meaningfully perturb the model's computation. The "mean_delta" baseline, which injects a non-problem-specific direction, performs slightly worse but not significantly so. All modes are within noise of baseline.

#### P2.4 — Adaptation Fine-tuning
- **Date:** —
- **Status:** Skipped. No injection benefit to preserve through fine-tuning.

### Phase 2 Analysis

**Why the injection doesn't work — three contributing factors:**

1. **The perturbation is too small.** At α=0.001, the injected delta has L2 norm ~0.056, while the hidden states at layer 21 have norms of ~100-200. The perturbation is <0.05% of the signal. But making it larger (α=0.005-0.1) slightly hurts rather than helps.

2. **The JEPA prediction adds minimal new information.** Phase 0/1 showed the mapping is strongly linear (R²=0.64) and the JEPA only adds +0.6% R² over Ridge. The residual (what JEPA learns beyond identity) is tiny — the model already encodes most of the first-step information in the problem representation. Injecting a prediction that's 98.8% correlated with what the model already has doesn't provide new steering signal.

3. **The model may be robust to this type of perturbation.** Transformer layer outputs feed through LayerNorm and attention mechanisms that can absorb small additive perturbations. The model's self-correction mechanisms (residual stream, attention re-weighting) may neutralize the injection within 1-2 subsequent layers.

---

## Phase 3: Continual Learning

### Experiments

#### P3.1 — Online JEPA Update
- **Date:**
- **Setup:** Replay buffer size, update frequency, evaluation schedule
- **Acquisition curve:**
- **Retention curve:**
- **Interpretation:**

---

## Running Notes

### Unexpected Observations
- Delta direction consistency is 0.97-0.99 across all layers — the problem→solution direction in hidden state space is nearly universal (not problem-specific) when using last-token extraction. This is a strong signal that the last token encodes "completion status" rather than "solution content."
- Delta norms grow dramatically in later layers (27→372 from layer 0 to 34), then drop at the final layer (239). The final layer norm may be compressing the representation.
- **First-step token is dramatically more predictable than last token.** R² jumps from 0.026 (last token) to 0.417 (first step). The model's very first generated token already encodes significant problem-specific solution information, and this is linearly accessible from the problem representation at mid-layers (12-19).
- The predictability plateau is in mid-layers (9-25), not early or late. Early layers may not have enough processing, late layers may be too specialized for next-token prediction.

### Ideas for Future Work
- Could try pooled/averaged states across all generated tokens instead of single-token extraction
- Sequence-level representations (e.g., mean of all reasoning tokens before the answer) might capture richer solution information
- Could investigate whether the universal direction is actually useful if injected (a "mean-delta injection" test from Phase 2)

### Technical Debt / Known Issues
- flash_attn package not installed; falling back to sdpa. Could install for additional 10-30% speedup.
- Original single-example extraction script still present in git history; current version is batched (v2).
