# LLM-JEPA: Solution-State Prediction for Guided Decoding

## Research Question

Can a JEPA module, trained on an LLM's own (problem-state, solution-state) hidden representation pairs, learn to predict useful "solution directions" that, when injected into the residual stream during early decoding, improve solution accuracy and efficiency?

## Sub-questions (in priority order)

1. **Do problem and solution hidden states have learnable, generalizable structure?** (Phase 0)
2. **Can a lightweight predictor bridge problem→solution state reliably on held-out problems?** (Phase 1)
3. **Does injecting predicted solution states actually help the model generate better/faster solutions?** (Phase 2)
4. **Can the predictor improve online without degrading on prior domains?** (Phase 3)

---

## Phase 0: Validate the Premise

**Goal:** Determine whether there is learnable, generalizable structure mapping problem hidden states to solution hidden states.

**Go/No-Go Gate:** Linear probe R² > 0.3 on held-out problems, AND JEPA-predicted states are closer to true solution states than the mean-solution baseline by a meaningful margin (>0.1 cosine similarity improvement).

### 0.1 Setup
- **Model:** Qwen3-4B (or Qwen2.5-3B as fallback)
- **Datasets:**
  - GSM8K (train split for JEPA training, test split held out entirely)
  - MATH (subset — start with algebra/number theory for clear problem/solution structure)
- **Hardware:** Single GPU with sufficient VRAM to run inference + store hidden states

### 0.2 Hidden State Extraction
- Run model on each problem, collect full successful completions (filter for verified-correct answers only)
- Extract hidden states at **every layer** (not just N/2 — we need the full picture)
- **Problem state:** hidden state at the last token of the problem/prompt, at each layer
- **Solution state:** hidden state at the last token before the final answer token, at each layer
  - Also try: hidden state at the first generated token (captures "initial direction")
- Store as fp16 to halve memory (at hidden_dim ~3584 for Qwen3-4B, each state is ~7KB in fp16)
- **Budget:** ~7K GSM8K train examples, ~5K MATH examples. At ~7KB per state per layer × 36 layers × 2 states × 12K examples ≈ 6GB. Manageable.

### 0.3 Analysis
1. **Cosine similarity distribution:** For each layer, compute cosine sim between problem and solution states. Plot distribution. Identify layers where they're neither too similar (nothing to learn) nor too dissimilar (gap too large).
2. **PCA / t-SNE visualization:** Project problem and solution states into 2D. Look for: are problem-solution pairs connected by consistent directions? Do clusters emerge by problem type?
3. **Linear probe:** For each layer, train a linear map (ridge regression) from problem state → solution state. Measure R² and cosine similarity on held-out problems.
4. **Predictability curve:** Plot linear probe R² across layers. This tells us WHERE the problem→solution mapping lives.
5. **Cross-domain check:** Train probe on GSM8K, evaluate on MATH (and vice versa). How domain-specific is the mapping?

### 0.4 Decision Criteria
- If linear probe R² > 0.3 at some layer → proceed with that layer (or nearby layers) to Phase 1
- If R² < 0.1 everywhere → the mapping may not exist in a learnable form; reconsider approach
- If R² is 0.1–0.3 → proceed cautiously; the MLP predictor in Phase 1 may bridge the gap but expectations should be tempered

---

## Phase 1: Train the JEPA Module

**Goal:** Train a predictor that maps problem states to solution states better than baselines.

**Go/No-Go Gate:** JEPA predictor achieves >0.15 cosine similarity improvement over mean-solution baseline on held-out problems.

### 1.1 Architecture
- **Predictor:** 2-layer MLP with residual connection
  - Input: problem hidden state (dim D)
  - Hidden: 2×D (with GELU activation)
  - Output: predicted solution state (dim D)
  - Residual: output = MLP(input) + input
- **Why residual:** problem and solution states may be close; predictor only needs to learn the delta

### 1.2 Training
- **Data split:** 80% train, 10% val, 10% test (stratified by difficulty/problem type)
- **CRITICAL: The test set problems must NEVER have their solution states seen by the JEPA.** This guards against causal contamination.
- **Loss options (try in order):**
  1. Cosine similarity loss (simple, directly optimizes what we measure)
  2. MSE in representation space
  3. VICReg-style (if collapse is observed with options 1-2)
- **Layer selection:** Use the best layer(s) from Phase 0's predictability curve
- **Training details:** AdamW, lr=1e-4 with cosine schedule, batch size 64, train until val loss plateaus
- **Total parameters:** ~2×D² for a 2-layer MLP ≈ 2 × 3584² ≈ 25M parameters. Tiny relative to the base model.

### 1.3 Baselines
1. **Mean predictor:** always predicts the mean solution hidden state from the training set
2. **Identity predictor:** predicts that solution state = problem state (tests whether the delta matters)
3. **k-NN predictor:** for a test problem, find the k nearest training problem states and average their solution states (tests whether the MLP learns more than nearest-neighbor interpolation)

### 1.4 Evaluation Metrics
- Cosine similarity between predicted and actual solution states (held-out)
- MSE between predicted and actual solution states (held-out)
- **Direction accuracy:** does the predicted delta (predicted - problem) point in roughly the same direction as the true delta (solution - problem)? Measure cosine sim of deltas.
- Generalization gap: performance on in-distribution (GSM8K→GSM8K) vs. cross-distribution (GSM8K→MATH)

---

## Phase 2: Injection During Decoding

**Goal:** Determine whether injecting JEPA-predicted solution states improves generation quality.

**Go/No-Go Gate:** Statistically significant improvement in solve rate over base model on held-out problems.

### 2.1 Injection Method
- **Residual injection:** At specified decode steps, add the predicted delta to the residual stream:
  ```
  h_modified = h_original + α × (JEPA_predict(h_problem) - h_problem)
  ```
  Note: we inject the *delta*, not the raw predicted state. This is more principled — we're nudging the model in the solution direction rather than overwriting its state.
- **Injection layer:** Same layer the JEPA was trained on
- **α schedule:** Start at 0.001 and sweep [0.001, 0.005, 0.01, 0.05, 0.1]. Expect useful range to be very small.
- **Injection duration (K):**
  - Fixed: inject for first K ∈ {1, 5, 10, 20} tokens
  - Adaptive: inject until model entropy drops below threshold (model has "found its footing")
  - Decaying: inject with α × decay^step for all steps

### 2.2 Evaluation
- **Primary:** Solve rate (% problems answered correctly) on held-out test sets
- **Secondary:**
  - Average tokens to correct answer (efficiency)
  - First-token accuracy (does the model "start right" more often?)
  - Perplexity of generated tokens
  - Calibration: is the model more/less well-calibrated with injection?
- **Sample size:** Minimum 500 held-out problems per condition for statistical power

### 2.3 Baselines & Controls
1. **Base model (no injection):** the floor to beat
2. **Random perturbation:** inject noise of the same L2 norm as the JEPA delta, in a random direction. Controls for "any perturbation helps" (implicit regularization / diversity)
3. **Directionally-matched random:** inject noise with the same average cosine similarity to the problem state as the JEPA delta. Controls for direction statistics.
4. **Mean-delta injection:** inject the mean (solution - problem) delta from training data. Controls for "is a problem-specific prediction better than a generic nudge?"

### 2.4 Optional: Adaptation Fine-tuning
If raw injection degrades output quality:
- Freeze JEPA, freeze most of the LLM
- Fine-tune only the LLM's final 2-4 layers (or use LoRA on them) with injection active
- This teaches the model to "listen to" the JEPA signal
- Train on a subset of correct completions WITH injection enabled

---

## Phase 3: Continual Learning (conditional on Phase 2 success)

**Goal:** Can the JEPA improve online as the model encounters new problems?

### 3.1 Setup
- Freeze base LLM entirely
- JEPA parameters are updatable
- Replay buffer (size 1000) to mitigate catastrophic forgetting

### 3.2 Protocol
1. Start with JEPA trained on GSM8K
2. Stream MATH problems in order of difficulty
3. For each problem:
   - Generate solution with JEPA injection
   - If solution is correct: add (problem_state, solution_state) to replay buffer, update JEPA
   - If incorrect: do not update (avoid learning from bad solutions)
4. Periodically evaluate on held-out GSM8K (retention) and held-out MATH (acquisition)

### 3.3 Metrics
- **Acquisition:** solve rate on new MATH problems over time
- **Retention:** solve rate on GSM8K over time (should not degrade)
- **Efficiency:** does the model need fewer tokens to solve problems it's seen similar versions of?

---

## Practical Details

### Model Choice
- **Primary:** Qwen3-4B (good balance of capability and iteration speed)
- **Fallback:** Qwen2.5-3B (smaller, faster iteration)
- Rationale: need a model that can solve some but not all problems in our dataset — too capable and there's nothing to improve; too weak and the hidden states may lack structure

### Compute Budget
- Phase 0: ~2-4 GPU-hours (inference + analysis)
- Phase 1: ~1-2 GPU-hours (training the small JEPA)
- Phase 2: ~4-8 GPU-hours (generation across multiple conditions)
- Phase 3: ~4-8 GPU-hours (online learning loop)
- **Total estimated: ~15-25 GPU-hours**

### Key Risks
1. **The mapping doesn't exist:** problem→solution may not be a learnable function in hidden state space. Phase 0 catches this early.
2. **Causal contamination:** leaking solution info rather than learning generalizable structure. Strict held-out evaluation catches this.
3. **Injection disrupts the model:** the base model never saw this signal during training. Starting with tiny α and considering adaptation fine-tuning mitigates this.
4. **Domain specificity:** the JEPA might only work for the exact distribution it was trained on. Cross-domain evaluation in Phases 0-1 measures this.
