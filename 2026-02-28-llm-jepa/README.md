# LLM-JEPA: Predicting and Injecting Solution-State Hidden Representations

**Date:** 2026-02-28 → 2026-03-01 · **Model:** Qwen3-4B (36 layers, d=2560) · **Data:** GSM8K ·
**Machine:** single GPU (the notes do not record which; sdpa attention, no flash-attn)

## Brief

Can a JEPA-style predictor, trained on an LLM's own (problem-state, solution-state)
hidden-representation pairs, learn a "solution direction" that improves accuracy when
injected into the residual stream during early decoding?

Four planned phases with explicit go/no-go gates (`PLAN.md`): (0) does learnable
problem→solution structure exist in the hidden states; (1) can a lightweight predictor
bridge it on held-out problems; (2) does injecting the prediction help decoding; (3) can
the predictor improve online. The work stopped after Phase 2 failed its gate.

## Headline results

- Last-token "solution states" are not predictable from the problem (best linear-probe
  R² 0.026). The problem→solution delta has cosine consistency 0.97–0.99 across
  examples: a single "I'm done" direction, not solution content.
- The first generated token is the one predictable point: R² 0.635 at layer 21. A
  26M-parameter residual MLP adds only +0.7% R² over ridge regression, so the mapping
  is essentially linear.
- Phase 1 passed its gate: JEPA (MSE loss) cos 0.9883 / R² 0.645 vs ridge 0.9875 /
  0.639. MSE beat cosine loss because injection needs magnitude, not just direction.
- Phase 2 failed: no α ∈ {0.001…0.1} or K ∈ {1…20} beat the 90.83% baseline solve
  rate. At the chosen config JEPA matched baseline exactly (McNemar p=0.855), flipping
  15 examples each way; random-perturbation and mean-delta controls were
  indistinguishable from baseline too.
- Read: predictability ≠ controllability. The prediction carries no information the
  model lacks at layer 21, perturbations small enough not to hurt are absorbed by the
  remaining 15 layers, and larger ones hurt.

## Contents

| Path | What |
|---|---|
| `REPORT.md` | Findings: extraction-point search, predictor training, injection sweeps, why injection failed, directions if continuing |
| `LABNOTES.md` | Experiment log P0.1–P2.3 with hypothesis, setup, result and interpretation per run, plus the phase decisions |
| `PLAN.md` | The original four-phase plan and its gates |
| `GUIDELINES.md` | Working discipline the run followed (logging, seeds, go/no-go rules) |
| `code/` | Extraction, analysis, probe, JEPA training and injection scripts; `code/README.md` has run commands |
| `results/` | Phase 1 predictor metrics (`jepa_phase1/`) and Phase 2 sweep/evaluation JSONs (`jepa_phase2/`) |

Hidden-state caches (~13 GB) and the trained predictor checkpoints were not committed;
the scripts regenerate them. Imported from `ianbarber/llmjepa` with history.
