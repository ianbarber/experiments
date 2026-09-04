# LLM operator-graph evolution 2022–2026: Core ATen density and mechanism heterogeneity

**Date:** 2026-06-12 → 2026-06-18 · **Machine:** AMD Strix Halo (Ryzen AI Max+ 395, gfx1151),
torch 2.11.0+rocm7.13.0, Python 3.13

## Brief

Has the *frontend* complexity of open causal language models grown over four years, and
by how much? Two lenses over 22 small open checkpoints (April 2022 → May 2026):

1. **Core ATen operator density** — nodes per transformer layer in a `torch.export`
   graph of one forward pass (batch 1, seq 128).
2. **Architectural mechanism inventory** — a hand-curated checklist of attention
   variants, MoE routing, positional encoding, normalisation, MTP, multimodal and
   model-level features, scored into a heterogeneity number.

Neither is FLOPs, parameters or wall-clock; both are structural.

## Headline results

- 2022 dense models sit at ~60–77 nodes/layer; the 2023–24 Llama-style recipe forms a
  plateau at ~90–105; MoE (Qwen1.5-MoE 153, DeepSeek-Coder-V2-Lite 181) is the first
  clear step up; 2025 dense models refine to ~110–130.
- 2026 splits three ways: SmolLM3 stays on the plateau (105), Gemma 4 E2B is denser
  (166), and Qwen 3.5 4B is an outlier at ~1293 nodes/layer.
- The outlier is real but not comparable. 24 of its 32 layers are Gated DeltaNet linear
  attention; without the `flash-linear-attention` / `causal-conv1d` kernels installed,
  `transformers` falls back to Python recurrence loops that `torch.export` unrolls into
  ~41K nodes. With FLA installed the export fails outright, because `torch.export`
  cannot trace Triton kernels.
- The mechanism inventory is the fairer throughline: heterogeneity 3.0–4.5 (2022) →
  4.0–5.5 (2023–25) → 5.5 / 7.5 / 8.0 (SmolLM3 / Gemma 4 / Qwen 3.5). Qwen 3.5 is
  barely more heterogeneous than Gemma 4 yet ~8× denser, which is the signature of
  the implementation artefact, not an architectural leap.
- Standing caveat: anything dispatched to a fused Triton / CUDA / ROCm kernel is
  invisible to this counter, so the metric is biased toward what the PyTorch frontend
  can see.

## Contents

| Path | What |
|---|---|
| `REPORT.md` | Method, dataset, the six findings, figure guide, caveats, reproduction |
| `LABNOTES.md` | Run record: the 2026-06-12 batch export, the later additions, the FLA attempt |
| `code/` | Model manifest (`config/models.yaml`), export/analysis package (`src/llm_op_evolution/`), drivers (`scripts/`); `code/README.md` has run commands |
| `results/` | `metrics.json` (aggregated), `summary.txt`, per-model JSONs, nine figures under `figures/` |

The 132 KB HTTP-level `run.log` was dropped from the tree on import; it remains in the
merged history. Imported from `ianbarber/llm-op-evolution` with history.
