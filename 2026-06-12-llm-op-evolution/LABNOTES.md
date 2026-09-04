# Lab notebook — LLM operator evolution

Reconstructed from the original `results/run.log` (dropped from the tree, still in
history), `results/metrics.json` and the commit history. The original repo kept no
separate notebook.

## 2026-06-12 — batch export, 16 models (10:04–10:29 local, ~25 min)

Strix Halo gfx1151, torch 2.11.0+rocm7.13.0, Python 3.13.6 via uv, Hugging Face hub
unauthenticated. `scripts/run_analysis.py --device cuda` over the manifest. Per model:
download the smallest checkpoint → `torch.export` one forward (batch 1, seq 128, fp32
unless noted, eager attention) → count Core ATen `call_function` nodes by category →
divide by transformer layers. All 16 succeeded; the Qwen 1.5 MoE export was the slow one
(~9 min, mostly download).

| model | nodes/layer (this pass) |
|---|--:|
| Pythia 410M | 77.0 |
| OPT-125M | 62.9 |
| BLOOM 560M | 60.0 |
| TinyLlama 1.1B | 103.5 |
| Phi-2 | 91.3 |
| StableLM 2 1.6B | 89.9 |
| OLMo 1B | 90.4 |
| SmolLM 360M | 102.4 |
| Qwen 2 0.5B | 103.2 |
| DeepSeek Coder 1.3B | 95.2 |
| Qwen 1.5 MoE A2.7B | 154.8 |
| Qwen 2.5 0.5B | 103.2 |
| Phi-3 Mini | 104.7 |
| Qwen 3 0.6B | 118.7 |
| OLMo 2 1B | 114.6 |
| SmolLM2 360M | 102.4 |

The final `results/summary.txt` reports Phi-2 90.3, Qwen 1.5 MoE 152.8 and Phi-3 Mini
103.7, so those three were re-exported later (fp16 for Phi-2 and the MoE, per the report's
dtype caveat); the re-runs are not in the saved log.

## 2026-06-12 → 06-18 — additions not in the saved log

- Six more models appear in `metrics.json`: DeepSeek-Coder-V2-Lite (180.6, fp16),
  DeepSeek-R1-Distill-Qwen-1.5B (109.9), Qwen3-1.7B (129.9), Qwen3.5-4B (1293.0),
  Gemma-4-E2B-it (165.6), SmolLM3-3B (105.2). Gated Gemma 3/4 small variants and some
  DeepSeek checkpoints were unavailable and skipped.
- **Qwen 3.5.** Inspected `Qwen3_5Config`: `layer_types` alternate 3× `linear_attention`
  (Gated DeltaNet) + 1× `full_attention`. Attributed the 41,375-node graph to the
  pure-Python DeltaNet fallback (explicit loops over sequence and chunk, many small
  reshape / pad / cumsum / where ops). Installed `flash-linear-attention` to get the
  kernelised graph: export failed with "Cannot access data pointer of Tensor" because
  `torch.export` cannot trace Triton kernels. Kept the fallback number and flagged it
  non-comparable.
- Added the second lens: `mechanisms.py` taxonomy, per-model `mechanisms:` lists in the
  manifest, `analyze_mechanisms.py`, and the heterogeneity score (categories touched +
  bonus for multiple mechanisms in one category). Regenerated the nine figures, three of
  them new (mechanism heatmap, heterogeneity timeline, density vs heterogeneity).

## 2026-06-18 — commits

Three commits: the analysis, a root README, then `pyproject.toml` metadata (MIT).
