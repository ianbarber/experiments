# LLM Operator Evolution

Track how the frontend / architectural complexity of open causal language models
has changed from 2022 to 2026, using two complementary lenses:

1. **Core ATen operator-graph density** — nodes per transformer layer from a
   `torch.export` Core ATen IR graph.
2. **Architectural mechanism inventory** — a curated checklist of attention
   variants, routing, normalization, position encoding, and model-level features,
   used to compute a heterogeneity score.

The project is built around a small, reproducible pipeline:
`config/models.yaml` → `torch.export` → `results/metrics.json` → figures.

## Quick start

Requires Python 3.11+ and a working PyTorch install (ROCm/CUDA/CPU).
The repo was developed on an AMD Strix Halo (gfx1151) machine with
`torch==2.11.0+rocm7.13.0`.

```bash
# Install dependencies (example with uv; pip works too)
uv pip install -e .

# Export all models and regenerate figures
HF_HUB_ENABLE_HF_TRANSFER=0 python scripts/run_analysis.py --device cuda

# Analyze architectural mechanisms and regenerate figures from existing results
python scripts/analyze_mechanisms.py
python scripts/run_analysis.py --plot-only
```

To export a single model:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 python scripts/run_analysis.py --device cuda --models qwen3.5-4b
```

## What is measured

For each model in `config/models.yaml`:

* Load the smallest publicly available checkpoint from a model family.
* Export a single forward pass (`batch_size=1`, `seq_len=128`) at the Core ATen
  IR level using `torch.export`.
* Count `call_function` nodes, split by coarse operator category, and normalize
  by the number of transformer layers.
* Manually annotate + config-infer the architectural mechanisms present
  (attention type, MoE, RoPE, RMSNorm, MTP, multimodal, etc.).

**Important:** this is not total FLOPs, parameter count, or wall-clock time.
It is a structural measure of graph density and architectural heterogeneity.

## Project structure

```
config/models.yaml              # Manifest of models to export
src/llm_op_evolution/
  export_model.py               # torch.export pipeline
  op_utils.py                   # Graph traversal and categorization
  plot.py                       # Figure generation
  mechanisms.py                 # Mechanism taxonomy + heterogeneity scoring
scripts/
  run_analysis.py               # End-to-end export + plot
  analyze_mechanisms.py         # Update metrics with mechanism inventory
  print_summary.py              # Compact CLI summary table
results/
  metrics.json                  # Aggregated per-model metrics
  summary.txt                   # Compact text table
  README.md                     # Detailed interpretation and caveats
  figures/                      # Generated PNG plots
```

## Key findings (high level)

See `results/README.md` for the full narrative.

* **2022 dense models** sit at ~60–77 Core ATen nodes/layer (Pythia, OPT, BLOOM).
* **2023–2024 Llama-style stacks** converge around ~90–105 nodes/layer.
* **2024 MoE models** push the ceiling to ~150–180 nodes/layer.
* **2025 models** refine the recipe without a large jump.
* **2026** is mixed: SmolLM3 stays on the plateau, Gemma 4 is denser but
  comparable, and Qwen 3.5 is a massive outlier at ~1293 nodes/layer.

The outlier is **real but not directly comparable**: Qwen 3.5 uses Gated
DeltaNet linear attention, but the current environment lacks the optimized
`flash-linear-attention`/`causal-conv1d` kernels, so `transformers` falls back
to pure-Python recurrence loops that `torch.export` materializes as thousands
of nodes. Installing FLA makes the export fail because its Triton kernels are
opaque to `torch.export`.

The **mechanism inventory** gives a fairer throughline:

* 2022 models: heterogeneity score 3.0–4.5
* 2023–2025 Llama/MoE models: 4.0–5.5
* 2026: SmolLM3 5.5, Gemma 4 7.5, Qwen 3.5 8.0

Qwen 3.5 is only slightly more heterogeneous than Gemma 4, but its ATen density
is ~8× higher — the gap is the implementation artifact, not an 8× architectural
leap.

## Figures

Generated in `results/figures/`:

| Figure | Description |
|--------|-------------|
| `timeline_density.png` | Linear-scale ATen density over time |
| `timeline_density_log.png` | Log-scale version |
| `timeline_density_dense_only.png` | Dense-transformer trend only |
| `year_density.png` | Year-binned median/mean density |
| `category_composition.png` | Operator category breakdown per model |
| `unique_ops.png` | Distinct ATen operator breadth |
| `mechanism_heatmap.png` | Models × mechanisms heatmap |
| `heterogeneity_timeline.png` | Heterogeneity score over time |
| `density_vs_heterogeneity.png` | ATen density vs. mechanism diversity |

## Caveats

* Small variants only; larger variants may differ.
* Forward pass only (`seq_len=128`, `batch_size=1`).
* Gated models are excluded.
* Custom/Triton kernels are opaque to `torch.export`, so counts are biased toward
  what the PyTorch frontend can observe.

## License

MIT — see `pyproject.toml` for package metadata.
