# Code — LLM operator evolution

The pipeline as run in the original repo (`ianbarber/llm-op-evolution`, archived):
`config/models.yaml` → `torch.export` → `../results/metrics.json` → figures.

```
config/models.yaml              # Manifest: 22 models, release dates, mechanism lists
src/llm_op_evolution/
  export_model.py               # torch.export pipeline (Core ATen IR)
  op_utils.py                   # Graph traversal and operator categorisation
  mechanisms.py                 # Mechanism taxonomy + heterogeneity scoring
  plot.py                       # Figure generation
scripts/
  run_analysis.py               # End-to-end export + plot
  analyze_mechanisms.py         # Add the mechanism inventory to metrics.json
  print_summary.py              # Compact CLI summary table
```

Requires Python 3.11+ and a working PyTorch (ROCm / CUDA / CPU). Developed on Strix Halo
(gfx1151) with `torch==2.11.0+rocm7.13.0`.

```bash
uv pip install -e .

# Export all models and regenerate figures (results live one level up)
HF_HUB_ENABLE_HF_TRANSFER=0 python scripts/run_analysis.py --device cuda --output ../results

# One model
HF_HUB_ENABLE_HF_TRANSFER=0 python scripts/run_analysis.py --device cuda --output ../results --models qwen3.5-4b

# Mechanism inventory + figures from the existing metrics.json
python scripts/analyze_mechanisms.py --output ../results
python scripts/run_analysis.py --plot-only --output ../results
python scripts/print_summary.py ../results/metrics.json
```

The scripts default to a `results/` directory beside `src/` (the original layout); pass
`--output ../results` as above to use this entry's `results/`. What is measured, and why
it is not FLOPs, is in `../REPORT.md`.
