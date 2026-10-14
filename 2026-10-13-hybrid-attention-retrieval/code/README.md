# Reproducing the hybrid recall experiment

Run the commands in this directory. The tested machine is `local-3090`: an RTX 3090 with 24 GiB VRAM. The scientific runtime dependencies are pinned in `pyproject.toml`; PyTorch uses the CUDA 13.0 wheel index. The model is BF16, with no training or quantization. The optional causal-convolution extension is unnecessary; DeltaNet must use FLA.

```bash
uv sync
uv run pytest -q
```

The model checkpoint is `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The first model load fetches the public weights if they are absent from the Hugging Face cache. The minimal MIT-licensed FactWorld data, oracle, and renderer snapshot is in `vendor/factworld/`, with attribution and the original source revision. No sibling checkout is required.

## Fixed fresh-seed expansion

Read [PROTOCOL.md](PROTOCOL.md) before execution. For each data seed:

The public CLI defaults to the fixed expansion at seed 42. Explicit options override that preset; the frozen pilot source retains its historical defaults.

```bash
uv run fwlayer --seed 42 --pairs 20 --causal-pairs 20 \
  --query-patch-mode cached --candidate-layers 19 23 24 26 27 28 30 31 \
  --min-answer-accuracy 0.9 --min-pair-accuracy 0.8 \
  --output results/expansion-seed42

uv run fwlayer --seed 314159 --pairs 20 --causal-pairs 20 \
  --query-patch-mode cached --candidate-layers 19 23 24 26 27 28 30 31 \
  --min-answer-accuracy 0.9 --min-pair-accuracy 0.8 \
  --output results/expansion-seed314159
```

Candidate indices in commands are zero-based; prose and plots use one-based layers. Each run contains 20 underlying examples at each of four distances. Identities stay fixed across distances, so statistics resample `(data seed, example)` clusters. Full-answer accuracy is unconditioned; logit margins supply only the shared answer prefix and compare the first differing token.

Outputs are written incrementally. Run metadata records seeds, model revision, software versions, source hashes, exact positions, and intervention settings. Repeating a command resumes an identical protocol. Selected-position tensor traces are needed to resume the original unsplit query-patch path; cached query patches capture their donor values from matching cached executions.

## Pilot and execution-path comparison

The frozen exploratory source is in `frozen_pilot/fwlayer/`. It preserves the scientific source files identified by the pilot's implementation hashes. Its dependencies use the same data snapshot.

```bash
PYTHONPATH=frozen_pilot uv run python -m fwlayer.run \
  --seed 20261014 --pairs 20 --causal-pairs 3 --output results/pilot-20261014

uv run python -m fwlayer.validate_cache --pilot results/pilot-20261014 \
  --output results/cache-validation.json
```

The comparison reruns the pilot's first example at 128 and 8,192 tokens through both paths in the same model session. Cached execution must stay within 0.03 of the unsplit path's transfer fraction, with matching self-patches. The published comparison passed at 0.014563.

## Published evidence

The entry's `results/` contains compact numerical evidence and run metadata. The entry's `images/` contains the figures used by the report. Reproduction runs create their detailed outputs under this directory's ignored `results/`: prompt token arrays, tensor traces, per-trial records, and generated reports. Model weights, raw traces, and runtime logs are local execution artifacts.

To rebuild a run's local summary and figures without loading model weights:

```bash
uv run fwlayer --stage report --output results/expansion-seed42
```

The protocol's stop conditions are executable accuracy and no-op gates. Interval computation and the locked primary/secondary tests are specified separately in the published analysis script.

To export a completed run and rebuild the publication tables and figures:

```bash
uv run python scripts/publication.py export --run results/expansion-seed42 \
  --destination ../results/seed42
uv run python scripts/publication.py export --run results/expansion-seed314159 \
  --destination ../results/seed314159
uv run python scripts/publication.py analyze --evidence ../results --images ../images
```
