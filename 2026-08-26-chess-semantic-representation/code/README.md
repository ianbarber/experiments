# Reproducing the chess-representation experiment

Run commands from this `code/` directory. Raw data, extracted images,
manifests, checkpoints, and run directories are intentionally omitted.

## Environment

Target: Python 3.12, ROCm 7.2, PyTorch 2.9.1 on `gfx1151`.

```bash
uv venv --python 3.12 .venv
uv sync --extra data --extra dev --active
bash scripts/install_rocm_torch.sh
uv run python scripts/smoke_rocm.py
uv run pytest
```

`uv sync` manages portable dependencies; the second script installs AMD's
machine-specific PyTorch/Torchvision/Triton wheels afterward.

## Data

```bash
bash scripts/acquire_dev_data.sh
```

This downloads and verifies roughly 5 GiB, expands to roughly 10 GiB, builds
the source manifests, audits them, and derives full and balanced
position-disjoint validation manifests. The data is non-commercial; see the
experiment README for licenses.

## Staged training

The selected model was produced by the following stages (single exploratory
seed until the final three replicas):

```bash
# Semantic encoder.
uv run python scripts/train.py \
  --train-manifest data/manifests/chesssight_train0.jsonl \
  --train-manifest data/manifests/chessred2k.jsonl \
  --validation-manifest data/manifests/chessred2k.jsonl \
  --source-ratio chesssight=0.5 --source-ratio chessred=0.5 \
  --stage semantic --max-steps 200 --batch-size 32 --workers 8 \
  --run-dir runs/B0

# Ground-truth-semantic decoder warm-up, then occupied-square weighting.
uv run python scripts/train.py \
  --train-manifest data/manifests/chesssight_train0.jsonl \
  --train-manifest data/manifests/chessred2k.jsonl \
  --source-ratio chesssight=0.5 --source-ratio chessred=0.5 \
  --stage decoder --resume runs/B0/checkpoint.pt \
  --max-steps 50 --batch-size 16 --workers 8 --run-dir runs/M1_decoder

uv run python scripts/train.py \
  --train-manifest data/manifests/chesssight_train0.jsonl \
  --train-manifest data/manifests/chessred2k.jsonl \
  --source-ratio chesssight=0.5 --source-ratio chessred=0.5 \
  --stage decoder --resume runs/M1_decoder/checkpoint.pt \
  --occupied-square-weight 4 --max-steps 100 --batch-size 16 --workers 8 \
  --run-dir runs/M1_piece

# Return to predicted hard semantics.
uv run python scripts/train.py \
  --train-manifest data/manifests/chesssight_train0.jsonl \
  --train-manifest data/manifests/chessred2k.jsonl \
  --source-ratio chesssight=0.5 --source-ratio chessred=0.5 \
  --stage joint --resume runs/M1_piece/checkpoint.pt \
  --occupied-square-weight 4 --max-steps 50 --batch-size 16 --workers 8 \
  --run-dir runs/M1_joint
```

`scripts/run_perceptual_replicas.sh` then runs the three class-aware replicas
from `runs/M1_joint/checkpoint.pt`. It uses `runs/B0/checkpoint.pt` as the frozen
semantic recognizer.

## Evaluation

The relevant entry points are:

```text
scripts/visualize_representations.py  reconstruction/base panels
scripts/semantic_intervention.py      fixed-appearance semantic swap
scripts/probe_latents.py              linear leakage/source probes
scripts/evaluate_codec.py             CRP1 bytes and quantized distortion
scripts/evaluate_standard_codecs.py   JPEG/WebP/PNG references
```

Small captured reports are in `../results/`; trained weights are not included.
