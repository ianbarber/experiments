#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

for experiment_seed in 20260826 20260827 20260828; do
  uv run python scripts/train.py \
    --train-manifest data/manifests/chesssight_train0.jsonl \
    --train-manifest data/manifests/chessred2k.jsonl \
    --validation-manifest data/manifests/position_disjoint_validation_balanced.jsonl \
    --run-dir "runs/E0008_perceptual_seed_${experiment_seed}" \
    --stage joint \
    --resume runs/M1_joint/checkpoint.pt \
    --source-ratio chesssight=0.5 \
    --source-ratio chessred=0.5 \
    --max-steps 150 \
    --batch-size 16 \
    --workers 8 \
    --log-every 50 \
    --occupied-square-weight 4 \
    --semantic-perceptual-checkpoint runs/B0/checkpoint.pt \
    --semantic-perceptual-weight 0.1 \
    --seed "$experiment_seed"
done
