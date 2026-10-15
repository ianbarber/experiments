#!/usr/bin/env bash
# Run a command inside the statelm research image with the repo, NAS HF cache and local HF cache mounted.
#   ./run.sh python tests/check_injection.py --model $Q35_4B
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec docker run --rm --gpus all --ipc=host --shm-size 16g \
  -v "$HERE":/work -v /mnt/nas/hf-cache:/nas-hf:ro -v "$HOME/.cache/huggingface":/hf -v "${STATELM_WEIGHTS:?set STATELM_WEIGHTS to the directory holding Qwen3.8-27B}":/weights:ro \
  -v "$HOME/.cache/statelm-triton":/root/.triton -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/work/code -w /work/code \
  ${DOCKER_EXTRA:-} statelm:fla052 "$@"
