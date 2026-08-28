#!/bin/bash
# Smoke test: one Python instance end-to-end with Arm A config (run on a worker host).
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export MSWEA_COST_TRACKING=ignore_errors
INSTANCE="${1:-albumentations-team__albumentations-2337}"
OUT="${2:-$HOME/refactorbench-eval/runs/smoke-arm-a}"
mini-extra swebench \
  --subset swe-bench-promax/SWE-Bench-ProMax --split test \
  --filter "$INSTANCE" \
  -c swebench.yaml -c "$HOME/refactorbench-eval/agent/arm_a.yaml" \
  -o "$OUT" -w 1 --redo-existing
echo "--- preds ---"
python3 -c "import json;d=json.load(open('$OUT/preds.json'));print({k: len(v['model_patch'] or '') for k,v in d.items()})"
