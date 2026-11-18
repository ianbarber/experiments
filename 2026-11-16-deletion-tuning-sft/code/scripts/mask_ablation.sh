#!/usr/bin/env bash
# Parts of the paper-rates deletion masks scored on their own, seen-list Wordle only.
# The masks are written by `python scripts/mask_anatomy.py --ablations`. Run from code/.
set -u
export PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
for part in only near far; do
  for s in 42 43 44; do
    $PY -m deltune.evaluate ablate_learned_${part}_dt_lr5e-4_a100-1000_s$s --steps 1600 --pools seen | grep ^RESULT
  done
done
$PY scripts/mask_anatomy.py ablation
