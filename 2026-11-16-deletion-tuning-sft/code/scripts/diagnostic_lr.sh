#!/usr/bin/env bash
# Diagnostic, one seed: forgetting against learning rate at one Wordle level.
# Rescores neighbouring learning rates that also reach the matching level, and retrains the two
# full fine-tuning runs whose checkpoints were not kept. Run from code/.
set -u
export PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
for lr in 3e-6 1e-5; do
  name=diag_fft_lr${lr}_s42
  [ -f runs/$name/done.json ] || $PY -m deltune.train --name $name --method fft --lr $lr --steps 1600 --stop-at 1200 --seed 42 --save-steps 1200
  $PY -m deltune.evaluate $name --steps 1200 --pools seen --suite | grep ^RESULT
done
$PY -m deltune.evaluate lora_r4_lr1e-4_s42 --steps 1600 --pools seen --suite | grep ^RESULT
$PY -m deltune.evaluate dt_lr5e-4_a100-1000_s42 --steps 400 --pools seen --suite | grep ^RESULT
$PY -m deltune.evaluate dt_lr3e-3_a100-1000_s42 --steps 400 --pools seen --suite | grep ^RESULT
# Control for "learning the mask drives the gain": the mask at step 0 (Wanda prior plus init noise, no training)
$PY -m deltune.evaluate dt_lr1e-3_a100-1000_s42 --steps 0 --pools seen,heldout_val | grep ^RESULT
# Raw HumanEval completions for reading: why do both adapters score higher than base?
$PY scripts/humaneval_read.py base 0 | grep ^RESULT
$PY scripts/humaneval_read.py dt_lr1e-3_a100-1000_s42 400 | grep ^RESULT
$PY scripts/humaneval_read.py lora_r4_lr3e-4_s42 1000 | grep ^RESULT
