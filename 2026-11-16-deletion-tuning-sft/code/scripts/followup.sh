#!/usr/bin/env bash
# The follow-up registered as amendment 11 (PROTOCOL.md section 7). Run from code/, one stage
# or several:   bash scripts/followup.sh base seed43 seed44 seed42 paper
#
#   base    re-score the base model's suite in this session (into runs/session2/base)
#   seedNN  every method at the paper's SFT learning rate, 1,600 steps, suite at step 1,600
#   paper   Deletion Tuning at the paper's exact SFT recipe: 400 steps, ramp over those 400
#
# One GPU job at a time. Finished runs are skipped, so a stage can be restarted.
set -u
export PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

train() {  # name, then training arguments
  local name=$1; shift
  mkdir -p runs/$name
  if [ -f runs/$name/done.json ]; then echo "[skip train] $name"; return 0; fi
  echo "[train] $name $*"
  $PY -m deltune.train --name $name "$@" >> runs/$name/train.log 2>&1 || { echo "[FAILED train] $name"; return 1; }
}

score() {  # name, then evaluation arguments
  local name=$1; shift
  $PY -m deltune.evaluate $name "$@" 2>&1 | tee -a runs/$name/evaluate.log | grep --line-buffered ^RESULT || echo "[FAILED score] $name $*"
}

conservative() {  # method, run name, seed, then method arguments
  local name=$2 seed=$3; shift 3
  train $name --steps 1600 --save-steps 400,800,1200,1600 --seed $seed "$@" || return
  score $name --steps 400,800,1200 --pools seen
  score $name --steps 1600 --pools seen --suite
}

for stage in "$@"; do
  echo "== stage $stage  $(date +%H:%M)"
  case $stage in
    base)
      mkdir -p runs/session2/base
      DELTUNE_RUNS=$PWD/runs/session2 $PY -m deltune.evaluate base --suite 2>&1 | tee -a runs/session2/base/evaluate.log | grep --line-buffered ^RESULT
      ;;
    seed43|seed44)
      s=${stage#seed}
      conservative dt dt_lr5e-4_a100-1000_s$s $s --method dt --lr 5e-4 --alpha 100,1000
      conservative lora lora_r4_lr1e-4_s$s $s --method lora --lr 1e-4 --rank 4
      conservative fft fft_lr3e-6_s$s $s --method fft --lr 3e-6
      ;;
    seed42)
      # LoRA is the pilot run (suite at step 1,600 already scored); Deletion Tuning is the tuning
      # run; full fine-tuning is retrained because its step-1,600 checkpoint was not kept.
      score dt_lr5e-4_a100-1000_s42 --steps 1600 --pools seen --suite
      conservative fft rerun_fft_lr3e-6_s42 42 --method fft --lr 3e-6
      ;;
    paper)
      for s in 42 43 44; do
        name=paper_dt_lr5e-4_s$s
        train $name --steps 400 --save-steps 200,400 --seed $s --method dt --lr 5e-4 --alpha 100,1000 || continue
        score $name --steps 400 --pools seen,heldout_test --suite
      done
      ;;
    *) echo "unknown stage $stage" ;;
  esac
done
echo "== done $(date +%H:%M)"
