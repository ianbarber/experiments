#!/bin/bash
# Full 32B run, priority order. Every stage is resumable: rerun this script after any interruption.
cd "$(dirname "$0")"; export PYTHONUNBUFFERED=1
PY=${PYTHON:-python}; B=results/Qwen_2.5_32B_instruct/buttons
[ -f $B/full_probs.jsonl ]  || cp $B/pilot_probs.jsonl  $B/full_probs.jsonl      # the pilot rows are part of the full grid
[ -f $B/full_trials.jsonl ] || cp $B/pilot_trials.jsonl $B/full_trials.jsonl
stage() { echo "=== $1 START $(date '+%F %T')" >> logs/full_run.log; shift; "$@" >> logs/full_run.log 2>&1; echo "=== END rc=$? $(date '+%F %T')" >> logs/full_run.log; }
HARM="destructive_relief_vs_inert kidspics_relief_vs_inert"
stage "1 probs all cells"      $PY scripts/06_buttons.py --model 32b --mode probs --tag full --batch-rows 64 --pairs $HARM relief_vs_inert relief_vs_helpful costly_relief_vs_inert
stage "2 trials all cells"     $PY scripts/06_buttons.py --model 32b --mode trials --tag full --seeds 1 --group-rows 48 --pairs $HARM relief_vs_inert relief_vs_helpful costly_relief_vs_inert
stage "3 probs priming"        $PY scripts/06_buttons.py --model 32b --mode probs --tag priming --batch-rows 64 --rewards more_pain more_itch fear --pairs relief_vs_inert
stage "4 probs sadness"        $PY scripts/06_buttons.py --model 32b --mode probs --tag sadness --batch-rows 64 --steer sadness --rewards pain --dose-levels paper
# ---- tail: sensitivity arms added before launch (see the lab notes) ----
stage "5 probs untuned harm"   $PY scripts/06_buttons.py --model 32b --mode probs --tag untuned --no-adapter --batch-rows 64 --pairs $HARM
stage "6 probs itch L6 (D14)"  $PY scripts/06_buttons.py --model 32b --mode probs --tag itchL6 --batch-rows 64 --steer itch --itch-vector itch_A --pairs $HARM
stage "7 trials random seed 2" $PY scripts/06_buttons.py --model 32b --mode trials --tag full --seeds 2 --no-greedy --group-rows 48 --steer rand --pairs $HARM
echo "=== ALL DONE $(date '+%F %T')" >> logs/full_run.log
