#!/usr/bin/env bash
# Phase 3 orchestration (docs/03 §4, §8; docs/05): the 9B main run with the v0 control set.
# Resumable: every step is idempotent through the loop's state file and the DB. Run detached:
#   setsid nohup scripts/run-main.sh > ${RSI_RUNS_DIR}/main/logs/run-main.log 2>&1 < /dev/null &
set -o pipefail
cd "$(dirname "$0")/.."
RUN=main; CFG=configs/main-9b.json; PY="venvs/loop/bin/python"; export PYTHONPATH=$PWD
L=${RSI_RUNS_DIR}/$RUN/logs; mkdir -p $L
step() { echo "##### [$(date '+%F %T')] $*"; }
run() { "$@" || { step "FAILED: $*"; exit 1; }; }
soft() { "$@" || step "NON-FATAL FAILURE: $*"; }   # side experiments must never stall the lineage

step "waiting for the M_0 canaries and the Terminal-Bench as-published evaluation (they share the base-model server that UPDATE will stop)"
until grep -q "CHAIN20_DONE" smoke/results/chain20-canaries.log 2>/dev/null && grep -q "CHAIN18_DONE" smoke/results/chain18-tbpub-eval.log 2>/dev/null; do sleep 60; done
step "iteration 0"
run $PY -m rsi.cli loop run --run $RUN --config $CFG --until 0
step "iteration-0 ablation: RFT vs PG from the same rollouts (dev x3)"
soft $PY -m rsi.cli control rft-vs-pg --run $RUN --config $CFG --iteration 0

for k in 1 2 3; do step "iteration $k"; run $PY -m rsi.cli loop run --run $RUN --config $CFG --until $k; done

step "branch the non-recursive control from M_3 (frozen M_0 + H_0 write the tasks)"
$PY -m rsi.cli loop branch --run $RUN --config $CFG --to ${RUN}-control --iteration 3 || step "branch exists"

for k in 4 5 6; do
  step "iteration $k (main)"; run $PY -m rsi.cli loop run --run $RUN --config $CFG --until $k
  step "iteration $k (control branch)"; run $PY -m rsi.cli loop run --run ${RUN}-control --config $CFG --set non_recursive=true --set proposer_model_path=${RSI_MODELS_DIR}/Qwen3.5-9B --until $k
done

step "final same-session held-out evaluation: M_final and M_0, alternating per seed block"
HEAD=$($PY -c "import json;print(json.load(open('${RSI_RUNS_DIR}/$RUN/state.json')).get('lineage_head',0))")
for seed in 1 2 3 4 5; do   # 10 seeds per model (pre-declared MDE fallback), alternating in blocks of 2
  run $PY -m rsi.cli eval heldout --run $RUN --config $CFG --policy M${HEAD} --seeds 2 --tag final-s$seed --iteration 7
  run $PY -m rsi.cli eval heldout --run $RUN --config $CFG --policy M0 --seeds 2 --tag end-s$seed --iteration 7
done
step "control branch final held-out x5"
CHEAD=$($PY -c "import json;print(json.load(open('${RSI_RUNS_DIR}/${RUN}-control/state.json')).get('lineage_head',0))")
run $PY -m rsi.cli eval heldout --run ${RUN}-control --config $CFG --policy M${CHEAD} --seeds 10 --tag final --iteration 7
step "TerminalWorld x3 and canaries at M_final; pass@k subset (first 40 held-out tasks x16) at M_0 and M_final"
run $PY -m rsi.cli eval tw --run $RUN --config $CFG --policy M${HEAD} --seeds 3 --tag final --iteration 7
run $PY -m rsi.cli eval canaries --run $RUN --config $CFG --policy M${HEAD}
run $PY -m rsi.cli eval heldout --run $RUN --config $CFG --policy M${HEAD} --seeds 16 --limit 40 --tag passk --iteration 7
run $PY -m rsi.cli eval heldout --run $RUN --config $CFG --policy M0 --seeds 16 --limit 40 --tag passk --iteration 7
step "control set from the stored iteration-0 rollouts (after the primary endpoint, so a control cannot stall the lineage)"
soft $PY -m rsi.cli control rft-vs-pg --run $RUN --config $CFG --iteration 0
soft $PY -m rsi.cli control random-reward --run $RUN --config $CFG --iteration 0
step "MAIN_RUN_DONE"
