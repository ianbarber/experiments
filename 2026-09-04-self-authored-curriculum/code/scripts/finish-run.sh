#!/usr/bin/env bash
# After MAIN_RUN_DONE: the evaluations the orchestrator does not cover, then the analysis bundle.
# Idempotent - every step is skipped if its marker is already in this log.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
RUN=main; CFG=configs/main-9b.json; PY="venvs/loop/bin/python"; export PYTHONPATH=$PWD
R=${RSI_RUNS_DIR}/$RUN; FLOG=$R/logs/finish-run.log
step() { echo "##### [$(date '+%F %T')] $*"; }
done_already() { grep -q "DONE: $1" "$FLOG" 2>/dev/null; }
soft() { local tag="$1"; shift; done_already "$tag" && { step "skip $tag"; return; }; "$@" && step "DONE: $tag" || step "FAILED: $tag"; }
HEAD=$($PY -c "import json;print(json.load(open('$R/state.json')).get('lineage_head',0))")
step "finisher for run $RUN at M$HEAD"

# Pre-registered but outside run-main.sh: Terminal-Bench 2.0 as published at M_final (network on, LAN blocked).
soft tbpub-final $PY -m rsi.cli eval tbpub --run $RUN --config $CFG --policy M$HEAD --seeds 3 --tag final --iteration 7 --concurrency 12
# Memorisation probe (docs/02): only if it has been implemented by then.
[ -f rsi/probe.py ] && soft probe $PY -m rsi.cli probe --run $RUN --config $CFG
# The M_1 dev point was measured on the bf16 merge, before the fp16 switch; re-measure it so the dev curve is
# one precision throughout (docs/07 amendment 2026-09-08). Stored under its own tag, the original rows are kept.
soft dev-M1-fp16 $PY -m rsi.cli eval dev --run $RUN --config $CFG --policy M1 --seeds 3 --tag fp16 --iteration 1
soft analysis $PY scripts/analyse-run.py --run $RUN
soft report $PY scripts/fill-writeup.py
soft dashboard $PY -c "from rsi import config as C; from rsi.db import DB; from rsi.dashboard import build_dashboard; r=C.run_paths('$RUN'); print(build_dashboard(r, DB(r.db)))"
step "FINISH_DONE"
