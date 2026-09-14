#!/usr/bin/env bash
# Watchdog for the Phase 3 orchestrator: if scripts/run-main.sh dies before MAIN_RUN_DONE, relaunch it (every step is
# idempotent through the loop state file and the DB). Bounded restarts so a hard failure cannot spin.
#   setsid nohup scripts/supervise-main.sh > ${RSI_RUNS_DIR}/main/logs/supervisor.log 2>&1 < /dev/null &
set -o pipefail
cd "$(dirname "$0")/.."
LOG=${RSI_RUNS_DIR}/main/logs/run-main.log
MAX=${MAX_RESTARTS:-24}; INTERVAL=${INTERVAL:-300}; n=0
say() { echo "##### [$(date '+%F %T')] supervisor: $*"; }
say "watching (interval ${INTERVAL}s, max ${MAX} restarts)"
while true; do
  sleep "$INTERVAL"
  if grep -q "MAIN_RUN_DONE" "$LOG" 2>/dev/null; then say "MAIN_RUN_DONE seen; exiting"; exit 0; fi
  if pgrep -f "bash scripts/run-mai[n].sh" > /dev/null; then continue; fi
  if [ "$n" -ge "$MAX" ]; then say "restart budget exhausted ($MAX); leaving it down"; exit 1; fi
  n=$((n + 1)); say "orchestrator is not running; restart $n/$MAX"
  tail -n 3 "$LOG" 2>/dev/null | sed 's/^/    last: /'
  setsid nohup scripts/run-main.sh >> "$LOG" 2>&1 < /dev/null &
  sleep 30
done
