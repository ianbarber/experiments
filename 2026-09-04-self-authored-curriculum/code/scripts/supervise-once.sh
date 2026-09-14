#!/usr/bin/env bash
# One watchdog check for the Phase 3 orchestrator, safe to call from cron every few minutes.
# Restarts scripts/run-main.sh only when neither it nor a loop child is running, under a lock so two
# checks can never start two orchestrators on the same run.
#   Kill switch: touch ${RSI_RUNS_DIR}/main/STOP
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
RUN=${RSI_RUNS_DIR}/main
LOG=$RUN/logs/run-main.log; WLOG=$RUN/logs/supervisor.log; CNT=$RUN/logs/.restarts; MAX=48
say() { echo "##### [$(date '+%F %T')] supervisor: $*" >> "$WLOG"; }

exec 9>"$RUN/.supervisor.lock" || exit 1
flock -n 9 || exit 0                                  # another check is already running

[ -e "$RUN/STOP" ] && exit 0                          # user kill switch
if grep -q "MAIN_RUN_DONE" "$LOG" 2>/dev/null; then      # the run is over: hand off to the finisher, once
  FLOG=$RUN/logs/finish-run.log
  grep -q "FINISH_DONE" "$FLOG" 2>/dev/null && exit 0
  pgrep -f "bash scripts/finish-ru[n].sh" > /dev/null && exit 0
  say "MAIN_RUN_DONE seen; starting the finisher"
  setsid nohup scripts/finish-run.sh >> "$FLOG" 2>&1 < /dev/null 9>&- &
  exit 0
fi
pgrep -f "bash scripts/run-mai[n].sh" > /dev/null && exit 0
pgrep -f "rsi.cli loop run --run mai[n]" > /dev/null && exit 0   # a child is still working; leave it alone
pgrep -f "rsi.cli (eval|control) .*--run mai[n]" > /dev/null && exit 0

n=$(cat "$CNT" 2>/dev/null || echo 0)
if [ "$n" -ge "$MAX" ]; then say "restart budget exhausted ($MAX); not restarting"; exit 1; fi
echo $((n + 1)) > "$CNT"
say "orchestrator down; restart $((n + 1))/$MAX"
# cron starts with a bare environment; the run needs the same caches the interactive shell has
export HF_HOME=${HOME}/.cache/huggingface HF_HUB_CACHE=/mnt/nas/hf-cache/hub \
       HF_DATASETS_CACHE=/mnt/nas/hf-cache/datasets HF_ASSETS_CACHE=${HOME}/.cache/huggingface/assets \
       HF_XET_CACHE=${HOME}/.cache/huggingface/xet VLLM_CACHE_ROOT=${HOME}/.cache/vllm-rsi \
       PATH=${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin
tail -n 2 "$LOG" 2>/dev/null | sed 's/^/    last: /' >> "$WLOG"
setsid nohup scripts/run-main.sh >> "$LOG" 2>&1 < /dev/null 9>&- &   # 9>&- or the child would hold the lock for days
