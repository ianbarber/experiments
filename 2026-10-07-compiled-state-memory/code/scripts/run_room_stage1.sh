#!/usr/bin/env bash
# Room check stage 1 (LABNOTES 2026-10-13, night), on the long-context SGLang profile:
#   smoke  2 calibration questions, every arm at small (+ A/T/D at large), to its own file
#   full   N, O; A, T, D at small/mid/large; M at small; the flip reruns (T and M at small, 40
#          questions, reverse order); then the Qwen judge on everything
# Needs the digests (run_room_digest.sh). Nothing else may hold the GPU.
set -e
MODE=${1:?usage: run_room_stage1.sh smoke|full}
cd "$(dirname "$0")/.."
P=../results/phase2/room
[ -s $P/prep/digests.jsonl ] || { echo "no digests"; exit 1; }
[ "$(docker ps -q | wc -l)" -eq 0 ] || { echo "a container is running"; exit 1; }
./serve_sglang.sh start-long
trap './serve_sglang.sh stop' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:30000/health >/dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:30000/health >/dev/null
RUN="uv run --with transformers==5.12.1 --with jinja2 python experiments/room_agent.py --prep $P/prep"
if [ "$MODE" = smoke ]; then
  Q=$(python3 -c "import json; print(','.join(json.load(open('$P/prep/split.json'))['cal'][:2]))")
  $RUN --qids $Q --arms N,O,A,T,D,M --sizes small --out $P/smoke.jsonl --workers 8
  $RUN --qids $Q --arms A,T,D --sizes large --out $P/smoke.jsonl --workers 8
  uv run --with transformers==5.12.1 --with jinja2 python experiments/room_judge.py --traj $P/smoke.jsonl --prep $P/prep --out $P/smoke_judge.jsonl
  echo "smoke done"
  exit 0
fi
$RUN --arms N,O --out $P/traj.jsonl --workers 16
$RUN --arms A,T,D --sizes small --out $P/traj.jsonl --workers 16
$RUN --arms M --sizes small --out $P/traj.jsonl --workers 8
$RUN --arms A,T,D --sizes mid --out $P/traj.jsonl --workers 16
$RUN --arms A,T,D --sizes large --out $P/traj.jsonl --workers 12
FLIP=$(python3 -c "import json; print(','.join(json.load(open('$P/prep/split.json'))['cal'][:40][::-1]))")
$RUN --qids $FLIP --arms T,M --sizes small --out $P/traj.jsonl --workers 8 --tag flip
uv run --with transformers==5.12.1 --with jinja2 python experiments/room_judge.py --traj $P/traj.jsonl --prep $P/prep --out $P/judge.jsonl
echo "stage 1 done"
