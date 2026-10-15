#!/usr/bin/env bash
# Room check stage 2 (LABNOTES 2026-10-13, night): M (whole memory in context, plus tools) at mid and
# large, on the long-context SGLang profile, then the Qwen judge on the new answers. The skip rule did
# not fire at either size (stage 1, 2026-10-14). Stop rule (logged before the run): after the first 10
# M@large questions, if the projected stage-2 total exceeds 24 h, stop and re-plan.
# Nothing else may hold the GPU.
set -e
cd "$(dirname "$0")/.."
P=../results/phase2/room
[ "$(docker ps -q | wc -l)" -eq 0 ] || { echo "a container is running"; exit 1; }
T0=$(date +%s)
./serve_sglang.sh start-long
trap './serve_sglang.sh stop' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:30000/health >/dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:30000/health >/dev/null
RUN="uv run --with transformers==5.12.1 --with jinja2 python experiments/room_agent.py --prep $P/prep --out $P/traj.jsonl --arms M"
$RUN --sizes mid --workers 2
FIRST=$(python3 -c "import json; print(','.join(json.load(open('$P/prep/split.json'))['cal'][:10]))")
$RUN --sizes large --workers 1 --qids $FIRST
python3 - "$T0" <<'EOF'
import json, sys, time
recs = [json.loads(ln) for ln in open("../results/phase2/room/traj.jsonl")]
large = [r["s"] for r in recs if r["arm"] == "M" and r["size"] == "large" and not r.get("tag")]
elapsed = time.time() - float(sys.argv[1])
proj = elapsed + (120 - len(large)) * sum(large) / len(large)
print(f"stop rule: {len(large)} M@large done, mean {sum(large) / len(large):.0f} s; elapsed {elapsed / 3600:.1f} h; "
      f"projected stage-2 total {proj / 3600:.1f} h (limit 24 h)", flush=True)
sys.exit(3 if proj > 24 * 3600 else 0)
EOF
$RUN --sizes large --workers 1
uv run --with transformers==5.12.1 --with jinja2 python experiments/room_judge.py --traj $P/traj.jsonl --prep $P/prep --out $P/judge.jsonl
echo "stage 2 done"
