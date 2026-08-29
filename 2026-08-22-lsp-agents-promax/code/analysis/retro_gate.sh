#!/bin/bash
# Retroactive gate false-block analysis (v2: instance-major ordering, per-instance image
# eviction, resume from existing valid JSONL rows).
set -uo pipefail
OUT=~/refactorbench-eval/runs/retro_gate.jsonl
NAS_LSP=/mnt/nas/refactorbench/images-lsp
GATE=/home/ianbarber/refactorbench-eval/agent/gate/typecheck_gate.py
touch "$OUT"
# drop a truncated trailing line if present
python3 - <<'EOF'
import json, os
p = os.path.expanduser("~/refactorbench-eval/runs/retro_gate.jsonl")
lines = open(p).read().splitlines()
good = []
for l in lines:
    try: json.loads(l); good.append(l)
    except Exception: pass
open(p, "w").write("\n".join(good) + ("\n" if good else ""))
print(f"[retro] kept {len(good)} valid rows")
EOF

python3 - <<'EOF' > /tmp/retro_jobs.txt
import json, os
done = set()
p = os.path.expanduser("~/refactorbench-eval/runs/retro_gate.jsonl")
for l in open(p):
    try:
        r = json.loads(l); done.add((r["run"], r["iid"]))
    except Exception: pass
jobs = []
for run in ["s1-arm-a-r1","s1-arm-a-r2","s1h-arm-a-r1","s1h-arm-a-r2"]:
    pr = {x["instance_id"]: x for x in json.load(open(f"/mnt/nas/refactorbench/runs/{run}/pass_rate.json"))}
    preds = json.load(open(f"/mnt/nas/refactorbench/runs/{run}/preds.json"))
    for iid, p2 in preds.items():
        if pr.get(iid, {}).get("passed") and (p2.get("model_patch") or "").strip() and (run, iid) not in done:
            fn = f"/tmp/retro_patch_{run}_{iid}.diff"
            open(fn, "w").write(p2["model_patch"])
            jobs.append((iid, run, fn))
jobs.sort()  # instance-major: each image loads once
for iid, run, fn in jobs:
    print(f"{run}\t{iid}\t{fn}")
EOF
TOTAL=$(wc -l < /tmp/retro_jobs.txt)
echo "[retro] $TOTAL remaining patches to check"

LAST_IID=""
N=0
while IFS=$'\t' read -r RUN IID PATCH; do
  N=$((N+1))
  if [[ -n "$LAST_IID" && "$IID" != "$LAST_IID" ]]; then
    docker rmi "promax-lsp:${LAST_IID}" >/dev/null 2>&1
    docker image prune -f >/dev/null 2>&1
  fi
  LAST_IID="$IID"
  IMG="promax-lsp:${IID}"
  if ! docker image inspect "$IMG" >/dev/null 2>&1; then
    [[ -f "$NAS_LSP/${IID}.tar.zst" ]] && zstd -dc "$NAS_LSP/${IID}.tar.zst" | docker load >/dev/null 2>&1
  fi
  C="retro_$$_${N}"
  docker run -d --name "$C" "$IMG" sleep 1200 >/dev/null 2>&1 || { echo "{\"run\":\"$RUN\",\"iid\":\"$IID\",\"error\":\"image\"}" >> "$OUT"; continue; }
  docker cp "$GATE" "$C":/usr/local/bin/typecheck_gate.py >/dev/null 2>&1
  docker exec "$C" bash -c "cd /testbed && /opt/lsp-tool/bin/python /usr/local/bin/typecheck_gate.py --record-baseline" >/dev/null 2>&1
  docker cp "$PATCH" "$C":/tmp/p.diff >/dev/null 2>&1
  APPLY=$(docker exec "$C" bash -c "cd /testbed && (git apply /tmp/p.diff 2>&1 || patch -p1 < /tmp/p.diff 2>&1) >/dev/null 2>&1; echo \$?")
  GOUT=$(docker exec "$C" bash -c "cd /testbed && /opt/lsp-tool/bin/python /usr/local/bin/typecheck_gate.py" 2>/dev/null)
  RC=$?
  NEWERR=$(grep -oE "reports [0-9]+ new" <<<"$GOUT" | grep -oE "[0-9]+" || echo 0)
  echo "{\"run\":\"$RUN\",\"iid\":\"$IID\",\"apply_rc\":$APPLY,\"would_reject\":$([ $RC -ne 0 ] && echo true || echo false),\"new_errors\":$NEWERR}" >> "$OUT"
  docker rm -f "$C" >/dev/null 2>&1
  echo "[retro] $N/$TOTAL $RUN $IID reject=$([ $RC -ne 0 ] && echo Y || echo n) new=$NEWERR"
done < /tmp/retro_jobs.txt
docker rmi "promax-lsp:${LAST_IID}" >/dev/null 2>&1
echo "[retro] complete"
