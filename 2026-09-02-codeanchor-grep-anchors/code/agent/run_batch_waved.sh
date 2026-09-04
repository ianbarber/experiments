#!/bin/bash
# Waved batch runner: bounded disk via load->rollout->eval->evict cycles.
# Usage: run_batch_waved.sh <arm_yaml> <run_name> <filter_regex> <mode: base|lsp|anchor> [wave_size] [workers] [eval_workers]
set -uo pipefail
ARM="$1"; NAME="$2"; FILTER="$3"; MODE="$4"; WAVE="${5:-6}"; WORKERS="${6:-3}"; EVAL_WORKERS="${7:-2}"
OUT="$HOME/refactorbench-eval/runs/$NAME"
HARNESS="$HOME/refactorbench-eval/harness"
NAS_BASE=/mnt/nas/refactorbench/images
NAS_LSP=/mnt/nas/refactorbench/images-lsp
export PATH="$HOME/.local/bin:$PATH"
export MSWEA_COST_TRACKING=ignore_errors
# arm E (anchor_env.AnchorDockerEnvironment) needs the anchor module importable and a
# telemetry sink; harmless for other arms.
export PYTHONPATH="$HOME/refactorbench-eval/anchor${PYTHONPATH:+:$PYTHONPATH}"
export ANCHOR_LOG="$OUT/anchor_log.jsonl"
mkdir -p "$OUT"

MINI_EXTRA="mini-extra"
if [[ "$MODE" == "lsp" || "$MODE" == "anchor" ]]; then
  SUBSET="$HOME/refactorbench-eval/promax-lsp-dataset"; ROLLOUT_PREFIX="promax-lsp:"
else
  SUBSET="swe-bench-promax/SWE-Bench-ProMax"; ROLLOUT_PREFIX="key4127/refactor-dockerhub:"
fi
if [[ "$MODE" == "anchor" ]]; then
  # arm E: same images/dataset as lsp mode, but the "docker" environment is rebound to
  # anchor_env.AnchorDockerEnvironment via the launcher shim (see anchor_runner.py).
  MINI_EXTRA="$HOME/.local/share/uv/tools/mini-swe-agent/bin/python $HOME/refactorbench-eval/anchor/anchor_runner.py"
fi

IDS=$(python3 - "$FILTER" <<'EOF'
import json, re, sys, os
pat = re.compile(sys.argv[1])
data = json.load(open(os.path.expanduser("~/refactorbench-eval/harness/data/swe-bench-promax.json")))
print("\n".join(x["instance_id"] for x in data if pat.match(x["instance_id"])))
EOF
)
TOTAL=$(wc -l <<<"$IDS")
export IDS_LIST="$IDS"
echo "[waved] $TOTAL instances, wave size $WAVE, mode $MODE"

load_base() {  # load original image (NAS first, hub fallback + NAS seed)
  local iid="$1" img="key4127/refactor-dockerhub:$1"
  docker image inspect "$img" >/dev/null 2>&1 && return 0
  if [[ -f "$NAS_BASE/${iid}.tar.zst" ]]; then zstd -dc "$NAS_BASE/${iid}.tar.zst" | docker load >/dev/null
  else docker pull -q "$img" >/dev/null && { docker save "$img" | zstd -T0 -o "$NAS_BASE/${iid}.tar.zst.tmp" && mv "$NAS_BASE/${iid}.tar.zst.tmp" "$NAS_BASE/${iid}.tar.zst"; }
  fi
}
load_lsp() {
  local iid="$1"
  docker image inspect "promax-lsp:${iid}" >/dev/null 2>&1 && return 0
  [[ -f "$NAS_LSP/${iid}.tar.zst" ]] || { echo "[waved] MISSING LSP image: $iid"; return 1; }
  zstd -dc "$NAS_LSP/${iid}.tar.zst" | docker load >/dev/null
}
evict() { for iid in $1; do docker rmi "${ROLLOUT_PREFIX}${iid}" "key4127/refactor-dockerhub:${iid}" >/dev/null 2>&1; done; docker image prune -f >/dev/null 2>&1; }

W=0
while IFS= read -r -d '' WAVE_IDS; do
  W=$((W+1))
  echo "[waved] === wave $W: $(tr '\n' ' ' <<<"$WAVE_IDS")"
  # 1. load rollout images; only successfully-loaded instances enter the rollout filter
  OK_IDS=""
  for iid in $WAVE_IDS; do
    if [[ "$MODE" == "lsp" || "$MODE" == "anchor" ]]; then load_lsp "$iid" && OK_IDS+="$iid"$'\n'; else load_base "$iid" && OK_IDS+="$iid"$'\n'; fi
  done
  [[ -z "$OK_IDS" ]] && { echo "[waved] wave $W: nothing loaded, skipping"; continue; }
  WAVE_RE="($(printf '%s' "$OK_IDS" | paste -sd'|' -))\$"
  # 2. rollout (resume-safe: existing preds are skipped by the runner)
  $MINI_EXTRA swebench --subset "$SUBSET" --split test --filter "$WAVE_RE" \
    -c swebench.yaml -c "$ARM" -o "$OUT" -w "$WORKERS" </dev/null
  # 3. lsp mode: drop rollout images before loading eval bases
  if [[ "$MODE" == "lsp" || "$MODE" == "anchor" ]]; then
    for iid in $WAVE_IDS; do docker rmi "promax-lsp:${iid}" >/dev/null 2>&1; done
    for iid in $WAVE_IDS; do load_base "$iid"; done
  fi
  # 4. eval just this wave
  python3 - "$OUT" "$WAVE_RE" "$W" <<'EOF'
import json, re, sys, os
out, wave_re, w = sys.argv[1], sys.argv[2], sys.argv[3]
pat = re.compile(wave_re)
preds = json.load(open(f"{out}/preds.json"))
sub = {k: v for k, v in preds.items() if pat.match(k)}
json.dump(sub, open(f"{out}/preds_wave{w}.json", "w"))
print(f"[waved] eval wave {w}: {len(sub)} preds")
EOF
  python3 "$HARNESS/src/evaluation/test_run.py" --pred "$OUT/preds_wave$W.json" \
    --golden "$HARNESS/data/swe-bench-promax.json" --eval "$HARNESS/data/eval.json" \
    --output "$OUT/pass_rate_wave$W.json" -w "$EVAL_WORKERS" </dev/null | tail -25
  # 5. evict everything from this wave
  evict "$WAVE_IDS"
done < <(python3 - "$WAVE" <<'EOF'
import sys, os, json, re
wave = int(sys.argv[1])
ids = [l for l in os.environ.get("IDS_LIST", "").split("\n") if l]
for i in range(0, len(ids), wave):
    sys.stdout.write("\n".join(ids[i:i+wave]) + "\0")
EOF
)
# merge waves
python3 - "$OUT" <<'EOF'
import json, glob, sys, collections
out = sys.argv[1]
allr = []
for f in sorted(glob.glob(f"{out}/pass_rate_wave*.json")):
    allr.extend(json.load(open(f)))
json.dump(allr, open(f"{out}/pass_rate.json", "w"), indent=2)
gold_ok = [r for r in allr if r["golden"]["final_result"] == "success"]
passed = sum(1 for r in gold_ok if r["passed"])
print(f"[waved] MERGED: {passed}/{len(gold_ok)} resolved ({len(allr)} evaluated, {len(allr)-len(gold_ok)} golden-invalid)")
by = collections.defaultdict(lambda: [0, 0])
for r in gold_ok:
    by[r["language"]][1] += 1
    by[r["language"]][0] += r["passed"]
for l, (p, t) in sorted(by.items()):
    print(f"  {l}: {p}/{t}")
EOF
mkdir -p /mnt/nas/refactorbench/runs
rsync -a "$OUT" /mnt/nas/refactorbench/runs/
echo "[waved] done: $NAME"
