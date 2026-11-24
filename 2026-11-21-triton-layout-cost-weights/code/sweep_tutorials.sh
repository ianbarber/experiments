#!/usr/bin/env bash
# Compile every tutorial sweep for every target, one process per (tutorial, target).
#   ./sweep_tutorials.sh <out_root> [extra run_census.py flags]
# Needs TRITON_ROOT. Set JOBS to change parallelism.
set -u
OUT=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/logs"
jobs=()
for arch in 80 90 100 120; do
  for w in tut02 tut03 tut05 tut06 tut08 tut09 tut10; do
    jobs+=("$w $arch")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P "${JOBS:-10}" -L 1 bash -c '
  w=$0; arch=$1; out='"$OUT"'/${w}_sm${arch}
  rm -rf "$out"
  TRITON_CACHE_DIR='"$OUT"'/cache/${w}_sm${arch} \
    python '"$HERE"'/run_census.py --workload $w --arch $arch --out "$out" '"$*"' \
    > '"$OUT"'/logs/${w}_sm${arch}.log 2>&1
  echo "done $w sm$arch rc=$? configs=$(wc -l < "$out/configs.jsonl" 2>/dev/null)"
'
