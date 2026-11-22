#!/usr/bin/env bash
# Rebuild every table, summary and figure of the entry from the raw runs.
#   RUNS=<raw output directory> ./assemble_results.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RES=$HERE/../results
IMG=$HERE/../images
mkdir -p "$RES" "$IMG"

python "$HERE/analyze_e3.py" --run "$RUNS/e3" --out "$RES" > "$RES/e3_tables.md"
python "$HERE/analyze_census.py" --runs "$RUNS/census" --out "$RES" --vanilla "$RUNS/vanilla" \
  --ratios "$RES/e3_ratios.json" > "$RES/e1_e2_tables.md"
python "$HERE/compare_reprice.py" --base "$RUNS/census" --repriced "$RUNS/reprice_0.12" --out "$RES" \
  --name reprice_0.12 > "$RES/reprice_0.12_tables.md"
python "$HERE/census_from_ttgir.py" --runs "$RUNS/e4" > /dev/null
python "$HERE/analyze_e4.py" --runs "$RUNS/e4" --out "$RES" > "$RES/e4_tables.md"
python "$HERE/report_tables.py" --results "$RES" > "$RES/report_tables.md"
python "$HERE/headline.py" --results "$RES" > "$RES/headline.txt"
python "$HERE/make_figures.py" --results "$RES" --out "$IMG"

# raw results small enough to keep
cp "$RUNS/e3/cases.json" "$RES/e3_cases.json"
cp "$RUNS/e3/cases_regperm.json" "$RES/e3_cases_register_permute.json"
cp "$RUNS/e3/results.jsonl" "$RES/e3_results.jsonl"
(cd "$RUNS/e4" && tar czf "$RES/e4_results.tar.gz" */results.jsonl */session.json */ttgir_census.json */ttgir_canonical.json)
python "$HERE/pack_ttgir.py" --runs "$RUNS/e4" --dirs tk_sm120 inductor_sm120 --out "$RES/e4_ttgir_sm120.tar.gz"
python "$HERE/pack_evidence.py" --runs "$RUNS/census" --arch 120 --out "$RES/census_evidence_sm120.tar.xz"
ls -la "$RES" "$IMG"
