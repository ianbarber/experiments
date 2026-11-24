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

# other ptxas releases: registers and frames of the triton_kernels builds under each
# (nothing launched), and the device studies timed again with 12.9.86
for arch in 80 90 100 120; do cp "$RUNS/ptxas_versions/tk_sm$arch.json" "$RES/ptxas_versions_sm$arch.json"; done
TMP=$(mktemp -d)
python "$HERE/census_from_ttgir.py" --runs "$RUNS/e4_ptxas12.9" > /dev/null
python "$HERE/analyze_e4.py" --runs "$RUNS/e4_ptxas12.9" --out "$TMP" > /dev/null
python "$HERE/analyze_ptxas.py" --versions "$RES/ptxas_versions_sm120.json" --base "$RES/e4_summary.json" --base-version 13.5.37 \
  --other "$TMP/e4_summary.json" --other-version 12.9.86 \
  --other-targets "$RES"/ptxas_versions_sm80.json "$RES"/ptxas_versions_sm90.json "$RES"/ptxas_versions_sm100.json \
  --json "$RES/ptxas_summary.json" > "$RES/ptxas_tables.md"
(cd "$RUNS/e4_ptxas12.9" && tar --sort=name --mtime=2026-11-23 --owner=0 --group=0 --numeric-owner -cf - \
  */results.jsonl */session.json */ttgir_census.json */ttgir_canonical.json | gzip -n > "$RES/e4_ptxas12.9_results.tar.gz")
rm -r "$TMP"
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
