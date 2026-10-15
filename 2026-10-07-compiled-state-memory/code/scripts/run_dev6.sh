#!/usr/bin/env bash
# Stage 1 dev run 6 (LABNOTES 2026-10-11): fact-level self-study rows (answer windows), memory-derived weights,
# held-out half and matched uniform control at 30% and 60% of the fit weight; then the absent store.
set -e
cd "$(dirname "$0")/.."
source models.sh
export DOCKER_EXTRA="-v ${STATELM_SEEDQ:?set STATELM_SEEDQ to a local cache dir for seeds and query sets}:/seedq"
./run.sh python -u experiments/lme_seed.py --model $Q38_27B --qids 0862e8bf,6f9b354f --ratios 64 \
  --refs sst,sst+ssfd%30,sst+ssfAd%30,sst+ssfAu%30,sst+ssfd%60,sst+ssfAd%60,sst+ssfAu%60 --ssf_rows ans \
  --ssf_dir /work/results/phase2/ssf --ss_cache /seedq --seed_dir /seedq/seeds6 --stores p \
  --out /work/results/phase2/lme_seed_dev6.jsonl
echo "present store done"
./run.sh python -u experiments/lme_seed.py --model $Q38_27B --qids 0862e8bf,6f9b354f --ratios 64 \
  --refs sst,sst+ssfd%30,sst+ssfd%60 --ssf_rows ans \
  --ssf_dir /work/results/phase2/ssf --ss_cache /seedq --seed_dir /seedq/seeds6 --stores a \
  --out /work/results/phase2/lme_seed_dev6a.jsonl
