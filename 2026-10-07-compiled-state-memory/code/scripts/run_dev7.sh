#!/usr/bin/env bash
# Stage 1 dev run 7 (LABNOTES 2026-10-11): k-means seeds loaded from dev run 6, mass-weighted keys (/mw) and
# query-metric mass keys (/mq) fitted on the same rows, and the drift diagnostic on every seed's ssf answer rows.
set -e
cd "$(dirname "$0")/.."
source models.sh
mkdir -p ${STATELM_SEEDQ:?set STATELM_SEEDQ to a local cache dir for seeds and query sets}/seeds7
for q in 0862e8bf 6f9b354f; do test -f ${STATELM_SEEDQ:?set STATELM_SEEDQ to a local cache dir for seeds and query sets}/seeds6/${q}_p_r64.pt; done
export DOCKER_EXTRA="-v ${STATELM_SEEDQ:?set STATELM_SEEDQ to a local cache dir for seeds and query sets}:/seedq"
./run.sh python -u experiments/lme_seed.py --model $Q38_27B --qids 0862e8bf,6f9b354f --ratios 64 \
  --refs sst,sst+ssfd%30,sst+ssfAd%30,sst/mw,sst+ssfd%30/mw,sst+ssfAd%30/mw,sst+ssfd%30/mq --ssf_rows ans \
  --ssf_dir /work/results/phase2/ssf --ss_cache /seedq --load_seeds /seedq/seeds6 --seed_dir /seedq/seeds7 --drift \
  --stores p --out /work/results/phase2/lme_seed_dev7.jsonl
echo "dev run 7 done"
