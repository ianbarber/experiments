#!/usr/bin/env bash
# Stage 1 dev run 5 (LABNOTES 2026-10-11): which probe rows carry the lookup, dose, bias-only seeds
set -e
cd "$(dirname "$0")/.."
source models.sh
export DOCKER_EXTRA="-v ${STATELM_SEEDQ:?set STATELM_SEEDQ to a local cache dir for seeds and query sets}:/seedq"
./run.sh python -u experiments/lme_seed.py --model $Q38_27B --qids 0862e8bf,6f9b354f --ratios 64 \
  --refs sstpw,sst+probe@5,sst+probe@100,sst+anchor@20,sst+sess@20,sst+gold@100,sst+gold@1000,probe --bias_only \
  --ss_cache /seedq --seed_dir /seedq/seeds5 --stores p --out /work/results/phase2/lme_seed_dev5.jsonl
