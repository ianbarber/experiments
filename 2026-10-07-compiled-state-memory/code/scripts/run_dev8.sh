#!/usr/bin/env bash
# Stage 1 dev run 8 (LABNOTES 2026-10-12): mass keys (/mw) only. 1) leak check on the absent store (k-means sst
# loaded from dev run 6a); 2) seed size r = 64 (loaded from dev run 7), 256, 128, 32; 3) dose: 60% at r = 64;
# 4) r = 16, last (longest fit, largest footprint).
set -e
cd "$(dirname "$0")/.."
source models.sh
S=${STATELM_SEEDQ:?set STATELM_SEEDQ to a local cache dir for seeds and query sets}
mkdir -p $S/seeds8a $S/seeds8 $S/seeds8d
for q in 0862e8bf 6f9b354f; do test -f $S/seeds6/${q}_a_r64.pt; test -f $S/seeds7/${q}_p_r64.pt; done
export DOCKER_EXTRA="-v $S:/seedq"
COMMON="--model $Q38_27B --qids 0862e8bf,6f9b354f --ssf_rows ans --ssf_dir /work/results/phase2/ssf --ss_cache /seedq"
./run.sh python -u experiments/lme_seed.py $COMMON --ratios 64 --refs sst,sst/mw,sst+ssfd%30/mw \
  --load_seeds /seedq/seeds6 --seed_dir /seedq/seeds8a --stores a --out /work/results/phase2/lme_seed_dev8a.jsonl
echo "part 1 (absent store) done"
./run.sh python -u experiments/lme_seed.py $COMMON --ratios 64,256,128,32 --refs sst/mw,sst+ssfd%30/mw,sst+ssfAd%30/mw \
  --load_seeds /seedq/seeds7 --seed_dir /seedq/seeds8 --drift --stores p --out /work/results/phase2/lme_seed_dev8.jsonl
echo "part 2 (size) done"
./run.sh python -u experiments/lme_seed.py $COMMON --ratios 64 --refs sst/mw,sst+ssfd%30/mw,sst+ssfd%60/mw \
  --load_seeds /seedq/seeds7 --seed_dir /seedq/seeds8d --drift --stores p --out /work/results/phase2/lme_seed_dev8d.jsonl
echo "part 3 (dose) done"
./run.sh python -u experiments/lme_seed.py $COMMON --ratios 16 --refs sst/mw,sst+ssfd%30/mw,sst+ssfAd%30/mw \
  --seed_dir /seedq/seeds8 --drift --stores p --out /work/results/phase2/lme_seed_dev8r16.jsonl
echo "dev run 8 done"
