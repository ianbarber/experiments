#!/usr/bin/env bash
# Room check stage 1a (LABNOTES 2026-10-13, night): arm D's one-line notes for every session of the
# calibration stores, on the SGLang generation profile. Nothing else may hold the GPU.
set -e
cd "$(dirname "$0")/.."
[ "$(docker ps -q | wc -l)" -eq 0 ] || { echo "a container is running"; exit 1; }
./serve_sglang.sh start
trap './serve_sglang.sh stop' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:30000/health >/dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:30000/health >/dev/null
uv run --with "transformers==5.12.1" --with jinja2 python experiments/room_digest.py --prep ../results/phase2/room/prep
echo "digests done"
