#!/usr/bin/env bash
# SGLang server for memory-free text generation (fact-level self-study, experiments/lme_ssf_gen.py).
# Same bf16 weights as the HF experiments; greedy requests come with explicit sampling params.
# Never run while an HF job holds the GPU (unified memory): the script refuses if MemAvailable < 100 GiB.
#   ./serve_sglang.sh start | stop
set -euo pipefail
#   ./serve_sglang.sh start-long   # room check (LABNOTES 2026-10-13): 262k context, radix cache on
LONG="--mem-fraction-static 0.72 --max-running-requests 16 --cuda-graph-max-bs-decode 16 --context-length 262144 --chunked-prefill-size 8192"
GEN="--mem-fraction-static 0.60 --disable-radix-cache --max-running-requests 48 --cuda-graph-max-bs-decode 48 --max-total-tokens 65536 --context-length 8192 --chunked-prefill-size 4096"
case "${1:-start}" in
stop) docker rm -f statelm-sglang ;;
start|start-long)
  [ "${1:-start}" = start-long ] && PROFILE=$LONG || PROFILE=$GEN
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 100 ] || { echo "MemAvailable ${avail} GiB < 100; is a GPU job running?"; exit 1; }
  mkdir -p "$HOME/.cache/sglang-triton" "$HOME/.cache/sglang-flashinfer"
  docker run -d --name statelm-sglang --gpus all --ipc=host --shm-size 16g -p 127.0.0.1:30000:30000 \
    -v "${STATELM_WEIGHTS:?set STATELM_WEIGHTS to the directory holding Qwen3.8-27B}":/weights:ro -v "$HOME/.cache/sglang-triton":/root/.triton \
    -v "$HOME/.cache/sglang-flashinfer":/root/.cache/flashinfer \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_AUTOTUNING=1 \
    lmsysorg/sglang:qwen38-27b \
    python3 -m sglang.launch_server --model-path /weights/Qwen3.8-27B --served-model-name qwen3.8-27b \
      --trust-remote-code --host 0.0.0.0 --port 30000 \
      --dtype bfloat16 --kv-cache-dtype bfloat16 --mamba-ssm-dtype float32 \
      --attention-backend flashinfer $PROFILE --disable-prefill-cuda-graph --watchdog-timeout 1200
  ;;
esac
