# Research image for statelm: the GB10-working SGLang image + flash-linear-attention kernels.
# Base pins torch 2.13.0+cu130 / transformers 5.12.1 / triton 3.7.1 (sm_121 capable).
FROM lmsysorg/sglang:qwen38-27b
RUN pip install --no-cache-dir --no-deps fla-core==0.5.2 flash-linear-attention==0.5.2 accelerate==1.13.0 \
 && pip install --no-cache-dir rank_bm25==0.2.2 pytest
# Cache Triton autotune choices in the (mounted, persistent) Triton cache so runs pick identical
# kernel configs across processes (dev vs test, A/A repeats).
ENV TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_AUTOTUNING=1
WORKDIR /work
