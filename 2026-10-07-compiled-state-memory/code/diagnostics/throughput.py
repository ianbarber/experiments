"""Measure HF-path throughput on dgx-spark: compile (prefill) tok/s at several lengths, greedy
decode tok/s at batch 1, and peak memory. Run alone on the GPU.

python diagnostics/throughput.py --model M --out /work/results/audit/throughput_<tag>.json
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from statelm.core import LM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="2048,8192,32768,65536")
    args = ap.parse_args()
    t0 = time.time()
    lm = LM(args.model)
    res = {"model": args.model, "load_s": time.time() - t0, "prefill": {}, "decode": {}}
    text = open("/work/data/raw/text/qwen38_model_card.md").read()
    base = lm.tok.encode(text, add_special_tokens=False)
    for L in [int(x) for x in args.lengths.split(",")]:
        ids = torch.tensor([(base * (L // len(base) + 1))[:L]], device=lm.device)
        lm.compile(ids[:, :512])                      # warm-up (triton autotune)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        lm.compile(ids)
        torch.cuda.synchronize()
        dt = time.time() - t
        res["prefill"][L] = dict(seconds=dt, tok_per_s=L / dt, peak_gb=torch.cuda.max_memory_allocated() / 2**30)
        print(L, res["prefill"][L], flush=True)
    for ctx in [512, 8192]:
        ids = torch.tensor([base[:ctx]], device=lm.device)
        cache = lm.new_cache()
        lm.forward(ids, cache)
        nxt = torch.tensor([[base[ctx]]], device=lm.device)
        for _ in range(5):
            lm.forward(nxt, cache)
        torch.cuda.synchronize()
        t = time.time()
        n = 64
        for _ in range(n):
            lm.forward(nxt, cache)
        torch.cuda.synchronize()
        dt = time.time() - t
        res["decode"][ctx] = dict(tok_per_s=n / dt)
        print("decode ctx", ctx, res["decode"][ctx], flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
