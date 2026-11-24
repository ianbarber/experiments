"""Stage 1: per-token loss of SmolLM-135M under every keep-table, on every dataset. No training.

Writes ../raw/stage1/<dataset>__<condition>.npz (per-token loss, and what each dataset's metrics need).
Existing files are skipped, so the script can be re-run or extended.

python stage1_eval.py [--datasets prose code ...] [--conds hisa-90 ...] [--mean-bias ../raw/dropped_mean.pt]
"""

import argparse
import re
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import keeptable as kt

MODEL = "HuggingFaceTB/SmolLM-135M"
DATASETS = ["prose", "code", "tables", "copy"] + [f"grid-{c}" for c in (8, 16, 32, 64, 128, 256)]
CONDS = (["dense"] + [f"hisa-{a}" for a in (45, 90, 180, 360, 720)] + list(kt.BANDS)
         + [f"ctrl-perm-s{s}" for s in range(3)] + ["ctrl-slow", "ctrl-iid-s0"])
OUT = "../raw/stage1"


def mean_bias(mean, M):
    """mean [L, H, P, T] (mean part of pair l at distance d); M [P, T]. -> [L, H, T]: sum over dropped pairs."""
    return (mean * (~M)[None, None].to(mean.dtype)).sum(2)


class BiasedTable(kt.TableAttention):
    """Table attention plus a per-head additive bias b[h, d] on the scores (mean ablation)."""

    bias = None

    def scores(self, q, k):
        s = super().scores(q, k)
        if self.bias is not None:
            T = s.shape[-1]
            pos = torch.arange(T, device=s.device)
            d = (pos[:, None] - pos[None, :]).clamp(min=0)
            s = s + self.bias[:, d] / 8.0
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--conds", nargs="+", default=CONDS)
    ap.add_argument("--mean-bias", default=None)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    theta = model.model.rotary_emb.inv_freq.detach().cpu()
    cores = kt.patch_model(model)
    for c in cores:
        c.__class__ = BiasedTable
    mean = torch.load(args.mean_bias) if args.mean_bias else None

    for cond in args.conds:
        base = cond[:-5] if cond.endswith("-mean") else cond
        keep = re.search(r"-keep(\d+)$", base)                  # e.g. hisa-90-keep1: first key keeps every pair
        base = base[:keep.start()] if keep else base
        M = kt.make_table(base, theta, 2048)
        kt.set_condition(cores, M)
        for li, c in enumerate(cores):
            c.bias = None
            c.exempt_keys = int(keep.group(1)) if keep else 0
            if cond.endswith("-mean"):
                c.bias = mean_bias(mean, M)[li].cuda().float()
                c.is_dense = False
        for ds in args.datasets:
            path = f"{OUT}/{ds}__{cond}.npz"
            if os.path.exists(path):
                continue
            d = np.load(f"../data/{ds}.npz")
            ids = torch.from_numpy(d["ids"]).long()
            t0 = time.time()
            loss, extra = [], {}
            p_nl, top1 = [], []
            nl = int(d["newline"]) if "newline" in d.files else None
            with torch.no_grad():
                for i in range(0, len(ids), args.bs):
                    x = ids[i:i + args.bs].cuda()
                    logits = model(x).logits[:, :-1]
                    lp = torch.log_softmax(logits, -1)
                    loss.append(-lp.gather(-1, x[:, 1:, None])[..., 0].float().cpu())
                    top1.append(logits.argmax(-1).int().cpu())
                    if nl is not None:
                        p_nl.append(lp[..., nl].exp().float().cpu())
            loss = torch.cat(loss).numpy()                     # [N, 2047]: loss of token t+1 given tokens <= t
            extra["top1"] = torch.cat(top1).numpy()
            if nl is not None:
                extra["p_newline"] = torch.cat(p_nl).numpy()   # P(next = newline) at every position
            np.savez_compressed(path, loss=loss, **extra)
            print(f"{cond:14s} {ds:9s} loss {loss.mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
