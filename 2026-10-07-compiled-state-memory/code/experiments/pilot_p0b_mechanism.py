"""Pilot P0b: where does the compiled-state fact signal come from? (follow-up to P0)

Same counterfactual shift as P0 (memory of 60 entries), with:
  pos    target entry index from the start {0,1,2,4,8} and from the end {0,1,2,4,8}
  band   inject state only into the first / middle / last third of GDN layers (others zero)
  heads  inject only heads with nominal half-life > 256 tokens ("long") or <= 256 ("short")
  scale  multiply the whole compiled state by {0.5, 2, 4}
Variants other than `pos` run at target index 0 (first entry) and n-1 (last entry).

python experiments/pilot_p0b_mechanism.py --model M --out /work/results/pilot/p0b_<tag>.jsonl
"""
import argparse
import json
import math
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from statelm.core import LM  # noqa: E402
from statelm.prompts import compile_text, session_text  # noqa: E402
from statelm.synth import ATTRS, make_memory  # noqa: E402

N = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs_per_attr", type=int, default=2)
    args = ap.parse_args()
    torch.manual_seed(42)
    lm = LM(args.model)
    out = open(args.out, "w")
    t0 = time.time()

    # nominal half-life per head (data-independent part of the gate)
    long_mask, short_mask = {}, {}
    for i in lm.gdn_layers:
        la = lm.tm.layers[i].linear_attn
        g0 = -la.A_log.float().exp() * F.softplus(la.dt_bias.float())
        hl = math.log(2) / g0.abs()
        long_mask[i] = (hl > 256).float()
        short_mask[i] = (hl <= 256).float()
    nl = len(lm.gdn_layers)
    bands = {"early": lm.gdn_layers[:nl // 3], "mid": lm.gdn_layers[nl // 3:2 * nl // 3],
             "late": lm.gdn_layers[2 * nl // 3:]}
    frac_long = float(torch.cat(list(long_mask.values())).mean())
    print(f"long heads (nominal hl > 256): {frac_long:.3f}", flush=True)

    def variants(idx):
        v = [("full", {})]
        if idx in (0, N - 1):
            for b, layers in bands.items():
                v.append((f"band_{b}", dict(layer_mask={i: (1.0 if i in layers else 0.0) for i in lm.gdn_layers})))
            v.append(("heads_long", dict(head_mask=long_mask)))
            v.append(("heads_short", dict(head_mask=short_mask)))
            for s in (0.5, 2.0, 4.0):
                v.append((f"scale_{s}", dict(scale=s)))
        return v

    positions = sorted({0, 1, 2, 4, 8} | {N - 1 - k for k in (0, 1, 2, 4, 8)})
    for ai, (key, tmpl, q, lead, pool) in enumerate(ATTRS):
        for p in range(args.pairs_per_attr):
            X, Y = pool[2 * p], pool[2 * p + 1]
            xa, ya = lm.ids(" " + X), lm.ids(" " + Y)
            sids = lm.ids(session_text(lm.tok, q, lead))
            seed = 1000 * ai + 10 * p + N
            for idx in positions:
                stX = lm.compile(lm.ids(compile_text(lm.tok, make_memory(key, X, N, idx, seed=seed))))
                stY = lm.compile(lm.ids(compile_text(lm.tok, make_memory(key, Y, N, idx, seed=seed))))
                for name, kw in variants(idx):
                    r = dict(attr=key, X=X, Y=Y, target_index=idx, variant=name)
                    for tag, st in (("C_X", stX), ("C_Y", stY)):
                        lx = lm.score(lm.cache_from_state(st, **kw), sids, xa).sum().item()
                        ly = lm.score(lm.cache_from_state(st, **kw), sids, ya).sum().item()
                        r[tag] = (lx, ly)
                    out.write(json.dumps(r) + "\n")
                out.flush()
            print(f"{key} pair {p} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
