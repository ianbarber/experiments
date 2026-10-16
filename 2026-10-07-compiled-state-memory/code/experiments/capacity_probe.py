"""Dev-only capacity stress for the storage probe (exploratory; dev items only).

Is user-bound content still *stored* in the compiled GDN state when many entries are written after
it, and does the model's own readout fall off faster than the stored signal? For each synth.ATTRS
attribute (X = pool[0], Y = pool[1]) and each distance stratum, 8 counterfactual pairs. A memory is
  PRE neutral entries | 10-entry slot block holding the user fact and the colleague fact at two
  random slots | AFTER neutral entries
so the facts sit AFTER..AFTER+9 entries from the end (~25 tokens per entry). Pair k shares filler,
slots and dates; X version: user -> X, colleague -> Y; Y version swapped (samples 2k = X, 2k+1 = Y).
Strata run in ascending order (d4000 on every other attribute only, for time), so a stopped run
still leaves complete strata.
Saved per (attribute, stratum): per-layer, per-head Grams [L, H, 16, 16] (data/capacity/, not in
git) and per sample the model's readout d at the first divergent target token (scale 1 and 2).

python experiments/capacity_probe.py --model M --out /work/results/phase1/capacity_probe_<tag>.jsonl
python experiments/capacity_probe.py --model M --timing      # one compile per stratum, then exit
"""
import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, ".")
from statelm.core import LM  # noqa: E402
from statelm.prompts import check_template, compile_text, session_text  # noqa: E402
from statelm.synth import ATTRS  # noqa: E402
from statelm.synth_v2 import colleague, make_memory2  # noqa: E402

PAIRS = 8
PRE, SLOTS = 30, 10
STRATA = {"d30": 30, "d250": 250, "d1000": 1000, "d4000": 4000}
STRIDE = {"d4000": 2}  # ~100k-token memories take ~195 s each: every other attribute only


def first_div(a, b):
    a, b = a[0].tolist(), b[0].tolist()
    return next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)) - 1)


def memory(tmpl, u, o, after, seed):
    rng = random.Random(seed)
    iu, ic = (PRE + s for s in rng.sample(range(SLOTS), 2))
    n = PRE + SLOTS + after
    return make_memory2({iu: tmpl.format(v=u), ic: colleague(tmpl.format(v=o))}, n, seed), iu, ic, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--gram_dir", default="/work/data/capacity")
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(42)
    lm = LM(args.model)
    check_template(lm.tok)

    if args.timing:
        key, tmpl, q, lead, pool = ATTRS[0]
        for name, after in STRATA.items():
            mem, *_ = memory(tmpl, pool[0], pool[1], after, 1)
            ids = lm.ids(compile_text(lm.tok, mem))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            lm.compile(ids)
            torch.cuda.synchronize()
            print(f"{name}: {ids.shape[1]} tokens, compile {time.time() - t:.1f}s, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        return

    os.makedirs(args.gram_dir, exist_ok=True)
    out = open(args.out, "w")
    out.write(json.dumps(dict(block="_header", model=args.model, pairs=PAIRS, pre=PRE, slots=SLOTS,
                              strata=STRATA, gdn_layers=lm.gdn_layers)) + "\n")
    t0 = time.time()
    for si, (stratum, after) in enumerate(STRATA.items()):
        for ai, (key, tmpl, q, lead, pool) in enumerate(ATTRS):
            if ai % STRIDE.get(stratum, 1):
                continue
            X, Y = pool[0], pool[1]
            sess = lm.ids(session_text(lm.tok, q, lead))
            targets = [lm.ids(" " + X), lm.ids(" " + Y)]
            j = first_div(*targets)
            a = [v[j].item() for v in lm.score_multi(lm.cache_from_state(None), sess, targets)]
            ns = 2 * PAIRS
            grams = torch.zeros(len(lm.gdn_layers), lm.state_shape[1], ns, ns, dtype=torch.float64,
                                device=lm.device)
            states = []
            for i in range(ns):
                seed = 95000 + 1000 * ai + 100 * si + i // 2
                label = i % 2
                u, o = (X, Y) if label == 0 else (Y, X)
                mem, iu, ic, n = memory(tmpl, u, o, after, seed)
                ids = lm.ids(compile_text(lm.tok, mem))
                st = lm.compile(ids)
                d = {}
                for g in (1.0, 2.0):
                    lp = lm.score_multi(lm.cache_from_state(st, scale=g), sess, targets)
                    d[f"d_g{g:g}"] = lp[0][j].item() - lp[1][j].item()
                out.write(json.dumps(dict(block=stratum, key=key, i=i, label=label, iu=iu, ic=ic, n=n,
                                          n_tokens=ids.shape[1], d_A=a[0] - a[1], **d)) + "\n")
                # keep states on the GPU only as flattened per-layer fp32 rows for the Gram
                states.append([st.recurrent[li][0].flatten(1) for li in lm.gdn_layers])  # each [H, K*V]
                del st
            for li in range(len(lm.gdn_layers)):
                R = torch.stack([s[li] for s in states]).double()  # [n, H, K*V]
                grams[li] = torch.einsum("nhd,mhd->hnm", R, R)
            torch.save(dict(grams=grams.cpu(), key=key, stratum=stratum),
                       os.path.join(args.gram_dir, f"{key}_{stratum}.pt"))
            del states
            out.flush()
            print(f"{stratum} {key} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
