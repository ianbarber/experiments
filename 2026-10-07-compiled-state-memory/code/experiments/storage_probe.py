"""Dev-only storage probe: is user-bound content *stored* in the compiled GDN state, whatever the
model manages to read out of it? (Exploratory; dev items only.)

For each of the 12 synth.ATTRS attributes (values X = pool[0], Y = pool[1]) and each task, 16
counterfactual *pairs* (32 memories). Pair k has its own filler seed and random entry positions;
its X version has the user holding X, its Y version the user holding Y, everything else equal
(samples 2k = X, 2k+1 = Y):
  raw60    one mention "The user ... {v}" among 60 entries
  bind60   user -> v and colleague -> the other value, 60 entries (identical token multisets)
  bind250  as bind60 with 250 entries (~6k tokens)
Saved per (attribute, task): per-layer, per-head Gram matrices of the flattened fp32 recurrent
states, [L, H, 32, 32] (data/probe/, not in git), from which difference-of-means probes and
kernel probes over any head/layer subset are computed offline. Also saved per sample: the model's
own readout d at the first divergent target token under the injected state (scale 1 and 2).

python experiments/storage_probe.py --model M --out /work/results/phase1/storage_probe_<tag>.jsonl
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

N_SAMPLES = 32
LISTS = (ATTRS,)
TASKS = {"raw60": (60, False), "bind60": (60, True), "bind250": (250, True)}


def first_div(a, b):
    a, b = a[0].tolist(), b[0].tolist()
    return next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items per list")
    ap.add_argument("--gram_dir", default="/work/data/probe")
    args = ap.parse_args()
    torch.manual_seed(42)
    if args.limit:
        for lst in LISTS:
            lst[:] = lst[:args.limit]
    os.makedirs(args.gram_dir, exist_ok=True)
    lm = LM(args.model)
    check_template(lm.tok)
    out = open(args.out, "w")
    out.write(json.dumps(dict(block="_header", model=args.model, n=N_SAMPLES, tasks=list(TASKS),
                              gdn_layers=lm.gdn_layers)) + "\n")
    t0 = time.time()

    for ai, (key, tmpl, q, lead, pool) in enumerate(ATTRS):
        X, Y = pool[0], pool[1]
        sess = lm.ids(session_text(lm.tok, q, lead))
        targets = [lm.ids(" " + X), lm.ids(" " + Y)]
        j = first_div(*targets)
        a = [v[j].item() for v in lm.score_multi(lm.cache_from_state(None), sess, targets)]
        for ti, (task, (n, bind)) in enumerate(TASKS.items()):
            grams = torch.zeros(len(lm.gdn_layers), lm.state_shape[1], N_SAMPLES, N_SAMPLES,
                                dtype=torch.float64, device=lm.device)
            states = []
            for i in range(N_SAMPLES):
                seed = 90000 + 1000 * ai + 100 * ti + i // 2
                rng = random.Random(seed)
                label = i % 2
                u, o = (X, Y) if label == 0 else (Y, X)
                if bind:
                    iu, ic = rng.sample(range(n), 2)
                    mem = make_memory2({iu: tmpl.format(v=u), ic: colleague(tmpl.format(v=o))}, n, seed)
                else:
                    iu, ic = rng.randrange(n), None
                    mem = make_memory2({iu: tmpl.format(v=u)}, n, seed)
                ids = lm.ids(compile_text(lm.tok, mem))
                st = lm.compile(ids)
                d = {}
                for g in (1.0, 2.0):
                    lp = lm.score_multi(lm.cache_from_state(st, scale=g), sess, targets)
                    d[f"d_g{g:g}"] = lp[0][j].item() - lp[1][j].item()
                out.write(json.dumps(dict(block=task, key=key, i=i, label=label, iu=iu, ic=ic, n=n,
                                          n_tokens=ids.shape[1], d_A=a[0] - a[1], **d)) + "\n")
                states.append([st.recurrent[li][0] for li in lm.gdn_layers])  # each [H, K, V] fp32
                del st
            for li in range(len(lm.gdn_layers)):
                R = torch.stack([s[li] for s in states]).flatten(2).double()  # [n, H, K*V]
                grams[li] = torch.einsum("nhd,mhd->hnm", R, R)
            torch.save(dict(grams=grams.cpu(), key=key, task=task),
                       os.path.join(args.gram_dir, f"{key}_{task}.pt"))
            del states
            out.flush()
            print(f"{key} {task} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
