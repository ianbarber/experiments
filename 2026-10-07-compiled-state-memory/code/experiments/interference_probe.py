"""Dev-only interference stress for the storage probe (exploratory; dev items only).

capacity_probe.py showed no loss over up to ~100k tokens of *neutral* filler. Under the delta rule,
what should erase a binding is later writes with similar keys, i.e. other facts of the same form.
Memory: PRE neutral entries | 10-entry slot block with the user fact and the colleague fact |
AFTER = 128 entries, of which K are interfering facts at random slots and the rest neutral filler.
Distance is fixed; only the number and kind of interferers varies:
  same   the target attribute held by other people ("The user's neighbour's dog is named Mochi.")
  other  other attributes, held by the user or by other people
  user   other attributes, held by the user        } follow-up cells (--cells), splitting
  person other attributes, held by other people    } *other* by subject; seed base 120000
Interferer values come from pool[2:] (never X or Y), and everything except the user/colleague swap
is shared within a pair. 12 attributes x 8 pairs per cell; cells K = 0 (shared), 8, 32, 128 per kind,
in ascending K. Saves per-head Grams [L, H, 16, 16] (data/interference/) and the model's readout d
(scale 1 and 2) per sample.

python experiments/interference_probe.py --model M --out /work/results/phase1/interference_probe_<tag>.jsonl
python experiments/interference_probe.py --show     # print one memory per cell (no model), then exit
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, ".")
from statelm.synth import ATTRS  # noqa: E402
from statelm.synth_v2 import colleague, make_memory2  # noqa: E402

PAIRS = 8
PRE, SLOTS, AFTER = 30, 10, 128
CELLS = [("none", 0), ("same", 8), ("other", 8), ("same", 32), ("other", 32), ("same", 128), ("other", 128)]
CELLS2 = [("user", 32), ("person", 32), ("user", 128), ("person", 128)]
ROLES = ["The user's neighbour", "The user's brother", "The user's manager", "The user's flatmate",
         "The user's cousin", "The user's dentist", "The user's old schoolfriend", "The user's landlord",
         "The user's aunt", "The user's teammate"]


def other_person(sentence, rng):
    return sentence.replace("The user", rng.choice(ROLES), 1)


def interferer(kind, ai, rng):
    if kind == "same":
        _, tmpl, _, _, pool = ATTRS[ai]
        return other_person(tmpl.format(v=rng.choice(pool[2:])), rng)
    oi = rng.choice([i for i in range(len(ATTRS)) if i != ai])
    _, tmpl, _, _, pool = ATTRS[oi]
    s = tmpl.format(v=rng.choice(pool[2:]))
    if kind == "user":
        return s
    if kind == "person":
        return other_person(s, rng)
    return s if rng.random() < 0.5 else other_person(s, rng)


def memory(ai, u, o, kind, k, seed):
    rng = random.Random(seed)
    _, tmpl, *_ = ATTRS[ai]
    iu, ic = (PRE + s for s in rng.sample(range(SLOTS), 2))
    at = {iu: tmpl.format(v=u), ic: colleague(tmpl.format(v=o))}
    for s in rng.sample(range(AFTER), k):
        at[PRE + SLOTS + s] = interferer(kind, ai, rng)
    return make_memory2(at, PRE + SLOTS + AFTER, seed), iu, ic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--gram_dir", default="/work/data/interference")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--cells", default="", help="comma-separated cell names from CELLS2 (default: CELLS)")
    args = ap.parse_args()
    if args.cells:
        named = {f"{k}{n}": (ci, (k, n)) for ci, (k, n) in enumerate(CELLS2)}
        run = [named[c] for c in args.cells.split(",")]
        base = 120000
    else:
        run, base = list(enumerate(CELLS)), 97000
    if args.show:
        for ci, (kind, k) in run:
            mem, iu, ic = memory(0, "Biscuit", "Pepper", kind, k, base + 100 * ci)
            lines = mem.split("\n")
            print(f"--- {kind} K={k}: {len(lines)} entries, facts at {iu},{ic}")
            print("\n".join(lines[iu - 1:ic + 2] if iu < ic else lines[ic - 1:iu + 2]))
            print("  ... after block sample:\n" + "\n".join(lines[PRE + SLOTS:PRE + SLOTS + 6]))
        return

    import torch
    from statelm.core import LM
    from statelm.prompts import check_template, compile_text, session_text
    torch.manual_seed(42)
    lm = LM(args.model)
    check_template(lm.tok)
    os.makedirs(args.gram_dir, exist_ok=True)
    out = open(args.out, "w")
    out.write(json.dumps(dict(block="_header", model=args.model, pairs=PAIRS, pre=PRE, slots=SLOTS,
                              after=AFTER, cells=[c for _, c in run], seed_base=base, roles=ROLES, gdn_layers=lm.gdn_layers)) + "\n")
    t0 = time.time()
    for ci, (kind, k) in run:
        cell = f"{kind}{k}"
        for ai, (key, tmpl, q, lead, pool) in enumerate(ATTRS):
            X, Y = pool[0], pool[1]
            sess = lm.ids(session_text(lm.tok, q, lead))
            targets = [lm.ids(" " + X), lm.ids(" " + Y)]
            j = next((t for t in range(min(targets[0].shape[1], targets[1].shape[1]))
                      if targets[0][0, t] != targets[1][0, t]), 0)
            a = [v[j].item() for v in lm.score_multi(lm.cache_from_state(None), sess, targets)]
            ns = 2 * PAIRS
            grams = torch.zeros(len(lm.gdn_layers), lm.state_shape[1], ns, ns, dtype=torch.float64,
                                device=lm.device)
            states = []
            for i in range(ns):
                seed = base + 1000 * ai + 100 * ci + i // 2
                label = i % 2
                u, o = (X, Y) if label == 0 else (Y, X)
                mem, iu, ic = memory(ai, u, o, kind, k, seed)
                ids = lm.ids(compile_text(lm.tok, mem))
                st = lm.compile(ids)
                d = {}
                for g in (1.0, 2.0):
                    lp = lm.score_multi(lm.cache_from_state(st, scale=g), sess, targets)
                    d[f"d_g{g:g}"] = lp[0][j].item() - lp[1][j].item()
                out.write(json.dumps(dict(block=cell, key=key, i=i, label=label, iu=iu, ic=ic,
                                          n_tokens=ids.shape[1], d_A=a[0] - a[1], **d)) + "\n")
                states.append([st.recurrent[li][0].flatten(1) for li in lm.gdn_layers])
                del st
            for li in range(len(lm.gdn_layers)):
                R = torch.stack([s[li] for s in states]).double()
                grams[li] = torch.einsum("nhd,mhd->hnm", R, R)
            torch.save(dict(grams=grams.cpu(), key=key, cell=cell),
                       os.path.join(args.gram_dir, f"{key}_{cell}.pt"))
            del states
            out.flush()
            print(f"{cell} {key} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
