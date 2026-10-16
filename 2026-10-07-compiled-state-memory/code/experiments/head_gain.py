"""Dev-only head-targeted gain (exploratory; dev items only, never the sealed v2 items).

The storage probe found user-bound values decodable from a few dozen GDN heads shared across
attributes, while the model's own readout is weak, and uniform gain amplifies familiarity (lures,
recency) along with binding. Does amplifying only those heads raise binding more selectively?
Same items and seeds as gain_sweep.py (blocks bind / raw / prefs / lures / guard), so the rows
are directly comparable and analysis/gain_sweep.py reads this file unchanged. Configurations
(multiplier on the listed heads' states, all other heads x1):
  g1, g2                    no gain / uniform x2 (reference, re-run here)
  h16x4, h16x8, h64x2, h64x4  top-16 / top-64 binding heads
  r16x8, r64x4              layer-matched random heads (controls)
Head sets come from analysis/binding_heads.py: for bind / raw, ranked on the *other* 11
attributes; for prefs / lures / guard, the global ranking (those items are not ATTRS).

python experiments/head_gain.py --model M --heads /work/results/phase1/binding_heads_<tag>.json \
    --out /work/results/phase1/head_gain_<tag>.jsonl
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from statelm.core import LM  # noqa: E402
from statelm.prompts import check_template, compile_text, session_text  # noqa: E402
from statelm.synth import ATTRS  # noqa: E402
from statelm.synth_prefs import CAPABILITY, CAPABILITY_DISTRACTORS, LURES, PREFS  # noqa: E402
from statelm.synth_v2 import colleague, make_memory2  # noqa: E402

N = 60
LAYOUTS = {"u0_c30": (0, 30), "c0_u30": (30, 0), "u59_c30": (59, 30), "c59_u30": (30, 59)}
CONFIGS = {"g1": None, "g2": None, "h16x4": ("top", 16, 4.0), "h16x8": ("top", 16, 8.0),
           "h64x2": ("top", 64, 2.0), "h64x4": ("top", 64, 4.0),
           "r16x8": ("rand", 16, 8.0), "r64x4": ("rand", 64, 4.0)}


def first_div(a, b):
    a, b = a[0].tolist(), b[0].tolist()
    return next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(42)
    hs = json.load(open(args.heads))
    lm = LM(args.model)
    check_template(lm.tok)
    H = lm.state_shape[1]
    assert H == hs["heads_per_layer"]
    out = open(args.out, "w")
    out.write(json.dumps(dict(block="_header", model=args.model, gains=list(CONFIGS), n=N, heads=args.heads)) + "\n")
    t0 = time.time()

    def kwargs(key):
        """Per-config cache_from_state kwargs; key = ATTRS key (leave-attribute-out heads) or None (global)."""
        res = {}
        for name, spec in CONFIGS.items():
            if spec is None:
                res[name] = {} if name == "g1" else dict(scale=2.0)
                continue
            kind, k, m = spec
            heads = (hs["by_attr"][key] if kind == "top" else hs["random"][key]) if key else \
                (hs["global_"] if kind == "top" else hs["random_global"])
            mask = {}
            for layer, h in heads[:k]:
                mask.setdefault(layer, torch.ones(H))[h] = m
            res[name] = dict(head_mask=mask)
        return res

    def sc(cache, sess, targets):
        return [v.tolist() for v in lm.score_multi(cache, sess, targets)]

    def compile_(mem):
        return lm.compile(lm.ids(compile_text(lm.tok, mem)))

    def pair(block, key, kw, memX, memY, q, lead, tX, tY, meta):
        sess = lm.ids(session_text(lm.tok, q, lead))
        targets = [lm.ids(tX), lm.ids(tY)]
        sts = {"X": compile_(memX), "Y": compile_(memY)}
        row = dict(block=block, key=key, j=first_div(*targets), **meta,
                   A=sc(lm.cache_from_state(None), sess, targets))
        row["C"] = {g: {t: sc(lm.cache_from_state(sts[t], **kw[g]), sess, targets) for t in "XY"} for g in kw}
        out.write(json.dumps(row) + "\n")
        out.flush()

    for ai, (key, tmpl, q, lead, pool) in enumerate(ATTRS):
        X, Y = pool[0], pool[1]
        seed = 80000 + 100 * ai
        kw = kwargs(key)
        for lname, (iu, ic) in LAYOUTS.items():
            memX = make_memory2({iu: tmpl.format(v=X), ic: colleague(tmpl.format(v=Y))}, N, seed)
            memY = make_memory2({iu: tmpl.format(v=Y), ic: colleague(tmpl.format(v=X))}, N, seed)
            pair("bind", key, kw, memX, memY, q, lead, " " + X, " " + Y, dict(layout=lname))
        for pos, idx in (("first", 0), ("last", N - 1)):
            pair("raw", key, kw, make_memory2({idx: tmpl.format(v=X)}, N, seed + 1),
                 make_memory2({idx: tmpl.format(v=Y)}, N, seed + 1), q, lead, " " + X, " " + Y, dict(pos=pos))
        print(f"facts {key} {time.time() - t0:.0f}s", flush=True)

    kw = kwargs(None)
    for pi, (key, sX, sY, q, cX, cY) in enumerate(PREFS):
        for pos, idx in (("first", 0), ("last", N - 1)):
            seed = 81000 + 100 * pi
            pair("prefs", key, kw, make_memory2({idx: sX}, N, seed), make_memory2({idx: sY}, N, seed),
                 q, "", cX, cY, dict(pos=pos))
    print(f"prefs {time.time() - t0:.0f}s", flush=True)

    yes, no = lm.ids(" Yes"), lm.ids(" No")
    for li, (lure, assoc, neutral) in enumerate(LURES):
        words = assoc.split(", ")
        present = ", ".join(words[:5] + [lure] + words[5:9])
        q = f"Did the list of words I wrote down include the word '{lure}'? Answer Yes or No."
        sess = lm.ids(session_text(lm.tok, q, "Answer:"))
        for kind, lst in (("assoc", assoc), ("present", present), ("neutral", neutral)):
            for pos, idx in (("first", 0), ("last", N - 1)):
                st = compile_(make_memory2({idx: f"The user wrote down this list of words: {lst}."}, N, 82000 + li))
                row = dict(block="lures", key=lure, kind=kind, pos=pos,
                           A=sc(lm.cache_from_state(None), sess, [yes, no]),
                           C={g: sc(lm.cache_from_state(st, **kw[g]), sess, [yes, no]) for g in kw})
                out.write(json.dumps(row) + "\n")
        out.flush()
    print(f"lures {time.time() - t0:.0f}s", flush=True)

    mems = []
    for i in range(2):
        key, tmpl, q, lead, pool = ATTRS[(i * 5) % len(ATTRS)]
        mems.append(("bind:" + key, make_memory2({0: tmpl.format(v=pool[0]), 30: colleague(tmpl.format(v=pool[1]))},
                                                 N, 83000 + i)))
        key, sX, *_ = PREFS[(i * 5) % len(PREFS)]
        mems.append(("pref:" + key, make_memory2({N - 1: sX}, N, 83100 + i)))
    states = [(name, compile_(m)) for name, m in mems]
    for qi, ((q, lead, gold), dis) in enumerate(zip(CAPABILITY, CAPABILITY_DISTRACTORS)):
        sess = lm.ids(session_text(lm.tok, q, lead))
        targets = [lm.ids(gold), lm.ids(dis)]
        row = dict(block="guard", key=qi, j=first_div(*targets), A=sc(lm.cache_from_state(None), sess, targets))
        row["C"] = {g: {name: sc(lm.cache_from_state(st, **kw[g]), sess, targets) for name, st in states} for g in kw}
        out.write(json.dumps(row) + "\n")
    out.flush()
    print(f"guard {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
