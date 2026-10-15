"""Dev-only readout-gain sweep (exploratory; dev items only, never the sealed v2 items).

Does amplifying the injected GDN state raise *binding* (not just priming), and at what cost to
unrelated questions? Items: synth.ATTRS (12 attributes, values pool[0]/pool[1]) and
synth_prefs.PREFS (12 dev preferences), v2 neutral filler, 60 entries, new seeds. Gains:
  g1 / g1.5 / g2 / g3          whole state x s
  late2 / late3                late third of GDN layers x s, the rest x1
  lateonly2                    late third x2, the rest zeroed
Blocks:
  bind   user->X & colleague->Y vs swapped, 4 mirrored layouts (as G1 B1)
  raw    single mention, first / last
  prefs  polarity (fact form), first / last
  lures  DRM lists (assoc / present / neutral) x first / last  [lists also used in G1]
  guard  40 general questions x 4 compiled memories, gold vs distractor  [also used in G1 K2]
Rows store per-token log-probs for A and, per gain, C_X / C_Y (guard: per memory).

python experiments/gain_sweep.py --model M --out /work/results/phase1/gain_sweep_<tag>.jsonl
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
LISTS = (ATTRS, PREFS, LURES, CAPABILITY, CAPABILITY_DISTRACTORS)
LAYOUTS = {"u0_c30": (0, 30), "c0_u30": (30, 0), "u59_c30": (59, 30), "c59_u30": (30, 59)}


def first_div(a, b):
    a, b = a[0].tolist(), b[0].tolist()
    return next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items per list")
    args = ap.parse_args()
    torch.manual_seed(42)
    if args.limit:
        for lst in LISTS:
            lst[:] = lst[:args.limit]
    lm = LM(args.model)
    check_template(lm.tok)
    out = open(args.out, "w")
    t0 = time.time()

    nl = len(lm.gdn_layers)
    late = set(lm.gdn_layers[2 * nl // 3:])
    gains = {"g1": {}, "g1.5": dict(scale=1.5), "g2": dict(scale=2.0), "g3": dict(scale=3.0),
             "late2": dict(layer_mask={i: (2.0 if i in late else 1.0) for i in lm.gdn_layers}),
             "late3": dict(layer_mask={i: (3.0 if i in late else 1.0) for i in lm.gdn_layers}),
             "lateonly2": dict(layer_mask={i: (2.0 if i in late else 0.0) for i in lm.gdn_layers})}
    out.write(json.dumps(dict(block="_header", model=args.model, gains=list(gains), n=N)) + "\n")

    def sc(cache, sess, targets):
        return [v.tolist() for v in lm.score_multi(cache, sess, targets)]

    def compile_(mem):
        return lm.compile(lm.ids(compile_text(lm.tok, mem)))

    def pair(block, key, memX, memY, q, lead, tX, tY, meta):
        sess = lm.ids(session_text(lm.tok, q, lead))
        targets = [lm.ids(tX), lm.ids(tY)]
        sts = {"X": compile_(memX), "Y": compile_(memY)}
        row = dict(block=block, key=key, j=first_div(*targets), **meta,
                   A=sc(lm.cache_from_state(None), sess, targets))
        row["C"] = {g: {t: sc(lm.cache_from_state(sts[t], **kw), sess, targets) for t in "XY"}
                    for g, kw in gains.items()}
        out.write(json.dumps(row) + "\n")
        out.flush()

    for ai, (key, tmpl, q, lead, pool) in enumerate(ATTRS):
        X, Y = pool[0], pool[1]
        seed = 80000 + 100 * ai
        for lname, (iu, ic) in LAYOUTS.items():
            memX = make_memory2({iu: tmpl.format(v=X), ic: colleague(tmpl.format(v=Y))}, N, seed)
            memY = make_memory2({iu: tmpl.format(v=Y), ic: colleague(tmpl.format(v=X))}, N, seed)
            assert sorted(memX.split()) == sorted(memY.split())
            pair("bind", key, memX, memY, q, lead, " " + X, " " + Y, dict(layout=lname))
        for pos, idx in (("first", 0), ("last", N - 1)):
            pair("raw", key, make_memory2({idx: tmpl.format(v=X)}, N, seed + 1),
                 make_memory2({idx: tmpl.format(v=Y)}, N, seed + 1), q, lead, " " + X, " " + Y, dict(pos=pos))
        print(f"facts {key} {time.time() - t0:.0f}s", flush=True)

    for pi, (key, sX, sY, q, cX, cY) in enumerate(PREFS):
        for pos, idx in (("first", 0), ("last", N - 1)):
            seed = 81000 + 100 * pi
            pair("prefs", key, make_memory2({idx: sX}, N, seed), make_memory2({idx: sY}, N, seed),
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
                           C={g: sc(lm.cache_from_state(st, **kw), sess, [yes, no]) for g, kw in gains.items()})
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
        row["C"] = {g: {name: sc(lm.cache_from_state(st, **kw), sess, targets) for name, st in states}
                    for g, kw in gains.items()}
        out.write(json.dumps(row) + "\n")
    out.flush()
    print(f"guard {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
