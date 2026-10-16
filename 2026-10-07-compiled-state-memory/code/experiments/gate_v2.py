"""Phase-1 sealed gate (v2): does compiled GDN state carry user-bound facts / preferences?

Sealed items: statelm/synth_v2.py (48 facts, 48 preferences), disjoint from the P0/P0b dev items.
All memories use the neutral FILLER2 bank. Default injection only (scale 1, all GDN layers,
conv reset, positions from 0). One prefill per memory serves C (GDN snapshot), M (continue the
full cache) and K (fork, zero GDN). Every condition is scored with score_multi; per-token
log-probs are stored so first-divergent-token and summed shifts can both be computed.

Blocks
  facts_raw    single mention (user -> X | user -> Y); sizes 60/250; target first/last; A C M K
  facts_bind   both values in both memories (user -> X & colleague -> Y | swapped); size 60;
               mirrored layouts (u0,c30) (c0,u30) (u59,c30) (c59,u30); A C M      [K1 block 1]
  prefs_fact   "loves {t}" | "can't stand {t}"; sizes 60/250; first/last; A C M K   [K1 block 2]
  prefs_instr  "Always favour {t} ..." | "Always avoid {t} ..."; size 60; first/last; A C M
  guard        40 general questions (gold vs distractor) under zero state and 12 compiled memories
  lures        DRM lists (assoc / present / neutral) in neutral filler; size 60; list first/last; A C M

python experiments/gate_v2.py --model M --out /work/results/phase1/gate_v2_<tag>.jsonl [--blocks ...]
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, ".")
from statelm.core import LM  # noqa: E402
from statelm.prompts import check_template, compile_text, session_text  # noqa: E402
from statelm.synth_prefs import CAPABILITY, CAPABILITY_DISTRACTORS, LURES  # noqa: E402
from statelm.synth_prefs import CAPABILITY as _CAP  # noqa: E402,F401
from statelm.synth_v2 import FACTS2, PREF_FORMS, PREFS2, colleague, make_memory2  # noqa: E402

ALL_BLOCKS = "facts_raw,facts_bind,prefs_fact,prefs_instr,guard,lures"


def code_hash():
    h = hashlib.sha256()
    for f in ("statelm/core.py", "statelm/prompts.py", "statelm/synth_v2.py", "statelm/synth_prefs.py",
              "experiments/gate_v2.py"):
        h.update(open(f, "rb").read())
    return h.hexdigest()[:16]


def items_hash():
    return hashlib.sha256(json.dumps([FACTS2, PREFS2, PREF_FORMS]).encode()).hexdigest()[:16]


def first_divergent(a, b):
    a, b = a[0].tolist(), b[0].tolist()
    for j in range(min(len(a), len(b))):
        if a[j] != b[j]:
            return j
    return min(len(a), len(b)) - 1


class Gate:
    def __init__(self, lm, out):
        self.lm, self.out = lm, out

    def score(self, cache, sess, targets):
        return [v.tolist() for v in self.lm.score_multi(cache, sess, targets)]

    def pair(self, block, key, memX, memY, q, lead, tX, tY, meta, conds=("C", "M", "K")):
        lm = self.lm
        sess = lm.ids(session_text(lm.tok, q, lead))
        targets = [lm.ids(tX), lm.ids(tY)]
        row = dict(block=block, key=key, j=first_divergent(*targets), n_tX=targets[0].shape[1],
                   n_tY=targets[1].shape[1], n_sess=sess.shape[1], **meta)
        row["A"] = self.score(lm.cache_from_state(None), sess, targets)
        for tag, mem in (("X", memX), ("Y", memY)):
            ids = lm.ids(compile_text(lm.tok, mem))
            full, st = lm.prefill_full(ids)
            row[f"n_mem_{tag}"], row[f"sha_{tag}"] = ids.shape[1], st.text_sha
            if "C" in conds:
                row[f"C_{tag}"] = self.score(lm.cache_from_state(st), sess, targets)
            if "M" in conds:
                row[f"M_{tag}"] = self.score(copy.deepcopy(full), sess, targets)
            if "K" in conds:
                row[f"K_{tag}"] = self.score(lm.zero_gdn(full), sess, targets)
            del full
        self.out.write(json.dumps(row) + "\n")
        self.out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default=ALL_BLOCKS)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N items per block")
    args = ap.parse_args()
    if args.limit:
        FACTS2[:] = FACTS2[:args.limit]
        PREFS2[:] = PREFS2[:args.limit]
    blocks = args.blocks.split(",")
    torch.manual_seed(42)
    lm = LM(args.model)
    check_template(lm.tok)
    import fla
    import transformers
    header = dict(block="_header", model=args.model, code=code_hash(), items=items_hash(),
                  torch=torch.__version__, transformers=transformers.__version__, fla=fla.__version__,
                  chunk=8192, state_dtype=str(lm.state_dtype), logits="fp32", model_dtype=str(lm.model_dtype))
    if os.path.exists(args.out):
        first = json.loads(open(args.out).readline())
        if {k: first.get(k) for k in header} != header:
            sys.exit(f"refusing to append: header differs\n{first}\n{header}")
        done = {(r["block"], str(r["key"]), json.dumps({k: r.get(k) for k in ("size", "pos", "layout", "form", "kind")}))
                for r in map(json.loads, open(args.out)) if r["block"] != "_header"}
        out = open(args.out, "a")
    else:
        done = set()
        out = open(args.out, "w")
        out.write(json.dumps(header) + "\n")
    g = Gate(lm, out)
    t0 = time.time()

    def todo(block, key, meta):
        return (block, str(key), json.dumps({k: meta.get(k) for k in ("size", "pos", "layout", "form", "kind")})) not in done

    if "facts_raw" in blocks:
        for fi, (key, tmpl, q, lead, X, Y) in enumerate(FACTS2):
            for n in (60, 250):
                for pos, idx in (("first", 0), ("last", n - 1)):
                    meta = dict(size=n, pos=pos)
                    if todo("facts_raw", key, meta):
                        seed = 20000 + 100 * fi + n
                        g.pair("facts_raw", key, make_memory2({idx: tmpl.format(v=X)}, n, seed),
                               make_memory2({idx: tmpl.format(v=Y)}, n, seed), q, lead, " " + X, " " + Y, meta)
            print(f"facts_raw {fi + 1}/48 {time.time() - t0:.0f}s", flush=True)

    if "facts_bind" in blocks:
        layouts = {"u0_c30": (0, 30), "c0_u30": (30, 0), "u59_c30": (59, 30), "c59_u30": (30, 59)}
        for fi, (key, tmpl, q, lead, X, Y) in enumerate(FACTS2):
            for lname, (iu, ic) in layouts.items():
                meta = dict(size=60, layout=lname)
                if todo("facts_bind", key, meta):
                    seed = 30000 + 100 * fi
                    memX = make_memory2({iu: tmpl.format(v=X), ic: colleague(tmpl.format(v=Y))}, 60, seed)
                    memY = make_memory2({iu: tmpl.format(v=Y), ic: colleague(tmpl.format(v=X))}, 60, seed)
                    assert sorted(memX.split()) == sorted(memY.split())
                    g.pair("facts_bind", key, memX, memY, q, lead, " " + X, " " + Y, meta, conds=("C", "M"))
            print(f"facts_bind {fi + 1}/48 {time.time() - t0:.0f}s", flush=True)

    for form, block, sizes, conds in (("fact", "prefs_fact", (60, 250), ("C", "M", "K")),
                                      ("instruction", "prefs_instr", (60,), ("C", "M"))):
        if block not in blocks:
            continue
        sX_t, sY_t = PREF_FORMS[form]
        for pi, (key, thing, q, cX, cY) in enumerate(PREFS2):
            for n in sizes:
                for pos, idx in (("first", 0), ("last", n - 1)):
                    meta = dict(size=n, pos=pos, form=form)
                    if todo(block, key, meta):
                        seed = 40000 + 100 * pi + n
                        g.pair(block, key, make_memory2({idx: sX_t.format(t=thing)}, n, seed),
                               make_memory2({idx: sY_t.format(t=thing)}, n, seed), q, "", cX, cY, meta, conds=conds)
            print(f"{block} {pi + 1}/48 {time.time() - t0:.0f}s", flush=True)

    if "guard" in blocks and todo("guard", "all", {}):
        mems = []
        for i in range(6):
            key, tmpl, q, lead, X, Y = FACTS2[i * 8]
            n, idx = (60, 0) if i % 2 == 0 else (250, 249)
            mems.append(("fact:" + key, make_memory2({idx: tmpl.format(v=X)}, n, 50000 + i)))
            key, thing, *_ = PREFS2[i * 8]
            mems.append(("pref:" + key, make_memory2({idx: PREF_FORMS["fact"][0].format(t=thing)}, n, 50100 + i)))
        states = [(name, lm.compile(lm.ids(compile_text(lm.tok, m)))) for name, m in mems]
        for qi, ((q, lead, gold), dis) in enumerate(zip(CAPABILITY, CAPABILITY_DISTRACTORS)):
            sess = lm.ids(session_text(lm.tok, q, lead))
            targets = [lm.ids(gold), lm.ids(dis)]
            row = dict(block="guard", key=qi, j=first_divergent(*targets),
                       A=g.score(lm.cache_from_state(None), sess, targets))
            row["C"] = {name: g.score(lm.cache_from_state(st), sess, targets) for name, st in states}
            out.write(json.dumps(row) + "\n")
        out.flush()
        print(f"guard {time.time() - t0:.0f}s", flush=True)

    if "lures" in blocks:
        yes, no = lm.ids(" Yes"), lm.ids(" No")
        for li, (lure, assoc, neutral) in enumerate(LURES):
            words = assoc.split(", ")
            present = ", ".join(words[:5] + [lure] + words[5:9])
            for kind, lst in (("assoc", assoc), ("present", present), ("neutral", neutral)):
                for pos, idx in (("first", 0), ("last", 59)):
                    meta = dict(size=60, pos=pos, kind=kind)
                    if not todo("lures", lure, meta):
                        continue
                    mem = make_memory2({idx: f"The user wrote down this list of words: {lst}."}, 60, 60000 + li)
                    q = f"Did the list of words I wrote down include the word '{lure}'? Answer Yes or No."
                    ids = lm.ids(compile_text(lm.tok, mem))
                    full, st = lm.prefill_full(ids)
                    sess = lm.ids(session_text(lm.tok, q, "Answer:"))
                    row = dict(block="lures", key=lure, **meta, A=g.score(lm.cache_from_state(None), sess, [yes, no]),
                               C=g.score(lm.cache_from_state(st), sess, [yes, no]), M=g.score(full, sess, [yes, no]))
                    out.write(json.dumps(row) + "\n")
            out.flush()
        print(f"lures {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
