"""Phase-1 gate (Qwen3.8-27B): does compiled GDN state carry user-specific facts / preferences?

Blocks (each writes JSONL rows tagged with `block`):
  facts   P0 counterfactual pairs (12 attrs x 2 pairs); sizes {60, 250}; positions first/mid/last;
          conditions A, C, M, K (K = memory attention KV kept, GDN zeroed)
  prefs   12 preference pairs stated as facts; 2 filler seeds; same sizes/positions/conditions
  dilute  facts + prefs, size 60, first & last, C with {512, 2048} tokens of unrelated exchange
  lures   DRM lists (associated / present / neutral), size 60, list at last & mid; A, C, M
  guard   40 general questions under zero state vs 4 compiled memories (size 250)
  binding lexical-priming control: both values appear in BOTH memories, one bound to the user and
          one to "the user's colleague"; roles swapped between memory X and memory Y (lexically
          identical files). Shift > 0 here requires the state to carry the user-value binding.

python experiments/gate_phase1.py --model M --out /work/results/phase1/gate_<tag>.jsonl [--blocks facts,prefs]
"""
import argparse
import json
import random
import sys
import time

import torch

sys.path.insert(0, ".")
from statelm.core import LM  # noqa: E402
from statelm.prompts import compile_text, session_text  # noqa: E402
from statelm.synth import ATTRS, make_memory, make_memory_with  # noqa: E402
from statelm.synth_prefs import CAPABILITY, LURES, PREFS  # noqa: E402

SIZES = [60, 250]


def positions(n):
    return {"first": 0, "mid": n // 2, "last": n - 1}


def filler_exchange(tok, prose, n_tokens):
    if n_tokens == 0:
        return ""
    body = tok.decode(tok.encode(prose, add_special_tokens=False)[:n_tokens])
    return (f"<|im_start|>user\nCan you tidy up the formatting of this text?\n{body}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\nDone - the formatting is tidied.<|im_end|>\n")


class Gate:
    def __init__(self, lm, out):
        self.lm, self.out = lm, out

    def pair_scores(self, mem_ids, sess_ids, targets, conds, st=None):
        lm = self.lm
        res = {}
        if "C" in conds:
            res["C"] = [v.sum().item() for v in lm.score_multi(lm.cache_from_state(st), sess_ids, targets)]
        if "M" in conds:
            res["M"] = [v.sum().item() for v in
                        lm.score_multi(lm.cache_from_state(None), torch.cat([mem_ids, sess_ids], 1), targets)]
        if "K" in conds:
            res["K"] = [v.sum().item() for v in
                        lm.score_multi(lm.kv_only_cache(mem_ids), sess_ids, targets,
                                       position_offset=mem_ids.shape[1])]
        return res

    def counterfactual(self, block, key, memX, memY, q, lead, tX, tY, meta, conds=("C", "M", "K"), dil=(0,),
                       prose=""):
        lm = self.lm
        idsX, idsY = lm.ids(compile_text(lm.tok, memX)), lm.ids(compile_text(lm.tok, memY))
        stX, stY = lm.compile(idsX), lm.compile(idsY)
        targets = [lm.ids(tX), lm.ids(tY)]
        for S in dil:
            sids = lm.ids(filler_exchange(lm.tok, prose, S) + session_text(lm.tok, q, lead))
            row = dict(block=block, key=key, n_mem_tokens=idsX.shape[1], dilution=S, n_sess=sids.shape[1], **meta)
            row["A"] = [v.sum().item() for v in lm.score_multi(lm.cache_from_state(None), sids, targets)]
            cs = conds if S == 0 else ("C",)
            rx = self.pair_scores(idsX, sids, targets, cs, stX)
            ry = self.pair_scores(idsY, sids, targets, cs, stY)
            for c in cs:
                row[f"{c}_X"], row[f"{c}_Y"] = rx[c], ry[c]
            self.out.write(json.dumps(row) + "\n")
        self.out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="facts,prefs,binding,dilute,lures,guard")
    args = ap.parse_args()
    blocks = args.blocks.split(",")
    torch.manual_seed(42)
    lm = LM(args.model)
    out = open(args.out, "a")
    g = Gate(lm, out)
    prose = open("/work/data/raw/text/qwen38_model_card.md").read()
    t0 = time.time()

    if "facts" in blocks:
        for ai, (key, tmpl, q, lead, pool) in enumerate(ATTRS):
            for p in range(2):
                X, Y = pool[2 * p], pool[2 * p + 1]
                for n in SIZES:
                    for pos, idx in positions(n).items():
                        seed = 1000 * ai + 10 * p + n
                        g.counterfactual("facts", key, make_memory(key, X, n, idx, seed), make_memory(key, Y, n, idx, seed),
                                         q, lead, " " + X, " " + Y, dict(X=X, Y=Y, n_entries=n, pos=pos, idx=idx))
            print(f"facts {key} {time.time() - t0:.0f}s", flush=True)

    if "prefs" in blocks:
        for pi, (key, sX, sY, q, cX, cY) in enumerate(PREFS):
            for rep in range(2):
                for n in SIZES:
                    for pos, idx in positions(n).items():
                        seed = 5000 + 100 * pi + 10 * rep + n
                        g.counterfactual("prefs", key, make_memory_with(sX, n, idx, seed),
                                         make_memory_with(sY, n, idx, seed), q, "", cX, cY,
                                         dict(rep=rep, n_entries=n, pos=pos, idx=idx))
            print(f"prefs {key} {time.time() - t0:.0f}s", flush=True)

    if "binding" in blocks:
        def colleague(sentence):
            return sentence.replace("The user", "The user's colleague", 1)

        def two_entry_memory(s_user, s_coll, n, i_user, i_coll, seed):
            # insert the later index first so both land where intended
            base = make_memory_with("@@A@@", n - 1, min(i_user, i_coll), seed)
            lines = base.split("\n")
            later = max(i_user, i_coll)
            lines.insert(later, "- [2026-09-30] @@B@@")
            first_s, second_s = (s_user, s_coll) if i_user < i_coll else (s_coll, s_user)
            return "\n".join(lines).replace("@@A@@", first_s).replace("@@B@@", second_s)

        layouts = {"user_first": (0, 30), "user_last": (59, 30), "user_first_coll_last": (0, 59)}
        for ai, (key, tmpl, q, lead, pool) in enumerate(ATTRS):
            for p in range(2):
                X, Y = pool[2 * p], pool[2 * p + 1]
                for lname, (iu, ic) in layouts.items():
                    seed = 1000 * ai + 10 * p + 60
                    memX = two_entry_memory(tmpl.format(v=X), colleague(tmpl.format(v=Y)), 60, iu, ic, seed)
                    memY = two_entry_memory(tmpl.format(v=Y), colleague(tmpl.format(v=X)), 60, iu, ic, seed)
                    assert sorted(memX.split()) == sorted(memY.split())
                    g.counterfactual("binding_facts", key, memX, memY, q, lead, " " + X, " " + Y,
                                     dict(X=X, Y=Y, n_entries=60, layout=lname), conds=("C", "M"))
        for pi, (key, sX, sY, q, cX, cY) in enumerate(PREFS):
            for lname, (iu, ic) in layouts.items():
                seed = 5000 + 100 * pi + 60
                memX = two_entry_memory(sX, colleague(sY), 60, iu, ic, seed)
                memY = two_entry_memory(sY, colleague(sX), 60, iu, ic, seed)
                g.counterfactual("binding_prefs", key, memX, memY, q, "", cX, cY,
                                 dict(n_entries=60, layout=lname), conds=("C", "M"))
        print(f"binding {time.time() - t0:.0f}s", flush=True)

    if "dilute" in blocks:
        for ai, (key, tmpl, q, lead, pool) in enumerate(ATTRS):
            X, Y = pool[0], pool[1]
            for pos, idx in (("first", 0), ("last", 59)):
                seed = 1000 * ai + 60
                g.counterfactual("dilute_facts", key, make_memory(key, X, 60, idx, seed),
                                 make_memory(key, Y, 60, idx, seed), q, lead, " " + X, " " + Y,
                                 dict(n_entries=60, pos=pos, idx=idx), conds=("C",), dil=(512, 2048), prose=prose)
        for pi, (key, sX, sY, q, cX, cY) in enumerate(PREFS):
            for pos, idx in (("first", 0), ("last", 59)):
                seed = 5000 + 100 * pi + 60
                g.counterfactual("dilute_prefs", key, make_memory_with(sX, 60, idx, seed),
                                 make_memory_with(sY, 60, idx, seed), q, "", cX, cY,
                                 dict(n_entries=60, pos=pos, idx=idx), conds=("C",), dil=(512, 2048), prose=prose)
        print(f"dilute {time.time() - t0:.0f}s", flush=True)

    if "lures" in blocks:
        yes, no = lm.ids(" Yes"), lm.ids(" No")
        for li, (lure, assoc, neutral) in enumerate(LURES):
            words = assoc.split(", ")
            present = ", ".join(words[:5] + [lure] + words[5:9])
            for kind, lst in (("assoc", assoc), ("present", present), ("neutral", neutral)):
                for pos, idx in (("last", 59), ("mid", 30)):
                    mem = make_memory_with(f"The user wrote down this list of words: {lst}.", 60, idx, 7000 + li)
                    q = f"Did the list of words I wrote down include the word '{lure}'? Answer Yes or No."
                    ids = lm.ids(compile_text(lm.tok, mem))
                    st = lm.compile(ids)
                    sids = lm.ids(session_text(lm.tok, q, "Answer:"))
                    row = dict(block="lures", key=lure, kind=kind, pos=pos, idx=idx)
                    row["A"] = [v.sum().item() for v in lm.score_multi(lm.cache_from_state(None), sids, [yes, no])]
                    row.update(g.pair_scores(ids, sids, [yes, no], ("C", "M"), st))
                    out.write(json.dumps(row) + "\n")
            out.flush()
        print(f"lures {time.time() - t0:.0f}s", flush=True)

    if "guard" in blocks:
        rng = random.Random(42)
        states = []
        for m in range(4):
            key, tmpl, q, lead, pool = ATTRS[m]
            states.append(lm.compile(lm.ids(compile_text(lm.tok, make_memory(key, rng.choice(pool), 250, 125, 9000 + m)))))
        for qi, (q, lead, ans) in enumerate(CAPABILITY):
            sids = lm.ids(session_text(lm.tok, q, lead))
            t = lm.ids(ans)
            row = dict(block="guard", key=qi, A=lm.score(lm.cache_from_state(None), sids, t).sum().item())
            row["C"] = [lm.score(lm.cache_from_state(st), sids, t).sum().item() for st in states]
            out.write(json.dumps(row) + "\n")
        out.flush()
        print(f"guard {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
