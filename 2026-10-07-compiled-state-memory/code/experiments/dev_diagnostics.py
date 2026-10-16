"""Dev-only mechanism diagnostics (P0/P0b items; never the sealed v2 items).

Raw counterfactual fact pairs on synth.ATTRS (12 attributes, first value pair), memory of 60
entries in the v1 format. All rows score C (and, where noted, M) with score_multi; per-token
log-probs stored. Variants:
  pos       target at start+0/1/4, mid, end-0
  preamble  ~200-token neutral preamble between the header and the list; target first / last
  neutral0  fixed neutral entry at index 0, target at index 1
  dates     dates shuffled (not sorted); target first / last
  dilute    target first/last; between state and question: structure-only exchange (~10-token
            body), 512 / 2048 tokens of prose in a separate exchange, 512 tokens in the same turn
  scale     x0.5, x2, x3 (target first/last)
  band      early / mid / late third of GDN layers only (target first/last)
  heads     nominal half-life > 256 ("long") / <= 256 ("short") only (target first/last)
  conv      conv state kept instead of reset (target first/last)
Preference polarity pairs on synth_prefs.PREFS topics (fact form, v2 neutral filler, dev topics)
run under pos(first/last) x {default, scale x2, band late}.

python experiments/dev_diagnostics.py --model M --out /work/results/phase1/dev_diag_<tag>.jsonl
"""
import argparse
import json
import math
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from statelm.core import LM  # noqa: E402
from statelm.prompts import compile_text, session_text  # noqa: E402
from statelm.synth import ATTRS, make_memory  # noqa: E402
from statelm.synth_prefs import PREFS  # noqa: E402
from statelm.synth_v2 import make_memory2  # noqa: E402

N = 60
PREAMBLE = ("These notes were collected over several months of conversations. They are listed in date "
            "order, one per line, and cover a mix of everyday topics. Some entries describe plans, some "
            "describe requests for help, and some record details the user mentioned in passing. The "
            "notes are informal and were written quickly, so the wording varies from line to line. "
            "Where an entry mentions another person, it says so explicitly. Nothing in this list has "
            "been verified independently, and some entries may have been superseded by later ones. ") * 2


def first_div(a, b):
    a, b = a[0].tolist(), b[0].tolist()
    return next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(42)
    lm = LM(args.model)
    out = open(args.out, "w")
    prose = open("/work/data/raw/text/qwen38_model_card.md").read()
    pids = lm.tok.encode(prose, add_special_tokens=False)
    t0 = time.time()

    long_mask, short_mask = {}, {}
    for i in lm.gdn_layers:
        la = lm.tm.layers[i].linear_attn
        hl = math.log(2) / (la.A_log.float().exp() * F.softplus(la.dt_bias.float())).abs()
        long_mask[i], short_mask[i] = (hl > 256).float(), (hl <= 256).float()
    nl = len(lm.gdn_layers)
    bands = {"early": lm.gdn_layers[:nl // 3], "mid": lm.gdn_layers[nl // 3:2 * nl // 3],
             "late": lm.gdn_layers[2 * nl // 3:]}

    def band_mask(b):
        return {i: (1.0 if i in bands[b] else 0.0) for i in lm.gdn_layers}

    def sess_with(q, lead, kind):
        base = session_text(lm.tok, q, lead)
        if kind == "none":
            return base
        if kind == "struct":
            return ("<|im_start|>user\nThanks for earlier.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    "You're welcome.<|im_end|>\n") + base
        if kind.startswith("sep"):
            body = lm.tok.decode(pids[:int(kind[3:])])
            return (f"<|im_start|>user\nCan you tidy up the formatting of this text?\n{body}<|im_end|>\n"
                    f"<|im_start|>assistant\n<think>\n\n</think>\n\nDone - the formatting is tidied.<|im_end|>\n") + base
        if kind.startswith("same"):
            body = lm.tok.decode(pids[:int(kind[4:])])
            return session_text(lm.tok, f"Here is some text I was reading:\n{body}\n\n{q}", lead)
        raise ValueError(kind)

    def run_pair(block, key, memX, memY, q, lead, tX, tY, variants, meta, sess_kind="none", with_M=False):
        targets = [lm.ids(tX), lm.ids(tY)]
        sess = lm.ids(sess_with(q, lead, sess_kind))
        sts = {}
        fulls = {}
        for tag, mem in (("X", memX), ("Y", memY)):
            fulls[tag], sts[tag] = lm.prefill_full(lm.ids(compile_text(lm.tok, mem)))
        rowA = [v.tolist() for v in lm.score_multi(lm.cache_from_state(None), sess, targets)]
        for vname, kw in variants:
            row = dict(block=block, key=key, variant=vname, sess=sess_kind, j=first_div(*targets), A=rowA, **meta)
            for tag in ("X", "Y"):
                row[f"C_{tag}"] = [v.tolist() for v in lm.score_multi(lm.cache_from_state(sts[tag], **kw), sess, targets)]
                if with_M and vname == "default":
                    row[f"M_{tag}"] = [v.tolist() for v in lm.score_multi(fulls[tag], sess, targets)]
            out.write(json.dumps(row) + "\n")
        out.flush()

    default = [("default", {})]
    mech = default + [("scale0.5", dict(scale=0.5)), ("scale2", dict(scale=2.0)), ("scale3", dict(scale=3.0)),
                      ("band_early", dict(layer_mask=band_mask("early"))), ("band_mid", dict(layer_mask=band_mask("mid"))),
                      ("band_late", dict(layer_mask=band_mask("late"))), ("heads_long", dict(head_mask=long_mask)),
                      ("heads_short", dict(head_mask=short_mask)), ("conv_kept", dict(keep_conv=True))]

    for ai, (key, tmpl, q, lead, pool) in enumerate(ATTRS):
        X, Y = pool[0], pool[1]
        seed = 1000 * ai + N
        mk = lambda v, idx: make_memory(key, v, N, idx, seed)  # noqa: E731
        for pos, idx in (("start+0", 0), ("start+1", 1), ("start+4", 4), ("mid", 30), ("end-0", N - 1)):
            vs = mech if pos in ("start+0", "end-0") else default
            run_pair("facts_pos", key, mk(X, idx), mk(Y, idx), q, lead, " " + X, " " + Y, vs, dict(pos=pos), with_M=True)
        for pos, idx in (("first", 0), ("last", N - 1)):
            # neutral preamble before the list
            pX, pY = PREAMBLE + "\n" + mk(X, idx), PREAMBLE + "\n" + mk(Y, idx)
            run_pair("facts_preamble", key, pX, pY, q, lead, " " + X, " " + Y, default, dict(pos=pos))
            # dates shuffled
            def shuf(m):
                lines = m.split("\n")
                dates = [ln[3:13] for ln in lines]
                random.Random(seed).shuffle(dates)
                return "\n".join(f"- [{d}]{ln[14:]}" for d, ln in zip(dates, lines))
            run_pair("facts_dates", key, shuf(mk(X, idx)), shuf(mk(Y, idx)), q, lead, " " + X, " " + Y, default, dict(pos=pos))
            for sk in ("struct", "sep512", "sep2048", "same512"):
                run_pair("facts_dilute", key, mk(X, idx), mk(Y, idx), q, lead, " " + X, " " + Y, default,
                         dict(pos=pos), sess_kind=sk)
        # neutral entry at index 0, target at index 1
        def neutral0(v):
            lines = mk(v, 1).split("\n")
            lines[0] = lines[0][:15] + "The user asked what the weather would be like tomorrow."
            return "\n".join(lines)
        run_pair("facts_neutral0", key, neutral0(X), neutral0(Y), q, lead, " " + X, " " + Y, default, dict(pos="start+1"))
        print(f"facts {key} {time.time() - t0:.0f}s", flush=True)

    pref_vars = default + [("scale2", dict(scale=2.0)), ("band_late", dict(layer_mask=band_mask("late")))]
    for pi, (key, sX, sY, q, cX, cY) in enumerate(PREFS):
        for pos, idx in (("first", 0), ("last", N - 1)):
            seed = 70000 + 100 * pi
            run_pair("prefs_dev", key, make_memory2({idx: sX}, N, seed), make_memory2({idx: sY}, N, seed),
                     q, "", cX, cY, pref_vars, dict(pos=pos), with_M=True)
        print(f"prefs {key} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
