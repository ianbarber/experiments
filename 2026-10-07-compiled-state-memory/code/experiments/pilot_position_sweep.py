"""Pilot: counterfactual position sweep (does a compiled GDN state carry a specific fact?).

For each (attribute, value pair X/Y, memory size, target position) we compile two memories that
differ only in the target value, and score both values under each state:
    d(state) = log p(X | state, q) - log p(Y | state, q)          (sum over answer tokens)
    shift    = d(state_X) - d(state_Y)                             (> 0 iff the state carries the value)
Conditions: A (zero state), C (compiled state only), M (memory text in context, full KV).
Optional dilution: S tokens of an unrelated prior exchange between the state and the question.

python experiments/pilot_position_sweep.py --model M --out /work/results/pilot/pos_sweep_q35-4b.jsonl
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from statelm.core import LM  # noqa: E402
from statelm.prompts import compile_text, session_text  # noqa: E402
from statelm.synth import ATTRS, make_memory  # noqa: E402

SIZES = [20, 60, 250]
FRACS_FROM_END = [0.0, 0.1, 0.25, 0.5, 1.0]
DILUTION = [0, 256, 1024]   # only for size 60, fracs {0, 0.5}


def filler_exchange(tok, prose: str, n_tokens: int) -> str:
    if n_tokens == 0:
        return ""
    ids = tok.encode(prose, add_special_tokens=False)[:n_tokens]
    body = tok.decode(ids)
    return (f"<|im_start|>user\nCan you tidy up the formatting of this text?\n{body}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\nDone - the formatting is tidied.<|im_end|>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs_per_attr", type=int, default=2)
    args = ap.parse_args()
    torch.manual_seed(42)
    lm = LM(args.model)
    prose = open("/work/data/raw/text/qwen38_model_card.md").read()
    out = open(args.out, "w")
    t0 = time.time()

    def score_pair(cache_fn, prompt_ids, xa, ya):
        lx = lm.score(cache_fn(), prompt_ids, xa).sum().item()
        ly = lm.score(cache_fn(), prompt_ids, ya).sum().item()
        return lx, ly

    for ai, (key, tmpl, q, lead, pool) in enumerate(ATTRS):
        for p in range(args.pairs_per_attr):
            X, Y = pool[2 * p], pool[2 * p + 1]
            xa, ya = lm.ids(" " + X), lm.ids(" " + Y)
            for n in SIZES:
                for f in FRACS_FROM_END:
                    idx = round((1 - f) * (n - 1))
                    dils = DILUTION if (n == 60 and f in (0.0, 0.5)) else [0]
                    seed = 1000 * ai + 10 * p + n
                    memX = compile_text(lm.tok, make_memory(key, X, n, idx, seed=seed))
                    memY = compile_text(lm.tok, make_memory(key, Y, n, idx, seed=seed))
                    idsX, idsY = lm.ids(memX), lm.ids(memY)
                    stX, stY = lm.compile(idsX), lm.compile(idsY)
                    # tokens from the end of the compiled text to the end of the target value
                    tail = memX.split(tmpl.format(v=X), 1)[1]
                    d_tok = len(lm.tok.encode(tail, add_special_tokens=False))
                    for S in dils:
                        sess = filler_exchange(lm.tok, prose, S) + session_text(lm.tok, q, lead)
                        sids = lm.ids(sess)
                        rec = dict(attr=key, X=X, Y=Y, n_entries=n, frac_from_end=f, target_index=idx,
                                   n_mem_tokens=idsX.shape[1], tokens_after_target=d_tok, dilution=S,
                                   n_sess_tokens=sids.shape[1])
                        rec["A"] = score_pair(lambda: lm.cache_from_state(None), sids, xa, ya)
                        rec["C_X"] = score_pair(lambda: lm.cache_from_state(stX), sids, xa, ya)
                        rec["C_Y"] = score_pair(lambda: lm.cache_from_state(stY), sids, xa, ya)
                        if S == 0:
                            rec["M_X"] = score_pair(lambda: lm.cache_from_state(None),
                                                    torch.cat([idsX, sids], 1), xa, ya)
                            rec["M_Y"] = score_pair(lambda: lm.cache_from_state(None),
                                                    torch.cat([idsY, sids], 1), xa, ya)
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
            print(f"{key} pair {p} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
