"""Dev-only follow-up to the retrieval-decision pilot: greedy generation of the first action
(exploratory; same dev items, stores and seeds as retrieval_pilot.py).

The pilot scored only log p(<tool_call>) at the first assistant token. The model often opens with
a preamble ("Let me check your memory ...") before calling, so a first-token score can miss calls
and misread level shifts. Here each session is decoded greedily: up to PRE_MAX tokens before a
<tool_call> (stop at <|im_end|> / EOS), and if a call starts, up to CALL_MAX more tokens or until
the first parameter closes, so the generated function and free-text query are recorded.
Extra condition C0: compiled state of the seed's 60 filler lines with no facts (same seed, so
the fact slots keep their filler line): a per-seed control for "any compiled memory-like state"
vs this store's content.
Conditions: A, Aw (per question); C0 (per seed); C, Cw, T, CT, M (per store), as in the pilot.

python experiments/retrieval_gen.py --model M --out /work/results/phase1/retrieval_gen_<tag>.jsonl
"""
import argparse
import copy
import json
import re
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from retrieval_pilot import AWARE, NEUTRAL, N, SEEDS, family_items, session, stores  # noqa: E402
from statelm.prompts import compile_text  # noqa: E402
from statelm.synth_v2 import make_memory2  # noqa: E402

PRE_MAX, CALL_MAX = 32, 28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N seeds and 2 questions per family")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from statelm.core import LM
    torch.manual_seed(42)
    lm = LM(args.model)
    tok = lm.tok
    TC = tok.convert_tokens_to_ids("<tool_call>")
    STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}

    def step(cache, tid):
        h = lm.forward(torch.tensor([[tid]], device=lm.device), cache)
        return lm.logits(h[:, -1])[0]

    def run(cache, sess_str):
        ids = tok.encode(sess_str, add_special_tokens=False)
        h = lm.forward(torch.tensor([ids], device=lm.device), cache)
        logits = lm.logits(h[:, -1])[0]
        row = dict(lp_call=F.log_softmax(logits, -1)[TC].item())
        pre, call = [], []
        for _ in range(PRE_MAX):
            t = int(logits.argmax())
            if t == TC or t in STOP:
                break
            pre.append(t)
            logits = step(cache, t)
        else:
            t = None
        row.update(called=t == TC, n_pre=len(pre), pre=tok.decode(pre))
        if t == TC:
            logits = step(cache, t)
            for _ in range(CALL_MAX):
                t = int(logits.argmax())
                if t in STOP:
                    break
                call.append(t)
                if "</parameter>" in tok.decode(call) or "</function>" in tok.decode(call):
                    break
                logits = step(cache, t)
            txt = tok.decode(call)
            fn = re.search(r"<function=([a-z_]+)>", txt)
            q = re.search(r"<parameter=\w+>\n?(.*?)(\n?</parameter>|$)", txt, re.S)
            row.update(call=txt, fn=fn.group(1) if fn else None, arg=q.group(1).strip() if q else None)
        return row

    out = open(args.out, "w")
    out.write(json.dumps(dict(block="_header", model=args.model, n=N, seeds=SEEDS, pre_max=PRE_MAX,
                              call_max=CALL_MAX, decoding="greedy")) + "\n")
    t0 = time.time()
    seeds = range(args.limit or SEEDS)
    for fam, items in family_items().items():
        items_q = items[:2] if args.limit else items
        for _, key, _, q, kw in items_q:
            for cond, system in (("A", NEUTRAL), ("Aw", AWARE)):
                r = run(lm.cache_from_state(None), session(tok, system, q))
                out.write(json.dumps(dict(fam=fam, key=key, kw=kw, cond=cond, seed=None, store=None, present=None, **r)) + "\n")
        for s in seeds:
            seed = 99000 + (1000 if fam == "F" else 0) + s
            c0 = lm.compile(lm.ids(compile_text(tok, make_memory2({}, N, seed))))
            for _, key, _, q, kw in items_q:
                r = run(lm.cache_from_state(c0), session(tok, NEUTRAL, q))
                out.write(json.dumps(dict(fam=fam, key=key, kw=kw, cond="C0", seed=s, store=None, present=None, **r)) + "\n")
            del c0
            for si, st in enumerate(stores(fam, items, s)):
                ids = lm.ids(compile_text(tok, st["memory"]))
                full, cst = lm.prefill_full(ids)
                dig = NEUTRAL + "\n\nTopics in your memory files: " + ", ".join(st["digest"]) + "."
                for _, key, _, q, kw in items_q:
                    meta = dict(fam=fam, key=key, kw=kw, seed=s, store=si, present=key in st["keys"])
                    for cond, cache, system in (("C", lm.cache_from_state(cst), NEUTRAL),
                                                ("Cw", lm.cache_from_state(cst), AWARE),
                                                ("T", lm.cache_from_state(None), dig),
                                                ("CT", lm.cache_from_state(cst), dig),
                                                ("M", copy.deepcopy(full), NEUTRAL)):
                        r = run(cache, session(tok, system, q))
                        out.write(json.dumps(dict(cond=cond, **meta, **r)) + "\n")
                del full, cst
                out.flush()
            print(f"{fam} seed {s} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
