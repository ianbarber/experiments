"""Dev-only retrieval-decision pilot: does a compiled-state hint change *whether* and *where* an
agent with memory tools looks? (Exploratory, likelihood-only; dev items only. Calibrates a later
pre-registered hint-vs-access study; nothing here is sealed.)

Setting: the session's system turn declares three memory tools (list files / search / read file;
one file per past session, named by date). We score only the agent's *first action*:
  lp_call   log p(<tool_call>) at the first assistant token (vs answering directly)
  lp_fn     given a call, log p of each function name (search / list / read)
  lp_kw     given a search call, log p of each candidate query (the family's 12 topic keywords,
            full token sequence + the closing newline, so "s" is not credit for "sister")
Families (12 items each; a store holds 6 of them as single entries among 60 neutral entries):
  P  implicit need: synth_prefs.PREFS fact form ("The user is vegetarian.") + its question
     ("Suggest one dinner idea for tonight."), which does not mention memory. diet and spice
     always share a store (both food).
  F  explicit: synth.ATTRS (value pool[0]) + its question ("What is my dog's name?").
Per seed, the 12 items are split into two complementary stores with the same filler and slots,
so every question is scored once with its fact present and once absent, against stores of equal
size and structure. 8 seeds per family.
Conditions (store-independent: A, Aw; the rest per store):
  A   zero state, neutral prompt            Aw  zero state, hint-aware prompt (prompt-only control)
  C   compiled state, neutral prompt        Cw  compiled state, hint-aware prompt
  T   zero state, neutral prompt + topic digest of the store ("Topics in your memory files: ...")
  CT  compiled state + topic digest         M   whole store as text in context (read-all ceiling)

python experiments/retrieval_pilot.py --model M --out /work/results/phase1/retrieval_pilot_<tag>.jsonl
python experiments/retrieval_pilot.py --show       # print stores and one session (tokenizer only)
"""
import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, ".")
from statelm.prompts import compile_text  # noqa: E402
from statelm.synth import ATTRS  # noqa: E402
from statelm.synth_prefs import PREFS  # noqa: E402
from statelm.synth_v2 import make_memory2  # noqa: E402

N, SEEDS = 60, 8
NEUTRAL = ("You are a helpful assistant. You have memory files from past conversations with this user "
           "(one file per session, named by date). Use the memory tools when something the user told you "
           "before could matter.")
AWARE = NEUTRAL + (" Before this conversation, your memory files were also read into your internal state. "
                   "That can give you a sense of what you know about the user, but not reliable details: when "
                   "something seems familiar, check the files before relying on it.")
TOOLS = [
    {"type": "function", "function": {"name": "list_memory", "description": "List the memory files (one per past session, named by date).",
                                      "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "search_memory", "description": "Search all memory files for a word or phrase. Returns matching lines with their file names.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Word or phrase to search for."}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "read_memory", "description": "Read one memory file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File name, e.g. 2026-03-14.md"}}, "required": ["path"]}}},
]
FNS = ("search_memory", "list_memory", "read_memory")
KW_P = {"diet": "diet", "holiday": "holiday", "verbosity": "answer length", "units": "units", "music": "music",
        "pet": "pet", "reading": "books", "exercise": "exercise", "budget": "budget", "tone": "tone",
        "drink": "drink", "spice": "spicy food"}
KW_F = {"dog": "dog", "city": "city", "employer": "work", "cuisine": "cuisine", "allergy": "allergy",
        "sister": "sister", "car": "car", "instrument": "instrument", "sport": "sport", "drink": "coffee",
        "book": "novel", "hometown": "grew up"}


def family_items():
    """[(family, key, store sentence, question, keyword)]"""
    P = [("P", k, sX, q, KW_P[k]) for k, sX, sY, q, cX, cY in PREFS]
    F = [("F", k, tmpl.format(v=pool[0]), q, KW_F[k]) for k, tmpl, q, lead, pool in ATTRS]
    return {"P": P, "F": F}


def stores(fam, items, s):
    """Two complementary stores for seed s: same filler seed and slots, items split 6 / 6."""
    seed = 99000 + (1000 if fam == "F" else 0) + s
    rng = random.Random(seed)
    keys = [it[1] for it in items]
    if fam == "P":  # diet and spice share a store
        single = [k for k in keys if k not in ("diet", "spice")]
        rng.shuffle(single)
        halves = [["diet", "spice"] + single[:4], single[4:]]
        rng.shuffle(halves)
    else:
        rng.shuffle(keys)
        halves = [keys[:6], keys[6:]]
    slots = sorted(rng.sample(range(N), 6))
    sent = {it[1]: it[2] for it in items}
    kw = {it[1]: it[4] for it in items}
    out = []
    for h in halves:
        order = h[:]
        rng.shuffle(order)
        mem = make_memory2({sl: sent[k] for sl, k in zip(slots, order)}, N, seed)
        digest = [kw[k] for k in sorted(h, key=lambda k: rng.random())]
        out.append(dict(keys=set(h), memory=mem, digest=digest))
    return out


def session(tok, system, question):
    s = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": question}],
                                tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert "Reasoning effort" not in s and s.startswith("<|im_start|>system\n# Tools"), s[:200]
    assert s.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), s[-80:]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N seeds and 2 questions per family")
    args = ap.parse_args()
    fams = family_items()

    if args.show:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model or os.path.expandvars("$STATELM_WEIGHTS/Qwen3.8-27B"))
        for fam, items in fams.items():
            for st in stores(fam, items, 0):
                print(f"--- {fam} store keys {sorted(st['keys'])} digest {st['digest']}")
                print("\n".join(l for l in st["memory"].split("\n") if not any(
                    w in l for w in ("asked", "shared", "finished", "said their", "meeting", "mentioned"))))
        s = session(tok, NEUTRAL + "\n\nTopics in your memory files: diet, music", fams["P"][0][3])
        print(s, "\n", len(tok.encode(s)), "tokens")
        return

    import torch
    import torch.nn.functional as F
    from statelm.core import LM
    torch.manual_seed(42)
    lm = LM(args.model)
    tok = lm.tok
    TC = tok.convert_tokens_to_ids("<tool_call>")

    def toks(s):
        return tok.encode(s, add_special_tokens=False)

    def branch(cache, prefix, suffixes):
        """Run prefix ids on cache (mutated), then score each suffix (list of ids) from there.
        Returns log p(prefix tokens after the first | ...) is not needed; returns
        (next-token log-dist after prefix, [sum log p(suffix_i)])."""
        h = lm.forward(torch.tensor([prefix], device=lm.device), cache)
        dist = F.log_softmax(lm.logits(h[:, -1]), dim=-1)[0]
        out = []
        for sfx in suffixes:
            lp = dist[sfx[0]].item()
            if len(sfx) > 1:
                c = copy.deepcopy(cache)
                h2 = lm.forward(torch.tensor([sfx[:-1]], device=lm.device), c)
                lp += F.log_softmax(lm.logits(h2), dim=-1)[0].gather(
                    -1, torch.tensor(sfx[1:], device=lm.device)[:, None]).sum().item()
            out.append(lp)
        return dist, out

    def split(prefix_str, full_strs):
        """Token ids of prefix and of each full string's remainder; asserts the prefix tokenises
        identically inside every full string."""
        p = toks(prefix_str)
        rest = []
        for s in full_strs:
            t = toks(s)
            assert t[:len(p)] == p, (prefix_str, s)
            rest.append(t[len(p):])
        return p, rest

    def score(cache, sess_str, kws):
        sess = toks(sess_str)
        h = lm.forward(torch.tensor([sess], device=lm.device), cache)
        dist = F.log_softmax(lm.logits(h[:, -1]), dim=-1)[0]
        top = [tok.decode([i]) for i in dist.topk(5).indices.tolist()]
        row = dict(lp_call=dist[TC].item(), top5=top)
        # function choice, conditional on <tool_call>
        fn_full = [f"<tool_call>\n<function={f}>" for f in FNS]
        p, rest = split("<tool_call>", fn_full)
        c1 = copy.deepcopy(cache)
        _, lfn = branch(c1, p, rest)
        row["lp_fn"] = dict(zip(FNS, lfn))
        # query choice, conditional on a search call with the query parameter open
        pre = "<tool_call>\n<function=search_memory>\n<parameter=query>\n"
        p, rest = split(pre, [pre + k + "\n</parameter>" for k in kws])
        rest = [r[:r.index(toks("\n</parameter>")[0]) + 1] if toks("\n</parameter>")[0] in r else r for r in rest]
        _, lkw = branch(cache, p, rest)
        row["lp_kw"] = dict(zip(kws, lkw))
        return row

    out = open(args.out, "w")
    out.write(json.dumps(dict(block="_header", model=args.model, n=N, seeds=SEEDS, neutral=NEUTRAL, aware=AWARE,
                              tools=TOOLS, kw_p=KW_P, kw_f=KW_F)) + "\n")
    t0 = time.time()
    seeds = range(args.limit or SEEDS)
    for fam, items in fams.items():
        if args.limit:
            items_q = items[:2]
        else:
            items_q = items
        kws = [it[4] for it in items]
        for _, key, _, q, kw in items_q:  # store-independent conditions
            for cond, system in (("A", NEUTRAL), ("Aw", AWARE)):
                r = score(lm.cache_from_state(None), session(tok, system, q), kws)
                out.write(json.dumps(dict(fam=fam, key=key, kw=kw, cond=cond, seed=None, store=None, present=None, **r)) + "\n")
        for s in seeds:
            for si, st in enumerate(stores(fam, items, s)):
                ids = lm.ids(compile_text(tok, st["memory"]))
                full, cst = lm.prefill_full(ids)
                dig = NEUTRAL + "\n\nTopics in your memory files: " + ", ".join(st["digest"]) + "."
                for _, key, _, q, kw in items_q:
                    meta = dict(fam=fam, key=key, kw=kw, seed=s, store=si, present=key in st["keys"], n_mem=ids.shape[1])
                    for cond, cache, system in (("C", lm.cache_from_state(cst), NEUTRAL),
                                                ("Cw", lm.cache_from_state(cst), AWARE),
                                                ("T", lm.cache_from_state(None), dig),
                                                ("CT", lm.cache_from_state(cst), dig),
                                                ("M", copy.deepcopy(full), NEUTRAL)):
                        r = score(cache, session(tok, system, q), kws)
                        out.write(json.dumps(dict(cond=cond, **meta, **r)) + "\n")
                del full, cst
                out.flush()
            print(f"{fam} seed {s} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
