"""LongMemEval_S hint check (PREREG.md §H1): on real conversation histories, does the compiled GDN
state tell an agent with memory tools *whether* the evidence it needs is in its memory?

Per question (statelm/lme.py builds the stores): the history as a store of files, one per session,
in date order; present / absent (evidence replaced in place by length-matched personal sessions
from other histories) and, for SSU and SSP, absent2 (an independent replacement draw: placebo
swap). All versions are compiled from one shared prefill of the tokens before the first evidence
file. The session declares the memory tools (Qwen3.8 native format), a neutral system prompt with
today's date, and the question. Conditions:
  A     zero state                         S     donor state (nearest earlier question in run order
                                                 whose present store holds none of this question's
                                                 evidence; question 1 takes one from the end)
  C     compiled state (p / a / a2)        T     zero state + keyword index of the store (p / a / a2)
  CT    compiled state + keyword index     M     whole store in context (p / a)
Recorded per session: log p(<tool_call>) and log p(anything else) at the first assistant token
(primary), the top-20 first tokens, and a greedy first action: up to PRE_MAX tokens before a call
(log p(<tool_call>) and log p(<|im_end|>) at every step), stop reason (call / eot / cap), then up
to CALL_MAX tokens of the call (function and argument parsed).
Anchor (secondary; SSU / KU, short non-yes/no gold): Σ log p(gold | question + "Answer with a
short phrase only.") under A, C (p / a), M (p / a), no tools.

python experiments/lme_hint.py --preflight                      # CPU: build everything, no model
python experiments/lme_hint.py --model M --out /work/results/phase2/lme_hint_<tag>.jsonl
python experiments/lme_hint.py --model M --out ... --resume     # after a crash: append, skip done
python experiments/lme_hint.py --model M --smoke --out ...      # the 2 registered smoke questions
"""
import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, ".")
from statelm import lme  # noqa: E402
from statelm.prompts import compile_text, session_text  # noqa: E402

PRE_MAX, CALL_MAX, TOPK = 96, 40, 20
ANCHOR_SUFFIX = "\nAnswer with a short phrase only."
DATA = "/work/data/raw/longmemeval/longmemeval_s_cleaned.json"
PLACEBO_TYPES = ("single-session-user", "single-session-preference")
DONOR_BACK = 4


def tool_session(tok, system, question):
    s = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": question}],
                                tools=lme.TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert "Reasoning effort" not in s and s.startswith("<|im_start|>system\n# Tools"), s[:200]
    assert s.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), s[-80:]
    return s


def encode_all(tok, texts, char):
    """Token ids of each compile text and a common split index b: every text tokenises identically
    before b and a token starts at character `char` in each (None if there is no such index).
    Also returns the offsets of the first text."""
    encs = [tok(t, add_special_tokens=False, return_offsets_mapping=True) for t in texts]
    assert all(t[:char] == texts[0][:char] for t in texts)
    bs = [next((i for i, (s, _) in enumerate(e["offset_mapping"]) if s == char), None) for e in encs]
    ids = [e["input_ids"] for e in encs]
    ok = bs[0] is not None and all(b == bs[0] for b in bs) and all(i[:bs[0]] == ids[0][:bs[0]] for i in ids)
    return ids, (bs[0] if ok else None), encs[0]["offset_mapping"]


class Plan:
    """Everything a question needs that does not need the model (also used by --preflight)."""

    def __init__(self, x, pool, byid, tok, idf):
        st = lme.stores(x, pool, byid)
        self.x, self.st = x, st
        names, ev = st["names"], st["ev"]
        self.versions = {"p": st["present"], "a": st["absent"]}
        if x["question_type"] in PLACEBO_TYPES and st["absent2"] is not None:
            self.versions["a2"] = st["absent2"]
        texts = {k: compile_text(tok, lme.memory_text(names, v)) for k, v in self.versions.items()}
        keys = list(texts)
        char = texts["p"].index(f"### {names[ev[0]]}\n")
        ids, self.b, offs = encode_all(tok, [texts[k] for k in keys], char)
        self.ids = dict(zip(keys, ids))
        starts = [texts["p"].index(f"### {names[i]}\n") for i in ev]
        n = len(self.ids["p"])
        self.ev_tok_from_end = [n - next(j for j, (s, _) in enumerate(offs) if s >= c) for c in starts]
        self.index = {k: lme.keyword_index(names, v, idf) for k, v in self.versions.items()}
        q, date = x["question"], x["question_date"]
        self.plain = tool_session(tok, lme.system_prompt(date), q)
        self.with_idx = {k: tool_session(tok, lme.system_prompt(date, self.index[k]), q) for k in self.versions}
        self.gold = lme.short_answer(x, tok)
        self.anchor_text = session_text(tok, q + ANCHOR_SUFFIX)
        self.ev_bodies = {st["present"][i] for i in ev}
        self.bodies_p = set(st["present"])


def donors(plans, qids):
    """Donor for question i: nearest earlier question in run order (cyclic for the first) whose
    present store contains none of question i's evidence bodies."""
    out = {}
    for i, q in enumerate(qids):
        for k in range(1, len(qids)):
            j = (i - k) % len(qids)
            if not (plans[qids[j]].bodies_p & plans[q].ev_bodies):
                out[q] = qids[j]
                break
    return out


def expected_keys(plan):
    keys = {("decide", c, s) for c in ("C", "T", "CT", "M") for s in ("p", "a")} | {("decide", "A", None), ("decide", "S", None)}
    if "a2" in plan.versions:
        keys |= {("decide", "C", "a2"), ("decide", "T", "a2")}
    if plan.gold:
        keys |= {("anchor", "A", None)} | {("anchor", c, s) for c in ("C", "M") for s in ("p", "a")}
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/statelm-weights/Qwen3.8-27B"))
    ap.add_argument("--out")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--smoke", action="store_true", help="run the 2 registered smoke questions instead")
    ap.add_argument("--resume", action="store_true", help="append to --out, skipping complete questions")
    ap.add_argument("--preflight", action="store_true", help="CPU only: build every question's inputs")
    ap.add_argument("--qids", default="", help="plumbing tests only: comma-separated frame question ids")
    args = ap.parse_args()
    data = lme.load(args.data)
    byid = {x["question_id"]: x for x in data}
    pool = lme.build_pool(data)
    idf = lme.build_idf(data)
    picked, smoke = lme.sample(data, pool, byid)
    qids = args.qids.split(",") if args.qids else smoke if args.smoke else picked
    assert not (args.qids and set(qids) & set(picked)), "plumbing ids must be outside the sample"

    if args.preflight:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        plans = {}
        for q in picked + smoke:
            p = plans[q] = Plan(byid[q], pool, byid, tok, idf)
            print(f"{q} {p.x['question_type'][:22]:22s} tokens {len(p.ids['p'])} versions {list(p.versions)} "
                  f"shared {p.b} ev_from_end {p.ev_tok_from_end} gold {p.gold!r} T-session "
                  f"{len(tok.encode(p.with_idx['p']))}")
            assert p.b, q
        for name, ids in (("sample", picked), ("smoke", smoke)):
            dn = donors(plans, ids)
            back = [(ids.index(q) - ids.index(dn[q])) % len(ids) for q in ids]
            print(name, "donor distances", sorted(back)[-5:], "max", max(back))
            assert len(dn) == len(ids) and max(back) <= DONOR_BACK
        print("preflight ok:", len(picked), "sample +", len(smoke), "smoke")
        return

    import torch
    import torch.nn.functional as F
    from statelm.core import LM, CompiledState
    torch.manual_seed(42)
    lm = LM(args.model)
    tok = lm.tok
    TC = tok.convert_tokens_to_ids("<tool_call>")
    END = tok.convert_tokens_to_ids("<|im_end|>")
    STOP = {END, tok.eos_token_id}
    plans = {q: Plan(byid[q], pool, byid, tok, idf) for q in qids}
    donor = donors(plans, qids)

    def step(cache, tid):
        return lm.logits(lm.forward(torch.tensor([[tid]], device=lm.device), cache)[:, -1])[0]

    def lp_split(logits):
        lp = F.log_softmax(logits.double(), -1)
        rest = torch.cat([lp[:TC], lp[TC + 1:]])
        return lp, lp[TC].item(), torch.logsumexp(rest, 0).item()

    def run(cache, sess_str):
        ids = tok.encode(sess_str, add_special_tokens=False)
        logits = lm.logits(lm.forward(torch.tensor([ids], device=lm.device), cache)[:, -1])[0]
        lp, lp_call, lp_not = lp_split(logits)
        top = lp.topk(TOPK)
        row = dict(lp_call=lp_call, lp_not_call=lp_not, n_sess=len(ids),
                   top=[[tok.decode([i]), round(v, 4)] for i, v in zip(top.indices.tolist(), top.values.tolist())])
        pre, path, stop = [], [], "cap"
        for _ in range(PRE_MAX):
            lp, lc, ln = lp_split(logits)
            path.append([lc, ln, lp[END].item()])
            t = int(logits.argmax())
            if t == TC:
                stop = "call"
                break
            if t in STOP:
                stop = "eot"
                break
            pre.append(t)
            logits = step(cache, t)
        row.update(stop=stop, called=stop == "call", n_pre=len(pre), pre=tok.decode(pre), path=path)
        if stop == "call":
            logits, call = step(cache, TC), []
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
            arg = re.search(r"<parameter=\w+>\n?(.*?)(\n?</parameter>|$)", txt, re.S)
            row.update(call=txt, fn=fn.group(1) if fn else None, arg=arg.group(1).strip() if arg else None)
        return row

    def anchor(cache, plan):
        lp = lm.score(cache, lm.ids(plan.anchor_text), lm.ids(plan.gold))
        return dict(lp=lp.tolist(), lp_sum=float(lp.sum()), n_tok=len(lp))

    def snapshot(cache, ids):
        rec = {i: cache.layers[i].recurrent_states.detach().clone() for i in lm.gdn_layers}
        conv = {i: cache.layers[i].conv_states.detach().clone() for i in lm.gdn_layers}
        sha = hashlib.sha256(torch.tensor(ids).numpy().tobytes()).hexdigest()[:16]
        return CompiledState(rec, conv, len(ids), sha, {})

    def fwd(ids, cache):
        lm.forward(torch.tensor([ids], device=lm.device), cache)

    def compile_all(plan, keep_kv):
        """Compile every version from the shared prefix. Returns ({version: CompiledState},
        {version: full cache} for p and a if keep_kv)."""
        b = plan.b
        base = lm.new_cache()
        fwd(plan.ids["p"][:b], base)
        cst, full = {}, {}
        keys = list(plan.versions)
        for n, k in enumerate(keys):
            c = base if n == len(keys) - 1 else copy.deepcopy(base)
            fwd(plan.ids[k][b:], c)
            cst[k] = snapshot(c, plan.ids[k])
            if keep_kv and k in ("p", "a"):
                full[k] = c
            else:
                del c
        del base
        return cst, full

    def present_state(q):
        return compile_all(plans[q], keep_kv=False)[0]["p"]

    # ------------------------------------------------------------------ resume bookkeeping
    header = dict(block="_header", model=args.model, data=args.data,
                  data_sha=hashlib.sha256(open(args.data, "rb").read()).hexdigest()[:16],
                  qids=qids, smoke=args.smoke or bool(args.qids), donors=donor, exclude=lme.EXCLUDE,
                  pre_max=PRE_MAX, call_max=CALL_MAX, neutral=lme.NEUTRAL, tools=lme.TOOLS,
                  anchor_suffix=ANCHOR_SUFFIX, decoding="greedy")
    done, rows_seen = set(), {}
    if args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        text = open(args.out).read()
        if not text.endswith("\n"):  # a half-written last line from a hard crash
            text = text[:text.rfind("\n") + 1]
            open(args.out, "w").write(text)
        prior = [json.loads(line) for line in text.splitlines() if line]
        assert prior and prior[0] == json.loads(json.dumps(header)), "resume: header differs from this run's"
        for r in prior[1:]:
            rows_seen.setdefault(r["qid"], set()).add((r["block"], r.get("cond"), r.get("store")))
        for q in qids:
            need = expected_keys(plans[q]) | {("meta", None, None)}
            if q == qids[0]:
                need.discard(("decide", "S", None))  # written at the end
            if need <= rows_seen.get(q, set()):
                done.add(q)
        out = open(args.out, "a")
        print(f"resume: {len(done)} of {len(qids)} questions complete", flush=True)
    else:
        out = open(args.out, "w")
        out.write(json.dumps(header) + "\n")
        out.flush()

    t0 = time.time()
    states = {}  # present states of recent questions (donors)
    for qi, q in enumerate(qids):
        if q in done:
            continue
        tq = time.time()
        torch.cuda.reset_peak_memory_stats()
        plan, x = plans[q], plans[q].x
        if qi > 0 and donor[q] not in states:
            states[donor[q]] = present_state(donor[q])  # only after a resume; before this question's caches exist
        cst, full = compile_all(plan, keep_kv=True)
        t_compile = time.time() - tq
        states[q] = cst["p"]
        out.write(json.dumps(dict(block="meta", qid=q, qtype=x["question_type"], question=x["question"],
                                  date=x["question_date"], gold=x["answer"], anchor_gold=plan.gold,
                                  n_files=len(plan.st["names"]), ev=plan.st["ev"],
                                  ev_names=[plan.st["names"][i] for i in plan.st["ev"]], rep=plan.st["rep"],
                                  rep2=plan.st["rep2"] if "a2" in plan.versions else None,
                                  n_tok={k: len(v) for k, v in plan.ids.items()}, shared_prefix=plan.b,
                                  ev_tok_from_end=plan.ev_tok_from_end, donor=donor[q],
                                  t_compile=round(t_compile, 1))) + "\n")

        def emit(cond, store, cache, sess):
            out.write(json.dumps(dict(block="decide", qid=q, qtype=x["question_type"], cond=cond, store=store,
                                      **run(cache, sess))) + "\n")

        def emit_anchor(cond, store, cache):
            out.write(json.dumps(dict(block="anchor", qid=q, qtype=x["question_type"], cond=cond, store=store,
                                      gold=plan.gold, **anchor(cache, plan))) + "\n")

        emit("A", None, lm.cache_from_state(None), plan.plain)
        if qi > 0:
            emit("S", None, lm.cache_from_state(states[donor[q]]), plan.plain)
        for s in plan.versions:
            emit("C", s, lm.cache_from_state(cst[s]), plan.plain)
            emit("T", s, lm.cache_from_state(None), plan.with_idx[s])
            if s != "a2":
                emit("CT", s, lm.cache_from_state(cst[s]), plan.with_idx[s])
        if plan.gold:
            emit_anchor("A", None, lm.cache_from_state(None))
            for s in ("p", "a"):
                emit_anchor("C", s, lm.cache_from_state(cst[s]))
        for s in ("p", "a"):
            if plan.gold:
                c = copy.deepcopy(full[s])
                emit_anchor("M", s, c)
                del c
            emit("M", s, full.pop(s), plan.plain)
            torch.cuda.empty_cache()
        keep = {qids[j] for j in range(qi + 1, min(qi + 1 + DONOR_BACK, len(qids)))}
        needed = {donor[k] for k in keep} | {donor[qids[0]]}
        states = {k: v for k, v in states.items() if k in needed}
        del cst
        torch.cuda.empty_cache()
        out.flush()
        print(f"{qi + 1}/{len(qids)} {q} {x['question_type']} tokens {len(plan.ids['p'])} prefix {plan.b} "
              f"compile {t_compile:.0f}s question {time.time() - tq:.0f}s total {time.time() - t0:.0f}s "
              f"peak alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
              f"reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB", flush=True)
    q0 = qids[0]
    if ("decide", "S", None) not in rows_seen.get(q0, set()):
        d = donor[q0]
        if d not in states:
            states[d] = present_state(d)
        out.write(json.dumps(dict(block="decide", qid=q0, qtype=plans[q0].x["question_type"], cond="S", store=None,
                                  **run(lm.cache_from_state(states[d]), plans[q0].plain))) + "\n")
    out.flush()
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
