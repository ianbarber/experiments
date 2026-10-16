"""Stage 0 (exploratory, dev; LABNOTES 2026-10-10): where does the memory's effect live?

Questions: dev16 (5 SSU, 2 SSP, 5 KU, 4 MS) drawn with random.Random(42) from the H1 frame minus the 80 H1 questions
and the plumbing ids. Stores and the tool session are as in H1 (lme_hint.py). Conditions:
  A                     no memory
  C / K / M             GDN state (conv reset) / memory KV only (GDN, conv zeroed) / full continuation
                        [every store version: p, a, a2 where buildable]
  CK                    memory KV + GDN state, conv reset                                       [p, a]
  Kstar4k / Kstar64k    memory KV with every key un-rotated to position 0; the session starts at
                        position 4,096 / 65,536 (every memory entry at one distance)            [p, a]
  Krt                   un-rotate, then re-rotate to the original positions (K* noise floor)    [p, a]
  Kearly / Kmid / Klate memory KV visible only at attention layers 3-19 / 23-43 / 47-63          [p, a]
Recorded per session: H1's first-action measures (first-token call log-odds, greedy path with
log p(call) per step, stop reason, call text, function and argument), plus a direct logit
attribution of z_call - E_q[z] (q = next-token distribution renormalised over non-call tokens) to the
embedding, the 64 mixers and the 64 MLPs at every greedy step before the stop. The per-component
contributions at the last prompt position (bf16, [129, hidden]) and the per-step attributions go to
<out>.d/<qid>.pt, with the step-0 readout vector w0 (so differences can use a common readout). Anchor (gold log-likelihood, SSU/KU short gold, no tools): A, and C, K, CK, M,
Kstar4k, Kearly, Kmid, Klate on p and a.

./run.sh python -u experiments/lme_where.py --model $Q38_27B --out /work/results/phase2/lme_where_q38-27b.jsonl
"""
import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, ".")
from experiments.lme_hint import ANCHOR_SUFFIX, CALL_MAX, DATA, PRE_MAX, TOPK, encode_all, tool_session  # noqa: E402
from statelm import lme  # noqa: E402
from statelm.prompts import compile_text, session_text  # noqa: E402

PLUMBING = ("09ba9854", "6a27ffc2", "b86304ba")
BANDS = {"Kearly": slice(0, 5), "Kmid": slice(5, 11), "Klate": slice(11, 16)}  # into lm.attn_layers
SHIFTS = {"Kstar4k": 4096, "Kstar64k": 65536}
# only 2 SSP questions remain outside the H1 sample and plumbing ids, so the other types make up 16
DEV_COUNTS = {"single-session-user": 5, "single-session-preference": 2, "knowledge-update": 5, "multi-session": 4}


def dev_sample(data, pool, byid, seed=42):
    picked, smoke = lme.sample(data, pool, byid)
    used = set(picked) | set(smoke) | set(PLUMBING)
    fr = lme.frame(data, pool, byid)
    rng = random.Random(seed)
    out = []
    for t, n in DEV_COUNTS.items():
        out += rng.sample(sorted(x["question_id"] for x in fr if x["question_type"] == t
                                 and x["question_id"] not in used), n)
    rng.shuffle(out)
    return out


class Plan:
    """Model-free inputs of one question; absent2 for every type where it can be built."""

    def __init__(self, x, pool, byid, tok):
        st = lme.stores(x, pool, byid)
        self.x, self.st = x, st
        names, ev = st["names"], st["ev"]
        self.versions = {"p": st["present"], "a": st["absent"]}
        if st["absent2"] is not None:
            self.versions["a2"] = st["absent2"]
        texts = {k: compile_text(tok, lme.memory_text(names, v)) for k, v in self.versions.items()}
        keys = list(texts)
        char = texts["p"].index(f"### {names[ev[0]]}\n")
        ids, self.b, _ = encode_all(tok, [texts[k] for k in keys], char)
        assert self.b, x["question_id"]
        self.ids = dict(zip(keys, ids))
        self.plain = tool_session(tok, lme.system_prompt(x["question_date"]), x["question"])
        self.gold = lme.short_answer(x, tok)
        self.anchor_text = session_text(tok, x["question"] + ANCHOR_SUFFIX)


def expected(plan):
    keys = {("decide", "A", None)} | {("decide", c, s) for c in ("C", "K", "M") for s in plan.versions}
    keys |= {("decide", c, s) for c in ("CK", "Krt", *SHIFTS, *BANDS) for s in ("p", "a")}
    if plan.gold:
        keys |= {("anchor", "A", None)} | {("anchor", c, s) for c in ANCHOR_CONDS for s in ("p", "a")}
    return keys


ANCHOR_CONDS = ("C", "K", "CK", "M", "Kstar4k", *BANDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expandvars("$STATELM_WEIGHTS/Qwen3.8-27B"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--qids", default="", help="plumbing only: comma-separated frame ids outside dev16")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    data = lme.load(args.data)
    byid = {x["question_id"]: x for x in data}
    pool = lme.build_pool(data)
    dev = dev_sample(data, pool, byid)
    qids = args.qids.split(",") if args.qids else dev
    assert not (args.qids and set(qids) & set(dev)), "plumbing ids must be outside dev16"

    import torch
    from statelm.core import LM, CompiledState
    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    lm = LM(args.model)
    lm.enable_kv_bias()
    tok = lm.tok
    TC = tok.convert_tokens_to_ids("<tool_call>")
    END = tok.convert_tokens_to_ids("<|im_end|>")
    STOP = {END, tok.eos_token_id}
    NEG = torch.finfo(torch.float32).min
    NL = len(lm.tm.layers)
    gnorm, eps = 1.0 + lm.tm.norm.weight.detach().float(), lm.tm.norm.eps
    plans = {q: Plan(byid[q], pool, byid, tok) for q in qids}
    trace_dir = args.out + ".d"
    os.makedirs(trace_dir, exist_ok=True)

    # ---------------------------------------------------------------- residual tracing
    buf, tracing = {}, [False]

    def keep(key):
        def hook(_m, _a, out):
            if tracing[0]:
                buf[key] = (out[0] if isinstance(out, tuple) else out)[0, -1].float()
        return hook

    lm.tm.embed_tokens.register_forward_hook(keep("emb"))
    for i, layer in enumerate(lm.tm.layers):
        (layer.linear_attn if i in lm.gdn_layers else layer.self_attn).register_forward_hook(keep(("mix", i)))
        layer.mlp.register_forward_hook(keep(("mlp", i)))
    lm.tm.norm.register_forward_pre_hook(lambda _m, a: buf.__setitem__("fin", a[0][0, -1].float()) if tracing[0] else None)

    def attribution(logits):
        """Contributions of [embedding, mixer 0..63, MLP 0..63] to z_call - E_q[z] through the final
        RMSNorm at its actual scale (first-order in nothing: exact for the pre-norm residual)."""
        z = logits.double()
        qr = torch.softmax(torch.cat([z[:TC], z[TC + 1:]]), 0)
        q = torch.cat([qr[:TC], qr.new_zeros(1), qr[TC:]])
        W = lm._head32
        d = W[TC] - q.float() @ W
        x = buf["fin"]
        w = d * gnorm * torch.rsqrt(x.pow(2).mean() + eps)
        C = torch.stack([buf["emb"]] + [buf[("mix", i)] for i in range(NL)] + [buf[("mlp", i)] for i in range(NL)])
        return C @ w, (z[TC] - (q * z).sum()).item(), (x @ w).item(), C, w

    def lp_split(logits):
        lp = torch.log_softmax(logits.double(), -1)
        rest = torch.cat([lp[:TC], lp[TC + 1:]])
        return lp, lp[TC].item(), torch.logsumexp(rest, 0).item()

    def fwd_logits(ids, cache):
        return lm.logits(lm.forward(torch.tensor([ids], device=lm.device), cache)[:, -1])[0]

    def run(cache, sess_str):
        ids = tok.encode(sess_str, add_special_tokens=False)
        tracing[0] = True
        logits = fwd_logits(ids, cache)
        contrib, target, xw, C0, w0 = attribution(logits)
        lp, lp_call, lp_not = lp_split(logits)
        top = lp.topk(TOPK)
        row = dict(lp_call=lp_call, lp_not_call=lp_not, n_sess=len(ids),
                   top=[[tok.decode([i]), round(v, 4)] for i, v in zip(top.indices.tolist(), top.values.tolist())],
                   dla0=[round(v, 5) for v in contrib.tolist()], dla_check=[target, contrib.sum().item(), xw])
        dla_steps = [contrib.cpu()]
        pre, path, stop = [], [], "cap"
        for n in range(PRE_MAX):
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
            logits = fwd_logits([t], cache)
            if n + 1 < PRE_MAX:
                dla_steps.append(attribution(logits)[0].cpu())
        tracing[0] = False
        row.update(stop=stop, called=stop == "call", n_pre=len(pre), pre=tok.decode(pre), path=path)
        if stop == "call":
            logits, call = fwd_logits([TC], cache), []
            for _ in range(CALL_MAX):
                t = int(logits.argmax())
                if t in STOP:
                    break
                call.append(t)
                if "</parameter>" in tok.decode(call) or "</function>" in tok.decode(call):
                    break
                logits = fwd_logits([t], cache)
            txt = tok.decode(call)
            fn = re.search(r"<function=([a-z_]+)>", txt)
            arg = re.search(r"<parameter=\w+>\n?(.*?)(\n?</parameter>|$)", txt, re.S)
            row.update(call=txt, fn=fn.group(1) if fn else None, arg=arg.group(1).strip() if arg else None)
        return row, dict(dla=torch.stack(dla_steps), C0=C0.bfloat16().cpu(), w0=w0.cpu())

    def anchor(cache, plan):
        lp = lm.score(cache, lm.ids(plan.anchor_text), lm.ids(plan.gold))
        return dict(lp=lp.tolist(), lp_sum=float(lp.sum()), n_tok=len(lp))

    def snapshot(cache, ids):
        rec = {i: cache.layers[i].recurrent_states.detach().clone() for i in lm.gdn_layers}
        conv = {i: cache.layers[i].conv_states.detach().clone() for i in lm.gdn_layers}
        sha = hashlib.sha256(torch.tensor(ids).numpy().tobytes()).hexdigest()[:16]
        return CompiledState(rec, conv, len(ids), sha, {})

    def versions(plan):
        """Yield (version, full cache) one at a time from the shared prefix."""
        base = lm.new_cache()
        lm.forward(torch.tensor([plan.ids["p"][:plan.b]], device=lm.device), base)
        keys = list(plan.versions)
        for n, k in enumerate(keys):
            c = base if n == len(keys) - 1 else copy.deepcopy(base)
            lm.forward(torch.tensor([plan.ids[k][plan.b:]], device=lm.device), c)
            yield k, c
            del c
        del base

    def kv_variants(full, T, s):
        """{cond: zero-arg cache factory} for the KV conditions of version s."""
        out = {"K": lambda: lm.zero_gdn(copy.deepcopy(full)), "M": lambda: copy.deepcopy(full)}
        if s == "a2":
            return out, None

        def ck():
            c = copy.deepcopy(full)
            for i in lm.gdn_layers:
                c.layers[i].conv_states.zero_()
            return c

        kpre = {i: lm.unrotate_keys(full.layers[i].keys) for i in lm.attn_layers}

        def with_keys(keys, shift=None, bias=None):
            def make():
                c = lm.zero_gdn(copy.deepcopy(full))
                for i in lm.attn_layers:
                    c.layers[i].keys = keys[i]
                c.statelm_pos_shift, c.statelm_kv_bias = shift, bias
                return c
            return make

        out["CK"] = ck
        for name, D in SHIFTS.items():
            out[name] = with_keys(kpre, shift=D - T)
        out["Krt"] = with_keys({i: lm.rope_keys(kpre[i], torch.arange(T)) for i in lm.attn_layers})
        orig = {i: full.layers[i].keys for i in lm.attn_layers}
        for name, sl in BANDS.items():
            vis = set(lm.attn_layers[sl])
            out[name] = with_keys(orig, bias={i: torch.full((T,), NEG, device=lm.device)
                                              for i in lm.attn_layers if i not in vis})
        return out, kpre

    # ---------------------------------------------------------------- output / resume
    header = dict(block="_header", model=args.model, data=args.data,
                  data_sha=hashlib.sha256(open(args.data, "rb").read()).hexdigest()[:16], qids=qids,
                  plumbing=bool(args.qids), bands={k: lm.attn_layers[v] for k, v in BANDS.items()}, shifts=SHIFTS,
                  pre_max=PRE_MAX, call_max=CALL_MAX, tools=lme.TOOLS, anchor_suffix=ANCHOR_SUFFIX,
                  anchor_conds=ANCHOR_CONDS, decoding="greedy")
    done = set()
    if args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        text = open(args.out).read()
        if not text.endswith("\n"):
            text = text[:text.rfind("\n") + 1]
            open(args.out, "w").write(text)
        prior = [json.loads(line) for line in text.splitlines() if line]
        assert prior and prior[0] == json.loads(json.dumps(header)), "resume: header differs"
        done = {r["qid"] for r in prior[1:] if r["block"] == "done"}
        out = open(args.out, "a")
        print(f"resume: {len(done)} of {len(qids)} questions complete", flush=True)
    else:
        out = open(args.out, "w")
        out.write(json.dumps(header) + "\n")
        out.flush()

    t0 = time.time()
    for qi, q in enumerate(qids):
        if q in done:
            continue
        tq = time.time()
        torch.cuda.reset_peak_memory_stats()
        plan, x = plans[q], plans[q].x
        traces = {}
        out.write(json.dumps(dict(block="meta", qid=q, qtype=x["question_type"], question=x["question"],
                                  date=x["question_date"], gold=x["answer"], anchor_gold=plan.gold,
                                  n_files=len(plan.st["names"]), ev=plan.st["ev"],
                                  ev_names=[plan.st["names"][i] for i in plan.st["ev"]], rep=plan.st["rep"],
                                  rep2=plan.st["rep2"] if "a2" in plan.versions else None,
                                  n_tok={k: len(v) for k, v in plan.ids.items()}, shared_prefix=plan.b)) + "\n")

        def emit(cond, store, cache):
            row, tr = run(cache, plan.plain)
            traces[(cond, store)] = tr
            out.write(json.dumps(dict(block="decide", qid=q, qtype=x["question_type"], cond=cond, store=store,
                                      **row)) + "\n")

        def emit_anchor(cond, store, cache):
            out.write(json.dumps(dict(block="anchor", qid=q, qtype=x["question_type"], cond=cond, store=store,
                                      gold=plan.gold, **anchor(cache, plan))) + "\n")

        emit("A", None, lm.cache_from_state(None))
        if plan.gold:
            emit_anchor("A", None, lm.cache_from_state(None))
        t_compile = 0.0
        tc = time.time()
        for s, full in versions(plan):
            t_compile += time.time() - tc
            T = len(plan.ids[s])
            st = snapshot(full, plan.ids[s])
            emit("C", s, lm.cache_from_state(st))
            if plan.gold and s != "a2":
                emit_anchor("C", s, lm.cache_from_state(st))
            makers, kpre = kv_variants(full, T, s)
            for cond, make in makers.items():
                if plan.gold and s != "a2" and cond in ANCHOR_CONDS:
                    emit_anchor(cond, s, make())
                emit(cond, s, make())
            del makers, kpre, st, full
            torch.cuda.empty_cache()
            out.flush()
            tc = time.time()
        torch.save(traces, os.path.join(trace_dir, f"{q}.pt"))
        out.write(json.dumps(dict(block="done", qid=q, t_compile=round(t_compile, 1),
                                  t_question=round(time.time() - tq, 1))) + "\n")
        out.flush()
        print(f"{qi + 1}/{len(qids)} {q} {x['question_type']} tokens {len(plan.ids['p'])} prefix {plan.b} "
              f"versions {list(plan.versions)} compile {t_compile:.0f}s question {time.time() - tq:.0f}s "
              f"total {time.time() - t0:.0f}s peak alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
              f"reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
