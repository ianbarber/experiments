"""Stage 1 development (exploratory, dev; LABNOTES 2026-10-11): fact-level self-study items (ssf).

Memory-free and text-only: for every user turn of a memory store (with the tail of the assistant
turn before it as context) the model, with no memory and without the session's question, lists
facts the user states and writes for each a question the user might ask later and a short answer
copied from the turn. The items are later teacher-forced on CKstar (experiments/lme_seed.py,
--ssf_dir) to give reference queries at the positions that predict each answer.

Generation runs on an SGLang server (native /generate, pre-rendered prompts, greedy), the same bf16
weights as the HF experiments. Output: one JSON per (question, store) with every prompt's raw
completion and the parsed items, plus a filter log.

  --scope all       every session of the store (reference items for the fit; R1 scores these same
                    items, restricted to the evidence sessions)
  --scope evidence  only the evidence sessions (quick checks only)
  --role assistant  (written for dev run 10, which was dropped; its partial output is not used) details the
                    assistant gives in its turns (SS_FACT_A; one excerpt per assistant turn
                    piece, after the tail of the user turn before it) instead of facts the user states.
                    The default (--role user) is unchanged, prompt, meta and file tag included.

uv run --with "transformers==5.12.1" --with jinja2 python experiments/lme_ssf_gen.py --qids <dev ids> --stores p --scope all \
    --out_dir ../results/phase2/ssf
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from experiments.lme_where import dev_sample  # noqa: E402
from statelm import lme  # noqa: E402
from statelm.prompts import session_text  # noqa: E402

SS_FACT = ("Below is an excerpt from a past conversation between a user and an AI assistant.\n\n"
           "{excerpt}\n\n"
           "List the specific facts that the user states about themselves or their life in their last message "
           "(for example names, numbers, quantities, dates, times, places, brands, models, colours or "
           "preferences). For each fact, write one question that the user could ask the assistant later to "
           "recall it, in the user's own voice, and its short answer copied from the user's message (one to five "
           "words). The answer must not appear in the question. Write at most {k} pairs, one per fact, in this "
           "format:\nQ: <question>\nA: <answer>\n"
           "If the user's last message states no such facts, reply NONE.")
SS_FACT_A = ("Below is an excerpt from a past conversation between a user and an AI assistant.\n\n"
             "{excerpt}\n\n"
             "List the specific details that the assistant gives in its last message that the user might want to "
             "recall later (for example names, titles, numbers, quantities, dates, times, places, brands, models, "
             "colours or recommendations). For each detail, write one question that the user could ask the assistant "
             "later to recall it, in the user's own voice, and its short answer copied from the assistant's message "
             "(one to five words). The answer must not appear in the question. Write at most {k} pairs, one per "
             "detail, in this format:\nQ: <question>\nA: <answer>\n"
             "If the assistant's last message gives no such details, reply NONE.")
TURN = re.compile(r"^\[(user|assistant)\]: ", re.M)
MARK = r"\s*(?:[-*]\s*|\d+[.)]\s*)?(?:\*\*)?"
PAIR = re.compile(rf"^{MARK}Q\d*:(?:\*\*)?\s*(.+?)\s*\n(?:\s*\n)?{MARK}A\d*:(?:\*\*)?\s*(.+?)\s*$", re.M)


def norm(s):
    return " ".join(re.sub(r"[^\w\s:.-]", " ", s.lower()).replace("_", " ").split()).strip(" .")


def turns(body):
    """[(role, text)] of a session body ('[user]: ...' / '[assistant]: ...' turns)."""
    marks = list(TURN.finditer(body))
    return [(m.group(1), body[m.end():marks[j + 1].start() if j + 1 < len(marks) else len(body)].strip())
            for j, m in enumerate(marks)]


def excerpts(tok, body, ctx, piece):
    """One excerpt per user turn: the last `ctx` tokens of the assistant turn before it (if any) and
    the user turn, cut into pieces of at most `piece` tokens (each piece its own excerpt)."""
    out, prev = [], None
    for k, (role, text) in enumerate(turns(body)):
        if role == "assistant":
            prev = text
            continue
        head = ""
        if prev:
            ids = tok.encode(prev, add_special_tokens=False)
            head = "[assistant]: " + ("..." if len(ids) > ctx else "") + tok.decode(ids[-ctx:]) + "\n"
        ids = tok.encode(text, add_special_tokens=False)
        for p, s0 in enumerate(range(0, max(len(ids), 1), piece)):
            user = tok.decode(ids[s0:s0 + piece])
            out.append(dict(turn=k, piece=p, user=user, excerpt=head + "[user]: " + user))
        prev = None
    return out


def excerpts_assistant(tok, body, ctx, piece):
    """One excerpt per assistant turn piece: the last `ctx` tokens of the user turn before it (if any) and
    the assistant turn, cut into pieces of at most `piece` tokens. The piece is kept as 'text'."""
    out, prev = [], None
    for k, (role, text) in enumerate(turns(body)):
        if role == "user":
            prev = text
            continue
        head = ""
        if prev:
            ids = tok.encode(prev, add_special_tokens=False)
            head = "[user]: " + ("..." if len(ids) > ctx else "") + tok.decode(ids[-ctx:]) + "\n"
        ids = tok.encode(text, add_special_tokens=False)
        for p, s0 in enumerate(range(0, max(len(ids), 1), piece)):
            a = tok.decode(ids[s0:s0 + piece])
            out.append(dict(turn=k, piece=p, text=a, excerpt=head + "[assistant]: " + a))
        prev = None
    return out


def server_info(url):
    with urllib.request.urlopen(url.rstrip("/") + "/get_model_info", timeout=60) as f:
        info = json.loads(f.read())
    assert info.get("model_path", "").rstrip("/").endswith("Qwen3.8-27B"), info
    return info


def generate(url, prompts, max_new, batch):
    outs = []
    for s in range(0, len(prompts), batch):
        req = dict(text=prompts[s:s + batch], sampling_params=dict(temperature=0, max_new_tokens=max_new))
        r = urllib.request.Request(url.rstrip("/") + "/generate", data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=3600) as f:
            res = json.loads(f.read())
        outs += [(x["text"], x["meta_info"].get("finish_reason")) for x in (res if isinstance(res, list) else [res])]
        print(f"  {len(outs)}/{len(prompts)}", flush=True)
    return outs


def parse(raw, user):
    """Parsed (question, answer) pairs and why any were dropped."""
    items, dropped = [], []
    pairs = PAIR.findall(raw)
    if not pairs and raw.strip().strip(".").upper() != "NONE":
        dropped.append(dict(question=None, answer=None, why="unparsed", raw=raw))
    for q, a in pairs:
        q, a = q.strip().strip("*").strip(), a.strip().strip("*").strip().strip('"').strip().rstrip(".")
        why = None
        if not a or len(a.split()) > 8:
            why = "length"
        elif norm(a) in norm(q):
            why = "answer in question"
        elif norm(a) not in norm(user):
            why = "not copied from the turn"
        (dropped if why else items).append(dict(question=q.strip(), answer=a, **({"why": why} if why else {})))
    return items, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", required=True)
    ap.add_argument("--stores", default="p")
    ap.add_argument("--scope", choices=("evidence", "all"), required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data", default="../data/raw/longmemeval/longmemeval_s_cleaned.json")
    ap.add_argument("--tokenizer", default=os.path.expandvars("$STATELM_WEIGHTS/Qwen3.8-27B"))
    ap.add_argument("--server", default="http://127.0.0.1:30000")
    ap.add_argument("--ctx", type=int, default=160, help="tokens of the preceding assistant turn")
    ap.add_argument("--piece", type=int, default=400, help="max tokens of a user turn per excerpt")
    ap.add_argument("--k", type=int, default=4, help="max pairs per excerpt")
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--role", choices=("user", "assistant"), default="user", help="whose turns the facts come from")
    ap.add_argument("--dry", action="store_true", help="print the file tag and prompt count, generate nothing")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data = lme.load(args.data)
    byid = {x["question_id"]: x for x in data}
    pool = lme.build_pool(data)
    dev = dev_sample(data, pool, byid)
    qids = args.qids.split(",")
    assert set(qids) <= set(dev), "Stage 1 development uses dev16 only"
    os.makedirs(args.out_dir, exist_ok=True)
    info = None if args.dry else server_info(args.server)
    fact, cut = (SS_FACT, excerpts) if args.role == "user" else (SS_FACT_A, excerpts_assistant)
    meta = dict(ss_fact=fact, ctx=args.ctx, piece=args.piece, k=args.k, max_new=args.max_new,
                tokenizer=os.path.basename(args.tokenizer.rstrip("/")), decoding="greedy", scope=args.scope)
    if args.role != "user":  # the user path keeps its meta, so its tag is unchanged
        meta["role"] = args.role
    tag = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
    for q in qids:
        x = byid[q]
        st = lme.stores(x, pool, byid)
        for s in args.stores.split(","):
            bodies = st[{"p": "present", "a": "absent"}[s]]
            sess = st["ev"] if args.scope == "evidence" else range(len(bodies))
            if args.scope == "evidence" and s != "p":
                continue
            ex = [dict(session=i, **e) for i in sess for e in cut(tok, bodies[i], args.ctx, args.piece)]
            prompts = [session_text(tok, fact.format(excerpt=e["excerpt"], k=args.k)) for e in ex]
            if args.dry:
                h = hashlib.sha256("\x00".join(prompts).encode()).hexdigest()[:12]
                print(f"{q} {s} {args.role}: tag {tag}, {len(prompts)} prompts (hash {h})")
                continue
            t0 = time.time()
            gen = generate(args.server, prompts, args.max_new, args.batch)
            raws = [g[0] for g in gen]
            items, drops = [], []
            for e, raw in zip(ex, raws):
                it, dr = parse(raw, e["user"] if args.role == "user" else e["text"])
                items += [dict(session=e["session"], turn=e["turn"], piece=e["piece"], **i,
                               **({} if args.role == "user" else dict(role=args.role))) for i in it]
                drops += [dict(session=e["session"], turn=e["turn"], **d) for d in dr]
            f = os.path.join(args.out_dir, f"{q}_{s}_{args.scope}_{tag}.json")
            json.dump(dict(meta=meta, server=info, qid=q, store=s, question_seen=False,
                           seconds=round(time.time() - t0, 1), n_prompts=len(prompts),
                           excerpts=[dict(e, raw=r, finish=fr) for e, (r, fr) in zip(ex, gen)],
                           items=items, dropped=drops), open(f, "w"), indent=1)
            n_un = sum(d["why"] == "unparsed" for d in drops)
            print(f"{q} {s}: {len(prompts)} prompts, {len(items)} items, {len(drops)} dropped ({n_un} unparsed), "
                  f"{time.time() - t0:.0f}s -> {f}", flush=True)


if __name__ == "__main__":
    main()
