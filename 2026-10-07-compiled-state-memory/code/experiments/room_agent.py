"""Room check (LABNOTES 2026-10-13): run the baseline arms as agents on an SGLang server.

Reads the prepared calibration questions (experiments/room_prep.py; CPU, before the server starts) and
writes one record per (question, arm, size) to --out (JSONL, resumable: done keys are skipped).

Arms (all greedy, thinking off, Qwen3.8's native template):
  N  no memory, no tools                         A  memory tools, no hint
  T  tools + keyword index of the store          D  tools + one-line note per file (room_digest.py)
  M  the store in context (H1's M format: compile_text prefix) + tools
  O  M on the evidence-only store (size "O"; N also runs once, under size "O")
Agents stop by themselves, at most CAP tool calls; one call per assistant turn (extra calls in a turn
are dropped from the turn and counted). After the CAP-th result, or if the next turn would overflow
the context, a user turn (room.FORCE) asks for the answer. Ending rules:
  - a turn without a call is the answer (a turn that only announces a search still counts, flagged);
  - a turn cut at max_new_tokens inside a call is no answer (scored wrong, flagged);
  - a forced reply that opens a call: the text before the call is the answer (empty: no answer).
Prompts are token ids: previous input + generated ids + the next turn, so the radix cache can resume
(checked against the chat template in LABNOTES; cached_tokens is logged per request).

uv run --with "transformers==5.12.1" --with jinja2 python experiments/room_agent.py --arms A,T,D --sizes small,mid,large \
    --prep ../results/phase2/room/prep --out ../results/phase2/room/traj.jsonl --workers 16
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, ".")
from statelm import lme, room  # noqa: E402
from statelm.prompts import compile_text  # noqa: E402

CAP, MAX_NEW = 8, 512
RESULT_MAX = 9_000      # tool-result tokens (a longer result is cut, flagged)
RESERVE = RESULT_MAX + 64 + MAX_NEW  # after a turn's call: its result, the forced turn, the forced answer
M_WORKERS = {"small": 6, "mid": 2, "large": 1}  # M keeps each question's prompt in the prefix cache
stop = threading.Event()
GEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
PLAIN = "You are a helpful assistant."
MEM_FLOOR_GIB = 8
lock = threading.Lock()


def mem_avail_gib():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) / 1048576
    return 0.0


class TooLong(Exception):
    pass


def server_ctx(url):
    for ep in ("/get_server_info", "/server_info"):
        try:
            with urllib.request.urlopen(url + ep, timeout=60) as f:
                info = json.load(f)
            break
        except urllib.error.HTTPError:
            continue
    n = info.get("max_req_input_len") or (info.get("internal_states") or [{}])[0].get("max_req_input_len")
    assert n, "server did not report max_req_input_len"
    return int(n)


def post(url, ids, max_new):
    req = urllib.request.Request(url + "/generate", data=json.dumps(dict(
        input_ids=ids, sampling_params=dict(temperature=0, max_new_tokens=max_new, skip_special_tokens=False))).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=7200) as f:
            return json.load(f)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code == 400 and ("length" in body or "context" in body):
            raise TooLong(body[:200])
        raise


class Agent:
    def __init__(self, tok, url, digests, ctx):
        self.tok, self.url, self.digests, self.ctx = tok, url, digests, ctx
        self.eos = tok.convert_tokens_to_ids("<|im_end|>")

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def gen(self, ids, steps):
        if stop.is_set():
            raise RuntimeError("run stopped")
        if mem_avail_gib() < MEM_FLOOR_GIB:
            stop.set()
            raise RuntimeError(f"MemAvailable below {MEM_FLOOR_GIB} GiB: stopping")
        t = time.time()
        r = post(self.url, ids, MAX_NEW)
        m = r["meta_info"]
        out = list(r.get("output_ids") or self.enc(r["text"]))
        steps.append(dict(prompt=m["prompt_tokens"], cached=m.get("cached_tokens"), out=m["completion_tokens"],
                          finish=(m.get("finish_reason") or {}).get("type"), s=round(time.time() - t, 2)))
        text = r["text"].replace("<|im_end|>", "").replace("<|endoftext|>", "")
        return text, out, (m.get("finish_reason") or {}).get("type") == "length"

    @staticmethod
    def files(q, size):
        return [(n, room.defuse(q["bodies"][sid])) for n, sid in q["sizes"][size]["files"]]

    def prefix(self, arm, q, size):
        """Prompt text up to the assistant's first turn."""
        date = q["question_date"]
        sysm = lme.system_prompt(date)
        if arm == "N":
            return self.tok.apply_chat_template(
                [{"role": "system", "content": PLAIN + f"\n\nToday's date: {date.split(')')[0]})."},
                 {"role": "user", "content": q["question"]}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        if arm == "T":
            sysm = lme.system_prompt(date, index=q["sizes"][size]["index_T"])
        if arm == "D":
            notes = [f"{n}: {room.defuse(self.digests[room.body_key(q['bodies'][sid])])}"
                     for n, sid in q["sizes"][size]["files"]]
            sysm += "\n\nIndex of your memory files (a one-line note per file):\n" + "\n".join(notes)
        s = self.tok.apply_chat_template([{"role": "system", "content": sysm}, {"role": "user", "content": q["question"]}],
                                         tools=lme.TOOLS, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        assert s.endswith(GEN) and "Reasoning effort" not in s
        if arm in ("M", "O"):
            files = self.files(q, size)
            s = compile_text(self.tok, lme.memory_text([n for n, _ in files], [b for _, b in files])) + s
        return s

    def run(self, arm, q, size):
        steps, calls = [], []
        flags = dict(dropped=0, cut=False, cap_in_call=False, malformed=False, forced=None, preamble=False,
                     overflow=False, result_cut=0)
        gen_ids = self.enc(GEN)
        ids = self.enc(self.prefix(arm, q, size))
        n_prefix = len(ids)
        if arm == "N":
            text, _, cut = self.gen(ids, steps)
            flags["cut"] = cut
            return dict(answer=text.strip(), calls=calls, steps=steps, flags=flags, n_prefix=n_prefix)
        store = room.Store(self.files(q, size))
        answer, need_force = None, False
        while True:  # ids end with the generation prompt here
            if len(ids) + MAX_NEW + RESERVE > self.ctx:
                flags["overflow"] = need_force = True
                ids = ids[:-len(gen_ids)]
                break
            try:
                text, out, cut = self.gen(ids, steps)
            except TooLong:
                flags["overflow"] = need_force = True
                ids = ids[:-len(gen_ids)]
                break
            if "<tool_call>" not in text:
                answer, flags["cut"] = text.strip(), cut
                flags["preamble"] = answer.endswith(":") or answer.lower().startswith(("let me", "i'll", "i will"))
                break
            parsed = room.parse_calls(text)
            if not parsed:  # cut inside a call, or a malformed call
                flags["cap_in_call"], flags["malformed"] = cut, not cut
                answer = None if cut else (text.split("<tool_call>")[0].strip() or None)
                break
            name, args = parsed[0]
            flags["dropped"] += len(parsed) - 1
            end = text.find("</tool_call>")
            first = text[:end + len("</tool_call>")] if end >= 0 else text
            if len(parsed) > 1 or end < 0 or text.strip() != first.strip():
                out = self.enc(first) + [self.eos]  # keep only the executed call
            elif out[-1] != self.eos:
                out = out + [self.eos]
            rt = self.enc(room.defuse(store.call(name, args)))
            if len(rt) > RESULT_MAX:
                rt = rt[:RESULT_MAX] + self.enc("\n[truncated]")
                flags["result_cut"] += 1
            calls.append(dict(name=name, args=args, result_tokens=len(rt)))
            ids = (ids + out + self.enc("\n<|im_start|>user\n<tool_response>\n") + rt
                   + self.enc("\n</tool_response><|im_end|>\n"))
            if len(calls) >= CAP:
                need_force = True
                break
            ids = ids + gen_ids
        if need_force:
            flags["forced"] = len(calls)
            force = self.enc("<|im_start|>user\n" + room.FORCE + "<|im_end|>\n") + gen_ids
            if len(ids) + len(force) + MAX_NEW <= self.ctx:
                text, _, cut = self.gen(ids + force, steps)
                flags["cut"] = cut
                answer = text.split("<tool_call>")[0].strip() or None
        return dict(answer=answer, calls=calls, steps=steps, flags=flags, n_prefix=n_prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--sizes", default="small,mid,large")
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--qids", default="", help="subset (smoke)")
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--tokenizer", default=os.path.expandvars("$STATELM_WEIGHTS/Qwen3.8-27B"))
    ap.add_argument("--tag", default="", help="written into each record (e.g. a rerun for the flip rate)")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    split = json.load(open(os.path.join(args.prep, "split.json")))
    qids = args.qids.split(",") if args.qids else split["cal"]
    assert not set(qids) & set(split["held"]), "held-out questions are never run here"
    done = set()
    if os.path.exists(args.out):
        for ln in open(args.out):
            r = json.loads(ln)
            done.add((r["qid"], r["arm"], r["size"], r.get("tag", "")))
    jobs = []
    for qid in qids:
        for arm in args.arms.split(","):
            for size in (["O"] if arm in ("N", "O") else args.sizes.split(",")):
                if (qid, arm, size, args.tag) not in done:
                    jobs.append((qid, arm, size))
    print(f"{len(jobs)} jobs ({len(done)} done)", flush=True)
    dpath = os.path.join(args.prep, "digests.jsonl")
    digests = {r["key"]: r["note"] for r in map(json.loads, open(dpath))} if os.path.exists(dpath) else {}
    if "D" in args.arms.split(","):
        need = {room.body_key(b) for qid in qids for b in json.load(open(os.path.join(args.prep, f"{qid}.json")))["bodies"].values()}
        assert need <= set(digests), f"arm D: {len(need - set(digests))} sessions have no note yet"
    ctx = server_ctx(args.url)
    print(f"server max input {ctx}", flush=True)
    agent = Agent(tok, args.url, digests, ctx)
    msizes = {size for _, arm, size in jobs if arm == "M"}
    if msizes:
        cap = min(M_WORKERS[s] for s in msizes)
        if args.workers > cap:
            print(f"M at {sorted(msizes)}: workers {args.workers} -> {cap}", flush=True)
            args.workers = cap
    t0 = time.time()

    def one(job):
        qid, arm, size = job
        q = json.load(open(os.path.join(args.prep, f"{qid}.json")))
        t = time.time()
        r = agent.run(arm, q, size)
        rec = dict(qid=qid, qtype=q["qtype"], arm=arm, size=size, tag=args.tag, s=round(time.time() - t, 1),
                   mem_gib=round(mem_avail_gib(), 1), **r)
        with lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return job

    ex = cf.ThreadPoolExecutor(args.workers)
    futs = [ex.submit(one, j) for j in jobs]
    try:
        for i, fut in enumerate(cf.as_completed(futs)):
            job = fut.result()  # a worker's error stops the run (resume picks up the rest)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)} {job} {time.time() - t0:.0f}s avail {mem_avail_gib():.0f} GiB", flush=True)
    except BaseException as e:
        stop.set()
        print(f"STOPPED: {type(e).__name__}: {e}", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        os._exit(1)
    ex.shutdown()


if __name__ == "__main__":
    main()
