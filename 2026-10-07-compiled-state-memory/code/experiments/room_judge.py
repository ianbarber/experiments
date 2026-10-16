"""Room check (LABNOTES 2026-10-13, night): LongMemEval's answer check (room.anscheck_prompt, verbatim
from evaluate_qa.py) on every distinct (question, answer) of the agent records, judged by Qwen3.8-27B on
the SGLang server: greedy, at most 10 tokens, correct = 'yes' in the reply (as the official script).
No answer (None) is wrong without a call. Resumable.

uv run --with "transformers==5.12.1" --with jinja2 python experiments/room_judge.py --traj ../results/phase2/room/traj.jsonl \
    --prep ../results/phase2/room/prep --out ../results/phase2/room/judge.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, ".")
from statelm import room  # noqa: E402
from statelm.prompts import session_text  # noqa: E402


def akey(qid, answer):
    return hashlib.sha1(f"{qid}\x00{answer}".encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="comma-separated agent record files")
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tokenizer", default=os.path.expandvars("$STATELM_WEIGHTS/Qwen3.8-27B"))
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    done = {json.loads(ln)["key"] for ln in open(args.out)} if os.path.exists(args.out) else set()
    meta, todo = {}, {}
    for path in args.traj.split(","):
        for r in map(json.loads, open(path)):
            if r["answer"] is None:
                continue
            k = akey(r["qid"], r["answer"])
            if k in done or k in todo:
                continue
            if r["qid"] not in meta:
                q = json.load(open(os.path.join(args.prep, f"{r['qid']}.json")))
                meta[r["qid"]] = (q["qtype"], q["question"], q["gold"])
            todo[k] = (r["qid"], r["answer"])
    items = sorted(todo.items())
    print(f"{len(items)} to judge ({len(done)} done)", flush=True)
    for s in range(0, len(items), args.batch):
        part = items[s:s + args.batch]
        prompts = [session_text(tok, room.anscheck_prompt(*meta[qid], ans)) for _, (qid, ans) in part]
        req = urllib.request.Request(args.url + "/generate", data=json.dumps(dict(
            text=prompts, sampling_params=dict(temperature=0, max_new_tokens=10))).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as f:
            res = json.load(f)
        with open(args.out, "a") as f:
            for (k, (qid, _)), g in zip(part, res):
                f.write(json.dumps(dict(key=k, qid=qid, label="yes" in g["text"].lower(), raw=g["text"])) + "\n")
        print(f"{s + len(part)}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
