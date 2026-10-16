"""Room check (LABNOTES 2026-10-13, night): arm D's index, one note per memory file, written by the same
model with greedy decoding, question-blind (it sees one session and nothing else), cached by session
text (room.body_key). Runs on the SGLang server (native /generate, pre-rendered prompts).

uv run --with "transformers==5.12.1" --with jinja2 python experiments/room_digest.py --prep ../results/phase2/room/prep
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from statelm import room  # noqa: E402
from statelm.prompts import session_text  # noqa: E402

NOTE = ("Below is one past conversation between a user and an AI assistant, saved as a memory file.\n\n"
        "{body}\n\n"
        "Write a one-line note (at most 25 words) for an index of memory files, saying what this conversation "
        "covers. Name the specific topics, people, places, items, numbers and dates that someone might later "
        "look for. Reply with the note only.")
MAX_NEW, MAX_WORDS, MAX_IN = 48, 40, 7_000  # MAX_IN: body tokens (the gen profile's context is 8,192)


def clean(text):
    line = next((ln.strip() for ln in text.strip().split("\n") if ln.strip()), "")
    return " ".join(line.split()[:MAX_WORDS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tokenizer", default=os.path.expandvars("$STATELM_WEIGHTS/Qwen3.8-27B"))
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out = os.path.join(args.prep, "digests.jsonl")
    done = {json.loads(ln)["key"] for ln in open(out)} if os.path.exists(out) else set()
    todo = [r for r in map(json.loads, open(os.path.join(args.prep, "digest_todo.jsonl"))) if r["key"] not in done]
    print(f"{len(todo)} to write ({len(done)} done)", flush=True)
    t0 = time.time()
    for s in range(0, len(todo), args.batch):
        part = todo[s:s + args.batch]
        prompts = [session_text(tok, NOTE.format(body=tok.decode(tok.encode(r["body"][:room.READ_CHARS],
                                                                          add_special_tokens=False)[:MAX_IN])))
                   for r in part]
        req = urllib.request.Request(args.url + "/generate", data=json.dumps(dict(
            text=prompts, sampling_params=dict(temperature=0, max_new_tokens=MAX_NEW))).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as f:
            res = json.load(f)
        with open(out, "a") as f:
            for r, g in zip(part, res):
                f.write(json.dumps(dict(key=r["key"], note=clean(g["text"]), raw=g["text"],
                                        cut=(g["meta_info"].get("finish_reason") or {}).get("type") == "length"),
                                   ensure_ascii=False) + "\n")
        n = s + len(part)
        print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
