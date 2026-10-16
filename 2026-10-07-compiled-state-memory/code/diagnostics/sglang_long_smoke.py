"""Room check infrastructure smoke (LABNOTES 2026-10-13): can the long-context SGLang profile
(serve_sglang.sh start-long) prefill ~200k tokens, reuse that prefix from the radix cache, and give
the same greedy output with the cache cold and warm? Filler text comes from H1 histories (already
used questions); no fresh question is touched.

uv run python diagnostics/sglang_long_smoke.py
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from statelm import lme  # noqa: E402

URL = "http://127.0.0.1:30000/generate"


def gen(text, n=32):
    t = time.time()
    req = urllib.request.Request(URL, data=json.dumps(dict(text=text, sampling_params=dict(
        temperature=0, max_new_tokens=n))).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    m = r["meta_info"]
    return r["text"], dict(s=round(time.time() - t, 1), prompt=m["prompt_tokens"], cached=m.get("cached_tokens"),
                           out=m["completion_tokens"])


def main():
    data = lme.load("../data/raw/longmemeval/longmemeval_s_cleaned.json")
    byid = {x["question_id"]: x for x in data}
    pool = lme.build_pool(data)
    h1, _ = lme.sample(data, pool, byid)
    bodies = [lme.session_body(s) for q in h1[:2] for _, _, s in lme.history(byid[q])]
    mem = "\n\n".join(bodies)
    mem = mem[:int(len(mem) * 0.86)]  # ~200k tokens
    short = "<|im_start|>user\nName three primary colours.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    a_cold, s0 = gen(short)
    print("short cold", s0, repr(a_cold[:80]))
    pre = f"<|im_start|>system\nYou are a helpful assistant. Here is your memory about the user:\n{mem}<|im_end|>\n"
    for i, q in enumerate(("What did I say about my car?", "What did I say about my job?")):
        out, st = gen(pre + f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", 64)
        print(f"long {i}", st, f"prefill tok/s {st['prompt'] / max(st['s'], 1e-6):.0f}", repr(out[:100]))
    a_warm, s1 = gen(short)
    print("short warm", s1, repr(a_warm[:80]))
    print("cold == warm:", a_cold == a_warm)
    _, sd = gen(short.replace("three", "ten"), 256)
    print("decode", sd, f"tok/s {sd['out'] / sd['s']:.1f}")


if __name__ == "__main__":
    main()
