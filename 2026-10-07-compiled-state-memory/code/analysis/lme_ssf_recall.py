"""Generator recall check R1 (dev; LABNOTES 2026-10-11): does fact-level self-study, run memory-free on
the evidence sessions' user turns, write an item whose answer is the gold fact?

uv run --with "transformers==5.12.1" --with jinja2 python analysis/lme_ssf_recall.py ../results/phase2/ssf/*_p_all_*.json

Scores the items a fit would use (--scope all files), restricted to the evidence sessions.
A question is covered if some item's normalised answer contains the normalised gold or is contained
in it (at least 3 characters), and the gold is not in the item's question. Number words and digits
are matched both ways. Evidence labels are used only here, to score the generator on dev.
"""
import json
import os
import sys

sys.path.insert(0, ".")
from experiments.lme_ssf_gen import norm  # noqa: E402
from statelm import lme  # noqa: E402

NUM = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def forms(s):
    s = norm(s)
    return {s, " ".join(NUM.get(w, w) for w in s.split())}


def covers(answer, question, gold):
    ga, aa, qa = forms(gold), forms(answer), forms(question)
    if any(g in q for g in ga for q in qa):
        return False
    return any(g in a or (len(a) >= 3 and a in g) for g in ga for a in aa)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.expandvars("$STATELM_WEIGHTS/Qwen3.8-27B"))
    data = lme.load("../data/raw/longmemeval/longmemeval_s_cleaned.json")
    byid = {x["question_id"]: x for x in data}
    pool = lme.build_pool(data)
    n = hit = 0
    by_type = {}
    for f in sys.argv[1:]:
        d = json.load(open(f))
        x = byid[d["qid"]]
        gold = lme.short_answer(x, tok)
        if gold is None:
            continue
        assert d["store"] == "p" and d["meta"]["scope"] == "all", f
        ev = set(lme.stores(x, pool, byid)["ev"])
        its = [it for it in d["items"] if it["session"] in ev]
        n += 1
        good = [it for it in its if covers(it["answer"], it["question"], gold)]
        near = [it for it in d["dropped"] if it["session"] in ev and it["question"]
                and covers(it["answer"], it["question"], gold)]
        hit += bool(good)
        t = by_type.setdefault(x["question_type"], [0, 0])
        t[0] += bool(good)
        t[1] += 1
        print(f"{d['qid']} {x['question_type'][:20]:20s} gold {gold!r:28s} evidence items {len(its):3d}  "
              f"{'COVERED' if good else 'missed '}  | {x['question']}")
        for it in good[:3]:
            print(f"    + {it['question']!r} -> {it['answer']!r}")
        for it in near[:2]:
            print(f"    (dropped: {it['why']}) {it['question']!r} -> {it['answer']!r}")
    print(f"covered {hit}/{n}  (" + ", ".join(f"{k} {a}/{b}" for k, (a, b) in by_type.items()) + ")")


if __name__ == "__main__":
    main()
