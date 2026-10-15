"""Room check (LABNOTES 2026-10-13, night), CPU only, before the server starts: the calibration
split, nested stores from LongMemEval_M (statelm/room.py), keyword indexes, the sessions to digest,
and a model-free retrieval check (BM25 with the question as the query: does an evidence file rank in
the top 3, and do all evidence files rank in the top 10, at each size).

uv run --with "transformers==5.12.1" --with jinja2 python experiments/room_prep.py --out ../results/phase2/room/prep
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, ".")
from statelm import lme, room  # noqa: E402

S_PATH = "../data/raw/longmemeval/longmemeval_s_cleaned.json"
M_PATH = "../data/raw/longmemeval/longmemeval_m_cleaned.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=os.path.expandvars("$STATELM_WEIGHTS/Qwen3.8-27B"))
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)
    data = lme.load(S_PATH)
    byid = {x["question_id"]: x for x in data}
    pool = lme.build_pool(data)
    idf = lme.build_idf(data)
    fr = room.fresh(data, room.used_ids(data, pool, byid))
    cal, held = room.calibration(fr)
    assert len(cal) == 120 and len(held) == 242
    json.dump(dict(cal=cal, held=held, held_flagged=list(room.HELD_FLAGGED)), open(os.path.join(args.out, "split.json"), "w"))
    del pool
    calset = set(cal)
    mdata = [x for x in json.load(open(M_PATH)) if x["question_id"] in calset]
    assert len(mdata) == 120
    todo, stats = {}, collections.defaultdict(list)
    for xm in mdata:
        q = xm["question_id"]
        xs = byid[q]
        assert (xs["question"], xs["answer"], xs["question_type"]) == (xm["question"], xm["answer"], xm["question_type"])
        assert set(xs["answer_session_ids"]) == set(xm["answer_session_ids"])
        stores, bodies, ntok = room.nested_stores(xm, tok)
        ev = set(xm["answer_session_ids"])
        sizes = {}
        for size, files in stores.items():
            names = [n for n, _ in files]
            bl = [bodies[s] for _, s in files]
            sizes[size] = dict(files=files, n_tok=ntok[size], n_files=len(files),
                               ev_names=[n for n, s in files if s in ev],
                               index_T=lme.keyword_index(names, bl, idf))
            if size != "O":
                ranks, _ = lme.bm25_ranks(bl, xm["question"])
                er = [r for (n, s), r in zip(files, ranks) if s in ev]
                stats[size].append((min(er) <= 3, max(er) <= 10, len(er)))
        used = {s for _, s in stores["large"]}
        for s in used:
            todo.setdefault(room.body_key(bodies[s]), bodies[s])
        json.dump(dict(qid=q, qtype=xm["question_type"], question=xm["question"], gold=xm["answer"],
                       question_date=xm["question_date"], sizes=sizes, bodies={s: bodies[s] for s in used}),
                  open(os.path.join(args.out, f"{q}.json"), "w"), ensure_ascii=False)
        print(q, xm["question_type"], {k: (v["n_tok"], v["n_files"]) for k, v in sizes.items()}, flush=True)
    with open(os.path.join(args.out, "digest_todo.jsonl"), "w") as f:
        for k, b in sorted(todo.items()):
            f.write(json.dumps(dict(key=k, body=b), ensure_ascii=False) + "\n")
    print(f"sessions to digest: {len(todo)}")
    for size, rows in stats.items():
        n = len(rows)
        print(f"BM25 (question as query) {size}: an evidence file in top 3 {sum(r[0] for r in rows) / n:.2f}, "
              f"all evidence files in top 10 {sum(r[1] for r in rows) / n:.2f} (n {n}, mean evidence files "
              f"{sum(r[2] for r in rows) / n:.1f})")


if __name__ == "__main__":
    main()
