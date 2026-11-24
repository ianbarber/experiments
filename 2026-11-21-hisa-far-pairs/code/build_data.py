"""Build the Stage 1 evaluation windows (token ids, 2048 tokens each) into ../data/*.npz (not committed).

  prose   OpenWebText documents of at least 2048 tokens (first 2048 tokens)
  code    Python files of at least 2048 tokens (codeparrot-clean-valid)
  tables  real CSV / JSON-lines / markdown-table text (vega-datasets, seaborn-data)
  grid-C  synthetic: rows of C single-token cells and a newline, C in 8..256
  copy    prose with a 32-token span repeated after a gap of 64..1536 tokens
  prose-stats  64 further prose windows (used only for the mean-of-dropped-parts check)

python build_data.py
"""

import csv
import io
import json
import os
import urllib.request

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

T = 2048
SEED = 42
OUT = "../data"
os.makedirs(OUT + "/raw", exist_ok=True)
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
rng = np.random.default_rng(SEED)


def enc(text):
    return tok(text, add_special_tokens=False).input_ids


# ----------------------------------------------------------------------------- prose, code


def long_docs(name, field, n, max_chars=200_000):
    out = []
    for r in load_dataset(name, split="train", streaming=True):
        text = r[field]
        if len(text) < 6000 or len(text) > max_chars:
            continue
        ids = enc(text)
        if len(ids) >= T:
            out.append(ids[:T])
            if len(out) == n:
                break
    return np.array(out, dtype=np.int32)


def build_prose_code():
    prose = long_docs("Skylion007/openwebtext", "text", 256 + 256 + 64)
    np.savez(f"{OUT}/prose.npz", ids=prose[:256])
    np.savez(f"{OUT}/prose-stats.npz", ids=prose[512:])
    # copy probe from the second 256 documents
    gaps = [64, 128, 256, 512, 1024, 1536]
    ids, meta = [], []
    for i, doc in enumerate(prose[256:512]):
        g = gaps[i % len(gaps)]
        p = int(rng.integers(16, T - 64 - g))            # span start; copy ends at p + 64 + g <= T
        seq = np.concatenate([doc[:p + 32 + g], doc[p:p + 32], doc[p + 32 + g:]])[:T]
        ids.append(seq)
        meta.append((p, p + 32 + g, g))
    np.savez(f"{OUT}/copy.npz", ids=np.array(ids, dtype=np.int32), meta=np.array(meta))
    code = long_docs("codeparrot/codeparrot-clean-valid", "content", 256)
    np.savez(f"{OUT}/code.npz", ids=code)
    print("prose", prose.shape, "code", code.shape)


# ----------------------------------------------------------------------------- natural tables

VEGA = "https://raw.githubusercontent.com/vega/vega-datasets/main/data/"
SEAB = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"
CSVS = [VEGA + f for f in ("airports.csv", "bird-migration.csv", "birdstrikes.csv", "flights-airport.csv",
                           "seattle-weather-hourly-normals.csv", "seattle-weather.csv", "sp500-2000.csv",
                           "species.csv", "weather.csv", "windvectors.csv", "zipcodes.csv")] + \
       [SEAB + f for f in ("diamonds.csv", "fmri.csv", "planets.csv", "seaice.csv", "taxis.csv", "titanic.csv")]
JSONS = [VEGA + f for f in ("cars.json", "countries.json", "flights-10k.json", "football.json", "gapminder.json",
                            "income.json", "jobs.json", "movies.json", "penguins.json", "budget.json",
                            "political-contributions.json", "unemployment-across-industries.json")]


def fetch(url):
    path = f"{OUT}/raw/" + url.split("/")[-1]
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return open(path, encoding="utf-8", errors="replace").read()


def line_windows(lines, n_max, header=None):
    """Non-overlapping 2048-token windows that start at a line boundary. The first starts at the top of the file;
    `header` lines (if any) are repeated at the top of later windows only for markdown."""
    wins, cur = [], []
    for ln in lines:
        cur.extend(enc(ln + "\n"))
        if len(cur) >= T:
            wins.append(cur[:T])
            cur = list(enc(header)) if header else []
            if len(wins) == n_max:
                break
    return wins


def build_tables():
    wins, kinds = [], []
    per = {"csv": 5, "jsonl": 8, "md": 5}
    for url in CSVS:
        lines = fetch(url).splitlines()
        w = line_windows(lines[:20000], per["csv"])
        wins += w
        kinds += ["csv"] * len(w)
        # markdown rendering of rows from the second half of the same file
        rows = list(csv.reader(io.StringIO("\n".join(lines[:20000]))))
        head, body = rows[0], rows[1 + len(rows) // 2:]
        md_head = "| " + " | ".join(head) + " |\n|" + "|".join(["---"] * len(head)) + "|\n"
        md = [l for l in md_head.rstrip("\n").split("\n")] + ["| " + " | ".join(r) + " |" for r in body]
        w = line_windows(md, per["md"])
        wins += w
        kinds += ["md"] * len(w)
    for url in JSONS:
        recs = json.loads(fetch(url))
        assert isinstance(recs, list) and isinstance(recs[0], dict), url
        w = line_windows([json.dumps(r) for r in recs[:20000]], per["jsonl"])
        wins += w
        kinds += ["jsonl"] * len(w)
    kinds = np.array(kinds)
    # 256 windows, as balanced over the three formats as the sources allow
    idx = []
    for kname, n in (("csv", 86), ("jsonl", 85), ("md", 85)):
        pool = np.where(kinds == kname)[0]
        idx += list(rng.permutation(pool)[:n])
    short = 256 - len(idx)
    if short > 0:
        rest = np.setdiff1d(np.arange(len(wins)), idx)
        idx += list(rng.permutation(rest)[:short])
    idx = np.array(sorted(idx))
    np.savez(f"{OUT}/tables.npz", ids=np.array([wins[i] for i in idx], dtype=np.int32), kind=kinds[idx])
    print("tables", len(idx), {k: int((kinds[idx] == k).sum()) for k in per})


# ----------------------------------------------------------------------------- vocabulary of single-token words


def word_vocab(n):
    """Lower-case alphabetic words (3-7 letters) for which both 'word' and ' word' are single tokens.
    Returns (ids with leading space, ids without), ordered by token id of the spaced form."""
    v = tok.get_vocab()
    out = []
    for t, i in sorted(v.items(), key=lambda x: x[1]):
        if t.startswith("Ġ") and t[1:].isalpha() and t[1:].islower() and t[1:].isascii() and 3 <= len(t) - 1 <= 7:
            if t[1:] in v and enc(" " + t[1:]) == [i] and enc(t[1:]) == [v[t[1:]]]:
                out.append((i, v[t[1:]]))
        if len(out) == n:
            break
    assert len(out) == n
    a = np.array(out)
    return a[:, 0], a[:, 1]


def build_grids():
    sp, nosp = word_vocab(256)
    nl = enc("\n")
    assert len(nl) == 1
    nl = nl[0]
    for C in (8, 16, 32, 64, 128, 256):
        ids = []
        for _ in range(64):
            n_rows = T // (C + 1) + 1
            cells = rng.integers(0, 256, size=(n_rows, C))
            row = np.concatenate([nosp[cells[:, :1]], sp[cells[:, 1:]], np.full((n_rows, 1), nl)], axis=1)
            seq = row.reshape(-1)[:T]
            ids.append(seq)
        ids = np.array(ids, dtype=np.int32)
        # token-regular check: the text round-trips to exactly these tokens
        for s in ids[:8]:
            assert enc(tok.decode(s)) == list(s), "grid does not round-trip through the tokenizer"
        np.savez(f"{OUT}/grid-{C}.npz", ids=ids, C=C, newline=nl)
    print("grids ok; sample:", repr(tok.decode(ids[0][:40])))
    np.savez(f"{OUT}/vocab.npz", spaced=word_vocab(1024)[0], unspaced=word_vocab(1024)[1])


if __name__ == "__main__":
    build_tables()
    build_grids()
    build_prose_code()
