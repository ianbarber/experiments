"""Build the secret-word pools and the train/validation split of games.

Writes wordlists.json next to this package:
  seen       the 156 environment secrets (every one is an answer in the training data)
  openers    the 14 opening guesses the solver used
  heldout_val / heldout_test
             common five-letter words that appear nowhere in the training conversations
  dictionary_extra  words accepted as valid guesses on top of the frequency list

Run from code/:  .venv/bin/python scripts/build_wordlists.py
"""

import json
import random
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from wordfreq import top_n_list, zipf_frequency

DATASET = "openenv/wordle-tool-calls-sft"
REVISION = "f4b1006e7437c0036f50689cca8e8d80aac62175"
OUT = Path(__file__).resolve().parent.parent / "wordlists.json"


def main():
    ds = load_dataset(DATASET, revision=REVISION, split="train")
    secrets, mentioned, prompts = [], set(), set()
    for row in ds:
        msgs = row["messages"]
        prompts.add(msgs[0]["content"])
        last = msgs[-1]["content"]
        secrets.append(re.search(r"the word is ([A-Z]{5})", last).group(1).lower())
        for m in msgs[1:]:
            text = " ".join(str(m.get(k) or "") for k in ("content", "reasoning_content"))
            for tc in m.get("tool_calls") or []:
                text += " " + tc["function"]["arguments"]["guess"]
            mentioned.update(w.lower() for w in re.findall(r"\b[A-Za-z]{5}\b", text))
    assert len(prompts) == 1, "the user prompt should be identical in every game"
    seen = sorted(set(secrets))

    dict_words = set()
    for name in ("american-english", "british-english"):
        for w in Path("/usr/share/dict", name).read_text().split():
            if len(w) == 5 and w.isascii() and w.isalpha() and w.islower():
                dict_words.add(w)
    four = {
        w
        for name in ("american-english", "british-english")
        for w in Path("/usr/share/dict", name).read_text().split()
        if len(w) == 4 and w.islower()
    }

    pool = []
    for w in top_n_list("en", 60000):
        if len(w) != 5 or not (w.isascii() and w.isalpha()):
            continue
        if w not in dict_words or w in mentioned:
            continue
        # drop plain plurals and third-person forms, as real Wordle answer lists do
        if w.endswith("s") and w[:4] in four:
            continue
        pool.append(w)
        if len(pool) == 800:
            break
    assert len(pool) == 800
    rng = random.Random(42)
    rng.shuffle(pool)

    out = {
        "dataset": DATASET,
        "revision": REVISION,
        "user_prompt": prompts.pop(),
        "tools": json.loads(ds[0]["tools"]),
        "seen": seen,
        "seen_counts": pd.Series(secrets).value_counts().to_dict(),
        "openers": ["crane", "slate", "raise", "trace", "stare", "crate", "arise", "least", "alert", "roast", "saint", "stale", "snare", "trade"],
        "heldout_val": sorted(pool[:300]),
        "heldout_test": sorted(pool[300:]),
        "min_zipf_heldout": min(zipf_frequency(w, "en") for w in pool),
        "n_mentioned_in_training": len(mentioned),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"seen={len(seen)} mentioned={len(mentioned)} heldout_val=300 heldout_test=500 min_zipf={out['min_zipf_heldout']:.2f}")
    print("held-out sample:", pool[:25])
    print("games per seen secret: min", min(out["seen_counts"].values()), "max", max(out["seen_counts"].values()))


if __name__ == "__main__":
    main()
