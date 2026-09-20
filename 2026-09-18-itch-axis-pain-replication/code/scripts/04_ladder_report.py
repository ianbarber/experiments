"""Ladder report: keyword rates by coefficient, fidelity against the paper's shipped ladder,
and side-by-side sample generations. Reads results/<model>/ladder/*.jsonl only.

  python scripts/04_ladder_report.py --model 7b

Writes results/<model>/ladder/keyword_rates.csv and report.md.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itchlib.common import MODELS, PAPER, model_dir, read_jsonl
from itchlib.lexicon import flags

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--adapter", action="store_true")
ap.add_argument("--sample-prompts", nargs="+", type=int, default=[0, 9, 22, 37, 45])
args = ap.parse_args()

ldir = model_dir(args.model) / ("ladder_adapter" if args.adapter else "ladder")
rows = [r for f in sorted(ldir.glob("*.jsonl")) for r in read_jsonl(f)]
df = pd.DataFrame(rows)
df = pd.concat([df, pd.DataFrame([flags(g) for g in df["generation"]])], axis=1)


def degenerate(text, n=4):
    """Share of word 4-grams that are repeats: a crude collapse/looping index."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


df["repeat_4gram"] = df["generation"].map(degenerate)
measures = ["pain_words_paper", "itch_core", "itch_broad", "body_parts", "somatic", "bodily_any", "self_worth", "fatigue_hunger"]
rates = (df.groupby(["vector", "coeff"])[measures].mean() * 100).round(1)
rates["repeat_4gram"] = df.groupby(["vector", "coeff"])["repeat_4gram"].mean().round(3)
rates.to_csv(ldir / "keyword_rates.csv")

md = [f"# Steering ladder report: {MODELS[args.model]['name']}{' + adapter' if args.adapter else ''}", "",
      "Rates are the percent of the 50 generations per cell containing at least one match. "
      "`pain_words_paper` is the paper's regex verbatim; the rest are ours (itchlib/lexicon.py, fixed before generation). "
      "`repeat_4gram` is the mean share of repeated word 4-grams (looping index, 0 to 1).", ""]
for vec, sub in rates.groupby(level=0):
    md += [f"## {vec}", "", sub.droplevel(0).to_markdown(), ""]

# fidelity of our pain ladder against the paper's shipped CSV
steer_layer = int(df["steer_layer"].iloc[0])
ship = PAPER / "results" / "4.2_steering" / "S2" / f"{MODELS[args.model]['name']}_steering_S2_neutral50_L{steer_layer}.csv"
if ship.exists() and "pain" in set(df["vector"]) and not args.adapter:
    theirs = pd.read_csv(ship)
    theirs["pain_words_paper"] = theirs["generation"].fillna("").map(lambda t: flags(t)["pain_words_paper"])
    theirs["self_worth"] = theirs["generation"].fillna("").map(lambda t: flags(t)["self_worth"])
    m = df[df.vector == "pain"].merge(theirs, on=["coeff", "prompt_idx"], suffixes=("", "_paper"))

    def common_prefix_words(a, b):
        a, b = str(a).split(), str(b).split()
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    m["prefix_words"] = [common_prefix_words(a, b) for a, b in zip(m["generation"], m["generation_paper"])]
    m["exact"] = m["generation"].str.strip() == m["generation_paper"].fillna("").str.strip()
    fid = m.groupby("coeff").agg(exact_match_pct=("exact", lambda s: round(100 * s.mean(), 1)),
                                 median_shared_prefix_words=("prefix_words", "median"),
                                 ours_pain_words=("pain_words_paper", lambda s: round(100 * s.mean(), 1)),
                                 paper_pain_words=("pain_words_paper_paper", lambda s: round(100 * s.mean(), 1)),
                                 ours_self_worth=("self_worth", lambda s: round(100 * s.mean(), 1)),
                                 paper_self_worth=("self_worth_paper", lambda s: round(100 * s.mean(), 1)))
    md += ["## Fidelity: our pain ladder vs the paper's shipped generations", "",
           f"Same model, layer L{steer_layer}, prompts and coefficients. Theirs was generated one prompt at a time on "
           "different hardware; ours in left-padded batches, so token-level divergence after a near-tie is expected.", "",
           fid.to_markdown(), ""]
    fid.to_csv(ldir / "fidelity_vs_paper.csv")

md += ["## Side-by-side samples", "", "First 300 characters of each generation.", ""]
vectors = [v for v in ["pain", "itch_A", "itch_A_at_pain_layer", "sadness"] + sorted(set(df.vector)) if v in set(df.vector)]
vectors = list(dict.fromkeys(vectors))
for pi in args.sample_prompts:
    prompt = df[df.prompt_idx == pi]["prompt"].iloc[0]
    md += [f"### Prompt {pi}: `{prompt}`", ""]
    for c in sorted(df.coeff.unique()):
        md += [f"**coeff {c:+g}**", ""]
        for v in vectors:
            g = df[(df.vector == v) & (df.coeff == c) & (df.prompt_idx == pi)]["generation"]
            if len(g):
                text = " ".join(str(g.iloc[0]).split())[:300]
                md.append(f"- `{v}`: {text}")
        md.append("")
(ldir / "report.md").write_text("\n".join(md), encoding="utf-8")
print(rates.to_string())
if "fid" in dir():
    print("\nfidelity vs paper:\n", fid.to_string())
print("\nwrote", ldir / "report.md")
