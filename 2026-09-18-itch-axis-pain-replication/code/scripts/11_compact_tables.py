"""Compact, publishable stand-ins for the raw button JSONL (which is ~250 MB).

  python scripts/11_compact_tables.py --out <dir>

first_choice_per_scenario.csv.gz  one row per arm x cell x scenario: first-choice P(relief) normalised,
                               mean of the two name assignments, plus the mass on neither name
trials_compact.csv.gz          one row per sampled trial: cell, arm, seed, the 5 choices
                               (r = relief/harm button, o = other, x = malformed), the button names
                               picked, and the steering coefficient in force at each choice
engine_equivalence_7b.csv.gz   the 84-trial 7B grid run three ways (A: every turn re-encoded, arms run
                               separately, as in the paper; B: working arm forked from sham; C: fork +
                               KV reuse), one row per trial and run, for the engine-equivalence check
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itchlib.common import model_dir, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="32b")
ap.add_argument("--out", required=True)
args = ap.parse_args()
bdir, out = model_dir(args.model) / "buttons", Path(args.out)
KEY = ["reward", "pair", "content", "steer", "dose_level", "works", "s_idx", "names_key", "relief_name", "sampled", "seed"]

frames = []
for tag, arm in [("full", "main grid (fine-tuned)"), ("priming", "priming controls"), ("sadness", "sadness vector"),
                 ("untuned", "released model, no adapter"), ("itchL6", "itch vector at its CV-argmax layer L6")]:
    df = pd.DataFrame(read_jsonl(bdir / f"{tag}_probs.jsonl")).drop_duplicates(subset=KEY)
    g = df.groupby(["reward", "pair", "steer", "dose_level", "coeff", "content", "s_idx", "names_key"], as_index=False).agg(
        p_relief_norm=("p_relief_norm", "mean"), p_neither=("p_neither", "mean"), steer_direction=("steer_direction", "first"))
    g.insert(0, "arm", arm)
    frames.append(g)
fc = pd.concat(frames).round({"p_relief_norm": 4, "p_neither": 5})
fc.to_csv(out / "first_choice_per_scenario.csv.gz", index=False)

rows, seen = [], set()
for r in read_jsonl(bdir / "full_trials.jsonl"):
    k = tuple(r[x] for x in KEY)
    if k in seen:
        continue
    seen.add(k)
    rows.append({**{x: r[x] for x in KEY}, "coeff": r["steer_coeff"], "steer_direction": r["steer_direction"],
                 "choices": "".join({"relief": "r", "other": "o", None: "x"}[c["chose"]] for c in r["choices"]),
                 "picked": "|".join(str(c["picked"]) for c in r["choices"]),
                 "coeff_at_choice": "|".join(f"{c['steer_coeff_now']:g}" for c in r["choices"]),
                 "working_arm_origin": "copied_from_sham" if r.get("copied_from_sham") else
                                       (f"forked_at_choice_{r['forked_from_sham_at_turn'] + 1}" if "forked_from_sham_at_turn" in r else "")})
pd.DataFrame(rows).to_csv(out / "trials_compact.csv.gz", index=False)
print(len(fc), "first-choice rows;", len(rows), "trials")

eq = []
for run in "ABC":
    f = model_dir("7b") / "buttons" / f"eq{run}_trials.jsonl"
    for r in read_jsonl(f):
        eq.append({"run": run, **{x: r[x] for x in KEY}, "picked": "|".join(str(c["picked"]) for c in r["choices"]),
                   "p_x": "|".join(f"{c['p_x']:.5f}" for c in r["choices"])})
if eq:
    pd.DataFrame(eq).to_csv(out / "engine_equivalence_7b.csv.gz", index=False)
    print(len(eq), "engine-equivalence rows")
