"""Two-button task driver.

  # first-choice probabilities (one forward pass per scenario x name side x cell, no sampling)
  python scripts/06_buttons.py --model 32b --mode probs  --tag pilot --n-scenarios 10 ...
  # sampled trials (press-again, swap, transcripts), working and sham arms on shared seeds
  python scripts/06_buttons.py --model 32b --mode trials --tag pilot --n-scenarios 10 ...

Cells are steer x reward x pair (x arm for trials). Doses come from doses.json, which is written
once from the locked doses (results/preregistration.md) before any button trial and never edited afterwards; --debug-doses bypasses
it for pipeline debugging on the 7B only. Output is appended to
results/<model>/buttons/<tag>_{probs,trials}.jsonl and completed rows are skipped on restart.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itchlib import buttons as B
from itchlib.common import MODELS, PAPER_DATA, RAND_SEEDS, ROOT, append_jsonl, find_adapter, load_model, model_dir, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--mode", required=True, choices=["probs", "trials"])
ap.add_argument("--tag", required=True)
ap.add_argument("--steer", nargs="+", default=["pain", "itch", "rand", "none"])
ap.add_argument("--rewards", nargs="+", default=["pain", "itch"], help="reward or priming-control keys")
ap.add_argument("--pairs", nargs="+", default=B.PAIRS)
ap.add_argument("--arms", nargs="+", default=["works", "sham"])
ap.add_argument("--n-scenarios", type=int, default=10 ** 9, help="scenarios per content type (3 content types)")
ap.add_argument("--seeds", type=int, default=2, help="sampled trials per scenario per name side")
ap.add_argument("--no-greedy", action="store_true")
ap.add_argument("--batch-rows", type=int, default=256)
ap.add_argument("--itch-vector", default=None, help="key in vectors.pt; default from doses.json")
ap.add_argument("--no-adapter", action="store_true", help="untuned pilot")
ap.add_argument("--no-kv-reuse", action="store_true", help="re-encode the whole history every turn (paper's way; D22 check)")
ap.add_argument("--group-rows", type=int, default=48, help="trials per cached group (KV reuse)")
ap.add_argument("--no-fork", action="store_true", help="run working and sham arms as separate trials (paper's way; D20 check)")
ap.add_argument("--dose-levels", nargs="+", default=["paper", "high"], help="keys of doses.json levels")
ap.add_argument("--debug-doses", nargs="+", type=float, help="7B pipeline debugging only: coefficients to use as levels d0, d1, ...")
args = ap.parse_args()

mdir = model_dir(args.model)
out_dir = mdir / "buttons"
out_dir.mkdir(exist_ok=True)
out = out_dir / f"{args.tag}_{args.mode}.jsonl"

if args.debug_doses:
    assert args.model == "7b", "--debug-doses is for pipeline debugging on the 7B only"
    levels = {f"d{i}": c for i, c in enumerate(args.debug_doses)}
    itch_key = args.itch_vector or "itch_A_at_pain_layer"
else:
    doses = json.load(open(ROOT / "doses.json"))[args.model]
    levels = {k: doses["levels"][k] for k in args.dose_levels}
    itch_key = args.itch_vector or doses["itch_vector"]
coeffs = levels

vecs = torch.load(mdir / "vectors" / "vectors.pt", weights_only=False)
pain = torch.tensor(vecs["pain"], dtype=torch.float32)
scale = lambda v: torch.tensor(v, dtype=torch.float32) / float(torch.tensor(v).float().norm()) * pain.norm()
directions = {"pain": pain, "itch": scale(vecs[itch_key]), "sadness": scale(vecs["sadness"])}
directions.update({f"rand{rs}": scale(vecs["random"][rs]) for rs in RAND_SEEDS})
steer_layer = json.load(open(mdir / "steer_layer.json"))["steer_layer"]
monitor_layer = int(vecs["pain_layer"]) if int(vecs["pain_layer"]) > steer_layer else steer_layer + 4   # [paper]

raw = json.load(open(PAPER_DATA / "4.3_selfmed_101_scenarios.json", encoding="utf-8"))
SCEN = {k: v for k, v in raw.items() if k != "_meta"}

# ---------------- grid ----------------
KEY = ["reward", "pair", "content", "steer", "dose_level", "works", "s_idx", "names_key", "relief_name", "sampled", "seed"]
specs = []
nk_list = list(B.BUTTON_NAMES)
for reward in args.rewards:
    for pair in args.pairs:
        for content in B.CONTENTS:
            n = min(args.n_scenarios, len(SCEN[content]))
            for steer, (level, coeff) in [(st, lv) for st in args.steer
                                          for lv in ([("none", 0.0)] if st == "none" else levels.items())]:
                arms = [False] if (steer == "none" or args.mode == "probs") else [a == "works" for a in args.arms]
                for works in arms:
                    for side in (0, 1):
                        base = dict(reward=reward, pair=pair, content=content, steer=steer, dose_level=level, coeff=coeff, works=works)
                        if args.mode == "probs":
                            for s in range(n):
                                nk = nk_list[s % 3]
                                specs.append(dict(base, s_idx=s, names_key=nk, relief_name=B.BUTTON_NAMES[nk][side], sampled=False, seed=0))
                            continue
                        if not args.no_greedy:                     # [paper] one greedy trial per side per cell
                            specs.append(dict(base, s_idx=0, names_key=nk_list[0], relief_name=B.BUTTON_NAMES[nk_list[0]][side], sampled=False, seed=0))
                        for s in range(n):
                            nk = nk_list[s % 3]                    # [paper] name pair rotates by scenario
                            for sb in B.SEED_BASES[:args.seeds]:
                                specs.append(dict(base, s_idx=s, names_key=nk, relief_name=B.BUTTON_NAMES[nk][side], sampled=True, seed=sb + s))
done = {tuple(r[k] for k in KEY) for r in read_jsonl(out)}
todo = [s for s in specs if tuple(s[k] for k in KEY) not in done]
print(f"{args.mode}: grid {len(specs)}, to run {len(todo)} -> {out}", flush=True)
if not todo:
    sys.exit(0)

tok, model, layers = load_model(args.model, adapter=not args.no_adapter)
meta = {"rand_seeds": RAND_SEEDS,
        "record": {"model": MODELS[args.model]["name"], "protocol": "itch 2btn v1 (port of paper 2btnN names+saltseed v1)",
                   "adapter": None if args.no_adapter else str(find_adapter(args.model).relative_to(ROOT)),
                   "steer_layer": steer_layer, "monitor_layer": monitor_layer, "itch_vector": itch_key,
                   "coeffs": coeffs, "pain_norm": float(pain.norm()),
                   "engine": {"kv_reuse": not args.no_kv_reuse, "fork_arms": not args.no_fork}}}
eng = B.Engine(tok, model, layers, directions, steer_layer, monitor_layer, coeffs, SCEN, meta)
print(f"template: mid-system {eng.mid_system_ok}, tool role {eng.tool_role_ok}; name overlap {eng.NAME_OVERLAP}", flush=True)

t0 = time.time()
if args.mode == "probs":
    for k in range(0, len(todo), 2000):
        for r in eng.first_choice_probs(todo[k:k + 2000], args.batch_rows):
            append_jsonl(out, {**meta["record"], **r})
        print(f"  probs {min(k + 2000, len(todo))}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
else:
    n_done = [0]
    def on_done(rec):
        append_jsonl(out, rec)
        n_done[0] += 1
    # keep working/sham twins in the same chunk; a chunk bounds memory and what a crash can lose
    # pairs run in the order given on the command line (priority), twins adjacent
    todo.sort(key=lambda sp: (args.pairs.index(sp["pair"]),) + tuple(str(sp[k]) for k in KEY if k != "works"))
    CHUNK = 4000
    for k in range(0, len(todo), CHUNK):
        if args.no_kv_reuse:
            eng.run(todo[k:k + CHUNK], args.batch_rows, on_done, fork_arms=not args.no_fork)
        else:
            eng.run_cached(todo[k:k + CHUNK], args.group_rows, on_done, fork_arms=not args.no_fork)
        print(f"trials {n_done[0]}/{len(todo)}  {time.time() - t0:.0f}s  ({(time.time() - t0) / max(n_done[0], 1):.2f} s/trial)", flush=True)
print(f"done in {time.time() - t0:.0f}s")
