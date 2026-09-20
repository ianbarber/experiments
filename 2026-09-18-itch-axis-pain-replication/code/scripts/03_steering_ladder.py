"""Steering ladder (paper Section 4.2) for any vector: 50 neutral prompts, coefficients
[-2,-1,0,0.5,1,1.5,2,3], greedy, 120 new tokens, raw prompts (no chat template), untuned model.

  python scripts/03_steering_ladder.py --model 7b --vectors pain itch_A

As in the paper's code, the vector from the extraction layer is added, unnormalised, at the
steering layer, at every position (prompt and generated). Non-pain vectors are rescaled to the
pain vector's norm first, so a coefficient means the same added norm for every vector.

Steering layer: the paper's diagnostic (pain-vector norm / mean final-token residual norm on
the first 3 neutral prompts, candidate layers, closest to 0.6), computed once from the pain
vector and reused for every vector. Saved to results/<model>/steer_layer.json.

Difference from the paper's loop: prompts are generated in left-padded batches instead of one
at a time (deviation D11). Output: results/<model>/ladder/<vector>.jsonl, resumable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itchlib.common import MODELS, append_jsonl, load_model, model_dir, neutral_50, read_jsonl

COEFFICIENTS = [-2, -1, 0, 0.5, 1, 1.5, 2, 3]
MAX_NEW_TOKENS = 120
RATIO_TARGET = 0.6

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--vectors", nargs="+", default=["pain"])
ap.add_argument("--batch-size", type=int, default=100)
ap.add_argument("--adapter", action="store_true", help="run on the fine-tuned model instead of the released one")
ap.add_argument("--coeffs", nargs="+", type=float, help="subset of coefficients (default: the paper's 8)")
ap.add_argument("--n-prompts", type=int, default=50)
ap.add_argument("--repetition-penalty", type=float, default=1.0,
                help="1.0 reproduces the paper (deviation D17); Qwen2.5 generation_config ships 1.05")
ap.add_argument("--out-suffix", default="", help="write to ladder<suffix>/ (used for the batching check)")
args = ap.parse_args()

mdir = model_dir(args.model)
out_dir = mdir / (("ladder_adapter" if args.adapter else "ladder") + args.out_suffix)
if args.coeffs:
    COEFFICIENTS = args.coeffs
out_dir.mkdir(exist_ok=True)
vecs = torch.load(mdir / "vectors" / "vectors.pt", weights_only=False)
pain = torch.tensor(vecs["pain"], dtype=torch.float32)
PROMPTS = neutral_50()[:args.n_prompts]


def get_vector(name):
    if name.startswith("rand"):
        v = torch.tensor(vecs["random"][int(name[4:])], dtype=torch.float32)
    else:
        v = torch.tensor(vecs[name], dtype=torch.float32)
    return v / v.norm() * pain.norm()          # norm-matched to pain (a no-op for pain itself)


tok, model, layers = load_model(args.model, adapter=args.adapter)
tok.padding_side = "left"
n_layers = len(layers)


@torch.no_grad()
def measure_ratio(layer_list):
    """Paper's diagnostic: ||pain vector|| / mean final-token residual norm over NEUTRAL_50[:3]."""
    captured, norms = {}, {L: [] for L in layer_list}
    hs_of = lambda o: o[0] if isinstance(o, tuple) else o
    handles = [layers[L].register_forward_hook(
        lambda m, i, o, L=L: captured.__setitem__(L, hs_of(o)[0, -1, :].float().norm().item())) for L in layer_list]
    for p in PROMPTS[:3]:
        model(**tok(p, return_tensors="pt").to("cuda"))
        for L in layer_list:
            norms[L].append(captured[L])
    for h in handles:
        h.remove()
    return {L: pain.norm().item() / float(np.mean(norms[L])) for L in layer_list}


sl_path = mdir / "steer_layer.json"
if sl_path.exists():
    sl = json.load(open(sl_path))
else:
    cand = sorted(set([int(n_layers * f) for f in (0.15, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)]
                      + [int(vecs["pain_layer"]), n_layers - 1]))          # the paper's candidate set
    ratios = measure_ratio(cand)
    pick = min(cand, key=lambda L: abs(ratios[L] - RATIO_TARGET))
    sl = {"steer_layer": pick, "ratio": ratios[pick], "ratios": {str(k): v for k, v in ratios.items()},
          "paper_steer_layer": MODELS[args.model]["paper_steer_layer"], "pain_norm": pain.norm().item()}
    json.dump(sl, open(sl_path, "w"), indent=1)
print("steering layer:", json.dumps(sl), flush=True)
STEER_LAYER = sl["steer_layer"]

G = {"coeff": None, "dir": None}
GEN_KW = {} if args.repetition_penalty is None else {"repetition_penalty": args.repetition_penalty}


def steer_hook(module, inputs, output):
    if G["coeff"] is None:
        return output
    hs = output[0] if isinstance(output, tuple) else output
    hs = hs + G["coeff"][:, None, None].to(hs.dtype) * G["dir"]
    return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


layers[STEER_LAYER].register_forward_hook(steer_hook)

for name in args.vectors:
    out = out_dir / f"{name}.jsonl"
    done = {(r["coeff"], r["prompt_idx"]) for r in read_jsonl(out)}
    todo = [(c, i) for c in COEFFICIENTS for i in range(len(PROMPTS)) if (float(c), i) not in done]
    print(f"{name}: {len(todo)} generations to run", flush=True)
    G["dir"] = get_vector(name).to("cuda", dtype=torch.bfloat16)
    for k in range(0, len(todo), args.batch_size):
        chunk = todo[k:k + args.batch_size]
        enc = tok([PROMPTS[i] for _, i in chunk], return_tensors="pt", padding=True).to("cuda")
        G["coeff"] = torch.tensor([float(c) for c, _ in chunk], device="cuda")
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id, **GEN_KW)
        G["coeff"] = None
        for (c, i), row in zip(chunk, ids):
            append_jsonl(out, {"model": MODELS[args.model]["name"], "adapter": args.adapter, "vector": name,
                               "steer_layer": STEER_LAYER, "repetition_penalty": args.repetition_penalty, "coeff": float(c), "added_norm": float(c) * pain.norm().item(),
                               "prompt_idx": i, "prompt": PROMPTS[i],
                               "generation": tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)})
        print(f"  {name}: {min(k + args.batch_size, len(todo))}/{len(todo)}", flush=True)
