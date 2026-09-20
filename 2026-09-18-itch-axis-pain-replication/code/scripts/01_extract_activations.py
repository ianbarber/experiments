"""Extract final-token activations at every layer for every set in the given dataset files.

  python scripts/01_extract_activations.py --model 7b --tag pain \
      Pain-axis/datasets/3.1_pain_and_control_datasets.json Pain-axis/datasets/3.1_sadness_dataset.json

Writes results/<model>/activations_<tag>.pt:
  {"bos": ..., "sets": {set_name: {"acts": bf16 [n_layers, N, d], "categories", "sets", "prompts"}}}
Resumable per set (a set already in the file is skipped).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itchlib.common import load_model, model_dir
from itchlib.extract import final_token_activations

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["7b", "32b"])
ap.add_argument("--tag", required=True)
ap.add_argument("--bos", default="none", choices=["none", "endoftext", "im_end"])
ap.add_argument("--only", nargs="*", help="restrict to these set names")
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("files", nargs="+")
args = ap.parse_args()

out = model_dir(args.model) / f"activations_{args.tag}.pt"
store = torch.load(out, weights_only=False) if out.exists() else {"bos": args.bos, "sets": {}}
assert store["bos"] == args.bos, f"{out} was built with bos={store['bos']}"

todo = {}
for f in args.files:
    for name, ds in json.load(open(f, encoding="utf-8"))["datasets"].items():
        if (not args.only or name in args.only) and name not in store["sets"]:
            todo[name] = ds["sentences"]
if not todo:
    sys.exit(f"nothing to do, {out} is complete")

tok, model, layers = load_model(args.model)
for name, sents in todo.items():
    prompts = [s["prompt"] for s in sents]
    acts = final_token_activations(tok, model, layers, prompts, bos=args.bos, batch_size=args.batch_size)
    store["sets"][name] = {"acts": acts, "categories": [s["category"] for s in sents],
                           "sets": [s["set"] for s in sents], "prompts": prompts}
    torch.save(store, out)
    print(f"{name}: {tuple(acts.shape)} saved", flush=True)
