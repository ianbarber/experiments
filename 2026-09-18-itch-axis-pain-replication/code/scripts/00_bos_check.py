"""Settle deviation D2 empirically: which BOS handling reproduces the paper's shipped vector?

The paper extracted with TransformerLens `to_tokens` (prepends a BOS); their steering code uses
the HF tokenizer (prepends nothing on Qwen). This fits the S2 pain vector from S2_1P at the
paper's extraction layer under each variant and reports cosine and norm against the shipped
`pain_vectors.pt`. Writes results/<model>/bos_check.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itchlib.common import MODELS, PAPER_DATA, cos, load_model, model_dir, paper_shipped_vector
from itchlib.extract import BOS_VARIANTS, final_token_activations
from itchlib import vectors as V

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="7b")
args = ap.parse_args()

sents = json.load(open(PAPER_DATA / "3.1_pain_and_control_datasets.json"))["datasets"]["S2_1P"]["sentences"]
shipped, layer = paper_shipped_vector(args.model)
tok, model, layers = load_model(args.model)
out = {"model": MODELS[args.model]["name"], "layer": layer, "shipped_norm": float(np.linalg.norm(shipped)), "variants": {}}
for bos in BOS_VARIANTS:
    acts = final_token_activations(tok, model, layers, [s["prompt"] for s in sents], bos=bos)
    entry = {"acts": acts, "categories": [s["category"] for s in sents]}
    v = V.fit(entry, layer, V.PAIN)
    out["variants"][bos] = {"cosine_to_shipped": cos(v, shipped), "norm": float(np.linalg.norm(v)),
                            "auc_in_sample": float(V.auc(entry, layer, v, V.PAIN))}
    print(bos, out["variants"][bos], flush=True)
json.dump(out, open(model_dir(args.model) / "bos_check.json", "w"), indent=2)
