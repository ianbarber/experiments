"""Where does cutting one pair beyond 90 degrees hurt? One layer at a time, then one head at a time in the
worst layers. Prose, first 64 windows; loss difference to dense over all tokens and over positions >= 1280.

python localise_pair.py [pair]      -> ../results/localise_pair<l>.json
"""

import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import keeptable as kt

l = int(sys.argv[1]) if len(sys.argv) > 1 else 23
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", dtype=torch.float32).cuda().eval()
theta = model.model.rotary_emb.inv_freq.detach().cpu()
cores = kt.patch_model(model)
ids = torch.from_numpy(np.load("../data/prose.npz")["ids"][:64]).long()
M = kt.make_table(f"pair-{l}-90", theta, 2048)


@torch.no_grad()
def loss():
    out = []
    for i in range(0, len(ids), 4):
        x = ids[i:i + 4].cuda()
        lp = torch.log_softmax(model(x).logits[:, :-1], -1)
        out.append(-lp.gather(-1, x[:, 1:, None])[..., 0].cpu())
    return torch.cat(out).numpy()


kt.set_condition(cores, kt.make_table("dense", theta, 2048))
L0 = loss()
res = {"pair": l, "n_windows": len(ids), "layers": [], "heads": {}}


def delta():
    d = loss() - L0
    return [float(d.mean()), float(d[:, 1279:].mean())]


kt.set_condition(cores, M)
res["all_layers"] = delta()
for layer in range(len(cores)):
    kt.set_condition(cores, M, layers={layer})
    res["layers"].append(delta())
    print("layer", layer, res["layers"][-1], flush=True)
worst = np.argsort([-d[1] for d in res["layers"]])[:3]
H = model.config.num_attention_heads
for layer in worst:
    res["heads"][int(layer)] = []
    for h in range(H):
        gate = torch.zeros(H, dtype=torch.bool, device="cuda")
        gate[h] = True
        kt.set_condition(cores, M, layers={int(layer)}, head_gate=gate)
        res["heads"][int(layer)].append(delta())
        print("layer", layer, "head", h, res["heads"][int(layer)][-1], flush=True)
json.dump(res, open(f"../results/localise_pair{l}.json", "w"), indent=1)
