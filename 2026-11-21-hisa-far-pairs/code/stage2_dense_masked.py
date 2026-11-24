"""Amendment 3: the fine-tuned dense Stage 2 models under keep-tables at inference (no further training).

Offset- and content-recall accuracy by distance on the Stage 2 held-out set (128 sequences, fp32).
python stage2_dense_masked.py      -> ../results/stage2_dense_masked.json
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import keeptable as kt
from stage2_task import DS, Vocab, make_batch

CONDS = (["dense"] + [f"hisa-{a}" for a in (45, 90, 180, 360, 720)] + list(kt.BANDS) + ["ctrl-perm-s0"]
         + [f"pair-{l}-90" for l in range(4)])           # single fast pairs: added after the band result
PATH = "../results/stage2_dense_masked.json"
eval_ids, eval_b, eval_m = make_batch(np.random.default_rng(10_000), Vocab(), 128)   # as in stage2_train.py
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", dtype=torch.float32).cuda().eval()
theta = model.model.rotary_emb.inv_freq.detach().cpu()
out = json.load(open(PATH)) if os.path.exists(PATH) else {}     # finished conditions are kept
patched = None
for seed in (42, 43):
    name = f"dense_lr0.0001_s{seed}"
    sd = torch.load(f"../raw/stage2/{name}.pt")
    model.load_state_dict(sd)
    if patched is None:
        patched = kt.patch_model(model)
    out.setdefault(name, {})
    for cond in CONDS:
        if cond in out[name]:
            continue
        kt.set_condition(patched, kt.make_table(cond, theta, 2048))
        ok = np.zeros(len(eval_m), dtype=bool)
        with torch.no_grad():
            for i in range(0, 128, 4):
                sel = np.where((eval_b >= i) & (eval_b < i + 4))[0]
                ids = torch.from_numpy(eval_ids[i:i + 4]).long().cuda()
                h = model.model(input_ids=ids).last_hidden_state
                pos = eval_m[sel, 0]
                b = eval_b[sel] - i
                pred = model.lm_head(h[torch.from_numpy(b).cuda(), torch.from_numpy(pos - 1).cuda()]).argmax(-1).cpu().numpy()
                ok[sel] = pred == eval_ids[eval_b[sel], pos]
        acc = {}
        for typ, tn in ((0, "offset"), (1, "content")):
            for di, D in enumerate(DS):
                q = (eval_m[:, 1] == typ) & (eval_m[:, 2] == di)
                if q.any():
                    acc[f"{tn}_{D}"] = [float(ok[q].mean()), int(q.sum())]
        out[name][cond] = acc
        print(name, f"{cond:14s}", "offset", " ".join(f"{acc[f'offset_{D}'][0]:.2f}" for D in DS),
              "| content", " ".join(f"{acc[f'content_{D}'][0]:.2f}" for D in DS[1:]), flush=True)
json.dump(out, open(PATH, "w"), indent=1)
