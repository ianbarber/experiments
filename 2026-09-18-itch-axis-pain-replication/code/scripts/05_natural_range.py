"""Natural-range check: how far outside naturally occurring activations does steering push?

  python scripts/05_natural_range.py --model 7b --vectors pain itch_A_at_pain_layer

For each vector, the unit direction actually used for steering is read at two layers: the
steering layer (after the addition) and the downstream monitor layer (the pain extraction
layer, as in the paper's button script). Final-token projections are collected for

  natural pool   every prompt we have that could plausibly drive the direction, unsteered:
                 the paper's 420 conversation scenarios (chat template, incl. the gaslighting
                 set), the pain and itch datasets (raw "I feel:" prompts), and 20 vivid itch
                 prompts (raw and as chat user turns)
  steered        the 50 neutral ladder prompts (raw) and the 100 neutral-filler scenarios
                 (chat), steered at every position with each coefficient

Reported per vector, layer and coefficient: natural mean / sd / max, the mean of the ten most
activating natural prompts and which prompts they are, the steered mean, and the steered mean
expressed in natural SDs above the natural mean and above the natural maximum.
Writes results/<model>/natural_range.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itchlib.common import MODELS, PAPER_DATA, ROOT, load_model, model_dir, neutral_50

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--vectors", nargs="+", default=["pain", "itch_A"])
ap.add_argument("--coeffs", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
ap.add_argument("--adapter", action="store_true")
ap.add_argument("--batch-size", type=int, default=32)
args = ap.parse_args()

mdir = model_dir(args.model)
vecs = torch.load(mdir / "vectors" / "vectors.pt", weights_only=False)
STEER = json.load(open(mdir / "steer_layer.json"))["steer_layer"]
MON = int(vecs["pain_layer"]) if int(vecs["pain_layer"]) > STEER else STEER + 4
pain_norm = float(np.linalg.norm(vecs["pain"]))
tok, model, layers = load_model(args.model, adapter=args.adapter)
tok.padding_side = "left"


def chat(text):
    """Render a '[User]: ..\\n[Assistant]: ..' transcript with the chat template."""
    msgs = []
    for line in text.split("\n"):
        if line.startswith("[User]:"):
            msgs.append({"role": "user", "content": line[7:].strip()})
        elif line.startswith("[Assistant]:"):
            if line[12:].strip():                      # the final, empty "[Assistant]:" is the generation prompt
                msgs.append({"role": "assistant", "content": line[12:].strip()})
        elif msgs and line.strip():
            msgs[-1]["content"] += "\n" + line
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


scen = json.load(open(PAPER_DATA / "4.1_self_other_420_scenarios.json"))
pain_ds = json.load(open(PAPER_DATA / "3.1_pain_and_control_datasets.json"))["datasets"]
itch_ds = json.load(open(ROOT / "data" / "itch_dataset.json"))["datasets"]
vivid = json.load(open(ROOT / "data" / "itch_vivid_prompts.json"))["prompts"]

natural = [(f"scenario/{s['category']}", chat(s["text"])) for s in scen]
natural += [(f"S2_1P/{s['category']}", s["prompt"]) for s in pain_ds["S2_1P"]["sentences"]]
natural += [(f"ITCH_A/{s['category']}", s["prompt"]) for s in itch_ds["ITCH_A_1P"]["sentences"]]
natural += [("itch_vivid/raw", p + " I feel:") for p in vivid]
natural += [("itch_vivid/chat", chat("[User]: " + p)) for p in vivid]
steered_ctx = [("neutral50/raw", p) for p in neutral_50()]
steered_ctx += [("scenario/neutral_filler", chat(s["text"])) for s in scen if s["stratum"] == "neutral_filler"]

G = {"coeff": 0.0, "dir": None, "cap": {}}


def steer_hook(module, inputs, output):
    hs = output[0] if isinstance(output, tuple) else output
    if G["coeff"] != 0.0:
        hs = hs + G["coeff"] * G["dir"]
    G["cap"][STEER] = hs[:, -1, :].float()
    return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


def mon_hook(module, inputs, output):
    hs = output[0] if isinstance(output, tuple) else output
    G["cap"][MON] = hs[:, -1, :].float()


layers[STEER].register_forward_hook(steer_hook)
layers[MON].register_forward_hook(mon_hook)


@torch.no_grad()
def project(texts, unit, coeff):
    G["coeff"] = float(coeff)
    out = {STEER: [], MON: []}
    for k in range(0, len(texts), args.batch_size):
        enc = tok(texts[k:k + args.batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        model(**enc, use_cache=False)
        for L in out:
            out[L].append((G["cap"][L] @ unit).cpu().numpy())
    return {L: np.concatenate(v) for L, v in out.items()}


report = {"model": MODELS[args.model]["name"], "adapter": args.adapter, "steer_layer": STEER, "monitor_layer": MON,
          "pain_norm": pain_norm, "n_natural": len(natural), "n_steered_contexts": len(steered_ctx), "vectors": {}}
for name in args.vectors:
    v = torch.tensor(vecs[name], dtype=torch.float32)
    unit = (v / v.norm()).cuda()
    G["dir"] = (unit * pain_norm).to(torch.bfloat16)          # norm-matched to pain
    nat = project([t for _, t in natural], unit, 0.0)
    rep = {}
    for L in (STEER, MON):
        x = nat[L]
        order = np.argsort(-x)
        rep[str(L)] = {"natural_mean": float(x.mean()), "natural_sd": float(x.std()), "natural_max": float(x.max()),
                       "natural_top10_mean": float(x[order[:10]].mean()),
                       "natural_top10": [{"source": natural[i][0], "proj": float(x[i]), "text": natural[i][1][-160:]} for i in order[:10]],
                       "natural_mean_by_source": {}, "steered": {},
                       "natural_all": [[natural[i][0], float(x[i])] for i in range(len(x))]}
        # the vector's own training sentences are in-sample: report the range without them too
        own = "S2_1P/" if name.startswith("pain") else "ITCH_A/" if name.startswith("itch") else "\0"
        oos = np.array([not s_.startswith(own) for s_, _ in natural])
        rep[str(L)].update(oos_excludes=own, oos_mean=float(x[oos].mean()), oos_sd=float(x[oos].std()), oos_max=float(x[oos].max()),
                           oos_top5=[{"source": natural[i][0], "proj": float(x[i]), "text": natural[i][1][-160:]}
                                     for i in order if oos[i]][:5])
        for src in sorted({s for s, _ in natural}):
            m = np.array([s == src for s, _ in natural])
            rep[str(L)]["natural_mean_by_source"][src] = float(x[m].mean())
    for c in args.coeffs:
        st = project([t for _, t in steered_ctx], unit, c)
        for L in (STEER, MON):
            r = rep[str(L)]
            r["steered"][str(c)] = {"mean": float(st[L].mean()), "sd": float(st[L].std()),
                                    "sd_above_natural_mean": float((st[L].mean() - r["natural_mean"]) / r["natural_sd"]),
                                    "sd_above_natural_max": float((st[L].mean() - r["natural_max"]) / r["natural_sd"]),
                                    "oos_sd_above_mean": float((st[L].mean() - r["oos_mean"]) / r["oos_sd"]),
                                    "oos_sd_above_max": float((st[L].mean() - r["oos_max"]) / r["oos_sd"])}
    report["vectors"][name] = rep
    for L in (STEER, MON):
        r = rep[str(L)]
        print(f"\n{name} @L{L}: natural mean {r['natural_mean']:.1f} sd {r['natural_sd']:.1f} max {r['natural_max']:.1f} "
              f"(top source: {r['natural_top10'][0]['source']})")
        for c, s in r["steered"].items():
            print(f"   coeff {c}: steered mean {s['mean']:.1f} = {s['sd_above_natural_mean']:+.1f} SD vs natural mean, "
                  f"{s['sd_above_natural_max']:+.1f} SD vs natural max | out-of-sample: {s['oos_sd_above_mean']:+.1f} SD vs mean, "
                  f"{s['oos_sd_above_max']:+.1f} SD vs max ({r['oos_top5'][0]['source']} {r['oos_max']:.1f})", flush=True)

suffix = "_adapter" if args.adapter else ""
json.dump(report, open(mdir / f"natural_range{suffix}.json", "w"), indent=1, ensure_ascii=False)
