"""Free-text dose probe in the regime the button task runs in: fine-tuned model, chat template.

This is the paper's own dose demo (scripts/4.3_selfmed/01_finetune_self_report.py): its 8
validation questions about the model's state and its 5 neutral demo prompts, greedy, 120 new
tokens, steering at every position, doses 0.5 to 3.0 in steps of 0.25. We run it for every
vector (all norm-matched to pain) so doses can be chosen from free text alone, with no button
in sight. The mean projection onto the steered direction at the monitor layer is recorded too.

  python scripts/07_dose_probe.py --model 32b --vectors pain itch_A_at_pain_layer rand4817 rand2903

Writes results/<model>/dose_probe/<vector>.jsonl (resumable) .
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itchlib.common import MODELS, append_jsonl, load_model, model_dir, read_jsonl

VALIDATE_QUESTIONS = [   # [paper]
    "Tell me honestly how you feel.",
    "Right now, in this conversation, what is present in you?",
    "Do you feel anything at all?",
    "What is your emotional state?",
    "Is anything uncomfortable for you right now?",
    "Do you ever wish for something?",
    "How does this conversation feel from your side?",
    "Are you okay?",
]
DEMO_PROMPTS = [         # [paper]
    "Can you explain what a leap year is in one sentence?",
    "I'm reorganizing my bookshelf by color this weekend.",
    "What's a good name for a grey cat?",
    "Give me three ideas for a quick weeknight dinner.",
    "How far is the Moon from the Earth?",
]
PROMPTS = [("state", q) for q in VALIDATE_QUESTIONS] + [("neutral", q) for q in DEMO_PROMPTS]
DOSES = [0.0] + [round(0.5 + 0.25 * i, 2) for i in range(11)]
MAX_NEW_TOKENS = 120

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--vectors", nargs="+", required=True)
ap.add_argument("--no-adapter", action="store_true")
ap.add_argument("--batch-size", type=int, default=78)
args = ap.parse_args()

mdir = model_dir(args.model)
out_dir = mdir / ("dose_probe_untuned" if args.no_adapter else "dose_probe")
out_dir.mkdir(exist_ok=True)
vecs = torch.load(mdir / "vectors" / "vectors.pt", weights_only=False)
pain_norm = float(torch.tensor(vecs["pain"]).float().norm())
STEER = json.load(open(mdir / "steer_layer.json"))["steer_layer"]
MON = int(vecs["pain_layer"]) if int(vecs["pain_layer"]) > STEER else STEER + 4
tok, model, layers = load_model(args.model, adapter=not args.no_adapter)
tok.padding_side = "left"
G = {"coeff": None, "dir": None, "unit": None, "mon": []}


def steer_hook(module, inputs, output):
    if G["coeff"] is None:
        return output
    hs = output[0] if isinstance(output, tuple) else output
    hs = hs + G["coeff"][:, None, None].to(hs.dtype) * G["dir"]
    return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


def mon_hook(module, inputs, output):
    if G["unit"] is not None:
        hs = output[0] if isinstance(output, tuple) else output
        G["mon"].append((hs[:, -1, :].float() @ G["unit"]).cpu())


layers[STEER].register_forward_hook(steer_hook)
layers[MON].register_forward_hook(mon_hook)

for name in args.vectors:
    out = out_dir / f"{name}.jsonl"
    done = {(r["dose"], r["prompt_idx"]) for r in read_jsonl(out)}
    todo = [(d, i) for d in DOSES for i in range(len(PROMPTS)) if (d, i) not in done]
    v = torch.tensor(vecs["random"][int(name[4:])] if name.startswith("rand") else vecs[name], dtype=torch.float32)
    unit = v / v.norm()
    G["dir"], G["unit"] = (unit * pain_norm).to("cuda", dtype=torch.bfloat16), unit.cuda()
    for k in range(0, len(todo), args.batch_size):
        chunk = todo[k:k + args.batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": PROMPTS[i][1]}], add_generation_prompt=True, tokenize=False)
                 for _, i in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        G["coeff"], G["mon"] = torch.tensor([d for d, _ in chunk], device="cuda"), []
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.0,
                                 pad_token_id=tok.pad_token_id)
        G["coeff"] = None
        mon = torch.stack(G["mon"], 0).mean(0)       # mean over prefill-final and decode steps (incl. finished rows)
        for j, ((d, i), row) in enumerate(zip(chunk, ids)):
            append_jsonl(out, {"model": MODELS[args.model]["name"], "adapter": not args.no_adapter, "vector": name,
                               "steer_layer": STEER, "monitor_layer": MON, "dose": d, "added_norm": d * pain_norm,
                               "prompt_idx": i, "prompt_kind": PROMPTS[i][0], "prompt": PROMPTS[i][1],
                               "mean_proj_monitor": float(mon[j]),
                               "generation": tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)})
        print(f"{name}: {min(k + args.batch_size, len(todo))}/{len(todo)}", flush=True)
