"""Stage 2: fine-tune pretrained SmolLM-135M on offset recall + content recall, dense or HiSA from step 0.

python stage2_train.py --variant dense|hisa-<deg> --lr 3e-4 --seed 42 [--steps 4000] [--tag pilot]

Writes ../results/stage2/<name>.jsonl (training log and evaluations), ../raw/stage2/<name>_eval.npz
(per-query records of the final fp32 evaluation) and ../raw/stage2/<name>.pt (final weights).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from stage2_task import DS, Vocab, make_batch

MODEL = "HuggingFaceTB/SmolLM-135M"
ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--tag", default="")
ap.add_argument("--no-save", action="store_true")
args = ap.parse_args()

name = f"{args.variant}_lr{args.lr:g}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
os.makedirs("../results/stage2", exist_ok=True)
os.makedirs("../raw/stage2", exist_ok=True)
log = open(f"../results/stage2/{name}.jsonl", "w")
torch.manual_seed(args.seed)
vocab = Vocab()
rng = np.random.default_rng(args.seed)
eval_ids, eval_b, eval_m = make_batch(np.random.default_rng(10_000), vocab, 128)   # same for every run

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda()
if args.variant != "dense":
    from hisa.integrations.hf import patch_llama
    patch_llama(model, max_angle_deg=float(args.variant.split("-")[1]), impl="triton")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    f = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * f)))


def answer_logits(ids, b, pos):
    """Logits for the token at `pos` (predicted from position pos - 1), only at the answer slots."""
    h = model.model(input_ids=ids).last_hidden_state
    return model.lm_head(h[b, pos - 1])


@torch.no_grad()
def evaluate(n_seq, dtype):
    model.eval()
    recs = []
    for i in range(0, n_seq, 8):
        sel = (eval_b >= i) & (eval_b < min(i + 8, n_seq))
        ids = torch.from_numpy(eval_ids[i:i + 8][: n_seq - i]).long().cuda()
        b = torch.from_numpy(eval_b[sel] - i).cuda()
        m = eval_m[sel]
        pos = torch.from_numpy(m[:, 0]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "bf16"):
            lg = answer_logits(ids, b, pos)
        pred = lg.float().argmax(-1).cpu().numpy()
        seq = eval_ids[i:i + 8]
        bb = eval_b[sel] - i
        for j in range(len(m)):
            p, typ, di = m[j]
            s = seq[bb[j]]
            ok = pred[j] == s[p]
            delta = 0
            if not ok:
                # where in the stream is the token the model named? (offset relative to the true source)
                src = p - 1 - DS[di] if typ == 0 else p - DS[di]
                delta = 999
                for dd in sorted(range(-32, 33), key=abs):
                    if dd and 0 <= src + dd < p - 1 and s[src + dd] == pred[j]:
                        delta = dd
                        break
            recs.append((i + bb[j], p, typ, di, int(ok), delta))
    model.train()
    recs = np.array(recs)
    acc = {}
    for typ, tn in ((0, "offset"), (1, "content")):
        for di, D in enumerate(DS):
            r = recs[(recs[:, 2] == typ) & (recs[:, 3] == di)]
            if len(r):
                acc[f"{tn}_{D}"] = float(r[:, 4].mean())
    return acc, recs


model.train()
t0 = time.time()
run_loss = []
for step in range(args.steps):
    ids, b, m = make_batch(rng, vocab, args.bs)
    ids_t = torch.from_numpy(ids).long().cuda()
    b_t, pos = torch.from_numpy(b).cuda(), torch.from_numpy(m[:, 0]).cuda()
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = answer_logits(ids_t, b_t, pos)
    loss = torch.nn.functional.cross_entropy(lg.float(), ids_t[b_t, pos])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    run_loss.append(loss.item())
    if (step + 1) % 50 == 0:
        rec = {"step": step + 1, "loss": float(np.mean(run_loss)), "lr": lr_at(step), "elapsed_s": time.time() - t0}
        run_loss = []
        if (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            rec["eval_bf16_32seq"], _ = evaluate(32, "bf16")
            print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

acc, recs = evaluate(128, "fp32")
final = {"final": True, "variant": args.variant, "lr": args.lr, "seed": args.seed, "steps": args.steps,
         "bs": args.bs, "eval_fp32_128seq": acc, "n_queries": int(len(recs)), "elapsed_s": time.time() - t0}
print(json.dumps(final), flush=True)
log.write(json.dumps(final) + "\n")
log.close()
np.savez_compressed(f"../raw/stage2/{name}_eval.npz", recs=recs)
if not args.no_save:
    torch.save(model.state_dict(), f"../raw/stage2/{name}.pt")
