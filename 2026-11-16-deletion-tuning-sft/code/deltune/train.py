"""SFT on the Wordle games with one of three parameterisations.

  python -m deltune.train --method {fft,lora,dt} --lr 1e-4 --steps 1600 --seed 42 --name NAME \
      --save-steps 200,400,...

Everything except the parameterisation is shared: data order, batch size, AdamW, fp32 master
weights with bf16 autocast, constant learning rate, loss on assistant tokens only. Training only
saves deployable checkpoints; scoring them is deltune.evaluate's job.
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F
import zstandard
from safetensors.torch import save_file

from . import data, methods
from .common import RUNS_DIR, dump_json, load_model, load_tokenizer, seed_everything


def sft_loss(core, ids, mask, labels):
    hidden = core.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
    target = labels[:, 1:]
    sel = target != -100
    logits = core.lm_head(hidden[sel])
    return F.cross_entropy(logits.float(), target[sel], reduction="sum"), int(sel.sum())


@torch.no_grad()
def val_loss(core, val_examples, pad_id, batch_size=10):
    total, count = 0.0, 0
    for i in range(0, len(val_examples), batch_size):
        ids, mask, labels = data.collate(val_examples[i : i + batch_size], pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, n = sft_loss(core, ids, mask, labels)
        total, count = total + float(loss), count + n
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fft", "lora", "dt"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=1600, help="length of the schedule (the alpha ramp runs over this)")
    ap.add_argument("--stop-at", type=int, default=0, help="stop early at this step of the same schedule")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-steps", default="", help="steps at which to save a deployable checkpoint")
    ap.add_argument("--no-checkpointing", action="store_true")
    # LoRA
    ap.add_argument("--rank", type=int, default=4)
    # Deletion Tuning
    ap.add_argument("--eta", type=float, default=1 / 2048)
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--alpha", default="100,1000")
    ap.add_argument("--calib", type=int, default=2048, help="training conversations used for Wanda scores")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    last = args.stop_at or args.steps
    save_steps = sorted(s for s in ({int(s) for s in args.save_steps.split(",") if s} | {last}) if s <= last)

    seed_everything(args.seed)
    tok = load_tokenizer()
    core = load_model(dtype=torch.float32)
    n_model = sum(p.numel() for p in core.parameters())
    train_games, val_games = data.load_games()
    train_ex = [data.encode(tok, g) for g in train_games]
    val_ex = [data.encode(tok, g) for g in val_games]
    pad_id = tok.pad_token_id

    peft_model, masked = None, None
    info = {"n_model_params": n_model}
    if args.method == "fft":
        params = [p for p in core.parameters()]
        for p in params:
            p.requires_grad_(True)
        info["adapter_bytes"] = 4 * n_model
    else:
        for p in core.parameters():
            p.requires_grad_(False)
        if args.method == "lora":
            peft_model = methods.install_lora(core, args.rank)
            params = [p for n, p in peft_model.named_parameters() if p.requires_grad]
            info["adapter_bytes"] = methods.lora_bytes(peft_model)
        else:
            calib = train_ex[: args.calib]
            norms, n_tok = methods.wanda_input_norms(
                core, (data.collate(calib[i : i + 16], pad_id) for i in range(0, len(calib), 16))
            )
            masked = methods.install_masks(core)
            methods.init_logits_from_wanda(masked, norms, args.eta, args.seed)
            params = [m.logits for m in masked]
            info["wanda_calibration"] = {"conversations": len(calib), "tokens": n_tok}
            info["mask_init"] = methods.mask_stats(masked, n_model)
    info["n_trainable"] = sum(p.numel() for p in params)
    if not args.no_checkpointing:
        core.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    a0, a1 = (float(x) for x in args.alpha.split(","))
    dump_json({"args": vars(args), "info": info, "train_games": len(train_ex), "val_games": len(val_ex)}, run_dir / "config.json")
    print(json.dumps({"args": vars(args), "info": info}), flush=True)
    log = open(run_dir / "log.jsonl", "w")

    def save(step):
        rec = {"step": step}
        if args.method == "fft":
            state = {n: p.detach().to(torch.bfloat16).cpu() for n, p in core.named_parameters()}
            save_file(state, str(run_dir / f"fft_step{step}.safetensors"))
        elif args.method == "lora":
            peft_model.save_pretrained(run_dir / f"lora_step{step}")
        else:
            blob = zstandard.ZstdCompressor(level=19).compress(methods.packed_mask(masked))
            (run_dir / f"mask_step{step}.zst").write_bytes(blob)
            rec["mask"] = methods.mask_stats(masked, n_model)
            rec["mask"]["adapter_bytes"] = len(blob)
            if step == 0:
                return rec
            # how far the soft mask used in training is from the hard mask that gets deployed
            core.eval()
            rec["val_loss_soft"] = val_loss(core, val_ex, pad_id)
            for m in masked:
                m.set_mode("hard")
            rec["val_loss_hard"] = val_loss(core, val_ex, pad_id)
            for m in masked:
                m.set_mode("soft")
            core.train()
        return rec

    if masked:
        rec = save(0)  # the Wanda-prior mask before any training
        log.write(json.dumps({"save": rec}) + "\n")

    stream = data.batches(train_ex, args.batch_size, args.seed)
    core.train()
    t0, ema = time.time(), None
    for step in range(1, last + 1):
        if masked:
            alpha = a0 + (step - 1) / args.steps * (a1 - a0)
            for m in masked:
                m.alpha = alpha
        ids, mask, labels = data.collate(next(stream), pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total, n = sft_loss(core, ids, mask, labels)
            loss = total / n
        loss.backward()
        reg = methods.regulariser_backward(masked, args.lam) if masked else 0.0
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema = float(loss) if ema is None else 0.9 * ema + 0.1 * float(loss)
        rec = {"step": step, "loss": float(loss), "reg": reg, "tokens": n, "elapsed": round(time.time() - t0, 1)}
        if masked:
            rec["alpha"] = alpha
        log.write(json.dumps(rec) + "\n")
        if step % 50 == 0 or step == 1:
            print(f"step {step} loss {float(loss):.4f} ema {ema:.4f} t {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
        if step in save_steps:
            rec = save(step)
            log.write(json.dumps({"save": rec}) + "\n")
            log.flush()
            print("SAVE", json.dumps(rec), flush=True)
    dump_json({"done": True, "seconds": round(time.time() - t0, 1), "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}, run_dir / "done.json")


if __name__ == "__main__":
    main()
