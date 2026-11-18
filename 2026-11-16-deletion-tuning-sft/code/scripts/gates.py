"""Implementation gates on the real model. Usage: gates.py {dt,lora}"""
import json, sys, time
import torch
from deltune import data, methods
from deltune.common import load_model, load_tokenizer
from deltune.train import sft_loss, val_loss

which = sys.argv[1]
tok = load_tokenizer()
core = load_model(dtype=torch.float32)
n_model = sum(p.numel() for p in core.parameters())
train_games, val_games = data.load_games()
train_ex = [data.encode(tok, g) for g in train_games]
val_ex = [data.encode(tok, g) for g in val_games]
pad = tok.pad_token_id
for p in core.parameters():
    p.requires_grad_(False)
core.eval()
out = {"n_model_params": n_model}

@torch.no_grad()
def logits_on(ex):
    ids, mask, _ = data.collate(ex, pad)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return core(input_ids=ids, attention_mask=mask).logits.float()

ref = logits_on(val_ex[:4])
out["val_loss_base"] = val_loss(core, val_ex, pad)

if which == "dt":
    t = time.time()
    norms, n_tok = methods.wanda_input_norms(core, (data.collate(train_ex[i : i + 16], pad) for i in range(0, len(train_ex), 16)))
    out["wanda_seconds"] = round(time.time() - t, 1); out["wanda_tokens"] = n_tok
    masked = methods.install_masks(core)
    for m in masked: m.set_mode("hard")
    out["gate_identity_all_kept"] = bool(torch.equal(logits_on(val_ex[:4]), ref))
    methods.init_logits_from_wanda(masked, norms, 1 / 2048, seed=42)
    out["mask_init"] = methods.mask_stats(masked, n_model)
    out["mask_init_zstd_bytes"] = methods.mask_bytes_zstd(masked)
    out["val_loss_init_hard"] = val_loss(core, val_ex, pad)
    for a in (100.0, 1000.0):
        for m in masked: m.set_mode("soft"); m.alpha = a
        out[f"val_loss_init_soft_alpha{int(a)}"] = val_loss(core, val_ex, pad)
    # noise-free Wanda prune at the same eta, for reference
    methods.init_logits_from_wanda(masked, norms, 1 / 2048, seed=42, noise=False)
    for m in masked: m.set_mode("hard")
    out["val_loss_wanda_only_noise_free"] = val_loss(core, val_ex, pad)
    out["mask_wanda_noise_free"] = methods.mask_stats(masked, n_model)
    methods.init_logits_from_wanda(masked, norms, 1 / 2048, seed=42)
    for m in masked: m.set_mode("soft"); m.alpha = 100.0
    params = [m.logits for m in masked]
else:
    peft_model = methods.install_lora(core, 4)
    params = [p for n, p in peft_model.named_parameters() if p.requires_grad]
    out["lora_modules"] = sorted({n.split(".lora_A")[0].split(".")[-1] for n, _ in peft_model.named_parameters() if "lora_A" in n})
    out["lora_module_count"] = sum("lora_A" in n for n, _ in peft_model.named_parameters())
    out["lora_bytes_r4"] = methods.lora_bytes(peft_model)
    out["gate_identity_at_init"] = bool(torch.allclose(logits_on(val_ex[:4]), ref, atol=1e-3))

out["n_trainable"] = sum(p.numel() for p in params)
core.gradient_checkpointing_enable()
core.train()
opt = torch.optim.AdamW(params, lr=5e-4 if which == "dt" else 1e-4, weight_decay=0.0, fused=True)
stream = data.batches(train_ex, 4, 42)
torch.cuda.reset_peak_memory_stats()
losses, t = [], time.time()
for step in range(12):
    ids, mask, labels = data.collate(next(stream), pad)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        total, n = sft_loss(core, ids, mask, labels)
        loss = total / n
    loss.backward()
    if which == "dt":
        methods.regulariser_backward(masked, 1e-5)
        if step == 0:
            g = torch.cat([m.logits.grad.flatten()[::997] for m in masked])
            out["grad_finite"] = bool(torch.isfinite(g).all()); out["grad_abs_median"] = float(g.abs().median())
            out["base_weights_have_grad"] = any(m.weight.grad is not None for m in masked)
    opt.step(); opt.zero_grad(set_to_none=True)
    losses.append(round(float(loss), 4))
out["losses_12_steps"] = losses
out["sec_per_step"] = round((time.time() - t) / 12, 3)
out["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
if which == "lora":
    core.eval()
    @torch.no_grad()
    def val_loss_fp32(model, n=20):
        total, count = 0.0, 0
        for i in range(0, n, 5):
            ids, mask, labels = data.collate(val_ex[i : i + 5], pad)
            loss, k = sft_loss(model, ids, mask, labels)
            total, count = total + float(loss), count + k
        return total / count

    a, a32 = val_loss(core, val_ex, pad), val_loss_fp32(core)
    merged = peft_model.merge_and_unload()
    b, b32 = val_loss(merged, val_ex, pad), val_loss_fp32(merged)
    out["gate_merge"] = {
        "bf16_autocast": {"unmerged": a, "merged": b, "abs_diff": abs(a - b)},
        "fp32": {"unmerged": a32, "merged": b32, "abs_diff": abs(a32 - b32)},
    }
print(json.dumps(out, indent=1))
json.dump(out, open(f"../results/gates_{which}.json", "w"), indent=1)
