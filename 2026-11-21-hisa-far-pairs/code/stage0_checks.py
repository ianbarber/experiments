"""Stage 0 sanity checks. Run before anything else; every check prints PASS/FAIL and the script exits non-zero on
a failure.  python stage0_checks.py"""

import json
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import keeptable as kt
from hisa import rope
from hisa.reference import hisa_attention_reference

MODEL = "HuggingFaceTB/SmolLM-135M"
torch.manual_seed(42)
dev = "cuda"
out, ok_all = {}, True


def report(name, ok, **info):
    global ok_all
    ok_all &= bool(ok)
    out[name] = {"pass": bool(ok), **info}
    print(("PASS " if ok else "FAIL ") + name, info, flush=True)


model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
P = cfg.hidden_size // cfg.num_attention_heads // 2
theta = model.model.rotary_emb.inv_freq.detach().cpu()

# 3. frequencies
expect = 10000.0 ** (-torch.arange(P, dtype=torch.float64) / P)
report("frequencies", torch.allclose(theta.double(), expect, rtol=1e-6),
       max_rel_err=float(((theta.double() - expect) / expect).abs().max()), P=P)

# 1. pair layout: RoPE preserves the norm of each true pair, and does not for a wrong pairing
T = 2048
x = torch.randn(1, 3, T, 2 * P, device=dev)
pos = torch.arange(T, device=dev)[None]
cos, sin = model.model.rotary_emb(x, pos)
xr, _ = apply_rotary_pos_emb(x, x, cos, sin)
n0 = (x[..., :P] ** 2 + x[..., P:] ** 2)
n1 = (xr[..., :P] ** 2 + xr[..., P:] ** 2)
err_split = float((n0 - n1).abs().max())
xi, xri = x.view(1, 3, T, P, 2), xr.view(1, 3, T, P, 2)          # interleaved pairing (2l, 2l+1): should fail
err_inter = float(((xi ** 2).sum(-1) - (xri ** 2).sum(-1)).abs().max())
report("pair_layout_split_half", err_split < 1e-4 and err_inter > 0.1, err_split_half=err_split,
       err_interleaved=err_inter)
# and the rotation angle of pair l at position p is p * theta_l
ang = torch.atan2(sin[0, :, :P], cos[0, :, :P])                    # [T, P]
want = torch.remainder(pos[0][:, None].double() * theta.to(dev).double()[None] + math.pi, 2 * math.pi) - math.pi
dang = torch.remainder(ang.double() - want + math.pi, 2 * math.pi) - math.pi
report("pair_angle", float(dang.abs().max()) < 1e-2, max_abs_err_rad=float(dang.abs().max()))

# 4. tables and attention agree with the paper repo's reference
q = torch.randn(2, 9, 512, 64, device=dev)
k = torch.randn(2, 3, 512, 64, device=dev)
v = torch.randn(2, 3, 512, 64, device=dev)
res = {}
for deg in (45.0, 90.0, 360.0):
    M = kt.make_table(f"hisa-{deg:g}", theta, 512)
    ref_mask = rope.pair_masks(theta, deg, 512)                    # [P, T, T]
    d = (torch.arange(512)[:, None] - torch.arange(512)[None, :])
    mine = M[:, d.clamp(min=0)] & (d >= 0)
    same_mask = bool((mine == ref_mask).all())
    core = kt.TableAttention(M, block=64, device=dev)
    o_ref = hisa_attention_reference(q, k, v, theta.to(dev), deg)
    o_blk = core(q, k, v)
    kk = k.repeat_interleave(3, 1)
    o_slow = torch.softmax(kt.masked_scores_slow(q, kk, M), -1) @ v.repeat_interleave(3, 1)
    res[deg] = dict(same_mask=same_mask, blocked_vs_ref=float((o_blk - o_ref).abs().max()),
                    slow_vs_ref=float((o_slow - o_ref).abs().max()))
report("reference_agreement", all(r["same_mask"] and r["blocked_vs_ref"] < 1e-4 and r["slow_vs_ref"] < 1e-4
                                  for r in res.values()), **{f"{k_:g}": r for k_, r in res.items()})

# 4b. blocked == slow on the non-window tables
res = {}
kk = k.repeat_interleave(3, 1)
for name in ("band-90-180", "band-720+", "ctrl-slow", "ctrl-perm-s0", "ctrl-iid-s0"):
    M = kt.make_table(name, theta, 512)
    core = kt.TableAttention(M, block=64, device=dev)
    s_blk, s_slow = core.scores(q, kk), kt.masked_scores_slow(q, kk, M)
    fin = torch.isfinite(s_slow)
    res[name] = dict(same_support=bool((torch.isfinite(s_blk) == fin).all()),
                     max_abs=float((s_blk[fin] - s_slow[fin]).abs().max()))
report("blocked_vs_slow_tables", all(r["same_support"] and r["max_abs"] < 1e-4 for r in res.values()), **res)

# control tables drop the same number of pairs per distance as hisa-90
n90 = kt.make_table("hisa-90", theta, T).sum(0)
same = {n: bool((kt.make_table(n, theta, T).sum(0) == n90).all())
        for n in ("ctrl-slow", "ctrl-perm-s0", "ctrl-perm-s1", "ctrl-perm-s2", "ctrl-iid-s0")}
report("control_counts", all(same.values()), kept_at_d=[int(n90[d]) for d in (0, 1, 8, 64, 512, 2047)], **same)

# 2. identity: patched model with an all-true table (forced through the blocked path) matches stock logits
ids = torch.randint(0, cfg.vocab_size, (2, T), device=dev)
text = ("The quick brown fox jumps over the lazy dog. " * 400)
ids[0] = torch.tensor(tok(text).input_ids[:T], device=dev)
with torch.no_grad():
    stock = model(ids).logits
    cores = kt.patch_model(model)
    kt.set_condition(cores, kt.make_table("dense", theta, T))
    for c in cores:
        c.is_dense = False                                         # force the blocked path
    patched = model(ids).logits
    kt.set_condition(cores, kt.make_table("dense", theta, T))    # SDPA path through the patched module
    patched_sdpa = model(ids).logits
    kt.set_condition(cores, kt.make_table("hisa-90", theta, T))
    h90 = model(ids).logits
report("identity_all_true", float((stock - patched).abs().max()) < 2e-3,
       max_abs_logit_diff_blocked=float((stock - patched).abs().max()),
       max_abs_logit_diff_sdpa=float((stock - patched_sdpa).abs().max()),
       hisa90_changes_logits=float((stock - h90).abs().max()))

# 4c. full model: blocked hisa-90 == the paper repo's patch_llama(impl="reference") on a short window
from hisa.integrations.hf import patch_llama
m2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
patch_llama(m2, max_angle_deg=90.0, impl="reference")
with torch.no_grad():
    kt.set_condition(cores, kt.make_table("hisa-90", theta, 512))
    a = model(ids[:, :512]).logits
    b = m2(ids[:, :512]).logits
report("model_vs_repo_reference_hisa90", float((a - b).abs().max()) < 2e-3, max_abs_logit_diff=float((a - b).abs().max()))

json.dump(out, open("../results/stage0_checks.json", "w"), indent=1)
sys.exit(0 if ok_all else 1)
