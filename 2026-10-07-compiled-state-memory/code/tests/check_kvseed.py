"""Correctness checks for attention-layer seeds (LABNOTES 2026-10-10; run inside the statelm image).

  python tests/check_kvseed.py --model <snapshot dir> --out <json>

S1  wrapper no-op     sdpa_kvbias with no bias == stock sdpa, bitwise (A, C, K, M sessions)
S2  zero bias         K with a zero bias on every layer == K (KL at bf16-mask noise)
S3  masked memory     K with the memory masked at every layer ~= A
S4  re-rotation       un-rotate + re-rotate to the original positions (Krt) ~= K: noise floor for K*
S5  RoPE round trip   rope_keys(unrotate_keys(k), 0..T-1) vs the cached keys (bitwise fraction, rel. error)
S6  position shift    A with session positions shifted by 65,536 ~= A
S7  deepcopy          pos_shift and bias survive copy.deepcopy (same logits)
S8  generate          greedy generate() on a biased, shifted cache == teacher-forced argmax
S9  partial bias      memory masked at half the layers differs from both K and A (the bias reaches layers)
S10 seed path         C cache + cache.update(un-rotated keys, values) + zero [Hkv, T] bias + shift ~= CK*
S11 per-head bias     a [Hkv, T] bias with equal rows == the same [T] bias (bitwise recorded)
S12 live consumption  a 64-entry seed with distinct per-KV-head biases: one layer's attention output,
                      recorded inside the model during a one-token session, == the seed formula
                      kvseed.fit_head assumes (KV head h serves query heads h*rep..(h+1)*rep-1); the
                      same formula with the wrong grouping (h = j mod Hkv) must disagree
"""
import argparse
import copy
import json
import sys

import torch

sys.path.insert(0, ".")
from statelm.core import LM  # noqa: E402
from statelm.prompts import compile_text, session_text  # noqa: E402
from statelm.synth import attr, make_memory  # noqa: E402


def cmp(a, b):
    a, b = a.float(), b.float()
    pa, pb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    k = (pa.exp() * (pa - pb)).sum(-1)
    return dict(max_abs=(a - b).abs().max().item(), max_kl=k.max().item(), median_kl=k.median().item(),
                argmax_agree=(a.argmax(-1) == b.argmax(-1)).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(42)
    lm = LM(args.model)
    res = {"model": args.model}
    key, _, q, lead, pool = attr("dog")
    mem = make_memory("dog", pool[0], n_entries=250, target_index=40, seed=42)
    mem_ids = lm.ids(compile_text(lm.tok, mem))
    sess_ids = lm.ids(session_text(lm.tok, q, lead))
    T = mem_ids.shape[1]
    res.update(n_mem=T, n_sess=sess_ids.shape[1])
    neg = torch.finfo(torch.float32).min

    def logits(cache, ids=sess_ids):
        return lm.logits(lm.forward(ids, cache))[0]

    full, st = lm.prefill_full(mem_ids)
    mk = {"A": lambda: lm.cache_from_state(None), "C": lambda: lm.cache_from_state(st),
          "K": lambda: lm.zero_gdn(copy.deepcopy(full)), "M": lambda: copy.deepcopy(full)}
    stock = {n: logits(f()) for n, f in mk.items()}

    lm.enable_kv_bias()
    res["S1_noop_bitwise"] = {n: bool(torch.equal(stock[n], logits(f()))) for n, f in mk.items()}

    def biased(layers, value):
        c = mk["K"]()
        c.statelm_kv_bias = {i: torch.full((T,), value, device=lm.device) for i in layers}
        return c

    res["S2_zero_bias_vs_K"] = cmp(stock["K"], logits(biased(lm.attn_layers, 0.0)))
    res["S3_masked_vs_A"] = cmp(stock["A"], logits(biased(lm.attn_layers, neg)))
    half = lm.attn_layers[::2]
    s9 = logits(biased(half, neg))
    res["S9_half_masked_vs_K"], res["S9_half_masked_vs_A"] = cmp(stock["K"], s9), cmp(stock["A"], s9)

    kpre = {i: lm.unrotate_keys(full.layers[i].keys) for i in lm.attn_layers}
    pos = torch.arange(T)
    krt = {i: lm.rope_keys(kpre[i], pos) for i in lm.attn_layers}
    bit = [(krt[i] == full.layers[i].keys).float().mean().item() for i in lm.attn_layers]
    rel = [((krt[i].float() - full.layers[i].keys.float()).norm() / full.layers[i].keys.float().norm()).item()
           for i in lm.attn_layers]
    res["S5_roundtrip"] = dict(min_bitwise_frac=min(bit), max_rel_err=max(rel))
    c = mk["K"]()
    for i in lm.attn_layers:
        c.layers[i].keys = krt[i]
    res["S4_Krt_vs_K"] = cmp(stock["K"], logits(c))

    c = mk["A"]()
    c.statelm_pos_shift = 65536
    res["S6_shift_vs_A"] = cmp(stock["A"], logits(c))

    c = mk["K"]()
    for i in lm.attn_layers:
        c.layers[i].keys = kpre[i]
    c.statelm_pos_shift = 4096 - T
    c.statelm_kv_bias = {i: torch.full((T,), neg, device=lm.device) for i in half}
    snap = copy.deepcopy(c)
    c2 = copy.deepcopy(c)
    res["S7_deepcopy_bitwise"] = bool(torch.equal(logits(c), logits(c2)))

    gen = lm.generate(copy.deepcopy(snap), sess_ids, max_new_tokens=12)
    res["S8_ran"] = bool(gen)
    if gen:
        tf = logits(copy.deepcopy(snap), torch.cat([sess_ids, torch.tensor([gen], device=lm.device)], 1))
        S = sess_ids.shape[1]
        res["S8_generate_matches_tf"] = tf[S - 1:S - 1 + len(gen)].argmax(-1).tolist() == gen
        res["S8_text"] = lm.tok.decode(gen)

    # Stage 1 seed path: a fresh C cache filled through cache.update with the un-rotated keys and
    # the values, a zero per-KV-head bias and the session at D, against CK* built from the full cache
    D, Hkv = 4096, full.layers[lm.attn_layers[0]].keys.shape[1]

    def ckstar():
        c = copy.deepcopy(full)
        for i in lm.gdn_layers:
            c.layers[i].conv_states.zero_()
        for i in lm.attn_layers:
            c.layers[i].keys = kpre[i]
        c.statelm_pos_shift = D - T
        return c

    def seeded(bias):
        c = lm.cache_from_state(st)
        for i in lm.attn_layers:
            c.update(kpre[i], full.layers[i].values, i)
        c.statelm_pos_shift, c.statelm_kv_bias = D - T, bias
        return c

    ref = logits(ckstar())
    res["S10_seed_path_vs_CKstar"] = cmp(ref, logits(seeded({i: torch.zeros(Hkv, T, device=lm.device)
                                                              for i in lm.attn_layers})))
    r = torch.randn(T, device=lm.device, generator=torch.Generator(device=lm.device).manual_seed(42))
    a1 = logits(seeded({i: r for i in lm.attn_layers}))
    a2 = logits(seeded({i: r.expand(Hkv, T).contiguous() for i in lm.attn_layers}))
    res["S11_per_head_bias"] = dict(bitwise=bool(torch.equal(a1, a2)), **cmp(a1, a2))

    # S12: record one layer's attention call during a one-token session on a seeded cache
    from transformers import AttentionInterface
    import statelm.core as core
    L0, n0 = lm.attn_layers[len(lm.attn_layers) // 2], 64
    g12 = torch.Generator(device=lm.device).manual_seed(42)
    seed = {i: (kpre[i][:, :, :n0], full.layers[i].values[:, :, :n0],
                2.0 * torch.randn(Hkv, n0, device=lm.device, generator=g12)) for i in lm.attn_layers}
    c = lm.cache_from_state(st)
    for i, (k, v, _) in seed.items():
        c.update(k, v, i)
    c.statelm_pos_shift, c.statelm_kv_bias = D - n0, {i: b for i, (_, _, b) in seed.items()}
    rec = {}

    def wrap(module, query, key, value, attention_mask, **kw):
        o = core._sdpa_kvbias(module, query, key, value, attention_mask, **kw)
        if module.layer_idx == L0:
            rec.update(q=query.float(), k=key.float(), v=value.float(), o=o[0].float(), s=module.scaling)
        return o

    AttentionInterface.register("sdpa_kvbias", wrap)
    try:
        lm.forward(sess_ids[:, :1], c)
    finally:
        AttentionInterface.register("sdpa_kvbias", core._sdpa_kvbias)
    Hq = rec["q"].shape[1]
    o_live = rec["o"][0, :, -1] if rec["o"].shape[1] == Hq else rec["o"][0, -1]  # [Hq, d] from [1,Hq,1,d] or [1,1,Hq,d]
    res["S12_shapes"] = dict(q=list(rec["q"].shape), k=list(rec["k"].shape), o=list(rec["o"].shape))
    bias = torch.cat([seed[L0][2], torch.zeros(Hkv, rec["k"].shape[2] - n0, device=lm.device)], 1)

    def formula(group):
        outs = []
        for j in range(Hq):
            h = group(j)
            p = torch.softmax(rec["s"] * rec["q"][0, j, -1] @ rec["k"][0, h].T + bias[h], 0)
            outs.append(p @ rec["v"][0, h])
        return torch.stack(outs)

    rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731
    res["S12_live"] = dict(layer=L0, rel_err=rel(formula(lambda j: j // (Hq // Hkv)), o_live),
                           rel_err_wrong_grouping=rel(formula(lambda j: j % Hkv), o_live))

    ok = (all(res["S1_noop_bitwise"].values()) and res["S2_zero_bias_vs_K"]["max_kl"] < 1e-3
          and res["S10_seed_path_vs_CKstar"]["max_kl"] < 1e-3 and res["S11_per_head_bias"]["max_kl"] < 1e-4
          and res["S3_masked_vs_A"]["max_kl"] < 1e-2 and res["S4_Krt_vs_K"]["max_kl"] < 1e-2
          and res["S6_shift_vs_A"]["max_kl"] < 1e-2 and res["S7_deepcopy_bitwise"]
          and res.get("S8_generate_matches_tf", True)
          and res["S9_half_masked_vs_K"]["max_kl"] > 1e-3 and res["S9_half_masked_vs_A"]["max_kl"] > 1e-3
          and res["S12_live"]["rel_err"] < 1e-2 and res["S12_live"]["rel_err_wrong_grouping"] > 0.1)
    res["PASS"] = bool(ok)
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
