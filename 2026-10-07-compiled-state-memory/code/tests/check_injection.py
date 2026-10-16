"""Correctness checks for state injection (run inside the statelm image, once per model).

  python tests/check_injection.py --model <snapshot dir> --out <json>

T0  template           compile_text + session_text == the template's own rendering (no thinking preamble)
T1  split == one pass  memory+session in one forward vs prefill_full(memory) then session (short session)
T1L                    same with a ~2,000-token session (median KL + argmax criteria; see note in code)
T2  zero inj == fresh  cache_from_state(None) vs a fresh cache
T3  RoPE offset        C with session positions from 0 vs from n_mem
T4  chunked compile    one chunk vs 512-token chunks (state rel. diff, logits)
T5  determinism        C twice -> bitwise identical
T6  bf16 vs fp32 state split pass with bf16 state buffers vs fp32 (short and long session)
T7  state is used      C differs from A
T8  score_multi        vs score for C, M and K (bf16 one-pass vs two-pass noise; all gate conditions
                       use score_multi, so this only bounds cross-path noise)
T9  K exactness        K cache + compiled GDN/conv copied back == the full split pass; K offset explicit == implicit
T10 generate           greedy generate() tokens == argmax of a teacher-forced pass over the same tokens
"""
import argparse
import copy
import json
import sys

import torch

sys.path.insert(0, ".")
from statelm.core import LM  # noqa: E402
from statelm.prompts import check_template, compile_text, session_text  # noqa: E402
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
    gd = lm.tm.layers[lm.gdn_layers[0]].linear_attn
    assert gd.chunk_gated_delta_rule.__module__.startswith("fla"), "fla kernels not in use"
    res = {"model": args.model}
    check_template(lm.tok)
    res["T0_template"] = True

    key, _, q, lead, pool = attr("dog")
    mem = make_memory("dog", pool[0], n_entries=60, target_index=40, seed=42)
    mem_ids = lm.ids(compile_text(lm.tok, mem))
    sess_ids = lm.ids(session_text(lm.tok, q, lead))
    prose = open("/work/data/raw/text/qwen38_model_card.md").read()
    long_sess = lm.ids(session_text(lm.tok, "Tidy this up:\n" + lm.tok.decode(
        lm.tok.encode(prose, add_special_tokens=False)[:2000]) + "\n" + q, lead))
    n_mem = mem_ids.shape[1]
    res.update(n_mem=n_mem, n_sess=sess_ids.shape[1], n_long_sess=long_sess.shape[1])

    def sess_logits(cache, ids, offset=None):
        return lm.logits(lm.forward(ids, cache, position_offset=offset))[0]

    def one_pass(ids):
        h = lm.forward(torch.cat([mem_ids, ids], 1), lm.new_cache())
        return lm.logits(h[:, -ids.shape[1]:])[0]

    # T1 / T1L
    full_cache, st = lm.prefill_full(mem_ids)
    one, one_L = one_pass(sess_ids), one_pass(long_sess)
    split = sess_logits(copy.deepcopy(full_cache), sess_ids)
    split_L = sess_logits(copy.deepcopy(full_cache), long_sess)
    res["T1_split_vs_one"], res["T1L_split_vs_one_long"] = cmp(one, split), cmp(one_L, split_L)
    # T2
    fresh = sess_logits(lm.new_cache(), sess_ids)
    res["T2_zeroinj_vs_fresh"] = cmp(fresh, sess_logits(lm.cache_from_state(None), sess_ids))
    # T3
    c0 = sess_logits(lm.cache_from_state(st), sess_ids, 0)
    res["T3_pos0_vs_posN"] = cmp(c0, sess_logits(lm.cache_from_state(st), sess_ids, n_mem))
    # T4
    st_chunk = lm.compile(mem_ids, chunk=512)
    res["T4_chunked_state_max_rel_diff"] = max(
        ((st.recurrent[i] - st_chunk.recurrent[i]).norm() / st.recurrent[i].norm()).item() for i in lm.gdn_layers)
    res["T4_chunked_logits"] = cmp(c0, sess_logits(lm.cache_from_state(st_chunk), sess_ids, 0))
    # T5
    res["T5_bitwise_identical"] = bool(torch.equal(c0, sess_logits(lm.cache_from_state(st), sess_ids, 0)))
    # T6 (bf16 state buffers)
    lm.state_dtype = torch.bfloat16
    c_bf = lm.new_cache()
    lm.forward(mem_ids, c_bf)
    res["T6_bf16_vs_fp32_short"] = cmp(split, sess_logits(copy.deepcopy(c_bf), sess_ids))
    res["T6_bf16_vs_fp32_long"] = cmp(split_L, sess_logits(c_bf, long_sess))
    lm.state_dtype = torch.float32
    # T7
    res["T7_C_vs_A"] = cmp(fresh, c0)
    # T8: score_multi vs score, for C, M (continue full cache) and K
    xa, ya = lm.ids(" " + pool[0]), lm.ids(" " + pool[1])
    t8 = {}
    for name, mk in [("C", lambda: lm.cache_from_state(st)), ("M", lambda: copy.deepcopy(full_cache)),
                     ("K", lambda: lm.zero_gdn(copy.deepcopy(full_cache)))]:
        single = [lm.score(mk(), sess_ids, t).sum().item() for t in (xa, ya)]
        multi = [v.sum().item() for v in lm.score_multi(mk(), sess_ids, [xa, ya])]
        t8[name] = max(abs(a - b) for a, b in zip(single, multi))
    res["T8_score_multi_max_abs"] = t8
    # T9: K exactness and offset
    k_cache = lm.zero_gdn(copy.deepcopy(full_cache))
    for i in lm.gdn_layers:
        k_cache.layers[i].recurrent_states.copy_(st.recurrent[i])
        k_cache.layers[i].conv_states.copy_(st.conv[i])
    res["T9_K_restored_vs_split"] = cmp(split, sess_logits(k_cache, sess_ids))
    k_impl = sess_logits(lm.zero_gdn(copy.deepcopy(full_cache)), sess_ids)
    k_expl = sess_logits(lm.zero_gdn(copy.deepcopy(full_cache)), sess_ids, n_mem)
    res["T9_K_offset_bitwise"] = bool(torch.equal(k_impl, k_expl))
    res["T9_K_vs_full"], res["T9_K_vs_A"] = cmp(split, k_impl), cmp(fresh, k_impl)
    # T10: generate == teacher-forced argmax
    gen = lm.generate(lm.cache_from_state(st), sess_ids, max_new_tokens=12)
    if gen:
        tf = sess_logits(lm.cache_from_state(st), torch.cat([sess_ids, torch.tensor([gen], device=lm.device)], 1))
        tf_arg = tf[sess_ids.shape[1] - 1:sess_ids.shape[1] - 1 + len(gen)].argmax(-1).tolist()
        res["T10_generate_matches_teacher_forced"] = tf_arg == gen
        res["T10_text"] = lm.tok.decode(gen)
    res["state_MB_fp32"] = st.nbytes() / 2**20

    ok = (res["T1_split_vs_one"]["argmax_agree"] == 1.0 and res["T1_split_vs_one"]["max_kl"] < 1e-2
          # long sessions are numerically chaotic on every path (one-pass vs token-wise decode is as far
          # apart as split vs one-pass; LABNOTES 2026-10-07), so judge them on median KL and argmax
          and res["T1L_split_vs_one_long"]["median_kl"] < 1e-3
          and res["T1L_split_vs_one_long"]["argmax_agree"] >= 0.95
          and res["T2_zeroinj_vs_fresh"]["max_kl"] < 1e-3 and res["T3_pos0_vs_posN"]["max_kl"] < 1e-2
          and res["T5_bitwise_identical"] and res["T7_C_vs_A"]["max_kl"] > 1e-2
          and max(t8.values()) < 0.15 and res["T9_K_restored_vs_split"]["max_kl"] < 1e-3
          and res["T9_K_offset_bitwise"] and res.get("T10_generate_matches_teacher_forced", True))
    res["PASS"] = bool(ok)
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
