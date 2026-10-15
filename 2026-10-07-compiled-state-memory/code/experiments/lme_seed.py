"""Stage 1 development (exploratory, dev; LABNOTES 2026-10-10): a derived attention seed.

For each store version (present, absent) the memory is compiled once with the full model while
QueryTap records the attention layers' pre-RoPE queries at a seeded sample of memory tokens. From
that compile's KV, every attention layer and KV head gets a seed of N = T / r entries
(statelm/kvseed.py: k-means centroid keys of the un-rotated keys, a fitted log-mass bias, ridge
least-squares values), fitted on reference queries rotated to distance D; the session's question is
never used to fit (except by the diagnostic `probe` set below). The seed is injected next to the
compiled GDN state:
  A         no memory
  C         GDN state (conv reset)
  CK        memory KV + GDN state, conv reset (Stage 0 ceiling)
  CKstar    CK with every key un-rotated to position 0, session at position D (uncompressed seed ceiling)
  S<r><ref> GDN state + seed at compression r fitted on reference set ref ('' for own)
  S0<r>     GDN state + unfitted seed (centroid keys, log cluster size, cluster-mean values)
  M         full continuation
Reference sets (--refs; the k-means is shared, so seeds differ only in bias and values):
  own    the memory's own compile queries at --q_tokens seeded token positions (dev run 1)
  rep    repeat-prefill queries [kim2025kvzip, zweiger2026fast]: after the memory, a user turn asks
         for excerpts word for word and the assistant turn is --rep_chunks seeded chunks of the
         memory, teacher-forced; queries at the chunk tokens (the first --rep_skip of each skipped)
  ss     self-study [eyuboglu2026cartridges, zweiger2026fast]: for --ss_n seeded memory chunks of
         --ss_chunk tokens the model, with no memory, writes one question the user might later ask
         about the chunk; each question is then asked on CKstar twice, in the tool session (queries at
         its last --sess_probe positions) and in the anchor format, whose short answer the model
         generates greedily (queries at every prompt and answer position). Queries are rotated to
         their own positions D + j. The questions and answers are logged.
  sst    self-study tiled over the whole memory: as ss, with consecutive non-overlapping chunks
         covering every memory token (queries kept in bf16 to fit in memory)
  mix    half of own and half of rep (same total)
  sstp   DIAGNOSTIC: sst plus the probe queries, always in the fit set, weight 1: can one seed serve
         the test lookups alongside the whole memory's?
  sstpw  DIAGNOSTIC: as sstp with weight --pw on the probe rows
  sst+<rows>@<w>  DIAGNOSTIC: sst plus a subset of the probe rows (probe = sess + anchor, sess, anchor,
         or gold, the anchor rows that predict the gold tokens) at weight w (sstpw = sst+probe@<pw>)
  sst+<ssf|ssfd|ssfA|ssfAd|ssfAu>%<share>  METHOD (dev): sst plus fact-level self-study rows (--ssf_dir,
         from experiments/lme_ssf_gen.py: memory-free questions about facts the user stated, one prompt
         per user turn, the session's question never seen), scaled so that they hold <share> percent of
         the fit weight (realised per head: extra_wsum / (extra_wsum + J_fit - extra)). Each item's
         question is asked on CKstar in the ssf template (SSF_SUFFIX, a paraphrase of the test's anchor
         template; note sst itself still uses the test's ANCHOR_SUFFIX) and its answer teacher-forced.
         Rows (--ssf_rows): 'ans', the positions that predict the answer tokens; 'q', also every prompt
         position from the user content on (question, suffix, template tail). Raw row weights: ssf,
         ssfA, ssfAu uniform; ssfd, ssfAd clip(delta, 0, 10), delta = log p(answer token) on CKstar
         minus on C (what the memory's attention adds over the GDN state; prompt rows take their item's
         largest delta). A: only items from a seeded half of the memory's sessions (the other half's
         items are held-out facts); ssfAu is ssfAd's uniform control at the same share. Every fitted
         seed also scores every item's answer (block 'items'), against C and CKstar from the
         teacher-forcing pass. ('@<w>' instead of '%' gives a fixed per-row weight, as for probe rows.)
--bias_only also runs every fitted seed with the unfitted cluster-mean values (S<r><ref>-b): what the
fitted bias alone carries.
Splits (kvseed.fit_head) are by group: the q heads of one token, or one self-study item.
  probe  DIAGNOSTIC ONLY, never a method: CKstar's own queries at the last --sess_probe session
         positions and at the anchor's positions. A ceiling for what fitting to the right queries
         can do at a given N.
Every fitted seed also reports its attention-matching error on those probe queries (diagnostic).
Measures as in Stage 0 (first-token call log-odds, greedy path, first call) without the attribution,
and the answer anchor.

./run.sh python -u experiments/lme_seed.py --model $Q38_27B --qids <dev ids> --out /work/results/phase2/lme_seed_dev.jsonl
"""
import argparse
import copy
import glob
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, ".")
from experiments.lme_hint import ANCHOR_SUFFIX, CALL_MAX, DATA, PRE_MAX, TOPK, tool_session  # noqa: E402
from experiments.lme_where import Plan, dev_sample  # noqa: E402
from statelm import lme  # noqa: E402
from statelm.prompts import session_text  # noqa: E402

REPEAT = "Repeat excerpts from the memory above, word for word."
REP_SEP = "\n\n[...]\n\n"
SYS_TAIL = 48  # self-study: the tool session's system-prompt tail is tapped once (first item), not per item
EXTRA = re.compile(r"sst\+(probe|sess|anchor|gold|ssf|ssfd|ssfA|ssfAd|ssfAu)([@%])(\d+(?:\.\d+)?)(/mw|/mq)?$")
SSF_SUFFIX = "\nReply with the answer only."  # ssf template (a paraphrase of the test's ANCHOR_SUFFIX)
USER_HEAD = 3  # '<|im_start|>', 'user', '\n' open every session_text prompt
DELTA_MAX = 10.0
SS_ASK = ("Here is part of a conversation between a user and an assistant:\n\n{chunk}\n\n"
          "Write one short question that the user could ask the assistant later about something the user "
          "said here. Reply with the question only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expandvars("$STATELM_WEIGHTS/Qwen3.8-27B"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--qids", required=True, help="comma-separated dev16 ids")
    ap.add_argument("--ratios", default="16,64,256")
    ap.add_argument("--D", type=int, default=65536)
    ap.add_argument("--q_tokens", type=int, default=6144, help="memory tokens whose queries are references")
    ap.add_argument("--unfitted", action="store_true", help="also run the unfitted seeds S0<r>")
    ap.add_argument("--refs", default="own", help="comma-separated: own, rep, ss, sst, mix, probe, sstp, sstpw, "
                    "sst+<probe|sess|anchor|gold>@<weight>")
    ap.add_argument("--stores", default="p,a")
    ap.add_argument("--rep_chunks", type=int, default=12)
    ap.add_argument("--rep_skip", type=int, default=8)
    ap.add_argument("--sess_probe", type=int, default=64)
    ap.add_argument("--ss_n", type=int, default=48)
    ap.add_argument("--ss_chunk", type=int, default=384)
    ap.add_argument("--ss_answer", type=int, default=24)
    ap.add_argument("--ss_cache", default=None, help="directory to save / reuse tiled self-study queries")
    ap.add_argument("--pw", type=int, default=20)
    ap.add_argument("--seed_dir", default=None, help="directory to save fitted seeds (for diagnostics)")
    ap.add_argument("--bias_only", action="store_true", help="also run each fitted seed with cluster-mean values")
    ap.add_argument("--ssf_dir", default=None, help="directory of lme_ssf_gen.py outputs ({qid}_{store}_all_*.json)")
    ap.add_argument("--ssf_rows", choices=("ans", "q"), default="ans", help="ssf: answer-predicting rows, or from the user content on")
    ap.add_argument("--load_seeds", default=None, help="DIAGNOSTIC: directory of seeds saved by --seed_dir; skip the fit "
                    "and use the saved seed of every ref (the k-means is not rerun)")
    ap.add_argument("--fit_seed", type=int, default=42, help="seed of the fit's own draws (the k-means start and the "
                    "fit / val / held split); the self-study, ssf halves and probe tokens keep seed 42. Added after the last run: "
                    "every run used 42")
    ap.add_argument("--drift", action="store_true", help="DIAGNOSTIC (ssf runs): per ssf answer row and head, each seed's "
                    "attention-matching error at CKstar's queries and at the queries its own seeded run produces")
    args = ap.parse_args()
    ratios = [int(r) for r in args.ratios.split(",")]
    refs = args.refs.split(",")
    stores = args.stores.split(",")
    plain = {"own", "rep", "ss", "sst", "mix", "probe", "sstp", "sstpw", "sst/mw", "sst/mq"}
    assert all(r in plain or EXTRA.match(r) for r in refs) and set(stores) <= {"p", "a"}
    on_sst = any(r in ("sstp", "sstpw", "sst/mw", "sst/mq") or EXTRA.match(r) for r in refs)
    is_ssf = lambda r: bool((m := EXTRA.match(r)) and m[1].startswith("ssf"))  # noqa: E731
    if any(map(is_ssf, refs)):
        assert not {"rep", "mix"} & set(refs), "repeat-prefill mutates the full cache before the ssf pass"
        for q in args.qids.split(","):
            for s in stores:
                n = len(glob.glob(os.path.join(args.ssf_dir, f"{q}_{s}_all_*.json")))
                assert n == 1, f"need exactly one ssf file for {q} {s} in {args.ssf_dir}, found {n}"
    else:
        is_ssf = None
    assert not args.drift or (is_ssf and args.ssf_rows == "ans"), "--drift reads the ssf answer rows"
    assert not (args.load_seeds and args.seed_dir and os.path.samefile(args.load_seeds, args.seed_dir)), \
        "--seed_dir would overwrite the seeds --load_seeds reads"
    assert not ((args.bias_only or args.unfitted) and any(r.endswith(("/mw", "/mq")) for r in refs)), \
        "the unfitted seed is built from the last ref's keys; with mixed key placement it would pair the wrong keys"
    data = lme.load(args.data)
    byid = {x["question_id"]: x for x in data}
    pool = lme.build_pool(data)
    dev = dev_sample(data, pool, byid)
    qids = args.qids.split(",")
    assert set(qids) <= set(dev), "Stage 1 development uses dev16 only"

    import torch
    import torch.nn.functional as F
    from statelm import kvseed
    from statelm.core import LM, CompiledState
    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    lm = LM(args.model)
    lm.enable_kv_bias()
    tok = lm.tok
    TC = tok.convert_tokens_to_ids("<tool_call>")
    END = tok.convert_tokens_to_ids("<|im_end|>")
    STOP = {END, tok.eos_token_id}
    attn0 = lm.tm.layers[lm.attn_layers[0]].self_attn
    s_att, Hq, Hkv = attn0.scaling, lm.cfg.num_attention_heads, lm.cfg.num_key_value_heads
    rep = Hq // Hkv
    angD = lm._rope_angles(torch.tensor([args.D]))[0]
    plans = {q: Plan(byid[q], pool, byid, tok) for q in qids}
    rep_head = tok.encode(session_text(tok, REPEAT), add_special_tokens=False)
    rep_sep = tok.encode(REP_SEP, add_special_tokens=False)

    def lp_split(logits):
        lp = torch.log_softmax(logits.double(), -1)
        return lp, lp[TC].item(), torch.logsumexp(torch.cat([lp[:TC], lp[TC + 1:]]), 0).item()

    def fwd_logits(ids, cache):
        return lm.logits(lm.forward(torch.tensor([ids], device=lm.device), cache)[:, -1])[0]

    def run(cache, sess_str):
        """H1/Stage 0 first-action measures (no attribution)."""
        logits = fwd_logits(tok.encode(sess_str, add_special_tokens=False), cache)
        lp, lp_call, lp_not = lp_split(logits)
        top = lp.topk(TOPK)
        row = dict(lp_call=lp_call, lp_not_call=lp_not,
                   top=[[tok.decode([i]), round(v, 4)] for i, v in zip(top.indices.tolist(), top.values.tolist())])
        pre, path, stop = [], [], "cap"
        for _ in range(PRE_MAX):
            lp, lc, ln = lp_split(logits)
            path.append([lc, ln, lp[END].item()])
            t = int(logits.argmax())
            if t == TC:
                stop = "call"
                break
            if t in STOP:
                stop = "eot"
                break
            pre.append(t)
            logits = fwd_logits([t], cache)
        row.update(stop=stop, called=stop == "call", n_pre=len(pre), pre=tok.decode(pre), path=path)
        if stop == "call":
            logits, call = fwd_logits([TC], cache), []
            for _ in range(CALL_MAX):
                t = int(logits.argmax())
                if t in STOP:
                    break
                call.append(t)
                if "</parameter>" in tok.decode(call) or "</function>" in tok.decode(call):
                    break
                logits = fwd_logits([t], cache)
            txt = tok.decode(call)
            fn = re.search(r"<function=([a-z_]+)>", txt)
            arg = re.search(r"<parameter=\w+>\n?(.*?)(\n?</parameter>|$)", txt, re.S)
            row.update(call=txt, fn=fn.group(1) if fn else None, arg=arg.group(1).strip() if arg else None)
        return row

    def anchor(cache, plan):
        lp = lm.score(cache, lm.ids(plan.anchor_text), lm.ids(plan.gold))
        return dict(lp=lp.tolist(), lp_sum=float(lp.sum()), n_tok=len(lp))

    def snapshot(cache, ids):
        rec = {i: cache.layers[i].recurrent_states.detach().clone() for i in lm.gdn_layers}
        conv = {i: cache.layers[i].conv_states.detach().clone() for i in lm.gdn_layers}
        sha = hashlib.sha256(torch.tensor(ids).numpy().tobytes()).hexdigest()[:16]
        return CompiledState(rec, conv, len(ids), sha, {})

    def versions(plan, keep):
        """Yield (version, full cache, own reference queries {layer: [n, Hq, d] bf16}) from the shared prefix."""
        base = lm.new_cache()
        with kvseed.QueryTap(lm, keep) as tap:
            lm.forward(torch.tensor([plan.ids["p"][:plan.b]], device=lm.device), base)
        q_pre = tap.result()
        todo = [k for k in ("p", "a") if k in stores]
        for n, k in enumerate(todo):
            c = base if n == len(todo) - 1 else copy.deepcopy(base)
            with kvseed.QueryTap(lm, keep, offset=plan.b) as tap:
                lm.forward(torch.tensor([plan.ids[k][plan.b:]], device=lm.device), c)
            q_suf = tap.result()
            yield k, c, {i: torch.cat([q_pre[i], q_suf[i]]) for i in lm.attn_layers}
            del c
        del base

    def repeat_queries(full, ids, g):
        """Repeat-prefill on `full` (mutates it; the caller slices the KV back to the memory). Returns
        ({layer: [n, Hq, d] bf16}, chunk starts)."""
        T, L = len(ids), args.q_tokens // args.rep_chunks + args.rep_skip
        starts = sorted(torch.randint(0, T - L, (args.rep_chunks,), generator=g).tolist())
        seq, keep = list(rep_head), []
        for j, s0 in enumerate(starts):
            if j:
                seq += rep_sep
            keep += range(len(seq) + args.rep_skip, len(seq) + L)
            seq += ids[s0:s0 + L]
        with kvseed.QueryTap(lm, torch.tensor(keep)) as tap:
            lm.forward(torch.tensor([seq], device=lm.device), full)
        return tap.result(), starts

    def probe_queries(make, plan):
        """CKstar's pre-RoPE queries at the last sess_probe session positions and at the anchor's
        positions, each rotated to its own position (D + index), as {name: {layer: [n, Hq, d] fp32}}."""
        out = {}
        sess = tok.encode(plan.plain, add_special_tokens=False)
        items = [("sess", sess, range(len(sess) - args.sess_probe, len(sess)))]
        if plan.gold:
            anc = lm.ids(plan.anchor_text)[0].tolist() + lm.ids(plan.gold)[0].tolist()
            items.append(("anchor", anc, range(len(anc) - 1)))
        for name, ids, keep in items:
            out[name] = tapped(make, ids, keep)[0]
        if plan.gold:  # the anchor rows that predict the gold tokens (a subset of 'anchor'; metrics only)
            ng = lm.ids(plan.gold).shape[1]
            out["gold"] = {i: v[-ng:] for i, v in out["anchor"].items()}
        return out

    def tapped(make, ids, keep, gen=0):
        """Queries at relative positions `keep` while `ids` (then `gen` greedy tokens) run on make(),
        rotated to D + position: ({layer: [n, Hq, d] fp32}, generated ids)."""
        keep = torch.tensor(list(keep))
        with kvseed.QueryTap(lm, keep) as tap:
            if gen:
                out = lm.generate(make(), torch.tensor([ids], device=lm.device), max_new_tokens=gen)
            else:
                lm.forward(torch.tensor([ids], device=lm.device), make())
                out = []
        res = tap.result()
        n = res[lm.attn_layers[0]].shape[0]
        ang = lm._rope_angles(keep[:n] + args.D)[:, None]
        return {i: kvseed.rotate(v.float(), ang) for i, v in res.items()}, out

    def selfstudy_queries(make, ids, plan, g, tile=False):
        """Self-study reference queries ({layer: [n, Hq, d]}, fp32, or bf16 when tiled), the item id of
        each query token ([n]; -1 for the system-prompt tail) and the log of questions/answers. Chunk
        starts: --ss_n seeded draws, or (tile) every multiple of --ss_chunk. In the tool session only
        the user turn on is tapped per item (the system prompt precedes the question, so its queries
        are identical across items); the first item also taps the last SYS_TAIL system positions."""
        T, L = len(ids), args.ss_chunk
        if tile:
            starts = list(range(0, T - L + 1, L))
        else:
            starts = sorted(torch.randint(0, T - L, (args.ss_n,), generator=g).tolist())
        dt = torch.bfloat16 if tile else torch.float32
        system = lme.system_prompt(plan.x["question_date"])
        acc, item, log = {i: [] for i in lm.attn_layers}, [], []
        for k, s0 in enumerate(starts):
            ask = lm.ids(session_text(tok, SS_ASK.format(chunk=tok.decode(ids[s0:s0 + L]))))
            qtxt = tok.decode(lm.generate(lm.cache_from_state(None), ask, max_new_tokens=48)).strip()
            stxt = tool_session(tok, system, qtxt)
            sess = tok.encode(stxt, add_special_tokens=False)
            nsys = len(tok.encode(stxt[:stxt.index("<|im_start|>user")], add_special_tokens=False))
            lo = nsys - SYS_TAIL if k == 0 else nsys
            qa, _ = tapped(make, sess, range(lo, len(sess)))
            anc = tok.encode(session_text(tok, qtxt + ANCHOR_SUFFIX), add_special_tokens=False)
            qb, ans = tapped(make, anc, range(len(anc) + args.ss_answer), gen=args.ss_answer)
            for i in lm.attn_layers:
                acc[i] += [qa[i].to(dt), qb[i].to(dt)]
            na, nb = qa[lm.attn_layers[0]].shape[0], qb[lm.attn_layers[0]].shape[0]
            item += [-1 if (k == 0 and j < SYS_TAIL) else k for j in range(na)] + [k] * nb
            log.append(dict(start=s0, question=qtxt, answer=tok.decode(ans), n_sess=na, n_anchor=nb))
        return {i: torch.cat(v) for i, v in acc.items()}, torch.tensor(item), log

    def ssf_items(q, s):
        fs = sorted(glob.glob(os.path.join(args.ssf_dir, f"{q}_{s}_all_*.json")))
        assert len(fs) == 1, fs
        d = json.load(open(fs[0]))
        assert not d["question_seen"]
        return fs[0], d["items"], len(plans[q].st["names"])

    def answer_lp(h, P, an):
        lp = F.log_softmax(lm.logits(h[:, P - 1:P + len(an) - 1]), -1)[0]
        return lp.gather(-1, torch.tensor(an, device=lm.device)[:, None])[:, 0].float().cpu()

    def fact_queries(make, stc, items):
        """ssf reference queries ({layer: [n, Hq, d] bf16}, rotated to D + position), the item id,
        delta and kind (0 prompt, 1 answer-predicting) of each query token, and per item log p(answer
        tokens) on CKstar (make()) and on C."""
        acc, item, delta, kind, log = {i: [] for i in lm.attn_layers}, [], [], [], []
        for k, it in enumerate(items):
            pr = tok.encode(session_text(tok, it["question"] + SSF_SUFFIX), add_special_tokens=False)
            an = tok.encode(it["answer"], add_special_tokens=False)
            P, ids = len(pr), torch.tensor([pr + an], device=lm.device)
            ctx = P - 1 - USER_HEAD if args.ssf_rows == "q" else 0
            keep = torch.arange(P - 1 - ctx, P + len(an) - 1)
            with kvseed.QueryTap(lm, keep) as tap:
                lp_s = answer_lp(lm.forward(ids, make()), P, an)
            lp_c = answer_lp(lm.forward(ids, lm.cache_from_state(stc)), P, an)
            d = lp_s - lp_c
            ang = lm._rope_angles(keep + args.D)[:, None]
            for i, v in tap.result().items():
                acc[i].append(kvseed.rotate(v.float(), ang).bfloat16())
            item.append(torch.full((len(keep),), k))
            delta.append(torch.cat([d.max().repeat(ctx), d]))
            kind.append(torch.cat([torch.zeros(ctx, dtype=torch.long), torch.ones(len(an), dtype=torch.long)]))
            log.append(dict(lp_star=lp_s.tolist(), lp_c=lp_c.tolist()))
        return ({i: torch.cat(v) for i, v in acc.items()}, torch.cat(item), torch.cat(delta), torch.cat(kind), log)

    def item_scores(make, items, tap=False):
        """Per item log p(answer tokens) on make(); with tap, also the queries at the answer-predicting
        positions (as fact_queries in 'ans' mode: {layer: [rows, Hq, d] bf16}, rotated to D + position)."""
        out, acc = [], {i: [] for i in lm.attn_layers}
        for it in items:
            pr, an = lm.ids(session_text(tok, it["question"] + SSF_SUFFIX)), lm.ids(it["answer"])
            if not tap:
                out.append(lm.score(make(), pr, an).tolist())
                continue
            keep = torch.arange(pr.shape[1] - 1, pr.shape[1] + an.shape[1] - 1)
            with kvseed.QueryTap(lm, keep) as qt:
                out.append(lm.score(make(), pr, an).tolist())
            ang = lm._rope_angles(keep + args.D)[:, None]
            for i, v in qt.result().items():
                acc[i].append(kvseed.rotate(v.float(), ang).bfloat16())
        return out, ({i: torch.cat(v) for i, v in acc.items()} if tap else None)

    def drift_stats(kv, seed, n, qstar, qrun):
        """DIAGNOSTIC (--drift): per (layer, KV head) and ssf answer row (mean over the head's rep q
        heads), the seed's relative output error at CKstar's queries against the memory's output there
        (e_star: what the fit was asked to match), at the queries of its own seeded run against the
        memory's output at those queries (e_run), and its own-run output against the memory's output at
        CKstar's queries (e_tot), plus the query drift and the memory-output drift between the two
        query sets, and the absolute log-mass errors."""
        res = []
        for i in lm.attn_layers:
            kpre = lm.unrotate_keys(kv[i][0])[0].float()
            vals = kv[i][1][0].float()
            C, U, b = seed[i]
            for h in range(Hkv):
                hs = slice(h * rep, (h + 1) * rep)
                qs, qr = (x[:, hs].reshape(-1, x.shape[-1]).float() for x in (qstar[i], qrun[i]))
                R = qs.shape[0] // rep
                ms, ys = kvseed.full_targets(qs, kpre[h], vals[h], s_att)
                mr, yr = kvseed.full_targets(qr, kpre[h], vals[h], s_att)
                Ch, Uh, bh = C[0, h].float(), U[0, h].float(), b[h].float()
                Ss, Sr = s_att * (qs @ Ch.T) + bh, s_att * (qr @ Ch.T) + bh
                os_, or_ = torch.softmax(Ss, 1) @ Uh, torch.softmax(Sr, 1) @ Uh
                rel = lambda a, t: ((a - t).norm(dim=1) / t.norm(dim=1).clamp_min(1e-12)).view(R, rep).mean(1)  # noqa: E731
                r4 = lambda t: [round(v, 4) for v in t.tolist()]  # noqa: E731
                res.append(dict(layer=i, head=h, e_star=r4(rel(os_, ys)), e_run=r4(rel(or_, yr)), e_tot=r4(rel(or_, ys)),
                                q_drift=r4(rel(qr, qs)), y_drift=r4(rel(yr, ys)),
                                lm_star=r4((torch.logsumexp(Ss, 1) - ms).abs().view(R, rep).mean(1)),
                                lm_run=r4((torch.logsumexp(Sr, 1) - mr).abs().view(R, rep).mean(1))))
                del Ss, Sr
            del kpre, vals
        return res

    def fit_seed(kv, refq, probes, n):
        """Per reference set {layer: (keys [1,Hkv,n,d] bf16, values bf16, bias [Hkv,n] fp32)}, the
        unfitted seed, and per-head stats (held-out and probe errors)."""
        fitted = {r: {} for r in refq}
        unfitted, stats = {}, []
        for i in lm.attn_layers:
            kpre = lm.unrotate_keys(kv[i][0])[0].float()
            vals = kv[i][1][0].float()
            parts = {r: {k: [] for k in "CbU"} for r in refq}
            p0 = {k: [] for k in "CbU"}
            for h in range(Hkv):
                hs = slice(h * rep, (h + 1) * rep)
                km = kvseed.kmeans(kpre[h], n, seed=args.fit_seed)
                pt = {pn: (P[i][:, hs].reshape(-1, P[i].shape[-1]),
                           *kvseed.full_targets(P[i][:, hs].reshape(-1, P[i].shape[-1]), kpre[h], vals[h], s_att))
                      for pn, P in probes.items()}
                for r, spec in refq.items():
                    Qd = spec["q"][i]
                    Q = Qd[:, hs].reshape(-1, Qd.shape[-1]).float()
                    tid = spec["ids"] if spec["ids"] is not None else torch.arange(Qd.shape[0])
                    extra, w = None, 1.0
                    if spec["extra"] is not None:
                        E, w = spec["extra"]
                        extra = E[i][:, hs].reshape(-1, E[i].shape[-1]).float()
                        if torch.is_tensor(w):  # per query token -> per row (token-major, rep q heads)
                            w = w.to(lm.device).float().repeat_interleave(rep)
                    keys = spec.get("keys", "kmeans")
                    C, b, U, b0, vbar, st = kvseed.fit_head(kpre[h], vals[h], Q, n, s_att, km=km if keys == "kmeans" else None,
                                                            extra=extra, extra_w=w, keys=keys, seed=args.fit_seed,
                                                            groups=tid.to(lm.device).repeat_interleave(rep))
                    for pn, (Qp, mp, yp) in pt.items():
                        st[f"probe_{pn}_init"] = kvseed.seed_metrics(Qp, C, b0, vbar, mp, yp, s_att)
                        st[f"probe_{pn}_fit"] = kvseed.seed_metrics(Qp, C, b, U, mp, yp, s_att)
                    for k, v in zip("CbU", (C, b, U)):
                        parts[r][k].append(v)
                    stats.append(dict(layer=i, head=h, ref=r, **st))
                for k, v in zip("CbU", (C, b0, vbar)):
                    p0[k].append(v)
            for r in refq:
                st = lambda key: torch.stack(parts[r][key])  # noqa: E731
                fitted[r][i] = (st("C")[None].bfloat16(), st("U")[None].bfloat16(), st("b"))
            unfitted[i] = (torch.stack(p0["C"])[None].bfloat16(), torch.stack(p0["U"])[None].bfloat16(),
                           torch.stack(p0["b"]))
            del kpre, vals
        return fitted, unfitted, stats

    def seeded(stc, seed, n):
        c = lm.cache_from_state(stc)
        for i, (k, v, _) in seed.items():
            c.update(k, v, i)
        c.statelm_pos_shift = args.D - n
        c.statelm_kv_bias = {i: b for i, (_, _, b) in seed.items()}
        return c

    header = dict(block="_header", model=args.model, data=args.data, qids=qids, ratios=ratios, D=args.D,
                  q_tokens=args.q_tokens, unfitted=args.unfitted, refs=refs, stores=stores,
                  rep_chunks=args.rep_chunks, rep_skip=args.rep_skip, sess_probe=args.sess_probe,
                  repeat_prompt=REPEAT, repeat_sep=REP_SEP, ss_n=args.ss_n, ss_chunk=args.ss_chunk,
                  ss_answer=args.ss_answer, ss_ask=SS_ASK, pw=args.pw, bias_only=args.bias_only, pre_max=PRE_MAX,
                  call_max=CALL_MAX, decoding="greedy", fit_seed=args.fit_seed)
    out = open(args.out, "w")
    out.write(json.dumps(header) + "\n")
    t0 = time.time()
    for qi, q in enumerate(qids):
        tq = time.time()
        torch.cuda.reset_peak_memory_stats()
        plan, x = plans[q], plans[q].x
        ip, ia = plan.ids["p"], plan.ids["a"]
        suf = next(k for k in range(min(len(ip), len(ia)) - plan.b) if ip[-1 - k] != ia[-1 - k])
        out.write(json.dumps(dict(block="meta", qid=q, qtype=x["question_type"], question=x["question"],
                                  gold=x["answer"], anchor_gold=plan.gold,
                                  n_tok={k: len(plan.ids[k]) for k in ("p", "a")}, shared_prefix=plan.b,
                                  evidence_p=[plan.b, len(ip) - suf])) + "\n")

        def emit(cond, store, make):
            row = run(make(), plan.plain)
            out.write(json.dumps(dict(block="decide", qid=q, qtype=x["question_type"], cond=cond, store=store,
                                      **row)) + "\n")
            if plan.gold:
                out.write(json.dumps(dict(block="anchor", qid=q, qtype=x["question_type"], cond=cond, store=store,
                                          gold=plan.gold, **anchor(make(), plan))) + "\n")
            out.flush()

        emit("A", None, lambda: lm.cache_from_state(None))
        T_min = min(len(plan.ids["p"]), len(plan.ids["a"]))
        g = torch.Generator().manual_seed(42)
        keep = torch.randperm(T_min, generator=g)[:args.q_tokens].sort().values
        for s, full, own in versions(plan, keep):
            T = len(plan.ids[s])
            stc = snapshot(full, plan.ids[s])
            emit("M", s, lambda: copy.deepcopy(full))
            emit("C", s, lambda: lm.cache_from_state(stc))

            def ck():
                c = copy.deepcopy(full)
                for i in lm.gdn_layers:
                    c.layers[i].conv_states.zero_()
                return c

            emit("CK", s, ck)

            def ckstar():
                c = ck()
                for i in lm.attn_layers:
                    c.layers[i].keys = lm.unrotate_keys(c.layers[i].keys)
                c.statelm_pos_shift = args.D - T
                return c

            emit("CKstar", s, ckstar)
            probes = probe_queries(ckstar, plan)
            refq = {}
            for ref in ("ss", "sst"):
                if ref not in refs and not (ref == "sst" and on_sst):
                    continue
                key = hashlib.sha256(json.dumps([os.path.basename(args.model.rstrip("/")), SS_ASK, ANCHOR_SUFFIX,
                                                 SYS_TAIL, args.ss_answer, args.ss_chunk, args.D, stc.text_sha,
                                                 plan.x["question_date"]]).encode()).hexdigest()[:12]
                cache_f = os.path.join(args.ss_cache, f"{q}_{s}_sst_{key}.pt") if args.ss_cache and ref == "sst" else None
                ts, hit = time.time(), bool(cache_f and os.path.exists(cache_f))
                if hit:
                    saved = torch.load(cache_f)
                    qd, ids_tok, ss_log = {i: v.to(lm.device) for i, v in saved["q"].items()}, saved["ids"], saved["log"]
                    del saved
                else:
                    star = ckstar()
                    g_ss = torch.Generator().manual_seed(42)
                    qd, ids_tok, ss_log = selfstudy_queries(lambda: copy.deepcopy(star), plan.ids[s], plan, g_ss,
                                                            tile=ref == "sst")
                    del star
                    if cache_f:
                        torch.save(dict(q={i: v.cpu() for i, v in qd.items()}, ids=ids_tok, log=ss_log), cache_f)
                refq[ref] = dict(q=qd, ids=ids_tok, extra=None)
                out.write(json.dumps(dict(block="selfstudy", qid=q, store=s, ref=ref, chunk=args.ss_chunk, cache=cache_f,
                                          seconds=round(time.time() - ts, 1), cached=hit,
                                          n_queries=qd[lm.attn_layers[0]].shape[0], items=ss_log)) + "\n")
                out.flush()
            if "own" in refs or "mix" in refs:
                refq["own"] = dict(q={i: kvseed.rotate(v.float(), angD) for i, v in own.items()}, ids=None, extra=None)
            if "rep" in refs or "mix" in refs:
                g_rep = torch.Generator().manual_seed(42)
                rq, starts = repeat_queries(full, plan.ids[s], g_rep)
                refq["rep"] = dict(q={i: kvseed.rotate(v.float(), angD) for i, v in rq.items()}, ids=None, extra=None)
                out.write(json.dumps(dict(block="repeat", qid=q, store=s, starts=starts,
                                          n_queries=rq[lm.attn_layers[0]].shape[0])) + "\n")
                del rq
            if "mix" in refs:
                g_mix = torch.Generator().manual_seed(42)
                h_own, h_rep = (torch.randperm(refq[k]["q"][lm.attn_layers[0]].shape[0], generator=g_mix)[:args.q_tokens // 2]
                                for k in ("own", "rep"))
                refq["mix"] = dict(q={i: torch.cat([refq["own"]["q"][i][h_own], refq["rep"]["q"][i][h_rep]])
                                      for i in lm.attn_layers}, ids=None, extra=None)
            pq = {i: torch.cat([probes[k][i] for k in ("sess", "anchor") if k in probes]) for i in lm.attn_layers}
            refq["probe"] = dict(q=pq, ids=None, extra=None)
            if "sst" in refq:
                refq["sstp"] = dict(refq["sst"], extra=(pq, 1.0))
                refq["sstpw"] = dict(refq["sst"], extra=(pq, float(args.pw)))
                for ref in refs:
                    if (m := EXTRA.match(ref)) and not m[1].startswith("ssf"):
                        assert m[2] == "@", ref
                        refq[ref] = dict(refq["sst"], extra=(pq if m[1] == "probe" else probes[m[1]], float(m[3])))
            items, f_log, drift_q = None, None, None
            if is_ssf and any(map(is_ssf, refs)):
                ts = time.time()
                f_file, items, n_sess = ssf_items(q, s)
                star = ckstar()
                fq, f_item, f_delta, f_kind, f_log = fact_queries(lambda: copy.deepcopy(star), stc, items)
                del star
                g_half = torch.Generator().manual_seed(42)
                half = set(torch.randperm(n_sess, generator=g_half)[:n_sess // 2].tolist())
                in_a = torch.tensor([items[k]["session"] in half for k in f_item.tolist()])
                n_sst = refq["sst"]["q"][lm.attn_layers[0]].shape[0]
                shares = {}
                for ref in filter(is_ssf, refs):
                    m = EXTRA.match(ref)
                    rows = in_a if "A" in m[1] else torch.ones_like(in_a)
                    w = f_delta.clamp(0, DELTA_MAX) if m[1] in ("ssfd", "ssfAd") else torch.ones_like(f_delta)
                    w = w[rows]
                    if m[2] == "%":  # extra rows hold `share` of the fit weight (sst fit set ~ (1 - held - val) of n_sst)
                        sh = float(m[3]) / 100
                        w = w * (sh / (1 - sh) * 0.8 * n_sst / w.sum().clamp_min(1e-9))
                    else:
                        w = w * float(m[3])
                    shares[ref] = dict(rows=int(rows.sum()), wsum=float(w.sum()),
                                       answer_rows_wshare=float(w[f_kind[rows] == 1].sum() / w.sum().clamp_min(1e-9)))
                    refq[ref] = dict(refq["sst"], extra=({i: v[rows.to(v.device)] for i, v in fq.items()}, w))
                ev_in_a = [e in half for e in plans[q].st["ev"]]  # dev diagnostic only; never feeds the fit
                out.write(json.dumps(dict(block="ssf", qid=q, store=s, file=f_file, n_items=len(items),
                                          n_rows=int(f_item.numel()), n_answer_rows=int(f_kind.sum()),
                                          seconds=round(time.time() - ts, 1), half_a=sorted(half), ev_in_a=ev_in_a,
                                          rows=args.ssf_rows, suffix=SSF_SUFFIX, n_sst_tokens=n_sst, shares=shares,
                                          items=[dict(it, **lg) for it, lg in zip(items, f_log)])) + "\n")
                out.flush()
                drift_q = fq if args.drift else None
                del fq, f_item, f_delta, f_kind, in_a
            for r_ in ("sst/mw", "sst/mq"):
                if r_ in refs:
                    refq[r_] = refq["sst"]
            KEYS = {"/mw": "mass", "/mq": "massq"}
            refq = {r: dict(refq[r], keys=KEYS.get(r[-3:], "kmeans")) for r in refs}
            kv = {i: (full.layers[i].keys[:, :, :T], full.layers[i].values[:, :, :T]) for i in lm.attn_layers}
            del own
            for r in ratios:
                n = T // r
                tf = time.time()
                fitted, unfitted, u_keys, loaded = {}, None, None, []
                sp = os.path.join(args.load_seeds, f"{q}_{s}_r{r}.pt") if args.load_seeds else None
                if sp and not os.path.exists(sp):  # no saved seeds at this ratio: every ref is fitted
                    out.write(json.dumps(dict(block="loaded", qid=q, store=s, ratio=r, n=n, path=None, refs=[])) + "\n")
                elif sp:  # refs saved by an earlier run are loaded, the rest are fitted
                    sv = torch.load(sp)
                    assert sv["n"] == n and sv["D"] == args.D, (sp, sv["n"], sv["D"])
                    assert sv.get("fit_seed", 42) == args.fit_seed, (sp, sv.get("fit_seed", 42), args.fit_seed)
                    on_dev = lambda d: {i: tuple(t.to(lm.device) for t in v) for i, v in d.items()}  # noqa: E731
                    loaded = [k for k in refs if k in sv["fitted"]]
                    fitted, unfitted = {k: on_dev(sv["fitted"][k]) for k in loaded}, on_dev(sv["unfitted"])
                    u_keys = sv.get("unfitted_keys")  # untagged files predate the tag
                    del sv
                    out.write(json.dumps(dict(block="loaded", qid=q, store=s, ratio=r, n=n, path=sp, refs=loaded)) + "\n")
                todo = [k for k in refs if k not in loaded]
                if todo:
                    f_new, u_new, stats = fit_seed(kv, {k: refq[k] for k in todo}, probes, n)
                    fitted.update(f_new)
                    if unfitted is None:  # fit_seed's unfitted seed sits on the last ref's keys
                        unfitted, u_keys = u_new, refq[todo[-1]]["keys"]
                    del f_new, u_new
                    out.write(json.dumps(dict(block="fit", qid=q, store=s, ratio=r, n=n, seconds=round(time.time() - tf, 1),
                                              refs=todo, heads=stats)) + "\n")
                if args.seed_dir:
                    cpu = lambda d: {i: tuple(t.cpu() for t in v) for i, v in d.items()}  # noqa: E731
                    torch.save(dict(fitted={k: cpu(v) for k, v in fitted.items()}, unfitted=cpu(unfitted), n=n, D=args.D,
                                    fit_seed=args.fit_seed, unfitted_keys=u_keys),
                               os.path.join(args.seed_dir, f"{q}_{s}_r{r}.pt"))
                assert not (args.bias_only or args.unfitted) or u_keys == "kmeans", \
                    f"the unfitted seed here sits on {u_keys} keys (None: an untagged saved file), not k-means"
                for ref in refs:
                    emit(f"S{r}" + ("" if ref == "own" else ref), s, lambda: seeded(stc, fitted[ref], n))
                    if items is not None:
                        sc, qrun = item_scores(lambda: seeded(stc, fitted[ref], n), items, tap=drift_q is not None)
                        out.write(json.dumps(dict(block="items", qid=q, store=s, cond=f"S{r}{ref}", lp=sc)) + "\n")
                        if qrun is not None:
                            td = time.time()
                            out.write(json.dumps(dict(block="drift", qid=q, store=s, cond=f"S{r}{ref}",
                                                      heads=drift_stats(kv, fitted[ref], n, drift_q, qrun),
                                                      seconds=round(time.time() - td, 1))) + "\n")
                            del qrun
                        out.flush()
                    if args.bias_only:  # fitted keys (= centroids) and bias, unfitted cluster-mean values
                        bo = {i: (unfitted[i][0], unfitted[i][1], fitted[ref][i][2]) for i in lm.attn_layers}
                        emit(f"S{r}{ref}-b", s, lambda: seeded(stc, bo, n))
                if args.unfitted:
                    emit(f"S0{r}", s, lambda: seeded(stc, unfitted, n))
                del fitted, unfitted
            del full, kv, refq, probes, stc
            torch.cuda.empty_cache()
        out.write(json.dumps(dict(block="done", qid=q, t_question=round(time.time() - tq, 1))) + "\n")
        out.flush()
        print(f"{qi + 1}/{len(qids)} {q} {x['question_type']} tokens {len(plan.ids['p'])} question "
              f"{time.time() - tq:.0f}s total {time.time() - t0:.0f}s peak alloc "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
