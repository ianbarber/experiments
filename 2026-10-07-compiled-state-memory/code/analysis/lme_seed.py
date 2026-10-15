"""Stage 1 development summary (exploratory, dev; LABNOTES 2026-10-10 night).

uv run --with numpy python code/analysis/lme_seed.py results/phase2/lme_seed_dev1.jsonl

1. Fit quality per compression ratio, reference set and attention layer: held-out log-mass RMSE and
   relative output error for the unfitted seed (init), after the bias fit, and after the value fit
   (median over the 4 KV heads, stores and questions). Runs with probe queries (dev run 2 on) add
   the relative output error and log-mass RMSE on CKstar's own session and anchor queries
   (diagnostic; the `probe` reference set is fitted on them, so its probe columns are in-sample).
   From dev run 4 on: the gold-predicting anchor rows separately ('gold'), and the 90th-percentile
   per-row error.
2. Per question and condition: L and H on present and absent, present - absent, first action, and the
   anchor (log p(gold)) on present and absent, then the anchor's per-token log-probs (present store):
   the format tokens and the content tokens differ (LABNOTES 2026-10-11, dev run 4 setup).
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

rows = [json.loads(line) for line in open(sys.argv[1])]
header, rows = rows[0], rows[1:]
print(f"ratios {header['ratios']}  D {header['D']}  reference tokens {header['q_tokens']}  "
      f"refs {header.get('refs', ['own'])}  stores {header.get('stores', ['p', 'a'])}")


def H(r):
    S = sum(float(np.log1p(-np.exp(st[0]))) if st[0] < -0.69 else st[1] for st in r["path"])
    return float(np.log(sum(np.exp(st[0]) for st in r["path"]))) if S > -1e-12 else float(np.log(-np.expm1(S)) - S)


print("\n== 1. fit quality (held-out; median over heads, stores and questions) ==")
fits = [r for r in rows if r["block"] == "fit"]
heads = defaultdict(list)
for r in fits:
    for h in r["heads"]:
        heads[r["ratio"], h.get("ref", "own")].append(h)
probe = any("probe_sess_fit" in h for hs in heads.values() for h in hs)
for ratio in header["ratios"]:
    for ref in header.get("refs", ["own"]):
        hs_all = heads.get((ratio, ref), [])
        if not hs_all:
            continue
        secs = sum(r["seconds"] for r in fits if r["ratio"] == ratio)
        mean = lambda f: float(np.mean([f(h) for h in hs_all]))  # noqa: E731
        line = (f"  r = {ratio} ref {ref} (N = {hs_all[0]['n']}; fit time {secs:.0f}s, all refs): mean over heads "
                f"held out_rel init {mean(lambda h: h['init']['out_rel']):.3f} fit {mean(lambda h: h['fit']['out_rel']):.3f}")
        if probe:
            for pn in ("sess", "anchor", "gold"):
                if f"probe_{pn}_fit" in hs_all[0]:
                    line += (f"; {pn} out_rel init {mean(lambda h: h[f'probe_{pn}_init']['out_rel']):.3f} "
                             f"fit {mean(lambda h: h[f'probe_{pn}_fit']['out_rel']):.3f}")
                    if "row_p90" in hs_all[0][f"probe_{pn}_fit"]:
                        line += f" (p90 {mean(lambda h: h[f'probe_{pn}_fit']['row_p90']):.3f})"
        print(line)
        by = defaultdict(list)
        for h in hs_all:
            by[h["layer"]].append(h)
        cols = f"    {'layer':>5s}  {'logmass init/bias':>18s}  {'out_rel init/bias/fit':>24s}  {'train':>6s} {'ridge':>6s} empty"
        if probe:
            cols += "   sess out init/fit  lm fit   anchor out init/fit  lm fit"
        print(cols)
        for layer in sorted(by):
            hs = by[layer]
            med = lambda f: float(np.median([f(h) for h in hs]))  # noqa: E731
            line = (f"    {layer:5d}  {med(lambda h: h['init']['logmass_rmse']):8.3f} {med(lambda h: h['bias_only']['logmass_rmse']):8.3f}"
                    f"   {med(lambda h: h['init']['out_rel']):7.3f} {med(lambda h: h['bias_only']['out_rel']):7.3f} "
                    f"{med(lambda h: h['fit']['out_rel']):7.3f}  {med(lambda h: h['fit_train']['out_rel']):6.3f} "
                    f"{med(lambda h: h['ridge']):6.0e} {int(sum(h['empty'] for h in hs)):5d}")
            for pn in ("sess", "anchor", "gold"):
                if f"probe_{pn}_fit" in hs[0]:
                    line += (f"   {med(lambda h: h[f'probe_{pn}_init']['out_rel']):7.3f} {med(lambda h: h[f'probe_{pn}_fit']['out_rel']):7.3f}"
                             f"  {med(lambda h: h[f'probe_{pn}_fit']['logmass_rmse']):6.3f}")
            print(line)

print("\n== 2. per question ==")
dec = {(r["qid"], r["cond"], r["store"]): r for r in rows if r["block"] == "decide"}
anc = {(r["qid"], r["cond"], r["store"]): r for r in rows if r["block"] == "anchor"}
qids = list(dict.fromkeys(r["qid"] for r in rows if r["block"] == "meta"))
pst = "p" if any(k[2] == "p" for k in dec) else "a"  # an absent-only run (dev run 6a) shows the absent store
L = lambda r: r["lp_call"] - r["lp_not_call"]  # noqa: E731
act = lambda r: f"{r['stop']}:{r.get('fn') or ''}:{(r.get('arg') or '')[:24]}"  # noqa: E731
for q in qids:
    meta = next(r for r in rows if r["block"] == "meta" and r["qid"] == q)
    print(f"  {q} {meta['qtype']}  tokens {meta['n_tok']}  gold {meta['anchor_gold']!r}")
    a = dec[q, "A", None]
    print(f"    {'A':8s} L {L(a):+6.2f}  H {H(a):+6.2f}  {act(a)}"
          + (f"  anchor {anc[q, 'A', None]['lp_sum']:+7.2f}" if (q, "A", None) in anc else ""))
    conds = list(dict.fromkeys(c for (qq, c, s) in dec if qq == q and s == pst))
    for c in conds:
        p, ab = dec[q, c, pst], dec.get((q, c, "a")) if pst == "p" else None
        if ab is not None:
            line = (f"    {c:8s} L p {L(p):+6.2f} a {L(ab):+6.2f} (p-a {L(p) - L(ab):+5.2f})  "
                    f"H p-a {H(p) - H(ab):+6.2f}  p {act(p):34s} a {act(ab):34s}")
        else:
            line = f"    {c:9s} L {pst} {L(p):+6.2f}  H {pst} {H(p):+6.2f}  {pst} {act(p):34s}"
        if (q, c, pst) in anc:
            ap = anc[q, c, pst]["lp_sum"]
            if pst == "p" and (q, c, "a") in anc:
                aa = anc[q, c, "a"]["lp_sum"]
                line += f"  anchor p {ap:+7.2f} a {aa:+7.2f} (p-a {ap - aa:+6.2f})"
            else:
                line += f"  anchor {pst} {ap:+7.2f}"
        print(line)

print(f"\n== 3. anchor per token, store {pst} (log p of each gold token) ==")
for q in qids:
    ks = [k for k in anc if k[0] == q and k[2] in (None, pst)]
    if not ks:
        continue
    print(f"  {q}")
    for k in ks:
        print(f"    {k[1]:9s} {anc[k]['lp_sum']:+7.2f}  " + " ".join(f"{x:+6.2f}" for x in anc[k]["lp"]))

ssf = {(r["qid"], r["store"]): r for r in rows if r["block"] == "ssf"}
if ssf:
    print("\n== 4. fact-level self-study items: recovery R = sum(lp_seed - lp_C) / sum(lp_CK* - lp_C) ==")
    print("  over items whose answer the memory's attention raises by >= 1 nat (delta = lp_CK* - lp_C, summed over")
    print("  answer tokens); A = items from the fitted half of the sessions (for ssfA* seeds), B = the held-out half")
    words = lambda s: set(re.findall(r"\w+", s.lower()))  # noqa: E731
    STOP = {"a", "an", "the", "of", "my", "to", "in"}
    for (q, s), f in ssf.items():
        its = f["items"]
        d = np.array([sum(it["lp_star"]) - sum(it["lp_c"]) for it in its])
        half = set(f["half_a"])
        ina = np.array([it["session"] in half for it in its])
        mem = d >= 1
        meta = next(r for r in rows if r["block"] == "meta" and r["qid"] == q)
        qw = words(meta["question"])
        sim = sorted(((len(words(it["question"]) & qw) / len(words(it["question"]) | qw), it["question"], it["answer"])
                      for it in its), reverse=True)[:3]
        print(f"  {q} {s}: {len(its)} items ({f['n_rows']} rows, {f['seconds']:.0f}s); delta >= 1: {mem.sum()}, "
              f">= 5: {(d >= 5).sum()}; fitted half A {sum(mem & ina)} / held B {sum(mem & ~ina)} memory items")
        print("    generated questions closest to the test question (word Jaccard): "
              + "; ".join(f"{j:.2f} {qq!r} -> {a!r}" for j, qq, a in sim))
        star = np.array([sum(it["lp_star"]) for it in its])
        c = np.array([sum(it["lp_c"]) for it in its])
        # G2 (dev run 6): items whose answer contains the anchor gold, in the ssf template on CK*
        gw = words(meta["anchor_gold"] or "") - STOP
        for k, it in enumerate(its):
            if gw and gw <= words(it["answer"]):
                dt = np.array(it["lp_star"]) - np.array(it["lp_c"])
                print(f"    gold item {it['question']!r} -> {it['answer']!r} (session {it['session']}, "
                      f"{'A' if ina[k] else 'B'}): CK* first token {it['lp_star'][0]:+.2f}, delta {d[k]:+.2f} "
                      f"(per token {' '.join(f'{x:+.2f}' for x in dt)})")
                # realised per-row weight of each answer row: clip(delta) x scale (ans rows; uniform sets: scale)
                for ref, sh in (f["shares"].items() if f["rows"] == "ans" else ()):
                    rows_a = [j for j, x in enumerate(its) if ("A" not in ref.split("+")[1] or ina[j])]
                    deltas = np.concatenate([np.array(its[j]["lp_star"]) - np.array(its[j]["lp_c"]) for j in rows_a])
                    unif = not re.search(r"ssfA?d%", ref)
                    scale = sh["wsum"] / (len(deltas) if unif else np.clip(deltas, 0, 10).sum())
                    if k in rows_a:
                        wk = (np.ones_like(dt) if unif else np.clip(dt, 0, 10)) * scale
                        print(f"      {ref:16s} per-row weight {' '.join(f'{x:6.1f}' for x in wk)}  "
                              f"(set: {sh['rows']} query tokens, weight sum {sh['wsum']:.0f})")
        res = {r["cond"]: np.array([sum(x) for x in r["lp"]])
               for r in rows if r["block"] == "items" and r["qid"] == q and r["store"] == s}
        res0 = {r["cond"]: np.array([x[0] for x in r["lp"]])  # first answer token only
                for r in rows if r["block"] == "items" and r["qid"] == q and r["store"] == s}
        star0, c0 = np.array([it["lp_star"][0] for it in its]), np.array([it["lp_c"][0] for it in its])
        sess = np.array([it["session"] for it in its])
        for cond, lp in res.items():
            R = lambda m: (lp[m] - c[m]).sum() / max((star[m] - c[m]).sum(), 1e-9)  # noqa: E731
            fr = lambda m: float(np.mean(((lp - c) / np.maximum(d, 1e-9))[m] >= 0.5))  # noqa: E731
            print(f"    {cond:18s} R all {R(mem):+.3f}  A {R(mem & ina):+.3f}  B {R(mem & ~ina):+.3f}  "
                  f"median item lp gap to CK* {np.median((star - lp)[mem]):+.2f}  "
                  f"items recovering >= 0.5: all {fr(mem):.2f} A {fr(mem & ina):.2f} B {fr(mem & ~ina):.2f}")

        def diff(x, y, m, label, first=False):
            """Paired difference of R on the same items; 95% session-cluster bootstrap (2,000 resamples,
            seed 42, fresh per call) and the leave-one-session-out range (LABNOTES dev run 6 amendment)."""
            rs, st_, cc = (res0, star0, c0) if first else (res, star, c)
            m = m & (star0 - c0 >= 1) if first else m  # first token: items whose first token the memory raises
            if x not in rs or y not in rs:
                return
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                print(f"    {label:40s} n/a (no items)")
                return
            us, inv = np.unique(sess[idx], return_inverse=True)
            N = np.bincount(inv, rs[x][idx] - rs[y][idx], len(us))
            D = np.bincount(inv, (st_ - cc)[idx], len(us))
            bs = np.random.default_rng(42).integers(0, len(us), (2000, len(us)))
            lo, hi = np.percentile(N[bs].sum(1) / np.maximum(D[bs].sum(1), 1e-9), [2.5, 97.5])
            loo = (N.sum() - N) / np.maximum(D.sum() - D, 1e-9)
            print(f"    {label:40s} {N.sum() / max(D.sum(), 1e-9):+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
                  f"LOSO [{loo.min():+.3f}, {loo.max():+.3f}]  ({len(idx)} items, {len(us)} sessions)")
        sst = next((k for k in res if k.endswith("sst")), None)
        shs = sorted({m[1] for k in res if (m := re.search(r"%(\d+(?:\.\d+)?)$", k))}, key=float)
        if sst and shs:
            dA = np.concatenate([np.array(its[j]["lp_star"]) - np.array(its[j]["lp_c"]) for j in np.flatnonzero(ina)])
            dall = np.concatenate([np.array(it["lp_star"]) - np.array(it["lp_c"]) for it in its])
            print(f"    dose factor (per-row weight of an A row, ssfAd over ssfd at one share): "
                  f"{np.clip(dall, 0, 10).sum() / max(np.clip(dA, 0, 10).sum(), 1e-9):.2f}")
            print("    paired differences (95% session-cluster bootstrap CI; leave-one-session-out range):")
        for sh in shs:
            pre = sst[:-3]
            diff(f"{pre}sst+ssfd%{sh}", sst, mem, f"CAPACITY dR_all ssfd%{sh} - sst")
            diff(f"{pre}sst+ssfAd%{sh}", sst, mem & ina, f"dose point dR_A ssfAd%{sh} - sst")
            diff(f"{pre}sst+ssfd%{sh}", f"{pre}sst+ssfAd%{sh}", mem & ina, f"crowding R_A ssfd%{sh} - ssfAd%{sh}")
            diff(f"{pre}sst+ssfAd%{sh}", sst, mem & ~ina, f"TRANSFER DiD R_B ssfAd%{sh} - sst")
            diff(f"{pre}sst+ssfAd%{sh}", sst, mem & ~ina, "  first answer token only (its delta >= 1)", first=True)
            diff(f"{pre}sst+ssfAu%{sh}", sst, mem & ~ina, f"transfer R_B ssfAu%{sh} - sst")
            diff(f"{pre}sst+ssfAd%{sh}", f"{pre}sst+ssfAu%{sh}", mem & ina, f"WEIGHTS R_A ssfAd%{sh} - ssfAu%{sh}")
            for x in (f"{pre}sst+ssfAd%{sh}", f"{pre}sst+ssfAu%{sh}"):
                if x in res and (~mem & ~ina).any():  # placebo: held-out items the memory doesn't raise (delta < 1)
                    pl = ~mem & ~ina
                    print(f"    placebo {x[len(pre):]:20s} mean item lp shift vs sst on B items with delta < 1: "
                          f"{np.mean(res[x][pl] - res[sst][pl]):+.3f} ({pl.sum()} items)")
        for a, b in zip(shs, shs[1:]):
            diff(f"{sst[:-3]}sst+ssfd%{b}", f"{sst[:-3]}sst+ssfd%{a}", mem, f"dose dR_all ssfd%{b} - ssfd%{a}")

if fits:
    print("\n== 5. item-held sst error per question (mean over heads; p90), realised share, change against sst ==")
    for r in fits:
        refs_r = list(dict.fromkeys(h["ref"] for h in r["heads"]))
        base = None
        for ref in refs_r:
            hs = [h for h in r["heads"] if h["ref"] == ref]
            e, p9 = float(np.mean([h["fit"]["out_rel"] for h in hs])), float(np.mean([h["fit"]["row_p90"] for h in hs]))
            h0 = hs[0]
            ew = h0.get("extra_wsum", h0.get("extra", 0))
            share = ew / (ew + h0["J_fit"] - h0.get("extra", 0)) if ew else 0.0
            if ref == "sst":
                base = (e, p9)
            dl = f"  {e - base[0]:+.3f} ({p9 - base[1]:+.3f})" if base and ref != "sst" else ""
            print(f"  {r['qid']} {r['store']} r{r['ratio']} {ref:16s} {e:.3f} ({p9:.3f})  share {share:6.1%}{dl}")

if any(c.startswith("S") and c.endswith("sst") for (_, c, _) in anc):
    print(f"\n== 6. anchor per token against this run's sst, store {pst}: t = (lp - lp_sst) / (lp_CK* - lp_C) ==")
    print("  (dev run 6 amendment: under ssfA* seeds 0862's 'L' is one held-out lookup, read beside the transfer DiD)")
    for q in qids:
        sst_c = next((c for (qq, c, s) in anc if qq == q and s == pst and c.startswith("S") and c.endswith("sst")), None)
        if sst_c is None or (q, "C", pst) not in anc:
            continue
        lp_ = lambda c: np.array(anc[q, c, pst]["lp"])  # noqa: E731
        gap = lp_("CKstar") - lp_("C")
        print(f"  {q}  CK* - C per token: " + " ".join(f"{x:+6.2f}" for x in gap))
        for c in (c for (qq, c, s) in anc if qq == q and s == pst and c.startswith("S") and c != sst_c):
            t = (lp_(c) - lp_(sst_c)) / np.where(np.abs(gap) > 1e-6, gap, np.nan)
            line = f"    {c:18s} t " + " ".join(f"{x:+6.2f}" for x in t)
            if pst == "a":  # absent store: rise of each token over max(sst, CK*) on the same store
                line += "   over max(sst, CK*) " + " ".join(
                    f"{x:+6.2f}" for x in lp_(c) - np.maximum(lp_(sst_c), lp_("CKstar")))
            print(line)
    if pst == "a":
        print("  plumbing (absent store): ssf file, gold-free items, sst cache, entry count")
        for q in qids:
            meta = next(r for r in rows if r["block"] == "meta" and r["qid"] == q)
            f = ssf.get((q, "a"))
            ss = next((r for r in rows if r["block"] == "selfstudy" and r["qid"] == q and r["store"] == "a"), None)
            ft = next((r for r in rows if r["block"] == "fit" and r["qid"] == q and r["store"] == "a"), None)
            gw = set(re.findall(r"\w+", (meta["anchor_gold"] or "").lower())) - {"a", "an", "the", "of", "my", "to", "in"}
            hits = [it["answer"] for it in (f["items"] if f else []) if gw and gw <= set(re.findall(r"\w+", it["answer"].lower()))]
            print(f"    {q}: ssf file {os.path.basename(f['file']) if f else None}; items with the gold {hits}; "
                  f"sst cache {os.path.basename(ss['cache']) if ss and ss['cache'] else None}; "
                  f"n {ft['n'] if ft else None} vs n_tok['a'] // r = {meta['n_tok']['a'] // ft['ratio'] if ft else None}")
