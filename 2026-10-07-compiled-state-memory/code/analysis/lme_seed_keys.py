"""Stage 1 dev run 7 summary (exploratory, dev; LABNOTES 2026-10-11 night): key placement and drift.

uv run --with numpy python analysis/lme_seed_keys.py ../results/phase2/lme_seed_dev7.jsonl ../results/phase2/lme_seed_dev6.jsonl

The k-means seeds (sst, sst+ssfd%30, sst+ssfAd%30) are loaded from dev run 6, which holds their fit
statistics. Dev run 7 fits the same reference sets with mass-weighted keys (/mw) and mass-weighted keys
in the query metric (/mq), and with --drift logs every seed's attention-matching error on the ssf
answer rows.
1. Reproduction: the loaded seeds' anchors and item scores against dev run 6's.
2. Fit: item-held sst error and p90 (G3) and the gold-row error of the anchor template (the dev runs
   4-6 calibration), per seed.
3. Recovery R (as dev run 6, section 4) and the key-placement contrasts with session-cluster intervals.
4. Drift: per seed, mean over heads of the per-row errors on the ssf answer rows: e_star (at CKstar's
   queries), e_tot (the seed's output at its own run's queries against the memory's output at
   CKstar's), e_run, query drift and memory-output drift. Rows: lookup rows (the memory raises that
   answer token by >= 1 nat), split by half, and the first answer token of memory items.
"""
import json
import re
import sys

import numpy as np

run = [json.loads(line) for line in open(sys.argv[1])][1:]
base = [json.loads(line) for line in open(sys.argv[2])][1:]
qids = list(dict.fromkeys(r["qid"] for r in run if r["block"] == "meta"))
KEYS = ("/mw", "/mq")
SH = "30"


def blocks(rows, kind, q, s="p"):
    return [r for r in rows if r["block"] == kind and r["qid"] == q and r.get("store", s) == s]


print("== 1. reproduction: loaded k-means seeds against dev run 6 ==")
for q in qids:
    ld = blocks(run, "loaded", q)
    refs = ld[0]["refs"] if ld else []
    for ref in refs:
        c = f"S64{ref}"
        a7 = next((r for r in blocks(run, "anchor", q) if r["cond"] == c), None)
        a6 = next((r for r in blocks(base, "anchor", q) if r["cond"] == c), None)
        i7 = next((r for r in blocks(run, "items", q) if r["cond"] == c), None)
        i6 = next((r for r in blocks(base, "items", q) if r["cond"] == c), None)
        da = max(abs(x - y) for x, y in zip(a7["lp"], a6["lp"])) if a7 and a6 else float("nan")
        di = (max(abs(x - y) for u, v in zip(i7["lp"], i6["lp"]) for x, y in zip(u, v))
              if i7 and i6 else float("nan"))
        print(f"  {q} {ref:16s} anchor max |diff| {da:.2e}  items max |diff| {di:.2e}")

print("\n== 2. fit: item-held sst error (mean over heads; p90) and anchor gold-row error, against dev run 6's sst ==")
for q in qids:
    hs = {}
    for rows in (base, run):
        for f in blocks(rows, "fit", q):
            if f["ratio"] != 64:
                continue
            for h in f["heads"]:
                hs.setdefault(h["ref"], []).append(h)
    if "sst" not in hs:
        continue
    m = lambda ref, f: float(np.mean([f(h) for h in hs[ref]]))  # noqa: E731
    e0, p0 = m("sst", lambda h: h["fit"]["out_rel"]), m("sst", lambda h: h["fit"]["row_p90"])
    for ref in hs:
        if not (ref in ("sst", f"sst+ssfd%{SH}", f"sst+ssfAd%{SH}") or ref.endswith(KEYS)):
            continue
        e, p9 = m(ref, lambda h: h["fit"]["out_rel"]), m(ref, lambda h: h["fit"]["row_p90"])
        g = m(ref, lambda h: h["probe_gold_fit"]["out_rel"]) if "probe_gold_fit" in hs[ref][0] else float("nan")
        an = m(ref, lambda h: h["probe_anchor_fit"]["out_rel"]) if "probe_anchor_fit" in hs[ref][0] else float("nan")
        anc = next((r for r in blocks(run, "anchor", q) if r["cond"] == f"S64{ref}"), None) or \
            next((r for r in blocks(base, "anchor", q) if r["cond"] == f"S64{ref}"), None)
        rid = np.unique([h["ridge"] for h in hs[ref]], return_counts=True)
        print(f"  {q} {ref:20s} held {e:.3f} ({p9:.3f})  G3 {e - e0:+.3f} ({p9 - p0:+.3f})  "
              f"gold rows {g:.3f}  anchor rows {an:.3f}  anchor lp "
              + (" ".join(f"{x:+6.2f}" for x in anc["lp"]) if anc else "-"))
        print(f"      bias loss {m(ref, lambda h: h['bias_loss']):.3f}  held logmass {m(ref, lambda h: h['fit']['logmass_rmse']):.3f}  "
              f"J_fit {hs[ref][0]['J_fit']} J_held {hs[ref][0]['J_held']} extra_wsum {hs[ref][0].get('extra_wsum', 0):.0f}  ridges "
              + " ".join(f"{r_:.0e}:{n_}" for r_, n_ in zip(*rid)))
    ss7 = blocks(run, "selfstudy", q)
    ss6 = blocks(base, "selfstudy", q)
    f7, f6 = blocks(run, "ssf", q), blocks(base, "ssf", q)
    if ss7 and ss6 and f7 and f6:
        same_items = all(a["lp_star"] == b["lp_star"] and a["lp_c"] == b["lp_c"] for a, b in zip(f7[0]["items"], f6[0]["items"]))
        sh7, sh6 = f7[0]["shares"], f6[0]["shares"]
        print(f"    plumbing: sst cache {ss7[0]['cache'] == ss6[0]['cache']} (cached {ss7[0]['cached']}), n_queries "
              f"{ss7[0]['n_queries'] == ss6[0]['n_queries']}; ssf file {f7[0]['file'] == f6[0]['file']}, half {f7[0]['half_a'] == f6[0]['half_a']}, "
              f"rows {f7[0]['n_rows'] == f6[0]['n_rows']}, items lp {same_items}; weight sums "
              + " ".join(f"{k}:{abs(sh7[k]['wsum'] - sh6[k.split('/')[0]]['wsum']) < 1e-3 * sh6[k.split('/')[0]]['wsum']}"
                         for k in sh7 if k.split("/")[0] in sh6))

print("\n== 3. recovery R over memory items (delta >= 1) and key-placement contrasts ==")
for q in qids:
    f = blocks(run, "ssf", q)
    if not f:
        continue
    f = f[0]
    its = f["items"]
    star = np.array([sum(it["lp_star"]) for it in its])
    c = np.array([sum(it["lp_c"]) for it in its])
    d = star - c
    mem = d >= 1
    ina = np.array([it["session"] in set(f["half_a"]) for it in its])
    sess = np.array([it["session"] for it in its])
    star0, c0 = np.array([it["lp_star"][0] for it in its]), np.array([it["lp_c"][0] for it in its])
    res = {r["cond"][3:]: np.array([sum(x) for x in r["lp"]]) for r in blocks(run, "items", q)}
    res0 = {r["cond"][3:]: np.array([x[0] for x in r["lp"]]) for r in blocks(run, "items", q)}
    print(f"  {q}: {len(its)} items; memory items {mem.sum()} (A {sum(mem & ina)}, B {sum(mem & ~ina)})")
    for cond, lp in res.items():
        R = lambda m_: (lp[m_] - c[m_]).sum() / max((star[m_] - c[m_]).sum(), 1e-9)  # noqa: E731
        fr = lambda m_: float(np.mean(((lp - c) / np.maximum(d, 1e-9))[m_] >= 0.5))  # noqa: E731
        print(f"    {cond:20s} R all {R(mem):+.3f}  A {R(mem & ina):+.3f}  B {R(mem & ~ina):+.3f}  "
              f"items recovering >= 0.5: all {fr(mem):.2f} A {fr(mem & ina):.2f} B {fr(mem & ~ina):.2f}")

    def diff(x, y, m_, label, first=False):
        """Paired difference of R on the same items: 95% session-cluster bootstrap (2,000 resamples, seed
        42, fresh per call) and the leave-one-session-out range (as dev run 6)."""
        lin([(x, 1), (y, -1)], m_, label, first)

    def lin(terms, m_, label, first=False):
        """As diff, for a linear combination of seeds' R, sum_k coef_k R(cond_k) (a difference of
        differences when the coefficients are +1, -1, -1, +1)."""
        rs, st_, cc = (res0, star0, c0) if first else (res, star, c)
        m_ = m_ & (star0 - c0 >= 1) if first else m_
        missing = [x for x, _ in terms if x not in rs]
        if missing:
            print(f"    {label:44s} n/a (missing {', '.join(missing)})")
            return
        idx = np.flatnonzero(m_)
        if len(idx) == 0:
            print(f"    {label:44s} n/a (no items)")
            return
        us, inv = np.unique(sess[idx], return_inverse=True)
        N = np.bincount(inv, sum(k * rs[x][idx] for x, k in terms), len(us))
        D = np.bincount(inv, (st_ - cc)[idx], len(us))
        bs = np.random.default_rng(42).integers(0, len(us), (2000, len(us)))
        lo, hi = np.percentile(N[bs].sum(1) / np.maximum(D[bs].sum(1), 1e-9), [2.5, 97.5])
        loo = (N.sum() - N) / np.maximum(D.sum() - D, 1e-9)
        print(f"    {label:44s} {N.sum() / max(D.sum(), 1e-9):+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"LOSO [{loo.min():+.3f}, {loo.max():+.3f}]  ({len(idx)} items, {len(us)} sessions)")

    fd, fa = f"sst+ssfd%{SH}", f"sst+ssfAd%{SH}"
    for k in KEYS:
        if not any(x.endswith(k) for x in res):
            continue
        print(f"    keys {k}:")
        diff(f"sst{k}", "sst", mem, f"KEYS sst: dR_all sst{k} - sst")
        diff(fd + k, fd, mem, f"KEYS: dR_all ssfd%{SH}{k} - ssfd%{SH}")
        diff(fd + k, f"sst{k}", mem, f"CAPACITY same keys dR_all ssfd%{SH}{k} - sst{k}")
        lin([(fd + k, 1), (f"sst{k}", -1), (fd, -1), ("sst", 1)], mem, "ENTRIES DiD (ssfd-sst) under keys - k-means")
        diff(fd + k, "sst", mem, f"NET dR_all ssfd%{SH}{k} - sst (k-means)")
        diff(fa + k, f"sst{k}", mem & ina, f"dose point dR_A ssfAd%{SH}{k} - sst{k}")
        diff(fa + k, f"sst{k}", mem & ~ina, f"TRANSFER DiD R_B ssfAd%{SH}{k} - sst{k}")
        diff(fa + k, f"sst{k}", mem & ~ina, "  first answer token only (its delta >= 1)", first=True)
        diff(fa + k, "sst", mem & ~ina, f"UNSEEN R_B ssfAd%{SH}{k} - sst")
        pl = ~mem & ~ina
        if fa + k in res and pl.any():
            print(f"    placebo {fa + k:22s} mean item lp shift vs sst{k} on B items with delta < 1: "
                  f"{np.mean(res[fa + k][pl] - res[f'sst{k}'][pl]):+.3f} ({pl.sum()} items)")
    if "sst+ssfd%30/mw" in res and "sst+ssfd%30/mq" in res:
        diff("sst+ssfd%30/mq", "sst+ssfd%30/mw", mem, "METRIC: dR_all ssfd%30/mq - ssfd%30/mw")

    # anchor: f against dev run 4's sst and probe (as dev run 6), and t against this question's sst
    meta = next(r for r in run if r["block"] == "meta" and r["qid"] == q)
    # the generated item closest to the test question (word Jaccard), in its own template (descriptive)
    w = lambda s: set(re.findall(r"[a-z']+", s.lower()))  # noqa: E731
    jac = [len(w(it["question"]) & w(meta["question"])) / len(w(it["question"]) | w(meta["question"])) for it in its]
    gi = int(np.argmax(jac))
    print(f"    test-matching item {its[gi]['question']!r} -> {its[gi]['answer']!r} (Jaccard {jac[gi]:.2f}, session "
          f"{its[gi]['session']}, half {'A' if ina[gi] else 'B'}; delta {d[gi]:+.2f}): item R, first-token lp (CK* "
          f"{star0[gi]:+.2f}, C {c0[gi]:+.2f})")
    print("      " + "  ".join(f"{cond} {(res[cond][gi] - c[gi]) / max(d[gi], 1e-9):+.2f} ({res0[cond][gi]:+.2f})" for cond in res))
    an = {r["cond"]: np.array(r["lp"]) for r in blocks(run, "anchor", q)}
    if "CKstar" in an and "S64sst" in an:
        if q == "0862e8bf":  # f from dev run 4's sst (-9.39) and probe (-0.03) on 'L' (dev runs 6 and 7)
            print("    anchor f('L') = (lp + 9.39) / 9.36: " + "  ".join(
                f"{cond[3:]} {(an[cond][0] + 9.39) / 9.36:+.2f}" for cond in an if cond.startswith("S64")))
        gap = an["CKstar"] - an["C"]
        print(f"    anchor {meta['anchor_gold']!r}: CK* - C per token " + " ".join(f"{x:+6.2f}" for x in gap))
        for cond in an:
            if cond.startswith("S64") and cond != "S64sst":
                t = (an[cond] - an["S64sst"]) / np.where(np.abs(gap) > 1e-6, gap, np.nan)
                print(f"      {cond[3:]:20s} lp " + " ".join(f"{x:+6.2f}" for x in an[cond])
                      + "   t " + " ".join(f"{x:+6.2f}" for x in t))

print("\n== 4. drift on the ssf answer rows (mean over the 64 heads of per-row errors) ==")
MET = ("e_star", "e_tot", "e_run", "q_drift", "y_drift")
for q in qids:
    f = blocks(run, "ssf", q)
    dr = blocks(run, "drift", q)
    if not f or not dr:
        continue
    its = f[0]["items"]
    half = set(f[0]["half_a"])
    tok_d = np.concatenate([np.array(it["lp_star"]) - np.array(it["lp_c"]) for it in its])
    item_d = np.concatenate([np.full(len(it["lp_star"]), sum(it["lp_star"]) - sum(it["lp_c"])) for it in its])
    first = np.concatenate([np.arange(len(it["lp_star"])) == 0 for it in its])
    a_row = np.concatenate([np.full(len(it["lp_star"]), it["session"] in half) for it in its])
    sets = {"lookup (all)": tok_d >= 1, "lookup A": (tok_d >= 1) & a_row, "lookup B": (tok_d >= 1) & ~a_row,
            "first tok, mem items": first & (item_d >= 1), "other rows": tok_d < 1}
    print(f"  {q}: rows {len(tok_d)}; " + ", ".join(f"{k} {v.sum()}" for k, v in sets.items()))
    print(f"    {'seed':20s} {'rows':22s} " + " ".join(f"{m_:>8s}" for m_ in MET))
    per_layer, pooled = {}, {}
    for r in dr:
        H = r["heads"]
        assert all(len(h["e_star"]) == len(tok_d) for h in H), "row count mismatch"
        arr = {m_: np.array([h[m_] for h in H]) for m_ in MET}  # [heads, rows]
        for name, sel in sets.items():
            print(f"    {r['cond'][3:]:20s} {name:22s} " + " ".join(f"{arr[m_][:, sel].mean():8.3f}" for m_ in MET))
        pooled[r["cond"][3:]] = {m_: float(arr[m_][:, sets["lookup (all)"]].mean()) for m_ in MET}
        layers = np.array([h["layer"] for h in H])
        per_layer[r["cond"][3:]] = {L: {m_: arr[m_][layers == L][:, sets["lookup A"] | sets["lookup B"]].mean() for m_ in MET}
                                    for L in sorted(set(layers.tolist()))}
    print("    drift reading (dev run 7 rule; pooled lookup rows): g = e_star(base) - e_star(seed), "
          "g_run = e_tot(base) - e_tot(seed), delta = e_tot - e_star")
    for seed, b in ((f"sst+ssfd%{SH}", "sst"), (f"sst+ssfd%{SH}/mw", "sst/mw"), (f"sst+ssfd%{SH}/mq", "sst/mw")):
        if seed not in pooled or b not in pooled:
            continue
        es, eb = pooled[seed]["e_star"], pooled[b]["e_star"]
        g, g_run = eb - es, pooled[b]["e_tot"] - pooled[seed]["e_tot"]
        budget, drift = es >= 0.8 * eb, g > 0 and g_run < 0.5 * g
        call = {(True, True): "both", (True, False): "budget", (False, True): "drift", (False, False): "neither"}[budget, drift]
        print(f"      {seed:18s} vs {b:7s} e_star {es:.3f} / {eb:.3f} (ratio {es / eb:.2f})  g {g:+.3f}  g_run {g_run:+.3f}  "
              f"delta seed {pooled[seed]['e_tot'] - es:+.3f}  delta base {pooled[b]['e_tot'] - eb:+.3f}  -> {call}")
    for cond in [k for k in per_layer if re.match(r"sst\+ssfd%\d+", k) or k == "sst"]:
        print(f"    per layer, lookup rows, {cond}: layer e_star / e_tot / q_drift")
        print("      " + "  ".join(f"{L}:{v['e_star']:.2f}/{v['e_tot']:.2f}/{v['q_drift']:.2f}" for L, v in per_layer[cond].items()))
