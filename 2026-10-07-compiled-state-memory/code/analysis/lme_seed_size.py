"""Stage 1 dev run 8 summary (exploratory, dev; LABNOTES 2026-10-12 morning, with its 06:07 amendment): seed
size under mass keys, the 60% dose, and the leak check for mass keys on the absent store.

uv run --with numpy python analysis/lme_seed_size.py --base ../results/phase2/lme_seed_dev7.jsonl \
  --size ../results/phase2/lme_seed_dev8.jsonl --r16 ../results/phase2/lme_seed_dev8r16.jsonl \
  --dose ../results/phase2/lme_seed_dev8d.jsonl \
  --absent ../results/phase2/lme_seed_dev8a.jsonl --absent_base ../results/phase2/lme_seed_dev6a.jsonl

Every file but --base is optional, and a file still being written is read as far as it goes; verdicts on
missing pieces print as "incomplete". dev run 7's file (--base) holds the r = 64 mass-key seeds and takes
precedence for them; the part files fill in only what it lacks.
0. Leak gate (part 1, absent store): reproduction of the loaded k-means sst(a) against dev run 6a, the
   plumbing, each gold token's rise over max(sst(a), CK*(a)) and the flag (a flag on either memory is a
   flag); capacity on the absent memories (descriptive).
1. Reproduction (G1') per part: the loaded r = 64 seeds' anchors and item scores against dev run 7's, and
   the plumbing. A part that fails is void and left out below.
2. Fit per r: item-held sst error and p90, G3 against sst/mw at the same r, fit seconds and seed size.
3. Recovery R per r and the contrasts (session-cluster intervals and LOSO ranges, as dev run 7): SIZE,
   SIZE sst, SIZE unseen, CAPACITY(r), TRANSFER/mw(r) and DOSE. A call is void on a memory if a ssf seed
   it reads fails G3 there.
4. Anchors and the test-matching item per r.
5. Drift per r on the lookup rows.
6. Calls, the dose rule, the compact default and the end-to-end gate.
"""
import argparse
import json
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="dev run 7 (the r = 64 mass-key seeds and their fit statistics)")
ap.add_argument("--size")
ap.add_argument("--r16")
ap.add_argument("--dose")
ap.add_argument("--absent")
ap.add_argument("--absent_base", help="dev run 6a (the k-means sst on the absent store)")
ap.add_argument("--json", help="write R per memory, r and seed, with seed MB, sst/mw held error and the calls (dev run 8's "
                "seed-42 summary, ../results/phase2/lme_seed_size_summary.json, is Figure 4's input; write other runs elsewhere)")
ap.add_argument("--ref", help="a --json summary from another fit seed to set this run beside (written for dev run 9, which was dropped and never run)")
args = ap.parse_args()
if not os.path.exists(args.base):
    ap.error(f"--base {args.base} does not exist")
FIT_SEED = json.loads(open(args.base).readline()).get("fit_seed", 42)


def load(p):
    if not p or not os.path.exists(p):
        return []
    rows = []
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a line still being written
            break
    return rows[1:]


base, absent, absent_base = load(args.base), load(args.absent), load(args.absent_base)
parts = [("size", load(args.size)), ("r16", load(args.r16)), ("dose", load(args.dose))]
QIDS = ("0862e8bf", "6f9b354f")
FD, FA, F60, SM = "sst+ssfd%30/mw", "sst+ssfAd%30/mw", "sst+ssfd%60/mw", "sst/mw"
RATIOS = (256, 128, 64, 32, 16)
COND = re.compile(r"^S(\d+)(.+)$")
BYTES_PER_ENTRY = 64 * (2 * 256 * 2 + 4)  # 64 KV heads, bf16 key and value (256-dim), fp32 bias


def blocks(rows, kind, q, s="p"):
    return [r for r in rows if r["block"] == kind and r["qid"] == q and r.get("store", s) == s]


def cond_key(c):
    m = COND.match(c)
    return (int(m[1]), m[2]) if m else None


def call3(est, lo, hi, loo):
    if est >= 0.1 and lo > 0 and loo.min() > 0:
        return "helps"
    if est <= -0.1 and hi < 0 and loo.max() < 0:
        return "hurts"
    return "no difference"


def cap_call(est):
    return "holds" if est >= 0.5 else "partial" if est >= 0.2 else "not"


def both(per):
    if any(per.get(q) is None for q in QIDS):
        return "incomplete"
    v = {per[q] for q in QIDS}
    return v.pop() if len(v) == 1 else "mixed"


def items_of(rows, q, s):
    f = blocks(rows, "ssf", q, s)
    return f[0] if f else None


def item_arrays(f):
    its = f["items"]
    star = np.array([sum(it["lp_star"]) for it in its])
    c = np.array([sum(it["lp_c"]) for it in its])
    star0, c0 = np.array([it["lp_star"][0] for it in its]), np.array([it["lp_c"][0] for it in its])
    ina = np.array([it["session"] in set(f["half_a"]) for it in its])
    sess = np.array([it["session"] for it in its])
    return its, star, c, star0, c0, ina, sess


def make_lin(res, res0, star, c, star0, c0, sess):
    def lin(terms, m_, label, first=False):
        """sum_k coef_k R(cond_k) over the items in m_: 95% session-cluster bootstrap (2,000 resamples, seed
        42, fresh per call) and the leave-one-session-out range (as dev runs 6 and 7). None if a seed is missing."""
        rs, st_, cc = (res0, star0, c0) if first else (res, star, c)
        m_ = m_ & (star0 - c0 >= 1) if first else m_
        missing = [x for x, _ in terms if x not in rs]
        if missing:
            print(f"    {label:46s} n/a (missing {', '.join(f'r{a} {b}' for a, b in missing)})")
            return None
        idx = np.flatnonzero(m_)
        us, inv = np.unique(sess[idx], return_inverse=True)
        N = np.bincount(inv, sum(k * rs[x][idx] for x, k in terms), len(us))
        D = np.bincount(inv, (st_ - cc)[idx], len(us))
        bs = np.random.default_rng(42).integers(0, len(us), (2000, len(us)))
        lo, hi = np.percentile(N[bs].sum(1) / np.maximum(D[bs].sum(1), 1e-9), [2.5, 97.5])
        loo = (N.sum() - N) / np.maximum(D.sum() - D, 1e-9)
        est = N.sum() / max(D.sum(), 1e-9)
        print(f"    {label:46s} {est:+.3f}  [{lo:+.3f}, {hi:+.3f}]  LOSO [{loo.min():+.3f}, {loo.max():+.3f}]  "
              f"({len(idx)} items, {len(us)} sessions)")
        return est, lo, hi, loo
    return lin


def mean(hs, k, fn):
    return float(np.mean([fn(h) for h in hs[k]]))


# ---------------------------------------------------------------- 0. leak gate
leak = {}  # qid -> "flag" | "no flag" | "void" | None (incomplete)
if absent:
    print("== 0. leak gate, absent store (part 1) ==")
    for q in QIDS:
        meta = next((r for r in absent if r["block"] == "meta" and r["qid"] == q), None)
        if meta is None:
            continue
        an = {r["cond"]: np.array(r["lp"]) for r in blocks(absent, "anchor", q, "a")}
        an0 = {r["cond"]: np.array(r["lp"]) for r in blocks(absent_base, "anchor", q, "a")}
        it8 = {r["cond"]: r["lp"] for r in blocks(absent, "items", q, "a")}
        it6 = {r["cond"]: r["lp"] for r in blocks(absent_base, "items", q, "a")}
        repro = None
        if "S64sst" in an and "S64sst" in an0 and "S64sst" in it8 and "S64sst" in it6:
            da = float(np.max(np.abs(an["S64sst"] - an0["S64sst"])))
            di = max(abs(x - y) for u, v in zip(it8["S64sst"], it6["S64sst"]) for x, y in zip(u, v))
            repro = da == 0 and di == 0
            print(f"  {q} reproduction of the loaded k-means sst(a) against dev run 6a: anchor max |diff| {da:.2e}  "
                  f"items max |diff| {di:.2e} -> {'pass' if repro else 'FAIL: part 1 void'}")
        f = items_of(absent, q, "a")
        ss = next(iter(blocks(absent, "selfstudy", q, "a")), None)
        ft = blocks(absent, "fit", q, "a")
        gw = set(re.findall(r"\w+", (meta["anchor_gold"] or "").lower())) - {"a", "an", "the", "of", "my", "to", "in"}
        hits = [it["answer"] for it in (f["items"] if f else []) if gw and gw <= set(re.findall(r"\w+", it["answer"].lower()))]
        n_ok = all(x["n"] == meta["n_tok"]["a"] // x["ratio"] for x in ft) if ft else None
        plumb = [bool(f and "_a_all_" in f["file"]), not hits, bool(ss and ss["cache"] and "_a_sst_" in ss["cache"]), bool(n_ok)]
        print(f"  {q} plumbing: ssf file {os.path.basename(f['file']) if f else None} (absent {plumb[0]}); "
              f"items with the gold {hits}; sst cache {os.path.basename(ss['cache']) if ss and ss['cache'] else None} "
              f"(absent {plumb[2]}); N = n_tok(a) // r {n_ok} ({[x['n'] for x in ft]} vs {meta['n_tok']['a']} // 64)")
        if not {"C", "CKstar", "S64sst"} <= set(an):
            continue
        env = np.maximum(an["S64sst"], an["CKstar"])
        print(f"    gold {meta['anchor_gold']!r}: C " + " ".join(f"{x:+6.2f}" for x in an["C"]) + "  CK*(a) "
              + " ".join(f"{x:+6.2f}" for x in an["CKstar"]) + "  sst(a) " + " ".join(f"{x:+6.2f}" for x in an["S64sst"]))
        flag, seen = False, 0
        for seed in (SM, FD):
            c_ = f"S64{seed}"
            if c_ not in an:
                continue
            seen += 1
            rise = an[c_] - env
            fl = bool((rise > 1).any() or (q == "0862e8bf" and an[c_][0] > -8))
            flag |= fl
            print(f"    {seed:16s} lp " + " ".join(f"{x:+6.2f}" for x in an[c_]) + "   over max(sst(a), CK*(a)) "
                  + " ".join(f"{x:+6.2f}" for x in rise) + f"   flag {fl}")
        if repro is False or not all(plumb):
            leak[q] = "void" if repro is False else "plumbing FAIL"
        elif flag:
            leak[q] = "flag"
        elif seen == 2 and repro:
            leak[q] = "no flag"
        if f is not None:
            its, star, c, star0, c0, ina, sess = item_arrays(f)
            res = {k: np.array([sum(x) for x in v]) for k, v in ((cond_key(c_), lp) for c_, lp in it8.items()) if k}
            res0 = {k: np.array([x[0] for x in v]) for k, v in ((cond_key(c_), lp) for c_, lp in it8.items()) if k}
            mem = star - c >= 1
            lin = make_lin(res, res0, star, c, star0, c0, sess)
            print(f"    absent memory items {mem.sum()} (descriptive)")
            lin([((64, FD), 1), ((64, SM), -1)], mem, "CAPACITY(a) ssfd%30/mw - sst/mw")
            lin([((64, SM), 1), ((64, "sst"), -1)], mem, "KEYS sst(a) sst/mw - sst")
            hs = {}
            for rows in (absent, absent_base):
                for fb in blocks(rows, "fit", q, "a"):
                    for h in fb["heads"]:
                        hs.setdefault((fb["ratio"], h["ref"]), []).append(h)
            if (64, "sst") in hs:
                e0 = mean(hs, (64, "sst"), lambda h: h["fit"]["out_rel"])
                for seed in (SM, FD):
                    if (64, seed) in hs:
                        print(f"    held {seed:16s} {mean(hs, (64, seed), lambda h: h['fit']['out_rel']):.3f} "
                              f"(p90 {mean(hs, (64, seed), lambda h: h['fit']['row_p90']):.3f}); k-means sst(a) {e0:.3f}")

# ---------------------------------------------------------------- 1. reproduction
print("\n== 1. reproduction (G1'): loaded r = 64 seeds against dev run 7 ==")
void = set()
for name, rows in parts:
    ok = True
    for q in QIDS:
        for ld in blocks(rows, "loaded", q):
            if not ld["refs"]:
                print(f"  {name} {q} r{ld['ratio']}: no saved seeds, all fitted")
                continue
            for ref in ld["refs"]:
                c_ = f"S{ld['ratio']}{ref}"
                a8 = next((r for r in blocks(rows, "anchor", q) if r["cond"] == c_), None)
                a7 = next((r for r in blocks(base, "anchor", q) if r["cond"] == c_), None)
                i8 = next((r for r in blocks(rows, "items", q) if r["cond"] == c_), None)
                i7 = next((r for r in blocks(base, "items", q) if r["cond"] == c_), None)
                if not (a8 and i8):
                    print(f"  {name} {q} {ref:16s} not scored yet")
                    continue
                da = max(abs(x - y) for x, y in zip(a8["lp"], a7["lp"])) if a7 else float("nan")
                di = max(abs(x - y) for u, v in zip(i8["lp"], i7["lp"]) for x, y in zip(u, v)) if i7 else float("nan")
                ok &= da == 0 and di == 0
                print(f"  {name} {q} {ref:16s} anchor max |diff| {da:.2e}  items max |diff| {di:.2e}")
        s7, s8 = blocks(base, "selfstudy", q), blocks(rows, "selfstudy", q)
        f7, f8 = items_of(base, q, "p"), items_of(rows, q, "p")
        if s7 and s8 and f7 and f8:
            same_items = len(f8["items"]) == len(f7["items"]) and all(
                a["lp_star"] == b["lp_star"] and a["lp_c"] == b["lp_c"] for a, b in zip(f8["items"], f7["items"]))
            pl = {"sst cache": s8[0]["cache"] == s7[0]["cache"], "ssf file": f8["file"] == f7["file"],
                  "half": f8["half_a"] == f7["half_a"], "rows": f8["n_rows"] == f7["n_rows"], "items lp": same_items}
            pl.update({f"wsum {k}": abs(f8["shares"][k]["wsum"] - f7["shares"][k]["wsum"]) < 1e-3 * f7["shares"][k]["wsum"]
                       for k in f8["shares"] if k in f7["shares"]})
            ok &= all(pl.values())
            print(f"    {name} {q} plumbing: " + ", ".join(f"{k} {v}" for k, v in pl.items()))
    if rows and not ok:
        void.add(name)
        print(f"  part {name}: VOID (G1'), left out below")
use = [("base", base)] + [(n, r) for n, r in parts if n not in void]

# ---------------------------------------------------------------- 2. fit and G3
print("\n== 2. fit per r: item-held sst error (mean over heads; p90), G3 against sst/mw at the same r ==")
g3 = {}  # (q, r, ref) -> passes
size_mb, held_sm = {}, {}  # (q, r) -> seed MB; sst/mw held error
for q in QIDS:
    hs, secs, src = {}, {}, {}
    for name, rows in use:  # base first: a (r, ref) fitted in an earlier file is not appended to
        for fb in blocks(rows, "fit", q):
            for h in fb["heads"]:
                k = (fb["ratio"], h["ref"])
                if src.setdefault(k, name) == name:
                    hs.setdefault(k, []).append(h)
            secs.setdefault(fb["ratio"], []).append(fb["seconds"] / max(len(fb["refs"]), 1))
    for r in sorted({r for r, _ in hs}, reverse=True):
        if (r, SM) not in hs:
            continue
        e0, p0 = mean(hs, (r, SM), lambda h: h["fit"]["out_rel"]), mean(hs, (r, SM), lambda h: h["fit"]["row_p90"])
        n = hs[(r, SM)][0]["n"]
        size_mb[q, r], held_sm[q, r] = n * BYTES_PER_ENTRY / 1e6, e0
        print(f"  {q} r{r:<3d} N {n:5d} ({n * BYTES_PER_ENTRY / 1e6:6.1f} MB)  fit s/seed "
              + "/".join(f"{x:.0f}" for x in secs.get(r, [])))
        for ref in (SM, FD, FA, F60):
            if (r, ref) not in hs:
                continue
            e, p9 = mean(hs, (r, ref), lambda h: h["fit"]["out_rel"]), mean(hs, (r, ref), lambda h: h["fit"]["row_p90"])
            ok = e - e0 <= 0.02 and p9 - p0 <= 0.03
            g3[q, r, ref] = ok
            an = mean(hs, (r, ref), lambda h: h["probe_anchor_fit"]["out_rel"]) if "probe_anchor_fit" in hs[(r, ref)][0] else float("nan")
            print(f"    {ref:16s} held {e:.3f} ({p9:.3f})  G3 {e - e0:+.3f} ({p9 - p0:+.3f}) {'pass' if ok else 'FAIL'}  "
                  f"anchor rows {an:.3f}  bias loss {mean(hs, (r, ref), lambda h: h['bias_loss']):.3f}")

# ---------------------------------------------------------------- 3. recovery and contrasts
print("\n== 3. recovery R over memory items (delta >= 1) per r, and the dev run 8 contrasts ==")
calls = {}  # readout -> {qid: call}
Rall, RA, RB, tm = {}, {}, {}, {}
for q in QIDS:
    f = items_of(base, q, "p")
    if f is None:
        continue
    its, star, c, star0, c0, ina, sess = item_arrays(f)
    res, res0 = {}, {}
    for name, rows in use:  # base first; part files fill in only keys it lacks
        g = items_of(rows, q, "p")
        if g is None:
            continue
        assert len(g["items"]) == len(f["items"]) and all(
            a["lp_star"] == b["lp_star"] and a["lp_c"] == b["lp_c"] for a, b in zip(g["items"], f["items"])), \
            f"item scores differ between files for {q} ({name})"
        for r in blocks(rows, "items", q):
            k = cond_key(r["cond"])
            if k is None or k in res:
                continue
            res[k] = np.array([sum(x) for x in r["lp"]])
            res0[k] = np.array([x[0] for x in r["lp"]])
    mem = star - c >= 1
    d = star - c
    print(f"  {q}: memory items {mem.sum()} (A {sum(mem & ina)}, B {sum(mem & ~ina)})")
    for k in sorted(res, key=lambda k: (-k[0], k[1])):
        if not k[1].endswith("/mw"):
            continue
        lp = res[k]
        R = lambda m_: (lp[m_] - c[m_]).sum() / max((star[m_] - c[m_]).sum(), 1e-9)  # noqa: E731
        fr = lambda m_: float(np.mean(((lp - c) / np.maximum(d, 1e-9))[m_] >= 0.5))  # noqa: E731
        Rall[q, k] = R(mem)
        RA[q, k] = R(mem & ina)
        RB[q, k] = R(mem & ~ina)
        print(f"    r{k[0]:<3d} {k[1]:16s} R all {R(mem):+.3f}  A {R(mem & ina):+.3f}  B {R(mem & ~ina):+.3f}  "
              f"items recovering >= 0.5: {fr(mem):.2f}")
    lin = make_lin(res, res0, star, c, star0, c0, sess)

    def rec(name, out, deps, kind="3"):
        """Record a call for this memory. deps: the ssf seeds (r, ref) it reads, each of which must pass G3."""
        if out is None:
            calls.setdefault(name, {})[q] = None
            return
        gs = [g3.get((q, r, ref)) for r, ref in deps]
        if any(x is None for x in gs):
            calls.setdefault(name, {})[q] = None
        elif not all(gs):
            calls.setdefault(name, {})[q] = "void (G3)"
        else:
            calls.setdefault(name, {})[q] = call3(*out) if kind == "3" else cap_call(out[0])

    for r in (256, 128, 32, 16):
        rec(f"SIZE r{r}", lin([((r, FD), 1), ((64, FD), -1)], mem, f"SIZE r{r}: R_all ssfd%30/mw r{r} - r64"),
            [(r, FD), (64, FD)])
        rec(f"SIZE sst r{r}", lin([((r, SM), 1), ((64, SM), -1)], mem, f"SIZE sst r{r}: R_all sst/mw r{r} - r64"), [])
        rec(f"SIZE unseen r{r}", lin([((r, FA), 1), ((64, FA), -1)], mem & ~ina, f"SIZE unseen r{r}: R_B ssfAd%30/mw r{r} - r64"),
            [(r, FA), (64, FA)])
    for r in RATIOS:
        rec(f"CAPACITY r{r}", lin([((r, FD), 1), ((r, SM), -1)], mem, f"CAPACITY r{r}: ssfd%30/mw - sst/mw"), [(r, FD)], "cap")
        t_all = lin([((r, FA), 1), ((r, SM), -1)], mem & ~ina, f"TRANSFER r{r}: R_B ssfAd%30/mw - sst/mw")
        t_one = lin([((r, FA), 1), ((r, SM), -1)], mem & ~ina, "  first answer token only (its delta >= 1)", first=True)
        rec(f"TRANSFER r{r}", t_all, [(r, FA)])
        tc = calls[f"TRANSFER r{r}"][q]
        if tc in ("helps", "hurts") and (t_one is None or call3(*t_one) != tc):
            calls[f"TRANSFER r{r}"][q] = "no difference (first token)"
        pl = ~mem & ~ina
        if (r, FA) in res and (r, SM) in res and pl.any():
            print(f"      placebo r{r}: mean item lp shift ssfAd%30/mw vs sst/mw on B items with delta < 1: "
                  f"{np.mean(res[(r, FA)][pl] - res[(r, SM)][pl]):+.3f} ({pl.sum()} items)")
    rec("DOSE", lin([((64, F60), 1), ((64, FD), -1)], mem, "DOSE: R_all ssfd%60/mw - ssfd%30/mw (r64)"), [(64, F60), (64, FD)])
    rec("CAPACITY 60%", lin([((64, F60), 1), ((64, SM), -1)], mem, "CAPACITY 60%: ssfd%60/mw - sst/mw (r64)"), [(64, F60)], "cap")

    # the test-matching item, per r (descriptive)
    meta = next(r for r in base if r["block"] == "meta" and r["qid"] == q)
    w = lambda s: set(re.findall(r"[a-z']+", s.lower()))  # noqa: E731
    jac = [len(w(it["question"]) & w(meta["question"])) / len(w(it["question"]) | w(meta["question"])) for it in its]
    gi = int(np.argmax(jac))
    tm[q] = (its[gi], jac[gi], ina[gi], d[gi], star0[gi], c0[gi],
             {k: ((res[k][gi] - c[gi]) / max(d[gi], 1e-9), res0[k][gi]) for k in res if k[1].endswith("/mw")})

# ---------------------------------------------------------------- 4. anchors and the test-matching item
print("\n== 4. test-matching item and anchor per r (descriptive) ==")
for q, (it, jc, a_, dd, s0, c0_, per) in tm.items():
    print(f"  {q} item {it['question']!r} -> {it['answer']!r} (Jaccard {jc:.2f}, half {'A' if a_ else 'B'}, delta {dd:+.2f}; "
          f"first-token lp CK* {s0:+.2f}, C {c0_:+.2f}): item R (first-token lp)")
    for k in sorted(per, key=lambda k: (-k[0], k[1])):
        print(f"    r{k[0]:<3d} {k[1]:16s} {per[k][0]:+.2f} ({per[k][1]:+.2f})")
    an = {}
    for _, rows in use:
        for r in blocks(rows, "anchor", q):
            an.setdefault(r["cond"], np.array(r["lp"]))
    if "C" in an and "CKstar" in an:
        gap = an["CKstar"] - an["C"]
        meta = next(r for r in base if r["block"] == "meta" and r["qid"] == q)
        print(f"    anchor {meta['anchor_gold']!r}: C " + " ".join(f"{x:+6.2f}" for x in an["C"]) + "  CK* "
              + " ".join(f"{x:+6.2f}" for x in an["CKstar"]) + "   per seed: lp | (lp - C) / (CK* - C)")
        for c_ in sorted((x for x in an if cond_key(x) and cond_key(x)[1].endswith("/mw")),
                         key=lambda x: (-cond_key(x)[0], cond_key(x)[1])):
            fr = (an[c_] - an["C"]) / np.where(np.abs(gap) > 1e-6, gap, np.nan)
            print(f"    {c_:20s} " + " ".join(f"{x:+6.2f}" for x in an[c_]) + " | " + " ".join(f"{x:+5.2f}" for x in fr))

# ---------------------------------------------------------------- 5. drift
print("\n== 5. drift per r on the lookup rows (tokens the memory raises by >= 1 nat; mean over heads) ==")
MET = ("e_star", "e_tot", "e_run", "q_drift")
for q in QIDS:
    f = items_of(base, q, "p")
    if f is None:
        continue
    tok_d = np.concatenate([np.array(it["lp_star"]) - np.array(it["lp_c"]) for it in f["items"]])
    look = tok_d >= 1
    pooled = {}
    for _, rows in use:
        for r in blocks(rows, "drift", q):
            k = cond_key(r["cond"])
            if not k or not k[1].endswith("/mw") or k in pooled:
                continue
            arr = {m_: np.array([h[m_] for h in r["heads"]]) for m_ in MET}
            pooled[k] = {m_: float(arr[m_][:, look].mean()) for m_ in MET}
    print(f"  {q}: lookup rows {look.sum()}")
    for k in sorted(pooled, key=lambda k: (-k[0], k[1])):
        p = pooled[k]
        line = f"    r{k[0]:<3d} {k[1]:16s} " + " ".join(f"{m_} {p[m_]:.3f}" for m_ in MET) + f"  delta {p['e_tot'] - p['e_star']:+.3f}"
        b = pooled.get((k[0], SM))
        if k[1] != SM and b:
            g, g_run = b["e_star"] - p["e_star"], b["e_tot"] - p["e_tot"]
            budget, drift = p["e_star"] >= 0.8 * b["e_star"], g > 0 and g_run < 0.5 * g
            line += (f"  vs sst/mw: ratio {p['e_star'] / b['e_star']:.2f} g {g:+.3f} g_run {g_run:+.3f} -> "
                     + {(True, True): "both", (True, False): "budget", (False, True): "drift", (False, False): "neither"}[budget, drift])
        print(line)

# ---------------------------------------------------------------- 6. calls
print("\n== 6. calls (dev run 8 rule with the 06:07 amendment) ==")
if absent:
    st = [leak.get(q) for q in QIDS]
    gate = ("FLAG" if "flag" in st else "VOID / plumbing" if any(s in ("void", "plumbing FAIL") for s in st)
            else "no flag" if all(s == "no flag" for s in st) else "incomplete")
    print(f"  LEAK gate (either memory flags): {', '.join(f'{q} {s}' for q, s in zip(QIDS, st))} -> {gate}")
if void:
    print(f"  VOID parts (G1'): {', '.join(sorted(void))}")
print("  (a call needs both memories; 'void (G3)' = a ssf seed it reads fails G3 on that memory)")
for name, per in calls.items():
    print(f"  {name:22s} " + "  ".join(f"{q} {per.get(q)}" for q in QIDS) + f"   -> {both(per)}")


def have(*keys):
    return all(k in Rall or k in RB for k in keys)


# dose rule (dev run 6): take 60% if it passes G3 and gains >= 0.05 in R_all over 30%, on both memories
dose_take = None
if all((q, (64, F60)) in Rall and (q, (64, FD)) in Rall and (q, 64, F60) in g3 for q in QIDS):
    gains = {q: Rall[q, (64, F60)] - Rall[q, (64, FD)] for q in QIDS}
    dose_take = "60%" if all(gains[q] >= 0.05 and g3[q, 64, F60] for q in QIDS) else "30%"
    print(f"  DOSE rule: gain {', '.join(f'{q} {v:+.3f}' for q, v in gains.items())}; G3 at 60% "
          f"{[g3[q, 64, F60] for q in QIDS]} -> take {dose_take}")
else:
    print("  DOSE rule: incomplete")


def compact_ok(r):
    """The compact-default criterion at r > 64 on both memories (amended): R_all(ssfd%30/mw) and R_B(ssfAd%30/mw)
    within 0.05 of r = 64's, SIZE sst not 'hurts', G3 passing for both ssf seeds. None if a piece is missing."""
    out = True
    for q in QIDS:
        need = [(q, (r, FD)), (q, (64, FD)), (q, (r, FA)), (q, (64, FA))]
        if (not all(k in Rall for k in need) or calls.get(f"SIZE sst r{r}", {}).get(q) is None
                or (q, r, FD) not in g3 or (q, r, FA) not in g3):
            return None
        out &= (Rall[q, (r, FD)] >= Rall[q, (64, FD)] - 0.05 and RB[q, (r, FA)] >= RB[q, (64, FA)] - 0.05
                and calls[f"SIZE sst r{r}"][q] != "hurts" and g3[q, r, FD] and g3[q, r, FA])
    return out


cmp_ = {r: compact_ok(r) for r in (256, 128)}
if all(v is not None for v in cmp_.values()):
    ok = [r for r, v in cmp_.items() if v]
    print(f"  compact default: criterion r256 {cmp_[256]}, r128 {cmp_[128]} -> r{max(ok) if ok else 64}")
else:
    print(f"  compact default: incomplete ({', '.join(f'r{r} {v}' for r, v in cmp_.items())})")
for q in QIDS:
    print(f"    {q} R_all ssfd%30/mw " + " ".join(f"r{r}:{Rall[q, (r, FD)]:.3f}" for r in RATIOS if (q, (r, FD)) in Rall)
          + " | R_B ssfAd%30/mw " + " ".join(f"r{r}:{RB[q, (r, FA)]:.3f}" for r in RATIOS if (q, (r, FA)) in RB))

# end-to-end gate (amended): CAPACITY holds on both memories at the same r with G3 passing; at r = 128/256 also
# the compact criterion; 60% only if DOSE takes 60%. The end-to-end seed is the smallest such seed.
passes, missing = [], []
for r in RATIOS:
    cap = calls.get(f"CAPACITY r{r}", {})
    if any(cap.get(q) is None for q in QIDS) or (r > 64 and cmp_.get(r) is None):
        missing.append(f"r{r}")
        continue
    if all(cap[q] == "holds" for q in QIDS) and (r <= 64 or cmp_[r]):
        passes.append(f"r{r}")
cap60 = calls.get("CAPACITY 60%", {})
if dose_take is None or any(cap60.get(q) is None for q in QIDS):
    missing.append("60%")
elif dose_take == "60%" and all(cap60[q] == "holds" for q in QIDS):
    passes.append("r64 at 60%")
print(f"  end-to-end gate: passes at {', '.join(passes) if passes else 'none'}"
      + (f"; incomplete, not yet scored: {', '.join(missing)}" if missing else "")
      + (f" -> end-to-end seed {passes[0]}" if passes and not missing else ""))
if args.ref:
    # dev run 9 (LABNOTES 2026-10-12 evening, as amended): this fit seed beside the --ref summary's
    ref = json.load(open(args.ref))
    print(f"\n== 7. fit seed {FIT_SEED} against {os.path.basename(args.ref)} (fit seed {ref.get('_fit_seed', 42)}) ==")
    dmax = {}
    for q in QIDS:
        for r in RATIOS:
            for name in (SM, FD, FA):
                if (q, (r, name)) not in Rall or name not in ref.get(q, {}).get(str(r), {}).get("R", {}):
                    continue
                o = ref[q][str(r)]["R"][name]
                now = dict(all=Rall[q, (r, name)], A=RA[q, (r, name)], B=RB[q, (r, name)])
                for k in now:
                    dmax[q, r] = max(dmax.get((q, r), 0.0), abs(now[k] - o[k]))
                print(f"  {q} r{r:<3d} {name:16s} " + "  ".join(f"R {k} {now[k]:+.3f} ({o[k]:+.3f}, d {now[k] - o[k]:+.3f})"
                                                               for k in now))
    for (q, r), v in sorted(dmax.items()):
        print(f"  fit noise {q} r{r}: max |dR| {v:.3f}")
    d64 = {q: Rall[q, (64, FD)] - ref[q]["64"]["R"][FD]["all"] for q in QIDS
           if (q, (64, FD)) in Rall and FD in ref.get(q, {}).get("64", {}).get("R", {})}
    if len(d64) == len(QIDS):
        print(f"  compact default (r64): R_all ssfd%30/mw moves {', '.join(f'{q} {v:+.3f}' for q, v in d64.items())} -> "
              + ("stands (|d| < 0.15 on both)" if all(abs(v) < 0.15 for v in d64.values()) else "does not stand"))
    for r in (64, 32, 16):
        cap = calls.get(f"CAPACITY r{r}", {})
        if r not in {r_ for _, (r_, _) in Rall} or any(cap.get(q) is None for q in QIDS):
            continue
        met = all(cap[q] == "holds" for q in QIDS)
        print(f"  gate at r{r} under fit seed {FIT_SEED}: " + ", ".join(f"{q} {cap[q]}" for q in QIDS)
              + (" -> met here: fit-seed dependent, still not met (a pass needs both fits)" if met else " -> failure stands"))
    rc = ref.get("_calls", {})
    for name, per in calls.items():
        if name in rc and both(per) != "incomplete":
            a, b = both(per), both(rc[name])
            print(f"  {name:20s} seed {FIT_SEED} {a:16s} seed {ref.get('_fit_seed', 42)} {b:16s} -> "
                  + ("same" if a == b else "different (fit-seed dependent)"))
if args.json:
    out = {q: {str(r): dict(mb=size_mb[q, r], held_sstmw=held_sm[q, r],
                            R={ref: dict(all=Rall[q, (r, ref)], A=RA[q, (r, ref)], B=RB[q, (r, ref)])
                               for (q_, (r_, ref)) in Rall if q_ == q and r_ == r})
               for (q_, r) in size_mb if q_ == q} for q in QIDS}
    out.update(_fit_seed=FIT_SEED, _calls=calls)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
