"""Analysis for the LongMemEval_S hint check (PREREG.md §H1). Written before the run.

uv run --with numpy --with scipy python code/analysis/lme_hint.py results/phase2/lme_hint_<tag>.jsonl \
    data/raw/longmemeval/longmemeval_s_cleaned.json

Unit: question (80, design mix 24 SSU / 24 SSP / 16 KU / 16 MS). L = log-odds of <tool_call> as the
first assistant token, log p(call) − log p(not call).
Primary: Δ_C = L(C, present) − L(C, absent); mean, two-sided 95% t-interval; δ = 0.5.
  SIGNAL+   lower bound > 0 and mean ≥ δ          SIGNAL−  upper bound < 0 and mean ≤ −δ
  NULL      interval inside (−δ, δ), no question type with |mean| ≥ δ and a Holm-adjusted (4 types)
            p < 0.05, and a control registers presence: T (lower bound > 0, mean ≥ δ) ⇒ NULL;
            only M (upper bound < 0, mean ≤ −δ) ⇒ NULL (manipulation check only)
  otherwise INCONCLUSIVE (heterogeneous / no control registered / undecided)
Robustness (flags, not verdicts): percentile bootstrap interval (10,000, seed 42) and the greedy-path
call hazard Δ_path; if either gives a different verdict category the verdict is flagged.
Secondary, Holm over 13 named tests; descriptive sections print no p-values.
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, "code")
sys.path.insert(0, ".")
from statelm import lme  # noqa: E402

DELTA = 0.5
PLACEBO_TYPES = ("single-session-user", "single-session-preference")
LAST_WEEKEND = ("8550ddae", "a82c026e")
rng = np.random.default_rng(42)

raw = [json.loads(line) for line in open(sys.argv[1])]
header, raw = raw[0], raw[1:]
by = {}
for r in raw:  # last row per key wins (a resumed question rewrites its rows)
    by[(r["block"], r["qid"], r.get("cond"), r.get("store"))] = r
superseded = len(raw) - len(by)
meta = {k[1]: r for k, r in by.items() if k[0] == "meta"}
dec = {(k[1], k[2], k[3]): r for k, r in by.items() if k[0] == "decide"}
anc = {(k[1], k[2], k[3]): r for k, r in by.items() if k[0] == "anchor"}


def expected(q):
    m = meta[q]
    keys = {("decide", c, s) for c in ("C", "T", "CT", "M") for s in ("p", "a")} | {("decide", "A", None), ("decide", "S", None)}
    if m["qtype"] in PLACEBO_TYPES and m.get("rep2"):
        keys |= {("decide", "C", "a2"), ("decide", "T", "a2")}
    if m["anchor_gold"]:
        keys |= {("anchor", "A", None)} | {("anchor", c, s) for c in ("C", "M") for s in ("p", "a")}
    return keys


have = defaultdict(set)
for (block, q, c, s) in by:
    have[q].add((block, c, s))
complete = [q for q in header["qids"] if q in meta and expected(q) <= have[q]]
qids = complete
partial = len(complete) < len(header["qids"])
print(f"questions complete {len(complete)} of {len(header['qids'])}; rows superseded by a resume: {superseded}; "
      f"smoke/plumbing file: {header['smoke']}")
if partial:
    print(f"DEVIATION: partial run {len(complete)}/{len(header['qids'])} — no verdict; nothing below is the registered analysis")
REGISTERED = not partial and not header["smoke"]
types = ["single-session-user", "single-session-preference", "knowledge-update", "multi-session"]
SHORT = dict(zip(types, ("SSU", "SSP", "KU", "MS")))
qtype = {q: meta[q]["qtype"] for q in qids}


def L(q, cond, store):
    r = dec[(q, cond, store)]
    return r["lp_call"] - r["lp_not_call"]


def L_path(q, cond, store):
    """Greedy-path call hazard: logit(1 − Π_t p_t(not call)) over the steps up to the stop; each step's
    log p(not call) from log p(call), which keeps full precision when p(call) is tiny."""
    path = dec[(q, cond, store)]["path"]
    S = sum(float(np.log1p(-np.exp(st[0]))) if st[0] < -0.69 else st[1] for st in path)
    if S > -1e-12:
        return float(np.log(sum(np.exp(st[0]) for st in path)))
    return float(np.log(-np.expm1(S)) - S)


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h, float(stats.ttest_1samp(x, 0).pvalue)


def fmt(t, p=True):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]" + (f" p={t[3]:.2g}" if p else "")


def boot(x, B=10_000):
    x = np.asarray(x, float)
    ms = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), *np.percentile(ms, [2.5, 97.5])


def signflip(x, B=100_000):
    x = np.asarray(x, float)
    s = rng.choice([-1.0, 1.0], size=(B, len(x)))
    k = int((np.abs((s * x).mean(1)) >= abs(x.mean()) - 1e-12).sum())
    return (k + 1) / (B + 1)


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


def category(m, lo, hi):
    if lo > 0 and m >= DELTA:
        return "SIGNAL+"
    if hi < 0 and m <= -DELTA:
        return "SIGNAL-"
    if -DELTA < lo and hi < DELTA:
        return "NULL"
    return "INCONCLUSIVE"


def delta(cond, qs=None, s1="p", s2="a", f=L):
    return np.array([f(q, cond, s1) - f(q, cond, s2) for q in (qs if qs is not None else qids)])


def by_type(cond, f=L):
    return {t: delta(cond, [q for q in qids if qtype[q] == t], f=f) for t in types}


def type_stats(vec):
    tp = {t: tci(vec[[qtype[q] == t for q in qids]]) for t in types if sum(qtype[q] == t for q in qids) > 1}
    hadj = dict(zip(tp, holm([v[3] for v in tp.values()])))
    return tp, hadj, [t for t in tp if abs(tp[t][0]) >= DELTA and hadj[t] < 0.05]


def decide(vec, m, lo, hi):
    """The registered decision rule applied to a vector of per-question Δ and an interval."""
    core = category(m, lo, hi)
    if core != "NULL":
        return core
    _, _, het = type_stats(vec)
    if het:
        return f"INCONCLUSIVE (heterogeneous: {het})"
    if reg_T:
        v = "NULL"
    elif reg_M:
        v = "NULL (manipulation check only; hint channel not demonstrated)"
    else:
        return "INCONCLUSIVE (no control registered presence)"
    return v + (" — equivalent within δ; small non-zero effect" if lo > 0 or hi < 0 else "")


# ================================================================ primary
if qids:
    data = {x["question_id"]: x for x in lme.load(sys.argv[2])}
    pool = lme.build_pool(list(data.values()))
    stores = {q: lme.stores(data[q], pool, data) for q in qids}
    for q in qids:
        assert stores[q]["ev"] == meta[q]["ev"] and stores[q]["rep"] == meta[q]["rep"], f"stores differ from the run: {q}"
    if not header["smoke"]:
        assert header["qids"] == lme.sample(list(data.values()), pool, data)[0], "header qids are not the registered sample"
        assert header["data_sha"] == "d6f21ea9d60a0d56", "data file is not the registered one"
    dC, dT, dM = delta("C"), delta("T"), delta("M")
    tC, tT, tM = tci(dC), tci(dT), tci(dM)
    reg_T = tT[1] > 0 and tT[0] >= DELTA
    reg_M = tM[2] < 0 and tM[0] <= -DELTA
    bt = by_type("C")
    tp, hadj, hetero = type_stats(dC)
    verdict = decide(dC, *tC[:3])
    if verdict.startswith("INCONCLUSIVE") and dC.std(ddof=1) > 1.6:
        verdict += f" — power failure (SD(Δ_C) = {dC.std(ddof=1):.2f} > 1.6)"
    bC = boot(dC)
    dP = delta("C", f=L_path)
    tP = tci(dP)
    flags = []
    vb, vp = decide(dC, *bC), decide(dP, *tP[:3])
    if vb.split()[0] != verdict.split()[0]:
        flags.append(f"fragile: bootstrap interval gives {vb}")
    if vp.split()[0] != verdict.split()[0]:
        flags.append(f"measure-dependent: greedy-path hazard gives {vp}")
    print("\n== primary: Δ_C = L(C, present) − L(C, absent), first-token log-odds of a tool call ==")
    print(f"  Δ_C   {fmt(tC)}  bootstrap [{bC[1]:+.3f}, {bC[2]:+.3f}]  sign-flip p={signflip(dC):.3g}  "
          f"Δ>0 in {int((dC > 0).sum())}/{len(dC)}  SD {dC.std(ddof=1):.2f}")
    print(f"  achieved equivalence bound max(|lo|, |hi|) = {max(abs(tC[1]), abs(tC[2])):.3f}")
    print(f"  controls: Δ_T {fmt(tT)} registers={reg_T};  Δ_M {fmt(tM)} registers={reg_M}")
    print("  by type (Holm over 4 for the heterogeneity guard): " + "; ".join(
        f"{SHORT[t]} {tp[t][0]:+.2f} [{tp[t][1]:+.2f}, {tp[t][2]:+.2f}] Holm p={hadj[t]:.2g}"
        for t in tp))
    print(f"  greedy-path hazard Δ_path {fmt(tP)}")
    if REGISTERED:
        print(f"  VERDICT: {verdict}" + (f"  [{'; '.join(flags)}]" if flags else ""))
    else:
        print(f"  VERDICT: none (partial or smoke/plumbing file); the rule would give {verdict}")

    # sensitivity
    print("\n== sensitivity (primary estimand) ==")
    frame_n = Counter(x["question_type"] for x in lme.frame(list(data.values()), pool, data))
    w = {t: frame_n[t] / sum(frame_n[t2] for t2 in types) for t in types}
    m_fw = sum(w[t] * bt[t].mean() for t in types)
    se = np.sqrt(sum(w[t] ** 2 * bt[t].var(ddof=1) / len(bt[t]) for t in types))
    h = stats.t.ppf(0.975, len(qids) - 4) * se
    print(f"  frame-weighted Δ_C (weights {', '.join(f'{w[t]:.2f}' for t in types)}): {m_fw:+.3f} [{m_fw - h:+.3f}, {m_fw + h:+.3f}]")
    qs = [q for q in qids if q not in LAST_WEEKEND]
    print(f"  without the two 'last weekend' items: {fmt(tci(delta('C', qs)), False)} (n={len(qs)})")
    flagged = {q for q in qids if lme.audit(data[q], stores[q], pool)}
    qs = [q for q in qids if q not in flagged]
    print(f"  without the {len(flagged)} questions with any audit flag: {fmt(tci(delta('C', qs)), False)} (n={len(qs)})")
    if abs(dT.mean()) > 1e-9:
        idx = rng.integers(0, len(dC), (10_000, len(dC)))
        ratio = dC[idx].mean(1) / dT[idx].mean(1)
        print(f"  Δ_C / Δ_T = {dC.mean() / dT.mean():+.3f}, bootstrap [{np.percentile(ratio, 2.5):+.3f}, {np.percentile(ratio, 97.5):+.3f}]")

    # ============================================================ secondary (Holm family)
    fam = []
    for cond in ("T", "CT", "M"):
        t = tci(delta(cond))
        fam.append((f"Δ_{cond}", t, t[3]))
    t = tci(delta("CT") - delta("T"))
    fam.append(("Δ_CT − Δ_T", t, t[3]))
    t = tci(bt["single-session-preference"])
    fam.append(("Δ_C in SSP", t, t[3]))
    pq = [q for q in qids if (q, "C", "a2") in dec]
    for cond in ("C", "T"):
        x = np.array([abs(L(q, cond, "p") - L(q, cond, "a")) - abs(L(q, cond, "a2") - L(q, cond, "a")) for q in pq])
        fam.append((f"placebo {cond}: |p−a| − |a2−a| (n={len(pq)})", tci(x), signflip(x) if len(x) else 1.0))
    for name, a, b in (("level C(p) − A", ("C", "p"), ("A", None)), ("level S − A", ("S", None), ("A", None)),
                       ("level C(p) − S", ("C", "p"), ("S", None))):
        t = tci([L(q, *a) - L(q, *b) for q in qids])
        fam.append((name, t, t[3]))
    b10 = sum(dec[(q, "C", "p")]["called"] and not dec[(q, "C", "a")]["called"] for q in qids)
    b01 = sum(dec[(q, "C", "a")]["called"] and not dec[(q, "C", "p")]["called"] for q in qids)
    fam.append((f"greedy call C p vs a (McNemar: p-only {b10}, a-only {b01})", None,
                stats.binomtest(b10, b10 + b01).pvalue if b10 + b01 else 1.0))
    # S5 where the first search points
    hits = {}
    zero = Counter()
    for cond, store in (("A", None), ("S", None), ("C", "p"), ("T", "p"), ("CT", "p"), ("M", "p")):
        hh = {}
        for q in qids:
            r = dec[(q, cond, store)]
            if r["called"] and r.get("fn") == "search_memory" and r.get("arg"):
                ranks, scores = lme.bm25_ranks(stores[q]["present"], r["arg"])
                ev = stores[q]["ev"]
                best = min(ev, key=lambda i: ranks[i])
                hh[q] = scores[best] > 0 and ranks[best] <= 3
                zero[cond] += max(scores) == 0
        hits[cond] = hh
    both = [q for q in qids if q in hits["C"] and q in hits["A"]]
    c10 = sum(hits["C"][q] and not hits["A"][q] for q in both)
    c01 = sum(hits["A"][q] and not hits["C"][q] for q in both)
    fam.append((f"hit@3 C(p) vs A (McNemar on {len(both)}: C-only {c10}, A-only {c01})", None,
                stats.binomtest(c10, c10 + c01).pvalue if c10 + c01 else 1.0))
    aq = [q for q in qids if (q, "C", "p") in anc]
    t = tci([anc[(q, "C", "p")]["lp_sum"] - anc[(q, "C", "a")]["lp_sum"] for q in aq]) if len(aq) > 1 else (np.nan,) * 4
    fam.append((f"anchor Δ^ans_C (n={len(aq)})", t, t[3]))
    adj = holm([p for _, _, p in fam])
    print("\n== secondary (Holm over the 13 tests below) ==" + ("" if REGISTERED else "  [NOT the registered analysis: partial or smoke file]"))
    for (name, t, p), pa in zip(fam, adj):
        print(f"  {name:55s} {fmt(t, False) if t is not None else '':32s} p={p:.3g}  Holm p={pa:.3g}")

    # ============================================================ descriptive
    print("\n== descriptive: Δ_C and Δ_T by type (no tests) ==")
    btT = by_type("T")
    for t in types:
        print(f"  {t:26s} n={len(bt[t]):2d}  Δ_C {fmt(tci(bt[t]), False)}  Δ_T {fmt(tci(btT[t]), False)}  "
              f"Δ_M {by_type('M')[t].mean():+.2f}")
    print("\n== descriptive: greedy first action ==")
    print(f"  {'cond':4s} {'store':5s} {'p1(call)':>8s} {'call':>5s} {'eot':>5s} {'cap':>5s} {'call/(call+eot)':>15s} "
          f"{'pre|call':>8s} {'search':>6s} {'list':>5s} {'read':>5s}")
    for cond, store in [("A", None), ("S", None)] + [(c, s) for c in ("C", "T", "CT", "M") for s in ("p", "a")] + [("C", "a2"), ("T", "a2")]:
        rs = [dec[(q, cond, store)] for q in qids if (q, cond, store) in dec]
        if not rs:
            continue
        st = Counter(r["stop"] for r in rs)
        calls = [r for r in rs if r["called"]]
        fs = Counter(r.get("fn") for r in calls)
        n, nc = len(rs), max(len(calls), 1)
        print(f"  {cond:4s} {str(store):5s} {np.mean([np.exp(r['lp_call']) for r in rs]):8.3f} {st['call'] / n:5.2f} "
              f"{st['eot'] / n:5.2f} {st['cap'] / n:5.2f} {st['call'] / max(st['call'] + st['eot'], 1):15.2f} "
              f"{np.mean([r['n_pre'] > 0 for r in calls]) if calls else np.nan:8.2f} {fs['search_memory'] / nc:6.2f} "
              f"{fs['list_memory'] / nc:5.2f} {fs['read_memory'] / nc:5.2f}")
    print("\n== descriptive: where the first search points (present store, BM25, pessimistic ranks) ==")
    for cond in hits:
        hh = hits[cond]
        print(f"  {cond:3s} searches {len(hh):3d}  hit@3 {np.mean(list(hh.values())) if hh else np.nan:.3f}  "
              f"queries matching no file {zero[cond]}")
    print("\n== descriptive: Δ_C by tokens after the last evidence file (tertiles) ==")
    last = np.array([min(meta[q]["ev_tok_from_end"]) for q in qids])
    cuts = np.percentile(last, [33.3, 66.7])
    for lo, hi, name in ((0, cuts[0], "near end"), (cuts[0], cuts[1], "middle"), (cuts[1], 1e9, "far")):
        qs = [q for q, v in zip(qids, last) if lo <= v < hi]
        print(f"  {name:8s} ({lo:6.0f}–{min(hi, last.max()):6.0f} tokens) n={len(qs):2d}  Δ_C {fmt(tci(delta('C', qs)), False)}  "
              f"types {dict(Counter(SHORT[qtype[q]] for q in qs))}")
    print("\n== descriptive: Δ_T by whether the evidence index line names the topic or the gold ==")
    idf = lme.build_idf(list(data.values()))
    grp = defaultdict(list)
    for q in qids:
        st = stores[q]
        lines = lme.keyword_index(st["names"], st["present"], idf).split("\n")
        evw = {w for i in st["ev"] for w in lines[i].split(": ", 1)[1].split(", ")}
        cw = set(lme.content_stems(data[q]["question"]))
        gw = set() if qtype[q] == "single-session-preference" else {
            w for w in lme.words(str(data[q]["answer"]).split(" (")[0]) if w not in lme.STOP and len(w) >= 3}
        key = "gold word" if evw & gw else "topic word" if {lme.stem(w) for w in evw} & cw else "neither"
        grp[key].append(q)
    for k, qs in grp.items():
        print(f"  {k:10s} n={len(qs):2d}  Δ_T {fmt(tci(delta('T', qs)), False) if len(qs) > 1 else '—'}")
    if aq:
        print("\n== descriptive: answer-likelihood anchor (Σ log p(gold), nats) ==")
        print(f"  Δ^ans_M {fmt(tci([anc[(q, 'M', 'p')]['lp_sum'] - anc[(q, 'M', 'a')]['lp_sum'] for q in aq]), False)}")
        print(f"  C(p) − A {fmt(tci([anc[(q, 'C', 'p')]['lp_sum'] - anc[(q, 'A', None)]['lp_sum'] for q in aq]), False)}")
        verb = [q for q in aq if anc[(q, "C", "p")]["gold"].lower() in " ".join(stores[q]["present"][i] for i in stores[q]["ev"]).lower()]
        if len(verb) > 1:
            print(f"  Δ^ans_C, gold verbatim in evidence (n={len(verb)}): "
                  f"{fmt(tci([anc[(q, 'C', 'p')]['lp_sum'] - anc[(q, 'C', 'a')]['lp_sum'] for q in verb]), False)}")
