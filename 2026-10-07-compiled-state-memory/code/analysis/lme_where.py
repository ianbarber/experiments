"""Stage 0 analysis (exploratory, dev; LABNOTES 2026-10-10): where does the memory's effect live?

uv run --with numpy --with scipy --with torch python code/analysis/lme_where.py results/phase2/lme_where_q38-27b.jsonl

L = first-token log-odds of <tool_call>; H = greedy-path call hazard (as in analysis/lme_hint.py).
1. Agreement with M over every session (store versions as run): slope of X - A on M - A (through 0,
   question-bootstrap CI), mean |X - M|, identical first action (stop, function, argument) with M.
2. Presence (present - absent) per condition, with the absent2 swap floor for C, K, M.
3. Anchor: gold log-likelihood, present - absent and level vs A.
4. Direct logit attribution of z_call - E_q[z] at the first assistant token, by component type and depth;
   differences between conditions under one readout per question (A's). The pre-planned reading;
   retracted for presence contrasts after the run (its MLP and total columns depend on the readout).
4b. Post hoc (added after the data were in): exact midpoint split of own-readout differences. Only
   its GDN and attention columns are robust to the readout choice (see lme_where_posthoc.py, P4).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

rng = np.random.default_rng(42)
raw = [json.loads(line) for line in open(sys.argv[1])]
header, raw = raw[0], raw[1:]
rows = {}
for r in raw:  # last row per key wins (a resumed question rewrites its rows)
    rows[(r["block"], r["qid"], r.get("cond"), r.get("store"))] = r
done = [q for q in header["qids"] if ("done", q, None, None) in rows]
print(f"questions done {len(done)} of {len(header['qids'])}; plumbing file: {header['plumbing']}")
dec = {(q, c, s): r for (b, q, c, s), r in rows.items() if b == "decide" and q in done}
anc = {(q, c, s): r for (b, q, c, s), r in rows.items() if b == "anchor" and q in done}
qtype = {q: rows[("meta", q, None, None)]["qtype"] for q in done}
meta = {q: rows[("meta", q, None, None)] for q in done}
KV = ["C", "K", "CK", "Kstar4k", "Kstar64k", "Krt", "Kearly", "Kmid", "Klate"]


def L(r):
    return r["lp_call"] - r["lp_not_call"]


def H(r):
    S = sum(float(np.log1p(-np.exp(st[0]))) if st[0] < -0.69 else st[1] for st in r["path"])
    return float(np.log(sum(np.exp(st[0]) for st in r["path"]))) if S > -1e-12 else float(np.log(-np.expm1(S)) - S)


def act(r):
    return (r["stop"], r.get("fn"), r.get("arg"))


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return f"{x.mean() if len(x) else float('nan'):+.3f} (n={len(x)})"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+.3f} [{x.mean() - h:+.3f}, {x.mean() + h:+.3f}]"


def slope(pairs_by_q, B=5000):
    """Slope through 0 of y on x over all pairs; percentile CI by resampling questions."""
    qs = [q for q in pairs_by_q if pairs_by_q[q]]

    def est(sel):
        xy = [p for q in sel for p in pairs_by_q[q]]
        x, y = np.array([p[0] for p in xy]), np.array([p[1] for p in xy])
        return float((x * y).sum() / (x * x).sum())
    b = [est(rng.choice(qs, len(qs))) for _ in range(B)]
    return est(qs), *np.percentile(b, [2.5, 97.5])


# ---------------------------------------------------------------- 1. agreement with M
def agreement(X, stores):
    pl, ph, absd, same = defaultdict(list), defaultdict(list), [], []
    for (q, c, s), r in dec.items():
        if c != X or s not in stores or (q, "M", s) not in dec:
            continue
        m, a = dec[(q, "M", s)], dec[(q, "A", None)]
        pl[q].append((L(m) - L(a), L(r) - L(a)))
        ph[q].append((H(m) - H(a), H(r) - H(a)))
        absd.append(abs(L(r) - L(m)))
        same.append(act(r) == act(m))
    return pl, ph, absd, same


print("\n== 1. agreement with M: X - A regressed on M - A (through 0; question-bootstrap 95% CI) ==")
print("  every condition on the same sessions (present and absent stores); C, K pooled with absent2 below")
print(f"  {'cond':12s} {'n':>3s}  {'slope L':22s} {'slope H':22s} {'mean|L-M|':>9s} {'same action as M':>17s}")
absd = [abs(L(dec[(q, "A", None)]) - L(m)) for (q, c, s), m in dec.items() if c == "M" and s in ("p", "a")]
same = [act(dec[(q, "A", None)]) == act(m) for (q, c, s), m in dec.items() if c == "M" and s in ("p", "a")]
print(f"  {'A':12s} {len(absd):3d}  {'(0 by definition)':22s} {'':22s} {np.mean(absd):9.3f} {np.mean(same):17.2f}")
for X, stores in [(x, ("p", "a")) for x in KV] + [("C", ("p", "a", "a2")), ("K", ("p", "a", "a2"))]:
    pl, ph, absd, same = agreement(X, stores)
    sl, sh = slope(pl), slope(ph)
    lab = X + (" +a2" if "a2" in stores else "")
    print(f"  {lab:12s} {len(absd):3d}  {sl[0]:+.2f} [{sl[1]:+.2f}, {sl[2]:+.2f}]{'':5s} "
          f"{sh[0]:+.2f} [{sh[1]:+.2f}, {sh[2]:+.2f}]{'':5s} {np.mean(absd):9.3f} {np.mean(same):17.2f}")

# ---------------------------------------------------------------- 2. presence
print("\n== 2. presence: present - absent (n = questions) ==")
print(f"  {'cond':9s} {'L':28s} {'H (path hazard)':28s} {'called p/a':>10s} {'eot p/a':>8s} {'read p/a':>9s} {'same act':>8s}")
for X in ["M"] + KV:
    qs = [q for q in done if (q, X, "p") in dec and (q, X, "a") in dec]
    p, a = [dec[(q, X, "p")] for q in qs], [dec[(q, X, "a")] for q in qs]
    print(f"  {X:9s} {tci([L(i) - L(j) for i, j in zip(p, a)]):28s} {tci([H(i) - H(j) for i, j in zip(p, a)]):28s} "
          f"{sum(r['called'] for r in p):4d}/{sum(r['called'] for r in a):<5d} "
          f"{sum(r['stop'] == 'eot' for r in p):3d}/{sum(r['stop'] == 'eot' for r in a):<4d} "
          f"{sum(r.get('fn') == 'read_memory' for r in p):4d}/{sum(r.get('fn') == 'read_memory' for r in a):<4d} "
          f"{sum(act(i) == act(j) for i, j in zip(p, a)):8d}")
print("  swap floor (questions with absent2): |p - a| - |a2 - a| on L")
for X in ("C", "K", "M"):
    qs = [q for q in done if (q, X, "a2") in dec]
    d = [abs(L(dec[(q, X, "p")]) - L(dec[(q, X, "a")])) - abs(L(dec[(q, X, "a2")]) - L(dec[(q, X, "a")])) for q in qs]
    print(f"    {X:3s} n={len(qs):2d} {tci(d)}")
print("  by type, L present - absent:")
for t in sorted(set(qtype.values())):
    cells = []
    for X in ("C", "K", "CK", "M"):
        qs = [q for q in done if qtype[q] == t and (q, X, "p") in dec]
        cells.append(f"{X} {np.mean([L(dec[(q, X, 'p')]) - L(dec[(q, X, 'a')]) for q in qs]):+.2f}")
    print(f"    {t:26s} " + "  ".join(cells))

# ---------------------------------------------------------------- 3. anchor
print("\n== 3. answer anchor: sum log p(gold), nats ==")
qa = sorted({q for (q, c, s) in anc})
for X in ["C", "K", "CK", "M", "Kstar4k", "Kearly", "Kmid", "Klate"]:
    qs = [q for q in qa if (q, X, "p") in anc and (q, X, "a") in anc]
    if not qs:
        continue
    d = [anc[(q, X, "p")]["lp_sum"] - anc[(q, X, "a")]["lp_sum"] for q in qs]
    lv = [anc[(q, X, "p")]["lp_sum"] - anc[(q, "A", None)]["lp_sum"] for q in qs]
    print(f"  {X:9s} n={len(qs):2d}  present - absent {tci(d):28s} present - A {tci(lv)}")

# ---------------------------------------------------------------- 4. attribution
print("\n== 4. direct logit attribution of z_call - E_q[z] at the first assistant token ==")
chk = np.array([r["dla_check"] for r in dec.values()])
print(f"  check (own readout): |sum - x.w| max {np.abs(chk[:, 1] - chk[:, 2]).max():.3f}; |x.w - target| median "
      f"{np.median(np.abs(chk[:, 2] - chk[:, 0])):.3f} max {np.abs(chk[:, 2] - chk[:, 0]).max():.3f}")
tdir = sys.argv[1] + ".d"
tr = {q: torch.load(os.path.join(tdir, f"{q}.pt")) for q in done if os.path.exists(os.path.join(tdir, f"{q}.pt"))}
print(f"  traces for {len(tr)} of {len(done)} questions. Differences between conditions use one readout per question,")
print("  A's direction and norm scale (w_A), so they show what the components write, not readout drift.")
NL = (len(next(iter(dec.values()))["dla0"]) - 1) // 2
attn = header["bands"]["Kearly"] + header["bands"]["Kmid"] + header["bands"]["Klate"]
gdn = [i for i in range(NL) if i not in attn]


def comp(q, c, s):
    """Per-component contributions [1 + 2 NL] at the last prompt position under the common readout w_A."""
    t = tr[q]
    return (t[(c, s)]["C0"].double() @ t[("A", None)]["w0"].double()).numpy()


def parts(v):
    mix, mlp = v[1:1 + NL], v[1 + NL:]
    return dict(emb=v[0], gdn=mix[gdn].sum(), attn=mix[attn].sum(), mlp=mlp.sum(), total=v.sum())


def show(label, vecs):
    P = [parts(v) for v in vecs]
    cells = []
    for k in ("emb", "gdn", "attn", "mlp", "total"):
        x = np.array([p[k] for p in P])
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        cells.append(f"{k} {x.mean():+7.2f}±{h:4.2f}")
    print(f"  {label:26s} " + "  ".join(cells))


Q = [q for q in done if q in tr]
show("A", [comp(q, "A", None) for q in Q])
for X in ("C", "K", "M"):
    show(f"{X} present", [comp(q, X, "p") for q in Q])
for X in ("C", "K", "CK", "M", "Kstar4k", "Kearly", "Kmid", "Klate"):
    show(f"{X} present - A", [comp(q, X, "p") - comp(q, "A", None) for q in Q])
for X in ("C", "K", "CK", "M"):
    show(f"{X} present - absent", [comp(q, X, "p") - comp(q, X, "a") for q in Q])
drift = [abs(tr[q][(X, "p")]["C0"].double().sum(0) @ (tr[q][(X, "p")]["w0"].double() - tr[q][("A", None)]["w0"].double())).item()
         for q in Q for X in ("C", "K", "M")]
print(f"  readout drift |x_X . (w_X - w_A)| over C, K, M present: median {np.median(drift):.2f} max {np.max(drift):.2f}")
print("  by depth (common readout; mean over questions; block = 3 GDN + 1 attention layer and their 4 MLPs):")
print(f"    {'layers':>6s} {'A':>22s} {'M p - A':>22s} {'M p - a':>22s} {'K p - A':>22s} {'K p - a':>22s}")
for blk in range(NL // 4):
    idx = list(range(4 * blk, 4 * blk + 4))
    cells = []
    for vecs in ([comp(q, "A", None) for q in Q],
                 [comp(q, "M", "p") - comp(q, "A", None) for q in Q],
                 [comp(q, "M", "p") - comp(q, "M", "a") for q in Q],
                 [comp(q, "K", "p") - comp(q, "A", None) for q in Q],
                 [comp(q, "K", "p") - comp(q, "K", "a") for q in Q]):
        V = np.asarray(vecs)
        g = V[:, 1 + np.array([i for i in idx if i in gdn])].sum(1).mean()
        a = V[:, 1 + np.array([i for i in idx if i in attn])].sum(1).mean()
        m = V[:, 1 + NL + np.array(idx)].sum(1).mean()
        cells.append(f"g{g:+5.1f} a{a:+5.1f} m{m:+5.1f}")
    print(f"    {4 * blk:2d}-{4 * blk + 3:2d} " + " ".join(f"{c:>22s}" for c in cells))

# ---------------------------------------------------------------- 4b. exact midpoint split
print("\n== 4b. exact split of own-readout differences: x_X.w_X - x_Y.w_Y = sum_i (C_X,i - C_Y,i).w_mid "
      "+ sum_i C_mid,i.(w_X - w_Y) ==")
print("  'written' = what the components write differently, read through the midpoint readout; 'readout' = the change in")
print("  the readout itself (final-norm scale and the non-call distribution q). The two sum exactly to the own-readout change.")


def split(q, X, Y):
    a, b = tr[q][X], tr[q][Y]
    Ca, Cb, wa, wb = a["C0"].double(), b["C0"].double(), a["w0"].double(), b["w0"].double()
    return (((Ca - Cb) @ ((wa + wb) / 2)).numpy(), (((Ca + Cb) / 2) @ (wa - wb)).sum().item(),
            (Ca.sum(0) @ wa - Cb.sum(0) @ wb).item())


def show_split(label, X, Y):
    S = [split(q, X, Y) for q in Q]
    P = [parts(s[0]) for s in S]
    cells = []
    for k, x in [(k, np.array([p[k] for p in P])) for k in ("gdn", "attn", "mlp", "total")] + [
            ("readout", np.array([s[1] for s in S])), ("own", np.array([s[2] for s in S]))]:
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        cells.append(f"{k} {x.mean():+6.2f}±{h:4.2f}")
    print(f"  {label:20s} " + "  ".join(cells))
    return S


cont = {}
for X in ("C", "K", "CK", "M", "Kstar4k", "Kmid"):
    cont[X, "p-A"] = show_split(f"{X} present - A", (X, "p"), ("A", None))
for X in ("C", "K", "CK", "M", "Kstar4k", "Kmid"):
    cont[X, "p-a"] = show_split(f"{X} present - absent", (X, "p"), (X, "a"))
print("  by depth, 'written' part (mean over questions; block = 3 GDN + 1 attention layer and their 4 MLPs):")
cols = [("M", "p-A"), ("M", "p-a"), ("K", "p-A"), ("K", "p-a"), ("CK", "p-a")]
print(f"    {'layers':>6s} " + " ".join(f"{X + ' ' + c:>25s}" for X, c in cols))
for blk in range(NL // 4):
    idx = list(range(4 * blk, 4 * blk + 4))
    cells = []
    for key in cols:
        V = np.asarray([s[0] for s in cont[key]])
        g = V[:, 1 + np.array([i for i in idx if i in gdn])].sum(1).mean()
        a = V[:, 1 + np.array([i for i in idx if i in attn])].sum(1).mean()
        m = V[:, 1 + NL + np.array(idx)].sum(1).mean()
        cells.append(f"g{g:+6.2f} a{a:+6.2f} m{m:+6.2f}")
    print(f"    {4 * blk:2d}-{4 * blk + 3:2d} " + " ".join(f"{c:>25s}" for c in cells))
