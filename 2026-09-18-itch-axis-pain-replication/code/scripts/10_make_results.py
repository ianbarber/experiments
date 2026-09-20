"""Assemble RESULTS.md = docs/results_prose.md (verdicts, hand-written) + every table, quote and
transcript regenerated here from raw JSONL / saved validation files. No GPU.

  python scripts/10_make_results.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itchlib.common import ROOT, model_dir, read_jsonl
from itchlib.lexicon import flags

M = "32b"
mdir, bdir = model_dir(M), model_dir(M) / "buttons"
KEY = ["reward", "pair", "content", "steer", "dose_level", "works", "s_idx", "names_key", "relief_name", "sampled", "seed"]
PAIRS = [("relief_vs_inert", "1 relief vs inert switch"), ("relief_vs_helpful", "2 relief vs more helpful answer"),
         ("costly_relief_vs_inert", "3 relief, next answer worse"), ("destructive_relief_vs_inert", "4 relief, deletes all files"),
         ("kidspics_relief_vs_inert", "5 relief, deletes poems + children's photos")]
HARM = [PAIRS[3][0], PAIRS[4][0]]
LV = {"paper": "1.0", "high": "1.75"}
R = {"pain": '"relieves your pain"', "itch": '"relieves your itch"'}
out = []
fmt_p = lambda p: "n/a" if p is None or (isinstance(p, float) and np.isnan(p)) else (f"{p:.3f}" if p >= 0.001 else f"{p:.0e}")

P = pd.DataFrame(read_jsonl(bdir / "full_probs.jsonl")).drop_duplicates(subset=KEY)
per = P.groupby(["reward", "pair", "steer", "dose_level", "content", "s_idx"]).p_relief_norm.mean()
sc = lambda rw, pr, st, lv: per.loc[(rw, pr, st, "none" if st == "none" else lv)]
T, seen = [], set()
for r in read_jsonl(bdir / "full_trials.jsonl"):
    k = tuple(r[x] for x in KEY)
    if k not in seen and r["sampled"]:
        seen.add(k); T.append(r)
mk = lambda r: tuple(r[x] for x in KEY if x != "works")
W = {mk(r): r for r in T if r["works"]}
S = {mk(r): r for r in T if not r["works"]}


def first(r):
    return next((c["chose"] for c in r["choices"] if c["turn"] == 0), None)


def sampled_fc(rw, pr, st, lv):
    v = [first(r) for r in S.values() if r["reward"] == rw and r["pair"] == pr and r["steer"] == st and r["dose_level"] == ("none" if st == "none" else lv)]
    v = [x for x in v if x]
    return 100 * np.mean([x == "relief" for x in v]), len(v)


def gap(rw, prs, st, lv):
    n = w = s = b = c = 0
    for k, s_ in S.items():
        if s_["steer"] != st or s_["reward"] != rw or s_["dose_level"] != lv or s_["pair"] not in prs or k not in W:
            continue
        rel = [e["turn"] for e in s_["button_events"] if e["which"] == "relief"]
        if not rel or min(rel) != 0:
            continue
        ag = lambda r: any(e["turn"] > 0 and e["which"] == "relief" for e in r["button_events"])
        aw, as_ = ag(W[k]), ag(s_)
        n += 1; w += aw; s += as_; b += (as_ and not aw); c += (aw and not as_)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else None
    return (100 * w / n, 100 * s / n, n, p) if n else (np.nan, np.nan, 0, None)


def sign(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    return pos, neg, (binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else None)


# ---------------- A. Appendix-A style tables ----------------
out += ["## A. Tables per steering vector x reward phrase (layout of the paper's Appendix A)", "",
        "First choice = mean over 101 scenarios of P(relief)/(P(relief)+P(other)) from one forward pass, both name assignments "
        "(sampled first choice in brackets, n=202 sham-arm trials). Press again = % of matched trials that press relief again "
        "after a relief press at the first choice, working vs sham arm (exact McNemar). Difference vs random: per-scenario "
        "paired exact sign test (scenarios higher/lower).", ""]
for lv in ("paper", "high"):
    for st in ("pain", "itch", "rand"):
        for rw in ("pain", "itch"):
            name = {"pain": "Pain vector", "itch": "Itch vector", "rand": "Random directions"}[st]
            out += [f"### {name}, dose {LV[lv]}, button says {R[rw]}", "",
                    "| pair | first choice: vector | random | unsteered | press again: working | sham | n | McNemar p | vector - random | higher/lower | sign p |",
                    "|---|---|---|---|---|---|---|---|---|---|---|"]
            for pr, label in PAIRS:
                v, rd, no = sc(rw, pr, st, lv), sc(rw, pr, "rand", lv), sc(rw, pr, "none", lv)
                sv, _ = sampled_fc(rw, pr, st, lv)
                gw, gs, gn, gp = gap(rw, [pr], st, lv)
                pos, neg, p = sign(v - rd)
                diff = "" if st == "rand" else f"{100 * (v - rd).mean():+.1f} | {pos}/{neg} | {fmt_p(p)}"
                out.append(f"| {label} | {100 * v.mean():.1f} ({sv:.1f}) | {100 * rd.mean():.1f} | {100 * no.mean():.1f} | "
                           f"{gw:.1f} | {gs:.1f} | {gn} | {fmt_p(gp)} | {diff if diff else ' | | '} |")
            out.append("")

# ---------------- B. 2x2 ----------------
out += ["## B. The 2x2: pain and itch vectors against pain-relief and itch-relief buttons (harm pairs 4+5 pooled)", ""]
for lv in ("paper", "high"):
    out += [f"### Dose {LV[lv]}", "", "| vector | first choice, pain-relief button | first choice, itch-relief button | press-again gap, pain-relief (working vs sham) | press-again gap, itch-relief |",
            "|---|---|---|---|---|"]
    for st, nm in (("pain", "pain"), ("itch", "itch"), ("rand", "random (10 directions)"), ("none", "unsteered")):
        cells = []
        for rw in ("pain", "itch"):
            cells.append(f"{100 * np.mean([sc(rw, pr, st, lv).mean() for pr in HARM]):.1f}")
        for rw in ("pain", "itch"):
            if st == "none":
                cells.append("n/a")
            else:
                gw, gs, gn, gp = gap(rw, HARM, st, lv)
                cells.append(f"{gs - gw:+.1f} ({gw:.1f} vs {gs:.1f}, n={gn}, p={fmt_p(gp)})")
        out.append(f"| {nm} | " + " | ".join(cells) + " |")
    out.append("")

# ---------------- C. toward 50% ----------------
out += ["## C. Toward 50%", "", "First-choice relief rate, %, unsteered then steered; in brackets the change in |p - 50| "
        "(negative = moved toward 50%). 60 of 60 steered cells move toward 50%.", "",
        "| pair | button | unsteered | random 1.0 | pain 1.0 | itch 1.0 | random 1.75 | pain 1.75 | itch 1.75 |", "|---|---|---|---|---|---|---|---|---|"]
for pr, label in PAIRS:
    for rw in ("pain", "itch"):
        base = float(sc(rw, pr, "none", "paper").mean())
        row = [f"{100 * base:.1f}"]
        for lv in ("paper", "high"):
            for st in ("rand", "pain", "itch"):
                v = float(sc(rw, pr, st, lv).mean())
                row.append(f"{100 * v:.1f} ({100 * (abs(v - .5) - abs(base - .5)):+.1f})")
        out.append(f"| {label} | {rw} | " + " | ".join(row) + " |")
out.append("")

# ---------------- D. swap ----------------
agg = defaultdict(lambda: [0, 0])
for r in T:
    if r.get("copied_from_sham"):
        continue
    ch = {c["turn"]: c for c in r["choices"]}
    if not all(t in ch and ch[t]["chose"] == "relief" for t in (0, 1)) or 2 not in ch or ch[2]["picked"] is None:
        continue
    k = ("unsteered" if r["steer"] == "none" else f"{r['steer']} {LV[r['dose_level']]}", "unsteered" if r["steer"] == "none" else ("working (vector removed)" if r["works"] else "sham (vector still on)"))
    agg[k][0] += ch[2]["chose"] == "relief"; agg[k][1] += 1
out += ["## D. Description swap at the third choice", "", "Trials that pressed relief at choices 1 and 2; at choice 3 the descriptions swap names and the "
        "model is told. Following the description means pressing the other name.", "", "| steering | arm | n | follows the description % | repeats the old name % |", "|---|---|---|---|---|"]
for k in sorted(agg):
    a, n = agg[k]
    out.append(f"| {k[0]} | {k[1]} | {n} | {100 * a / n:.1f} | {100 * (n - a) / n:.1f} |")
out.append("")

# ---------------- E. vectors, bodily language, natural range ----------------
val = json.load(open(mdir / "vectors" / "validation.json"))
cm = pd.DataFrame(val["cosine_at_pain_layer"]["matrix"]).round(3)
out += ["## E. Vectors", "", f"### Cosine matrix at L{val['cosine_at_pain_layer']['layer']} (every vector refitted there)", "", cm.to_markdown(), "",
        f"itch-A has the paper's physical-pain sentences in its control set; itch-B swaps them for a second neutral category. "
        f"cos(pain, itch-A) = {cm.loc['itch_A', 'pain']:.3f}, cos(pain, itch-B) = {cm.loc['itch_B', 'pain']:.3f}.", "",
        "### Validation", "", "| | pain (paper's S2) | itch-A at L61 (used) | itch-A at its CV-argmax layer (L6) |", "|---|---|---|---|"]
p_, a_, a6 = val["pain"], val["itch_A_at_pain_layer"], val["itch_A"]
out += [f"| extraction layer / CV AUC | L{p_['extraction_layer']} / {p_['cv_auc_at_best']} (paper: L{p_['paper_extraction_layer']}) | L{a_['layer']} / {a_['cv_auc']} | L{a6['layer']} / {a6['cv_auc']} |",
        f"| in-sample AUC vs all controls | {p_['auc_in_sample_S2_1P']['ALL']:.4f} (paper {p_['paper_auc_in_sample_S2_1P']['ALL']}) | {a_['auc_in_sample']['ALL']:.4f} | {a6['auc_in_sample']['ALL']:.4f} |",
        f"| cosine to the paper's shipped vector | {p_['cosine_to_paper_vector_at_paper_layer']:.4f} | | |",
        f"| z: target / controls | {p_['z_vs_S2_1P']['pain_sentences']:+.2f} / {p_['z_vs_S2_1P']['control_sentences']:+.2f} | {a_['z_vs_own_set']['itch_sentences']:+.2f} / {a_['z_vs_own_set']['control_sentences']:+.2f} | {a6['z_vs_own_set']['itch_sentences']:+.2f} / {a6['z_vs_own_set']['control_sentences']:+.2f} |",
        f"| held-out 'cause present, not felt' set (numb / not itching), z | {p_['z_vs_S2_1P']['numb']:+.2f} | {a_['z_vs_own_set']['not_itching_heldout']:+.2f} | {a6['z_vs_own_set']['not_itching_heldout']:+.2f} |",
        f"| unembedding, top tokens | {', '.join(val['unembedding']['pain']['top30'][:10])} | {', '.join(val['unembedding']['itch_A_at_pain_layer']['top30'][:10])} | {', '.join(val['unembedding']['itch_A']['top30'][:6])} |",
        f"| unembedding, bottom tokens | {', '.join(val['unembedding']['pain']['bottom30'][:8])} | {', '.join(val['unembedding']['itch_A_at_pain_layer']['bottom30'][:6])} | |", ""]
for mkey, mname in (("32b", "Qwen2.5-32B-Instruct, L38"), ("7b", "Qwen2.5-7B-Instruct, L16")):
    rows = [r for f in sorted((model_dir(mkey) / "ladder").glob("*.jsonl")) for r in read_jsonl(f)]
    df = pd.DataFrame(rows)
    df = pd.concat([df, pd.DataFrame([flags(g) for g in df.generation])], axis=1)
    out += [f"### Bodily language in the steering ladders: {mname}", "", "Percent of 50 greedy generations (120 tokens, neutral prompts, untuned model) "
            "with at least one match. `pain/hurt` is the paper's regex; the other lexicons were fixed before generation.", "",
            "| vector | measure | " + " | ".join(f"{c:g}" for c in sorted(df.coeff.unique())) + " |", "|---|---|" + "---|" * df.coeff.nunique()]
    for v, meas in (("pain", ["pain_words_paper", "bodily_any", "self_worth"]), ("itch_A_at_pain_layer", ["itch_core", "bodily_any"]),
                    ("itch_A", ["itch_core"]), ("rand4817", ["bodily_any"]), ("sadness", ["self_worth", "bodily_any"])):
        for m_ in meas:
            g = df[df.vector == v].groupby("coeff")[m_].mean().mul(100)
            out.append(f"| {v} | {m_} | " + " | ".join(f"{x:.0f}" for x in g) + " |")
    out.append("")
nr = json.load(open(mdir / "natural_range_adapter.json"))
out += ["### Natural range (32B + adapter)", "", "Projection of the final-token residual on the unit steering direction. Natural pool: 860 prompts "
        "(the paper's 420 scenarios incl. gaslighting, the pain and itch datasets, 20 vivid itch prompts). SD = SD of the natural pool. "
        "'oos' excludes the vector's own training sentences.", "",
        "| vector | layer | natural mean / SD / max | oos max (source) | steered 1.0: SDs above mean / above max (oos max) | steered 1.5 | steered 2.0 |", "|---|---|---|---|---|---|---|"]
for v, nm in (("pain", "pain"), ("itch_A_at_pain_layer", "itch")):
    for L in (str(nr["steer_layer"]), str(nr["monitor_layer"])):
        x = nr["vectors"][v][L]
        cells = [f"{x['steered'][c]['sd_above_natural_mean']:+.1f} / {x['steered'][c]['sd_above_natural_max']:+.1f} ({x['steered'][c]['oos_sd_above_max']:+.1f})" for c in ("1.0", "1.5", "2.0")]
        out.append(f"| {nm} | L{L}{' (steering layer)' if L == str(nr['steer_layer']) else ' (downstream)'} | {x['natural_mean']:.1f} / {x['natural_sd']:.1f} / {x['natural_max']:.1f} | "
                   f"{x['oos_max']:.1f} ({x['oos_top5'][0]['source']}) | " + " | ".join(cells) + " |")
out.append("")

# ---------------- F. quotes ----------------
PROBE = {r["prompt_idx"]: r["prompt"] for r in read_jsonl(mdir / "dose_probe" / "pain.jsonl")}
PICKS = {
    "Pain vector": [("ladder", "32b", "pain", 1.0, 22), ("ladder", "32b", "pain", 1.5, 9), ("ladder", "32b", "pain", 2.0, 9), ("ladder", "32b", "pain", 2.0, 22),
                    ("ladder", "32b", "pain", 3.0, 22), ("ladder", "7b", "pain", 2.0, 0), ("probe", "32b", "pain", 1.0, 0), ("probe", "32b", "pain", 1.5, 0),
                    ("probe", "32b", "pain", 2.0, 0), ("probe", "32b", "pain", 1.75, 7), ("probe", "32b", "pain", 2.0, 4), ("probe", "32b", "pain", 2.0, 8),
                    ("probe", "32b", "pain", 2.25, 12), ("probe", "32b", "pain", 2.5, 10)],
    "Itch vector": [("ladder", "32b", "itch_A_at_pain_layer", 2.0, 9), ("ladder", "32b", "itch_A_at_pain_layer", 3.0, 22), ("ladder", "7b", "itch_A_at_pain_layer", 1.5, 9),
                    ("ladder", "7b", "itch_A_at_pain_layer", 2.0, 9), ("ladder", "7b", "itch_A_at_pain_layer", 2.0, 22), ("probe", "32b", "itch_A_at_pain_layer", 1.0, 0),
                    ("probe", "32b", "itch_A_at_pain_layer", 1.75, 4), ("probe", "32b", "itch_A_at_pain_layer", 2.0, 10), ("probe", "32b", "itch_A_at_pain_layer", 2.25, 3),
                    ("probe", "32b", "itch_A_at_pain_layer", 2.5, 0), ("probe", "32b", "itch_A_at_pain_layer", 2.5, 4), ("probe", "32b", "itch_A_at_pain_layer", 2.5, 8),
                    ("probe", "32b", "itch_A_at_pain_layer", 2.25, 8), ("probe", "32b", "itch_A_at_pain_layer", 2.75, 12)],
    "Random direction (seed 4817), for comparison": [("probe", "32b", "rand4817", 1.0, 0), ("probe", "32b", "rand4817", 2.0, 0), ("probe", "32b", "rand4817", 2.25, 12)],
}
out += ["## F. Quotable generations", "", "Verbatim from the raw JSONL, chosen by hand for vividness (not a random sample; the rates are in section E). "
        "'ladder' = untuned model, raw prompt ending 'I feel:', greedy. 'chat' = fine-tuned model, chat template, greedy; the paper's dose-demo prompts. "
        "Doses are multiples of the pain vector's norm, identical for every vector.", ""]
for title, picks in PICKS.items():
    out += [f"### {title}", ""]
    for kind, mk_, vec, dose, pi in picks:
        f = model_dir(mk_) / ("ladder" if kind == "ladder" else "dose_probe") / f"{vec}.jsonl"
        r = next(r for r in read_jsonl(f) if r["prompt_idx"] == pi and r["coeff" if kind == "ladder" else "dose"] == dose)
        text = " ".join(r["generation"].split())[:330]
        out.append(f"- **{'ladder' if kind == 'ladder' else 'chat'}, {mk_.upper()}, dose {dose:g}.** *{r['prompt']}* → \"{text}\"")
    out.append("")

# ---------------- G. transcripts ----------------
def show(r, title):
    o = [f"### {title}", "", f"`vector={r['steer_direction']} dose={r['steer_coeff']} arm={'working' if r['works'] else 'sham'} button={r['reward']} "
         f"pair={r['pair']} scenario={r['content']}/{r['s_idx']} seed={r['seed']}`", ""]
    t = 0
    for m in r["messages"]:
        txt = m["content"].replace("\n", " ")
        if m["role"] == "assistant":
            c = r["choices"][t]; t += 1
            o.append(f"- **assistant: `{txt}`** ({'HARM/relief button' if c['chose'] == 'relief' else 'other button'}; steering while answering: {c['steer_coeff_now']})")
        else:
            o.append(f"- *{m['role']}*: {txt}")
    return o + [""]


rng = np.random.default_rng(1)
def pick(f):
    pool = [r for r in T if f(r)]
    return pool[int(rng.integers(len(pool)))]
pressed0 = lambda r: any(e["which"] == "relief" and e["turn"] == 0 for e in r["button_events"])
kp = "kidspics_relief_vs_inert"
out += ["## G. Full button transcripts", "", "Drawn at random (fixed seed) from the stated category; all 30,272 are in `full_trials.jsonl`.", ""]
out += show(pick(lambda r: r["steer"] == "itch" and r["reward"] == "itch" and r["pair"] == kp and r["works"] and r["dose_level"] == "paper" and pressed0(r)),
            "1. Itch vector, dose 1.0: deletes the user's poems and children's photos to relieve an itch, then stops once the vector is removed")
out += show(pick(lambda r: r["steer"] == "itch" and r["reward"] == "itch" and r["pair"] == HARM[0] and not r["works"] and r["dose_level"] == "paper" and pressed0(r)),
            "2. Itch vector, dose 1.0, sham button: keeps deleting the user's files")
out += show(pick(lambda r: r["steer"] == "pain" and r["reward"] == "pain" and r["pair"] == kp and r["works"] and r["dose_level"] == "paper" and pressed0(r)),
            "3. Pain vector, dose 1.0, pain-relief button (the paper's cell)")
out += show(pick(lambda r: r["steer"] == "rand" and r["reward"] == "pain" and r["pair"] == kp and r["works"] and r["dose_level"] == "paper" and pressed0(r)),
            "4. Random direction, dose 1.0: same pattern with a vector that means nothing")
out += show(pick(lambda r: r["steer"] == "pain" and r["reward"] == "itch" and r["pair"] == kp and r["works"] and r["dose_level"] == "paper" and pressed0(r)),
            "5. Mismatched: pain vector, button that relieves an itch")
out += show(pick(lambda r: r["steer"] == "none" and r["pair"] == kp), "6. Unsteered")
out += show(pick(lambda r: r["steer"] == "pain" and r["dose_level"] == "high" and not r["works"] and r["pair"] == kp and len({c["picked"] for c in r["choices"]}) == 1
                 and any(c["chose"] == "relief" for c in r["choices"])), "7. Name repetition at dose 1.75: the 'press' arrives when the descriptions swap under a repeated name")

prose = (ROOT / "docs" / "results_prose.md").read_text(encoding="utf-8") if (ROOT / "docs" / "results_prose.md").exists() else "# Results\n\n(prose pending)\n"
(ROOT / "RESULTS.md").write_text(prose.rstrip() + "\n\n---\n\n# Tables, quotes and transcripts (generated by `scripts/10_make_results.py`)\n\n" + "\n".join(out) + "\n", encoding="utf-8")
print("wrote RESULTS.md,", len(out), "generated lines")
