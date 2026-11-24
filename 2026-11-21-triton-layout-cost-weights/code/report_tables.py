#!/usr/bin/env python
"""Compact tables for REPORT.md, one per experiment, from the result summaries.

    python report_tables.py --results ../results > ../results/report_tables.md
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ORDER = ("tutorials", "inductor", "triton_kernels", "triton_kernels_fpsan")
NAME = {"tutorials": "Tutorials", "inductor": "Inductor", "triton_kernels": "triton_kernels",
        "triton_kernels_fpsan": "triton_kernels under fpsan"}
ARCHS = (80, 90, 100, 120)


def pct(n, d, nd=0):
    return f"{100 * n / d:.{nd}f}%" if d else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    res = Path(args.results)
    s = json.loads((res / "e1_e2_summary.json").read_text())

    def get(key, section, arch):
        for x in s[key]:
            if x["section"] == section and x["arch"] == arch:
                return x
        return None

    # ------------------------------------------------------------ corpus
    print("### Corpus: kernel configurations per target\n")
    print("| Section | sm_80 | sm_90 | sm_100 | sm_120 | Kernel functions (sm_120) |")
    print("|---|---|---|---|---|---|")
    funcs = defaultdict(set)
    for row in s["corpus"]:
        if row["sm120"]:
            funcs[row["section"]].add(row["kernel"])
    for sec in ORDER:
        cells = [get("e1_conflicts", sec, a) for a in ARCHS]
        if not any(cells):
            continue
        print(f"| {NAME[sec]} | " + " | ".join(str(c["N"]) if c else "0" for c in cells) + f" | {len(funcs[sec])} |")
    tot = [sum(c["N"] for c in s["e1_conflicts"] if c["arch"] == a and c["section"] != "triton_kernels_fpsan") for a in ARCHS]
    print("| **Total without fpsan** | " + " | ".join(f"**{t}**" for t in tot) + " | |")

    # ------------------------------------------------------------ E1 conflicts
    rules = defaultdict(Counter)
    kinds = defaultdict(Counter)
    for d in s["e1_conflict_detail"]:
        rules[(d["section"], d["arch"])][d["rule"]] += d["values"]
        kinds[(d["section"], d["arch"])][" vs ".join(sorted(set(d["candidates"])))] += d["values"]
    print("\n### E1a: configurations that reach `resolveConflicts` with a choice\n")
    print("| Section | Target | Configurations | With a conflict | Conflict values | In pipeline invocation 1 / 2 / 3 / 4 | Settled by: first candidate / MMA rule / blocked-for-memory-op |")
    print("|---|---|---|---|---|---|---|")
    for sec in ORDER:
        for a in ARCHS:
            c = get("e1_conflicts", sec, a)
            if not c:
                continue
            bi = c["by_invocation"]
            inv = " / ".join(str(bi.get(i, 0)) for i in ("1", "2", "3", "4"))
            extra = []
            if bi.get("nested"):
                extra.append(f'{bi["nested"]} nested')
            if bi.get("fpsan"):
                extra.append(f'{bi["fpsan"]} in the fpsan run')
            r = rules[(sec, a)]
            print(f"| {NAME[sec]} | sm_{a} | {c['N']} | {c['K']} ({pct(c['K'], c['N'])}) | {c['conflict_values']} | "
                  f"{inv}{' (+' + ', '.join(extra) + ')' if extra else ''} | "
                  f"{r.get('first_candidate', 0)} / {r.get('mma', 0)} / {r.get('blocked_for_memory_op', 0)} |")
    for a in ARCHS:
        n = sum(c["N"] for c in s["e1_conflicts"] if c["arch"] == a and c["section"] != "triton_kernels_fpsan")
        k = sum(c["K"] for c in s["e1_conflicts"] if c["arch"] == a and c["section"] != "triton_kernels_fpsan")
        print(f"| **Total without fpsan** | sm_{a} | **{n}** | **{k} ({pct(k, n)})** | | | |")

    print("\nCandidate sets on sm_120, by the kinds of layout in them (conflict values): " + "; ".join(
        f"{k}: {v}" for k, v in sum((kinds[(sec, 120)] for sec in ORDER[:3]), Counter()).most_common(6)) + ".")

    # ------------------------------------------------------------ E1 survivors
    print("\n### E1b: `convert_layout` ops in the final TTGIR\n")
    print("| Section | Target | Conversions | Shared memory (warp / CTA / cluster barrier) | Warp shuffle | Register permute | No-op | Inside a loop | Configurations with none |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sec in ORDER:
        for a in ARCHS:
            c = get("e1_final_census", sec, a)
            if not c:
                continue
            cl, b, m = c["classes"], c["barriers"], c["M"]
            sm = cl.get("shared_memory", 0)
            print(f"| {NAME[sec]} | sm_{a} | {m} | {sm} ({pct(sm, m)}; {b.get('warp', 0)} / {b.get('cta', 0)} / {b.get('cluster', 0)}) | "
                  f"{cl.get('warp_shuffle', 0)} ({pct(cl.get('warp_shuffle', 0), m)}) | {cl.get('register_permute', 0)} "
                  f"({pct(cl.get('register_permute', 0), m)}) | {cl.get('noop', 0)} ({pct(cl.get('noop', 0), m)}) | "
                  f"{c['in_loop']} ({pct(c['in_loop'], m)}) | {c['configs_without']} of {c['configs']} |")
    for a in ARCHS:
        cs = [c for c in s["e1_final_census"] if c["arch"] == a and c["section"] != "triton_kernels_fpsan"]
        m = sum(c["M"] for c in cs)
        cl = Counter()
        for c in cs:
            cl.update(c["classes"])
        free = cl["noop"] + cl["register_permute"]
        print(f"| **Total without fpsan** | sm_{a} | **{m}** | **{cl['shared_memory']} ({pct(cl['shared_memory'], m)})** | "
              f"**{cl['warp_shuffle']} ({pct(cl['warp_shuffle'], m)})** | {cl['register_permute']} | {cl['noop']} "
              f"(free: **{pct(free, m)}**) | {sum(c['in_loop'] for c in cs)} | {sum(c['configs_without'] for c in cs)} of "
              f"{sum(c['configs'] for c in cs)} |")

    # ------------------------------------------------------------ E2
    print("\n### E2a: what `getConvertCost` charged against how the conversion would lower\n")
    print("| Section | Target | Calls | Priced free | Priced as shared memory | of those, lower to: shared memory | warp shuffle | register permute or no-op | Mispriced |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sec in ORDER:
        for a in ARCHS:
            c = get("e2_pricing", sec, a)
            if not c:
                continue
            pc = c["priced_classes"]
            free = pc.get("noop", 0) + pc.get("register_permute", 0)
            wrong = c["priced_smem"] - pc.get("shared_memory", 0)
            print(f"| {NAME[sec]} | sm_{a} | {c['calls']} | {c['priced_free']} | {c['priced_smem']} | {pc.get('shared_memory', 0)} | "
                  f"{pc.get('warp_shuffle', 0)} | {free} | {pct(wrong, c['priced_smem'], 1)} |")
    for a in ARCHS:
        cs = [c for c in s["e2_pricing"] if c["arch"] == a and c["section"] != "triton_kernels_fpsan"]
        priced = sum(c["priced_smem"] for c in cs)
        wrong = sum(c["priced_smem"] - c["priced_classes"].get("shared_memory", 0) for c in cs)
        print(f"| **Total without fpsan** | sm_{a} | {sum(c['calls'] for c in cs)} | {sum(c['priced_free'] for c in cs)} | "
              f"**{priced}** | | **{wrong}** | | **{pct(wrong, priced, 1)}** |")

    names = s["reprice_grid"]
    shuffle_measured = next(n for n in names if n.startswith("warp shuffle only, at measured"))
    print("\n### E2b: `isRematBeneficial` decisions and what repricing does to them\n")
    print(f"| Section | Target | Cost decisions | Subject priced free | Subject priced, remat chosen | Subject priced, conversion kept | Flips: {shuffle_measured} | warp shuffle only, at 0.01x | every class at its measured ratio (lowest to highest measured) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sec in ORDER:
        for a in ARCHS:
            d = get("e2_decisions", sec, a)
            r = get("e2_reprice", sec, a)
            if not d:
                continue
            be = [x for x in s["e2_break_even"] if x["section"] == sec and x["arch"] == a]
            sub = d["subject_classes"]
            f = r["flips"]
            lo, hi = sorted((f["every class, lowest measured in bucket"], f["every class, highest measured in bucket"]))
            print(f"| {NAME[sec]} | sm_{a} | {d['R']} | {sub.get('noop', 0) + sub.get('register_permute', 0)} | "
                  f"{sum(x['n'] for x in be if x['remat'])} | {sum(x['n'] for x in be if not x['remat'])} | "
                  f"{f[shuffle_measured]} | {f['warp shuffle only, at 0.01x']} | {f['every class, measured ratio']} ({lo} to {hi}) |")
    for a in ARCHS:
        ds = [c for c in s["e2_decisions"] if c["arch"] == a and c["section"] != "triton_kernels_fpsan"]
        rs = [c for c in s["e2_reprice"] if c["arch"] == a and c["section"] != "triton_kernels_fpsan"]
        be = [x for x in s["e2_break_even"] if x["arch"] == a and x["section"] != "triton_kernels_fpsan"]
        R = sum(c["R"] for c in ds)
        fm = sum(c["flips"]["every class, measured ratio"] for c in rs)
        print(f"| **Total without fpsan** | sm_{a} | **{R}** | | {sum(x['n'] for x in be if x['remat'])} | "
              f"{sum(x['n'] for x in be if not x['remat'])} | **{sum(c['flips'][shuffle_measured] for c in rs)}** | "
              f"{sum(c['flips']['warp shuffle only, at 0.01x'] for c in rs)} | **{fm} ({pct(fm, R, 1)})** |")
    print("\n\"Subject priced\" counts decisions with no new conversion on the other side. Decisions that weigh new "
          "conversions (hoists) are in the totals and in the flip counts.")

    # ------------------------------------------------------------ E3
    e3 = json.loads((res / "e3_summary.json").read_text())
    ratios = json.loads((res / "e3_ratios.json").read_text())
    print("\n### E3: cost of one conversion on the RTX 5090, default route, tensors of 1 KB and over\n")
    print("| Class | Element width | Conversions measured | Measured, in elementwise adds of the same tensor: median (range) | Assumed by the pass | Measured / assumed |")
    print("|---|---|---|---|---|---|")
    for g in e3["by_class"]:
        a = "/".join(map(str, g["assumed"]))
        alpha = "n/a" if g["alpha_median"] is None else f"{g['alpha_median']:.2f}"
        print(f"| {g['class'].replace('_', ' ')} | {g['bits']} bits | {g['n']} | {g['adds_median']:.1f} ({g['adds_min']:.1f} to "
              f"{g['adds_max']:.1f}) | {a} | {alpha} |")
    so = ratios["shuffle_over_smem"]
    print(f"\nThe same conversion lowered both ways, {so['n']} conversions: warp shuffle costs {so['median']:.2f} times "
          f"the shared-memory route at the median (range {so['min']:.2f} to {so['max']:.2f}); "
          f"{so['median_without_f64']:.2f} over the {so['n_without_f64']} that are not fp64.")
    ov = ratios["overall"]
    print(f"Measured over assumed price, all {ov['n']} conversions with a usable add unit: median {ov['median']:.2f}, "
          f"range {ov['min']:.3f} to {ov['max']:.2f}.")

    # ------------------------------------------------------------ E4
    p4 = res / "e4_summary.json"
    if p4.exists():
        e4 = json.loads(p4.read_text())
        print("\n### E4: alternative resolutions against the default\n")
        print("| Section | Target | Kernels with a conflict studied | Alternatives tried | Same program as the default | Different from the default | Distinct programs among those | of which the default's operations in another order | Do not compile | Distinct programs: more / same / fewer non-free conversions | Timed | Faster (3%, intervals apart, both runs) | Slower |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        cen = {(c["section"], c["target"]): c for c in e4["census"]}
        st_by = defaultdict(list)
        for x in e4["studies"]:
            st_by[(x["section"], x["target"])].append(x)
        for row in sorted(e4["table"], key=lambda r: (ORDER.index(r["section"]) if r["section"] in ORDER else 9, r["target"])):
            key = (row["section"], row["target"])
            c = cen.get(key, {})
            ss = st_by[key]
            timed = sum(len(x["timed"]) for x in ss)
            faster = sum(1 for x in ss for v in x["timed"] if v["faster"])
            slower = sum(1 for x in ss for v in x["timed"] if v["slower"])
            launchable = any(x["default_ms"] for x in ss) or row["target"] == 120
            print(f"| {NAME.get(row['section'], row['section'])} | sm_{row['target']} | {row['kernels']} | {row['tried']} | "
                  f"{row['same_as_default']} | {row['differ_from_default']} | {row['distinct_code']} | {row['reordered_only']} | "
                  f"{row['compile_error']} | "
                  f"{c.get('more', 0)} / {c.get('same', 0)} / {c.get('fewer', 0)} | "
                  f"{timed if launchable else 'not runnable here'} | {faster if launchable else ''} | {slower if launchable else ''} |")
        h = e4["headline"]
        if h:
            print(f"\nOn the device: {h.get('alternatives_timed', 0)} alternatives timed on {h.get('kernels_timed', 0)} "
                  f"kernels; {h.get('reportably_faster', 0)} faster and {h.get('reportably_slower', 0)} slower by the "
                  f"reporting rule. Over all {h.get('device_kernels_with_conflict_studied', 0)} conflicted kernels studied, "
                  f"the reportable gap to the best alternative has median {h.get('gap_median_pct', 0):.1f}% and maximum "
                  f"{h.get('gap_max_pct', 0):.1f}%; counting only alternatives that change an operation or a layout, "
                  f"median {h.get('gap_median_pct_different_operations', 0):.1f}% and maximum "
                  f"{h.get('gap_max_pct_different_operations', 0):.1f}%.")


if __name__ == "__main__":
    main()
