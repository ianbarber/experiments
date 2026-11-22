#!/usr/bin/env python
"""E4 tables: alternative conflict resolutions against the default.

    python analyze_e4.py --runs runs/e4 --out ../results

Reads every `<workload>_sm<target>/results.jsonl` under --runs. Writes e4_summary.json
and prints markdown tables.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

SECTION = {"tut06": "tutorials", "inductor": "inductor", "tk": "triton_kernels",
           "inductor_run2": "inductor_run2", "tk_run2": "triton_kernels_run2",
           "inductor_reprice": "inductor_reprice", "tk_reprice": "triton_kernels_reprice",
           "inductor_reprice_run2": "inductor_reprice_run2", "tk_run2_reprice": "triton_kernels_reprice_run2"}


DTYPE = {"float8_e5m2": "fp8e5", "float8_e4m3fn": "fp8e4", "mxfloat4_e2m1": "mxfp4", "mxfloat8_e4m3fn": "mxfp8",
         "nvfp4_e2m1": "nvfp4", "nvfp4_e2m1_fiber": "nvfp4f", "bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def short_name(r):
    """A row label a reader can place: Inductor kernels lose their model prefix, triton_kernels
    studies get their operand types and M tile from the test that launched them."""
    label = r["label"]
    m = re.search(r"-(?:plain|batched|ragged)-([a-z0-9_]+)-([a-z0-9_]+)-", r.get("test") or "")
    n = re.search(r"_(r?\d+)$", label)
    if m and n and not label.startswith(("resnet50", "vit_b_16", "bert_base", "gpt2")):
        x, w = (DTYPE.get(t, t) for t in m.groups())
        tile = (r.get("constants") or {}).get("BLOCK_M") if isinstance(r.get("constants"), dict) else None
        return f'{r["kernel"]} {x}×{w}' + (f" M{tile}" if tile else "") + f" #{n.group(1).lstrip('r').lstrip('0')}"
    name = re.sub(r"^(resnet50|vit_b_16|bert_base|gpt2)__triton_", "", label)
    # Inductor names share long prefixes and differ at the end: keep both ends
    return name if len(name) <= 40 else name[:18] + "…" + name[-21:]


def n_conversions(census, where=None):
    """Number of non-free conversions in a final-TTGIR census, optionally only those in a loop."""
    n = 0
    for k, v in (census or {}).items():
        cls = k.split("@")[0].split(":")[0]
        if cls in ("noop", "register_permute"):
            continue
        if where == "loop" and "@loop" not in k:
            continue
        n += v
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    studies = []
    scanned = Counter()
    sessions = {}
    for d in sorted(root.glob("*_sm*")):
        f = d / "results.jsonl"
        if not f.exists():
            continue
        m = re.match(r"(.+)_sm(\d+)$", d.name)
        workload, target = m.group(1), int(m.group(2))
        if (d / "session.json").exists():
            sessions[d.name] = json.loads((d / "session.json").read_text())
        # Runs on the device do not log the final-TTGIR census while compiling; it is recomputed
        # from the saved TTGIR by census_from_ttgir.py.
        posthoc = json.loads((d / "ttgir_census.json").read_text()) if (d / "ttgir_census.json").exists() else {}
        canon = json.loads((d / "ttgir_canonical.json").read_text()) if (d / "ttgir_canonical.json").exists() else {}
        for line in open(f):
            r = json.loads(line)
            scanned[(SECTION.get(workload, workload), target, r.get("status"))] += 1
            if r.get("status") != "ok":
                continue
            variants = r["variants"]
            # The harness compared TTGIR as text. Two builds that differ only in the names of
            # their SSA values are the same program: fold those into what they duplicate.
            base_entry = canon.get(f'{r["label"]}__default.ttgir') or {}
            base = base_entry.get("program")
            first, alias, renamed_timed = {}, {}, []
            for name, v in variants.items():
                entry = canon.get(f'{r["label"]}__{name}.ttgir') or {}
                c = entry.get("program")
                if v["status"] == "identical_code" or c is None or base is None:
                    continue
                # same operations on the same layouts as the default, in a different order
                v["reordered_only"] = entry.get("operations") == base_entry.get("operations")
                if c == base:
                    alias[name] = "default"
                elif c in first:
                    alias[name] = first[c]
                else:
                    first[c] = name
            for name, target_name in alias.items():
                v = variants[name]
                if "vs_default_pct" in v and target_name == "default":
                    renamed_timed.append({"name": name, "vs_default_pct": v["vs_default_pct"],
                                          "faster": v["faster"], "slower": v["slower"]})
                v.pop("vs_default_pct", None)
                v.update(status="identical_code", same_as=target_name, renamed_only=True)
            for v in variants.values():
                if v["status"] == "identical_code":
                    v["same_as"] = alias.get(v.get("same_as"), v.get("same_as"))
            same_default = sum(1 for v in variants.values() if v["status"] == "identical_code" and v["same_as"] == "default")
            duplicates = sum(1 for v in variants.values() if v["status"] == "identical_code" and v["same_as"] != "default")
            key = f'{r["label"]}__default.ttgir'
            if key in posthoc and "error" not in posthoc[key]:
                r["default_census"] = posthoc[key]
                for name, v in variants.items():
                    c = posthoc.get(f'{r["label"]}__{name}.ttgir')
                    if c is not None and "error" not in c:
                        v["census"] = c
            states = Counter(v["status"] for v in variants.values())
            timed = [dict(name=k, vs_default_pct=v["vs_default_pct"], faster=v["faster"], slower=v["slower"],
                          ms=v["ms"], census=v.get("census"), flipped=v["flipped"], binary=v.get("binary"),
                          output_exact=v.get("output", {}).get("exact"), reordered_only=v.get("reordered_only"))
                     for k, v in variants.items() if "vs_default_pct" in v]
            distinct = [v for v in variants.values() if v["status"] == "distinct_code"]
            base = n_conversions(r["default_census"])
            base_loop = n_conversions(r["default_census"], "loop")
            delta = Counter()
            for v in distinct:
                n = n_conversions(v.get("census"))
                nl = n_conversions(v.get("census"), "loop")
                delta["more" if n > base else "fewer" if n < base else "same"] += 1
                delta["more_in_loop" if nl > base_loop else "fewer_in_loop" if nl < base_loop else "same_in_loop"] += 1
            studies.append({
                "section": SECTION.get(workload, workload), "target": target, "label": r["label"], "kernel": r["kernel"],
                "short": short_name(r),
                "conflicts": len(r["conflicts"]),
                "conflict_invocations": sorted({c["inv"] for c in r["conflicts"]}),
                "alternatives_total": r["alternatives_total"], "tried": r["alternatives_tried"], "sampled": r["sampled"],
                "states": dict(states), "default_census": r["default_census"], "census_delta": dict(delta),
                "same_as_default": same_default, "duplicates_of_alternatives": duplicates,
                "reordered_only": sum(1 for v in distinct if v.get("reordered_only")),
                "differ_from_default": states.get("distinct_code", 0) + duplicates,
                "renamed_only": len(alias), "renamed_timed": renamed_timed,
                "default_ms": r.get("default_ms"), "inner_repeats": r.get("inner_repeats"), "timed": timed,
                "timing_mode": r.get("timing_mode"), "wall_ms_per_launch": r.get("wall_ms_per_launch"),
                "control": r.get("control"), "default_binary": r.get("default_binary"),
                "output_check": r.get("output_check"), "output_args": r.get("output_args"),
                "constants": r.get("constants"), "options": r.get("options"),
                "default_repeatable": r.get("default_repeatable"),
                "output_checks": Counter("exact" if v["output"]["exact"] else "within_tolerance"
                                         for v in variants.values() if "output" in v and v["status"] == "distinct_code"),
                "test": r.get("test"),
            })

    # A second, independent run of a section replicates the first; a repriced build is a different experiment.
    second = {(x["section"][:-5], x["target"], x["label"]): x for x in studies if x["section"].endswith("_run2")}
    repriced = [x for x in studies if x["section"].endswith("_reprice")]
    studies = [x for x in studies if not x["section"].endswith(("_run2", "_reprice"))]
    replication = {"pairs": 0, "faster_first": 0, "faster_both": 0, "slower_first": 0, "slower_both": 0, "controls": []}
    unpaired = Counter()
    for x in studies + repriced:
        y = second.get((x["section"], x["target"], x["label"]))
        if y is None or y["kernel"] != x["kernel"]:
            # no second run for this kernel: nothing here can be reported as a difference
            if x["timed"] and any(k[0] == x["section"] for k in second):
                unpaired[x["section"]] += 1
                for v in x["timed"]:
                    v["first_run_faster"], v["first_run_slower"] = v["faster"], v["slower"]
                    v["faster"] = v["slower"] = False
            continue
        x["replicated"] = True
        if y.get("control"):
            replication["controls"].append(y["control"])
        other = {v["name"]: v for v in y["timed"]}
        for v in x["timed"]:
            w = other.get(v["name"])
            if w is None:
                continue
            replication["pairs"] += 1
            v["second_run_pct"] = w["vs_default_pct"]
            v["first_run_faster"], v["first_run_slower"] = v["faster"], v["slower"]
            replication["faster_first"] += v["faster"]
            replication["slower_first"] += v["slower"]
            # a difference is reported only if both runs report it, at the smaller of the two sizes
            v["faster"] = bool(v["faster"] and w["faster"])
            v["slower"] = bool(v["slower"] and w["slower"])
            v["first_run_pct"] = v["vs_default_pct"]
            if v["faster"]:
                v["vs_default_pct"] = max(v["first_run_pct"], w["vs_default_pct"])
            elif v["slower"]:
                v["vs_default_pct"] = min(v["first_run_pct"], w["vs_default_pct"])
            replication["faster_both"] += v["faster"]
            replication["slower_both"] += v["slower"]
    # A kernel whose null control moves by 3% or more in either run cannot be timed to that
    # resolution: nothing on it is reported.
    unreliable = []
    for x in studies + repriced:
        ctls = [c for c in (x.get("control"), (second.get((x["section"], x["target"], x["label"])) or {}).get("control")) if c]
        if x.get("control"):
            replication["controls"].append(x["control"])
        if any(abs(c["vs_default_pct"]) >= 3.0 for c in ctls):
            unreliable.append(x["label"])
            x["unreliable_timing"] = True
            for v in x["timed"]:
                v["faster"] = v["slower"] = False
    replication["unpaired_kernels"] = dict(unpaired)
    replication["unreliable_kernels"] = unreliable
    # tally per study, after the exclusions, from the first-run flags and the final ones
    for x in studies + repriced:
        if x.get("unreliable_timing"):
            continue
        for v in x["timed"]:
            if "second_run_pct" not in v:
                continue
            c = replication.setdefault("by_section", {}).setdefault(
                x["section"], {"pairs": 0, "faster_first": 0, "faster_both": 0, "slower_first": 0, "slower_both": 0})
            c["pairs"] += 1
            c["faster_first"] += bool(v["first_run_faster"])
            c["slower_first"] += bool(v["first_run_slower"])
            c["faster_both"] += bool(v["faster"])
            c["slower_both"] += bool(v["slower"])

    print("## E4: alternative conflict resolutions\n")
    print("| Section | Target | Kernels studied | Alternatives that exist (median per kernel) | Tried | Sampled kernels | Same program as the default | Different from the default | Distinct programs among those | of which the default's operations in another order | Fail to compile | Wrong output | Launch error | Output not checkable |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for s in studies:
        groups[(s["section"], s["target"])].append(s)
    table = []
    for (section, target), ss in sorted(groups.items()):
        tot = Counter()
        for s in ss:
            tot.update(s["states"])
        tried = sum(s["tried"] for s in ss)
        med_alt = st.median(s["alternatives_total"] for s in ss)
        row = {"section": section, "target": target, "kernels": len(ss), "median_alternatives": med_alt, "tried": tried,
               "sampled_kernels": sum(1 for s in ss if s["sampled"]),
               "same_as_default": sum(s["same_as_default"] for s in ss),
               "differ_from_default": sum(s["differ_from_default"] for s in ss),
               "renamed_only": sum(s["renamed_only"] for s in ss),
               "reordered_only": sum(s["reordered_only"] for s in ss),
               **{k: tot.get(k, 0) for k in
                  ("identical_code", "distinct_code", "compile_error", "wrong_output", "launch_error", "unverified_output")}}
        table.append(row)
        print(f"| {section} | sm_{target} | {len(ss)} | {med_alt:,.0f} | {tried} | {row['sampled_kernels']} | "
              f"{row['same_as_default']} | {row['differ_from_default']} | {row['distinct_code']} | {row['reordered_only']} | "
              f"{row['compile_error']} | {row['wrong_output']} | {row['launch_error']} | {row['unverified_output']} |")

    print("\nA build is the same program as another when their final TTGIR is equal up to debug locations and "
          "the names of SSA values. \"Different from the default\" counts alternatives; several of them usually "
          "produce the same program, and each distinct program is checked and timed once.")
    print("\n### Kernels where no alternative changed the code\n")
    print("| Section | Target | Kernels | of which: every alternative tried is the same program as the default |")
    print("|---|---|---|---|")
    inert = []
    for (section, target), ss in sorted(groups.items()):
        n = sum(1 for s in ss if s["same_as_default"] == s["tried"])
        inert.append({"section": section, "target": target, "kernels": len(ss), "all_identical": n})
        print(f"| {section} | sm_{target} | {len(ss)} | {n} |")

    print("\n### Different programs: conversions left in the final TTGIR against the default\n")
    print("| Section | Target | Distinct programs other than the default | fewer non-free conversions | same | more | fewer inside a loop | more inside a loop |")
    print("|---|---|---|---|---|---|---|---|")
    census_rows = []
    for (section, target), ss in sorted(groups.items()):
        d = Counter()
        for s in ss:
            d.update(s["census_delta"])
        n = sum(s["states"].get("distinct_code", 0) for s in ss)
        if not n:
            continue
        census_rows.append({"section": section, "target": target, "distinct": n, **dict(d)})
        print(f"| {section} | sm_{target} | {n} | {d['fewer']} | {d['same']} | {d['more']} | {d['fewer_in_loop']} | {d['more_in_loop']} |")

    # ---- timing (device target only)
    timed_studies = [s for s in studies if s["timed"]]
    print("\n### Timing on the device\n")
    headline = {}
    if timed_studies:
        print("| Section | Kernel | Conflicts | Different-code alternatives timed | Default, ms per launch [95% interval] | Best alternative vs default | Worst | Reportably faster | Reportably slower |")
        print("|---|---|---|---|---|---|---|---|---|")
        best_per = []
        for s in sorted(timed_studies, key=lambda s: (s["section"], s["label"])):
            vs = [v["vs_default_pct"] for v in s["timed"]]
            faster = sum(1 for v in s["timed"] if v["faster"])
            slower = sum(1 for v in s["timed"] if v["slower"])
            d = s["default_ms"]
            best_per.append((min(vs), s))
            print(f"| {s['section']} | `{s['short']}` | {s['conflicts']} | {len(vs)} | {d['median']:.4f} "
                  f"[{d['ci'][0]:.4f}, {d['ci'][1]:.4f}] | {min(vs):+.1f}% | {max(vs):+.1f}% | {faster} | {slower} |")
        best_all = sorted(b for b, _ in best_per)
        reportable = [(v["vs_default_pct"], s["label"], v["name"]) for s in timed_studies for v in s["timed"] if v["faster"]]
        headline = {
            "kernels_timed": len(timed_studies),
            "alternatives_timed": sum(len(s["timed"]) for s in timed_studies),
            "reportably_faster": len(reportable),
            "reportably_slower": sum(1 for s in timed_studies for v in s["timed"] if v["slower"]),
            "largest_reportable_gain_pct": -min((r[0] for r in reportable), default=0.0),
            "largest_reportable_gain_kernel": min(reportable)[1] if reportable else None,
            "largest_reportable_gain_section": next((s["section"] for s in timed_studies
                                                     if reportable and s["label"] == min(reportable)[1]), None),
            "best_alternative_median_pct": st.median(best_all),
            "best_alternative_min_pct": best_all[0],
            "kernels_with_reportable_gain": len({r[1] for r in reportable}),
        }
        # the extremes per section, for the headline's last sentence
        extremes = {}
        for s_ in timed_studies:
            for v in s_["timed"]:
                if not (v["faster"] or v["slower"]):
                    continue
                e = extremes.setdefault(s_["section"], {})
                key = "largest_gain" if v["faster"] else "largest_loss"
                if key not in e or abs(v["vs_default_pct"]) > abs(e[key]["pct"]):
                    e[key] = {"pct": v["vs_default_pct"], "label": s_["label"], "short": s_["short"],
                              "default_us": s_["default_ms"]["median"] * 1e3, "variant_us": v["ms"]["median"] * 1e3}
        headline["extremes"] = extremes
        print(f"\n{headline['alternatives_timed']} alternatives with different code were timed on "
              f"{headline['kernels_timed']} kernels. {headline['reportably_faster']} are faster than the default by 3% "
              f"or more with non-overlapping intervals, on {headline['kernels_with_reportable_gain']} kernels; "
              f"{headline['reportably_slower']} are slower by the same test. Best alternative per kernel: median "
              f"{headline['best_alternative_median_pct']:+.1f}%, most favourable {headline['best_alternative_min_pct']:+.1f}%.\n")
    else:
        print("No alternative produced different code on the device's own target, so nothing was timed.\n")

    # ---- all kernels with a conflict on the device target: gap to best alternative, zero when code is identical
    device_targets = {s["target"] for s in studies if s["default_ms"] or s["timed"]} or {120}
    gaps = []
    for s in studies:
        if s["target"] not in device_targets:
            continue
        reportable = [v["vs_default_pct"] for v in s["timed"] if v["faster"]]
        gaps.append(-min(reportable) if reportable else 0.0)
    if gaps:
        headline["device_kernels_with_conflict_studied"] = len(gaps)
        headline["gap_median_pct"] = st.median(gaps)
        headline["gap_max_pct"] = max(gaps)
        print(f"Over all {len(gaps)} conflicted kernels studied on the device (identical-code alternatives count as "
              f"no gap): reportable gap to the best alternative, median {st.median(gaps):.1f}%, maximum {max(gaps):.1f}%.\n")

    if replication["pairs"]:
        ctl = replication["controls"]
        dev = sorted(abs(c["vs_default_pct"]) for c in ctl)
        print("### Replication and null control\n")
        print("Studies on the device were run twice in separate processes. Only differences both runs report are "
              "counted, at the smaller of the two sizes.\n")
        print("| Study | Builds timed in both runs | Faster: first run | both runs | Slower: first run | both runs |")
        print("|---|---|---|---|---|---|")
        for sec, c in sorted(replication.get("by_section", {}).items()):
            print(f"| {sec} | {c['pairs']} | {c['faster_first']} | {c['faster_both']} | {c['slower_first']} | {c['slower_both']} |")
        print()
        if dev:
            flagged = sum(1 for c in ctl if c["faster"] or c["slower"])
            print(f"Null control (the default built a second time and timed as if it were an alternative), "
                  f"{len(dev)} kernels: median absolute difference {st.median(dev):.2f}%, largest {dev[-1]:.2f}%; "
                  f"{flagged} would have been reported as a difference.\n")
        renamed = [abs(v["vs_default_pct"]) for x in studies + repriced for v in x.get("renamed_timed", [])]
        if renamed:
            flagged_r = sum(1 for x in studies + repriced for v in x.get("renamed_timed", []) if v["faster"] or v["slower"])
            print(f"A second null control: {len(renamed)} builds that the harness timed as different code are the "
                  f"default with its values renamed. Largest difference from the default {max(renamed):.2f}%; "
                  f"{flagged_r} would have been reported.\n")
            replication["renamed_default_timed"] = {"n": len(renamed), "abs_max": max(renamed), "flagged": flagged_r}
        replication["control_abs_median"] = st.median(dev) if dev else None
        replication["control_abs_max"] = dev[-1] if dev else None
        replication["control_flagged"] = sum(1 for c in ctl if c["faster"] or c["slower"])
        if unreliable:
            names = sorted({u.split("__")[-1] for u in unreliable})
            print(f"Kernels left out because their null control moved by 3% or more in either run: {len(names)} "
                  f"({', '.join('`' + u + '`' for u in names)}), in {len(unreliable)} studies "
                  f"(alternative resolutions and repriced builds are separate studies).\n")
        headline["replication"] = {k: v for k, v in replication.items() if k != "controls"}

    # ---- do the faster builds have fewer conversions? and what else differs in the binary?
    crosstab, differences = {}, []
    for x in studies:
        if not x["timed"] or x.get("unreliable_timing"):
            continue
        base = n_conversions(x["default_census"])
        base_loop = n_conversions(x["default_census"], "loop")
        db = x.get("default_binary") or {}
        for v in x["timed"]:
            n = n_conversions(v.get("census"))
            rel = ("reordered" if v.get("reordered_only") else
                   "more" if n > base else "fewer" if n < base else "same")
            outcome = "faster" if v["faster"] else "slower" if v["slower"] else "no difference"
            c = crosstab.setdefault(x["section"], {}).setdefault(rel, {"faster": 0, "no difference": 0, "slower": 0})
            c[outcome] += 1
            if v["faster"] or v["slower"]:
                vb = v.get("binary") or {}
                differences.append({
                    "section": x["section"], "label": x["label"], "short": x["short"], "variant": v["name"],
                    "pct": v["vs_default_pct"], "first_run_pct": v.get("first_run_pct"), "second_run_pct": v.get("second_run_pct"),
                    "default_ms": x["default_ms"]["median"], "variant_ms": v["ms"]["median"],
                    "conversions": [base, n], "conversions_in_loop": [base_loop, n_conversions(v.get("census"), "loop")],
                    "registers": [db.get("n_regs"), vb.get("n_regs")], "spills": [db.get("n_spills"), vb.get("n_spills")],
                    "output_exact": v.get("output_exact"), "reordered_only": v.get("reordered_only"),
                    "test": x.get("test")})
    if crosstab:
        print("### Distinct programs: what differs from the default, and the timing verdict\n")
        print("| Section | Against the default | Faster | No reportable difference | Slower |")
        print("|---|---|---|---|---|")
        names = {"reordered": "same operations on the same layouts, in a different order",
                 "fewer": "different operations, fewer non-free conversions",
                 "same": "different operations, as many non-free conversions",
                 "more": "different operations, more non-free conversions"}
        for sec in sorted(crosstab):
            for rel in ("reordered", "fewer", "same", "more"):
                c = crosstab[sec].get(rel)
                if c:
                    print(f"| {sec} | {names[rel]} | {c['faster']} | {c['no difference']} | {c['slower']} |")
        print()
    if differences:
        print("### Every reportable difference\n")
        print("| Section | Kernel | Alternative | vs default (smaller of two runs) | Default, µs | Alternative, µs | "
              "Non-free conversions, default → alternative (in a loop) | Same operations reordered | Registers per thread | Stack frame, bytes |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        fmt = lambda pair, k=1: ("n/a" if pair[0] is None or pair[1] is None else f"{pair[0] * k:,} → {pair[1] * k:,}")
        for d in sorted(differences, key=lambda d: (d["section"], d["pct"])):
            print(f"| {d['section']} | `{d['short']}` | {d['variant']} | {d['pct']:+.1f}% | {d['default_ms'] * 1e3:.2f} | "
                  f"{d['variant_ms'] * 1e3:.2f} | {d['conversions'][0]} → {d['conversions'][1]} "
                  f"({d['conversions_in_loop'][0]} → {d['conversions_in_loop'][1]}) | {'yes' if d['reordered_only'] else 'no'} | "
                  f"{fmt(d['registers'])} | {fmt(d['spills'], 4)} |")
        print("\nRegisters and stack frame are what the CUDA driver reports when the binary is loaded: registers per "
              "thread, and the per-thread local stack frame in bytes (Triton's `n_spills` times four).\n")
    headline["crosstab"] = crosstab

    reprice_rows = []
    if repriced:
        print("### Rebuilt with repriced conversions, on the device\n")
        print("| Section | Kernels rebuilt | Same code | Different code | Timed (stable timing) | Faster (3%, intervals apart, both runs) | Slower | Median difference | Range |")
        print("|---|---|---|---|---|---|---|---|---|")
        by = defaultdict(list)
        for x in repriced:
            by[(x["section"], x["target"])].append(x)
        for (section, target), ss in sorted(by.items()):
            same = sum(1 for x in ss if x["states"].get("identical_code"))
            diff = sum(1 for x in ss if x["states"].get("distinct_code"))
            timed = [v for x in ss for v in x["timed"] if not x.get("unreliable_timing")]
            unstable = sum(1 for x in ss if x["timed"] and x.get("unreliable_timing"))
            pcts = sorted(v["vs_default_pct"] for v in timed)
            row = {"section": section, "target": target, "kernels": len(ss), "same": same, "different": diff,
                   "timed": len(timed), "unstable_kernels": unstable,
                   "faster": sum(v["faster"] for v in timed), "slower": sum(v["slower"] for v in timed),
                   "median_pct": st.median(pcts) if pcts else None, "min_pct": pcts[0] if pcts else None,
                   "max_pct": pcts[-1] if pcts else None,
                   "by_kernel": dict(Counter(x["kernel"] for x in ss if x["states"].get("distinct_code")))}
            reprice_rows.append(row)
            rng = f"{pcts[0]:+.1f}% to {pcts[-1]:+.1f}%" if pcts else "n/a"
            med = f"{st.median(pcts):+.1f}%" if pcts else "n/a"
            print(f"| {section.replace('_reprice', '')} | {len(ss)} | {same} | {diff} | {len(timed)} | {row['faster']} | "
                  f"{row['slower']} | {med} | {rng} |")
        print()

    (out / "e4_summary.json").write_text(json.dumps({
        "sessions": sessions, "scanned": {f"{k[0]}|sm{k[1]}|{k[2]}": v for k, v in scanned.items()},
        "table": table, "inert": inert, "census": census_rows, "headline": headline, "studies": studies,
        "differences": differences, "reprice": reprice_rows, "reprice_studies": repriced}, indent=1,
        default=str))


if __name__ == "__main__":
    main()
