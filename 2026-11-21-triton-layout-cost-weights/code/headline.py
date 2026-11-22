#!/usr/bin/env python
"""The headline block, computed from the result summaries so no number is typed by hand.

    python headline.py --results ../results [--arch 120] [--json]

--json also prints the numbers behind the sentences, for every target.

Totals are over the tutorials, Inductor and triton_kernels sections. Configurations compiled
under the fpsan instrumentation mode are a separate section and are left out.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

SECTIONS = ("tutorials", "inductor", "triton_kernels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--arch", type=int, default=120)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = Path(args.results)
    s = json.loads((res / "e1_e2_summary.json").read_text())
    e3 = json.loads((res / "e3_summary.json").read_text())
    ratios = json.loads((res / "e3_ratios.json").read_text())
    e4 = json.loads((res / "e4_summary.json").read_text()) if (res / "e4_summary.json").exists() else None
    out = {}

    def rows(key, arch):
        return [x for x in s[key] if x["arch"] == arch and x["section"] in SECTIONS]

    per_arch = {}
    for arch in (80, 90, 100, 120):
        c = rows("e1_conflicts", arch)
        n, k = sum(x["N"] for x in c), sum(x["K"] for x in c)
        cen = rows("e1_final_census", arch)
        m = sum(x["M"] for x in cen)
        cls = {}
        for x in cen:
            for name, v in x["classes"].items():
                cls[name] = cls.get(name, 0) + v
        pr = rows("e2_pricing", arch)
        priced = sum(x["priced_smem"] for x in pr)
        cheaper = sum(x["priced_smem"] - x["priced_classes"].get("shared_memory", 0) for x in pr)
        dec = rows("e2_decisions", arch)
        r = sum(x["R"] for x in dec)
        flips = {}
        for x in rows("e2_reprice", arch):
            for name, v in x["flips"].items():
                flips[name] = flips.get(name, 0) + v
        be = rows("e2_break_even", arch)
        per_arch[arch] = {
            "N": n, "K": k, "K_pct": 100 * k / n, "M": m,
            "shared_memory_pct": 100 * cls.get("shared_memory", 0) / m, "warp_shuffle_pct": 100 * cls.get("warp_shuffle", 0) / m,
            "free_pct": 100 * (cls.get("noop", 0) + cls.get("register_permute", 0)) / m,
            "classes": cls, "priced": priced, "priced_cheaper": cheaper, "P_pct": 100 * cheaper / priced,
            "R": r, "flips": flips,
            "priced_subject_remat": sum(x["n"] for x in be if x["remat"]),
            "priced_subject_keep": sum(x["n"] for x in be if not x["remat"]),
            "in_loop": sum(x["in_loop"] for x in cen), "configs_without": sum(x["configs_without"] for x in cen),
        }
    out["per_arch"] = per_arch
    a = per_arch[args.arch]

    shuffle_only = next(v for k, v in a["flips"].items() if k.startswith("warp shuffle only, at measured"))
    all_measured = a["flips"]["every class, measured ratio"]

    def pick(name, route="default"):
        return next(r for r in e3["rows"] if r["case"] == name and r["route"] == route)

    sh = pick("syn_inwarp_32x32_f32", "default")
    sm = pick("syn_inwarp_32x32_f32", "smem")
    out["e3_example"] = {"case": "32x32 fp32, one warp, same conversion both ways",
                         "shuffle_ns": sh["cvt_ns"], "smem_ns": sm["cvt_ns"], "shuffle_adds": sh["cvt_per_add"],
                         "smem_adds": sm["cvt_per_add"], "assumed_adds": sh["assumed_per_add"],
                         "shuffle_cycles": sh["cvt_sm_cycles"], "smem_cycles": sm["cvt_sm_cycles"],
                         "assumed_cycles": sh["assumed_sm_cycles"], "clock_mhz": sh["clock_mhz"]}
    out["e3_ratios"] = ratios

    lines = []
    lines.append(f'Across {a["N"]:,} kernel configurations, the pass faced a real choice in {a["K"]:,} ({a["K_pct"]:.0f}%).')
    lines.append(f'Of {a["M"]:,} conversions that survive the pass, {a["shared_memory_pct"]:.0f}% are shared-memory round '
                 f'trips, {a["warp_shuffle_pct"]:.0f}% warp shuffles, {a["free_pct"]:.0f}% free.')
    lines.append(f'{a["P_pct"]:.1f}% of conversions the pass priced as shared-memory round trips lower to something '
                 f'cheaper; repricing them flips {shuffle_only} of {a["R"]:,} rematerialization decisions.')
    ex = out["e3_example"]
    lines.append(f'On the RTX 5090, a warp-shuffle conversion costs {ex["shuffle_adds"]:.1f} elementwise adds of the same '
                 f'tensor and a shared-memory one costs {ex["smem_adds"]:.1f}, against an assumed {ex["assumed_adds"]:.0f} '
                 f'({ex["shuffle_ns"]:.1f} and {ex["smem_ns"]:.1f} ns of SM time for a 4 KB fp32 tensor; '
                 f'{ex["shuffle_cycles"]:.0f} and {ex["smem_cycles"]:.0f} SM cycles against an assumed {ex["assumed_cycles"]:.0f}).')
    if e4:
        h = e4["headline"]
        if h.get("reportably_faster"):
            sec = h.get("largest_reportable_gain_section")
            name = (h.get("extremes", {}).get(sec, {}).get("largest_gain", {}).get("short")
                    or h["largest_reportable_gain_kernel"])
            lines.append(f'The best alternative resolution beat the default by at most {h["largest_reportable_gain_pct"]:.1f}% '
                         f'(kernel: {sec} {name}); median {h["gap_median_pct"]:.0f}%.')
            for section, e in sorted(h.get("extremes", {}).items()):
                for key, word in (("largest_gain", "largest gain"), ("largest_loss", "largest loss")):
                    if key in e:
                        x = e[key]
                        lines.append(f'  ({section}, {word} among the alternatives tried: {x["pct"]:+.1f}%, {x["short"]}, '
                                     f'{x["default_us"]:.2f} µs default against {x["variant_us"]:.2f} µs)')
        else:
            lines.append(f'The best alternative resolution beat the default by at most 0% (no alternative on any of '
                         f'{h.get("device_kernels_with_conflict_studied", 0)} conflicted kernels was faster by 3% with '
                         f'non-overlapping intervals); median 0.')
        out["e4"] = h
    out["extra"] = {"all_classes_measured_flips": all_measured, "R": a["R"]}
    print("\n".join(f"- {line}" for line in lines))
    if args.json:
        print()
        print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
