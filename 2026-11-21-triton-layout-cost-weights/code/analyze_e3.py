#!/usr/bin/env python
"""E3 tables: measured cost of a conversion per lowering class, against what the pass assumes.

    python analyze_e3.py --run runs/e3 --out ../results

Writes e3_summary.json (every case and route), e3_ratios.json (measured/assumed price
ratios for the E2 reprice) and prints markdown tables.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

BITS = {"i8": 8, "f16": 16, "bf16": 16, "i32": 32, "f32": 32, "i64": 64, "f64": 64}


def load(run: Path):
    lowered, detail = defaultdict(list), defaultdict(list)
    for line in open(run / "events.jsonl"):
        e = json.loads(line)
        if e["event"] == "lowered":
            lowered[e["tag"]].append(e)
        elif e["event"] == "lowered_smem_detail":
            detail[e["tag"]].append(e)
    rows = []
    errors = []
    for line in open(run / "results.jsonl"):
        r = json.loads(line)
        c = r["case"]
        if "add_ns" not in r:
            errors.append({"case": c["name"].replace("_other_", "_triton_kernels_"), "error": r.get("error", "")})
            continue
        for route, v in r["routes"].items():
            if v.get("status") != "ok":
                errors.append({"case": c["name"].replace("_other_", "_triton_kernels_"), "route": route,
                               "error": v.get("status") + ": " + v.get("error", "")})
                continue
            lo = lowered.get(v["tag"], [])
            classes = sorted({x["lowered_class"] for x in lo})
            barriers = sorted({x["barrier"] for x in lo if x["lowered_class"] == "shared_memory"})
            # the two directions of a pair can lower differently; label the pair by what it contains
            cls = classes[0] if len(classes) == 1 else "mixed"
            first = lo[0] if lo else {}
            elems = first.get("elems", 0)
            bits = BITS[c["dtype"]]
            assumed = (0 if first.get("priced_free") else 32 * first.get("priced_bytes", 0))
            clock = r["clock"].get("clock_mhz_median")
            is_f64 = c["dtype"] == "f64"
            rows.append({
                "case": c["name"].replace("_other_", "_triton_kernels_"), "source": c.get("source"),
                "section": "triton_kernels" if c.get("section") == "other" else c.get("section"), "route": route,
                "shape": c["shape"], "dtype": c["dtype"], "bits": bits, "elems": elems, "bytes": r["bytes"],
                "num_warps": r["num_warps"], "class": cls, "classes": classes, "barrier": "/".join(barriers) or "none",
                "reps": [d["reps"] for d in detail.get(v["tag"], [])],
                "instr": [[d["store_instr"], d["load_instr"]] for d in detail.get(v["tag"], [])],
                "add_ns": r["add_ns"], "add_src_ns": r["add_src_ns"], "add_dst_ns": r["add_dst_ns"],
                "cvt_ns": v["cvt_ns"], "cvt_ns_ci": v["cvt_ns_ci"],
                "cvt_per_add": None if is_f64 else v["cvt_per_add"],
                "cvt_per_add_ci": None if is_f64 else v["cvt_per_add_ci"],
                # what the pass charges, in units of its own price for one elementwise op on this tensor
                "assumed_per_add": assumed / r["bytes"] if r["bytes"] else None,
                "assumed_cost": assumed,
                "clock_mhz": clock, "cvt_sm_cycles": v["cvt_ns"] * clock / 1e3 if clock else None,
                "assumed_sm_cycles": assumed / 1e3,
                "separated": v["separated"], "dst_baseline": r.get("dst_baseline"), "count": c.get("count"),
                "grid": r["grid"], "n_iter": r["n_iter"], "runs": r["runs"],
            })
    return rows, errors


def fmt(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, errors = load(run)
    session = json.loads((run / "session.json").read_text())

    print("## E3: measured cost of one conversion\n")
    print(f"Device: {session['nvidia_smi']}, compute capability {session['compute_capability']}, CUDA runtime "
          f"{session['cuda_runtime']}, {session['n_sm']} SMs. {session['runs']} timed runs per variant after "
          f"{session['warmup']} warm-up, {session['grid_mult']} CTAs per SM.\n")

    # ---- same conversion through both routes
    print("### The same conversion through warp shuffles and through shared memory\n")
    print("| Conversion | bytes | warps | default route | shuffle: SM-ns | adds | shared memory: SM-ns (barrier) | adds | shuffle / shared memory | assumed (adds) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    by_case = defaultdict(dict)
    for r in rows:
        by_case[r["case"]][r["route"]] = r
    pairs = []
    for name, routes in by_case.items():
        d = routes.get("default")
        if d is None:
            continue
        other = routes.get("smem") or routes.get("shuffle")
        if other is None:
            continue
        # which is the shuffle route and which the shared-memory route
        if "smem" in routes and d["class"] == "warp_shuffle":
            sh, sm = d, routes["smem"]
        elif "shuffle" in routes and d["class"] == "shared_memory":
            sh, sm = routes["shuffle"], d
        else:
            continue  # a pair whose two directions take different routes by default
        if sh["class"] != "warp_shuffle" or sm["class"] != "shared_memory":
            continue
        ratio = sh["cvt_ns"] / sm["cvt_ns"]
        pairs.append({"case": name, "bytes": d["bytes"], "dtype": d["dtype"], "shuffle_ns": sh["cvt_ns"],
                      "smem_ns": sm["cvt_ns"], "ratio": ratio, "barrier": sm["barrier"],
                      "shuffle_adds": sh["cvt_per_add"], "smem_adds": sm["cvt_per_add"],
                      "ci_overlap": not (sh["cvt_ns_ci"][1] < sm["cvt_ns_ci"][0] or sm["cvt_ns_ci"][1] < sh["cvt_ns_ci"][0])})
        print(f"| `{name}` | {d['bytes']} | {d['num_warps']} | {d['class'].replace('_', ' ')} | {fmt(sh['cvt_ns'])} | "
              f"{fmt(sh['cvt_per_add'])} | {fmt(sm['cvt_ns'])} ({sm['barrier']}) | {fmt(sm['cvt_per_add'])} | "
              f"{ratio:.2f} | {fmt(d['assumed_per_add'], 0)} |")
    ratios = sorted(p["ratio"] for p in pairs)
    ratios_no64 = sorted(p["ratio"] for p in pairs if p["dtype"] != "f64")
    if ratios:
        print(f"\nShuffle cost over shared-memory cost for the same conversion, {len(ratios)} conversions: "
              f"median {st.median(ratios):.2f}, range {ratios[0]:.2f} to {ratios[-1]:.2f}. Without the three fp64 "
              f"cases, where the conversion is a small difference of two long loops: median "
              f"{st.median(ratios_no64):.2f}, range {ratios_no64[0]:.2f} to {ratios_no64[-1]:.2f}.\n")

    # ---- by class
    print("### Default route, by class\n")
    print("| Conversion | source | dtype | bytes | warps | class (barrier, rounds) | SM-ns per conversion [95% interval] | SM cycles | adds [95% interval] | assumed (adds) | assumed (SM cycles) | measured / assumed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    table = []
    for r in rows:
        if r["route"] != "default":
            continue
        alpha = None
        if r["cvt_per_add"] is not None and r["assumed_per_add"]:
            alpha = r["cvt_per_add"] / r["assumed_per_add"]
        r["alpha"] = alpha
        table.append(r)
        cls = r["class"].replace("_", " ")
        if r["class"] in ("shared_memory", "mixed"):
            cls += f" ({r['barrier']}, {'+'.join(map(str, r['reps']))})"
        ci = r["cvt_per_add_ci"]
        adds = "n/a" if r["cvt_per_add"] is None else f"{r['cvt_per_add']:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]"
        flag = "" if r["separated"] else " (not separated)"
        print(f"| `{r['case']}` | {r['source']} | {r['dtype']} | {r['bytes']} | {r['num_warps']} | {cls} | "
              f"{r['cvt_ns']:.1f} [{r['cvt_ns_ci'][0]:.1f}, {r['cvt_ns_ci'][1]:.1f}]{flag} | {fmt(r['cvt_sm_cycles'], 0)} | "
              f"{adds} | {fmt(r['assumed_per_add'], 0)} | {fmt(r['assumed_sm_cycles'], 0)} | {fmt(alpha, 3)} |")

    # ---- summary per class and dtype width
    print("\n### Summary: conversions of at least 1 KB, by class and element width\n")
    print("| Class | Element width | Cases | adds per conversion: median (range) | assumed | measured / assumed: median |")
    print("|---|---|---|---|---|---|")
    summary = []
    groups = defaultdict(list)
    for r in table:
        if r["cvt_per_add"] is None or r["bytes"] < 1024 or r["class"] == "mixed":
            continue
        cls = r["class"] if r["class"] != "shared_memory" else f"shared_memory ({r['barrier']} barrier)"
        groups[(cls, r["bits"])].append(r)
    for (cls, bits), rs in sorted(groups.items()):
        adds = sorted(x["cvt_per_add"] for x in rs)
        alphas = sorted(x["alpha"] for x in rs if x["alpha"] is not None)
        alpha_med = st.median(alphas) if alphas else None
        assumed = sorted({round(x["assumed_per_add"]) for x in rs})
        print(f"| {cls.replace('_', ' ')} | {bits} bits | {len(rs)} | {st.median(adds):.1f} ({adds[0]:.1f} to {adds[-1]:.1f}) | "
              f"{'/'.join(map(str, assumed))} | {fmt(alpha_med, 3)} |")
        summary.append({"class": cls, "bits": bits, "n": len(rs), "adds_median": st.median(adds), "adds_min": adds[0],
                        "adds_max": adds[-1], "assumed": assumed, "alpha_median": alpha_med})

    # ---- ratios for the E2 reprice: measured / assumed by element width and size
    def bucket(r):
        size = "tiny" if r["elems"] < 32 else ("small" if r["bytes"] < 1024 else "large")
        return f"{r['bits']}:{size}"

    buckets = defaultdict(list)
    for r in table:
        if r["alpha"] is not None and r["class"] != "mixed":
            buckets[bucket(r)].append(r["alpha"])
    all_alpha = sorted(a for v in buckets.values() for a in v)
    ratio_table = {k: {"median": st.median(v), "min": min(v), "max": max(v), "n": len(v)} for k, v in sorted(buckets.items())}
    print("\n### Measured over assumed price, by element width and size (input to the E2 reprice)\n")
    print("| Element width | Size | Cases | median | min | max |")
    print("|---|---|---|---|---|---|")
    for k, v in ratio_table.items():
        bits, size = k.split(":")
        label = {"tiny": "under 32 elements", "small": "32 elements to 1 KB", "large": "1 KB and over"}[size]
        print(f"| {bits} bits | {label} | {v['n']} | {v['median']:.3f} | {v['min']:.3f} | {v['max']:.3f} |")
    print(f"\nAll {len(all_alpha)} measured conversions: median {st.median(all_alpha):.3f}, "
          f"range {all_alpha[0]:.3f} to {all_alpha[-1]:.3f}.\n")

    if errors:
        print("### Cases that could not be measured\n")
        for e in errors:
            print(f"- `{e['case']}`" + (f" ({e['route']} route)" if e.get("route") else "") + f": {e['error'][:160]}")

    (out / "e3_summary.json").write_text(json.dumps({"session": session, "rows": rows, "pairs": pairs,
                                                    "by_class": summary, "errors": errors}, indent=1))
    (out / "e3_ratios.json").write_text(json.dumps({
        "by_bucket": ratio_table,
        "overall": {"median": st.median(all_alpha), "min": all_alpha[0], "max": all_alpha[-1], "n": len(all_alpha)},
        "shuffle_over_smem": {"median": st.median(ratios), "min": ratios[0], "max": ratios[-1], "n": len(ratios),
                              "median_without_f64": st.median(ratios_no64), "n_without_f64": len(ratios_no64)} if ratios else None,
    }, indent=1))


if __name__ == "__main__":
    main()
