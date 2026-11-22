#!/usr/bin/env python
"""Choose the conversions E3 times.

Two sources:

1. The census. The most frequent conversions that survive into the final TTGIR and the
   most frequent conversions the pass put a non-zero price on, on the device's own target.
2. A synthetic sweep of dtype width and tensor size around one in-warp pair (shuffle by
   default) and one cross-warp pair (shared memory by default).

    python make_e3_cases.py --runs runs/census --arch 120 --out runs/e3/cases.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layoutlab.events import family, load_runs  # noqa: E402

DTYPE_OK = {"f16", "bf16", "f32", "f64", "i8", "i32", "i64"}


def blocked(size, threads, warps, order):
    return (f"#ttg.blocked<{{sizePerThread = {list(size)}, threadsPerWarp = {list(threads)}, "
            f"warpsPerCTA = {list(warps)}, order = {list(order)}}}>")


def routes_for(desc):
    routes = ["default"]
    if desc["class"] == "warp_shuffle":
        routes.append("smem")
    elif desc["class"] == "shared_memory" and desc.get("can_shuffle"):
        routes.append("shuffle")
    return routes


def census_cases(runs, arch, top):
    configs = [c for c in load_runs(runs) if c.ok and c.row.get("arch") == arch]
    survive, priced = Counter(), Counter()
    desc = {}
    for c in configs:
        sec = c.section
        for e in c.final_census():
            if e["class"] in ("noop", "register_permute"):
                continue
            k = (tuple(e["shape"]), e["dtype"], e["src"], e["dst"])
            survive[(sec, k)] += 1
            desc[k] = e
        for e in c.of("convert_cost"):
            if e["cost"] <= 0 or e["class"] in ("noop", "register_permute"):
                continue
            k = (tuple(e["shape"]), e["dtype"], e["src"], e["dst"])
            priced[(sec, k)] += 1
            desc.setdefault(k, e)
    cases = []
    seen = set()

    def add(source, sec, k, n):
        if k in seen:
            return
        shape, dtype, src, dst = k
        note = ""
        if dtype == "i1":
            dtype, note = "i8", "i1 tensor timed as i8 (the lowering widens it)"
        if dtype.startswith("!tt.ptr"):
            dtype, note = "i64", "pointer tensor timed as i64"
        if dtype not in DTYPE_OK:
            return
        seen.add(k)
        d = desc[k]
        cases.append({
            "name": f"{source}{len(cases):02d}_{sec}_{d['class']}_{'x'.join(map(str, shape))}_{dtype}",
            "source": source, "section": sec, "count": n, "shape": list(shape), "dtype": dtype, "src": src, "dst": dst,
            "class": d["class"], "barrier": d["barrier"], "can_shuffle": d.get("can_shuffle"),
            "n_mixed": d.get("n_mixed"), "routes": routes_for(d), "note": note,
        })

    # The most frequent conversions per corpus section, class and role.
    for sec in sorted({s for s, _ in survive} | {s for s, _ in priced}):
        for cls in ("shared_memory", "warp_shuffle"):
            for source, counter in (("survive", survive), ("priced", priced)):
                ranked = [(k, n) for (s, k), n in counter.most_common() if s == sec and desc[k]["class"] == cls]
                for k, n in ranked[:top]:
                    add(source, sec, k, n)
    return cases


def canonical_cases():
    """The conversions a matmul with MMA v2 layouts is built around, at fixed sizes."""
    mma = "#ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>"
    out = [
        {"name": "canon_mma_to_blocked_128x128_f16", "shape": [128, 128], "dtype": "f16", "src": mma,
         "dst": blocked([1, 8], [2, 16], [4, 1], [1, 0]), "class": "shared_memory", "routes": ["default"],
         "note": "accumulator to store layout, as in the matmul epilogue"},
        {"name": "canon_mma_to_blocked_128x128_f32", "shape": [128, 128], "dtype": "f32", "src": mma,
         "dst": blocked([1, 4], [4, 8], [4, 1], [1, 0]), "class": "shared_memory", "routes": ["default"],
         "note": "accumulator to store layout, fp32"},
        {"name": "canon_mma_to_dotop_64x32_f16", "shape": [64, 32], "dtype": "f16", "src": mma,
         "dst": f"#ttg.dot_op<{{opIdx = 0, parent = {mma}, kWidth = 4}}>", "class": "warp_shuffle",
         "routes": ["default", "smem"], "note": "dot result fed to the next dot as its left operand"},
    ]
    for c in out:
        c["source"] = "canonical"
    return out


def synthetic_cases():
    cases = []
    # In-warp: rows spread over lanes vs. half the lanes per row. One warp.
    for dtype in ("i8", "f16", "f32", "f64"):
        for n in (32, 64, 128):
            src = blocked([1, 1], [32, 1], [1, 1], [1, 0])
            dst = blocked([1, 1], [16, 2], [1, 1], [1, 0])
            cases.append({"name": f"syn_inwarp_32x{n}_{dtype}", "source": "synthetic", "shape": [32, n],
                          "dtype": dtype, "src": src, "dst": dst, "class": "warp_shuffle",
                          "routes": ["default", "smem"], "note": "one warp; elements per thread grows with N"})
    # Cross-warp: 4 warps stacked along rows vs. along columns.
    for dtype in ("i8", "f16", "f32", "f64"):
        for n in (32, 64, 128):
            src = blocked([1, 1], [32, 1], [4, 1], [1, 0])
            dst = blocked([1, 1], [1, 32], [1, 4], [0, 1])
            cases.append({"name": f"syn_crosswarp_{n}x{n}_{dtype}", "source": "synthetic", "shape": [n, n],
                          "dtype": dtype, "src": src, "dst": dst, "class": "shared_memory",
                          "routes": ["default"], "note": "four warps; data moves between warps"})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--arch", type=int, default=120)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = census_cases(args.runs, args.arch, args.top) + canonical_cases() + synthetic_cases()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cases, indent=1))
    for c in cases:
        print(c["name"], c.get("count", ""), c["routes"])


if __name__ == "__main__":
    main()
