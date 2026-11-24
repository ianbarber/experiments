#!/usr/bin/env python
"""Closed-loop check of the E2 reprice: recompile with `TRITON_LAYOUT_REPRICE` set and see what changes.

    python compare_reprice.py --base runs/census --repriced runs/reprice_0.12 --out ../results --name reprice_0.12

Matches configurations by what was compiled and compares the final TTGIR and the census of
conversions left in it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layoutlab.events import family, load_runs  # noqa: E402


def spec_key(row):
    return json.dumps([row["workload"], row["kernel"], row.get("arch"), row.get("signature"), row.get("constants"),
                       row.get("options")], sort_keys=True, default=str)


def census(c):
    out = Counter()
    for e in c.final_census():
        key = e["class"] if e["class"] != "shared_memory" else f'shared_memory:{e["barrier"]}'
        out[key + ("@loop" if e["loop_depth"] > 0 else "")] += 1
    return out


def nonfree(cen, loop_only=False):
    return sum(v for k, v in cen.items()
               if k.split("@")[0].split(":")[0] not in ("noop", "register_permute") and (not loop_only or "@loop" in k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--repriced", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    rep = [c for c in load_runs(args.repriced) if c.ok and c.row.get("ttgir")]
    dirs = {Path(c.row["run_dir"]).name for c in rep}
    base = {}
    for c in load_runs(args.base):
        if c.ok and c.row.get("ttgir") and Path(c.row["run_dir"]).name in dirs:
            base.setdefault(spec_key(c.row), []).append(c)

    rows = defaultdict(lambda: Counter())
    changed = []
    for c in rep:
        key = spec_key(c.row)
        sec, arch = c.section, c.row.get("arch")
        others = base.get(key)
        if not others:
            rows[(sec, arch)]["unmatched"] += 1
            continue
        text = (Path(c.row["run_dir"]) / "ttgir" / c.row["ttgir"]).read_bytes()
        same = any((Path(o.row["run_dir"]) / "ttgir" / o.row["ttgir"]).read_bytes() == text for o in others)
        rows[(sec, arch)]["configs"] += 1
        if same:
            rows[(sec, arch)]["identical"] += 1
            continue
        rows[(sec, arch)]["changed"] += 1
        b, r = census(others[0]), census(c)
        d = nonfree(r) - nonfree(b)
        dl = nonfree(r, True) - nonfree(b, True)
        rows[(sec, arch)]["more_conversions" if d > 0 else "fewer_conversions" if d < 0 else "same_conversions"] += 1
        rows[(sec, arch)]["more_in_loop" if dl > 0 else "fewer_in_loop" if dl < 0 else "same_in_loop"] += 1
        changed.append({"section": sec, "arch": arch, "kernel": c.row["kernel"], "constants": c.row.get("constants"),
                        "options": c.row.get("options"), "base": dict(b), "repriced": dict(r)})

    print(f"## Recompiled with repriced conversions ({args.name})\n")
    print("| Section | Target | Configurations | Final TTGIR identical | Changed | of which more non-free conversions | same | fewer | more inside a loop | fewer inside a loop |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    table = []
    for (sec, arch), c in sorted(rows.items()):
        print(f"| {sec} | sm_{arch} | {c['configs']} | {c['identical']} | {c['changed']} | {c['more_conversions']} | "
              f"{c['same_conversions']} | {c['fewer_conversions']} | {c['more_in_loop']} | {c['fewer_in_loop']} |")
        table.append({"section": sec, "arch": arch, **dict(c)})
    kernels = Counter((x["section"], x["kernel"]) for x in changed)
    if kernels:
        print("\nKernels that changed: " + ", ".join(f"`{k}` ({n})" for (_, k), n in kernels.most_common(12)) + ".")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.name}.json").write_text(json.dumps({"table": table, "changed": changed[:400]}, indent=1))


if __name__ == "__main__":
    main()
