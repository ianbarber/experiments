#!/usr/bin/env python
"""E1 (census) and E2 (pricing) tables from census runs.

    python analyze_census.py --runs runs/census --out ../results [--vanilla runs/vanilla]

Writes e1_e2_summary.json and prints markdown tables. With --vanilla it also diffs the
final TTGIR of every configuration against the unpatched build.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layoutlab.events import ARCHS, family, frontend_rejected, layout_kind, load_runs  # noqa: E402

CLASSES = ("shared_memory", "warp_shuffle", "register_permute", "noop")


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def reprice_flips(decisions, ratio):
    """Count decisions whose outcome changes when conversion costs are repriced.

    `ratio` maps a class (or "class:barrier") to a multiplier on the pass's price, or to a
    callable `(description, old_cost) -> new_cost`. The key "*" applies to every class.
    This is first-order: the pass is not re-run, so later decisions that would have seen a
    different IR are not followed.
    """

    def new_cost(desc, old):
        key = f'{desc["class"]}:{desc["barrier"]}'
        r = ratio.get(key, ratio.get(desc["class"], ratio.get("*")))
        if r is None:
            return old
        if callable(r):
            return r(desc, old)
        return r * old

    flips = []
    for e in decisions:
        if e["reason"] != "cost":
            continue
        c, r, n = e["convert_cost"], e["remat_cost"], e["new_cvt_cost"]
        c2 = new_cost(e["subject"], c)
        n2 = sum(new_cost(d, d["cost"]) for d in e.get("new_conversions", []))
        r2 = r - n + n2
        before, after = c >= r, c2 >= r2
        assert before == e["outcome"], e
        if before != after:
            flips.append((e, after))
    return flips


def measured_ratio(table):
    """Measured / assumed price for a conversion, from E3's table by element width and size."""
    by_bucket = table["by_bucket"]
    overall = table["overall"]["median"]

    def lookup(desc, which="median"):
        bits = desc["bits"]
        width = 8 if bits <= 8 else 16 if bits <= 16 else 32 if bits <= 32 else 64
        size = "tiny" if desc["elems"] < 32 else ("small" if desc["bytes"] < 1024 else "large")
        for cand in (size, "large", "small", "tiny"):
            v = by_bucket.get(f"{width}:{cand}")
            if v:
                return v[which]
        return overall

    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vanilla", default=None)
    ap.add_argument("--ratios", default=None, help="JSON file with measured class ratios from E3")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = load_runs(args.runs)
    summary = {"sections": {}}

    # ------------------------------------------------------------------ corpus
    by_section = defaultdict(list)
    for c in configs:
        by_section[c.section].append(c)

    print("## Corpus\n")
    print("| Section | Workload | Kernel | " + " | ".join(f"sm_{a}" for a in ARCHS) + " |")
    print("|---|---|---|" + "---|" * len(ARCHS))
    corpus_rows = []
    cells = defaultdict(lambda: Counter())
    rejected = Counter()
    failed = defaultdict(list)
    for c in configs:
        key = (c.section, c.row["workload"].split(":")[0], c.row["kernel"])
        if c.ok:
            cells[key][c.row.get("arch")] += 1
        elif frontend_rejected(c):
            rejected[(key, c.row.get("arch"))] += 1
        else:
            failed[key].append(c)
    for key in sorted(cells):
        row = [str(cells[key].get(a, 0)) for a in ARCHS]
        print(f"| {key[0]} | {key[1]} | `{key[2]}` | " + " | ".join(row) + " |")
        corpus_rows.append({"section": key[0], "workload": key[1], "kernel": key[2],
                            **{f"sm{a}": cells[key].get(a, 0) for a in ARCHS}})
    summary["corpus"] = corpus_rows
    summary["frontend_rejected"] = sum(rejected.values())
    summary["failed_after_events"] = {f"{k}": len(v) for k, v in failed.items()}
    print(f"\nConfigurations rejected by the kernel's own compile-time checks (never reach the pass): "
          f"{sum(rejected.values())}. Failed after the pass started: {sum(len(v) for v in failed.values())}.\n")

    ok = [c for c in configs if c.ok]

    # ------------------------------------------------------------------ E1 conflicts
    print("## E1: values reaching resolveConflicts with more than one candidate\n")
    print("| Section | Target | Configurations (N) | With a conflict (K) | Conflict values | By pipeline invocation 1/2/3/4 | Configurations with nested runs |")
    print("|---|---|---|---|---|---|---|")
    e1 = []
    for section in sorted(by_section):
        for arch in ARCHS:
            cs = [c for c in by_section[section] if c.ok and c.row.get("arch") == arch]
            if not cs:
                continue
            with_conflict = [c for c in cs if c.conflicts()]
            n_values = sum(len(c.conflicts()) for c in cs)
            by_inv = Counter()
            for c in cs:
                labels = c.invocation_labels()
                for e in c.conflicts():
                    by_inv[labels.get(e["inv"], "?")] += 1
            inv = "/".join(str(by_inv.get(i, 0)) for i in ("1", "2", "3", "4"))
            extra = ", ".join(f"{by_inv[k]} {name}" for k, name in (("nested", "in nested runs"), ("fpsan", "in the fpsan run"))
                              if by_inv.get(k))
            nested_cfg = sum(1 for c in cs if "nested" in c.invocation_labels().values())
            print(f"| {section} | sm_{arch} | {len(cs)} | {len(with_conflict)} | {n_values} | {inv}"
                  f"{' (+' + extra + ')' if extra else ''} | {nested_cfg} |")
            e1.append({"section": section, "arch": arch, "N": len(cs), "K": len(with_conflict),
                       "conflict_values": n_values, "by_invocation": dict(by_inv), "configs_with_nested_runs": nested_cfg})
    summary["e1_conflicts"] = e1

    # what the conflicts look like
    print("\n### Conflicts by kernel, candidates and rule\n")
    print("| Section | Kernel | Target | Configs with conflict / all | Op | Candidates (chosen first) | Rule | Values |")
    print("|---|---|---|---|---|---|---|---|")
    detail = Counter()
    kernel_tot = Counter()
    kernel_with = Counter()
    for c in ok:
        k = (c.section, c.row["kernel"], c.row.get("arch"))
        kernel_tot[k] += 1
        if c.conflicts():
            kernel_with[k] += 1
        for e in c.conflicts():
            kinds = [layout_kind(x) for x in e["candidates"]]
            chosen = kinds[e["chosen"]]
            others = sorted(kinds[:e["chosen"]] + kinds[e["chosen"] + 1:])
            op = e["op"] if e["op"] != "block_arg" else f'block_arg({e.get("block_parent", "")})'
            detail[(k, op, (chosen, *others), e["rule"])] += 1
    conflict_detail = []
    for (k, op, kinds, rule), n in sorted(detail.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f"| {k[0]} | `{k[1]}` | sm_{k[2]} | {kernel_with[k]}/{kernel_tot[k]} | `{op}` | {' vs '.join(kinds)} | {rule} | {n} |")
        conflict_detail.append({"section": k[0], "kernel": k[1], "arch": k[2], "configs_with": kernel_with[k],
                                "configs": kernel_tot[k], "op": op, "candidates": kinds, "rule": rule, "values": n})
    summary["e1_conflict_detail"] = conflict_detail

    # ------------------------------------------------------------------ E1 final census
    print("\n## E1: convert_layout ops in the final TTGIR\n")
    print("| Section | Target | Conversions (M) | shared memory (warp / CTA / cluster barrier) | warp shuffle | register permute | no-op | In a loop | Configs with none |")
    print("|---|---|---|---|---|---|---|---|---|")
    e1c = []
    for section in sorted(by_section):
        for arch in ARCHS:
            cs = [c for c in by_section[section] if c.ok and c.row.get("arch") == arch]
            if not cs:
                continue
            ev = [e for c in cs for e in c.final_census()]
            cls = Counter(e["class"] for e in ev)
            bar = Counter(e["barrier"] for e in ev if e["class"] == "shared_memory")
            in_loop = sum(1 for e in ev if e["loop_depth"] > 0)
            none = sum(1 for c in cs if not c.final_census())
            m = len(ev)
            print(f"| {section} | sm_{arch} | {m} | {cls['shared_memory']} ({pct(cls['shared_memory'], m)}; "
                  f"{bar['warp']} / {bar['cta']} / {bar['cluster']}) | {cls['warp_shuffle']} ({pct(cls['warp_shuffle'], m)}) | "
                  f"{cls['register_permute']} ({pct(cls['register_permute'], m)}) | {cls['noop']} ({pct(cls['noop'], m)}) | "
                  f"{in_loop} ({pct(in_loop, m)}) | {none}/{len(cs)} |")
            e1c.append({"section": section, "arch": arch, "M": m, "classes": dict(cls), "barriers": dict(bar),
                        "in_loop": in_loop, "configs_without": none, "configs": len(cs),
                        "bytes": sorted(Counter(e["bytes"] for e in ev).items())})
    summary["e1_final_census"] = e1c

    # ground truth check
    lowered = Counter()
    mismatch = 0
    for c in ok:
        for e in c.of("lowered"):
            lowered[(e["class"], e["lowered_class"])] += 1
        mismatch += sum(1 for e in c.events if e.get("classifier_mismatch"))
    disagree = sum(n for (a, b), n in lowered.items() if a != b)
    print(f"\nClassifier check: {sum(lowered.values())} conversions reached the lowering; the static class and the "
          f"branch actually taken disagree on {disagree}. Internal cross-check flags: {mismatch}.\n")
    summary["classifier_check"] = {"lowered": sum(lowered.values()), "disagree": disagree, "flags": mismatch,
                                   "table": {f"{a}->{b}": n for (a, b), n in lowered.items()}}

    # ------------------------------------------------------------------ E2 pricing
    print("## E2: what getConvertCost charged against the class the conversion lowers to\n")
    print("| Section | Target | getConvertCost calls | priced free (0) | priced as shared memory | of those: shared memory | warp shuffle | register permute | no-op | mispriced share (P) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    e2 = []
    for section in sorted(by_section):
        for arch in ARCHS:
            cs = [c for c in by_section[section] if c.ok and c.row.get("arch") == arch]
            if not cs:
                continue
            calls = [e for c in cs for e in c.of("convert_cost")]
            free = [e for e in calls if e["cost"] == 0]
            priced = [e for e in calls if e["cost"] > 0]
            cls = Counter(e["class"] for e in priced)
            other = len(priced) - cls["shared_memory"]  # priced as shared memory, lowers some other way
            free_wrong = Counter(e["class"] for e in free if e["class"] in ("shared_memory", "warp_shuffle"))
            print(f"| {section} | sm_{arch} | {len(calls)} | {len(free)} | {len(priced)} | {cls['shared_memory']} | "
                  f"{cls['warp_shuffle']} | {cls['register_permute']} | {cls['noop']} | {pct(other, len(priced))} |")
            e2.append({"section": section, "arch": arch, "calls": len(calls), "priced_free": len(free),
                       "priced_smem": len(priced), "priced_classes": dict(cls), "free_but_not_free": dict(free_wrong)})
    summary["e2_pricing"] = e2

    # decisions
    print("\n## E2: isRematBeneficial decisions\n")
    print("| Section | Target | Evaluations | Bail-outs (no cost test) | Cost decisions (R) | subject free | subject shared memory | subject warp shuffle | remat chosen | with new conversions |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    e2d = []
    all_decisions = defaultdict(list)
    for section in sorted(by_section):
        for arch in ARCHS:
            cs = [c for c in by_section[section] if c.ok and c.row.get("arch") == arch]
            if not cs:
                continue
            ds = [e for c in cs for e in c.decisions()]
            cost = [e for e in ds if e["reason"] == "cost"]
            all_decisions[(section, arch)] = cost
            sub = Counter(e["subject"]["class"] for e in cost)
            chosen = sum(1 for e in cost if e["outcome"])
            hoist = sum(1 for e in cost if e.get("new_conversions"))
            free = sub["noop"] + sub["register_permute"]
            print(f"| {section} | sm_{arch} | {len(ds)} | {len(ds) - len(cost)} | {len(cost)} | {free} | "
                  f"{sub['shared_memory']} | {sub['warp_shuffle']} | {chosen} | {hoist} |")
            e2d.append({"section": section, "arch": arch, "evaluations": len(ds), "bailouts": len(ds) - len(cost),
                        "R": len(cost), "subject_classes": dict(sub), "remat_chosen": chosen, "with_new_conversions": hoist})
    summary["e2_decisions"] = e2d

    # break-even ratios: how much cheaper than assumed must the conversion be for the decision to flip
    print("\n### How far the price would have to move\n")
    print("For a decision with a priced subject and no new conversions, the outcome flips when the conversion's real "
          "cost falls below `remat_cost / convert_cost` of the assumed price (if rematerialization was chosen) or rises "
          "above it (if not).\n")
    print("| Section | Target | Class | Outcome | Decisions | break-even ratio: min | median | max | flip at 0.5x | at 0.25x | at 0.1x | at 0.03x | at 0.01x | at 2x | at 4x |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    be = []
    for (section, arch), cost in sorted(all_decisions.items()):
        for cls in ("shared_memory", "warp_shuffle"):
            for outcome in (True, False):
                sel = [e for e in cost if e["subject"]["class"] == cls and e["outcome"] == outcome
                       and e["convert_cost"] > 0 and not e.get("new_conversions")]
                if not sel:
                    continue
                ratios = sorted(e["remat_cost"] / e["convert_cost"] for e in sel)
                med = ratios[len(ratios) // 2]

                def flips(x):
                    return sum(1 for e in sel if (x * e["convert_cost"] >= e["remat_cost"]) != e["outcome"])

                print(f"| {section} | sm_{arch} | {cls} | {'remat' if outcome else 'keep convert'} | {len(sel)} | "
                      f"{ratios[0]:.4g} | {med:.4g} | {ratios[-1]:.4g} | " +
                      " | ".join(str(flips(x)) for x in (0.5, 0.25, 0.1, 0.03, 0.01, 2, 4)) + " |")
                be.append({"section": section, "arch": arch, "class": cls, "remat": outcome, "n": len(sel),
                           "min": ratios[0], "median": med, "max": ratios[-1],
                           "flips": {str(x): flips(x) for x in (0.5, 0.25, 0.1, 0.03, 0.01, 2, 4)}})
    summary["e2_break_even"] = be

    # parametrised reprice
    grid = {}
    for x in (0.5, 0.1, 0.01):
        grid[f"warp shuffle only, at {x}x"] = {"warp_shuffle": x}
    if args.ratios:
        measured = json.loads(Path(args.ratios).read_text())
        lookup = measured_ratio(measured)
        shuffle_ratio = measured["shuffle_over_smem"]["median"]
        grid[f"warp shuffle only, at measured {shuffle_ratio:.2f}x"] = {"warp_shuffle": shuffle_ratio}
        grid["every class, measured ratio"] = {"*": lambda d, old: lookup(d) * old}
        grid["every class, lowest measured in bucket"] = {"*": lambda d, old: lookup(d, "min") * old}
        grid["every class, highest measured in bucket"] = {"*": lambda d, old: lookup(d, "max") * old}
        summary["measured_ratios"] = measured
    for x in (0.25, 0.1, 0.05):
        grid[f"every class, at {x}x"] = {"*": x}
    print("\n### Reprice: decisions that flip\n")
    print("| Section | Target | R | " + " | ".join(grid) + " |")
    print("|---|---|---|" + "---|" * len(grid))
    rp = []
    flip_detail = defaultdict(Counter)
    for (section, arch), cost in sorted(all_decisions.items()):
        row = {}
        for name, ratio in grid.items():
            flips = reprice_flips(cost, ratio)
            row[name] = len(flips)
            if name == "every class, measured ratio":
                for e, after in flips:
                    sub = e["subject"]
                    flip_detail[(section, arch)][(sub["class"], sub["dtype"], "remat -> keep" if not after else "keep -> remat",
                                                  e["phase"])] += 1
        print(f"| {section} | sm_{arch} | {len(cost)} | " + " | ".join(str(row[n]) for n in grid) + " |")
        rp.append({"section": section, "arch": arch, "R": len(cost), "flips": row})
    summary["e2_reprice"] = rp
    summary["reprice_grid"] = list(grid)
    if flip_detail:
        print("\n### What flips at the measured ratios\n")
        print("| Section | Target | Subject class | dtype | Direction | Phase | Decisions |")
        print("|---|---|---|---|---|---|---|")
        fd = []
        for (section, arch), c in sorted(flip_detail.items()):
            for (cls, dtype, direction, phase), n in c.most_common(6):
                print(f"| {section} | sm_{arch} | {cls} | {dtype} | {direction} | {phase} | {n} |")
                fd.append({"section": section, "arch": arch, "class": cls, "dtype": dtype, "direction": direction,
                           "phase": phase, "n": n})
        summary["e2_flip_detail"] = fd

    # ------------------------------------------------------------------ patch interference
    if args.vanilla:
        # The JIT's source hash differs between the two Python trees, so configurations are
        # matched by what was compiled: workload, kernel, target, signature, constants, options.
        import hashlib
        import re

        # Debug locations name the Triton source tree, and the two builds live in different
        # directories. Nothing else in the text depends on where the compiler is installed.
        loc_path = re.compile(rb'loc\("[^"]*/triton/')

        def spec_key(row):
            return json.dumps([row["workload"], row["kernel"], row.get("arch"), row.get("signature"),
                               row.get("constants"), row.get("options")], sort_keys=True, default=str)

        def digests(cs):
            out = defaultdict(list)
            for c in cs:
                f = c.row.get("ttgir")
                if c.ok and f:
                    data = (Path(c.row["run_dir"]) / "ttgir" / f).read_bytes()
                    data = loc_path.sub(b'loc("<triton>/', data)
                    out[spec_key(c.row)].append(hashlib.sha256(data).hexdigest())
            return {k: sorted(v) for k, v in out.items()}

        patched = digests(configs)
        vanilla = digests(load_runs(args.vanilla))
        same = diff = missing = 0
        differing = []
        kernels = set()
        for key, d in patched.items():
            if key not in vanilla:
                missing += len(d)
            elif vanilla[key] == d:
                same += len(d)
                kernels.add(json.loads(key)[1])
            else:
                diff += len(d)
                differing.append(json.loads(key)[:3])
        only_vanilla = sum(len(v) for k, v in vanilla.items() if k not in patched)
        print(f"\n## Patch interference\n\nFinal TTGIR compared with the unpatched build, configuration by "
              f"configuration: {same} identical, {diff} different, {missing} only in the patched run, "
              f"{only_vanilla} only in the unpatched run, across {len(kernels)} kernel functions.\n")
        summary["patch_check"] = {"identical": same, "different": diff, "only_patched": missing,
                                  "only_unpatched": only_vanilla, "kernels": sorted(kernels),
                                  "differing": differing[:20]}

    (out / "e1_e2_summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
