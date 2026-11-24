#!/usr/bin/env python
"""Tables for the ptxas check: does the register fallback depend on the assembler release?

    python analyze_ptxas.py --versions ../results/ptxas_versions_sm120.json \
        --base ../results/e4_summary.json --base-version 13.5.37 \
        --other /tmp/ptxas12.9/e4_summary.json --other-version 12.9.86 > ../results/ptxas_tables.md

--versions is what ptxas_versions.py wrote: registers and stack frame of each saved build
under each release, nothing launched. --base and --other are E4 summaries of the same
triton_kernels study timed on the device with two releases as Triton's assembler.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SECTIONS = (("triton_kernels", "triton_kernels"), ("inductor", "Inductor"))


def device_studies(summary, section):
    """Studies of one section on the device target, keyed by what was launched (not by label
    for triton_kernels, whose labels are numbered in the order the suite reaches them)."""
    out = {}
    for s in summary["studies"]:
        if s["section"] != section or s["target"] != 120:
            continue
        # Inductor's autotuner can pick another configuration of a kernel from run to run, so
        # its label is not enough either: the specialization is part of the key.
        key = json.dumps([s["label"] if section == "inductor" else s["kernel"], s.get("test"), s.get("constants"),
                          s.get("options")], sort_keys=True, default=str)
        assert key not in out, key
        out[key] = s
    return out


FRAME = 4096


def is_fallback(r):
    return "error" not in r and (r.get("stack_frame") or 0) >= FRAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-version", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--other-version", required=True)
    ap.add_argument("--other-targets", nargs="*", default=[], help="ptxas_versions.py output for targets not launched")
    ap.add_argument("--json", help="also write the numbers behind the tables here")
    args = ap.parse_args()
    pv = json.loads(Path(args.versions).read_text())
    base_summary, other_summary = json.loads(Path(args.base).read_text()), json.loads(Path(args.other).read_text())
    base = device_studies(base_summary, "triton_kernels")
    versions = pv["versions"]

    # ---- registers and stack frame, every saved build, every release
    by_build = defaultdict(dict)
    for r in pv["rows"]:
        by_build[(r["kernel"], r["build"])][r["ptxas"]] = r
    short = {s["label"]: s["short"] for s in base.values()}
    kernels = sorted({k for k, _ in by_build})
    n_err = sum(1 for r in pv["rows"] if "error" in r)
    print("## The same builds under other ptxas releases\n")
    print(f"Final TTGIR of {len(by_build)} builds ({len(kernels)} configurations of `_matmul` and `_p_matmul`, the "
          f"default and each distinct alternative), compiled again from the saved TTGIR with each release as "
          f"Triton's assembler and assembled with Triton's flags for {pv['rows'][0].get('target', 'sm_120a')}. "
          f"Nothing is launched. {n_err} builds failed to assemble.\n")

    def cell(r):
        return "error" if "error" in r else f"{r['registers']} / {r['stack_frame']:,}"

    # The fallback shows as a stack frame of kilobytes: ptxas keeps few values in registers and
    # puts the rest on the stack. 4 KB separates the two groups of builds with room to spare.
    fallback = defaultdict(set)
    for (k, b), per in by_build.items():
        for v, r in per.items():
            if is_fallback(r):
                fallback[v].add((k, b))
    sets = {v: sorted(fallback[v]) for v in versions}
    same = all(sets[v] == sets[versions[0]] for v in versions)
    regs = sorted({by_build[kb][v]["registers"] for v in versions for kb in sets[v]})
    rest = sorted({r["registers"] for r in pv["rows"] if "error" not in r and not is_fallback(r)})
    print(f"Builds with a stack frame of {FRAME // 1024} KB or more per thread: "
          + "; ".join(f"{v}: {len(sets[v])}" for v in versions)
          + (f". They are the same {len(sets[versions[0]])} builds under every release: "
             + ", ".join(f"`{short.get(k, k)}` {'default' if b == 'default' else 'alternative'}" for k, b in sets[versions[0]])
             if same else ". The sets differ between releases")
          + f". Each is given {'/'.join(map(str, regs))} registers; every other build gets {rest[0]} to {rest[-1]}.\n")

    print("Registers per thread / stack frame per thread in bytes, for the configurations where two builds differ "
          "in either under some release:\n")
    print("| Configuration | Build | " + " | ".join(versions) + " |")
    print("|---|---|" + "---|" * len(versions))
    for k in kernels:
        builds = sorted((b for kk, b in by_build if kk == k), key=lambda b: (b != "default", b))
        cells = {b: tuple(cell(by_build[(k, b)][v]) for v in versions) for b in builds}
        if len(set(cells.values())) < 2:
            continue
        seen = set()
        for b in builds:
            if cells[b] in seen:
                continue  # an alternative with the same numbers as one already listed
            seen.add(cells[b])
            print(f"| `{short.get(k, k)}` | {'default' if b == 'default' else 'alternative ' + b} | "
                  + " | ".join(cells[b]) + " |")

    flag = [r for r in pv["rows"] if "without_regalloc_flag" in r and "error" not in r]
    if flag:
        moved = [r for r in flag if r["without_regalloc_flag"].get("registers") != r["registers"]]
        low_moved = [r for r in flag if is_fallback(r) != is_fallback(r["without_regalloc_flag"])]
        print(f"\nTriton always passes `--regAllocOptLevel=2`. Without it, {len(moved)} of {len(flag)} assemblies get a "
              f"different register count, and {len(low_moved)} move into or out of the group with a frame of "
              f"{FRAME // 1024} KB or more.\n")

    # ---- timing on the device under two releases
    print(f"## Timed on the device with ptxas {args.other_version} against {args.base_version}\n")
    print("The same harness, two runs each, the same reporting rule (3% or more, non-overlapping intervals, both "
          "runs, null control under 3%). Rows are the builds either release reports as different from the default; "
          "microseconds are the first run's medians and the percentage is the smaller of the two runs'.\n")

    def timed(s):
        return {v["name"]: v for v in s["timed"]}

    def part(s, name):
        v = timed(s).get(name)
        if v is None:
            return "not timed | | |", None
        d = s["default_ms"]["median"] * 1e3
        word = "" if (v["faster"] or v["slower"]) else " (not reportable)"
        regs = f'{(s.get("default_binary") or {}).get("n_regs")} → {(v.get("binary") or {}).get("n_regs")}'
        return f'{d:,.2f} | {v["ms"]["median"] * 1e3:,.2f} | {v["vs_default_pct"]:+.1f}%{word} | {regs}', v

    def record(s, v, state):
        return None if v is None else {"default_us": s["default_ms"]["median"] * 1e3, "variant_us": v["ms"]["median"] * 1e3,
                                       "pct": v["vs_default_pct"], "reported": state,
                                       "reordered_only": bool(v.get("reordered_only")),
                                       "registers": [(s.get("default_binary") or {}).get("n_regs"),
                                                     (v.get("binary") or {}).get("n_regs")]}

    timing, by_release = [], {}
    for section, title in SECTIONS:
        sb_all, so_all = device_studies(base_summary, section), device_studies(other_summary, section)
        if not so_all:
            continue
        common = sorted(set(sb_all) & set(so_all), key=lambda k: sb_all[k]["label"])
        print(f"### {title}\n")
        print(f"{len(sb_all)} conflicted kernels studied under {args.base_version}, {len(so_all)} under "
              f"{args.other_version}; {len(common)} are the same launches.")
        # a kernel launched with another specialization under the other release is not compared
        only_other = {x["label"]: x for k, x in so_all.items() if k not in sb_all}
        for k, x in sorted(sb_all.items(), key=lambda kv: kv[1]["label"]):
            y = only_other.get(x["label"])
            if k in so_all or y is None:
                continue
            ox, oy = x.get("options") or {}, y.get("options") or {}
            diff = ", ".join(f"{f} {ox.get(f)} then {oy.get(f)}" for f in sorted(set(ox) | set(oy)) if ox.get(f) != oy.get(f))
            both = [abs(v["vs_default_pct"]) for z in (x, y) for v in z["timed"]]
            print(f"Not compared, because it was launched with another specialization under {args.other_version}: "
                  f"`{x['short']}` ({diff or 'constants differ'}); "
                  + (f"its alternatives are within {max(both):.1f}% of the default under both." if both
                     and not any(v["faster"] or v["slower"] for z in (x, y) for v in z["timed"]) else "see each summary."))
        print()
        print(f"| Kernel | Build | {args.base_version}: default, µs | alternative, µs | against default | registers | "
              f"{args.other_version}: default, µs | alternative, µs | against default | registers |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        counts = defaultdict(int)
        for key in common:
            sb, so = sb_all[key], so_all[key]
            for name in sorted(set(timed(sb)) | set(timed(so))):
                pb, vb = part(sb, name)
                po, vo = part(so, name)
                state = tuple("not timed" if v is None else "faster" if v["faster"] else "slower" if v["slower"] else "none"
                              for v in (vb, vo))
                counts[state] += 1
                timing.append({"section": section, "label": sb["label"], "short": sb["short"], "build": name,
                               args.base_version: record(sb, vb, state[0]), args.other_version: record(so, vo, state[1])})
                if set(state) <= {"none", "not timed"}:
                    continue
                print(f"| `{sb['short']}` | {name} | {pb} | {po} |")
        print(f"\nBuilds by what each release reports ({args.base_version} / {args.other_version}): "
              + "; ".join(f"{a} / {b}: {n}" for (a, b), n in sorted(counts.items())) + ".")
        for name, studies in ((args.base_version, sb_all), (args.other_version, so_all)):
            ctl = sorted(abs(x["control"]["vs_default_pct"]) for x in studies.values() if x.get("control"))
            if ctl:
                print(f"Null control under {name}: largest {ctl[-1]:.2f}% over {len(ctl)} kernels"
                      + (f", {sum(c >= 3 for c in ctl)} at 3% or more and left out" if ctl[-1] >= 3 else "") + ".")
        print()
        by_release[section] = {f"{a}/{b}": n for (a, b), n in sorted(counts.items())}

    # ---- targets the card cannot run: assembled only
    if args.other_targets:
        print("\n## Other targets, assembled only\n")
        print("The builds saved by the code-only E4 runs, assembled the same way. Nothing here can be launched on "
              f"this card. A cell is the number of builds with a stack frame of {FRAME // 1024} KB or more, of those "
              "that assembled.\n")
        names = {(s_["target"], s_["label"]): s_["short"] for s_ in base_summary["studies"]
                 if s_["section"] == "triton_kernels"}
        tables = [json.loads(Path(p).read_text()) for p in args.other_targets]
        vs = tables[0]["versions"]
        print("| Target | Configurations | Builds | " + " | ".join(vs) + " |")
        print("|---|---|---|" + "---|" * len(vs))
        notes = []
        for t in tables:
            rows = t["rows"]
            builds = {(r["kernel"], r["build"]) for r in rows}
            cells = []
            for v in vs:
                rv = [r for r in rows if r["ptxas"] == v]
                ok = [r for r in rv if "error" not in r]
                cells.append(f"{sum(map(is_fallback, ok))} of {len(ok)}" if ok else "does not assemble")
            print(f"| sm_{t['arch']} | {len({k for k, _ in builds})} | {len(builds)} | " + " | ".join(cells) + " |")
            for k in sorted({r["kernel"] for r in rows if is_fallback(r)}):
                per = [r for r in rows if r["kernel"] == k and "error" not in r]
                n_builds = len({r["build"] for r in per})
                hit = [r for r in per if is_fallback(r)]
                hit_builds = {r["build"] for r in hit}
                hit_versions = [v for v in vs if any(r["ptxas"] == v for r in hit)]
                which = ("every build, the default and all alternatives" if len(hit_builds) == n_builds and n_builds > 1
                         else "the default only" if hit_builds == {"default"}
                         else f"{len(hit_builds)} of {n_builds} builds" + (", the default among them" if "default" in hit_builds else ", not the default"))
                under = "every release" if hit_versions == vs else ", ".join(hit_versions)
                rest = [r["registers"] for r in per if not is_fallback(r)]
                notes.append(f"- sm_{t['arch']} `{names.get((t['arch'], k), k)}`: {which}, under {under} "
                             f"({min(r['registers'] for r in hit)} registers, frame {min(r['stack_frame'] for r in hit):,} to "
                             f"{max(r['stack_frame'] for r in hit):,} bytes"
                             + (f"; otherwise {min(rest)}" + (f" to {max(rest)}" if max(rest) != min(rest) else "")
                                + " registers" if rest else "") + ")")
        if notes:
            print("\nWhich builds:\n")
            print("\n".join(notes))

    if args.json:
        Path(args.json).write_text(json.dumps({
            "base_version": args.base_version, "other_version": args.other_version, "versions": versions,
            "frame_threshold_bytes": FRAME,
            "large_frame_builds": {v: [list(kb) for kb in sets[v]] for v in versions},
            "same_builds_under_every_release": same,
            "timing": timing,
            "reported_by_release": by_release,
            # the largest gain among alternatives that change an operation or a layout, and among
            # those that only reorder the default's operations, under each release
            "largest_gain_by_kind": {args.base_version: base_summary["headline"].get("largest_gain_by_kind"),
                                     args.other_version: other_summary["headline"].get("largest_gain_by_kind")},
        }, indent=1))


if __name__ == "__main__":
    main()
