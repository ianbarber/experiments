#!/usr/bin/env python
"""Reduce the raw event logs to what the tables are computed from, and pack them.

    python pack_evidence.py --runs $RUNS/census --arch 120 --out ../results/census_evidence_sm120.tar.xz

The raw logs are several hundred MB per target, mostly per-stage censuses and repeated
layout strings. This keeps every configuration row and, per configuration, the events the
analysis reads (conflicts, prices, decisions, the final-TTGIR census, what the lowering
did), with layout strings replaced by a per-file dictionary index. Unpacked, the archive is
a runs directory `analyze_census.py --runs` reads directly (`layoutlab.events.load_runs`
expands the index).
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

KEEP = {"conflict", "convert_cost", "remat_decision", "remat_skip", "uncosted_hoist", "pass_begin", "pass_end", "lowered",
        "lowered_smem_detail"}
DROP = {"seq", "users", "src_op", "index", "func", "classifier_mismatch", "force_warp_shuffle", "priced_free",
        "priced_bytes", "assumed_cost"}


def reduce_file(path: Path, arch=None):
    table, lines = {}, []
    ops = {}  # per compile: op identity (an address in the raw log) -> small integer
    suffix = f"|sm{arch}" if arch is not None else None

    def intern(s):
        return table.setdefault(s, len(table))

    def slim(d, tag):
        out = {}
        for k, v in d.items():
            if k in DROP:
                continue
            if k == "op_id":
                seen = ops.setdefault(tag, {})
                out[k] = seen.setdefault(v, len(seen))
            elif k in ("src", "dst", "current") and isinstance(v, str):
                out[k] = intern(v)
            elif k == "candidates":
                out[k] = [intern(x) for x in v]
            elif k in ("subject",):
                out[k] = slim(v, tag)
            elif k == "new_conversions":
                out[k] = [slim(x, tag) for x in v]
            else:
                out[k] = v
        return out

    with open(path) as fh:
        for line in fh:
            if suffix is not None and suffix not in line:
                continue
            e = json.loads(line)
            kind = e["event"]
            if kind == "census":
                if e["stage"] != "final_ttgir":
                    continue
            elif kind not in KEEP:
                continue
            lines.append(json.dumps(slim(e, e.get("tag")), separators=(",", ":")))
    header = json.dumps({"layouts": [k for k, _ in sorted(table.items(), key=lambda kv: kv[1])]}, separators=(",", ":"))
    return (header + "\n" + "\n".join(lines) + "\n").encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--arch", type=int, default=None, help="keep only this target")
    args = ap.parse_args()
    root = Path(args.runs)
    with tarfile.open(args.out, "w:xz", preset=9) as tar:
        for d in sorted(root.glob(args.pattern)):
            if not (d / "configs.jsonl").exists():
                continue
            if args.arch is not None:
                # keep one target: rows and events are filtered by the tag's suffix
                suffix = f"|sm{args.arch}"
                rows = [l for l in (d / "configs.jsonl").read_text().splitlines() if suffix in l]
                if not rows:
                    continue
            configs = (d / "configs.jsonl").read_bytes()
            if args.arch is not None:
                configs = ("\n".join(rows) + "\n").encode()
            events = reduce_file(d / "events.jsonl", args.arch) if (d / "events.jsonl").exists() else b""
            for name, data in (("configs.jsonl", configs), ("events.reduced.jsonl", events)):
                info = tarfile.TarInfo(f"{d.name}/{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            if (d / "session.json").exists():
                tar.add(d / "session.json", arcname=f"{d.name}/session.json")
    print(args.out, round(Path(args.out).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
