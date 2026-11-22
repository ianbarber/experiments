#!/usr/bin/env python
"""Sanity set: run the lit tests that exercise RemoveLayoutConversions with the event log on.

Each RUN line that names the pass is executed without FileCheck. These are
hand-written IR, so the counts say whether the logger sees the conflicts the
tests were built to create. They are not part of the kernel corpus.

    python run_lit_census.py --triton-root $TRITON_ROOT --triton-opt $BUILD/bin/triton-opt --out runs/census/lit
"""

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

PASS = "tritongpu-remove-layout-conversions"


def run_lines(path: Path):
    text = path.read_text()
    # RUN lines may be continued with a trailing backslash.
    joined = re.sub(r"\\\n\s*//\s*RUN:\s*", " ", text)
    for m in re.finditer(r"//\s*RUN:\s*(.*)", joined):
        cmd = m.group(1).split("|")[0].strip()
        if PASS in cmd and cmd.startswith("triton-opt"):
            yield cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triton-root", required=True)
    ap.add_argument("--triton-opt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.triton_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events = out / "events.jsonl"
    rows = []
    for path in sorted((root / "test").rglob("*.mlir")):
        if PASS not in path.read_text():
            continue
        for i, cmd in enumerate(run_lines(path)):
            rel = path.relative_to(root)
            tag = f"lit|{rel}|run{i}"
            argv = [a for a in shlex.split(cmd.replace("%s", str(path))) if a != "2>&1"]
            argv[0] = args.triton_opt
            env = dict(os.environ, TRITON_LAYOUT_LOG=str(events), TRITON_LAYOUT_TAG=tag)
            proc = subprocess.run(argv, env=env, capture_output=True, text=True)
            rows.append({
                "tag": tag,
                "workload": "lit",
                "kernel": str(rel),
                "command": cmd,
                "status": "ok" if proc.returncode == 0 else "error",
                "error": proc.stderr[-300:] if proc.returncode else "",
            })
    with open(out / "configs.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"{len(rows)} RUN lines, {sum(r['status'] == 'ok' for r in rows)} ok")


if __name__ == "__main__":
    main()
