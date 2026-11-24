#!/usr/bin/env python
"""Assemble saved TTGIR with several ptxas releases and report registers and stack frame.

The E4 runs save the final TTGIR of the default build and of every alternative. This
compiles each one again from that file, once per ptxas release, with Triton's own backend:
the PTX version follows the assembler and ptxas gets the flags Triton gives it, as in a
normal build. Registers and stack frame are read from the log of that ptxas run. Nothing
is launched.

    python ptxas_versions.py --ttgir runs/e4/tk_sm120/ttgir --ptxas-root ptxas \
        --match _matmul_ _p_matmul_ --out ../results/ptxas_versions_sm120.json

`--ptxas-root` holds one directory per release, named by its version, each with a `ptxas`
binary in it. `--no-regalloc-flag` assembles the same PTX a second time by hand, with
Triton's flags at the pinned commit less `--regAllocOptLevel=2`. `--jobs` runs that many
releases at once.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Runs in a fresh interpreter per release. TRITON_DUMP_PTXAS_LOG makes Triton print what
# `ptxas -v` reported for its own invocation.
CHILD = r"""
import contextlib, io, json, sys
import triton
from triton.backends.compiler import GPUTarget
out = {}
for path in sys.argv[2:]:
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            k = triton.compile(path, target=GPUTarget("cuda", int(sys.argv[1]), 32))
        out[path] = {"ptx": k.asm["ptx"], "log": log.getvalue()}
    except Exception as e:  # an assembler that rejects the PTX raises here
        out[path] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
json.dump(out, sys.__stdout__)
"""


def compile_with(ptxas: Path, arch: int, files, cache: str):
    """PTX and ptxas log of each TTGIR file, as Triton builds it when `ptxas` is its assembler."""
    env = dict(os.environ, TRITON_PTXAS_PATH=str(ptxas), TRITON_CACHE_DIR=cache, TRITON_ALWAYS_COMPILE="1",
               TRITON_DUMP_PTXAS_LOG="1")
    for k in list(env):
        if k.startswith("TRITON_LAYOUT_"):
            del env[k]  # the experiment's override switches stay off
    # run in the empty cache directory: a checkout named `triton` in the working directory
    # would otherwise shadow the installed package
    r = subprocess.run([sys.executable, "-c", CHILD, str(arch), *map(str, files)], env=env, cwd=cache,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout[r.stdout.index("{"):])


def parse_log(log: str):
    regs = re.search(r"Used (\d+) registers", log)
    frame = re.search(r"(\d+) bytes stack frame", log)
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
    return {"registers": int(regs.group(1)) if regs else None,
            "stack_frame": int(frame.group(1)) if frame else None,
            "spill_stores": int(spill.group(1)) if spill else None,
            "spill_loads": int(spill.group(2)) if spill else None}


def assemble_without_regalloc_flag(ptxas: Path, ptx: str):
    """ptxas by hand: Triton's flags at the pinned commit (fp fusion is off by default there,
    so it passes --fmad=false), without --regAllocOptLevel=2."""
    target = re.search(r"^\.target\s+([A-Za-z0-9_]+)", ptx, re.MULTILINE).group(1)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "k.ptx"
        src.write_text(ptx)
        r = subprocess.run([str(ptxas), "-lineinfo", "--fmad=false", "-v", f"--gpu-name={target}", str(src),
                            "-o", str(Path(tmp) / "k.o")], capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": (r.stdout + r.stderr).strip()[-300:]}
    return parse_log(r.stdout + r.stderr)


def one_release(version, args, files):
    ptxas = (Path(args.ptxas_root) / version / "ptxas").resolve()
    with tempfile.TemporaryDirectory() as cache:
        built = compile_with(ptxas, args.arch, files, cache)
    rows = []
    for f in files:
        kernel, build = f.stem.split("__", 1)
        b = built[str(f)]
        row = {"ptxas": version, "kernel": kernel, "build": build}
        if "error" in b:
            row["error"] = b["error"]
        else:
            m = re.search(r"^\.version\s+(\S+)", b["ptx"], re.MULTILINE)
            t = re.search(r"^\.target\s+([A-Za-z0-9_]+)", b["ptx"], re.MULTILINE)
            row.update(ptx_version=m.group(1) if m else None, target=t.group(1) if t else None, **parse_log(b["log"]))
            if args.no_regalloc_flag:
                row["without_regalloc_flag"] = assemble_without_regalloc_flag(ptxas, b["ptx"])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttgir", required=True, help="directory of <kernel>__<build>.ttgir files")
    ap.add_argument("--ptxas-root", required=True)
    ap.add_argument("--match", nargs="*", default=[], help="keep files whose name starts with one of these")
    ap.add_argument("--arch", type=int, default=120)
    ap.add_argument("--no-regalloc-flag", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(p.resolve() for p in Path(args.ttgir).glob("*.ttgir")
                   if not args.match or any(p.name.startswith(m) for m in args.match))
    versions = sorted((p.name for p in Path(args.ptxas_root).iterdir() if (p / "ptxas").exists()),
                      key=lambda v: tuple(map(int, v.split("."))))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        per_release = list(pool.map(lambda v: one_release(v, args, files), versions))
    rows = [row for rows in per_release for row in rows]
    Path(args.out).write_text(
        json.dumps({"arch": args.arch, "versions": versions})[:-1] + ', "rows": [\n'
        + ",\n".join(json.dumps(row) for row in rows) + "\n]}\n")


if __name__ == "__main__":
    main()
