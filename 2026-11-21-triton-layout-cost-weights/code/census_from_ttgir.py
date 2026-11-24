#!/usr/bin/env python
"""Census of the conversions in saved TTGIR files.

    python census_from_ttgir.py --runs $RUNS/e4

The E4 runs on the device save the final TTGIR of the default build and of every
alternative with different code. This parses each file back into a module and asks the
instrumented compiler for the same census the census sweep logs, so the conversions left
in an alternative can be compared with the default's. Writes `ttgir_census.json` next to
each `ttgir/` directory.

It also writes `ttgir_canonical.json`, two digests per file. `program` is the module with
debug locations removed and SSA values renamed in order of first appearance: the harness
compares TTGIR as text, so two builds that differ only in the names of their values count
as different code there, and the analysis uses this digest to treat them as the same
program. `operations` is the sorted list of operations with value names removed and layout
aliases written out: two builds with the same digest contain the same operations on the
same layouts, in a different order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

_VALUE = re.compile(r"%[A-Za-z0-9_.$-]+")


def strip_locations(text):
    """Remove debug locations from printed MLIR: every ` loc(...)` and every `#loc` definition.

    Locations sit at the end of an operation, but also inside a function signature after
    each argument, so the closing parenthesis has to be matched (quotes may hold any text)."""
    out = []
    for line in text.splitlines():
        if line.startswith("#loc"):
            continue
        while True:
            i = line.find(" loc(")
            if i < 0:
                break
            j, depth, quoted = i + 5, 1, False
            while j < len(line) and depth:
                ch = line[j]
                if quoted:
                    if ch == "\\":
                        j += 1
                    elif ch == '"':
                        quoted = False
                elif ch == '"':
                    quoted = True
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                j += 1
            line = line[:i] + line[j:]
        out.append(line)
    return "\n".join(out) + "\n"


def canonical_digest(text):
    """Digest of a TTGIR module up to debug locations and the names of SSA values."""
    text = strip_locations(text)
    names = {}
    text = _VALUE.sub(lambda m: names.setdefault(m.group(0), f"%v{len(names)}"), text)
    return hashlib.sha256(text.encode()).hexdigest()


_ALIAS_DEF = re.compile(r"(#[A-Za-z_]+\d*) = (.*)$")
_ALIAS_USE = re.compile(r"#[A-Za-z_]+\d*\b")


def operations_digest(text):
    """Digest of the operations of a TTGIR module regardless of their order.

    Layout aliases (`#blocked3`, `#linear7`) are numbered in order of first use, so they are
    replaced by their definitions before comparing; SSA value names are dropped."""
    alias, body = {}, []
    for line in (ln.strip() for ln in strip_locations(text).splitlines()):
        if not line:
            continue
        m = _ALIAS_DEF.match(line)
        if m:
            alias[m.group(1)] = m.group(2)
        else:
            body.append(line)

    def expand(t, depth=0):
        return _ALIAS_USE.sub(
            lambda m: expand(alias[m.group(0)], depth + 1) if m.group(0) in alias and depth < 8 else m.group(0), t)

    ops = sorted(re.sub(r"%[\w.#:$-]+", "%", expand(line)) for line in body)
    return hashlib.sha256("\n".join(ops).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    args = ap.parse_args()

    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend

    backend = make_backend(GPUTarget("cuda", 120, 32))
    for d in sorted(Path(args.runs).glob("*/ttgir")):
        out = {}
        files = sorted(d.glob("*.ttgir"))
        texts = {f.name: f.read_text() for f in files}
        (d.parent / "ttgir_canonical.json").write_text(json.dumps(
            {name: {"program": canonical_digest(t), "operations": operations_digest(t)} for name, t in texts.items()},
            indent=0))
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "events.jsonl"
            os.environ["TRITON_LAYOUT_LOG"] = str(log)
            for f in files:
                os.environ["TRITON_LAYOUT_TAG"] = f.name
                context = ir.context()
                ir.load_dialects(context)
                backend.load_dialects(context)
                try:
                    module = ir.parse_mlir_module(str(f), context)
                    module.log_layout_census("final_ttgir")
                except Exception as e:  # noqa: BLE001
                    out[f.name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
                    continue
                out.setdefault(f.name, {})
            if log.exists():
                for line in open(log):
                    e = json.loads(line)
                    if e["event"] != "census":
                        continue
                    key = e["class"] if e["class"] != "shared_memory" else f'shared_memory:{e["barrier"]}'
                    key += "@loop" if e["loop_depth"] > 0 else ""
                    c = out.setdefault(e["tag"], {})
                    c[key] = c.get(key, 0) + 1
            os.environ.pop("TRITON_LAYOUT_LOG", None)
            os.environ.pop("TRITON_LAYOUT_TAG", None)
        (d.parent / "ttgir_census.json").write_text(json.dumps(out, indent=0))
        n = Counter("error" if "error" in v else "ok" for v in out.values())
        print(d.parent.name, dict(n))


if __name__ == "__main__":
    main()
