"""Load census runs (configs.jsonl + events.jsonl) and reduce them to per-configuration records.

A run directory may hold `events.reduced.jsonl` instead of `events.jsonl`: the form
`pack_evidence.py` writes, with layout strings replaced by an index into a table on the
first line.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ARCHS = (80, 90, 100, 120)


def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_reduced(path: Path):
    """Events from a file written by pack_evidence.py, with layout strings restored."""
    with open(path) as fh:
        layouts = json.loads(fh.readline())["layouts"]

        def expand(d):
            for k in ("src", "dst", "current"):
                if isinstance(d.get(k), int):
                    d[k] = layouts[d[k]]
            if "candidates" in d:
                d["candidates"] = [layouts[i] for i in d["candidates"]]
            if isinstance(d.get("subject"), dict):
                expand(d["subject"])
            for x in d.get("new_conversions", ()):
                expand(x)
            return d

        return [expand(json.loads(line)) for line in fh if line.strip()]


def layout_kind(enc: str) -> str:
    """Short name for an encoding string as MLIR prints it."""
    m = re.match(r"#ttg\.(\w+)<", enc)
    if not m:
        return enc[:20]
    kind = m.group(1)
    if kind == "slice":
        inner = re.search(r"parent = #ttg\.(\w+)<", enc)
        return f"slice({inner.group(1) if inner else '?'})"
    if kind == "dot_op":
        inner = re.search(r"parent = #ttg\.(\w+)<", enc)
        return f"dot_op({inner.group(1) if inner else '?'})"
    return kind


def family(workload: str) -> str:
    """Corpus section a workload belongs to."""
    if workload.startswith("tut"):
        return "tutorials"
    if workload.startswith("inductor"):
        return "inductor"
    if workload == "lit":
        return "lit"
    if workload == "tk":
        return "triton_kernels"
    return "other"


class Config:
    """One kernel configuration on one target: its row and its events."""

    __slots__ = ("row", "events")

    def __init__(self, row):
        self.row = row
        self.events = []

    @property
    def tag(self):
        return self.row["tag"]

    @property
    def ok(self):
        return self.row.get("status") == "ok"

    @property
    def instrumented(self):
        """Compiled under the fpsan instrumentation mode, which adds a run with remat splitting off."""
        return any(e["event"] == "pass_begin" and e.get("disable_remat_splitting") for e in self.events)

    def invocation_labels(self):
        """Map the raw invocation index to its place in the pipeline.

        make_ttgir adds the pass four times. On the Blackwell pipeline OptimizePartitionWarps
        also runs it, inside its own pass manager, between the third and the fourth; and the
        fpsan mode adds one at the end with remat splitting disabled.
        """
        begins = sorted((e["inv"], bool(e.get("disable_remat_splitting"))) for e in self.events
                        if e["event"] == "pass_begin")
        labels = {}
        plain = [inv for inv, fpsan in begins if not fpsan]
        for inv, fpsan in begins:
            if fpsan:
                labels[inv] = "fpsan"
            elif inv <= 3:
                labels[inv] = str(inv)
            elif inv == plain[-1]:
                labels[inv] = "4"
            else:
                labels[inv] = "nested"
        return labels

    @property
    def section(self):
        sec = family(self.row["workload"])
        return sec + "_fpsan" if self.instrumented else sec

    def of(self, kind):
        return [e for e in self.events if e["event"] == kind]

    def conflicts(self):
        return self.of("conflict")

    def final_census(self):
        return [e for e in self.events if e["event"] == "census" and e["stage"] == "final_ttgir"]

    def decisions(self):
        """Cost comparisons, with re-evaluations of the same op at the same costs collapsed."""
        seen = set()
        out = []
        for e in self.of("remat_decision"):
            s = e["subject"]
            key = (e["inv"], e["phase"], s.get("op_id"), s["src"], s["dst"], e["convert_cost"], e["remat_cost"],
                   e["reason"])
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out


def load_runs(root: str | Path, pattern: str = "*"):
    """Yield Config objects for every run directory under root matching pattern."""
    root = Path(root)
    configs = []
    for d in sorted(root.glob(pattern)):
        if not (d / "configs.jsonl").exists():
            continue
        by_tag = {}
        for row in read_jsonl(d / "configs.jsonl"):
            row["run_dir"] = str(d)
            by_tag[row["tag"]] = Config(row)
        raw = d / "events.jsonl"
        events = read_jsonl(raw) if raw.exists() or not (d / "events.reduced.jsonl").exists() \
            else read_reduced(d / "events.reduced.jsonl")
        for e in events:
            c = by_tag.get(e.get("tag"))
            if c is not None:
                c.events.append(e)
        configs.extend(by_tag.values())
    return configs


def frontend_rejected(c: Config) -> bool:
    """A configuration the kernel's own static checks reject never reaches the pass."""
    return (not c.ok) and not c.events


def group(items, key):
    out = defaultdict(list)
    for it in items:
        out[key(it)].append(it)
    return out
