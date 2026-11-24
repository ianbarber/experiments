"""Turn a layout attribute as MLIR prints it into the Gluon layout object that produces it.

This lets the E3 microbenchmarks time exactly the conversions the census found:
`#ttg.blocked<{sizePerThread = [1, 4], ...}>` becomes `ttgl.BlockedLayout([1, 4], ...)`.
"""

from __future__ import annotations

import ast
import re

from triton.experimental.gluon import language as ttgl


def _split_top(text: str):
    """Split `a = 1, b = [1, 2], c = #x<{..}>` on top-level commas."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "[{<(":
            depth += 1
        elif ch in "]}>)":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _fields(body: str) -> dict:
    out = {}
    for part in _split_top(body):
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _lit(text: str):
    return ast.literal_eval(text)


def parse(enc: str, shape=None):
    """`enc` is e.g. '#ttg.blocked<{...}>'. `shape` is only needed for linear layouts."""
    m = re.match(r"#ttg\.(\w+)<\{(.*)\}>$", enc.strip(), re.S)
    if not m:
        raise ValueError(f"cannot parse layout: {enc}")
    kind, body = m.group(1), m.group(2)
    f = _fields(body)
    if kind == "blocked":
        extra = {}
        if "CGALayout" in f:
            extra["cga_layout"] = _lit(f["CGALayout"])
        return ttgl.BlockedLayout(_lit(f["sizePerThread"]), _lit(f["threadsPerWarp"]), _lit(f["warpsPerCTA"]),
                                  _lit(f["order"]), **extra)
    if kind == "slice":
        dim = int(f["dim"])
        parent_shape = None
        if f["parent"].startswith("#ttg.linear") and shape is not None:
            # The parent has one more dimension. Its extent along the sliced dimension is
            # whatever its bases span; the others are the tensor's own.
            pf = _fields(re.match(r"#ttg\.linear<\{(.*)\}>$", f["parent"], re.S).group(1))
            bases = _lit(pf["register"]) + _lit(pf["lane"]) + _lit(pf["warp"]) + _lit(pf["block"])
            extent = 1 + sum(b[dim] for b in bases)
            parent_shape = list(shape[:dim]) + [extent] + list(shape[dim:])
        return ttgl.SliceLayout(dim, parse(f["parent"], parent_shape))
    if kind == "nvidia_mma":
        extra = {}
        if "CGALayout" in f:
            extra["cga_layout"] = _lit(f["CGALayout"])
        return ttgl.NVMMADistributedLayout([int(f["versionMajor"]), int(f["versionMinor"])], _lit(f["warpsPerCTA"]),
                                           _lit(f["instrShape"]), **extra)
    if kind == "dot_op":
        return ttgl.DotOperandLayout(int(f["opIdx"]), parse(f["parent"]), int(f.get("kWidth", 0)))
    if kind == "linear":
        if shape is None:
            raise ValueError("linear layout needs the tensor shape")
        return ttgl.DistributedLinearLayout(_lit(f["register"]), _lit(f["lane"]), _lit(f["warp"]), _lit(f["block"]),
                                            list(shape))
    raise ValueError(f"unsupported layout kind {kind}: {enc}")


def num_warps(layout, shape) -> int:
    n = 1
    for w in ttgl._layouts.warps_per_cta(layout, list(shape)):
        n *= w
    return int(n)
