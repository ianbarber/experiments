"""Sweeps of the Triton tutorials' own block sizes and dtypes.

Each tutorial file is executed one top-level statement at a time, with the benchmark
runner disabled, so its kernels, wrappers and autotune configurations are the
tutorial's own. The sweep then calls the tutorial's wrappers for each dtype. Every
compilation is recorded by `instrument.install`.
"""

from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path

import torch

from .instrument import attempt

TRITON_ROOT = Path(os.environ.get("TRITON_ROOT", "")).expanduser()

FILES = {
    "tut02": "02-fused-softmax.py",
    "tut03": "03-matrix-multiplication.py",
    "tut05": "05-layer-norm.py",
    "tut06": "06-fused-attention.py",
    "tut08": "08-grouped-gemm.py",
    "tut09": "09-persistent-matmul.py",
    "tut10": "10-block-scaled-matmul.py",
}

FLOATS = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _stub_profiler():
    """Tutorials 09 and 10 import the profiler; a build without it still has to load them."""
    try:
        import triton.profiler  # noqa: F401
        return
    except Exception:
        pass
    import contextlib

    mod = types.ModuleType("triton.profiler")
    for name in ("start", "finalize", "activate", "deactivate", "enter_scope", "exit_scope"):
        setattr(mod, name, lambda *a, **k: None)
    mod.scope = lambda *a, **k: contextlib.nullcontext()
    sys.modules["triton.profiler"] = mod
    import triton
    triton.profiler = mod


def load(name: str) -> dict:
    """Execute a tutorial file statement by statement and return its globals."""
    import triton.testing

    triton.testing.Mark.run = lambda self, *a, **k: None
    _stub_profiler()
    path = TRITON_ROOT / "python" / "tutorials" / FILES[name]
    tree = ast.parse(path.read_text(), filename=str(path))
    g = {"__name__": f"layoutlab_{name}", "__file__": str(path)}
    for node in tree.body:
        code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
        attempt(f"{name}: top-level statement at line {node.lineno}", exec, code, g)
    return g


def _reset_autotune(g: dict):
    """Autotune keys do not include every constexpr, so force a full sweep on each call."""
    from triton.runtime.autotuner import Autotuner
    for v in g.values():
        if isinstance(v, Autotuner):
            v.cache.clear()


def _capability() -> int:
    return torch.cuda.get_device_capability()[0]


def tut02(g):
    for dname, dtype in FLOATS.items():
        for n_cols in (256, 512, 1024, 2048, 4096, 8192, 16384):
            x = torch.randn(64, n_cols, device="cuda", dtype=dtype)
            attempt(f"tut02 softmax {dname} n_cols={n_cols}", g["softmax"], x)


def tut03(g):
    for dname, dtype in FLOATS.items():
        for activation in ("", "leaky_relu"):
            a = torch.randn((512, 512), device="cuda", dtype=dtype)
            b = torch.randn((512, 512), device="cuda", dtype=dtype)
            _reset_autotune(g)
            attempt(f"tut03 matmul {dname} activation={activation!r}", g["matmul"], a, b, activation)


def tut05(g):
    for dname, dtype in FLOATS.items():
        for n in (1024, 2048, 4096, 8192, 16384):
            attempt(f"tut05 layer_norm {dname} N={n}", g["test_layer_norm"], 128, n, dtype)


def tut06(g):
    cap = _capability()
    for head_dim in (64, 128):
        for causal in (False, True):
            ws_options = [False]
            if cap == 10 or (cap == 9 and not causal):
                ws_options.append(True)
            for ws in ws_options:
                # The kernel is written for fp16 (and fp8 forward); bf16 and fp32 inputs are
                # rejected by the frontend, so there is no dtype sweep beyond the tutorial's own.
                _reset_autotune(g)
                attempt(f"tut06 attention fwd+bwd fp16 head_dim={head_dim} causal={causal} ws={ws}", g["test_op"], 1, 2,
                        1024, head_dim, causal, ws, "bwd", "triton-fp16", torch.float16)
                if g.get("TORCH_HAS_FP8"):
                    _reset_autotune(g)
                    attempt(f"tut06 attention fwd fp8 head_dim={head_dim} causal={causal} ws={ws}", g["test_op"], 1, 2,
                            1024, head_dim, causal, ws, "fwd", "triton-fp8", torch.float16)


def tut08(g):
    sizes = [1024, 512, 256, 128]
    dtypes = {"fp16": torch.float16}
    if hasattr(torch, "float8_e4m3fn"):
        dtypes["fp8"] = torch.float8_e4m3fn
    for dname, dtype in dtypes.items():
        group_a = [torch.rand((s, s), device="cuda", dtype=torch.float16).to(dtype) for s in sizes]
        group_b = [torch.rand((s, s), device="cuda", dtype=torch.float16).to(dtype) for s in sizes]
        group_bt = [b.T.contiguous() for b in group_b]
        if dname == "fp16":
            _reset_autotune(g)
            attempt(f"tut08 grouped gemm {dname}", g["group_gemm_fn"], group_a, group_b)
        if g["supports_tma"]():
            _reset_autotune(g)
            attempt(f"tut08 grouped gemm tma {dname}", g["group_gemm_tma_fn"], group_a, group_bt)


def tut09(g):
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}
    if hasattr(torch, "float8_e4m3fn"):
        dtypes["fp8"] = torch.float8_e4m3fn
    for dname, dtype in dtypes.items():
        _reset_autotune(g)
        attempt(f"tut09 persistent matmul {dname}", g["validate"], 2048, 2048, 512, dtype)


def tut10(g):
    if not g["supports_block_scaling"]():
        return
    for fmt in ("nvfp4", "mxfp4", "mxfp8", "mixed"):
        for clc in (False, True):
            attempt(f"tut10 block scaled matmul {fmt} clc={clc}", g["validate_block_scaled"], 2048, 2048, 2048, fmt, clc)


SWEEPS = {"tut02": tut02, "tut03": tut03, "tut05": tut05, "tut06": tut06, "tut08": tut08, "tut09": tut09, "tut10": tut10}


def run(name: str):
    g = load(name)
    SWEEPS[name](g)
