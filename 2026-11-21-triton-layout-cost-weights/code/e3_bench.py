#!/usr/bin/env python
"""E3: what one convert_layout costs on the device, per lowering class.

For each case (shape, dtype, source layout, destination layout) a Gluon kernel loads a
tensor once and then loops `n_iter` times over

    x = x + c                      (source layout)
    y = convert_layout(x, DST)
    y = y + c                      (destination layout)
    x = convert_layout(y, SRC)

Baselines run the same loop with no conversions and 2 or 4 elementwise adds per
iteration, once with the tensor in the source layout and once in the destination
layout. With the GPU saturated (many more CTAs than fit at once):

    add_src    = (t_base4_src - t_base2_src) / 2
    add_dst    = (t_base4_dst - t_base2_dst) / 2
    conversion = (t_convert - t_base2_src + add_src - add_dst) / 2

i.e. the convert loop minus the loop overhead, one add in each layout, halved: the mean
of SRC->DST and DST->SRC. Everything is SM time per CTA per iteration,
t * n_SM / (grid * n_iter). For integer dtypes the filler op is `x ^ (x << 1)`, two
instructions, so one "add" is half of it.

Routes: "default" is whatever the lowering picks. "shuffle" adds force_warp_shuffle to
both conversions (only legal inside a warp). "smem" compiles with
TRITON_LAYOUT_DISABLE_WARP_SHUFFLE=1 so shuffle-by-default conversions use shared memory.

    python e3_bench.py --cases cases.json --out runs/e3 [--runs 25]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
from triton.experimental import gluon  # noqa: E402
from triton.experimental.gluon import language as ttgl  # noqa: E402

from layoutlab import gluon_layouts  # noqa: E402

DTYPES = {
    "f16": (torch.float16, ttgl.float16),
    "bf16": (torch.bfloat16, ttgl.bfloat16),
    "f32": (torch.float32, ttgl.float32),
    "f64": (torch.float64, ttgl.float64),
    "i8": (torch.int8, ttgl.int8),
    "i32": (torch.int32, ttgl.int32),
    "i64": (torch.int64, ttgl.int64),
}


@gluon.jit(do_not_specialize=["n_iter"])
def cvt_loop(x_ptr, y_ptr, n_iter, M: ttgl.constexpr, N: ttgl.constexpr, SRC: ttgl.constexpr, DST: ttgl.constexpr,
             N_CVT: ttgl.constexpr, N_OPS: ttgl.constexpr, IS_FLOAT: ttgl.constexpr):
    if N == 0:
        offs = ttgl.arange(0, M, layout=SRC)
    else:
        offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, SRC))[:, None]
        offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, SRC))[None, :]
        offs = offs_m * N + offs_n
    x = ttgl.load(x_ptr + offs)
    for _ in range(n_iter):
        for _ in ttgl.static_range(N_OPS):
            if IS_FLOAT:
                x = x + 0.001953125
            else:
                x = x ^ (x << 1)
        for _ in ttgl.static_range(N_CVT):
            if IS_FLOAT:
                x = x + 0.001953125
            else:
                x = x ^ (x << 1)
            y = ttgl.convert_layout(x, DST)
            if IS_FLOAT:
                y = y + 0.001953125
            else:
                y = y ^ (y << 1)
            x = ttgl.convert_layout(y, SRC)
    ttgl.store(y_ptr + offs, x)


def reference(x, n_ops_total, is_float):
    ref = x.clone()
    for _ in range(n_ops_total):
        ref = ref + 0.001953125 if is_float else ref ^ (ref << 1)
    return ref


class ClockSampler:
    """Reads the GPU core clock while a measurement runs."""

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.current.graphics,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
                mhz, temp, power = [float(v) for v in out.strip().split(",")]
                self.samples.append((mhz, temp, power))
            except Exception:
                pass
            self._stop.wait(0.25)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self):
        if not self.samples:
            return {}
        a = np.array(self.samples)
        return {"clock_mhz_median": float(np.median(a[:, 0])), "clock_mhz_min": float(a[:, 0].min()),
                "temp_c_max": float(a[:, 1].max()), "power_w_median": float(np.median(a[:, 2])), "samples": len(a)}


def compile_variant(case, n_cvt, n_ops, route, workdir: Path, events: Path | None, in_dst: bool = False):
    """Compile one loop variant and return (CompiledKernel from IR, tag, warps, conversions left)."""
    shape = case["shape"]
    if len(shape) > 2:
        raise ValueError("the harness handles 1-D and 2-D tensors only")
    m, n = (shape[0], 0) if len(shape) == 1 else shape
    src = gluon_layouts.parse(case["src"], shape)
    dst = gluon_layouts.parse(case["dst"], shape)
    torch_dtype, _ = DTYPES[case["dtype"]]
    is_float = torch_dtype.is_floating_point
    # A layout that repeats data across warps reports fewer warps than the kernel has.
    warps = max(gluon_layouts.num_warps(src, shape), gluon_layouts.num_warps(dst, shape))
    if in_dst:
        src = dst
    x = torch.zeros(shape, dtype=torch_dtype, device="cuda")
    y = torch.zeros_like(x)

    env_before = {k: os.environ.get(k) for k in ("TRITON_LAYOUT_DISABLE_WARP_SHUFFLE", "TRITON_LAYOUT_LOG", "TRITON_LAYOUT_TAG")}
    tag = f'e3|{case["name"]}|cvt{n_cvt}_ops{n_ops}{"_dst" if in_dst else ""}|{route}'
    try:
        os.environ.pop("TRITON_LAYOUT_LOG", None)
        if route == "smem":
            os.environ["TRITON_LAYOUT_DISABLE_WARP_SHUFFLE"] = "1"
        jit_kernel = cvt_loop.warmup(x, y, 7, m, n, src, dst, n_cvt, n_ops, is_float, grid=(1, ), num_warps=warps)
        ttgir = jit_kernel.asm["ttgir"]
        if route == "shuffle":
            lines = []
            for line in ttgir.splitlines():
                if "ttg.convert_layout" in line and "force_warp_shuffle" not in line:
                    line = line.replace(" : ", " {force_warp_shuffle} : ", 1)
                lines.append(line)
            ttgir = "\n".join(lines) + "\n"
        path = workdir / (re.sub(r"[^A-Za-z0-9_.-]", "_", tag) + ".ttgir")
        path.write_text(ttgir)
        if events is not None:
            os.environ["TRITON_LAYOUT_LOG"] = str(events)
            os.environ["TRITON_LAYOUT_TAG"] = tag
        prev = triton.knobs.compilation.always_compile
        triton.knobs.compilation.always_compile = True
        try:
            ck = triton.compile(str(path))
        finally:
            triton.knobs.compilation.always_compile = prev
    finally:
        for k, v in env_before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    n_cvt_ops = ttgir.count("ttg.convert_layout")
    return ck, tag, warps, n_cvt_ops


def time_runs(runners, n_runs, warmup):
    """Interleaved timing. Returns {name: [seconds]}."""
    times = {name: [] for name in runners}
    for i in range(warmup + n_runs):
        for name, fn in runners.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= warmup:
                times[name].append(dt)
    return times


def boot_ci(fn, arrays, n_boot=2000, seed=42):
    """Bootstrap 95% interval of fn(*arrays), resampling run indices jointly (runs are interleaved)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(*[a[idx] for a in arrays])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def run_case(case, args, workdir, events, n_sm):
    shape = case["shape"]
    torch_dtype, _ = DTYPES[case["dtype"]]
    is_float = torch_dtype.is_floating_point
    routes = case.get("routes", ["default"])
    result = {"case": case, "routes": {}}

    torch.manual_seed(42)
    if is_float:
        x = torch.randn(shape, dtype=torch_dtype, device="cuda")
    else:
        info = torch.iinfo(torch_dtype)
        x = torch.randint(info.min, info.max, shape, dtype=torch_dtype, device="cuda")
    y = torch.zeros_like(x)

    variants = {}
    base2, _, warps, _ = compile_variant(case, 0, 2, "default", workdir, None)
    base4, _, _, _ = compile_variant(case, 0, 4, "default", workdir, None)
    variants["base2"] = (base2, 2)
    variants["base4"] = (base4, 4)
    try:
        base2d, _, _, _ = compile_variant(case, 0, 2, "default", workdir, None, in_dst=True)
        base4d, _, _, _ = compile_variant(case, 0, 4, "default", workdir, None, in_dst=True)
        variants["base2d"] = (base2d, 2)
        variants["base4d"] = (base4d, 4)
        result["dst_baseline"] = True
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, KeyboardInterrupt):
            raise
        # Some layouts cannot be loaded into directly. Fall back to assuming an add costs
        # the same in both layouts, and say so.
        result["dst_baseline"] = False
        result["dst_baseline_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    tags = {}
    for route in routes:
        try:
            ck, tag, _, n_cvt_ops = compile_variant(case, 1, 0, route, workdir, events)
        except BaseException as e:  # noqa: BLE001
            result["routes"][route] = {"status": "compile_error", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            continue
        if n_cvt_ops != 2:
            result["routes"][route] = {"status": "folded", "error": f"{n_cvt_ops} convert_layout ops left, expected 2"}
            continue
        variants[route] = (ck, 2)
        tags[route] = tag

    # correctness on a single CTA, three iterations
    for name, (ck, ops_per_iter) in list(variants.items()):
        y.zero_()
        ck[(1, 1, 1)](x, y, 3)
        torch.cuda.synchronize()
        ref = reference(x, 3 * ops_per_iter, is_float)
        if not torch.equal(y, ref):
            bad = int((y != ref).sum())
            if name in ("base2", "base4"):
                raise RuntimeError(f"baseline {name} is wrong for {case['name']}: {bad} elements differ")
            if name in ("base2d", "base4d"):
                variants.pop("base2d", None)
                variants.pop("base4d", None)
                result["dst_baseline"] = False
                result["dst_baseline_error"] = f"{name} output wrong: {bad} elements differ"
                continue
            result["routes"][name] = {"status": "wrong_output", "mismatched_elements": bad}
            del variants[name]

    grid = n_sm * args.grid_mult
    # pick n_iter so the slowest variant takes about target_ms
    n_iter = 64
    while True:
        slow = 0.0
        for name, (ck, _) in variants.items():
            ck[(grid, 1, 1)](x, y, n_iter)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ck[(grid, 1, 1)](x, y, n_iter)
            torch.cuda.synchronize()
            slow = max(slow, time.perf_counter() - t0)
        if slow * 1e3 >= args.target_ms / 4 or n_iter >= (1 << 22):
            n_iter = max(16, int(n_iter * (args.target_ms / 1e3) / slow))
            break
        n_iter *= 8

    runners = {name: (lambda ck=ck: ck[(grid, 1, 1)](x, y, n_iter)) for name, (ck, _) in variants.items()}
    with ClockSampler() as clock:
        times = time_runs(runners, args.runs, args.warmup)
    arr = {k: np.array(v) for k, v in times.items()}
    scale = n_sm / (grid * n_iter) * 1e9  # seconds per run -> SM-nanoseconds per CTA-iteration
    bytes_ = int(np.prod(shape)) * torch.tensor([], dtype=torch_dtype).element_size()

    def med(a):
        return float(np.median(a))

    ops = 1 if is_float else 2  # instructions in one filler op

    def add_src_f(b4, b2):
        return (np.median(b4) - np.median(b2)) / 2 * scale / ops

    def cvt_f(c, b4, b2, b4d, b2d):
        a_src = (np.median(b4) - np.median(b2)) / 2
        a_dst = (np.median(b4d) - np.median(b2d)) / 2
        return (np.median(c) - np.median(b2) + a_src - a_dst) / 2 * scale

    def add_mean_f(b4, b2, b4d, b2d):
        return ((np.median(b4) - np.median(b2)) + (np.median(b4d) - np.median(b2d))) / 4 * scale / ops

    if "base2d" not in arr:
        arr["base2d"], arr["base4d"] = arr["base2"], arr["base4"]
    base = [arr["base4"], arr["base2"], arr["base4d"], arr["base2d"]]
    add_src = add_src_f(arr["base4"], arr["base2"])
    add_dst = add_src_f(arr["base4d"], arr["base2d"])
    add_ns = add_mean_f(*base)
    result.update({
        "grid": grid, "n_iter": n_iter, "num_warps": warps, "bytes": bytes_, "n_sm": n_sm, "runs": args.runs,
        "clock": clock.summary(), "ops_per_filler": ops,
        "raw_ms": {k: {"median": med(v) * 1e3, "min": float(v.min()) * 1e3, "max": float(v.max()) * 1e3,
                       "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)) * 1e3} for k, v in arr.items()},
        "iter_ns": {k: med(v) * scale for k, v in arr.items()},
        "add_src_ns": add_src, "add_dst_ns": add_dst,
        "add_ns": add_ns, "add_ns_ci": boot_ci(add_mean_f, base),
    })
    for route in routes:
        if route not in variants:
            continue
        cvt_ns = cvt_f(arr[route], *base)
        ci = boot_ci(cvt_f, [arr[route], *base])
        ratio_ci = boot_ci(lambda c, b4, b2, b4d, b2d: cvt_f(c, b4, b2, b4d, b2d) / add_mean_f(b4, b2, b4d, b2d),
                           [arr[route], *base])
        # separated from the baseline?  the convert loop against the run-to-run spread of the baseline
        spread = float(np.percentile(arr["base2"], 97.5) - np.percentile(arr["base2"], 2.5))
        diff = med(arr[route]) - med(arr["base2"])
        result["routes"][route] = {
            "status": "ok", "tag": tags[route], "cvt_ns": cvt_ns, "cvt_ns_ci": ci,
            "cvt_per_add": cvt_ns / add_ns if add_ns > 0 else None, "cvt_per_add_ci": ratio_ci,
            "separated": bool(diff > spread and ci[0] > 0),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--grid-mult", type=int, default=32, help="CTAs launched per SM")
    ap.add_argument("--target-ms", type=float, default=60.0)
    ap.add_argument("--only", default=None, help="regex on case names")
    args = ap.parse_args()

    out = Path(args.out)
    workdir = out / "ttgir"
    workdir.mkdir(parents=True, exist_ok=True)
    events = out / "events.jsonl"
    cases = json.loads(Path(args.cases).read_text())
    if args.only:
        cases = [c for c in cases if re.search(args.only, c["name"])]
    props = triton.runtime.driver.active.utils.get_device_properties(0)
    n_sm = props["multiprocessor_count"]
    cap = torch.cuda.get_device_capability()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.graphics", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    session = {"device": torch.cuda.get_device_name(0), "compute_capability": f"{cap[0]}.{cap[1]}", "nvidia_smi": smi,
               "cuda_runtime": torch.version.cuda, "torch": torch.__version__, "triton": triton.__version__,
               "n_sm": n_sm, "grid_mult": args.grid_mult, "runs": args.runs, "warmup": args.warmup}
    (out / "session.json").write_text(json.dumps(session, indent=1))
    print(session, flush=True)

    with open(out / "results.jsonl", "a") as fh:
        for case in cases:
            try:
                res = run_case(case, args, workdir, events, n_sm)
            except BaseException as e:  # noqa: BLE001
                if isinstance(e, KeyboardInterrupt):
                    raise
                res = {"case": case, "status": "error", "error": f"{type(e).__name__}: {str(e)[:400]}"}
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            line = [case["name"]]
            if "routes" in res and "add_ns" in res:
                line.append(f'add={res["add_ns"]:.2f}ns')
                for r, v in res["routes"].items():
                    if v.get("status") == "ok":
                        line.append(f'{r}={v["cvt_ns"]:.1f}ns ({v["cvt_per_add"]:.1f} adds)')
                    else:
                        line.append(f'{r}:{v.get("status")}')
            else:
                line.append(res.get("error", ""))
            print(" | ".join(line), flush=True)


if __name__ == "__main__":
    main()
