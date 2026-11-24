"""Capture real kernel launches, rebuild the kernel under compiler overrides, and replay.

A *launch* is one compiled kernel specialization together with the call that first
used it. Two kinds are supported: kernels launched through Triton's JIT
(`JitLaunch`, recorded by `Capture`) and kernels launched by Inductor's own
launcher (`InductorLaunch`, recorded by `InductorCapture`). Both can be rebuilt with
override environment variables set (`build`) and re-run with the original arguments
(`run`), so every variant goes through the same launch path on the same inputs.

`study()` does the E4 procedure for one launch: enumerate alternative conflict
resolutions, rebuild, compare code, check outputs, time.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

_local = threading.local()
_counter = [0]
CAP = 64


def _tensors(obj, out):
    """Collect torch tensors reachable from call arguments (host tensor descriptors included)."""
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif hasattr(obj, "base") and isinstance(getattr(obj, "base"), torch.Tensor):
        out.append(obj.base)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _tensors(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _tensors(x, out)


def _unique(tensors):
    uniq, seen = [], set()
    for t in tensors:
        if id(t) not in seen:
            seen.add(id(t))
            uniq.append(t)
    return uniq


class Built:
    def __init__(self, handle, events, ttgir):
        self.handle, self.events, self.ttgir = handle, events, ttgir


class _Env:
    """Set the event log, a fresh tag and any overrides for the duration of one compile."""

    def __init__(self, label, name, events_path, env):
        _counter[0] += 1
        self.tag = f"{label}|{name}|{_counter[0]}"
        self.events_path = Path(events_path)
        self.vars = dict(env or {}, TRITON_LAYOUT_LOG=str(events_path), TRITON_LAYOUT_TAG=self.tag)

    def __enter__(self):
        from triton import knobs
        self.saved = {k: os.environ.get(k) for k in self.vars}
        self.prev = knobs.compilation.always_compile
        os.environ.update(self.vars)
        knobs.compilation.always_compile = True
        return self

    def __exit__(self, *exc):
        from triton import knobs
        knobs.compilation.always_compile = self.prev
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def events(self):
        out = []
        if self.events_path.exists():
            with open(self.events_path) as fh:
                for line in fh:
                    if self.tag in line:
                        e = json.loads(line)
                        if e.get("tag") == self.tag:
                            out.append(e)
            # each build reads only its own events; keep the file from growing without bound
            self.events_path.unlink()
        return out


class JitLaunch:
    """A specialization compiled by Triton's JIT and the call that first used it."""

    scramble_bytes = True  # 8-bit tensors here are mxfp values and scales

    kind = "jit"

    def __init__(self, fn, key, signature, device, constexprs, options, attrs):
        self.fn, self.key, self.signature, self.device = fn, key, signature, device
        self.constexprs, self.options, self.attrs = constexprs, options, attrs
        self.args = self.kwargs = None

    @property
    def name(self):
        return self.fn.__name__

    def constants(self):
        names = self.fn.arg_names
        return {names[k[0]]: v for k, v in self.constexprs.items() if len(k) == 1}

    def describe(self):
        opts = self.options.__dict__
        return {"kernel": self.name, "constants": {k: str(v) for k, v in self.constants().items()},
                "signature": {k: str(v) for k, v in self.signature.items()},
                "options": {k: opts.get(k) for k in ("num_warps", "num_stages", "num_ctas", "maxnreg")}}

    def tensors(self):
        out = []
        _tensors(self.args, out)
        _tensors(self.kwargs, out)
        return _unique(out)

    def build(self, label, events_path, env=None):
        import triton
        src = self.fn.ASTSource(self.fn, self.signature, self.constexprs, self.attrs)
        target = self.fn.device_caches[self.device][2]
        with _Env(label, self.name, events_path, env) as e:
            binary = triton.compile(src, target=target, options=self.options.__dict__)
        return Built(binary, e.events(), binary.asm["ttgir"])

    def run(self, handle):
        self.fn.device_caches[self.device][0][self.key] = handle
        self.fn.run(*self.args, warmup=False, **self.kwargs)


class Capture:
    """Record JIT compilations and the calls that trigger them.

    With `events_path`, each compilation is logged and the number of conflicts it met is
    stored on the launch, so kernels without a conflict can be skipped without rebuilding.
    """

    def __init__(self, events_path=None):
        from triton.runtime.jit import JITFunction

        self.launches = []
        self.events_path = events_path
        self._JIT = JITFunction
        self._orig_compile = JITFunction._do_compile
        self._orig_run = JITFunction.run
        capture = self

        def run(fn, *args, grid, warmup, **kwargs):
            prev = getattr(_local, "call", None)
            _local.call = (args, dict(kwargs, grid=grid))
            try:
                return capture._orig_run(fn, *args, grid=grid, warmup=warmup, **kwargs)
            finally:
                _local.call = prev

        def _do_compile(fn, key, signature, device, constexprs, options, attrs, warmup):
            n_conflicts = None
            if capture.events_path is not None:
                with _Env("capture", fn.__name__, capture.events_path, None) as e:
                    kernel = capture._orig_compile(fn, key, signature, device, constexprs, options, attrs, warmup)
                n_conflicts = len(conflicts_of(e.events()))
            else:
                kernel = capture._orig_compile(fn, key, signature, device, constexprs, options, attrs, warmup)
            launch = JitLaunch(fn, key, signature, device, constexprs, options, attrs)
            launch.n_conflicts = n_conflicts
            call = getattr(_local, "call", None)
            if call is not None:
                # Autotune pre-hooks rewrite host tensor descriptors in place for each
                # configuration, so keep this configuration's own copies.
                args, kw = call
                launch.args = tuple(copy.copy(a) if hasattr(a, "block_shape") else a for a in args)
                launch.kwargs = {k: (copy.copy(v) if hasattr(v, "block_shape") else v) for k, v in kw.items()}
            capture.launches.append(launch)
            return kernel

        JITFunction.run = run
        JITFunction._do_compile = _do_compile

    def stop(self):
        self._JIT.run = self._orig_run
        self._JIT._do_compile = self._orig_compile


class InductorLaunch:
    """A kernel Inductor compiled and launched through its own autotuner object."""

    scramble_bytes = False  # 8-bit tensors here are max-pool offsets, guarded by device asserts

    kind = "inductor"

    def __init__(self, autotuner, args, kwargs, stream):
        self.autotuner, self.args, self.kwargs, self.stream = autotuner, args, kwargs, stream
        self.config = autotuner.launchers[0].config

    @property
    def name(self):
        return self.autotuner.fn.__name__

    def describe(self):
        return {"kernel": self.name, "config": str(self.config),
                "options": {"num_warps": getattr(self.config, "num_warps", None),
                            "num_stages": getattr(self.config, "num_stages", None)}}

    def tensors(self):
        out = []
        _tensors(self.args, out)
        _tensors(self.kwargs, out)
        return _unique(out)

    def build(self, label, events_path, env=None):
        with _Env(label, self.name, events_path, env) as e:
            result = self.autotuner._precompile_config(self.config)
        launcher = result.make_launcher()
        # make_launcher loads the binary; carry what the driver reported onto the handle
        for attr in ("n_regs", "n_spills", "metadata"):
            try:
                setattr(launcher, attr, getattr(result.kernel, attr))
            except Exception:  # noqa: BLE001 - best effort, only used for the record
                pass
        return Built(launcher, e.events(), result.kernel.asm["ttgir"])

    def run(self, handle):
        # the stream of the moment, so launches land in a CUDA graph when one is being captured
        stream = torch._C._cuda_getCurrentRawStream(torch.cuda.current_device())
        handle(*self.args, **self.kwargs, stream=stream)


class InductorCapture:
    """Record the first launch of every Inductor kernel."""

    def __init__(self):
        import torch._inductor.config as config
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        # Keep the compiled kernel object (and its IR) reachable from every launcher.
        config.use_static_cuda_launcher = False
        self.launches = []
        self._seen = set()
        self._cls = CachingAutotuner
        self._orig_run = CachingAutotuner.run
        capture = self

        def run(autotuner, *args, stream, benchmark_run=False, **kwargs):
            result = capture._orig_run(autotuner, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)
            if id(autotuner) not in capture._seen and not benchmark_run and len(autotuner.launchers) == 1:
                capture._seen.add(id(autotuner))
                capture.launches.append(InductorLaunch(autotuner, args, kwargs, stream))
            return result

        CachingAutotuner.run = run

    def stop(self):
        self._cls.run = self._orig_run


# --------------------------------------------------------------------------- comparison and timing


def _is_data(t, bytes_ok):
    """May this tensor be overwritten with a pattern to see whether the kernel writes it?

    Floating-point tensors hold values. Integer tensors may be indices, and a kernel given
    garbage indices reads out of bounds or trips a device assert, which kills the CUDA
    context. 8-bit tensors are data in triton_kernels (mxfp values and scales) and indices in
    Inductor (max-pool offsets), so the launch says whether they may be touched. Wider
    integers and booleans never are."""
    return t.dtype.is_floating_point or (bytes_ok and t.dtype in (torch.uint8, torch.int8))


def _scramble(t, reference, bytes_ok=True):
    """Overwrite a data tensor with something that is not `reference`. Returns False if it could not."""
    if not _is_data(t, bytes_ok) or t.numel() == 0:
        return False
    # 0x3C bytes read as small floats (0.01 for fp32 and bf16, 1.06 for fp16), so what is left
    # of the pattern after a partial write does not swamp the scale of a relative comparison.
    for value in (0x3C, 0x5A, 0x00):
        try:
            if t.is_contiguous():
                t.view(torch.uint8).fill_(value)
            elif value:
                t.fill_(1)
            else:
                t.zero_()
        except Exception:  # noqa: BLE001 - exotic dtypes
            continue
        if not torch.equal(t, reference):
            return True
    return False


def _same(a, b):
    """Bitwise equality, so that NaNs compare equal to themselves."""
    try:
        return torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
    except Exception:  # noqa: BLE001 - dtypes that cannot be viewed as bytes
        return torch.equal(a, b)


def find_outputs(launch, handle, snapshot, tol=1e-2):
    """Which tensor arguments does the launch write, and how can a comparison see it?

    The arguments were captured after the kernel had already run once, so a tensor the
    kernel only writes already holds the right answer: a rebuilt kernel that wrote nothing
    would pass a plain comparison. Written tensors come in two kinds.

    - Self-checking: launched from the snapshot, the tensor changes. That is a running mean
      or an in-place activation, or an output that a later kernel overwrote before the
      snapshot was taken. A build has to reproduce the change, so nothing more is needed.
    - Outputs: launched from the snapshot the tensor comes back the same, but if it is
      overwritten first the launch changes it, in whole or in part. These are overwritten
      before every launch; whatever the kernel does not write stays overwritten in every
      build and compares equal.

    Returns (indices of outputs to overwrite, indices of tensors updated in place).
    """
    tensors = launch.tensors()

    def restore():
        for tt, s in zip(tensors, snapshot):
            tt.copy_(s)

    with torch.no_grad():
        restore()
        launch.run(handle)
        torch.cuda.synchronize()
        in_place = []
        for i, t in enumerate(tensors):
            if _same(t, snapshot[i]):
                continue
            c = compare([snapshot[i]], [t])
            # a change the comparison's tolerance would wave through cannot check anything
            if c["nan_mismatch"] or (c["max_abs"] > tol and c["max_rel"] > tol):
                in_place.append(i)
        outputs = []
        for i, t in enumerate(tensors):
            if i in in_place:
                continue
            restore()
            if not _scramble(t, snapshot[i], getattr(launch, "scramble_bytes", False)):
                continue
            before = t.detach().clone()
            launch.run(handle)
            torch.cuda.synchronize()
            if not _same(t, before):
                outputs.append(i)
        restore()
    return outputs, in_place


def outputs_after(launch, handle, snapshot, outputs=()):
    """Restore every tensor argument, scramble the known outputs, launch once, return copies."""
    tensors = launch.tensors()
    with torch.no_grad():
        for t, s in zip(tensors, snapshot):
            t.copy_(s)
        for i in outputs:
            _scramble(tensors[i], snapshot[i])
        launch.run(handle)
        torch.cuda.synchronize()
        return [t.detach().clone() for t in tensors]


def compare(reference, candidate):
    """Bit-exact, or the largest absolute and relative difference over all tensor arguments."""
    exact, max_abs, max_rel, nan_mismatch = True, 0.0, 0.0, False
    for a, b in zip(reference, candidate):
        if a.dtype != b.dtype or a.shape != b.shape:
            return {"exact": False, "max_abs": float("inf"), "max_rel": float("inf"), "nan_mismatch": True}
        if torch.equal(a, b):
            continue
        fa, fb = a.double(), b.double()
        nan_a, nan_b = torch.isnan(fa), torch.isnan(fb)
        if not torch.equal(nan_a, nan_b):
            nan_mismatch = True
        both = ~(nan_a | nan_b)
        if both.any():
            x, y = fa[both], fb[both]
            same = x == y  # equal values, infinities of the same sign included
            if not bool(same.all()):
                exact = False
                # infinite when an infinity meets a finite value or the opposite infinity
                worst = float((x[~same] - y[~same]).abs().max())
                max_abs = max(max_abs, worst)
                ref = x[torch.isfinite(x)]
                scale = float(ref.abs().max()) if ref.numel() else 1.0
                max_rel = max(max_rel, worst / max(scale, 1e-12))
        if nan_mismatch:
            exact = False
    return {"exact": exact, "max_abs": max_abs, "max_rel": max_rel, "nan_mismatch": nan_mismatch}


def time_variants(launch, handles: dict, n_runs: int, warmup: int, inner: int = 1, graph: bool = True):
    """Interleaved timing of every variant.

    Each sample is `inner` launches. With `graph`, those launches are captured once into a
    CUDA graph per variant and the graph is replayed, so the host's launch overhead (tens of
    microseconds, more than many of these kernels take) is not in the measurement. This is
    what Triton's own `do_bench_cudagraph` does. Falls back to plain launches if capture fails.

    Returns ({name: seconds per launch, one entry per run}, timing mode).
    """
    graphs, mode = {}, "wall clock"
    if graph:
        try:
            with torch.cuda.stream(torch.cuda.Stream()):
                for name, handle in handles.items():
                    launch.run(handle)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for _ in range(inner):
                            launch.run(handle)
                    torch.cuda.synchronize()
                    graphs[name] = g
            mode = "cuda graph replay"
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            graphs, mode = {}, f"wall clock (graph capture failed: {type(e).__name__})"
    times = {name: [] for name in handles}
    for i in range(warmup + n_runs):
        for name, handle in handles.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if graphs:
                graphs[name].replay()
            else:
                for _ in range(inner):
                    launch.run(handle)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / inner
            if i >= warmup:
                times[name].append(dt)
    del graphs
    return {k: np.array(v) for k, v in times.items()}, mode


def median_ci(x, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def gpu_state():
    return subprocess.run(["nvidia-smi", "--query-gpu=clocks.current.graphics,temperature.gpu,power.draw",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()


# --------------------------------------------------------------------------- the E4 procedure


def conflicts_of(events):
    out = []
    for e in events:
        if e["event"] == "conflict":
            out.append({"inv": e["inv"], "index": e["index"], "n": len(e["candidates"]), "default": e["default"],
                        "chosen": e["chosen"], "op": e["op"], "shape": e["shape"], "dtype": e["dtype"],
                        "loop_depth": e["loop_depth"], "candidates": e["candidates"], "rule": e["rule"]})
    return out


def census_of(events):
    """Conversions in the final TTGIR: class (with barrier), split by inside or outside a loop."""
    c = {}
    for e in events:
        if e["event"] == "census" and e["stage"] == "final_ttgir":
            key = e["class"] if e["class"] != "shared_memory" else f'shared_memory:{e["barrier"]}'
            key += "@loop" if e["loop_depth"] > 0 else ""
            c[key] = c.get(key, 0) + 1
    return c


def enumerate_variants(conflicts, cap=CAP, seed=42):
    """Return ({name: {(inv, index): candidate}}, number of alternatives that exist)."""
    out = {}
    taken = set()

    def add(name, choice):
        if not choice or len(out) >= cap:
            return
        key = tuple(sorted(choice.items()))
        if key in taken:
            return
        taken.add(key)
        out[name] = dict(choice)

    alts = {(c["inv"], c["index"]): [i for i in range(c["n"]) if i != c["default"]] for c in conflicts}
    add("all", {k: v[0] for k, v in alts.items()})
    for inv in sorted({c["inv"] for c in conflicts}):
        add(f"inv{inv}", {k: v[0] for k, v in alts.items() if k[0] == inv})
    for k, v in alts.items():
        for cand in v:
            add(f"one_{k[0]}_{k[1]}_{cand}", {k: cand})
    total = 1
    for c in conflicts:
        total *= c["n"]
        if total > 10**9:
            break
    rng = random.Random(seed)
    tries = 0
    while len(out) < min(cap, total - 1) and tries < 50 * cap:
        tries += 1
        choice = {}
        for k, v in alts.items():
            pick = rng.randrange(len(v) + 1)
            if pick:
                choice[k] = v[pick - 1]
        add(f"rand{tries}", choice)
    return out, total - 1


def override_string(choice):
    return ",".join(f"{inv}:{idx}={cand}" for (inv, idx), cand in sorted(choice.items()))


def study(launch, label, out: Path, n_runs=30, warmup=5, min_ms=20.0, launchable=True, tol=1e-2, env_variants=None):
    """Enumerate, compare, verify and time the alternative resolutions of one launch.

    With `launchable=False` (a target the device cannot run) only the code comparison is done.
    With `env_variants` ({name: {VARIABLE: value}}) those builds are studied instead of the
    conflict alternatives, e.g. one build with `TRITON_LAYOUT_REPRICE` set.
    """
    out = Path(out)
    (out / "ttgir").mkdir(parents=True, exist_ok=True)
    events_path = out / f"build_events_{os.getpid()}.jsonl"
    rec = {"label": label, "kind": launch.kind, **launch.describe()}

    default = launch.build(f"{label}|default", events_path)
    conflicts = conflicts_of(default.events)
    rec["conflicts"] = conflicts
    rec["default_census"] = census_of(default.events)
    if env_variants is None:
        if not conflicts:
            rec["status"] = "no_conflict"
            return rec
        choices, n_alternatives = enumerate_variants(conflicts)
        plan = {name: ({"TRITON_LAYOUT_RESOLVE_OVERRIDE": override_string(c)},
                       {"override": override_string(c), "flipped": len(c)}) for name, c in choices.items()}
        rec["alternatives_total"] = n_alternatives
        rec["alternatives_tried"] = len(choices)
        rec["sampled"] = n_alternatives > len(choices)
    else:
        plan = {name: (dict(env), {"env": dict(env), "flipped": 0}) for name, env in env_variants.items()}
        rec["alternatives_total"] = rec["alternatives_tried"] = len(plan)
        rec["sampled"] = False
    (out / "ttgir" / f"{label}__default.ttgir").write_text(default.ttgir)

    if launchable:
        snapshot = [t.detach().clone() for t in launch.tensors()]
        # What the output check can see: the tensor arguments, and which of them the launch writes.
        outputs, in_place = find_outputs(launch, default.handle, snapshot, tol)
        rec["tensor_args"] = [[list(t.shape), str(t.dtype).replace("torch.", "")] for t in snapshot]
        rec["output_args"] = outputs
        rec["in_place_args"] = in_place
        rec["tensor_args_written"] = len(outputs) + len(in_place)
        reference = outputs_after(launch, default.handle, snapshot, outputs)
        rec["default_repeatable"] = compare(reference, outputs_after(launch, default.handle, snapshot, outputs))
        # With nothing the launch visibly writes, a kernel that wrote nothing would pass.
        checkable = bool(outputs or in_place)
        rec["output_check"] = ("outputs overwritten before each launch" if outputs else
                               "tensor updated in place" if in_place else "vacuous")

    handles = {"default": default.handle}
    by_code = {hashlib.sha256(default.ttgir.encode()).hexdigest(): "default"}
    rec["variants"] = {}
    for name, (env, v) in plan.items():
        rec["variants"][name] = v
        try:
            built = launch.build(f"{label}|{name}", events_path, env)
        except BaseException as e:  # noqa: BLE001 - the compiler can abort in many ways
            if isinstance(e, KeyboardInterrupt):
                raise
            v["status"] = "compile_error"
            v["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            continue
        realized = conflicts_of(built.events)
        v["realized_conflicts"] = len(realized)
        v["realized_overrides"] = sum(1 for c in realized if c["rule"] == "override")
        v["census"] = census_of(built.events)
        digest = hashlib.sha256(built.ttgir.encode()).hexdigest()
        if digest in by_code:
            v["status"] = "identical_code"
            v["same_as"] = by_code[digest]
            continue
        by_code[digest] = name
        (out / "ttgir" / f"{label}__{name}.ttgir").write_text(built.ttgir)
        if not launchable:
            v["status"] = "distinct_code"
            continue
        try:
            got = outputs_after(launch, built.handle, snapshot, outputs)
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            v["status"] = "launch_error"
            v["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            continue
        cmp = compare(reference, got)
        v["output"] = cmp
        if not (cmp["exact"] or (not cmp["nan_mismatch"] and (cmp["max_abs"] <= tol or cmp["max_rel"] <= tol))):
            v["status"] = "wrong_output"
            continue
        if not checkable:
            v["status"] = "unverified_output"
            continue
        v["status"] = "distinct_code"
        handles[name] = built.handle

    if launchable:
        with torch.no_grad():
            for t, s in zip(launch.tensors(), snapshot):
                t.copy_(s)
    if launchable and len(handles) > 1:
        # Null control: the default built a second time. It is the same program, so whatever
        # difference it shows from the default is the method's own noise.
        try:
            handles["control_default"] = launch.build(f"{label}|control", events_path).handle
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
        launch.run(default.handle)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            launch.run(default.handle)
        torch.cuda.synchronize()
        one = (time.perf_counter() - t0) / 3
        inner = min(2000, max(1, int(np.ceil(min_ms / 1e3 / max(one, 1e-7)))))
        rec["inner_repeats"] = inner
        rec["wall_ms_per_launch"] = one * 1e3
        rec["gpu_before"] = gpu_state()
        times, rec["timing_mode"] = time_variants(launch, handles, n_runs, warmup, inner)
        rec["gpu_after"] = gpu_state()
        # Cross-check on the graph replay: plain launches, host overhead included.
        plain, _ = time_variants(launch, handles, 5, 1, min(inner, 50), graph=False)
        rec["plain_launch_ms"] = {k: float(np.median(v)) * 1e3 for k, v in plain.items()}
        d_med, d_lo, d_hi = median_ci(times["default"])
        rec["default_ms"] = {"median": d_med * 1e3, "ci": [d_lo * 1e3, d_hi * 1e3],
                             "min": float(times["default"].min()) * 1e3, "max": float(times["default"].max()) * 1e3,
                             "runs": len(times["default"])}
        rec["default_binary"] = binary_stats(default.handle)
        for name, handle in handles.items():
            if name in rec["variants"]:
                rec["variants"][name]["binary"] = binary_stats(handle)
        for name, t in times.items():
            if name == "default":
                continue
            med, lo, hi = median_ci(t)
            if name == "control_default":
                rec["control"] = {"vs_default_pct": (med / d_med - 1) * 100, "ms": {"median": med * 1e3, "ci": [lo * 1e3, hi * 1e3]},
                                  "faster": bool(hi < d_lo and med <= d_med * 0.97),
                                  "slower": bool(lo > d_hi and med >= d_med * 1.03)}
                continue
            v = rec["variants"][name]
            v["ms"] = {"median": med * 1e3, "ci": [lo * 1e3, hi * 1e3], "min": float(t.min()) * 1e3,
                       "max": float(t.max()) * 1e3}
            v["vs_default_pct"] = (med / d_med - 1) * 100
            # a difference is reported only at 3% or more with non-overlapping intervals
            v["faster"] = bool(hi < d_lo and med <= d_med * 0.97)
            v["slower"] = bool(lo > d_hi and med >= d_med * 1.03)
        rec["raw_times_ms"] = {k: [round(float(x) * 1e3, 6) for x in v] for k, v in times.items()}
    # restore the default binary for anything that runs afterwards
    if launchable and launch.kind == "jit":
        launch.fn.device_caches[launch.device][0][launch.key] = default.handle
    rec["status"] = "ok"
    return rec


def binary_stats(handle):
    """Registers per thread and `n_spills` (the per-thread local stack frame in 4-byte words) that
    the CUDA driver reports for a loaded Triton binary, where available."""
    out = {}
    for name in ("n_regs", "n_spills"):
        v = getattr(handle, name, None)
        if isinstance(v, int):
            out[name] = v
    meta = getattr(handle, "metadata", None)
    if meta is not None and hasattr(meta, "shared"):
        out["shared_bytes"] = int(meta.shared)
    return out


def summarize(rec):
    states = {}
    for v in rec.get("variants", {}).values():
        states[v.get("status")] = states.get(v.get("status"), 0) + 1
    best = min((v["vs_default_pct"] for v in rec.get("variants", {}).values() if "vs_default_pct" in v), default=None)
    return (f'{rec["label"]} {rec.get("status")} conflicts={len(rec.get("conflicts", []))} {states} '
            f'default_ms={rec.get("default_ms", {}).get("median")} best_vs_default_pct={best}')
