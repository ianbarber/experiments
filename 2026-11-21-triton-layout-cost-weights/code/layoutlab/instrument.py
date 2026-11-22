"""Record every Triton compilation of a workload, tagged for the layout event log.

`install()` wraps `triton.compile` so that each distinct (kernel, specialization,
options, target) is compiled exactly once with `TRITON_LAYOUT_TAG` set, its final
TTGIR saved, and one row appended to `configs.jsonl`.

With `launch=False` (the census mode) kernels are compiled as far as LLVM IR and
never assembled, loaded or run, so the same sweep works for any target. With
`arch` different from the device's own, the compiler targets that architecture
and `torch.cuda.get_device_capability` reports it, so workload code takes the
branches it would take on that hardware.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import traceback
from pathlib import Path

_STATE = {}

# Options a caller chooses. Everything else in CUDAOptions is derived from the target.
USER_OPTIONS = ("num_warps", "num_ctas", "num_stages", "enable_fp_fusion", "maxnreg", "launch_cooperative_grid",
                "launch_pdl", "clc", "debug", "sanitize_overflow")


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return repr(v)


def _noop(*args, **kwargs):
    return None


def native_arch() -> int:
    import torch
    major, minor = _STATE.get("real_capability") or torch.cuda.get_device_capability()
    return major * 10 + minor


def install(workload: str, out_dir: str, arch: int | None = None, launch: bool = False, log_events: bool = True):
    """Start recording. Returns a dict describing the session."""
    import torch
    import triton
    from triton import knobs
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler import compiler as tc
    from triton.runtime import driver

    assert not _STATE.get("installed"), "install() called twice"
    out = Path(out_dir)
    (out / "ttgir").mkdir(parents=True, exist_ok=True)
    real_capability = torch.cuda.get_device_capability()
    real_arch = real_capability[0] * 10 + real_capability[1]
    arch = arch or real_arch
    emulated = arch != real_arch
    patched_build = hasattr(triton._C.libtriton.ir.module, "log_layout_census")
    _STATE.update(installed=True, workload=workload, out=out, arch=arch, emulated=emulated, launch=launch,
                  real_capability=real_capability, seen=set(), sources=[], lock=threading.RLock(), patched_build=patched_build)

    if log_events and patched_build:
        os.environ["TRITON_LAYOUT_LOG"] = str(out / "events.jsonl")
    else:
        os.environ.pop("TRITON_LAYOUT_LOG", None)

    if emulated:
        assert not launch, "cannot launch kernels built for another architecture"
        target = GPUTarget("cuda", arch, 32)
        driver.active.get_current_target = lambda: target
        torch.cuda.get_device_capability = lambda *a, **k: (arch // 10, arch % 10)
    if not launch:
        # Never assemble, load or launch. E1 and E2 need the pipeline up to LLVM IR
        # (for the lowering events), not ptxas.
        CUDABackend.make_cubin = lambda self, src, metadata, opt, capability: b""
        tc.CompiledKernel._init_handles = _noop
        tc.CompiledKernel.run = property(lambda self: _noop)
        # The autotuner would otherwise spend seconds per configuration timing a no-op.
        driver.active.get_benchmarker = lambda: (lambda kernel_call, quantiles=None, **kw:
                                                 (kernel_call(), [0.0, 0.0, 0.0])[1])
        # Some workloads read these after a warmup compile.
        tc.CompiledKernel.n_regs = 64
        tc.CompiledKernel.n_spills = 0
        tc.CompiledKernel.n_max_threads = 1024
        # Outputs are never written, so correctness checks in the workload are meaningless.
        torch.allclose = lambda *a, **k: True
        torch.testing.assert_close = _noop
        triton.testing.assert_close = _noop

    # Census of the final TTGIR, straight after the make_ttgir pipeline.
    orig_make_ttgir = CUDABackend.make_ttgir

    def make_ttgir(mod, metadata, opt, capability):
        mod = orig_make_ttgir(mod, metadata, opt, capability)
        if patched_build and os.environ.get("TRITON_LAYOUT_LOG"):
            mod.log_layout_census("final_ttgir")
        return mod

    CUDABackend.make_ttgir = staticmethod(make_ttgir)

    orig_compile = tc.compile

    def compile(src, target=None, options=None, _env_vars=None):
        with _STATE["lock"]:
            if emulated and target is not None and target.arch != arch:
                target = GPUTarget(target.backend, arch, target.warp_size)
            if os.environ.get("TRITON_LAYOUT_TAG"):
                # Another driver (the E4 variant builder) is tagging this compile itself.
                return orig_compile(src, target, options, _env_vars)
            tgt = target or driver.active.get_current_target()
            opts = dict(options or {})
            if hasattr(src, "hash"):
                name, src_key = src.name, src.hash()
            else:
                name, src_key = Path(str(src)).name, hashlib.sha256(Path(str(src)).read_bytes()).hexdigest()
            opt_key = sorted((k, repr(v)) for k, v in opts.items())
            h = hashlib.sha1(f"{src_key}|{opt_key}|{tgt.arch}".encode()).hexdigest()[:12]
            tag = f"{workload}|{name}|{h}|sm{tgt.arch}"
            if tag in _STATE["seen"]:
                return orig_compile(src, target, options, _env_vars)
            _STATE["seen"].add(tag)
            if hasattr(src, "fn") and not emulated and tgt.arch == real_arch:
                _STATE["sources"].append((src, {k: v for k, v in opts.items() if k in USER_OPTIONS}))

            row = {"tag": tag, "workload": workload, "kernel": name, "arch": tgt.arch, "emulated": emulated, "launched": launch}
            if hasattr(src, "signature"):
                row["signature"] = _jsonable(src.signature)
            if hasattr(src, "constants") and hasattr(src, "fn"):
                names = src.fn.arg_names
                row["constants"] = {
                    ".".join([names[k[0]]] + [str(i) for i in k[1:]]): _jsonable(v)
                    for k, v in src.constants.items()
                }
            row["options"] = {
                k: _jsonable(v)
                for k, v in opts.items() if k in USER_OPTIONS
            }
            os.environ["TRITON_LAYOUT_TAG"] = tag
            prev = knobs.compilation.always_compile
            knobs.compilation.always_compile = True
            t0 = time.time()
            try:
                ck = orig_compile(src, target, options, _env_vars)
                row["status"] = "ok"
                ttgir = ck.asm.get("ttgir") if hasattr(ck.asm, "get") else None
                if ttgir:
                    f = out / "ttgir" / f"{h}_sm{tgt.arch}.ttgir"
                    f.write_text(ttgir)
                    row["ttgir"] = f.name
                return ck
            except BaseException as e:
                row["status"] = "error"
                row["error"] = f"{type(e).__name__}: {str(e)[:400]}"
                raise
            finally:
                row["compile_s"] = round(time.time() - t0, 3)
                knobs.compilation.always_compile = prev
                os.environ["TRITON_LAYOUT_TAG"] = ""
                with open(out / "configs.jsonl", "a") as fh:
                    fh.write(json.dumps(row) + "\n")

    tc.compile = compile
    triton.compiler.compile = compile
    triton.compile = compile

    session = {
        "workload": workload,
        "arch": arch,
        "emulated": emulated,
        "launch": launch,
        "device": torch.cuda.get_device_name(0),
        "device_capability": f"{real_capability[0]}.{real_capability[1]}",
        "patched_build": patched_build,
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    (out / "session.json").write_text(json.dumps(session, indent=1))
    return session


def attempt(label: str, fn, *args, **kwargs):
    """Run one workload step; a failure is recorded and does not stop the sweep."""
    try:
        return fn(*args, **kwargs)
    except BaseException as e:  # noqa: BLE001 - compiler aborts surface as many types
        if isinstance(e, KeyboardInterrupt):
            raise
        out = _STATE.get("out")
        if out is not None:
            with open(out / "driver_errors.jsonl", "a") as fh:
                fh.write(
                    json.dumps({
                        "step": label,
                        "error": f"{type(e).__name__}: {str(e)[:400]}",
                        "trace": traceback.format_exc()[-1500:],
                    }) + "\n")
        return None


def replay(archs):
    """Recompile every recorded AST source for other targets, without launching.

    Only valid for workloads whose Python code does not branch on the device's
    capability (Inductor output, for example).
    """
    import triton
    from triton.backends.compiler import GPUTarget

    sources = list(_STATE["sources"])
    for arch in archs:
        target = GPUTarget("cuda", arch, 32)
        for src, opts in sources:
            attempt(f"replay {src.name} sm{arch}", triton.compile, src, target=target, options=opts)
