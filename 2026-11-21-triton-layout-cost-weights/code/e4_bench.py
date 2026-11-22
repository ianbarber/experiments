#!/usr/bin/env python
"""E4: does a different conflict resolution run faster?

For each kernel specialization that showed a conflict, enumerate alternative resolutions
(at most 64 per kernel), rebuild the kernel with `TRITON_LAYOUT_RESOLVE_OVERRIDE`, and
compare the final TTGIR with the default build. Alternatives that produce different code
are checked against the default's output on the same inputs and timed, interleaved with
the default. Alternatives that produce byte-identical code are the same program.

Variants: every conflict flipped; every conflict of one invocation flipped; each single
conflict flipped to each of its other candidates; then random combinations (seed 42) up
to the cap.

    python e4_bench.py --workload tut06 --out runs/e4/tut06_sm120
    python e4_bench.py --workload tut06 --arch 90 --out runs/e4/tut06_sm90     # code comparison only
    python e4_bench.py --workload inductor --out runs/e4/inductor_sm120

Test suites go through `layoutlab.e4_plugin` instead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from layoutlab import variants as V  # noqa: E402


def _match(launches, config):
    if config is None:
        return None
    for la in launches:
        consts, opts = la.constants(), la.options.__dict__
        if all(consts.get(k) == v for k, v in config.kwargs.items()) and opts.get("num_warps") == config.num_warps \
                and opts.get("num_stages") == config.num_stages:
            return la
    return None


def workload_tut06(out, args, launchable):
    """Tutorial 06 forward attention at the tutorial's benchmark shape.

    The backward kernels are left out: the census found no conflict in them, and the
    tutorial's backward configuration needs more shared memory than the RTX 5090 has.
    """
    from layoutlab import tutorials

    cap = V.Capture(out / "capture_events.jsonl")
    g = tutorials.load("tut06")
    torch.manual_seed(42)
    batch, heads, n_ctx = 4, 32, 4096
    for head_dim in (64, 128):
        for causal in (False, True):
            q = torch.randn((batch, heads, n_ctx, head_dim), dtype=torch.float16, device="cuda")
            k = torch.randn((batch, heads, n_ctx, head_dim), dtype=torch.float16, device="cuda")
            v = torch.randn((batch, heads, n_ctx, head_dim), dtype=torch.float16, device="cuda")
            tutorials._reset_autotune(g)
            n_before = len(cap.launches)
            with torch.no_grad():
                g["attention"](q, k, v, causal, 0.5, False)
            torch.cuda.synchronize()
            fwd = [la for la in cap.launches[n_before:] if la.name == "_attn_fwd" and la.args is not None]
            label = f"attn_fwd_hd{head_dim}_{'causal' if causal else 'full'}"
            chosen = []
            if launchable:
                # what the autotuner would deploy, then the fastest configurations that have a conflict
                best = _match(fwd, g["_attn_fwd"].best_config)
                if best is not None:
                    chosen.append((f"{label}__autotuned_best", best))
                timings = getattr(g["_attn_fwd"], "configs_timings", None) or {}
                ranked = sorted(timings.items(), key=lambda kv: kv[1][0] if isinstance(kv[1], (list, tuple)) else kv[1])
                n = 0
                for cfg, _ in ranked:
                    la = _match(fwd, cfg)
                    if la is not None and la.n_conflicts and la is not best and n < args.per_group:
                        n += 1
                        chosen.append((f"{label}__conflicted_rank{n}", la))
            else:
                conflicted = [la for la in fwd if la.n_conflicts]
                for i, la in enumerate(conflicted[:args.per_group]):
                    chosen.append((f"{label}__conflicted_{i}", la))
            for name, la in chosen:
                yield name, la
    cap.stop()


def workload_inductor(out, args, launchable):
    """Every Inductor kernel of the four census models, launched with the arguments it first saw."""
    from layoutlab import inductor

    assert launchable, "Inductor kernels are identical across targets; study them on the device"
    cap = V.InductorCapture()
    for model in inductor.MODELS:
        n_before = len(cap.launches)
        inductor._infer(model)
        if model in inductor.TRAIN:
            inductor._train(model)
        torch.cuda.synchronize()
        seen = set()
        for la in cap.launches[n_before:]:
            if la.name in seen:
                continue
            seen.add(la.name)
            yield f"{model}__{la.name}", la
        torch._dynamo.reset()
    cap.stop()


WORKLOADS = {"tut06": workload_tut06, "inductor": workload_inductor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", type=int, default=None, help="other target: compare code only, nothing is launched")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--min-ms", type=float, default=20.0, help="each timed sample repeats the launch to last this long")
    ap.add_argument("--per-group", type=int, default=3, help="conflicted configurations studied per kernel group")
    ap.add_argument("--only", default=None, help="regex on study labels; others are skipped")
    ap.add_argument("--reprice", default=None, help="study one build with this TRITON_LAYOUT_REPRICE instead of the "
                    "conflict alternatives")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = torch.cuda.get_device_capability()
    native = cap[0] * 10 + cap[1]
    launchable = args.arch is None or args.arch == native
    if not launchable:
        from layoutlab import instrument
        instrument.install(f"e4_{args.workload}", str(out / "census"), arch=args.arch, launch=False, log_events=False)
    session = {"device": torch.cuda.get_device_name(0), "device_capability": f"{cap[0]}.{cap[1]}",
               "target": args.arch or native, "launchable": launchable, "cuda_runtime": torch.version.cuda,
               "torch": torch.__version__, "runs": args.runs, "warmup": args.warmup, "min_ms": args.min_ms,
               "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                            capture_output=True, text=True).stdout.strip()}
    (out / "session.json").write_text(json.dumps(session, indent=1))
    with open(out / "results.jsonl", "a") as fh:
        for label, launch in WORKLOADS[args.workload](out, args, launchable):
            if args.only and not re.search(args.only, label):
                continue
            try:
                env_variants = {"repriced": {"TRITON_LAYOUT_REPRICE": args.reprice}} if args.reprice else None
                rec = V.study(launch, label, out, args.runs, args.warmup, args.min_ms, launchable,
                              env_variants=env_variants)
            except BaseException as e:  # noqa: BLE001
                if isinstance(e, KeyboardInterrupt):
                    raise
                rec = {"label": label, "status": "error", "error": f"{type(e).__name__}: {str(e)[:400]}"}
            rec["target"] = args.arch or native
            # keep the file small: conflicts are described once in the census
            for c in rec.get("conflicts", []):
                c.pop("candidates", None)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            if rec.get("status") != "no_conflict":
                print(V.summarize(rec), flush=True)


if __name__ == "__main__":
    main()
