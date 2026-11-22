#!/usr/bin/env python
"""Compile one workload for one target with the layout event log on.

    python run_census.py --workload tut03 --arch 90 --out runs/census/tut03_sm90

Workloads: tut02 tut03 tut05 tut06 tut08 tut09 tut10 inductor inductor:<model>.
Test suites are recorded through `layoutlab.pytest_plugin` instead.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", type=int, default=None, help="target compute capability, e.g. 90; default is the device's")
    ap.add_argument("--launch", action="store_true", help="assemble and run kernels (native target only)")
    ap.add_argument("--replay", default="", help="comma-separated targets to recompile recorded kernels for")
    ap.add_argument("--no-events", action="store_true", help="do not write the event log (unpatched-build comparison)")
    args = ap.parse_args()

    from layoutlab import instrument
    session = instrument.install(args.workload, args.out, arch=args.arch, launch=args.launch,
                                 log_events=not args.no_events)
    print(session, flush=True)

    if args.workload.startswith("tut"):
        from layoutlab import tutorials
        tutorials.run(args.workload)
    elif args.workload.startswith("inductor"):
        from layoutlab import inductor
        inductor.run(args.workload)
    else:
        raise SystemExit(f"unknown workload {args.workload}")

    if args.replay:
        instrument.replay([int(a) for a in args.replay.split(",")])


if __name__ == "__main__":
    main()
