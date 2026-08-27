#!/usr/bin/env python3
"""Summarize validation metrics across controlled training replicas."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    for path in args.metrics:
        payload = json.loads(path.read_text())
        if payload["validation"] is None:
            raise ValueError(f"{path} contains no validation metrics")
        runs.append({"metrics_path": str(path), **payload["validation"]})
    metric_names = sorted(set(runs[0]) - {"metrics_path"})
    summary = {}
    for name in metric_names:
        values = [float(run[name]) for run in runs]
        summary[name] = {
            "mean": statistics.fmean(values),
            "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    best = min(runs, key=lambda run: float(run["loss"]))
    report = {
        "replicas": len(runs),
        "selection_rule": "lowest validation total loss",
        "selected_metrics_path": best["metrics_path"],
        "runs": runs,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
