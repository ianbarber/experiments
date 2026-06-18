#!/usr/bin/env python3
"""Run LLM operator export pipeline and generate visualizations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_op_evolution.export_model import run_exports
from llm_op_evolution.plot import generate_all_plots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "config" / "models.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Subset of model ids (default: all models in manifest)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # --skip-export is a legacy alias for --plot-only.
    args.plot_only = args.plot_only or args.skip_export

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    metrics_path = args.output / "metrics.json"

    if not args.plot_only:
        run_exports(
            args.manifest,
            args.output,
            model_ids=args.models,
            device=args.device,
            strict=args.strict,
        )

    if not metrics_path.exists():
        print(f"No metrics at {metrics_path}", file=sys.stderr)
        return 1

    try:
        figures = generate_all_plots(metrics_path, args.output / "figures")
        print("Generated figures:")
        for fig in figures:
            print(f"  {fig}")
    except ValueError as exc:
        print(f"Plotting skipped: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())