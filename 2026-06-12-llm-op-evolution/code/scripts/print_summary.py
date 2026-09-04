#!/usr/bin/env python3
"""Print a compact summary table from results/metrics.json.

Usage: print_summary.py [path/to/metrics.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    metrics_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "metrics.json"
    if not metrics_path.exists():
        print(f"No metrics at {metrics_path}", file=sys.stderr)
        return 1

    with metrics_path.open() as f:
        rows = json.load(f)

    print(
        f"{'Model':<22} {'Date':<8} {'Layers':>6} {'Nodes':>7} {'N/Layer':>8} "
        f"{'Unique':>7} {'Het':>5} {'Mechs':>5} {'OK':>4}"
    )
    print("-" * 84)
    for row in sorted(rows, key=lambda r: r["model"]["release_date"]):
        m = row["model"]
        core = row.get("metrics", {}).get("core_aten") or {}
        mech = row.get("mechanisms") or {}
        nlayer = core.get("nodes_per_layer")
        if nlayer is None and core.get("compute_nodes") and row.get("num_layers"):
            nlayer = core["compute_nodes"] / row["num_layers"]
        nlayer_str = f"{nlayer:8.1f}" if nlayer is not None else f"{'-':>8}"
        het = mech.get("heterogeneity_score")
        het_str = f"{het:5.1f}" if het is not None else f"{'-':>5}"
        n_mech = mech.get("num_mechanisms")
        n_mech_str = f"{n_mech:5d}" if n_mech is not None else f"{'-':>5}"
        print(
            f"{m['name']:<22} {m['release_date']:<8} "
            f"{row.get('num_layers') or '-':>6} "
            f"{core.get('compute_nodes') or '-':>7} "
            f"{nlayer_str} "
            f"{core.get('unique_ops') or '-':>7} "
            f"{het_str} "
            f"{n_mech_str} "
            f"{'yes' if row['success'] else 'no':>4}"
        )
        if not row["success"] and row.get("error"):
            print(f"  error: {row['error'][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())