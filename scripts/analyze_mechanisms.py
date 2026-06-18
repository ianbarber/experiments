#!/usr/bin/env python3
"""Update metrics with architectural mechanism inventory and heterogeneity scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_op_evolution.mechanisms import compute_heterogeneity, get_model_mechanisms, load_manifest


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
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    metrics_path = args.output / "metrics.json"

    if not metrics_path.exists():
        print(f"No metrics at {metrics_path}", file=sys.stderr)
        return 1

    with metrics_path.open() as f:
        rows = json.load(f)

    model_by_id = {m["id"]: m for m in manifest.get("models", [])}
    updated = 0
    for row in rows:
        model_id = row["model"]["id"]
        model_cfg = model_by_id.get(model_id)
        if model_cfg is None:
            print(f"Warning: {model_id} not in manifest", file=sys.stderr)
            continue

        mechanisms = get_model_mechanisms(model_cfg)
        het = compute_heterogeneity(mechanisms)

        row["mechanisms"] = het

        # Also update per-model JSON file.
        per_model_path = args.output / f"{model_id}.json"
        if per_model_path.exists():
            with per_model_path.open() as f:
                per_model = json.load(f)
            per_model["mechanisms"] = het
            per_model_path.write_text(json.dumps(per_model, indent=2))

        updated += 1

    metrics_path.write_text(json.dumps(rows, indent=2))
    print(f"Updated mechanisms for {updated} models in {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
