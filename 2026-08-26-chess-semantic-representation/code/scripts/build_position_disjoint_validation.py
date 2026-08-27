#!/usr/bin/env python3
"""Build a validation manifest excluding every train-seen board placement."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from chess_repr.data.audit import audit_records
from chess_repr.data.manifest import read_manifest, summarize_manifest, write_manifest
from chess_repr.data.splits import position_disjoint_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--validation-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--balanced-per-source", type=int)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    training = [record for path in args.train_manifest for record in read_manifest(path)]
    validation = [
        record for path in args.validation_manifest for record in read_manifest(path)
    ]
    kept, excluded = position_disjoint_validation(training, validation)
    eligible_samples = len(kept)
    if args.balanced_per_source is not None:
        if args.balanced_per_source <= 0:
            raise ValueError("--balanced-per-source must be positive")
        random_generator = random.Random(args.seed)
        by_source: dict[str, list] = {}
        for record in kept:
            by_source.setdefault(record.dataset_source, []).append(record)
        selected = []
        for source_records in by_source.values():
            random_generator.shuffle(source_records)
            selected.extend(source_records[: args.balanced_per_source])
        random_generator.shuffle(selected)
        kept = selected
    digest = write_manifest(kept, args.output)
    report = {
        "output": str(args.output),
        "sha256": digest,
        "summary": summarize_manifest(kept),
        "excluded_samples": len(excluded),
        "eligible_samples": eligible_samples,
        "balanced_per_source": args.balanced_per_source,
        "seed": args.seed,
        "excluded_by_source": dict(
            sorted(Counter(record.dataset_source for record in excluded).items())
        ),
        "audit": audit_records(kept).to_dict(),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
