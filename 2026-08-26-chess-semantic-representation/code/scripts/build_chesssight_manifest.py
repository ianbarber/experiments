#!/usr/bin/env python3
"""Extract ChessSight images and build an audited normalized manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_repr.data.audit import audit_records
from chess_repr.data.chesssight import extract_chesssight_records
from chess_repr.data.manifest import summarize_manifest, write_manifest
from chess_repr.data.schema import Split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/processed/chesssight"),
    )
    parser.add_argument("--split", choices=tuple(Split), default=Split.TRAIN.value)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/chesssight_train0.jsonl"),
    )
    args = parser.parse_args()

    records = extract_chesssight_records(
        args.parquet,
        image_root=args.image_root,
        split=args.split,
        limit=args.limit,
    )
    digest = write_manifest(records, args.output)
    result = {
        "output": str(args.output),
        "sha256": digest,
        "summary": summarize_manifest(records),
        "audit": audit_records(records).to_dict(),
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["audit"]["ok"] else 1)


if __name__ == "__main__":
    main()
