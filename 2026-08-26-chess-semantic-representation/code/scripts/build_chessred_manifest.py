#!/usr/bin/env python3
"""Build and audit a normalized manifest from ChessReD annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_repr.data.audit import audit_records
from chess_repr.data.chessred import load_chessred_records
from chess_repr.data.manifest import summarize_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/chessred/annotations.json"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/raw/chessred/chessred2k"),
    )
    parser.add_argument("--subset", choices=("full", "chessred2k"), default="chessred2k")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/chessred2k.jsonl"),
    )
    args = parser.parse_args()

    records = load_chessred_records(
        args.annotations,
        image_root=args.image_root,
        subset=args.subset,
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

