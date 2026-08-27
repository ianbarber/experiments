#!/usr/bin/env python3
"""Validate a normalized JSONL data manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_repr.data.audit import audit_records
from chess_repr.data.manifest import manifest_digest, read_manifest, summarize_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = [record for path in args.manifest for record in read_manifest(path)]
    report = audit_records(records)
    result = {
        "manifests": {
            str(path): manifest_digest(path)
            for path in args.manifest
        },
        "summary": summarize_manifest(records),
        "audit": report.to_dict(),
    }
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
