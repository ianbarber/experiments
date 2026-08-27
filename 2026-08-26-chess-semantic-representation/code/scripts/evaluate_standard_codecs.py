#!/usr/bin/env python3
"""Measure JPEG, WebP, and PNG on the same rectified validation crops."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from chess_repr.data.dataset import ManifestImageDataset
from chess_repr.data.schema import Split


def _round_trip(image: Image.Image, *, image_format: str, quality: int | None) -> tuple[int, float]:
    buffer = BytesIO()
    options = {} if quality is None else {"quality": quality}
    image.save(buffer, format=image_format, **options)
    encoded = buffer.getvalue()
    with Image.open(BytesIO(encoded)) as decoded:
        decoded_array = np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0
    original_array = np.asarray(image, dtype=np.float32) / 255.0
    mse = float(np.square(decoded_array - original_array).mean())
    return len(encoded), mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(Split), default=Split.VALIDATION.value)
    parser.add_argument("--max-examples", type=int, default=512)
    args = parser.parse_args()

    dataset = ManifestImageDataset.from_manifests([args.manifest], splits=(args.split,))
    settings = [
        *[("JPEG", quality) for quality in (1, 5, 10, 20, 40, 60, 80)],
        *[("WEBP", quality) for quality in (1, 5, 10, 20, 40, 60, 80)],
        ("PNG", None),
    ]
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    examples = min(len(dataset), args.max_examples)
    for index in range(examples):
        tensor = dataset[index]["image"]
        image = Image.fromarray(
            tensor.mul(255).round().byte().permute(1, 2, 0).numpy(),
            mode="RGB",
        )
        for image_format, quality in settings:
            name = image_format if quality is None else f"{image_format.lower()}_q{quality}"
            byte_count, mse = _round_trip(
                image,
                image_format=image_format,
                quality=quality,
            )
            totals[name]["bytes"] += byte_count
            totals[name]["mse"] += mse

    report = {
        "manifest": str(args.manifest),
        "examples": examples,
        "image_size": 256,
        "results": {},
    }
    for name, values in sorted(totals.items()):
        mean_bytes = values["bytes"] / examples
        mean_mse = values["mse"] / examples
        report["results"][name] = {
            "mean_bytes": mean_bytes,
            "bits_per_pixel": mean_bytes * 8 / (256 * 256),
            "mean_mse": mean_mse,
            "psnr_db": -10 * math.log10(max(mean_mse, 1e-12)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
