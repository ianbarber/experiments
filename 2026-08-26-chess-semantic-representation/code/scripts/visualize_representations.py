#!/usr/bin/env python3
"""Render input, full reconstruction, and semantic-base reconstruction rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from chess_repr.data.dataset import ManifestImageDataset
from chess_repr.data.schema import Split
from chess_repr.fen import grid_to_placement
from chess_repr.models.structured_vae import StructuredVAE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(Split), default=Split.VALIDATION.value)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    configuration = checkpoint["configuration"]
    model = StructuredVAE(
        appearance_dimensions=int(configuration["appearance_dimensions"]),
        base_channels=int(configuration["base_channels"]),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = ManifestImageDataset.from_manifests(
        [args.manifest],
        splits=(args.split,),
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.examples, shuffle=False)))
    images = batch["image"].to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        output = model(images, sample=False)
        semantic_base = model.decode_semantic_base(output.hard_semantic)

    tiles = torch.stack(
        [
            tile.detach().cpu()
            for row in zip(images, output.reconstruction, semantic_base, strict=True)
            for tile in row
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(tiles, nrow=3, padding=4), args.output)

    predictions = output.semantic_logits.argmax(dim=1).detach().cpu()
    report = []
    for sample_id, expected, predicted in zip(
        batch["sample_id"],
        batch["grid"],
        predictions,
        strict=True,
    ):
        report.append(
            {
                "sample_id": sample_id,
                "expected": grid_to_placement(expected.flatten().tolist()),
                "predicted": grid_to_placement(predicted.flatten().tolist()),
                "incorrect_squares": int(expected.ne(predicted).sum()),
            }
        )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
