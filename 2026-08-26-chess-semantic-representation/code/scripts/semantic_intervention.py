#!/usr/bin/env python3
"""Swap a semantic board state while holding the appearance code fixed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as functional
from torchvision.utils import make_grid, save_image

from chess_repr.data.dataset import ManifestImageDataset
from chess_repr.data.schema import Split
from chess_repr.fen import NUM_CLASSES, grid_to_placement
from chess_repr.models.structured_vae import StructuredVAE


def _one_hot(grid: torch.Tensor) -> torch.Tensor:
    return functional.one_hot(grid, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(Split), default=Split.VALIDATION.value)
    parser.add_argument("--index-a", type=int, default=0)
    parser.add_argument("--index-b", type=int, default=5)
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

    dataset = ManifestImageDataset.from_manifests([args.manifest], splits=(args.split,))
    item_a = dataset[args.index_a]
    item_b = dataset[args.index_b]
    images = torch.stack((item_a["image"], item_b["image"])).to(device)
    grids = torch.stack((item_a["grid"], item_b["grid"])).to(device)
    semantics = _one_hot(grids)

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        appearance_a = model.encoder(images[:1]).appearance_mu
        reconstruction_a = model.decoder(semantics[:1], appearance_a)
        semantic_swap = model.decoder(semantics[1:2], appearance_a)

    difference = (reconstruction_a - semantic_swap).abs().mean(dim=1, keepdim=True)
    changed_squares = grids[:1].ne(grids[1:2]).unsqueeze(1)
    changed_pixels = functional.interpolate(
        changed_squares.float(),
        size=difference.shape[-2:],
        mode="nearest",
    ).bool()
    changed_difference = float(difference[changed_pixels].mean())
    unchanged_difference = float(difference[~changed_pixels].mean())

    difference_visual = difference.repeat(1, 3, 1, 1).mul(5).clamp(0, 1)
    tiles = torch.cat(
        (
            images.detach().cpu(),
            reconstruction_a.detach().cpu(),
            semantic_swap.detach().cpu(),
            difference_visual.detach().cpu(),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(tiles, nrow=2, padding=4), args.output)

    report = {
        "layout": [
            "input A",
            "input B",
            "decode(s_A, z_A)",
            "decode(s_B, z_A)",
            "5x absolute decode difference",
        ],
        "sample_a": item_a["sample_id"],
        "sample_b": item_b["sample_id"],
        "fen_a": grid_to_placement(grids[0].flatten().tolist()),
        "fen_b": grid_to_placement(grids[1].flatten().tolist()),
        "changed_squares": int(changed_squares.sum()),
        "mean_change_in_changed_squares": changed_difference,
        "mean_change_in_unchanged_squares": unchanged_difference,
        "locality_ratio": changed_difference / max(unchanged_difference, 1e-12),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
