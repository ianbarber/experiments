#!/usr/bin/env python3
"""Measure actual structured-representation bytes and quantization distortion."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from chess_repr.codec import decode_representation, encode_representation
from chess_repr.data.dataset import ManifestImageDataset
from chess_repr.data.schema import Split
from chess_repr.fen import NUM_CLASSES
from chess_repr.models.structured_vae import StructuredVAE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(Split), default=Split.VALIDATION.value)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-bitstreams", type=int, default=8)
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    byte_counts: list[int] = []
    base_byte_counts: list[int] = []
    totals: dict[str, float] = defaultdict(float)
    source_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    examples = 0
    bitstream_dir = args.output.parent / "bitstreams"

    for batch in loader:
        if args.max_examples is not None and examples >= args.max_examples:
            break
        images = batch["image"].to(device)
        grids = batch["grid"].to(device)
        if args.max_examples is not None:
            remaining = args.max_examples - examples
            images = images[:remaining]
            grids = grids[:remaining]
            batch["sample_id"] = batch["sample_id"][:remaining]
            batch["source"] = batch["source"][:remaining]

        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(images, sample=False)

        decoded_grids = []
        decoded_appearances = []
        for offset in range(images.shape[0]):
            predicted_grid = output.semantic_logits[offset].argmax(dim=0)
            encoded = encode_representation(predicted_grid, output.appearance_mu[offset])
            decoded = decode_representation(encoded.data)
            decoded_grids.append(torch.tensor(decoded.grid, dtype=torch.long).reshape(8, 8))
            decoded_appearances.append(decoded.appearance)
            byte_counts.append(len(encoded.data))
            base_byte_counts.append(encoded.base_bytes)
            if examples + offset < args.save_bitstreams:
                bitstream_dir.mkdir(parents=True, exist_ok=True)
                (bitstream_dir / f"{batch['sample_id'][offset]}.crp").write_bytes(encoded.data)

        decoded_grid_tensor = torch.stack(decoded_grids).to(device)
        decoded_semantic = functional.one_hot(
            decoded_grid_tensor,
            num_classes=NUM_CLASSES,
        ).permute(0, 3, 1, 2).float()
        decoded_appearance = torch.stack(decoded_appearances).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            quantized_reconstruction = model.decoder(decoded_semantic, decoded_appearance)
            base_reconstruction = model.decode_semantic_base(decoded_semantic)

        prediction_correct = decoded_grid_tensor.eq(grids)
        batch_metrics = {
            "square_accuracy": prediction_correct.float().mean(dim=(1, 2)),
            "exact_board_accuracy": prediction_correct.flatten(1).all(dim=1).float(),
            "full_l1": (output.reconstruction - images).abs().mean(dim=(1, 2, 3)),
            "full_mse": (output.reconstruction - images).square().mean(dim=(1, 2, 3)),
            "quantized_l1": (quantized_reconstruction - images).abs().mean(dim=(1, 2, 3)),
            "quantized_mse": (
                quantized_reconstruction - images
            ).square().mean(dim=(1, 2, 3)),
            "base_l1": (base_reconstruction - images).abs().mean(dim=(1, 2, 3)),
            "quantization_delta_l1": (
                quantized_reconstruction - output.reconstruction
            ).abs().mean(dim=(1, 2, 3)),
        }
        for name, values in batch_metrics.items():
            totals[name] += float(values.sum())
            for source, value in zip(batch["source"], values, strict=True):
                source_totals[source][name] += float(value)
        for source in batch["source"]:
            source_totals[source]["examples"] += 1
        examples += images.shape[0]

    if examples == 0:
        raise ValueError("no examples evaluated")
    mean_bytes = statistics.fmean(byte_counts)
    mean_base_bytes = statistics.fmean(base_byte_counts)
    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "examples": examples,
        "image_size": 256,
        "appearance_dimensions": int(configuration["appearance_dimensions"]),
        "mean_total_bytes": mean_bytes,
        "median_total_bytes": statistics.median(byte_counts),
        "min_total_bytes": min(byte_counts),
        "max_total_bytes": max(byte_counts),
        "mean_semantic_base_bytes": mean_base_bytes,
        "mean_enhancement_bytes": mean_bytes - mean_base_bytes,
        "mean_bits_per_pixel": mean_bytes * 8 / (256 * 256),
        "mean_base_bits_per_pixel": mean_base_bytes * 8 / (256 * 256),
        "metrics": {name: value / examples for name, value in sorted(totals.items())},
        "by_source": {
            source: {
                name: value / values["examples"]
                for name, value in sorted(values.items())
                if name != "examples"
            }
            for source, values in sorted(source_totals.items())
        },
    }
    report["metrics"]["full_psnr_db"] = -10 * math.log10(
        max(report["metrics"]["full_mse"], 1e-12)
    )
    report["metrics"]["quantized_psnr_db"] = -10 * math.log10(
        max(report["metrics"]["quantized_mse"], 1e-12)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
