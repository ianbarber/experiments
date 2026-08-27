#!/usr/bin/env python3
"""Fit linear probes for board leakage and source/style separation."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from chess_repr.data.dataset import ManifestImageDataset
from chess_repr.data.manifest import read_manifest
from chess_repr.data.schema import SampleRecord, Split
from chess_repr.fen import NUM_CLASSES
from chess_repr.models.structured_vae import StructuredVAE
from chess_repr.training.objectives import board_metrics


def _balanced_records(
    records: list[SampleRecord],
    *,
    split: Split,
    per_source: int,
    seed: int,
) -> list[SampleRecord]:
    by_source: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        if record.split == split:
            by_source[record.dataset_source].append(record)
    generator = random.Random(seed)
    selected = []
    for source_records in by_source.values():
        generator.shuffle(source_records)
        selected.extend(source_records[:per_source])
    generator.shuffle(selected)
    return selected


@torch.inference_mode()
def _extract(
    model: StructuredVAE,
    records: list[SampleRecord],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    source_to_class: dict[str, int],
) -> dict[str, Tensor]:
    loader = DataLoader(
        ManifestImageDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    extracted: dict[str, list[Tensor]] = defaultdict(list)
    for batch in loader:
        images = batch["image"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            encoded = model.encoder(images)
        hard = functional.one_hot(
            encoded.semantic_logits.argmax(dim=1),
            num_classes=NUM_CLASSES,
        ).flatten(1).float()
        extracted["appearance"].append(encoded.appearance_mu.float().cpu())
        extracted["semantic"].append(hard.cpu())
        extracted["grid"].append(batch["grid"].long())
        extracted["source"].append(
            torch.tensor([source_to_class[source] for source in batch["source"]])
        )
    return {name: torch.cat(values) for name, values in extracted.items()}


def _fit_linear(
    inputs: Tensor,
    targets: Tensor,
    *,
    outputs: int,
    steps: int,
    seed: int,
    device: torch.device,
) -> nn.Linear:
    torch.manual_seed(seed)
    model = nn.Linear(inputs.shape[1], outputs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    inputs = inputs.to(device)
    targets = targets.to(device)
    for _ in range(steps):
        indices = torch.randint(0, inputs.shape[0], (min(256, inputs.shape[0]),), device=device)
        logits = model(inputs[indices])
        selected_targets = targets[indices]
        if selected_targets.ndim == 2:
            loss = functional.cross_entropy(
                logits.reshape(-1, NUM_CLASSES),
                selected_targets.flatten(),
            )
        else:
            loss = functional.cross_entropy(logits, selected_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.inference_mode()
def _source_accuracy(
    model: nn.Linear,
    inputs: Tensor,
    targets: Tensor,
    device: torch.device,
) -> float:
    predictions = model(inputs.to(device)).argmax(dim=1).cpu()
    return float(predictions.eq(targets).float().mean())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-source", type=int, default=512)
    parser.add_argument("--validation-per-source", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    configuration: dict[str, Any] = checkpoint["configuration"]
    model = StructuredVAE(
        appearance_dimensions=int(configuration["appearance_dimensions"]),
        base_channels=int(configuration["base_channels"]),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_training = [
        record for path in args.train_manifest for record in read_manifest(path)
    ]
    all_validation = read_manifest(args.validation_manifest)
    train_records = _balanced_records(
        all_training,
        split=Split.TRAIN,
        per_source=args.train_per_source,
        seed=args.seed,
    )
    validation_records = _balanced_records(
        all_validation,
        split=Split.VALIDATION,
        per_source=args.validation_per_source,
        seed=args.seed,
    )
    sources = sorted({record.dataset_source for record in train_records})
    source_to_class = {source: index for index, source in enumerate(sources)}
    train = _extract(
        model,
        train_records,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        source_to_class=source_to_class,
    )
    validation = _extract(
        model,
        validation_records,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        source_to_class=source_to_class,
    )

    appearance_mean = train["appearance"].mean(dim=0)
    appearance_std = train["appearance"].std(dim=0).clamp_min(1e-5)
    train_appearance = (train["appearance"] - appearance_mean) / appearance_std
    validation_appearance = (validation["appearance"] - appearance_mean) / appearance_std

    appearance_board_probe = _fit_linear(
        train_appearance,
        train["grid"].flatten(1),
        outputs=64 * NUM_CLASSES,
        steps=args.steps,
        seed=args.seed,
        device=device,
    )
    with torch.inference_mode():
        board_logits = appearance_board_probe(validation_appearance.to(device)).reshape(
            -1, 64, NUM_CLASSES
        ).permute(0, 2, 1).reshape(-1, NUM_CLASSES, 8, 8)
    board_probe_metrics = {
        name: float(value)
        for name, value in board_metrics(
            board_logits.cpu(),
            validation["grid"],
        ).items()
    }

    semantic_source_probe = _fit_linear(
        train["semantic"],
        train["source"],
        outputs=len(sources),
        steps=args.steps,
        seed=args.seed + 1,
        device=device,
    )
    appearance_source_probe = _fit_linear(
        train_appearance,
        train["source"],
        outputs=len(sources),
        steps=args.steps,
        seed=args.seed + 2,
        device=device,
    )

    majority_grid = []
    for square in range(64):
        square_classes = train["grid"][:, square // 8, square % 8].tolist()
        majority_grid.append(Counter(square_classes).most_common(1)[0][0])
    majority = torch.tensor(majority_grid).reshape(1, 8, 8).expand_as(validation["grid"])
    majority_correct = majority.eq(validation["grid"])
    report = {
        "checkpoint": str(args.checkpoint),
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "sources": sources,
        "position_disjoint_validation": True,
        "appearance_board_probe": board_probe_metrics,
        "majority_board_baseline": {
            "square_accuracy": float(majority_correct.float().mean()),
            "exact_board_accuracy": float(
                majority_correct.flatten(1).all(dim=1).float().mean()
            ),
        },
        "semantic_source_probe_accuracy": _source_accuracy(
            semantic_source_probe,
            validation["semantic"],
            validation["source"],
            device,
        ),
        "appearance_source_probe_accuracy": _source_accuracy(
            appearance_source_probe,
            validation_appearance,
            validation["source"],
            device,
        ),
        "source_chance_accuracy": 1 / len(sources),
        "caveat": (
            "Hard-semantic source prediction includes real differences in position "
            "distributions and is not a pure visual-style leakage measure."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
