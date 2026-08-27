#!/usr/bin/env python3
"""Train the semantic baseline or the hard-bottleneck structured VAE."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from chess_repr.data.dataset import ManifestImageDataset
from chess_repr.data.manifest import manifest_digest
from chess_repr.data.schema import Split
from chess_repr.fen import NUM_CLASSES
from chess_repr.models.structured_vae import BoardEncoder, StructuredVAE
from chess_repr.training.objectives import (
    board_metrics,
    reconstruction_region_metrics,
    structured_vae_loss,
)
from chess_repr.training.sampling import domain_balanced_sampler


def _source_ratios(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        source, separator, raw_ratio = value.partition("=")
        if not separator:
            raise ValueError(f"source ratio must be SOURCE=RATIO, got {value!r}")
        result[source] = float(raw_ratio)
    return result


def _average(totals: dict[str, float], examples: int) -> dict[str, float]:
    return {name: value / examples for name, value in sorted(totals.items())}


def _add_metrics(totals: dict[str, float], metrics: dict[str, Tensor], batch_size: int) -> None:
    for name, value in metrics.items():
        totals[name] += float(value.detach()) * batch_size


def _load_frozen_semantic_encoder(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> BoardEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    configuration = checkpoint["configuration"]
    source_model = StructuredVAE(
        appearance_dimensions=int(configuration["appearance_dimensions"]),
        base_channels=int(configuration["base_channels"]),
    )
    source_model.load_state_dict(checkpoint["model"])
    encoder = source_model.encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def _semantic_perceptual_loss(
    encoder: BoardEncoder | None,
    reconstruction: Tensor,
    grid: Tensor,
) -> Tensor | None:
    if encoder is None:
        return None
    return functional.cross_entropy(encoder(reconstruction).semantic_logits, grid)


@torch.inference_mode()
def evaluate(
    model: StructuredVAE,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    stage: str,
    max_batches: int | None,
    semantic_weight: float,
    reconstruction_weight: float,
    occupied_square_weight: float,
    semantic_perceptual_encoder: BoardEncoder | None,
    semantic_perceptual_weight: float,
    beta: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    examples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        grid = batch["grid"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            if stage == "semantic":
                encoded = model.encoder(images)
                loss = functional.cross_entropy(encoded.semantic_logits, grid)
                metrics = {"loss": loss, **board_metrics(encoded.semantic_logits, grid)}
            else:
                semantic_override = None
                if stage == "decoder":
                    semantic_override = functional.one_hot(
                        grid,
                        num_classes=NUM_CLASSES,
                    ).permute(0, 3, 1, 2).float()
                output = model(images, sample=False, semantic_override=semantic_override)
                perceptual_loss = _semantic_perceptual_loss(
                    semantic_perceptual_encoder,
                    output.reconstruction,
                    grid,
                )
                losses = structured_vae_loss(
                    output,
                    images,
                    grid,
                    semantic_weight=semantic_weight,
                    reconstruction_weight=reconstruction_weight,
                    occupied_square_weight=occupied_square_weight,
                    semantic_perceptual_loss=perceptual_loss,
                    semantic_perceptual_weight=semantic_perceptual_weight,
                    beta=beta,
                )
                metrics = {
                    "loss": losses["total"],
                    "semantic_loss": losses["semantic"],
                    "reconstruction_l1": losses["reconstruction"],
                    "reconstruction_objective": losses["reconstruction_objective"],
                    "semantic_perceptual_loss": losses["semantic_perceptual"],
                    "kl": losses["kl"],
                    **board_metrics(output.semantic_logits, grid),
                    **reconstruction_region_metrics(output.reconstruction, images, grid),
                }
        batch_size = images.shape[0]
        _add_metrics(totals, metrics, batch_size)
        examples += batch_size
    if examples == 0:
        raise ValueError("evaluation loader produced no examples")
    return _average(totals, examples)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--validation-manifest", type=Path, action="append", default=[])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("semantic", "decoder", "joint"),
        default="semantic",
    )
    parser.add_argument("--source-ratio", action="append", default=[])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validation-max-batches", type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--appearance-dimensions", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--semantic-temperature", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--occupied-square-weight", type=float, default=0.0)
    parser.add_argument("--semantic-perceptual-checkpoint", type=Path)
    parser.add_argument("--semantic-perceptual-weight", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ManifestImageDataset.from_manifests(
        args.train_manifest,
        splits=(Split.TRAIN,),
    )
    validation_dataset = None
    if args.validation_manifest:
        validation_dataset = ManifestImageDataset.from_manifests(
            args.validation_manifest,
            splits=(Split.VALIDATION,),
        )

    source_ratios = _source_ratios(args.source_ratio)
    sampler = None
    if source_ratios:
        sampler = domain_balanced_sampler(
            train_dataset.records,
            source_ratios,
            seed=args.seed,
        )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=sampler is None,
        **loader_options,
    )
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **loader_options)
        if validation_dataset is not None
        else None
    )

    model = StructuredVAE(
        appearance_dimensions=args.appearance_dimensions,
        base_channels=args.base_channels,
    ).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
    if args.semantic_perceptual_weight and not args.semantic_perceptual_checkpoint:
        raise ValueError(
            "--semantic-perceptual-weight requires --semantic-perceptual-checkpoint"
        )
    semantic_perceptual_encoder = (
        _load_frozen_semantic_encoder(
            args.semantic_perceptual_checkpoint,
            device=device,
        )
        if args.semantic_perceptual_checkpoint
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        **vars(args),
        "train_manifest": [str(path) for path in args.train_manifest],
        "validation_manifest": [str(path) for path in args.validation_manifest],
        "resume": str(args.resume) if args.resume else None,
        "semantic_perceptual_checkpoint": (
            str(args.semantic_perceptual_checkpoint)
            if args.semantic_perceptual_checkpoint
            else None
        ),
        "run_dir": str(args.run_dir),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": str(torch.__version__),
        "manifest_sha256": {
            str(path): manifest_digest(path)
            for path in [*args.train_manifest, *args.validation_manifest]
        },
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset) if validation_dataset else 0,
    }
    (args.run_dir / "config.json").write_text(json.dumps(configuration, indent=2, default=str))

    if args.eval_only:
        if validation_loader is None:
            raise ValueError("--eval-only requires --validation-manifest")
        validation_metrics = evaluate(
            model,
            validation_loader,
            device=device,
            stage=args.stage,
            max_batches=args.validation_max_batches,
            semantic_weight=args.semantic_weight,
            reconstruction_weight=args.reconstruction_weight,
            occupied_square_weight=args.occupied_square_weight,
            semantic_perceptual_encoder=semantic_perceptual_encoder,
            semantic_perceptual_weight=args.semantic_perceptual_weight,
            beta=args.beta,
        )
        result = {
            "elapsed_seconds": 0.0,
            "steps": 0,
            "train": None,
            "validation": validation_metrics,
        }
        (args.run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return

    started = time.perf_counter()
    global_step = 0
    train_totals: dict[str, float] = defaultdict(float)
    train_examples = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            grid = batch["grid"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                if args.stage == "semantic":
                    encoded = model.encoder(images)
                    loss = functional.cross_entropy(encoded.semantic_logits, grid)
                    metrics = {"loss": loss, **board_metrics(encoded.semantic_logits, grid)}
                else:
                    semantic_override = None
                    if args.stage == "decoder":
                        semantic_override = functional.one_hot(
                            grid,
                            num_classes=NUM_CLASSES,
                        ).permute(0, 3, 1, 2).float()
                    output = model(
                        images,
                        semantic_temperature=args.semantic_temperature,
                        semantic_override=semantic_override,
                    )
                    perceptual_loss = _semantic_perceptual_loss(
                        semantic_perceptual_encoder,
                        output.reconstruction,
                        grid,
                    )
                    losses = structured_vae_loss(
                        output,
                        images,
                        grid,
                        semantic_weight=args.semantic_weight,
                        reconstruction_weight=args.reconstruction_weight,
                        occupied_square_weight=args.occupied_square_weight,
                        semantic_perceptual_loss=perceptual_loss,
                        semantic_perceptual_weight=args.semantic_perceptual_weight,
                        beta=args.beta,
                    )
                    loss = losses["total"]
                    metrics = {
                        "loss": loss,
                        "semantic_loss": losses["semantic"],
                        "reconstruction_l1": losses["reconstruction"],
                        "reconstruction_objective": losses["reconstruction_objective"],
                        "semantic_perceptual_loss": losses["semantic_perceptual"],
                        "kl": losses["kl"],
                        **board_metrics(output.semantic_logits, grid),
                        **reconstruction_region_metrics(
                            output.reconstruction,
                            images,
                            grid,
                        ),
                    }
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.shape[0]
            _add_metrics(train_totals, metrics, batch_size)
            train_examples += batch_size
            global_step += 1
            if global_step % args.log_every == 0:
                progress = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    **_average(train_totals, train_examples),
                }
                print(json.dumps(progress), flush=True)
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    train_metrics = _average(train_totals, train_examples)
    validation_metrics = (
        evaluate(
            model,
            validation_loader,
            device=device,
            stage=args.stage,
            max_batches=args.validation_max_batches,
            semantic_weight=args.semantic_weight,
            reconstruction_weight=args.reconstruction_weight,
            occupied_square_weight=args.occupied_square_weight,
            semantic_perceptual_encoder=semantic_perceptual_encoder,
            semantic_perceptual_weight=args.semantic_perceptual_weight,
            beta=args.beta,
        )
        if validation_loader is not None
        else None
    )
    result = {
        "elapsed_seconds": time.perf_counter() - started,
        "steps": global_step,
        "train": train_metrics,
        "validation": validation_metrics,
    }
    torch.save(
        {"model": model.state_dict(), "configuration": configuration, "result": result},
        args.run_dir / "checkpoint.pt",
    )
    (args.run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
