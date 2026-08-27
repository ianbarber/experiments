"""Objectives and board-level metrics for the structured VAE."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor
from torch.nn import functional as functional

from chess_repr.models.structured_vae import StructuredVAEOutput


class Losses(TypedDict):
    total: Tensor
    semantic: Tensor
    reconstruction: Tensor
    reconstruction_objective: Tensor
    semantic_perceptual: Tensor
    kl: Tensor


def structured_vae_loss(
    output: StructuredVAEOutput,
    images: Tensor,
    grid: Tensor,
    *,
    semantic_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
    occupied_square_weight: float = 0.0,
    semantic_perceptual_loss: Tensor | None = None,
    semantic_perceptual_weight: float = 0.0,
    beta: float = 1e-4,
) -> Losses:
    semantic = functional.cross_entropy(output.semantic_logits, grid)
    absolute_error = (output.reconstruction - images).abs()
    reconstruction = absolute_error.mean()
    occupancy = functional.interpolate(
        grid.ne(0).unsqueeze(1).float(),
        size=images.shape[-2:],
        mode="nearest",
    )
    pixel_weights = 1.0 + occupied_square_weight * occupancy
    reconstruction_objective = (absolute_error * pixel_weights).sum() / (
        pixel_weights.sum() * images.shape[1]
    )
    kl = -0.5 * torch.mean(
        torch.sum(
            1
            + output.appearance_log_variance
            - output.appearance_mu.square()
            - output.appearance_log_variance.exp(),
            dim=1,
        )
    )
    semantic_perceptual = (
        output.reconstruction.new_zeros(())
        if semantic_perceptual_loss is None
        else semantic_perceptual_loss
    )
    total = (
        semantic_weight * semantic
        + reconstruction_weight * reconstruction_objective
        + semantic_perceptual_weight * semantic_perceptual
        + beta * kl
    )
    return {
        "total": total,
        "semantic": semantic,
        "reconstruction": reconstruction,
        "reconstruction_objective": reconstruction_objective,
        "semantic_perceptual": semantic_perceptual,
        "kl": kl,
    }


def board_metrics(logits: Tensor, grid: Tensor) -> dict[str, Tensor]:
    predictions = logits.argmax(dim=1)
    correct = predictions.eq(grid)
    errors_per_board = (~correct).flatten(1).sum(dim=1).float()
    return {
        "square_accuracy": correct.float().mean(),
        "exact_board_accuracy": correct.flatten(1).all(dim=1).float().mean(),
        "mean_square_errors": errors_per_board.mean(),
    }


def reconstruction_region_metrics(
    reconstruction: Tensor,
    images: Tensor,
    grid: Tensor,
) -> dict[str, Tensor]:
    absolute_error = (reconstruction - images).abs().mean(dim=1, keepdim=True)
    occupancy = functional.interpolate(
        grid.ne(0).unsqueeze(1).float(),
        size=images.shape[-2:],
        mode="nearest",
    ).bool()
    return {
        "occupied_square_l1": absolute_error[occupancy].mean(),
        "empty_square_l1": absolute_error[~occupancy].mean(),
    }
