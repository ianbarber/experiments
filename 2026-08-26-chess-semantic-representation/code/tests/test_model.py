from __future__ import annotations

import torch

from chess_repr.models.structured_vae import StructuredVAE
from chess_repr.training.objectives import board_metrics, structured_vae_loss


def test_structured_vae_shapes_and_gradients() -> None:
    model = StructuredVAE(appearance_dimensions=8, base_channels=8)
    images = torch.rand(2, 3, 256, 256)
    grid = torch.randint(0, 13, (2, 8, 8))
    output = model(images, sample=False)
    assert output.semantic_logits.shape == (2, 13, 8, 8)
    assert output.hard_semantic.shape == (2, 13, 8, 8)
    assert output.appearance_mu.shape == (2, 8)
    assert output.reconstruction.shape == images.shape
    assert torch.allclose(output.hard_semantic.sum(dim=1), torch.ones(2, 8, 8))

    losses = structured_vae_loss(output, images, grid)
    assert torch.allclose(losses["reconstruction"], losses["reconstruction_objective"])
    perceptual = torch.tensor(0.75)
    weighted_losses = structured_vae_loss(
        output,
        images,
        grid,
        semantic_perceptual_loss=perceptual,
        semantic_perceptual_weight=0.2,
    )
    assert torch.allclose(
        weighted_losses["total"],
        losses["total"] + 0.2 * perceptual,
    )
    losses["total"].backward()
    assert model.encoder.semantic_head.weight.grad is not None
    assert model.decoder.upsample[-2].weight.grad is not None


def test_exact_board_metric_is_stricter_than_square_accuracy() -> None:
    grid = torch.zeros(2, 8, 8, dtype=torch.long)
    logits = torch.zeros(2, 13, 8, 8)
    logits[:, 0] = 1
    logits[0, 1, 0, 0] = 2
    metrics = board_metrics(logits, grid)
    assert metrics["square_accuracy"].item() == 127 / 128
    assert metrics["exact_board_accuracy"].item() == 0.5
    assert metrics["mean_square_errors"].item() == 0.5
