"""Hard semantic bottleneck plus rate-limited appearance VAE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from chess_repr.fen import NUM_CLASSES


def _encoder_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(min(16, output_channels), output_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.GroupNorm(min(16, output_channels), output_channels),
        nn.SiLU(inplace=True),
    )


def _decoder_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.GroupNorm(min(16, output_channels), output_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.GroupNorm(min(16, output_channels), output_channels),
        nn.SiLU(inplace=True),
    )


@dataclass(slots=True)
class EncoderOutput:
    semantic_logits: Tensor
    appearance_mu: Tensor
    appearance_log_variance: Tensor


@dataclass(slots=True)
class StructuredVAEOutput:
    reconstruction: Tensor
    semantic_logits: Tensor
    hard_semantic: Tensor
    appearance_mu: Tensor
    appearance_log_variance: Tensor
    appearance_sample: Tensor


class BoardEncoder(nn.Module):
    """Encode a 256px rectified board to named square states and appearance."""

    def __init__(self, *, appearance_dimensions: int = 64, base_channels: int = 32) -> None:
        super().__init__()
        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 6,
            base_channels * 8,
        )
        blocks: list[nn.Module] = []
        input_channels = 3
        for output_channels in channels:
            blocks.append(_encoder_block(input_channels, output_channels))
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.semantic_head = nn.Conv2d(channels[-1], NUM_CLASSES, kernel_size=1)
        self.appearance_pool = nn.AdaptiveAvgPool2d(1)
        self.appearance_head = nn.Linear(channels[-1], appearance_dimensions * 2)

    def forward(self, images: Tensor) -> EncoderOutput:
        features = self.features(images)
        if features.shape[-2:] != (8, 8):
            features = functional.adaptive_avg_pool2d(features, (8, 8))
        semantic_logits = self.semantic_head(features)
        appearance_parameters = self.appearance_head(
            self.appearance_pool(features).flatten(1)
        )
        appearance_mu, appearance_log_variance = appearance_parameters.chunk(2, dim=1)
        return EncoderOutput(semantic_logits, appearance_mu, appearance_log_variance)


class StructuredDecoder(nn.Module):
    def __init__(self, *, appearance_dimensions: int = 64, base_channels: int = 32) -> None:
        super().__init__()
        self.semantic_embedding = nn.Sequential(
            nn.Conv2d(NUM_CLASSES, base_channels * 2, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.appearance_projection = nn.Linear(
            appearance_dimensions,
            base_channels * 4 * 8 * 8,
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.GroupNorm(16, base_channels * 8),
            nn.SiLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            _decoder_block(base_channels * 8, base_channels * 8),
            _decoder_block(base_channels * 8, base_channels * 6),
            _decoder_block(base_channels * 6, base_channels * 4),
            _decoder_block(base_channels * 4, base_channels * 2),
            _decoder_block(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.base_channels = base_channels

    def forward(self, hard_semantic: Tensor, appearance: Tensor) -> Tensor:
        semantic_features = self.semantic_embedding(hard_semantic)
        appearance_features = self.appearance_projection(appearance).reshape(
            appearance.shape[0], self.base_channels * 4, 8, 8
        )
        return self.upsample(
            self.fusion(torch.cat((semantic_features, appearance_features), dim=1))
        )


class StructuredVAE(nn.Module):
    """A decoder-bottlenecked model with explicit 64-way semantic variables."""

    def __init__(self, *, appearance_dimensions: int = 64, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = BoardEncoder(
            appearance_dimensions=appearance_dimensions,
            base_channels=base_channels,
        )
        self.decoder = StructuredDecoder(
            appearance_dimensions=appearance_dimensions,
            base_channels=base_channels,
        )

    @staticmethod
    def reparameterize(mu: Tensor, log_variance: Tensor, *, sample: bool) -> Tensor:
        if not sample:
            return mu
        standard_deviation = torch.exp(0.5 * log_variance)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    @staticmethod
    def discretize(logits: Tensor, *, temperature: float, sample: bool) -> Tensor:
        if sample:
            return functional.gumbel_softmax(
                logits,
                tau=temperature,
                hard=True,
                dim=1,
            )
        indices = logits.argmax(dim=1)
        return functional.one_hot(indices, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()

    def forward(
        self,
        images: Tensor,
        *,
        semantic_temperature: float = 1.0,
        sample: bool | None = None,
        semantic_override: Tensor | None = None,
    ) -> StructuredVAEOutput:
        should_sample = self.training if sample is None else sample
        encoded = self.encoder(images)
        hard_semantic = (
            self.discretize(
                encoded.semantic_logits,
                temperature=semantic_temperature,
                sample=should_sample,
            )
            if semantic_override is None
            else semantic_override
        )
        appearance = self.reparameterize(
            encoded.appearance_mu,
            encoded.appearance_log_variance,
            sample=should_sample,
        )
        reconstruction = self.decoder(hard_semantic, appearance)
        return StructuredVAEOutput(
            reconstruction=reconstruction,
            semantic_logits=encoded.semantic_logits,
            hard_semantic=hard_semantic,
            appearance_mu=encoded.appearance_mu,
            appearance_log_variance=encoded.appearance_log_variance,
            appearance_sample=appearance,
        )

    def decode_semantic_base(self, semantic: Tensor) -> Tensor:
        appearance_dimensions = self.decoder.appearance_projection.in_features
        appearance = torch.zeros(
            semantic.shape[0],
            appearance_dimensions,
            device=semantic.device,
            dtype=semantic.dtype,
        )
        return self.decoder(semantic, appearance)
