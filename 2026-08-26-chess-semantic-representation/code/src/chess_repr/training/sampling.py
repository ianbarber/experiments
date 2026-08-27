"""Sampling helpers for explicitly controlled source blends."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import WeightedRandomSampler

from chess_repr.data.schema import SampleRecord


def domain_balanced_sampler(
    records: Sequence[SampleRecord],
    source_ratios: Mapping[str, float],
    *,
    seed: int,
) -> WeightedRandomSampler:
    """Sample each source at a requested probability, independent of archive size."""
    if not records:
        raise ValueError("cannot sample an empty dataset")
    counts = Counter(record.dataset_source for record in records)
    missing = set(counts) - set(source_ratios)
    unknown = set(source_ratios) - set(counts)
    if missing or unknown:
        raise ValueError(
            f"source ratios mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if any(value <= 0 for value in source_ratios.values()):
        raise ValueError("source ratios must be positive")
    total = sum(source_ratios.values())
    weights = [
        source_ratios[record.dataset_source] / total / counts[record.dataset_source]
        for record in records
    ]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )
