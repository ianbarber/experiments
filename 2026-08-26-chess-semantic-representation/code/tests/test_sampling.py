from __future__ import annotations

from chess_repr.data.schema import SampleRecord
from chess_repr.training.sampling import domain_balanced_sampler


def _record(sample_id: str, source: str) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        dataset_source=source,
        image_path=f"{sample_id}.jpg",
        placement_fen="8/8/8/8/8/8/8/8",
        split="train",
        group_id=sample_id,
        license="test-only",
    )


def test_domain_balanced_sampler_offsets_archive_sizes() -> None:
    records = [*[_record(f"a{i}", "a") for i in range(9)], _record("b0", "b")]
    sampler = domain_balanced_sampler(records, {"a": 0.5, "b": 0.5}, seed=4)
    sampled_sources = [records[index].dataset_source for index in list(sampler) * 100]
    fraction_b = sampled_sources.count("b") / len(sampled_sources)
    assert 0.35 < fraction_b < 0.65
