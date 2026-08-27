from __future__ import annotations

from chess_repr.data.schema import SampleRecord
from chess_repr.data.splits import position_disjoint_validation


def _record(sample_id: str, placement: str, split: str) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        dataset_source="test",
        image_path=f"{sample_id}.jpg",
        placement_fen=placement,
        split=split,
        group_id=sample_id,
        license="test-only",
    )


def test_position_disjoint_validation_excludes_train_seen_placements() -> None:
    starting = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    kings = "4k3/8/8/8/8/8/8/4K3"
    training = [_record("train", starting, "train")]
    validation = [
        _record("seen", starting, "validation"),
        _record("unseen", kings, "validation"),
        _record("wrong-split", kings, "test"),
    ]
    kept, excluded = position_disjoint_validation(training, validation)
    assert [record.sample_id for record in kept] == ["unseen"]
    assert [record.sample_id for record in excluded] == ["seen"]
