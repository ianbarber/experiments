from __future__ import annotations

from pathlib import Path

import pytest

from chess_repr.data.audit import audit_records
from chess_repr.data.manifest import read_manifest, summarize_manifest, write_manifest
from chess_repr.data.schema import SampleRecord, Split

EMPTY = "8/8/8/8/8/8/8/8"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def record(sample_id: str, split: Split, group: str, placement: str = EMPTY) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        dataset_source="test",
        image_path=f"images/{sample_id}.png",
        placement_fen=placement,
        split=split,
        group_id=group,
        license="CC0-1.0",
    )


def test_record_normalizes_grid_and_position_id() -> None:
    sample = record("one", Split.TRAIN, "g1")
    assert len(sample.grid) == 64
    assert set(sample.grid) == {0}
    assert sample.position_id


def test_mismatched_grid_rejected() -> None:
    with pytest.raises(ValueError, match="different board states"):
        SampleRecord(
            sample_id="bad",
            dataset_source="test",
            image_path="bad.png",
            placement_fen=EMPTY,
            grid=[1] + [0] * 63,
            split=Split.TRAIN,
            group_id="bad",
            license="CC0-1.0",
        )


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    original = [record("one", Split.TRAIN, "g1"), record("two", Split.TEST, "g2", START)]
    digest = write_manifest(original, path)
    restored = read_manifest(path)
    assert restored == original
    assert len(digest) == 64
    assert summarize_manifest(restored)["unique_positions"] == 2


def test_group_cross_split_is_error() -> None:
    report = audit_records(
        [record("one", Split.TRAIN, "same"), record("two", Split.TEST, "same", START)]
    )
    assert not report.ok
    assert {issue.code for issue in report.issues} >= {"group_crosses_splits"}


def test_position_cross_split_is_warning_only() -> None:
    report = audit_records(
        [record("one", Split.TRAIN, "g1"), record("two", Split.TEST, "g2")]
    )
    assert report.ok
    assert {issue.code for issue in report.issues} == {"position_crosses_splits"}


def test_sealed_physical_board_overlap_is_error() -> None:
    train = record("one", Split.TRAIN, "g1")
    sealed = record("two", Split.SEALED_TEST, "g2", START)
    object.__setattr__(train, "physical_board_id", "board-a")
    object.__setattr__(sealed, "physical_board_id", "board-a")
    assert not audit_records([train, sealed]).ok
