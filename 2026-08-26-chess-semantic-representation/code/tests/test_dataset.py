from __future__ import annotations

from pathlib import Path

from PIL import Image

from chess_repr.data.dataset import ManifestImageDataset
from chess_repr.data.schema import SampleRecord, Split


def test_manifest_dataset_rectifies_and_returns_grid(tmp_path: Path) -> None:
    image_path = tmp_path / "board.jpg"
    Image.new("RGB", (100, 80), color=(100, 120, 140)).save(image_path)
    record = SampleRecord(
        sample_id="sample",
        dataset_source="test",
        image_path=str(image_path),
        placement_fen="8/8/8/8/8/8/8/8",
        split=Split.TRAIN,
        group_id="group",
        license="test-only",
        corners_px=((10, 5), (90, 5), (90, 75), (10, 75)),
    )
    item = ManifestImageDataset([record], output_size=64)[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["grid"].shape == (8, 8)
    assert 0 <= item["image"].min() <= item["image"].max() <= 1
