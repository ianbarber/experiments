from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from chess_repr.data.chesssight import CORNER_ORDER, extract_chesssight_records
from chess_repr.data.schema import Split


def _jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color=(20, 40, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_extract_chesssight_records(tmp_path: Path) -> None:
    parquet_path = tmp_path / "data.parquet"
    grid = [0] * 64
    grid[4] = 12
    grid[60] = 6
    table = pa.table(
        {
            "image": [{"bytes": _jpeg_bytes(), "path": "frame.jpg"}],
            "id": ["train0/000001"],
            "source_run": ["train0"],
            "fen": ["4k3/8/8/8/8/8/8/4K3 w - - 0 1"],
            "grid": [grid],
            "corners_px": [[[1.0, 2.0], [14.0, 2.0], [14.0, 14.0], [1.0, 14.0]]],
            "width": [16],
            "height": [16],
        }
    )
    pq.write_table(table, parquet_path)

    records = extract_chesssight_records(
        parquet_path,
        image_root=tmp_path / "extracted",
        split=Split.TRAIN,
    )
    assert len(records) == 1
    record = records[0]
    assert record.sample_id == "chesssight-train0-000001"
    assert record.placement_fen == "4k3/8/8/8/8/8/8/4K3"
    assert record.grid == tuple(grid)
    assert record.split == Split.TRAIN
    assert Path(record.image_path).is_file()
    assert record.metadata["corner_order"] == list(CORNER_ORDER)
