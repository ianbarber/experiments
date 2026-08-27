from __future__ import annotations

import json
from pathlib import Path

from chess_repr.data.chessred import load_chessred_records, square_index
from chess_repr.data.schema import Split
from chess_repr.fen import placement_to_grid


def test_square_index_uses_a8_row_major_order() -> None:
    assert square_index("a8") == 0
    assert square_index("h8") == 7
    assert square_index("a1") == 56
    assert square_index("h1") == 63


def test_chessred_adapter(tmp_path: Path) -> None:
    raw = {
        "images": [
            {
                "id": 3,
                "path": "images/9/example.jpg",
                "camera": "camera-a",
                "height": 100,
                "width": 120,
                "game_id": 9,
                "move_id": 2,
            }
        ],
        "categories": [
            {"id": 0, "name": "white-king"},
            {"id": 1, "name": "black-king"},
            {"id": 12, "name": "empty"},
        ],
        "annotations": {
            "pieces": [
                {"image_id": 3, "category_id": 0, "chessboard_position": "e1"},
                {"image_id": 3, "category_id": 1, "chessboard_position": "e8"},
            ],
            "corners": [
                {
                    "image_id": 3,
                    "corners": {
                        "top_left": [0, 0],
                        "top_right": [100, 0],
                        "bottom_right": [100, 100],
                        "bottom_left": [0, 100],
                    },
                }
            ],
        },
        "splits": {
            "train": {"image_ids": [3], "n_samples": 1},
            "val": {"image_ids": [], "n_samples": 0},
            "test": {"image_ids": [], "n_samples": 0},
            "chessred2k": {
                "train": {"image_ids": [3], "n_samples": 1},
                "val": {"image_ids": [], "n_samples": 0},
                "test": {"image_ids": [], "n_samples": 0},
            },
        },
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(raw))
    records = load_chessred_records(path, image_root=Path("root"))
    assert len(records) == 1
    record = records[0]
    assert record.split == Split.TRAIN
    assert record.group_id == "chessred-game-9"
    assert record.image_path == "root/images/9/example.jpg"
    grid = placement_to_grid(record.placement_fen)
    assert grid[square_index("e1")] == 6
    assert grid[square_index("e8")] == 12
    assert record.corners_px == ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))

