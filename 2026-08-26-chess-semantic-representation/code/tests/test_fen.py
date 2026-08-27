from __future__ import annotations

import chess
import pytest

from chess_repr.fen import grid_to_placement, placement_to_grid, position_id


@pytest.mark.parametrize(
    "placement",
    [
        "8/8/8/8/8/8/8/8",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        "3r3r/b3npp1/P1k4p/2pp4/2P2B2/5B2/5PPP/R4RK1",
    ],
)
def test_placement_round_trip(placement: str) -> None:
    assert grid_to_placement(placement_to_grid(placement)) == placement


def test_full_fen_uses_only_placement_field() -> None:
    full_fen = "8/8/8/8/4N3/8/8/8 b - - 17 42"
    assert grid_to_placement(placement_to_grid(full_fen)) == full_fen.split()[0]


def test_legal_game_positions_round_trip() -> None:
    board = chess.Board()
    for move in list(board.legal_moves)[:10]:
        candidate = board.copy()
        candidate.push(move)
        placement = candidate.board_fen()
        assert grid_to_placement(placement_to_grid(placement)) == placement


@pytest.mark.parametrize("invalid", ["8/8", "9/8/8/8/8/8/8/8", "x7/8/8/8/8/8/8/8"])
def test_invalid_placement_rejected(invalid: str) -> None:
    with pytest.raises(ValueError):
        placement_to_grid(invalid)


def test_position_id_ignores_nonvisual_fen_fields() -> None:
    placement = "8/8/8/8/8/8/8/8"
    assert position_id(placement) == position_id(f"{placement} b KQkq e3 10 50")

