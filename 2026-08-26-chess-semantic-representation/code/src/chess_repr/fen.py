"""Canonical conversion between piece-placement FEN and semantic class grids."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

BOARD_SIZE = 8
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
NUM_CLASSES = 13

CLASS_TO_SYMBOL: tuple[str, ...] = (
    ".",
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K",
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",
)
SYMBOL_TO_CLASS = {symbol: index for index, symbol in enumerate(CLASS_TO_SYMBOL)}
CLASS_NAMES: tuple[str, ...] = (
    "empty",
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
)


def placement_field(fen_or_placement: str) -> str:
    """Return and syntactically validate the piece-placement field of a FEN."""
    if not isinstance(fen_or_placement, str) or not fen_or_placement.strip():
        raise ValueError("FEN must be a non-empty string")
    placement = fen_or_placement.strip().split()[0]
    placement_to_grid(placement)
    return placement


def placement_to_grid(fen_or_placement: str) -> tuple[int, ...]:
    """Decode FEN rank order (a8..h1) into 64 semantic class integers."""
    if not isinstance(fen_or_placement, str) or not fen_or_placement.strip():
        raise ValueError("FEN must be a non-empty string")
    placement = fen_or_placement.strip().split()[0]
    ranks = placement.split("/")
    if len(ranks) != BOARD_SIZE:
        raise ValueError(f"piece placement must contain 8 ranks, got {len(ranks)}")

    grid: list[int] = []
    for rank_index, rank in enumerate(ranks):
        decoded_rank: list[int] = []
        for token in rank:
            if token.isdigit():
                empty_count = int(token)
                if not 1 <= empty_count <= BOARD_SIZE:
                    raise ValueError(f"invalid empty-square count {token!r} in rank {rank_index}")
                decoded_rank.extend([0] * empty_count)
            elif token in SYMBOL_TO_CLASS and token != ".":
                decoded_rank.append(SYMBOL_TO_CLASS[token])
            else:
                raise ValueError(f"invalid piece-placement token {token!r} in rank {rank_index}")
        if len(decoded_rank) != BOARD_SIZE:
            raise ValueError(
                f"rank {8 - rank_index} expands to {len(decoded_rank)} squares, expected 8"
            )
        grid.extend(decoded_rank)
    return tuple(grid)


def validate_grid(grid: Sequence[int] | Iterable[int]) -> tuple[int, ...]:
    """Return a normalized tuple after validating semantic grid dimensions/classes."""
    normalized = tuple(int(value) for value in grid)
    if len(normalized) != NUM_SQUARES:
        raise ValueError(f"grid must contain 64 values, got {len(normalized)}")
    invalid = sorted({value for value in normalized if not 0 <= value < NUM_CLASSES})
    if invalid:
        raise ValueError(f"grid contains invalid classes: {invalid}")
    return normalized


def grid_to_placement(grid: Sequence[int] | Iterable[int]) -> str:
    """Serialize 64 semantic classes to canonical piece-placement FEN."""
    normalized = validate_grid(grid)
    ranks: list[str] = []
    for rank_start in range(0, NUM_SQUARES, BOARD_SIZE):
        rank_tokens: list[str] = []
        empty_run = 0
        for class_id in normalized[rank_start : rank_start + BOARD_SIZE]:
            if class_id == 0:
                empty_run += 1
                continue
            if empty_run:
                rank_tokens.append(str(empty_run))
                empty_run = 0
            rank_tokens.append(CLASS_TO_SYMBOL[class_id])
        if empty_run:
            rank_tokens.append(str(empty_run))
        ranks.append("".join(rank_tokens))
    return "/".join(ranks)


def position_id(fen_or_placement: str) -> str:
    """Stable content identifier for a canonical visible board state."""
    canonical = grid_to_placement(placement_to_grid(fen_or_placement))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()

