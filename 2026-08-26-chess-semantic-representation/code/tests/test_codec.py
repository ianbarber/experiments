from __future__ import annotations

import pytest
import torch

from chess_repr.codec import (
    HEADER,
    decode_representation,
    decode_semantic_base,
    encode_representation,
)


def test_representation_round_trip_and_progressive_prefix() -> None:
    grid = torch.arange(64).remainder(13)
    appearance = torch.linspace(-2, 3, 64)
    encoded = encode_representation(grid, appearance)
    decoded = decode_representation(encoded.data)
    assert decoded.grid == tuple(grid.tolist())
    assert decode_semantic_base(encoded.data[: encoded.base_bytes]) == decoded.grid
    assert encoded.base_bytes == HEADER.size + encoded.semantic_stream_bytes
    maximum_error = (decoded.appearance - appearance).abs().max()
    assert maximum_error <= encoded.quantization_scale / 2 + 1e-6


def test_representation_checksum_rejects_corruption() -> None:
    encoded = encode_representation(torch.zeros(64), torch.ones(8))
    corrupted = bytearray(encoded.data)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError):
        decode_representation(bytes(corrupted))
