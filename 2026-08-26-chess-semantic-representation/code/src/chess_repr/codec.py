"""A small, versioned semantic-base plus quantized-appearance bitstream."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

from chess_repr.fen import validate_grid

MAGIC: Final = b"CRP1"
VERSION: Final = 1
HEADER = struct.Struct("<4sBHfHHI")


@dataclass(frozen=True, slots=True)
class EncodedRepresentation:
    data: bytes
    base_bytes: int
    semantic_stream_bytes: int
    appearance_stream_bytes: int
    quantization_scale: float


@dataclass(frozen=True, slots=True)
class DecodedRepresentation:
    grid: tuple[int, ...]
    appearance: Tensor
    quantization_scale: float


def _pack_grid(grid: tuple[int, ...]) -> bytes:
    return bytes(grid[index] | (grid[index + 1] << 4) for index in range(0, 64, 2))


def _unpack_grid(data: bytes) -> tuple[int, ...]:
    if len(data) != 32:
        raise ValueError(f"semantic payload must be 32 bytes, got {len(data)}")
    grid = tuple(value for packed in data for value in (packed & 0x0F, packed >> 4))
    return validate_grid(grid)


def encode_representation(
    grid: Tensor | tuple[int, ...] | list[int],
    appearance: Tensor,
) -> EncodedRepresentation:
    """Encode a 64-category grid and one appearance vector to finite bytes."""
    if isinstance(grid, Tensor):
        normalized_grid = validate_grid(grid.detach().cpu().flatten().tolist())
    else:
        normalized_grid = validate_grid(grid)
    flat_appearance = appearance.detach().float().cpu().flatten()
    if not 0 < flat_appearance.numel() <= 65535:
        raise ValueError("appearance vector length must fit uint16 and be nonzero")

    maximum = float(flat_appearance.abs().max())
    scale = maximum / 127.0 if maximum > 0 else 1.0
    quantized = torch.round(flat_appearance / scale).clamp(-127, 127).to(torch.int8)
    semantic_raw = _pack_grid(normalized_grid)
    appearance_raw = quantized.numpy().tobytes()
    semantic_stream = zlib.compress(semantic_raw, level=9)
    appearance_stream = zlib.compress(appearance_raw, level=9)
    checksum = zlib.crc32(semantic_raw + appearance_raw)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        flat_appearance.numel(),
        scale,
        len(semantic_stream),
        len(appearance_stream),
        checksum,
    )
    data = header + semantic_stream + appearance_stream
    return EncodedRepresentation(
        data=data,
        base_bytes=len(header) + len(semantic_stream),
        semantic_stream_bytes=len(semantic_stream),
        appearance_stream_bytes=len(appearance_stream),
        quantization_scale=scale,
    )


def _header(data: bytes) -> tuple[int, float, int, int, int]:
    if len(data) < HEADER.size:
        raise ValueError("truncated representation header")
    magic, version, dimensions, scale, semantic_length, appearance_length, checksum = (
        HEADER.unpack_from(data)
    )
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported representation bitstream")
    if dimensions <= 0 or scale <= 0:
        raise ValueError("invalid representation header values")
    return dimensions, scale, semantic_length, appearance_length, checksum


def decode_semantic_base(data: bytes) -> tuple[int, ...]:
    """Decode the independently useful prefix containing only semantic state."""
    _, _, semantic_length, _, _ = _header(data)
    semantic_end = HEADER.size + semantic_length
    if len(data) < semantic_end:
        raise ValueError("truncated semantic base layer")
    try:
        semantic_raw = zlib.decompress(data[HEADER.size:semantic_end])
    except zlib.error as error:
        raise ValueError("invalid semantic stream") from error
    return _unpack_grid(semantic_raw)


def decode_representation(data: bytes) -> DecodedRepresentation:
    dimensions, scale, semantic_length, appearance_length, checksum = _header(data)
    semantic_end = HEADER.size + semantic_length
    appearance_end = semantic_end + appearance_length
    if len(data) != appearance_end:
        raise ValueError("truncated or trailing representation bytes")
    try:
        semantic_raw = zlib.decompress(data[HEADER.size:semantic_end])
        appearance_raw = zlib.decompress(data[semantic_end:appearance_end])
    except zlib.error as error:
        raise ValueError("invalid compressed representation stream") from error
    if len(appearance_raw) != dimensions:
        raise ValueError("appearance payload has the wrong length")
    if zlib.crc32(semantic_raw + appearance_raw) != checksum:
        raise ValueError("representation checksum mismatch")
    quantized = torch.frombuffer(bytearray(appearance_raw), dtype=torch.int8).clone()
    return DecodedRepresentation(
        grid=_unpack_grid(semantic_raw),
        appearance=quantized.float().mul_(scale),
        quantization_scale=scale,
    )
