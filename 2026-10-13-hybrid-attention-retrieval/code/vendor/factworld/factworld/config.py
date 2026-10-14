"""World configuration.

Sizes are calibrated against canonical recall/state-tracking benchmarks (canonical MQAR only became a
real recall stress test at vocab 8192 / 64-KV / train-len 512 / ~10M params; the S_k word
problem at train-len 32 still extrapolated to 0% at 2-4x). They remain knobs — the exact
recall scale is meant to be empirically calibrated in the model phase (down until pure
Mamba-2 fails and the GDN-hybrid succeeds, mirroring the MQAR separation) — but the defaults
are deliberately in the non-trivial regime, NOT the 256-fact lookup-table regime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    seed: int = 0
    # Distinct namespace -> fully disjoint symbol sets. Used to mint auxiliary operator-worlds
    # whose agents/roles do NOT overlap the target world, so the transition operator must be
    # learned over name-binding rather than memorised as a fixed multiplication table (R2).
    id_namespace: str = ""

    # --- recall (static, multi-attribute; MQAR-calibrated scale) ---
    n_entities: int = 2048           # 2048 x 4 attrs = 8192 associations (was a trivial 256)
    n_attributes: int = 4
    value_vocab_size: int = 1024     # SHARED, opaque value pool across all attributes
    freq_skew: float = 1.0           # Zipf exponent for entity document-frequency (0 = uniform)

    # --- easy-state (possession/location, last-write-wins) ---
    # A holder may be ANY location or agent, and both event kinds draw from that union, so the
    # holder value never reveals which event last touched the object (R4-ii).
    n_objects: int = 24
    n_locations: int = 16

    # --- hard-state (role permutation; word problem over S_k) ---
    k: int = 5                       # parameterise {3,4,5} across worlds; k=4 is the control rung
    swap_prob: float = 0.7           # P(transposition); the rest are cycles of length >= min_cycle_len
    min_cycle_len: int = 3           # >= 3 kills the degenerate 2-cycle == transposition (R4-iii)

    # --- planned chain-length bands (in events), per task; episode generation lives in M2/M3 ---
    easy_train_lengths: tuple[int, ...] = (8, 16)
    easy_ood_lengths: tuple[int, ...] = (4, 32, 64, 128)
    # hard-state trains at >= 32 to match the proven-hard S_k regime, with a wide OOD fan (R3)
    hard_train_lengths: tuple[int, ...] = (16, 32)
    hard_ood_lengths: tuple[int, ...] = (8, 48, 64, 96, 128)
