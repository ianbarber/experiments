"""Architectural mechanism inventory and heterogeneity metrics.

This module complements ATen node counts with a hand-curated + config-derived
feature inventory. The goal is to capture *architectural* complexity (how many
distinct ideas a model combines) independently of *implementation* complexity
(how many ATen ops the exporter happens to see).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoConfig


@dataclass(frozen=True)
class Mechanism:
    id: str
    label: str
    category: str
    description: str


TAXONOMY: list[Mechanism] = [
    # Attention variants
    Mechanism(
        "full_attention",
        "Full softmax attention",
        "attention",
        "Standard multi-head softmax attention over the full context.",
    ),
    Mechanism(
        "linear_attention",
        "Linear / state-space attention",
        "attention",
        "Kernelized or recurrent linear attention (e.g., DeltaNet, state-space).",
    ),
    Mechanism(
        "sliding_window_attention",
        "Sliding / local attention",
        "attention",
        "Attention restricted to a local sliding window.",
    ),
    Mechanism(
        "gqa",
        "GQA",
        "attention",
        "Grouped-query attention: fewer KV heads than query heads.",
    ),
    Mechanism(
        "mqa",
        "MQA",
        "attention",
        "Multi-query attention: a single KV head shared across query heads.",
    ),
    # Routing / mixture
    Mechanism(
        "moe_routing",
        "MoE routing",
        "routing",
        "Mixture-of-experts token routing and gating.",
    ),
    # Position encoding
    Mechanism(
        "rope",
        "RoPE",
        "position",
        "Rotary position embedding.",
    ),
    Mechanism(
        "alibi",
        "ALiBi",
        "position",
        "Attention with linear biases (ALiBi).",
    ),
    Mechanism(
        "learned_pos_embed",
        "Learned pos. embed.",
        "position",
        "Learned absolute positional embeddings.",
    ),
    # Normalization
    Mechanism(
        "rms_norm",
        "RMSNorm",
        "normalization",
        "Root-mean-square normalization (no mean-centering).",
    ),
    Mechanism(
        "layer_norm",
        "LayerNorm",
        "normalization",
        "Standard mean-and-variance layer normalization.",
    ),
    # FFN
    Mechanism(
        "gated_mlp",
        "Gated MLP",
        "ffn",
        "Gated feed-forward network (SwiGLU, GeGLU, etc.).",
    ),
    # Model-level features
    Mechanism(
        "mtp",
        "MTP",
        "prediction",
        "Multi-token prediction head(s).",
    ),
    Mechanism(
        "multimodal",
        "Multimodal",
        "modalities",
        "Non-text modalities (vision, audio, etc.).",
    ),
    Mechanism(
        "tied_embeddings",
        "Tied embeddings",
        "weights",
        "Input and output embeddings share weights.",
    ),
    Mechanism(
        "kv_sharing",
        "KV sharing",
        "attention",
        "Key/value projections shared across some layers.",
    ),
]

TAXONOMY_BY_ID: dict[str, Mechanism] = {m.id: m for m in TAXONOMY}


def _get_text_config(config: Any) -> Any:
    """Return the text decoder config, handling multi-modal nested configs."""
    if hasattr(config, "text_config") and config.text_config is not None:
        return config.text_config
    return config


def _has_rms_norm(config: Any) -> bool:
    text_cfg = _get_text_config(config)
    return getattr(text_cfg, "rms_norm_eps", None) is not None


def _has_layer_norm(config: Any) -> bool:
    text_cfg = _get_text_config(config)
    # Only infer LayerNorm if the config explicitly carries layer_norm_eps and no
    # RMSNorm marker. Many models store RMSNorm under other fields, so this is
    # intentionally conservative; manual annotations handle ambiguous cases.
    return getattr(text_cfg, "layer_norm_eps", None) is not None and not _has_rms_norm(config)


def _is_gqa(config: Any) -> bool:
    text_cfg = _get_text_config(config)
    n_heads = getattr(text_cfg, "num_attention_heads", None)
    n_kv_heads = getattr(text_cfg, "num_key_value_heads", None)
    return (
        isinstance(n_heads, int)
        and isinstance(n_kv_heads, int)
        and n_kv_heads > 1
        and n_kv_heads < n_heads
    )


def _is_mqa(config: Any) -> bool:
    text_cfg = _get_text_config(config)
    n_kv_heads = getattr(text_cfg, "num_key_value_heads", None)
    return isinstance(n_kv_heads, int) and n_kv_heads == 1


def _has_gated_mlp(config: Any) -> bool:
    text_cfg = _get_text_config(config)
    act = getattr(text_cfg, "hidden_act", None) or getattr(text_cfg, "hidden_activation", None)
    if act is None:
        return False
    act = act.lower() if isinstance(act, str) else ""
    # Silu is unambiguously SwiGLU in modern LLMs. GeGLU (gelu_pytorch_tanh) is
    # also gated in some families, but gelu_new (Phi) is not; rely on manual
    # annotation for those cases to avoid false positives.
    return act == "silu"


def _attention_layer_types(config: Any) -> set[str]:
    text_cfg = _get_text_config(config)
    layer_types = getattr(text_cfg, "layer_types", None)
    if not isinstance(layer_types, (list, tuple)):
        return set()
    return set(str(t).lower() for t in layer_types)


def infer_mechanisms(config: Any) -> set[str]:
    """Infer architectural mechanisms from a HuggingFace config object."""
    mechs: set[str] = set()
    text_cfg = _get_text_config(config)

    # Attention layer types
    layer_types = _attention_layer_types(config)
    if "full_attention" in layer_types or not layer_types:
        mechs.add("full_attention")
    if "linear_attention" in layer_types:
        mechs.add("linear_attention")
    if "sliding_attention" in layer_types:
        mechs.add("sliding_window_attention")

    # Sliding window (if not already captured by layer_types)
    sliding_window = getattr(text_cfg, "sliding_window", None)
    if isinstance(sliding_window, int) and sliding_window > 0:
        mechs.add("sliding_window_attention")

    # GQA / MQA
    if _is_gqa(config):
        mechs.add("gqa")
    elif _is_mqa(config):
        mechs.add("mqa")

    # MoE
    if any(
        getattr(text_cfg, attr, None) is not None
        for attr in ("num_experts", "n_routed_experts", "num_local_experts", "moe_intermediate_size")
    ):
        # Distinguish real MoE from false positives.
        num_experts = getattr(text_cfg, "num_experts", 0) or 0
        n_routed = getattr(text_cfg, "n_routed_experts", 0) or 0
        if num_experts > 1 or n_routed > 1:
            mechs.add("moe_routing")

    # Position encoding — keep this conservative. transformers sometimes adds
    # default rope_parameters to configs that actually use learned embeddings.
    pos_type = getattr(text_cfg, "position_embedding_type", None)
    if pos_type == "alibi":
        mechs.add("alibi")
    elif pos_type == "rope":
        mechs.add("rope")
    elif pos_type == "learned":
        mechs.add("learned_pos_embed")
    # If the config explicitly declares rope_scaling, that's a strong RoPE signal.
    if getattr(text_cfg, "rope_scaling", None) is not None:
        mechs.add("rope")

    # Normalization
    if _has_rms_norm(config):
        mechs.add("rms_norm")
    if _has_layer_norm(config):
        mechs.add("layer_norm")

    # FFN
    if _has_gated_mlp(config):
        mechs.add("gated_mlp")

    # MTP
    if getattr(text_cfg, "mtp_num_hidden_layers", 0):
        mechs.add("mtp")

    # Multimodal
    if any(
        getattr(config, attr, None) is not None
        for attr in ("vision_config", "audio_config", "image_token_id", "video_token_id")
    ):
        mechs.add("multimodal")

    # Tied embeddings
    if getattr(text_cfg, "tie_word_embeddings", False) or getattr(config, "tie_word_embeddings", False):
        mechs.add("tied_embeddings")

    # KV sharing across layers
    if getattr(text_cfg, "num_kv_shared_layers", 0):
        mechs.add("kv_sharing")

    return mechs


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open() as f:
        return yaml.safe_load(f)


def get_model_mechanisms(model_cfg: dict[str, Any], config: Any | None = None) -> set[str]:
    """Return the union of manually annotated and inferred mechanisms.

    Manual annotations take precedence for ambiguous pairs (norm type, position
    encoding), so inferred mechanisms that contradict a manual choice are
    dropped.
    """
    manual = set(model_cfg.get("mechanisms", []))
    if config is None:
        try:
            config = AutoConfig.from_pretrained(model_cfg["hf_id"])
        except Exception:
            return manual
    inferred = infer_mechanisms(config)

    # Resolve contradictions in favor of manual annotations.
    if "rms_norm" in manual or "layer_norm" in manual:
        inferred.discard("rms_norm")
        inferred.discard("layer_norm")
    if "rope" in manual or "learned_pos_embed" in manual or "alibi" in manual:
        inferred.discard("rope")
        inferred.discard("learned_pos_embed")
        inferred.discard("alibi")
    if "gated_mlp" in manual:
        inferred.discard("gated_mlp")

    return manual | inferred


def compute_heterogeneity(mechanisms: set[str]) -> dict[str, Any]:
    """Simple heterogeneity metrics from a mechanism set."""
    by_category: dict[str, set[str]] = {}
    for mid in mechanisms:
        mech = TAXONOMY_BY_ID.get(mid)
        if mech is None:
            continue
        by_category.setdefault(mech.category, set()).add(mid)

    n_categories = len(by_category)
    n_mechanisms = len(mechanisms)
    # Heterogeneity score: number of distinct categories touched, plus a small
    # bonus for multiple mechanisms within a category (e.g., full + linear attention).
    intra_category_bonus = sum(max(0, len(v) - 1) for v in by_category.values())
    score = n_categories + 0.5 * intra_category_bonus

    return {
        "mechanisms": sorted(mechanisms),
        "num_mechanisms": n_mechanisms,
        "num_categories": n_categories,
        "by_category": {k: sorted(v) for k, v in by_category.items()},
        "heterogeneity_score": round(score, 2),
    }
