"""Utilities for extracting and categorizing operators from exported graphs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.fx import GraphModule, Node


def op_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    if hasattr(target, "name"):
        return str(target.name())
    return str(target)


def is_aten_op(name: str) -> bool:
    return name.startswith("aten::") or name.startswith("aten.")


def is_custom_op(name: str) -> bool:
    if is_aten_op(name) or name.startswith("<built-in"):
        return False
    return name.startswith("torch.ops.") or "::" in name


# Category patterns ordered by specificity (first match wins).
CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("moe", re.compile(r"expert|moe|router|topk|index_put", re.I)),
    ("attention", re.compile(r"scaled_dot_product|_scaled_dot_product|softmax", re.I)),
    ("norm", re.compile(r"layer_norm|rms_norm|group_norm|batch_norm", re.I)),
    ("activation", re.compile(r"gelu|silu|relu|tanh|sigmoid|softplus|glu|gelu", re.I)),
    ("embedding", re.compile(r"embedding", re.I)),
    ("linear", re.compile(r"::mm|matmul|linear|addmm|bmm|einsum", re.I)),
    (
        "reshape",
        re.compile(
            r"reshape|view|permute|transpose|squeeze|unsqueeze|expand|cat|split",
            re.I,
        ),
    ),
]


def categorize_op(name: str) -> str:
    for category, pattern in CATEGORY_PATTERNS:
        if pattern.search(name):
            return category
    if is_custom_op(name):
        return "custom"
    return "other"


@dataclass
class GraphMetrics:
    model_id: str
    ir_level: str
    total_nodes: int
    compute_nodes: int
    unique_ops: int
    nodes_per_layer: float | None
    num_layers: int | None
    op_counts: dict[str, int] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    custom_ops: list[str] = field(default_factory=list)
    export_success: bool = True
    error: str | None = None


def count_graph_nodes(
    gm: GraphModule,
    *,
    model_id: str,
    ir_level: str,
    num_layers: int | None = None,
) -> GraphMetrics:
    op_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    custom_ops: set[str] = set()
    compute_nodes = 0

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        compute_nodes += 1
        name = op_name(node.target)
        op_counter[name] += 1
        category_counter[categorize_op(name)] += 1
        if is_custom_op(name):
            custom_ops.add(name)

    nodes_per_layer = None
    if num_layers and num_layers > 0:
        nodes_per_layer = compute_nodes / num_layers

    return GraphMetrics(
        model_id=model_id,
        ir_level=ir_level,
        total_nodes=len(gm.graph.nodes),
        compute_nodes=compute_nodes,
        unique_ops=len(op_counter),
        nodes_per_layer=nodes_per_layer,
        num_layers=num_layers,
        op_counts=dict(op_counter.most_common()),
        category_counts=dict(category_counter.most_common()),
        custom_ops=sorted(custom_ops),
        export_success=True,
    )


def diff_ops(
    baseline_ops: set[str], model_ops: set[str]
) -> tuple[list[str], list[str]]:
    novel = sorted(model_ops - baseline_ops)
    removed = sorted(baseline_ops - model_ops)
    return novel, removed


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    inner = model
    for _ in range(3):
        if hasattr(inner, "model") and isinstance(inner.model, torch.nn.Module):
            inner = inner.model
        else:
            break
    return inner


def infer_num_layers(model: torch.nn.Module) -> int | None:
    inner = _unwrap_model(model)

    config = getattr(inner, "config", None)
    if config is not None:
        # Top-level config (most models)
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            val = getattr(config, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        # Nested text config (multimodal models such as Gemma 4)
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            for attr in ("num_hidden_layers", "n_layer", "num_layers"):
                val = getattr(text_config, attr, None)
                if isinstance(val, int) and val > 0:
                    return val

    for attr in ("model", "transformer", "gpt_neox", "language_model"):
        submodule = getattr(inner, attr, None)
        if submodule is None:
            continue
        # For multimodal wrappers, the language model may itself contain a model
        # with the transformer layers.
        targets = [submodule]
        if hasattr(submodule, "model") and isinstance(submodule.model, torch.nn.Module):
            targets.append(submodule.model)
        for target in targets:
            for layer_attr in ("layers", "h", "blocks"):
                layers = getattr(target, layer_attr, None)
                if layers is not None:
                    try:
                        return len(layers)
                    except TypeError:
                        pass
    return None