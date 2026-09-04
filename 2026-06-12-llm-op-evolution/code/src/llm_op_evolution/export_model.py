"""Load HuggingFace models and export to Core ATen IR."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_op_evolution.op_utils import GraphMetrics, count_graph_nodes, infer_num_layers

logger = logging.getLogger(__name__)


class LogitsOnlyWrapper(torch.nn.Module):
    """Export-friendly wrapper: logits tensor only, no DynamicCache in outputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(
            input_ids,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


@dataclass
class ModelConfig:
    id: str
    hf_id: str
    name: str
    release_date: str
    family: str
    arch_notes: str
    trust_remote_code: bool = False
    optional: bool = False
    dtype: str | None = None


@dataclass
class ExportResult:
    model: ModelConfig
    metrics_training: GraphMetrics | None
    metrics_inference: GraphMetrics | None
    metrics_core_aten: GraphMetrics | None
    num_layers: int | None
    vocab_size: int | None
    export_path: str | None
    success: bool
    error: str | None = None


def load_model_manifest(path: Path) -> tuple[dict[str, Any], list[ModelConfig]]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    defaults = raw.get("defaults", {})
    models = [
        ModelConfig(
            id=m["id"],
            hf_id=m["hf_id"],
            name=m["name"],
            release_date=m["release_date"],
            family=m["family"],
            arch_notes=m["arch_notes"],
            trust_remote_code=m.get("trust_remote_code", False),
            optional=m.get("optional", False),
            dtype=m.get("dtype"),
        )
        for m in raw["models"]
    ]
    return defaults, models


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def load_hf_model(
    model_cfg: ModelConfig,
    defaults: dict[str, Any],
    *,
    device: str = "cpu",
) -> tuple[torch.nn.Module, torch.Tensor, int | None]:
    dtype_name = model_cfg.dtype or defaults.get("dtype", "float32")
    dtype = _dtype_from_name(dtype_name)
    attn = defaults.get("attn_implementation", "eager")

    logger.info("Loading %s (%s)", model_cfg.name, model_cfg.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id,
        torch_dtype=dtype,
        attn_implementation=attn,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    model.eval()
    model.to(device)
    model = LogitsOnlyWrapper(model)

    seq_len = int(defaults.get("seq_len", 128))
    batch = int(defaults.get("batch_size", 1))

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.hf_id,
            trust_remote_code=model_cfg.trust_remote_code,
        )
        vocab_size = len(tokenizer)
        input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    except Exception:
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", 32000)
        input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)

    return model, input_ids, vocab_size


def export_model_graph(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    strict: bool = False,
) -> tuple[Any, Any, Any]:
    """Export and decompose to Training, Inference, and Core ATen IR levels."""
    from torch.export import export

    with torch.no_grad():
        ep = export(model, args=(input_ids,), strict=strict)
        ep_inference = ep.run_decompositions(decomp_table={})
        ep_core = ep.run_decompositions(decomp_table=None)

    return ep, ep_inference, ep_core


def analyze_exported_model(
    model_cfg: ModelConfig,
    defaults: dict[str, Any],
    *,
    output_dir: Path,
    device: str = "cpu",
    strict: bool = False,
    save_export: bool = False,
) -> ExportResult:
    try:
        model, input_ids, vocab_size = load_hf_model(
            model_cfg, defaults, device=device
        )
        num_layers = infer_num_layers(model)

        ep, ep_inference, ep_core = export_model_graph(
            model, input_ids, strict=strict
        )

        gm_train = ep.graph_module
        gm_inf = ep_inference.graph_module
        gm_core = ep_core.graph_module

        metrics_training = count_graph_nodes(
            gm_train,
            model_id=model_cfg.id,
            ir_level="training",
            num_layers=num_layers,
        )
        metrics_inference = count_graph_nodes(
            gm_inf,
            model_id=model_cfg.id,
            ir_level="inference",
            num_layers=num_layers,
        )
        metrics_core = count_graph_nodes(
            gm_core,
            model_id=model_cfg.id,
            ir_level="core_aten",
            num_layers=num_layers,
        )

        export_path = None
        if save_export:
            export_path = str(output_dir / f"{model_cfg.id}.pt2")
            torch.export.save(ep_core, export_path)

        del model
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ExportResult(
            model=model_cfg,
            metrics_training=metrics_training,
            metrics_inference=metrics_inference,
            metrics_core_aten=metrics_core,
            num_layers=num_layers,
            vocab_size=vocab_size,
            export_path=export_path,
            success=True,
        )
    except Exception as exc:
        logger.exception("Export failed for %s", model_cfg.id)
        return ExportResult(
            model=model_cfg,
            metrics_training=None,
            metrics_inference=None,
            metrics_core_aten=None,
            num_layers=None,
            vocab_size=None,
            export_path=None,
            success=False,
            error=str(exc),
        )


def export_result_to_dict(result: ExportResult) -> dict[str, Any]:
    def metrics_dict(m: GraphMetrics | None) -> dict[str, Any] | None:
        return asdict(m) if m else None

    return {
        "model": asdict(result.model),
        "num_layers": result.num_layers,
        "vocab_size": result.vocab_size,
        "export_path": result.export_path,
        "success": result.success,
        "error": result.error,
        "metrics": {
            "training": metrics_dict(result.metrics_training),
            "inference": metrics_dict(result.metrics_inference),
            "core_aten": metrics_dict(result.metrics_core_aten),
        },
    }


def _load_existing_metrics(metrics_path: Path) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    if metrics_path.exists():
        with metrics_path.open() as f:
            for row in json.load(f):
                merged[row["model"]["id"]] = row
    return merged


def _save_metrics(metrics_path: Path, merged: dict[str, dict[str, Any]]) -> None:
    with metrics_path.open("w") as f:
        json.dump(list(merged.values()), f, indent=2)


def run_exports(
    manifest_path: Path,
    output_dir: Path,
    *,
    model_ids: list[str] | None = None,
    device: str = "cpu",
    strict: bool = False,
) -> list[ExportResult]:
    defaults, models = load_model_manifest(manifest_path)
    if model_ids is not None:
        models = [m for m in models if m.id in model_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    merged = _load_existing_metrics(metrics_path)
    results: list[ExportResult] = []

    for model_cfg in models:
        result = analyze_exported_model(
            model_cfg,
            defaults,
            output_dir=output_dir / "raw",
            device=device,
            strict=strict,
        )
        results.append(result)

        out_file = output_dir / f"{model_cfg.id}.json"
        with out_file.open("w") as f:
            json.dump(export_result_to_dict(result), f, indent=2)
        logger.info(
            "Finished %s: success=%s nodes_per_layer=%s",
            model_cfg.id,
            result.success,
            (
                result.metrics_core_aten.nodes_per_layer
                if result.metrics_core_aten
                else None
            ),
        )

        # Update metrics.json after each model so long runs are resumable.
        merged[model_cfg.id] = export_result_to_dict(result)
        _save_metrics(metrics_path, merged)

    return results