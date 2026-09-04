"""Visualization for LLM operator graph density over time."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from llm_op_evolution.mechanisms import TAXONOMY, TAXONOMY_BY_ID


def _parse_date(date_str: str) -> datetime:
    for fmt in ("%Y-%m", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime(2020, 1, 1)


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    with metrics_path.open() as f:
        rows = json.load(f)

    records: list[dict[str, Any]] = []
    for row in rows:
        model = row["model"]
        core = row.get("metrics", {}).get("core_aten") or {}
        records.append(
            {
                "id": model["id"],
                "name": model["name"],
                "release_date": model["release_date"],
                "family": model["family"],
                "arch_notes": model["arch_notes"],
                "success": row["success"],
                "error": row.get("error"),
                "num_layers": row.get("num_layers"),
                "compute_nodes": core.get("compute_nodes"),
                "unique_ops": core.get("unique_ops"),
                "nodes_per_layer": core.get("nodes_per_layer"),
                "category_counts": core.get("category_counts", {}),
                "custom_ops": core.get("custom_ops", []),
                "mechanisms": row.get("mechanisms"),
            }
        )

    df = pd.DataFrame(records)
    df["date"] = df["release_date"].map(_parse_date)
    return df.sort_values("date")


def _compute_density(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["success"] == True].copy()  # noqa: E712
    ok["density"] = ok.apply(
        lambda r: (
            r["nodes_per_layer"]
            if pd.notna(r["nodes_per_layer"])
            else (r["compute_nodes"] or 0) / max(r["num_layers"] or 1, 1)
        ),
        axis=1,
    )
    ok["year"] = ok["date"].dt.year
    ok["is_moe"] = ok["family"].str.contains("moe", case=False, na=False)
    ok["arch_class"] = "dense"
    ok.loc[
        ok["family"].str.contains("moe", case=False, na=False)
        | ok["id"].isin(["deepseek-coder-v2-lite"]),
        "arch_class",
    ] = "moe"
    ok.loc[ok["id"] == "qwen3.5-4b", "arch_class"] = "hybrid-linear-attn"
    return ok


def plot_timeline(df: pd.DataFrame, output_dir: Path) -> Path:
    ok = _compute_density(df)
    if ok.empty:
        raise ValueError("No successful exports to plot")

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    ax0 = axes[0]
    sns.scatterplot(
        data=ok,
        x="date",
        y="density",
        hue="family",
        style="family",
        s=120,
        ax=ax0,
        legend="brief",
    )
    # Annotate only a curated subset to avoid clutter.
    label_ids = {
        "qwen3.5-4b",
        "deepseek-coder-v2-lite",
        "qwen1.5-moe-a2.7b",
        "gemma-4-e2b-it",
        "qwen3-1.7b",
        "qwen3-0.6b",
        "deepseek-r1-distill-qwen-1.5b",
        "tinyllama-1.1b",
        "phi-2",
    }
    for _, row in ok.iterrows():
        if row["id"] in label_ids or row["density"] > 150:
            ax0.annotate(
                row["name"],
                (row["date"], row["density"]),
                textcoords="offset points",
                xytext=(4, 4),
                ha="left",
                fontsize=8,
                alpha=0.9,
            )
    ax0.set_ylabel("Core ATen nodes / layer")
    ax0.set_title("LLM Operator Graph Density Over Time (2022–2026, open models)")
    ax0.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    ax1 = axes[1]
    sns.lineplot(
        data=ok.sort_values("date"),
        x="date",
        y="compute_nodes",
        marker="s",
        color="tab:orange",
        ax=ax1,
    )
    ax1.set_ylabel("Total Core ATen compute nodes (full forward pass)")
    ax1.set_xlabel("Release date")

    fig.autofmt_xdate()
    fig.tight_layout()
    out = output_dir / "timeline_density.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_timeline_log(df: pd.DataFrame, output_dir: Path) -> Path:
    ok = _compute_density(df)
    if ok.empty:
        raise ValueError("No successful exports to plot")

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.scatterplot(
        data=ok,
        x="date",
        y="density",
        hue="family",
        style="family",
        s=150,
        ax=ax,
        legend="brief",
    )
    for _, row in ok.iterrows():
        ax.annotate(
            row["name"],
            (row["date"], row["density"]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            fontsize=7,
            alpha=0.85,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Core ATen nodes / layer (log scale)")
    ax.set_xlabel("Release date")
    ax.set_title(
        "LLM Operator Graph Density Over Time — Log Scale\n"
        "(shows both the gradual rise and the Qwen 3.5 outlier)"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = output_dir / "timeline_density_log.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_timeline_dense_only(df: pd.DataFrame, output_dir: Path) -> Path:
    ok = _compute_density(df)
    dense = ok[ok["arch_class"] == "dense"].copy()
    if dense.empty:
        raise ValueError("No dense models to plot")

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.scatterplot(
        data=dense,
        x="date",
        y="density",
        hue="family",
        style="family",
        s=120,
        ax=ax,
        legend="brief",
    )
    for _, row in dense.iterrows():
        ax.annotate(
            row["name"],
            (row["date"], row["density"]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            fontsize=7,
            alpha=0.85,
        )
    ax.set_ylabel("Core ATen nodes / layer")
    ax.set_xlabel("Release date")
    ax.set_title(
        "Dense-Transformer Operator Graph Density Over Time\n"
        "(MoE and hybrid linear-attention models excluded for readability)"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = output_dir / "timeline_density_dense_only.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_year_binned(df: pd.DataFrame, output_dir: Path) -> Path:
    ok = _compute_density(df)
    if ok.empty:
        raise ValueError("No successful exports to plot")

    yearly = (
        ok.groupby("year")
        .agg(
            mean_density=("density", "mean"),
            median_density=("density", "median"),
            max_density=("density", "max"),
            min_density=("density", "min"),
            count=("density", "size"),
        )
        .reset_index()
        .sort_values("year")
    )

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = yearly["year"].values
    width = 0.35
    ax.bar(
        x - width / 2,
        yearly["median_density"],
        width=width,
        label="Median",
        color="tab:blue",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        yearly["mean_density"],
        width=width,
        label="Mean",
        color="tab:orange",
        alpha=0.8,
    )
    # Overlay min/max range
    for _, row in yearly.iterrows():
        ax.plot(
            [row["year"], row["year"]],
            [row["min_density"], row["max_density"]],
            color="gray",
            linewidth=2,
            zorder=0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(y)) for y in x])
    ax.set_xlabel("Release year")
    ax.set_ylabel("Core ATen nodes / layer")
    ax.set_title(
        "Operator Graph Density by Year\n"
        "(bars: median/mean; vertical lines: min–max range)"
    )
    ax.legend()
    fig.tight_layout()
    out = output_dir / "year_density.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_category_stacked(df: pd.DataFrame, output_dir: Path) -> Path:
    ok = df[df["success"] == True].copy()  # noqa: E712
    if ok.empty:
        raise ValueError("No successful exports to plot")

    categories = [
        "attention",
        "moe",
        "norm",
        "linear",
        "activation",
        "reshape",
        "embedding",
        "output",
        "custom",
        "other",
    ]

    rows = []
    for _, row in ok.iterrows():
        counts = row["category_counts"] or {}
        total = sum(counts.values()) or 1
        num_layers = row["num_layers"] or 1
        for cat in categories:
            rows.append(
                {
                    "name": row["name"],
                    "date": row["date"],
                    "category": cat,
                    "count_per_layer": counts.get(cat, 0) / num_layers,
                    "fraction": counts.get(cat, 0) / total,
                }
            )

    cat_df = pd.DataFrame(rows)
    pivot = cat_df.pivot_table(
        index="name",
        columns="category",
        values="count_per_layer",
        fill_value=0,
    )
    # Preserve chronological order
    order = ok.sort_values("date")["name"].tolist()
    pivot = pivot.reindex(order)

    n = len(pivot)
    fig_h = max(7, 0.35 * n + 3)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    pivot.plot(kind="barh", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Operator nodes per layer (by category)")
    ax.set_ylabel("Model (chronological)")
    ax.set_title("Operator Category Composition (Core ATen IR)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    out = output_dir / "category_composition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_unique_ops(df: pd.DataFrame, output_dir: Path) -> Path:
    ok = df[df["success"] == True].copy().sort_values("date")  # noqa: E712
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = sns.color_palette("husl", n_colors=ok["family"].nunique())
    color_map = dict(zip(ok["family"].unique(), palette))
    colors = ok["family"].map(color_map)
    bars = ax.bar(ok["name"], ok["unique_ops"], color=colors, alpha=0.85)
    ax.set_ylabel("Unique Core ATen operators used")
    ax.set_xlabel("Model")
    ax.set_title("Operator Vocabulary Breadth")
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, ok["unique_ops"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out = output_dir / "unique_ops.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _prepare_mechanism_df(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["success"] == True].copy()  # noqa: E712
    if ok.empty or "mechanisms" not in ok.columns:
        raise ValueError("No mechanism data available")

    records = []
    for _, row in ok.iterrows():
        mechs = set(row["mechanisms"].get("mechanisms", []))
        for mech in TAXONOMY:
            records.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "date": row["date"],
                    "family": row["family"],
                    "mechanism_id": mech.id,
                    "mechanism_label": mech.label,
                    "category": mech.category,
                    "present": 1.0 if mech.id in mechs else 0.0,
                    "heterogeneity_score": row["mechanisms"].get(
                        "heterogeneity_score", 0
                    ),
                    "num_mechanisms": row["mechanisms"].get("num_mechanisms", 0),
                    "density": row["density"] if pd.notna(row.get("density")) else 0,
                }
            )
    return pd.DataFrame(records)


def plot_mechanism_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    mech_df = _prepare_mechanism_df(df)
    ok = df[df["success"] == True].copy()  # noqa: E712
    ok = ok.sort_values("date")

    # Build a matrix: rows = models (chronological), cols = mechanisms (by category)
    pivot = mech_df.pivot_table(
        index="name",
        columns="mechanism_id",
        values="present",
        fill_value=0,
    )
    # Order rows by date
    order = ok["name"].tolist()
    pivot = pivot.reindex(order)
    # Order columns by taxonomy order
    ordered_cols = [m.id for m in TAXONOMY if m.id in pivot.columns]
    pivot = pivot[ordered_cols]

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        cbar_kws={"label": "Present"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_xlabel("Architectural mechanism")
    ax.set_ylabel("Model (chronological)")
    ax.set_title("Architectural Mechanism Inventory (2022–2026)")
    # Rotate x labels
    ax.set_xticklabels(
        [TAXONOMY_BY_ID[c].label for c in pivot.columns],
        rotation=45,
        ha="right",
    )
    fig.tight_layout()
    out = output_dir / "mechanism_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heterogeneity_timeline(df: pd.DataFrame, output_dir: Path) -> Path:
    ok = df[df["success"] == True].copy()  # noqa: E712
    if "mechanisms" not in ok.columns:
        raise ValueError("No mechanism data available")

    ok["heterogeneity_score"] = ok["mechanisms"].map(
        lambda x: x.get("heterogeneity_score", 0) if isinstance(x, dict) else 0
    )
    ok["num_mechanisms"] = ok["mechanisms"].map(
        lambda x: x.get("num_mechanisms", 0) if isinstance(x, dict) else 0
    )
    ok = ok.sort_values("date")

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.scatterplot(
        data=ok,
        x="date",
        y="heterogeneity_score",
        hue="family",
        style="family",
        s=150,
        ax=ax,
        legend="brief",
    )
    for _, row in ok.iterrows():
        ax.annotate(
            row["name"],
            (row["date"], row["heterogeneity_score"]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            fontsize=7,
            alpha=0.85,
        )
    ax.set_ylabel("Heterogeneity score")
    ax.set_xlabel("Release date")
    ax.set_title(
        "Architectural Heterogeneity Over Time\n"
        "(higher = more distinct mechanisms / categories per model)"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = output_dir / "heterogeneity_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_density_vs_heterogeneity(df: pd.DataFrame, output_dir: Path) -> Path:
    ok = df[df["success"] == True].copy()  # noqa: E712
    if "mechanisms" not in ok.columns:
        raise ValueError("No mechanism data available")

    ok["heterogeneity_score"] = ok["mechanisms"].map(
        lambda x: x.get("heterogeneity_score", 0) if isinstance(x, dict) else 0
    )
    ok["density"] = ok.apply(
        lambda r: (
            r["nodes_per_layer"]
            if pd.notna(r["nodes_per_layer"])
            else (r["compute_nodes"] or 0) / max(r["num_layers"] or 1, 1)
        ),
        axis=1,
    )

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=ok,
        x="heterogeneity_score",
        y="density",
        hue="family",
        style="family",
        s=150,
        ax=ax,
        legend="brief",
    )
    for _, row in ok.iterrows():
        ax.annotate(
            row["name"],
            (row["heterogeneity_score"], row["density"]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            fontsize=7,
            alpha=0.85,
        )
    ax.set_xlabel("Heterogeneity score (mechanism diversity)")
    ax.set_ylabel("Core ATen nodes / layer")
    ax.set_yscale("log")
    ax.set_title(
        "Implementation Density vs. Architectural Heterogeneity\n"
        "(log-scale y-axis; Qwen 3.5 is an outlier in density, not heterogeneity)"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    out = output_dir / "density_vs_heterogeneity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_all_plots(metrics_path: Path, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(metrics_path)
    ok = df[df["success"] == True]  # noqa: E712
    if ok.empty:
        raise ValueError("No successful exports to plot")

    outputs = [
        plot_timeline(df, figures_dir),
        plot_timeline_log(df, figures_dir),
        plot_timeline_dense_only(df, figures_dir),
        plot_year_binned(df, figures_dir),
        plot_category_stacked(df, figures_dir),
        plot_unique_ops(df, figures_dir),
    ]
    if "mechanisms" in df.columns:
        outputs.extend(
            [
                plot_mechanism_heatmap(df, figures_dir),
                plot_heterogeneity_timeline(df, figures_dir),
                plot_density_vs_heterogeneity(df, figures_dir),
            ]
        )
    return outputs