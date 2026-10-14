"""Rebuild summaries and standalone scientific figures from recorded results."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#4477aa", "#228833", "#cc6677", "#aa3377"]


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def causal_stats(rows):
    # Directions within a pair are dependent; average them before calculating uncertainty.
    by_pair = defaultdict(list)
    for r in rows:
        if r["fraction_of_counterfactual_gap"] is not None:
            by_pair[r["pair_id"]].append(r["fraction_of_counterfactual_gap"])
    values = np.array([np.mean(v) for v in by_pair.values()])
    margin_by_pair = defaultdict(list)
    for r in rows:
        margin_by_pair[r["pair_id"]].append(r["correct_margin_change"])
    margins = [np.mean(v) for v in margin_by_pair.values()]
    # The same underlying example is reused at different distances. Bootstrap whole
    # example clusters, keeping both directions and every distance together.
    clusters = defaultdict(list)
    for pair_id, fractions in by_pair.items():
        clusters[int(pair_id.rsplit("-p", 1)[1])].append(float(np.mean(fractions)))
    if len(clusters) > 1:
        cluster_sums = np.array([sum(v) for v in clusters.values()])
        cluster_counts = np.array([len(v) for v in clusters.values()])
        draws = np.random.default_rng(0).integers(0, len(clusters), (2000, len(clusters)))
        boot = cluster_sums[draws].sum(-1) / cluster_counts[draws].sum(-1)
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
    else:
        ci = None
    return {"n_pairs": len(margin_by_pair), "n_trials": len(rows),
            "n_example_clusters": len(clusters),
            "mean_gap_fraction": float(values.mean()) if len(values) else None,
            "bootstrap_95_interval": ci,
            "mean_correct_margin_change": float(np.mean(margins)) if margins else None}


def build_report(output: Path):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    records = load_lines(output / "baseline.jsonl")
    causal = load_lines(output / "causal.jsonl")
    cache_controls = load_lines(output / "cache_controls.jsonl")
    kinds = manifest.get("layer_types", [])
    distances = manifest["protocol"]["distances"]
    usable = [r for r in records if r["both_answers_correct"]]
    summary = {"model": manifest["protocol"]["model"], "revision": manifest["protocol"]["revision"],
               "n_pairs": len(records), "both_answers_correct": len(usable), "by_distance": {},
               "peak_reserved_gib": max([r["peak_reserved_gib"] for r in records] + [manifest.get("peak_reserved_gib", 0)]),
               "analysis_condition": "Both unconditioned greedy answers correct; causal pairs also require both first-difference predictions correct",
               "limitations": ["Single model, 16 distractors by default, neutral filler, and a small deterministic pilot",
                               "First-difference scoring supplies a shared answer prefix; full-answer accuracy is reported separately",
                               "Candidate layers selected using these pilot data; interventions are exploratory",
                               "Patches at one layer do not exclude historical information arriving through other layers",
                               "A donor patch measures transfer; history restrictions test a specific form of necessity",
                               "Layerwise unembedding is a logit lens, not a causal attribution"]}
    for d in distances:
        group = [r for r in records if r["nominal_distance"] == d]
        n = len(group)
        summary["by_distance"][str(d)] = {
            "pairs": n,
            "correct_answers": sum(r[s]["answer_correct"] for r in group for s in ("a", "b")),
            "answer_accuracy": sum(r[s]["answer_correct"] for r in group for s in ("a", "b")) / (2*n) if n else None,
            "both_correct_pairs": sum(r["both_answers_correct"] for r in group),
            "both_first_difference_correct_pairs": sum(r["both_first_differences_correct"] for r in group),
            "mean_trace_forward_seconds": float(np.mean([r[s]["seconds"] for r in group for s in ("a", "b")])) if n else None,
        }
    summary["failed_answers"] = [{"pair_id": r["pair_id"], "a": r["a"]["generated_answer"],
                                  "b": r["b"]["generated_answer"]} for r in records if not r["both_answers_correct"]]
    summary["max_self_patch_margin_drift"] = max([abs(r["margin_delta"]) for r in causal if r["control"]] + [0])
    summary["causal_trials"] = len(causal)
    summary["self_patch_controls"] = sum(r["control"] for r in causal)
    summary["completed_causal_pairs"] = len(load_lines(output / "causal_completed.jsonl"))
    summary["unique_baseline_examples"] = len({int(r["pair_id"].rsplit("-p", 1)[1]) for r in records})
    summary["max_full_to_chunked_margin_drift"] = max([abs(v) for r in cache_controls
                                                     for v in r["full_to_chunked_margin_drift"].values()] + [0])
    groups = defaultdict(list)
    for r in causal:
        if not r["control"]:
            groups[(r["component"], r["layer"])].append(r)
    summary["causal"] = [{"component": component, "layer_zero_based": layer, "layer_type": kinds[layer],
                          **causal_stats(rows), "by_distance": {
                              str(d): causal_stats([r for r in rows if r["nominal_distance"] == d]) for d in distances}}
                         for (component, layer), rows in sorted(groups.items())]
    if usable:
        separation = np.array([np.array(r["a"]["trace"]["lens_mixer_change"]) -
                               np.array(r["b"]["trace"]["lens_mixer_change"]) for r in usable])
        summary["mixer_changes_by_type"] = {kind: {
            "layers": sum(k == kind for k in kinds),
            "mean_absolute_paired_lens_change_per_layer": float(np.abs(separation[:, [i for i,k in enumerate(kinds) if k == kind]]).mean()),
            "mean_signed_paired_lens_change_per_layer": float(separation[:, [i for i,k in enumerate(kinds) if k == kind]].mean()),
        } for kind in sorted(set(kinds))}
        summary["largest_mean_paired_changes"] = {}
        for component in ("mixer", "mlp"):
            changes = np.mean([np.array(r["a"]["trace"][f"lens_{component}_change"]) -
                               np.array(r["b"]["trace"][f"lens_{component}_change"]) for r in usable], axis=0)
            summary["largest_mean_paired_changes"][component] = [
                {"layer_zero_based": int(i), "layer_type": kinds[i], "mean_paired_lens_change": float(changes[i])}
                for i in np.argsort(np.abs(changes))[::-1][:5]]
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 130, "savefig.dpi": 180, "svg.fonttype": "none"})

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(figures / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)

    def mark_globals(ax):
        for i, kind in enumerate(kinds):
            if kind == "full_attention":
                ax.axvspan(i + 0.65, i + 1.35, color="#888888", alpha=0.13, zorder=0)
        ax.set_xlabel("Decoder layer (1-based)")
        ax.set_xticks(np.arange(4, len(kinds) + 1, 4))

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(distances))
    ax.bar(xs - .17, [100 * (summary["by_distance"][str(d)]["answer_accuracy"] or 0) for d in distances], .34,
           label="Individual answers", color=COLORS[0])
    ax.bar(xs + .17, [100 * summary["by_distance"][str(d)]["both_correct_pairs"] /
                      max(1, summary["by_distance"][str(d)]["pairs"]) for d in distances], .34,
           label="Both answers in pair", color=COLORS[1])
    ax.set(xticks=xs, xticklabels=[str(d) for d in distances], ylim=(0, 105),
           xlabel="Oldest source → scoring position (tokens)", ylabel="Greedy full-answer accuracy (%)")
    ax.legend()
    save(fig, "accuracy")
    if usable:
        layers = np.arange(1, len(kinds) + 1)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for d, color in zip(distances, COLORS):
            group = [r for r in usable if r["nominal_distance"] == d]
            if not group:
                continue
            values = np.array([np.array(r["a"]["trace"]["lens_after"]) -
                               np.array(r["b"]["trace"]["lens_after"]) for r in group])
            ax.plot(layers, values.mean(0), label=f"{d} tokens (n={len(group)})", color=color)
        mark_globals(ax)
        ax.set_ylabel("Paired separation: margin(A) − margin(B)")
        ax.legend()
        save(fig, "layerwise_lens")
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for ax, d in zip(axes.flat, distances):
            group = [r for r in usable if r["nominal_distance"] == d]
            if not group:
                continue
            for offset, component, color in ((-.18, "mixer", COLORS[0]), (.18, "mlp", COLORS[2])):
                values = np.array([np.array(r["a"]["trace"][f"lens_{component}_change"]) -
                                   np.array(r["b"]["trace"][f"lens_{component}_change"]) for r in group])
                ax.bar(layers + offset, values.mean(0), .36, label=component, color=color)
            mark_globals(ax)
            ax.set_title(f"{d} tokens; {len(group)} solved pairs")
            ax.set_ylabel("Change in paired logit-lens separation")
            ax.axhline(0, color="black", lw=.6)
        axes.flat[0].legend()
        save(fig, "mixer_and_mlp")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for position, ax in enumerate(axes):
            for d, color in zip(distances, COLORS):
                group = [r for r in usable if r["nominal_distance"] == d]
                if group:
                    values = np.array([r["source_sensitivity"]["relative_paired_difference"] for r in group])
                    ax.plot(layers, values[:, :, position].mean(0), label=str(d), color=color)
            mark_globals(ax)
            ax.set_title(("Historical V1", "Historical V2", "Scoring position")[position])
            ax.set_ylabel("Relative paired representation difference")
        axes[0].legend(title="Distance")
        save(fig, "source_contextualization")
        fig, ax = plt.subplots(figsize=(8, 4))
        for d, color in zip(distances, COLORS):
            group = [r for r in usable if r["nominal_distance"] == d]
            if not group:
                continue
            values = [np.array(r[s]["trace"]["attention_value_mass"])[:, :, v].mean(1)
                      for r in group for s, v in (("a", 0), ("b", 1))]
            globals_ = np.array(group[0]["a"]["trace"]["attention_layers"]) + 1
            ax.plot(globals_, np.mean(values, axis=0), marker="o", label=str(d), color=color)
        ax.set(xlabel="Global attention layer (1-based)", ylabel="Mean attention mass on correct value span")
        ax.legend(title="Distance")
        save(fig, "global_value_attention")
    if summary["causal"]:
        for filename, components, ylabel, metric in (
            ("query_patches", ("residual", "mixer", "mlp"), "Mean fraction of answer-margin gap transferred", "mean_gap_fraction"),
            ("state_and_kv_patches", ("recurrent_state", "value_kv"), "Mean fraction of answer-margin gap transferred", "mean_gap_fraction"),
            ("history_restriction", ("history_restriction",), "Correct-answer margin change", "mean_correct_margin_change"),
        ):
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for component, color in zip(components, COLORS):
                group = [r for r in summary["causal"] if r["component"] == component and r[metric] is not None]
                if not group:
                    continue
                xx = [r["layer_zero_based"] + 1 for r in group]
                yy = [r[metric] for r in group]
                ax.plot(xx, yy, marker="o", color=color, label=component)
                if metric == "mean_gap_fraction":
                    intervals = [r for r in group if r["bootstrap_95_interval"] is not None]
                    if intervals:
                        ax.fill_between([r["layer_zero_based"] + 1 for r in intervals],
                                        [r["bootstrap_95_interval"][0] for r in intervals],
                                        [r["bootstrap_95_interval"][1] for r in intervals], color=color, alpha=.13)
            mark_globals(ax)
            ax.axhline(0, color="black", lw=.6)
            ax.set_ylabel(ylabel)
            ax.legend()
            save(fig, filename)

    lines = ["# Qwen3.5-4B paired-recall pilot", "",
             f"Frozen checkpoint `{summary['revision']}` on {manifest.get('gpu', 'CUDA')}; BF16, batch size 1.", "",
             "| Nominal distance | Correct answers | Both-correct pairs | Both first-difference predictions correct |",
             "|---:|---:|---:|---:|"]
    for d in distances:
        r = summary["by_distance"][str(d)]
        lines.append(f"| {d} | {r['correct_answers']}/{2*r['pairs']} | {r['both_correct_pairs']}/{r['pairs']} | "
                     f"{r['both_first_difference_correct_pairs']}/{r['pairs']} |")
    lines += ["", f"Peak reserved GPU memory: **{summary['peak_reserved_gib']:.2f} GiB**.", "",
              "The distance coordinate is from the older value's final token to the scoring position. The other value is nearby; both actual distances are recorded. Unconditioned greedy generation determines answer accuracy. The logit margin compares the two first differing answer tokens after teacher-forcing only their shared prefix.", "",
              "## Layerwise observations", "",
              "Gray bands mark global attention layers. Mixer and MLP logit-lens changes are reported separately; these measurements are descriptive.", "",
              "![Layerwise logit lens](figures/layerwise_lens.png)", "",
              "![Mixer and MLP contributions](figures/mixer_and_mlp.png)", "",
              "![Source contextualization](figures/source_contextualization.png)", ""]
    if usable:
        global_change = summary["mixer_changes_by_type"]["full_attention"]["mean_absolute_paired_lens_change_per_layer"]
        recurrent_change = summary["mixer_changes_by_type"]["linear_attention"]["mean_absolute_paired_lens_change_per_layer"]
        largest_mlp = max(summary["largest_mean_paired_changes"]["mlp"], key=lambda r: r["mean_paired_lens_change"])
        lines += [f"Global mixers average **{global_change:.2f}** units of absolute paired logit-lens change per layer, compared with **{recurrent_change:.2f}** for recurrent mixers. This comparison accounts for the model's 8 global versus 24 recurrent layers. The largest positive mean MLP change among the leading layers is at layer **{largest_mlp['layer_zero_based'] + 1}** ({largest_mlp['mean_paired_lens_change']:.2f} units). Changes in either direction count in the absolute metric; they are not all improvements in retrieval.", "",
                  "![Global attention on the correct historical value](figures/global_value_attention.png)", ""]
    if summary["causal"]:
        lines += ["## Causal pilot", "",
                  f"Completed causal pairs: **{summary['completed_causal_pairs']}**; recorded trials: **{summary['causal_trials']}**, including **{summary['self_patch_controls']}** self-patch controls.", "",
                  "Both patch directions are run and averaged within each pair. Bootstrap intervals resample underlying examples, keeping all their distances together; the causal pilot normally has three independent example clusters. Candidate selection uses the pilot's solved pairs and is exploratory; a subsequent confirmatory sample should use fixed candidates.", "",
                  "Query residual, mixer-output, and MLP-output patches use uncached executions. Recurrent-state and historical value-span K/V patches use a checkpoint before the final " + str(manifest['protocol']['history_window']) + " tokens. History restriction clears recurrent and convolution states, or masks prefix K/V access, at one selected layer.", "",
                  f"Maximum self-patch margin drift: **{summary['max_self_patch_margin_drift']:.6f}**. Maximum full-to-chunked margin drift: **{summary['max_full_to_chunked_margin_drift']:.6f}**. Cached effects use the corresponding unmodified cached baseline; states are stored in FP32.", "",
                  "![Query patches](figures/query_patches.png)", "",
                  "![State and K/V patches](figures/state_and_kv_patches.png)", "",
                  "![History restriction](figures/history_restriction.png)", ""]
        for component, description in (("value_kv", "Historical value-span K/V exchange"),
                                       ("recurrent_state", "Recurrent-state exchange")):
            entries = [r for r in summary["causal"] if r["component"] == component and r["mean_gap_fraction"] is not None]
            if entries:
                strongest = max(entries, key=lambda r: r["mean_gap_fraction"])
                if component == "recurrent_state" and max(abs(r["mean_gap_fraction"]) for r in entries) < .01:
                    low, high = min(r["mean_gap_fraction"] for r in entries), max(r["mean_gap_fraction"] for r in entries)
                    lines += [f"Mean recurrent-state transfer is near zero at these selected checkpoints: candidate-layer means range from **{low:.4f}** to **{high:.4f}** of the counterfactual margin gap. Other layers can still carry or retrieve historical information.", ""]
                else:
                    lines += [f"{description} has its largest mean donor transfer at layer **{strongest['layer_zero_based'] + 1}**: **{strongest['mean_gap_fraction']:.3f}** of the counterfactual margin gap. This intervention targets the selected component at the recorded checkpoint. Other layers can still carry or retrieve historical information.", ""]
    lines += ["## Interpretation limits", ""] + [f"- {s}." for s in summary["limitations"]]
    lines += ["", "This experiment fixes distractor count, key–value separation, and one history cutoff. It does not compare sliding-window hybrids or identify a universal memory horizon.", "",
              "Machine-readable results: [summary](summary.json), [baseline](baseline.jsonl), [causal trials](causal.jsonl), [run provenance](manifest.json). Figures also have PDF and SVG versions.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
