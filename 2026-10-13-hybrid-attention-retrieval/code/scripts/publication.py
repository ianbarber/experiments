"""Export small numerical evidence and analyze the fixed two-seed expansion."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_write(path, rows):
    if not rows:
        raise ValueError("Refusing an empty evidence table")
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: int(v) if isinstance(v, bool) else v for k, v in row.items()})


def csv_read(path):
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    integers = {"seed", "example", "distance", "layer", "control", "cached_baseline", "correct", "strict_format",
                "first_difference_correct", "source_distance", "scoring_tokens"}
    strings = {"side", "component", "layer_type"}
    return [{key: (None if value == "" else int(value) if key in integers else
                   value if key in strings else float(value)) for key, value in row.items()} for row in rows]


def export_run(run, destination):
    manifest = json.loads((run / "manifest.json").read_text())
    if "finished_utc" not in manifest or manifest.get("stopped_reason"):
        raise ValueError("Export a completed run; preserve stopped runs separately")
    seed = manifest["protocol"]["seed"]
    baseline = jsonl(run / "baseline.jsonl")
    causal = jsonl(run / "causal.jsonl")
    kinds = manifest["layer_types"]
    endpoints, layers, trials = [], [], []
    for r in baseline:
        example = int(r["pair_id"].rsplit("-p", 1)[1])
        common = {"seed": seed, "example": example, "distance": r["nominal_distance"]}
        for side in ("a", "b"):
            endpoints.append(common | {"side": side, "correct": r[side]["answer_correct"],
                 "strict_format": bool(re.fullmatch(r"\s*v\d{3}\.?\s*", r[side]["generated_answer"])),
                 "first_difference_correct": r[side]["first_difference_correct"],
                 "margin": r[side]["margin_a_minus_b"], "trace_seconds": r[side]["seconds"],
                 "source_distance": r["distance_" + side], "scoring_tokens": r["scoring_tokens"]})
        for i, kind in enumerate(kinds):
            a, b = r["a"]["trace"], r["b"]["trace"]
            mass = None
            if kind == "full_attention":
                slot = a["attention_layers"].index(i)
                mass = float((np.array(a["attention_value_mass"])[slot, :, 0].mean() +
                              np.array(b["attention_value_mass"])[slot, :, 1].mean()) / 2)
            sensitivity = r["source_sensitivity"]["relative_paired_difference"][i]
            layers.append(common | {"layer": i + 1, "layer_type": kind,
                "separation": a["lens_after"][i] - b["lens_after"][i],
                "mixer_change": a["lens_mixer_change"][i] - b["lens_mixer_change"][i],
                "mlp_change": a["lens_mlp_change"][i] - b["lens_mlp_change"][i],
                "source_1_relative_difference": sensitivity[0], "source_2_relative_difference": sensitivity[1],
                "query_relative_difference": sensitivity[2], "correct_value_attention_mass": mass})
    for r in causal:
        trials.append({"seed": seed, "example": int(r["pair_id"].rsplit("-p", 1)[1]),
            "distance": r["nominal_distance"], "side": r["side"], "layer": r["layer"] + 1,
            "layer_type": r["layer_type"], "component": r["component"], "control": r["control"],
            "cached_baseline": r["cached_baseline"], "baseline_margin": r["baseline_margin"],
            "donor_margin": r["donor_margin"], "patched_margin": r["patched_margin"],
            "margin_delta": r["margin_delta"], "gap_fraction": r["fraction_of_counterfactual_gap"],
            "correct_margin_change": r["correct_margin_change"]})
    destination.mkdir(parents=True, exist_ok=True)
    csv_write(destination / "answers.csv.gz", endpoints)
    csv_write(destination / "layers.csv.gz", layers)
    csv_write(destination / "interventions.csv.gz", trials)
    factworld = manifest["protocol"]["factworld"]
    factworld.pop("root", None)
    factworld["source"] = "FactWorld minimal data/oracle/renderer snapshot"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    detailed_summary = json.loads((run / "summary.json").read_text())
    summary = {key: detailed_summary[key] for key in ("model", "revision", "n_pairs", "both_answers_correct",
        "by_distance", "peak_reserved_gib", "max_self_patch_margin_drift", "max_full_to_chunked_margin_drift",
        "causal_trials", "self_patch_controls", "completed_causal_pairs", "unique_baseline_examples")}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cache = [{"example": int(r["pair_id"].rsplit("-p", 1)[1]), "seed": seed,
              "boundary": r["boundary"], "window": r["history_window"],
              "state_dtype": r["recurrent_state_dtype"], "margin_drift": r["full_to_chunked_margin_drift"]}
             for r in jsonl(run / "cache_controls.jsonl")]
    (destination / "cache_controls.json").write_text(json.dumps(cache, indent=2) + "\n")
    return {"seed": seed, "paired_cases": len(baseline), "interventions": len(causal)}


def interval(values, draws=10000):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Endpoint values must be finite and nonempty")
    resampled = np.random.default_rng(0).choice(values, (draws, len(values)), replace=True).mean(-1)
    low, high = np.quantile(resampled, [.025, .975])
    return {"n_examples": len(values), "mean": float(values.mean()), "ci95": [float(low), float(high)]}


def participant_means(rows, component, distance=8192, layer=None):
    groups = defaultdict(list)
    for r in rows:
        if not r["control"] and r["distance"] == distance and r["component"] == component:
            if layer is None or r["layer"] == layer:
                if r["gap_fraction"] is not None:
                    groups[(r["seed"], r["example"])].append(r["gap_fraction"])
    return {key: float(np.mean(values)) for key, values in groups.items()}


def locked_endpoints(rows, seed=None):
    selected = [r for r in rows if seed is None or r["seed"] == seed]
    kv = participant_means(selected, "value_kv")
    state = participant_means(selected, "recurrent_state")
    mlp = participant_means(selected, "mlp", layer=31)
    common = sorted(kv.keys() & state.keys())
    primary = interval([kv[key] - state[key] for key in common])
    secondary = interval(list(mlp.values()))
    for result in (primary, secondary):
        result["decision"] = ("support" if result["mean"] >= .05 and result["ci95"][0] > 0 else
                              "reject" if result["mean"] < .05 or result["ci95"][1] <= 0 else "inconclusive")
    return {"global_kv_minus_recurrent_state_8192": primary, "layer31_mlp_transfer_8192": secondary,
            "global_value_kv_transfer_8192": interval(list(kv.values())),
            "recurrent_state_transfer_8192": interval(list(state.values()))}


def analyze(evidence, images):
    endpoints, layers, trials, manifests = [], [], [], []
    for seed in (42, 314159):
        root = evidence / f"seed{seed}"
        endpoints += csv_read(root / "answers.csv.gz")
        layers += csv_read(root / "layers.csv.gz")
        trials += csv_read(root / "interventions.csv.gz")
        manifests.append(json.loads((root / "manifest.json").read_text()))
    result = {"data_seeds": [42, 314159], "unique_examples": len({(r['seed'], r['example']) for r in endpoints}),
              "paired_cases": len(endpoints)//2, "answers": len(endpoints),
              "correct_answers": sum(r["correct"] for r in endpoints),
              "correct_first_differences": sum(r["first_difference_correct"] for r in endpoints),
              "malformed_answers": sum(not r["strict_format"] for r in endpoints),
              "causal_trials": len(trials), "locked_endpoints": locked_endpoints(trials),
              "by_seed": {str(seed): locked_endpoints(trials, seed) for seed in (42, 314159)},
              "excluded_small_denominator_trials": sum(not r["control"] and r["gap_fraction"] is None for r in trials),
              "max_self_patch_margin_drift": max(abs(r["margin_delta"]) for r in trials if r["control"]),
              "peak_reserved_gib": max(m["peak_reserved_gib"] for m in manifests), "by_distance": {},
              "layer_effects": [], "bootstrap_draws": 10000,
              "bootstrap_unit": "data seed and underlying example; both directions and all distances stay together"}
    for distance in (128, 1024, 4096, 8192):
        cell = [r for r in endpoints if r["distance"] == distance]
        paired = defaultdict(list)
        for r in cell:
            paired[(r["seed"], r["example"])].append(r["correct"])
        success = sum(all(v) for v in paired.values())
        n = len(paired)
        z, rate = 1.959963984540054, success/n
        denominator = 1 + z*z/n
        center = (rate + z*z/(2*n)) / denominator
        half = z * np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / denominator
        result["by_distance"][str(distance)] = {"n_pairs": n, "correct_answers": sum(r["correct"] for r in cell),
             "both_correct_pairs": success, "pair_accuracy": rate, "pair_accuracy_wilson95": [center-half, center+half]}
    for component in ("mixer", "mlp", "recurrent_state", "value_kv", "history_restriction"):
        chosen_layers = sorted({r["layer"] for r in trials if r["component"] == component})
        for layer in chosen_layers:
            relevant = [r for r in trials if not r["control"] and r["component"] == component and
                         r["layer"] == layer and r["distance"] == 8192]
            if component == "history_restriction":
                groups = defaultdict(list)
                for r in relevant:
                    groups[(r["seed"], r["example"])].append(r["correct_margin_change"])
                stats = interval([np.mean(v) for v in groups.values()])
            else:
                stats = interval(list(participant_means(trials, component, layer=layer).values()))
            result["layer_effects"].append({"component": component, "layer": layer,
                "layer_type": relevant[0]["layer_type"], **stats})
    (evidence / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    images.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    colors = ["#4477aa", "#228833", "#cc6677", "#aa3377"]
    globals_ = [4, 8, 12, 16, 20, 24, 28, 32]

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(images / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    def shade(ax):
        for layer in globals_:
            ax.axvspan(layer-.35, layer+.35, color="gray", alpha=.12, zorder=0)
        ax.set_xlabel("Decoder layer (1-based; gray = global)")
        ax.set_xticks(globals_)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for distance, color in zip((128, 1024, 4096, 8192), colors):
        means = [np.mean([r["separation"] for r in layers if r["distance"]==distance and r["layer"]==layer])
                 for layer in range(1,33)]
        ax.plot(range(1,33), means, label=f"{distance} tokens (n=40)", color=color)
    shade(ax); ax.set_ylabel("Paired normalized logit-lens separation"); ax.legend()
    finish(fig, "layerwise_lens")
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    for ax, distance in zip(axes, (128,8192)):
        for offset, component, color in ((-.18,"mixer",colors[0]),(.18,"mlp",colors[2])):
            values = [np.mean([r[component+"_change"] for r in layers if r["distance"]==distance and r["layer"]==layer])
                      for layer in range(1,33)]
            ax.bar(np.arange(1,33)+offset, values,.36,label=component,color=color)
        shade(ax); ax.set_title(f"{distance} tokens"); ax.axhline(0,color="black",lw=.6)
        ax.set_ylabel("Change in paired lens separation")
    axes[0].legend(); finish(fig,"mixer_and_mlp")
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    components = (("mixer","mlp"),("value_kv","recurrent_state"),("history_restriction",))
    labels = {"mixer": "Mixer", "mlp": "MLP", "value_kv": "Historical value K/V",
              "recurrent_state": "Recurrent state", "history_restriction": "History restriction"}
    for ax, panel in zip(axes,components):
        for component, color in zip(panel,colors):
            group=[r for r in result["layer_effects"] if r["component"]==component]
            xs=[r["layer"] for r in group]; ys=[r["mean"] for r in group]
            ax.plot(xs,ys,marker="o",label=labels[component],color=color)
            ax.fill_between(xs,[r["ci95"][0] for r in group],[r["ci95"][1] for r in group],color=color,alpha=.14)
        shade(ax); ax.axhline(0,color="black",lw=.6); ax.legend()
        ax.set_xlabel("Decoder layer (1-based)")
        ax.set_ylabel("Change in correct answer margin" if panel[0]=="history_restriction" else "Fraction of margin gap transferred")
    finish(fig,"causal_components")
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    for ax, field, title in zip(axes,("source_1_relative_difference","source_2_relative_difference","query_relative_difference"),
                                ("Historical V1","Historical V2","Scoring position")):
        for distance,color in zip((128,1024,4096,8192),colors):
            values=[np.mean([r[field] for r in layers if r["distance"]==distance and r["layer"]==layer]) for layer in range(1,33)]
            ax.plot(range(1,33),values,label=str(distance),color=color)
        shade(ax); ax.set_title(title); ax.set_xlabel("Decoder layer (1-based)")
        ax.set_ylabel("Relative paired representation difference")
    axes[0].legend(title="Distance"); finish(fig,"source_contextualization")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    exp=sub.add_parser("export"); exp.add_argument("--run",type=Path,required=True); exp.add_argument("--destination",type=Path,required=True)
    ana=sub.add_parser("analyze"); ana.add_argument("--evidence",type=Path,default=Path("../results")); ana.add_argument("--images",type=Path,default=Path("../images"))
    args=parser.parse_args()
    result=export_run(args.run,args.destination) if args.command=="export" else analyze(args.evidence,args.images)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
