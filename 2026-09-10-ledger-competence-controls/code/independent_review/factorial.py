"""Completion-gated independent factorial review; CPU saved data only.

Requires all 34 scoped stage audits and the frozen selection/validation records.
Does not read repair outcomes or treat a partial factorial as a final analysis.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import review

STRATA = ("trigger_report", "control_report", "clear")
EFFECTS = ("reweighting", "decision_last", "interaction")
COEFFICIENTS = np.array([[-.5, .5, -.5, .5], [-.5, -.5, .5, .5], [1., -1., -1., 1.]])


def shared_draws(strata, draws=10000, seed=20260910):
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    blocks = []
    for label in STRATA:
        indices = np.flatnonzero(strata == label)
        if len(indices) == 0:
            raise ValueError("Missing authored stratum")
        blocks.append(indices[rng.integers(0, len(indices), size=(draws, len(indices)))])
    return np.concatenate(blocks, axis=1), blocks[-1]


def effect_cases(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[:2] != (2, 4) or not np.isin(values, [0., 1.]).all():
        raise ValueError("Expected binary case scores for exactly two seeds and four recipes")
    # Construct contrasts independently from the main analysis using a linear
    # contrast matrix; each case retains all recipe values and both seeds.
    return np.einsum("ec,scn->sen", COEFFICIENTS, values)


def summarize_effects(values, indices, point_indices):
    effects = effect_cases(values)
    curves = np.concatenate([effects, effects.mean(axis=0, keepdims=True)], axis=0)
    bootstrap = np.empty((3, 3, len(indices)))
    for begin in range(0, len(indices), 250):
        chosen = indices[begin:begin + 250]
        bootstrap[:, :, begin:begin + len(chosen)] = curves[:, :, chosen].mean(axis=-1)
    point = curves[:, :, point_indices].mean(axis=-1)
    results = {}
    for effect, name in enumerate(EFFECTS):
        values_by_seed = {}
        for seed_index, seed_name in enumerate(("1729", "2718", "observed_seed_mean")):
            low, high = np.quantile(bootstrap[seed_index, effect], (.025, .975), method="linear")
            values_by_seed[seed_name] = {"difference": float(point[seed_index, effect]),
                                        "ci95": [float(low), float(high)]}
        results[name] = {"role": "secondary_interaction" if name == "interaction" else "prespecified_main_effect",
                         "values": values_by_seed}
    return results


def official_summary(report):
    return {name: {"n": group["n"], **{metric: group["rates"][metric] for metric in review.METRICS}}
            for name, group in report["groups"].items()}


def compute(root):
    directory = root / "results/independent_review"
    program, freeze, plan = review.frozen_program(root)
    stages, training = review.scoped_stages(program)
    required = [directory / (name + suffix) for name in stages for suffix in (".json", ".cases.jsonl")]
    required += [root / "results/SELECTED_RECIPE.json", root / "results/VALIDATION_RESULT.json"]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Final factorial audit requires every completed scoped audit and selection/validation record; missing: " + ", ".join(missing))
    checker = review.contract_checker(root, program, freeze)
    checked, per_case = {}, {}
    inputs = {}
    for name, stage in stages.items():
        saved_path = directory / (name + ".json")
        cases_path = directory / (name + ".cases.jsonl")
        saved = json.loads(saved_path.read_text())
        if review.sha(cases_path) != saved["per_case_review_sha256"]:
            raise ValueError("Saved independent case audit changed")
        actual, cases = review.review(root, stage, program, freeze, plan, checker, training)
        for key in ("stage_spec", "groups", "competence_gate", "completion_sha256", "outputs_sha256", "qualitative_examples"):
            if actual[key] != saved[key]:
                raise ValueError("Fresh complete-data recomputation differs from saved independent audit: " + name + "/" + key)
        if cases != review.read_rows(cases_path):
            raise ValueError("Complete casewise independent recomputation differs")
        checked[name], per_case[name] = actual, cases
        inputs[str(saved_path.relative_to(root))] = review.sha(saved_path)
        inputs[str(cases_path.relative_to(root))] = review.sha(cases_path)
        inputs[str((root / stage["output"] / "COMPLETED.json").relative_to(root))] = actual["completion_sha256"]
    calibration = {recipe: {} for recipe in review.RECIPE_NAMES}
    validation = {recipe: {} for recipe in review.RECIPE_NAMES}
    cal_contracts, adapters = {}, {}
    for recipe in review.RECIPE_NAMES:
        for seed in (1729, 2718):
            name = f"competence_{recipe}_i{seed}_epoch3_calibration"
            report = checked[name]
            calibration[recipe][str(seed)] = official_summary(report)
            validation[recipe][str(seed)] = official_summary(checked[f"validation_{recipe}_i{seed}"])
            cal_contracts[f"{recipe}:{seed}"] = report["completion_sha256"]
            adapters[f"{recipe}:{seed}"] = report["initial_adapter_sha256"]
    selected = program.select_recipe(calibration)
    selected.update(source_snapshot=freeze["source_snapshot"], calibration_contracts=cal_contracts,
                    epoch3_adapter_hashes=adapters)
    selection_path = root / "results/SELECTED_RECIPE.json"
    if selected != json.loads(selection_path.read_text()):
        raise ValueError("Frozen recipe differs from independent calibration-only selection")
    validation_result = program.validation_decision(selected, validation)
    expected_validation = {**validation_result, "selected_recipe_sha256": review.sha(selection_path)}
    validation_path = root / "results/VALIDATION_RESULT.json"
    if expected_validation != json.loads(validation_path.read_text()):
        raise ValueError("Frozen validation decision differs from independent no-fallback check")
    data = review.read_rows(root / "data/validation.jsonl")
    if len(data) != 384:
        raise ValueError("Final validation must contain exactly 384 cases")
    ids = [row["id"] for row in data]
    strata = [row["stratum"] for row in data]
    if any(strata.count(label) != 128 for label in STRATA):
        raise ValueError("Expected 128 cases per validation stratum")
    for recipe in review.RECIPE_NAMES:
        for seed in (1729, 2718):
            if [case["id"] for case in per_case[f"validation_{recipe}_i{seed}"]] != ids:
                raise ValueError("Cross-recipe case pairing changed")
    full_indices, clear_indices = shared_draws(strata)
    results, cells = {}, {}
    for endpoint, metric, indices in (("clear_decision_accuracy", "decision_correct", clear_indices),
                                      ("complete_correctness", "full_correct", full_indices)):
        values = np.asarray([[[case["independent"][metric] for case in per_case[f"validation_{recipe}_i{seed}"]]
                              for recipe in review.RECIPE_NAMES] for seed in (1729, 2718)], dtype=float)
        subset = np.flatnonzero(np.asarray(strata) == "clear") if endpoint == "clear_decision_accuracy" else np.arange(384)
        results[endpoint] = summarize_effects(values, indices, subset)
        cells[endpoint] = {recipe: {str(seed): {"correct": int(values[i, j, subset].sum()),
                                             "n": len(subset), "accuracy": float(values[i, j, subset].mean())}
                                   for i, seed in enumerate((1729, 2718))}
                           for j, recipe in enumerate(review.RECIPE_NAMES)}
    inputs["results/SELECTED_RECIPE.json"] = review.sha(selection_path)
    inputs["results/VALIDATION_RESULT.json"] = review.sha(validation_path)
    return {"scope": "Completed competence factorial only; repair outcomes are outside this independent audit.",
            "scoped_stages_recomputed": len(checked), "case_checkpoint_records_recomputed": sum(len(v) for v in per_case.values()),
            "validation_cases": 384, "calibration_cases": 192, "validation_models": 8,
            "selected_recipe": selected["selected_recipe"], "both_selected_validation_gates_passed": validation_result["passed"],
            "calibration_selection_exact_match": True, "validation_no_fallback_exact_match": True,
            "cells": cells, "effects": results,
            "bootstrap": {"draws": 10000, "seed": 20260910, "strata_order": list(STRATA),
                          "within_stratum_draws": 128, "paired_across_all_recipes_and_both_seeds": True,
                          "shared_draws_sha256": hashlib.sha256(full_indices.astype("<i8").tobytes()).hexdigest(),
                          "quantile_method": "linear", "interval_scope": "Unadjusted descriptive intervals conditional on these authored cases and two observed seeds; no seed-population resampling."},
            "source_snapshot": freeze["source_snapshot"], "inputs_sha256": inputs}


def markdown(result):
    lines = ["# Independent completed factorial review", "", result["scope"], "",
             "All 34 scoped stages and 8,064 case/checkpoint records were recomputed. Recipe selection and the no-fallback validation decision agree exactly with the frozen records.", "",
             "Selected recipe: `" + str(result["selected_recipe"]) + "`; both selected validation gates passed: `" + str(result["both_selected_validation_gates_passed"]) + "`.", "",
             "Differences and intervals below are percentage points. Positive main effects favor reweighting or decision-last. Interactions are secondary.", "",
             "| Endpoint | Effect | Seed 1729 | Seed 2718 | Observed-seed mean [95% interval] |",
             "|---|---|---:|---:|---:|"]
    for endpoint, effects in result["effects"].items():
        for name, effect in effects.items():
            values = effect["values"]
            mean = values["observed_seed_mean"]
            lines.append(f"| {endpoint} | {name} | {100*values['1729']['difference']:.3f} | {100*values['2718']['difference']:.3f} | {100*mean['difference']:.3f} [{100*mean['ci95'][0]:.3f}, {100*mean['ci95'][1]:.3f}] |")
    lines += ["", result["bootstrap"]["interval_scope"], "", "Cell numerators, denominators, seed-specific intervals, exact pairing/draw identity, and source contracts are retained in the JSON. This independent review changes neither selection nor program progression.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=review.ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = compute(root)
    output = args.output_dir or root / "results/independent_review/factorial_final"
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "REVIEW.md").write_text(markdown(result))
    proof = {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
             "analysis_script_sha256": review.sha(Path(__file__)),
             "checker_script_sha256": review.sha(Path(__file__).with_name("review.py")),
             "artifacts_sha256": {p.name: review.sha(p) for p in output.iterdir() if p.is_file()},
             "model_or_service_calls": False}
    (output / "COMPLETED.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"status": "passed", "output": str(output), "selected_recipe": result["selected_recipe"]}))


if __name__ == "__main__":
    main()
