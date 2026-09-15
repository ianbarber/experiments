"""Project completed independent reviews for publication; never publish or run models."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
import review
import public_replay

CORE_SOURCES = ("review.py", "factorial.py", "phase2.py", "test_review.py", "test_factorial.py",
                "test_phase2.py", "test_analysis_parity.py", "PLAN.json", "PHASE2_PLAN.json")
EXTRA_SOURCES = ("execution_diagnostics.py", "test_execution_diagnostics.py")
FORBIDDEN_KEYS = {"token_ids", "input_ids", "logits", "token_logprobs", "container_id", "container_image",
                  "hostname", "pid", "command", "test_command", "interpreter_path", "private_transcript"}
PRIVATE_PATTERNS = (
    r"(?:/home/|/Users/)[A-Za-z0-9_.-]+/", r"\b(?:192\.168|10\.[0-9]+)\.[0-9]+\.[0-9]+\b", r"\.ts\.net\b",
    r"sk-[A-Za-z0-9]{16,}", r"ghp_[A-Za-z0-9]{16,}", r"hf_[A-Za-z0-9]{16,}", r"AKIA[A-Z0-9]{12,}", r"PRIVATE[ ]KEY",
)


def scan_text(text):
    if any(re.search(pattern, text) for pattern in PRIVATE_PATTERNS):
        raise ValueError("Publication scan rejected private path, network, or credential material")


def scan_object(value):
    if isinstance(value, dict):
        if set(value) & FORBIDDEN_KEYS:
            raise ValueError("Publication projection contains forbidden operational or token fields")
        for item in value.values():
            scan_object(item)
    elif isinstance(value, list):
        for item in value:
            scan_object(item)
    elif isinstance(value, str):
        scan_text(value)


def write_json(path, value):
    scan_object(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def expected_stages():
    names = ["base_calibration_decision_first", "base_calibration_decision_last"]
    names += [f"competence_{recipe}_i{seed}_epoch{epoch}_calibration"
              for recipe in review.RECIPE_NAMES for seed in (1729, 2718) for epoch in (1, 2, 3)]
    names += [f"validation_{recipe}_i{seed}" for recipe in review.RECIPE_NAMES for seed in (1729, 2718)]
    return names


def require_final(root, names):
    if set(names) != set(expected_stages()):
        raise ValueError("Final publication requires every one of the 34 completed independent stage audits")
    directory = root / "results/independent_review"
    required = [root / "results/PROGRAM_COMPLETED.json", directory / "factorial_final/COMPLETED.json",
                directory / "factorial_final/analysis.json", root / "results/SELECTED_RECIPE.json", root / "results/VALIDATION_RESULT.json"]
    if any(not p.is_file() for p in required):
        raise ValueError("Final publication requires terminal program and completed independent factorial review")
    proof = json.loads(required[1].read_text())
    analysis = json.loads(required[2].read_text())
    if proof["status"] != "passed" or review.sha(required[2]) != proof["artifacts_sha256"]["analysis.json"]:
        raise ValueError("Final independent factorial proof differs")
    if analysis["scoped_stages_recomputed"] != 34 or analysis["case_checkpoint_records_recomputed"] != 8064:
        raise ValueError("Final independent factorial coverage differs")
    for relative, digest in analysis["inputs_sha256"].items():
        if review.sha(root / relative) != digest:
            raise ValueError("Final independent audit source changed")
    terminal = json.loads(required[0].read_text())
    if terminal["outcome"] not in {"full_fixed_comparison_completed", "feasibility_gate_failed"}:
        raise ValueError("Unexpected terminal scientific outcome")
    repair = None
    if terminal["outcome"] == "full_fixed_comparison_completed":
        path = directory / "phase2/FINAL_REPAIR_REVIEW.json"
        if not path.is_file():
            raise ValueError("Full program requires completed independent repair review")
        repair = json.loads(path.read_text())
        if repair["checkpoints"] != 34 or repair["case_checkpoint_records"] != 26112 or repair["program_completion_sha256"] != review.sha(required[0]):
            raise ValueError("Completed repair audit coverage or terminal source differs")
        for filename, digest in repair["stage_audit_sha256"].items():
            if review.sha(directory / "phase2" / filename) != digest:
                raise ValueError("Repair audit source changed")
    elif analysis["both_selected_validation_gates_passed"]:
        stops = [json.loads(p.read_text()) for p in (directory / "phase2").glob("*_stop_*.json")]
        if not any(stop["status"] == "verified_scientific_stop" and stop["reason"] == terminal["reason"] and
                   stop["program_completion_sha256"] == review.sha(required[0]) for stop in stops):
            raise ValueError("Conditional feasibility stop requires its independent completed audit")
    return terminal, analysis, repair


def project_stage(root, name):
    directory = root / "results/independent_review"
    path = directory / (name + ".json")
    report = json.loads(path.read_text())
    cases_path = directory / (name + ".cases.jsonl")
    actual = root / report["stage_spec"]["output"]
    if not (actual / "COMPLETED.json").is_file():
        raise ValueError("Only completed stage evidence can be projected")
    if (review.sha(actual / "COMPLETED.json") != report["completion_sha256"] or
            review.sha(actual / "outputs.jsonl") != report["outputs_sha256"] or
            review.sha(cases_path) != report["per_case_review_sha256"]):
        raise ValueError("Audited stage or case source changed")
    source = root / report["stage_spec"]["args"]["data"]
    if review.sha(source) != report["source_data_sha256"]:
        raise ValueError("Audited source data changed")
    cases = review.read_rows(cases_path)
    source_rows = review.read_rows(source)
    if [row["id"] for row in cases] != [row["id"] for row in source_rows]:
        raise ValueError("Projected source/case coverage differs")
    by_id = {row["id"]: row for row in source_rows}
    scores = []
    for case in cases:
        if review.row_sha(by_id[case["id"]]) != case["source_row_sha256"]:
            raise ValueError("Source case bytes differ")
        scores.append({"id": case["id"], "position": case["frozen_file_position_zero_based"],
                       "stratum": case["stratum"], "family": case["family"],
                       **{metric: int(case["independent"][metric]) for metric in public_replay.METRICS}})
    examples = []
    for case in report["qualitative_examples"]:
        source_case = by_id[case["id"]]
        if case["policy"] != source_case["policy"] or case["ledger"] != source_case["ledger"]:
            raise ValueError("Illustrative ledger or policy differs from its source")
        selections = []
        if "fixed_before_outputs_first_two_source_IDs_in_stratum" in case["selection_reasons"]:
            selections.append("fixed_longitudinal")
        if "prespecified_per_stage_first_two_full_errors_in_stratum" in case["selection_reasons"]:
            selections.append("per_stage_first_two_errors")
        examples.append({"id": case["id"], "position": case["frozen_file_position_zero_based"],
                         "stratum": case["stratum"], "family": case["family"], "selection_rules": selections,
                         "source_row_sha256": case["source_row_sha256"],
                         "domain": source_case["domain"], "style": source_case["style"],
                         "pressure": source_case["pressure"], "oversight": source_case["oversight"],
                         "policy": case["policy"], "ledger": case["ledger"],
                         "raw_response": case["generated"]["text"], "finish_reason": case["generated"]["finish_reason"],
                         "oracle": case["independent"]["expected"], "full_correct": case["independent"]["full_correct"],
                         "error_components": case["independent"]["error_components"]})
    summary = {"stage": name, "split": report["split"], "completed_utc": report["completed_utc"],
               "reviewed_utc": report["reviewed_utc"], "groups": report["groups"],
               "descriptive_competence_gate": report["competence_gate"],
               "gate_scope": "Only epoch-3 calibration and the selected recipe's validation can control progression; base and intermediate checks are descriptive.",
               "requested_output_order": report["stage_spec"]["args"]["output_order"],
               "source_sha256": {"independent_audit": review.sha(path), "complete_case_audit": review.sha(cases_path),
                                 "stage_completion": report["completion_sha256"], "original_raw_outputs": report["outputs_sha256"],
                                 "source_data": report["source_data_sha256"], "initial_adapter": report["initial_adapter_sha256"],
                                 "review_script": report["review_script_sha256"], "plan": report["plan_sha256"]}}
    for item in (summary, scores, examples):
        scan_object(item)
    return summary, scores, examples


def build(root, output, mode, requested=None):
    if mode not in {"development", "final"}:
        raise ValueError("Publication mode must be explicit")
    if output.exists():
        raise ValueError("Public projection must use a new output directory")
    directory = root / "results/independent_review"
    freeze = json.loads((directory / "IMPLEMENTATION_FREEZE.json").read_text())
    addition = json.loads((directory / "EXECUTION_DIAGNOSTIC_ADDITION.json").read_text())
    for relative, digest in freeze["files_sha256"].items():
        if review.sha(root / relative) != digest:
            raise ValueError("Frozen independent source changed")
    for name in EXTRA_SOURCES:
        if review.sha(directory / name) != addition["files_sha256"][name]:
            raise ValueError("Logged additive diagnostic source changed")
    allowed = expected_stages()
    names = requested or [name for name in allowed if (directory / (name + ".json")).is_file()]
    if not names or len(set(names)) != len(names) or not set(names) <= set(allowed):
        raise ValueError("Expected unique in-scope completed stage names")
    names = [name for name in allowed if name in names]
    terminal, final_factorial, final_repair = require_final(root, names) if mode == "final" else (None, None, None)
    sources = []
    with tempfile.TemporaryDirectory(prefix="independent-public-projection-") as temporary:
        package = Path(temporary) / "package"
        code, results = package / "code/independent_review", package / "results/independent_review"
        code.mkdir(parents=True)
        results.mkdir(parents=True)
        for name in CORE_SOURCES + EXTRA_SOURCES:
            source = directory / name
            scan_text(source.read_text())
            shutil.copyfile(source, code / name)
            sources.append({"public_path": "code/independent_review/" + name, "original_relative_path": "results/independent_review/" + name,
                            "sha256": review.sha(source), "copy_kind": "exact_original_bytes"})
        for name in ("public_replay.py", "package.py"):
            source = HERE / name
            scan_text(source.read_text())
            shutil.copyfile(source, code / name)
            sources.append({"public_path": "code/independent_review/" + name, "sha256": review.sha(source), "copy_kind": "additive_publication_utility"})
        stages = []
        fixed_success = {}
        for name in names:
            summary, scores, examples = project_stage(root, name)
            stages.append(summary)
            fixed_cases = [row for row in examples if "fixed_longitudinal" in row["selection_rules"]]
            fixed_success[name] = (sum(row["full_correct"] for row in fixed_cases), len(fixed_cases))
            scores_path = results / "cases" / (name + ".csv")
            scores_path.parent.mkdir(exist_ok=True)
            with scores_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
                writer.writeheader(); writer.writerows(scores)
            example_path = results / "illustrations" / (name + ".jsonl")
            example_path.parent.mkdir(exist_ok=True)
            example_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples))
            diagnostic = directory / "execution_diagnostics" / (name + ".json")
            if diagnostic.is_file():
                record = json.loads(diagnostic.read_text())
                scan_object(record)
                expected = summary["source_sha256"]
                if (record["source_hashes"] != {"completion": expected["stage_completion"], "outputs": expected["original_raw_outputs"],
                                               "cases": expected["complete_case_audit"], "audit": expected["independent_audit"]}
                        or record["script_sha256"] != addition["files_sha256"]["execution_diagnostics.py"]
                        or record["source_data_sha256"] != expected["source_data"]):
                    raise ValueError("Logged component diagnostic source differs")
                target = results / "execution_diagnostics" / (name + ".json")
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(diagnostic, target)
                sources.append({"public_path": str(target.relative_to(package)),
                                "original_relative_path": "results/independent_review/execution_diagnostics/" + name + ".json",
                                "sha256": review.sha(diagnostic), "copy_kind": "exact_original_scientific_component_audit"})
        write_json(results / "stage_summaries.json", stages)
        write_json(results / "source_manifest.json", sources)
        write_json(results / "implementation_history.json", {
            "projection_of_original_records": True,
            "original_implementation_freeze_sha256": review.sha(directory / "IMPLEMENTATION_FREEZE.json"),
            "created_utc": freeze["created_utc"], "scope": freeze["scope"], "tests_passed_at_freeze": freeze["tests"],
            "program_pins_verified_unchanged_at_freeze": freeze["verified_unchanged_program_pins"],
            "later_execution_diagnostic_addition": {"original_sha256": review.sha(directory / "EXECUTION_DIAGNOSTIC_ADDITION.json"),
                                                     "created_utc": addition["created_utc"], "scope": addition["scope"],
                                                     "tests_passed": addition["tests"]["count"]}})
        status = "final_independent_review" if mode == "final" else "development_evidence_only"
        if final_factorial is not None:
            write_json(results / "final_factorial_summary.json", {key: final_factorial[key] for key in (
                "scoped_stages_recomputed", "case_checkpoint_records_recomputed", "validation_cases", "calibration_cases", "validation_models",
                "selected_recipe", "both_selected_validation_gates_passed", "calibration_selection_exact_match", "validation_no_fallback_exact_match", "cells", "effects", "bootstrap")})
            if final_repair is not None:
                write_json(results / "final_repair_summary.json", {key: final_repair[key] for key in (
                    "checkpoints", "case_checkpoint_records", "primary_ids", "primary_cell_counts", "contrasts", "control_preservation", "control_direction", "interval_scope")})
        lines = ["# Independent review evidence", "", "Status: **" + status + "**.", "",
                 f"This projection contains {len(stages)} completed stage audits and {sum(s['groups']['all']['n'] for s in stages):,} case/checkpoint records. Repeated cases are not independent observations.", "",
                 "The CSV files retain every audited binary score. The illustration files retain full ledger facts, policy and raw response text for the fixed longitudinal cases and the separately preplanned first two errors per stratum. The two selection rules are explicitly labelled; examples are not prevalence samples. Token-level records and local operational details are omitted.", "",
                 "The original checker and plan files in `code/independent_review` are exact source copies, identified in `source_manifest.json`. This summary, the CSV tables, illustration records and implementation history are public projections, not the original audit bytes.", "",
                 "Run from the package root:", "", "```sh", "python3 code/independent_review/public_replay.py --package .", "```", "",
                 "This weights-free check recounts every projected score, recounts any included execution-component tables, and re-executes every published illustration. The separate main saved-data replay checks all original response text and final numerical analysis. Exact-source `review.py`, `factorial.py` and `phase2.py` also offer stronger original-workspace contract and adapter-byte checks; those require the retained original source layout, completion contracts and checkpoint files, and are not implied by this public projection replay.", "",
                 "Only epoch 3 can select a recipe, selection uses calibration alone, and validation allows no fallback. An intermediate gate pass does not establish final eligibility. A scientific prerequisite failure is not a repair null result.", ""]
        if terminal is not None:
            lines += ["Terminal scientific outcome: `" + terminal["outcome"] + "`.", ""]
        else:
            lines += ["Final factorial, validation and conditional repair conclusions remain pending in this development projection.", ""]
        lines += ["## Completed-stage evidence", "",
                  "| Stage | Full correct | Decision correct | CLEAR decision correct | Fixed examples full correct | Descriptive gate |",
                  "|---|---:|---:|---:|---:|---|"]
        for stage in stages:
            group = stage["groups"]
            n = group["all"]["n"]
            fixed_numerator, fixed_denominator = fixed_success[stage["stage"]]
            evidence = "illustrations/" + stage["stage"] + ".jsonl"
            lines.append(f"| [{stage['stage']}]({evidence}) | {group['all']['counts']['full_correct']}/{n} | {group['all']['counts']['decision_correct']}/{n} | {group['clear_control']['counts']['decision_correct']}/{group['clear_control']['n']} | {fixed_numerator}/{fixed_denominator} | {'pass' if stage['descriptive_competence_gate']['passed'] else 'fail'} |")
        lines += ["", "Fixed examples are selected before reviewer outcome access and are followed unchanged. Their success rate is an illustration count, not an accuracy estimate; the full denominator is shown alongside it. Each linked file also contains the separately labelled first-error examples, with their complete source facts and observed responses.", ""]
        (results / "REVIEW.md").write_text("\n".join(lines))
        (code / "README.md").write_text("# Independent review source\n\nThese source files are identified as exact originals or additive publication utilities in `../../results/independent_review/source_manifest.json`. The accompanying review explains the distinction between weights-free public recounting and optional original-workspace adapter-byte verification. From the entry root, run `python3 code/independent_review/public_replay.py --package .`. The two owned subtrees can be merged into the full entry without replacing its main code README. The source tests for the full original workspace additionally require its frozen source/data contracts and the separately frozen main analyzer; they are not standalone public replay commands.\n")
        for path in package.rglob("*"):
            if path.is_file():
                scan_text(path.read_text())
                if path.stat().st_size >= 2_000_000:
                    raise ValueError("Public review artifact is unexpectedly large")
        artifacts = {str(path.relative_to(package)): {"sha256": review.sha(path), "bytes": path.stat().st_size}
                     for path in sorted(package.rglob("*")) if path.is_file()}
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "publication_status": status,
                    "completed_stage_count": len(stages), "case_checkpoint_records": sum(s["groups"]["all"]["n"] for s in stages),
                    "illustration_scope": "Fixed longitudinal cases and per-stage first two errors are separately labelled illustrations, not representative samples.",
                    "artifacts": artifacts, "source_originals_preserved": True,
                    "original_stage_weight_byte_reverification_in_this_projection": False,
                    "privacy_scope": "Allowlisted scientific fields; no token IDs, local home paths, operational IDs, raw HTTP logs or private transcripts."}
        write_json(results / "MANIFEST.json", manifest)
        proof = public_replay.verify(package)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(package, output)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE.parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("development", "final"), required=True)
    parser.add_argument("--stage", action="append")
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.output.resolve(), args.mode, args.stage), indent=2))


if __name__ == "__main__":
    main()
