"""Weights-free checks of the published independent-review projection.

Recounts all projected scores and re-executes every published illustration.
It does not replace the main replay of all saved responses or prove weight bytes.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import review
import execution_diagnostics

METRICS = review.METRICS + review.EXTRA_METRICS + review.ORDER_METRICS


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_scores(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["position"] = int(row["position"])
        for metric in METRICS:
            if row[metric] not in {"0", "1"}:
                raise ValueError("Case score must be binary")
            row[metric] = int(row[metric])
    if [row["position"] for row in rows] != list(range(len(rows))) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Projected case IDs/order are incomplete or duplicated")
    return rows


def recount(rows):
    groups = defaultdict(list)
    for row in rows:
        group = {"trigger_report": "target", "control_report": "report_control", "clear": "clear_control"}[row["stratum"]]
        for name in ("all", group, "family:" + row["family"]):
            groups[name].append(row)
    return {name: {"n": len(values), "counts": {metric: sum(v[metric] for v in values) for metric in METRICS},
                   "rates": {metric: sum(v[metric] for v in values) / len(values) for metric in METRICS}}
            for name, values in groups.items()}


def verify(package):
    directory = package / "results/independent_review"
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    # Validate only the two owned subtrees, so this projection can be merged into
    # a full experiment entry with its own report, main analysis and figures.
    actual_files = {str(path.relative_to(package)) for subtree in (package / "code/independent_review", directory)
                    for path in subtree.rglob("*") if path.is_file()}
    if actual_files != set(manifest["artifacts"]) | {"results/independent_review/MANIFEST.json"}:
        raise ValueError("Public artifact inventory differs")
    for relative, record in manifest["artifacts"].items():
        path = package / relative
        if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Public artifact bytes differ: " + relative)
    stages = json.loads((directory / "stage_summaries.json").read_text())
    plan = json.loads((package / "code/independent_review/PLAN.json").read_text())
    records, illustrations, component_records = 0, 0, 0
    for stage in stages:
        rows = read_scores(directory / "cases" / (stage["stage"] + ".csv"))
        expected_n = 384 if stage["split"] == "validation" else 192
        if len(rows) != expected_n:
            raise ValueError("Wrong completed-stage denominator")
        groups = recount(rows)
        if groups != stage["groups"] or review.independent_gate(groups) != stage["descriptive_competence_gate"]:
            raise ValueError("Projected complete-case count/gate mismatch")
        by_id = {row["id"]: row for row in rows}
        fixed = {i for group in plan["fixed_longitudinal_ids_by_frozen_source_order"][stage["split"]].values() for i in group}
        errors = {row["id"] for stratum in ("trigger_report", "control_report", "clear")
                  for row in [r for r in rows if r["stratum"] == stratum and not r["full_correct"]][:2]}
        examples = [json.loads(line) for line in (directory / "illustrations" / (stage["stage"] + ".jsonl")).read_text().splitlines()]
        if len({row["id"] for row in examples}) != len(examples) or {row["id"] for row in examples} != fixed | errors:
            raise ValueError("Fixed/first-error illustration membership differs")
        for example in examples:
            source = {"policy": example["policy"], "ledger": example["ledger"]}
            parsed = review.independent_parse(example["raw_response"], source)
            row = by_id[example["id"]]
            expected_selection = (["fixed_longitudinal"] if example["id"] in fixed else []) + (["per_stage_first_two_errors"] if example["id"] in errors else [])
            if example["selection_rules"] != expected_selection or example["position"] != row["position"]:
                raise ValueError("Illustration selection rule/order differs")
            if parsed["expected"] != example["oracle"]:
                raise ValueError("Published ledger/policy oracle differs")
            for metric in review.METRICS + tuple(k for k in review.EXTRA_METRICS if k != "eos_finished"):
                if parsed[metric] != bool(row[metric]):
                    raise ValueError("Published raw illustration disagrees with complete-case score")
            if (example["finish_reason"] == "eos") != bool(row["eos_finished"]):
                raise ValueError("Finish summary differs")
        diagnostic_path = directory / "execution_diagnostics" / (stage["stage"] + ".json")
        if diagnostic_path.is_file():
            diagnostic = json.loads(diagnostic_path.read_text())
            component_cases = diagnostic["cases"]
            if [row["id"] for row in component_cases] != [row["id"] for row in rows]:
                raise ValueError("Component diagnostic case coverage differs")
            grouped = defaultdict(list)
            for row in component_cases:
                scored = by_id[row["id"]]
                if bool(scored["full_correct"]) != row["full_correct"]:
                    raise ValueError("Component full-correctness flag differs")
                if bool(scored["format_valid"]) != (row["selection_category"] != "not_schema_valid"):
                    raise ValueError("Component schema denominator differs")
                if scored["format_valid"] and bool(scored["selected_exact"]) != (row["selection_category"] == "exact_gold_order_and_membership"):
                    raise ValueError("Component exact-selection flag differs")
                for name in ("all", "stratum:" + row["stratum"], "family:" + row["family"]):
                    grouped[name].append(row)
            if {name: execution_diagnostics.summarize(items) for name, items in grouped.items()} != diagnostic["groups"]:
                raise ValueError("Component diagnostic aggregation differs")
            component_records += len(component_cases)
        records += len(rows)
        illustrations += len(examples)
    if len(stages) != manifest["completed_stage_count"] or records != manifest["case_checkpoint_records"]:
        raise ValueError("Manifest coverage mismatch")
    if manifest["publication_status"] == "final_independent_review" and len(stages) != 34:
        raise ValueError("Final factorial coverage is incomplete")
    return {"status": "verified_public_projection", "publication_status": manifest["publication_status"],
            "completed_stages": len(stages), "case_checkpoint_records_recounted": records,
            "published_illustrations_reexecuted": illustrations,
            "execution_component_records_recounted": component_records,
            "scope": "Recounts projected case/component scores and re-executes all published illustrations. Full-response main replay and original-workspace adapter-byte verification are distinct checks."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package.resolve()), indent=2))


if __name__ == "__main__":
    main()
