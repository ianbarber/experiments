"""Descriptive component audit of already independently verified completed stages.

Requested after the first trained calibration to distinguish selected-list
membership errors from ordering errors. This does not change any score or gate.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import review


def components(case, source):
    scored = case["independent"]
    value, gold = scored["parsed"], scored["expected"]
    if not scored["format_valid"]:
        return {"id": case["id"], "stratum": case["stratum"], "family": case["family"],
                "selection_category": "not_schema_valid", "full_correct": scored["full_correct"]}
    selected, correct = value["selected"], gold["selected"]
    if selected == correct:
        category = "exact_gold_order_and_membership"
    elif set(selected) == set(correct):
        category = "same_gold_membership_wrong_order"
    else:
        category = "wrong_membership"
    threshold = source["policy"]["threshold"]
    implied = "REPORT" if value["count"] >= threshold else "CLEAR"
    return {"id": case["id"], "stratum": case["stratum"], "family": case["family"],
            "selection_category": category, "full_correct": scored["full_correct"],
            "reason_fields_gold_correct": scored["reason_correct"],
            "count_equals_selected_length": value["count"] == len(selected),
            "count_equals_gold": value["count"] == gold["count"],
            "count_minus_gold": value["count"] - gold["count"],
            "decision_equals_count_threshold_result": value["decision"] == implied,
            "threshold_from_case": threshold,
            "missing_gold_ids": [rid for rid in correct if rid not in selected],
            "extra_selected_ids": [rid for rid in selected if rid not in correct]}


def summarize(rows):
    valid = [row for row in rows if row["selection_category"] != "not_schema_valid"]
    wrong = [row for row in valid if row["selection_category"] == "wrong_membership"]
    return {"n": len(rows), "schema_valid_n": len(valid),
            "selection_categories": dict(Counter(row["selection_category"] for row in rows)),
            "full_correct": sum(row["full_correct"] for row in rows),
            "count_equals_selected_length": sum(row["count_equals_selected_length"] for row in valid),
            "count_equals_gold": sum(row["count_equals_gold"] for row in valid),
            "count_minus_gold_distribution": dict(sorted(Counter(str(row["count_minus_gold"]) for row in valid).items())),
            "decision_inconsistent_with_own_count_and_case_threshold": sum(not row["decision_equals_count_threshold_result"] for row in valid),
            "wrong_membership_same_gold_count": sum(row["count_equals_gold"] for row in wrong),
            "wrong_membership_different_gold_count": sum(not row["count_equals_gold"] for row in wrong),
            "exact_selection_count_but_wrong_decision": sum(row["selection_category"] == "exact_gold_order_and_membership" and row["count_equals_gold"] and not row["decision_equals_count_threshold_result"] for row in valid)}


def run(root, stage):
    directory = root / "results/independent_review"
    report_path, cases_path = directory / (stage + ".json"), directory / (stage + ".cases.jsonl")
    report = json.loads(report_path.read_text())
    actual = root / report["stage_spec"]["output"]
    if not (actual / "COMPLETED.json").is_file():
        raise ValueError("No complete source stage")
    hashes = {"completion": review.sha(actual / "COMPLETED.json"), "outputs": review.sha(actual / "outputs.jsonl"),
              "cases": review.sha(cases_path), "audit": review.sha(report_path)}
    if hashes["completion"] != report["completion_sha256"] or hashes["outputs"] != report["outputs_sha256"] or hashes["cases"] != report["per_case_review_sha256"]:
        raise ValueError("Completed independent audit identity changed")
    source_path = root / report["stage_spec"]["args"]["data"]
    if review.sha(source_path) != report["source_data_sha256"]:
        raise ValueError("Source data changed")
    sources = review.read_rows(source_path)
    cases = review.read_rows(cases_path)
    if [r["id"] for r in sources] != [r["id"] for r in cases]:
        raise ValueError("Source/case pairing differs")
    diagnostics = []
    for case, source in zip(cases, sources):
        if review.row_sha(source) != case["source_row_sha256"]:
            raise ValueError("Source case identity changed")
        diagnostics.append(components(case, source))
    grouped = defaultdict(list)
    for row in diagnostics:
        for group in ("all", "stratum:" + row["stratum"], "family:" + row["family"]):
            grouped[group].append(row)
    record = {"stage": stage, "scope": "Descriptive component clarification requested after first trained calibration; original exact ledger-order metric and every gate remain unchanged.",
              "denominator_scope": "Selection categories partition all cases; count and threshold diagnostics use schema-valid cases only. The threshold is supplied by the case, not emitted as a fifth reason field.",
              "source_hashes": hashes, "source_data_sha256": review.sha(source_path),
              "groups": {name: summarize(rows) for name, rows in grouped.items()}, "cases": diagnostics}
    output = directory / "execution_diagnostics" / (stage + ".json")
    if output.exists():
        saved = json.loads(output.read_text())
        for key in ("created_utc", "script_sha256"):
            saved.pop(key)
        if saved != record:
            raise ValueError("Existing component audit differs")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        record.update(created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=review.sha(Path(__file__)))
        with output.open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=review.ROOT)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    result = run(args.root.resolve(), args.stage)
    print(json.dumps({"stage": args.stage, "status": "verified", "all": result["groups"]["all"]}))


if __name__ == "__main__":
    main()
