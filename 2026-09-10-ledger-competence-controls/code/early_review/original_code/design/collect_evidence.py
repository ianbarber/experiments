"""Bounded CPU evidence for the in-flight scientific-identification audit.

Reads frozen sources/data and only completed model-stage artifacts. The donor
witness uses authored oracle alternatives, explicitly not sampled model output.
No model, tokenizer, GPU, service, or experiment-controller imports/calls.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def main():
    started = datetime.now(timezone.utc).isoformat()
    freeze_path = ROOT / "results/FROZEN_PROGRAM.json"
    freeze = json.loads(freeze_path.read_text())
    snapshot = ROOT / freeze["source_snapshot"]
    assert json.loads((snapshot / "manifest.json").read_text()) == freeze
    for relative, digest in freeze["pins"].items():
        assert sha(ROOT / relative) == digest, relative
        if not relative.startswith("data/"):
            assert sha(snapshot / relative) == digest, relative
    sys.path.insert(0, str(snapshot / "scripts"))
    import task
    import serialization
    sys.path.insert(0, str(ROOT / "results/independent_review"))
    import review
    assert not any(name in sys.modules for name in ("torch", "transformers", "peft"))

    datasets = {name: read(ROOT / "data" / (name + ".jsonl")) for name in
                ("competence", "calibration", "validation", "induction", "pool", "preservation", "evaluation")}
    seen = set()
    prior = json.loads((ROOT / "data/prior_semantic_exclusions.json").read_text())
    previous = {signature for item in prior["files"].values() for signature in item["semantic_hashes"]}
    summaries = {}
    for name, rows in datasets.items():
        signatures = [task.semantic_signature(row) for row in rows]
        assert len(set(signatures)) == len(rows)
        assert not (set(signatures) & seen) and not (set(signatures) & previous)
        seen.update(signatures)
        summaries[name] = {"rows": len(rows), "strata": dict(Counter(r["stratum"] for r in rows)),
                           "families": dict(Counter(r["family"] for r in rows)),
                           "gold_decisions": dict(Counter(task.gold(r)["decision"] for r in rows)),
                           "supervised_decisions": dict(Counter(json.loads(r["target"])["decision"] for r in rows)),
                           "REPORT_cases_with_supported_CLEAR_shortcut": sum(task.gold(r)["decision"] == "REPORT" and bool(task.available_operators(r)) for r in rows),
                           "zero_count_CLEAR": sum(task.gold(r)["decision"] == "CLEAR" and task.gold(r)["count"] == 0 for r in rows)}
    # A deterministic structural witness, selected only from frozen source data.
    groups = defaultdict(list)
    for row in datasets["pool"]:
        for operator in task.available_operators(row):
            wrong = task.bad_options(row)[operator]
            groups[task.donor_key(row, operator)].append((row, wrong))
    witness = None
    for key in sorted(groups):
        first_row, first_wrong = groups[key][0]
        for second_row, second_wrong in groups[key][1:]:
            if task.canonical_json(first_wrong) != task.canonical_json(second_wrong):
                witness = (key, first_row, first_wrong, second_row, second_wrong)
                break
        if witness:
            break
    assert witness is not None
    key, a, aw, b, bw = witness
    assert a["id"] != b["id"] and task.donor_key(a, key[2]) == task.donor_key(b, key[2])
    assert aw["reason"] == bw["reason"] and aw["count"] == bw["count"] and aw["decision"] == bw["decision"] == "CLEAR"
    assert aw["selected"] != bw["selected"] and task.parse_output(task.canonical_json(aw), a)["eligible_failure"]
    assert task.parse_output(task.canonical_json(bw), b)["eligible_failure"]
    donor_witness = {"scope": "Authored structural counterexample using frozen source facts and executable alternatives; not an actual model failure or an accepted future cohort.",
                     "donor_key": list(key), "current": {"case": a, "wrong_audit": aw, "gold_audit": task.gold(a)},
                     "donor": {"case": b, "wrong_audit": bw, "gold_audit": task.gold(b)},
                     "trace_fields_equal": [field for field in ("decision", "selected", "count", "reason") if aw[field] == bw[field]],
                     "trace_fields_different": [field for field in ("decision", "selected", "count", "reason") if aw[field] != bw[field]],
                     "own_trace_executes_own_facts": True, "donor_trace_executes_donor_facts": True,
                     "same_wrong_mechanism_by_construction": True,
                     "interpretation": "Whole-instance archive alignment differs. Self-authorship, failure mechanism, scalar result, and declared rule do not differ within the donor pair."}
    # Metadata poisoning demonstrates that truth labels are not prompt fields.
    source = datasets["competence"][0]
    altered = copy.deepcopy(source)
    for field in ("gold", "gold_target", "correct_action", "oracle_count", "oracle_selected", "semantic_hash", "target", "split", "stratum"):
        altered[field] = "AUDIT_METADATA_SENTINEL"
    assert serialization.render_case(source, "decision_first") == serialization.render_case(altered, "decision_first")
    first = serialization.render_case(source, "decision_first")
    last = serialization.render_case(source, "decision_last")
    assert last == first.replace(serialization.FIRST_CONTRACT_FRAGMENT, serialization.LAST_CONTRACT_FRAGMENT, 1)
    assert json.loads(serialization.answer_text(source, "decision_first")) == json.loads(serialization.answer_text(source, "decision_last"))

    calibration_ids = [row["id"] for row in datasets["calibration"]]
    current = []
    for directory in sorted((ROOT / "results/development").glob("*")):
        completed = directory / "COMPLETED.json"
        if not completed.is_file():
            continue
        contract = json.loads(completed.read_text())
        assert contract["status"] == "complete"
        assert contract["data_sha256"] == freeze["pins"]["data/calibration.jsonl"]
        assert contract["source_sha256"] == {key: value for key, value in freeze["pins"].items() if key.startswith("scripts/") and key.endswith(".py")}
        assert contract["args"]["score_decision"] is False and contract["args"]["sample"] is False
        output = directory / "outputs.jsonl"
        assert sha(output) == contract["artifacts_sha256"]["outputs.jsonl"]
        rows = read(output)
        assert [row["id"] for row in rows] == calibration_ids
        order = ("decision", "selected", "count", "reason") if contract["args"]["output_order"] == "decision_first" else ("selected", "count", "reason", "decision")
        counts = Counter()
        stratum_counts = defaultdict(Counter)
        for observed, row in zip(rows, datasets["calibration"]):
            assert observed["source_row_sha256"] == review.row_sha(row)
            parsed = review.independent_parse(observed["generated"]["text"], row)
            for metric in review.METRICS[:4]:
                assert parsed[metric] == observed["parsed"][metric]
                counts[metric] += int(parsed[metric])
                stratum_counts[row["stratum"]][metric] += int(parsed[metric])
            stratum_counts[row["stratum"]]["n"] += 1
            counts["eos_finished"] += observed["generated"]["finish_reason"] == "eos"
            value = parsed["parsed"]
            counts["requested_order_followed"] += isinstance(value, dict) and tuple(value) == order
        current.append({"stage": directory.name, "finished_utc": contract["finished_at"],
                        "completion_sha256": sha(completed), "outputs_sha256": sha(output),
                        "requested_order": contract["args"]["output_order"], "n": len(rows), "counts": dict(counts),
                        "strata": {name: dict(c) for name, c in stratum_counts.items()}})
    # Read only completed training metadata to test actual paired exposure.
    chains = defaultdict(list)
    for directory in sorted((ROOT / "checkpoints").glob("competence_*")):
        completed = directory / "COMPLETED.json"
        if not completed.is_file():
            continue
        c = json.loads(completed.read_text())
        assert c["status"] == "complete" and c["steps"] == 96 and len(c["sample_order"]) == 1536
        assert len(set(c["sample_order"])) == 1536
        seed, epoch = directory.name.split("_i", 1)[1].split("_epoch")
        chains[seed + ":" + epoch].append({"name": directory.name, "class_weighting": c["args"]["class_weighting"],
                                           "output_order": c["args"]["output_order"], "target_tokens": c["target_tokens"],
                                           "prefix_tokens": c["prefix_tokens"], "processed_class_budget": c["processed_class_budget"],
                                           "sample_order_sha256": hashlib.sha256(canonical(c["sample_order"]).encode()).hexdigest(),
                                           "completion_sha256": sha(completed)})
    assert all(len({entry["sample_order_sha256"] for entry in cells}) == 1 for cells in chains.values())
    events = []
    for line in (ROOT / "results/program.jsonl").read_text().splitlines(keepends=True):
        if line.endswith("\n"):
            event = json.loads(line)
            events.append({key: event[key] for key in ("at", "event", "name", "recipe", "installation", "epoch") if key in event})
    state = {"selection_record_present": (ROOT / "results/SELECTED_RECIPE.json").exists(),
             "validation_result_present": (ROOT / "results/VALIDATION_RESULT.json").exists(),
             "validation_started_directories": sorted(p.name for p in (ROOT / "results/validation").glob("*") if p.is_dir()),
             "induction_directories": sorted(p.name for p in (ROOT / "checkpoints").glob("induction_*")),
             "repair_evaluation_directories": sorted(p.name for p in (ROOT / "results/evaluation").glob("*") if p.is_dir()),
             "last_events": events[-6:]}
    generation = ROOT.parent / "reactivation/models/Qwen2.5-3B-Instruct/generation_config.json"
    assert sha(generation) == freeze["base_model_hashes"]["generation_config.json"]
    result = {"review_started_utc": started, "evidence_captured_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "In-flight scientific identification review; observed timing explicitly after completed development calibrations, before any validation or conditional outcomes if state fields confirm that condition.",
              "frozen_program_sha256": sha(freeze_path), "source_snapshot": freeze["source_snapshot"],
              "verified_live_and_archived_pins": freeze["pins"],
              "original_research_plan_sha256": sha(ROOT.parent / "reactivation/RESEARCH_PLAN.md"),
              "current_readme_sha256": sha(ROOT / "README.md"),
              "data": summaries, "unique_new_semantic_cases": len(seen), "previous_semantic_cases_excluded": len(previous),
              "metadata_poison_render_unchanged": True, "order_changes_only_contract_and_answer_key_order_in_witness": True,
              "actual_completed_calibrations": current, "actual_completed_training_exposure": dict(chains),
              "same_observed_seed_epoch_sample_order_across_recipes": True, "state": state,
              "base_generation_config_sha256": sha(generation), "base_generation_config": json.loads(generation.read_text()),
              "base_generation_config_note": "Saved base defaults, not an independently probed effective model GenerationConfig. Explicit common.py overrides and installed-library resolution must be considered.",
              "donor_witness": donor_witness, "evidence_script_sha256": sha(Path(__file__)),
              "model_tokenizer_gpu_or_service_calls": False}
    for relative, digest in freeze["pins"].items():
        assert sha(ROOT / relative) == digest
    with (DEST / "EVIDENCE.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"new_cases": len(seen), "completed_calibrations": len(current),
                      "state": state, "donor_witness_ids": [a["id"], b["id"]], "different_trace_fields": donor_witness["trace_fields_different"]}, indent=2))


if __name__ == "__main__":
    main()
