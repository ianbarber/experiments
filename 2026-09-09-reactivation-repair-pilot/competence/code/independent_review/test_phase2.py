"""Synthetic CPU checks of conditional audit arithmetic and fail-closed bounds."""
import copy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import phase2
import review
from test_review import case


def groups():
    return {"all": {"n": 192, "counts": {"format_valid": 189, "internally_consistent": 183}},
            "target": {"n": 64, "counts": {"false_clear": 20}},
            "report_control": {"n": 64, "counts": {"decision_correct": 58, "full_correct": 52}},
            "clear_control": {"n": 64, "counts": {"decision_correct": 58, "full_correct": 52}}}


class ConditionalReviewTests(unittest.TestCase):
    def test_complete_synthetic_cohort_audit_in_both_orders(self):
        program, freeze, _ = review.frozen_program(review.ROOT)
        pool = review.read_rows(review.ROOT / "data/pool.jsonl")
        for order in ("decision_first", "decision_last"):
            with self.subTest(order=order), TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "data").mkdir()
                for name in ("pool.jsonl", "preservation.jsonl"):
                    shutil.copy2(review.ROOT / "data" / name, root / "data" / name)
                collection = root / "synthetic_collection"
                collection.mkdir()
                outputs, cases = [], []
                keys = (("decision", "selected", "count", "reason") if order == "decision_first"
                        else ("selected", "count", "reason", "decision"))
                for row in pool:
                    answer = phase2.alternatives(row)[row["designated_operator"]]
                    text = json.dumps({key: answer[key] for key in keys}, separators=(",", ":"))
                    generated = {"text": text, "finish_reason": "eos"}
                    outputs.append({"id": row["id"], "source_row_sha256": review.row_sha(row),
                                    "rendered_row_sha256": review.row_sha(program.serialized_case(row, order)),
                                    "generated": generated, "parsed": program.parse_output(text, row)})
                    parsed = review.independent_parse(text, row)
                    parsed["eos_finished"] = True
                    cases.append({"id": row["id"], "source": row, "generated": generated,
                                  "independent": parsed, "classification": phase2.classify(row, parsed)})
                (collection / "outputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in outputs))
                program.construct(root / "data/pool.jsonl", collection / "outputs.jsonl", root / "data/preservation.jsonl",
                                  root / "data/cohort_1729", 1729, output_order=order)
                checker = review.contract_checker(root, program, freeze)
                actual = phase2.audit_cohort(root, 1729, collection, cases, checker, program, order)
                self.assertEqual(actual["pool_denominator"], 1536)
                self.assertEqual(actual["eligible_supported_finished_failures"], 1536)
                self.assertEqual(actual["selected_failures"], 192)
                self.assertEqual(actual["closed_two_cycles"], 96)
                self.assertEqual(actual["identical_preservation_examples"], 320)
                self.assertTrue(actual["complete_artifacts_and_deterministic_reconstruction_match"])
                mapping_path = root / "data/cohort_1729/matched_cases.jsonl"
                mapping = review.read_rows(mapping_path)
                mapping[0]["donor_trace"] = mapping[0]["trace"]
                mapping_path.write_text("".join(json.dumps(row) + "\n" for row in mapping))
                completion_path = mapping_path.with_name("COMPLETED.json")
                completion = json.loads(completion_path.read_text())
                completion["artifacts_sha256"][mapping_path.name] = review.sha(mapping_path)
                completion_path.write_text(json.dumps(completion))
                with self.assertRaisesRegex(ValueError, "deterministic reconstruction"):
                    phase2.audit_cohort(root, 1729, collection, cases, checker, program, order)

    def test_integer_induction_boundaries_and_per_control_preservation(self):
        competent = groups()
        competent["report_control"]["counts"]["full_correct"] = 55
        competent["clear_control"]["counts"]["full_correct"] = 55
        actual = groups()
        self.assertTrue(phase2.induction_gate(actual, competent)["passed"])
        for count, passed in ((19, False), (20, True), (44, True), (45, False)):
            actual["target"]["counts"]["false_clear"] = count
            self.assertEqual(phase2.induction_gate(actual, competent)["checks"]["target_false_clear"], passed)
        for name in ("report_control", "clear_control"):
            altered = copy.deepcopy(actual)
            altered[name]["counts"]["full_correct"] -= 1
            self.assertFalse(phase2.induction_gate(altered, competent)["checks"][name + "_full_preserved"])
        for metric in ("format_valid", "internally_consistent"):
            altered = copy.deepcopy(actual)
            altered["all"]["counts"][metric] -= 1
            self.assertFalse(phase2.induction_gate(altered, competent)["passed"])

    def test_actual_supported_failure_requires_declared_program_and_execution(self):
        row = case()
        wrong = phase2.alternatives(row)["recent_only"]
        parsed = review.independent_parse(json.dumps(wrong), row)
        self.assertEqual(phase2.classify(row, parsed)["operator"], "recent_only")
        self.assertTrue(phase2.classify(row, parsed)["eligible_failure"])
        inconsistent = copy.deepcopy(wrong)
        inconsistent["count"] = 1
        self.assertFalse(phase2.classify(row, review.independent_parse(json.dumps(inconsistent), row))["eligible_failure"])
        right = review.independent_parse("{}", row)["expected"]
        right["decision"] = "CLEAR"
        self.assertFalse(phase2.classify(row, review.independent_parse(json.dumps(right), row))["eligible_failure"])

    def test_canonical_trace_identity_ignores_key_order_but_not_program(self):
        row = case()
        wrong = phase2.alternatives(row)["recent_only"]
        reordered = {k: wrong[k] for k in reversed(wrong)}
        self.assertEqual(phase2.canonical(wrong), phase2.canonical(reordered))
        changed = copy.deepcopy(wrong)
        changed["reason"]["unit"] = "events"
        self.assertNotEqual(phase2.canonical(wrong), phase2.canonical(changed))

    def test_donor_key_catches_count_and_policy_mismatch(self):
        row = case()
        key = phase2.independent_donor_key(row, "recent_only")
        self.assertEqual(key, ["current__approved_only__issues", 2, "recent_only", 8, "REPORT", "CLEAR", 2, 0])
        changed = copy.deepcopy(row)
        changed["policy"]["threshold"] = 3
        self.assertNotEqual(key, phase2.independent_donor_key(changed, "recent_only"))

    def test_paired_identity_direction_and_exact_shared_draws(self):
        rng = np.random.default_rng(8)
        first = rng.integers(0, 2, size=(3, 12))
        identity = phase2.paired_effect(first, first, draws=37, seed=23)
        self.assertEqual(identity["observed_seed_mean"], {"difference": 0., "ci95": [0., 0.]})
        other = rng.integers(0, 2, size=(3, 12))
        actual = phase2.paired_effect(first, other, draws=37, seed=23)
        delta = (first - other).mean(axis=0)
        indices = np.random.default_rng(23).integers(0, 12, size=(37, 12))
        np.testing.assert_array_equal(actual["observed_seed_mean"]["ci95"], np.quantile(delta[indices].mean(axis=1), [.025, .975], method="linear"))
        reversed_result = phase2.paired_effect(other, first, draws=37, seed=23)
        np.testing.assert_allclose(reversed_result["observed_seed_mean"]["ci95"], -np.asarray(actual["observed_seed_mean"]["ci95"])[::-1])
        self.assertEqual(reversed_result["observed_seed_mean"]["difference"], -actual["observed_seed_mean"]["difference"])

    def test_repair_score_shape_binary_and_final_checkpoint_guard(self):
        for bad in (np.zeros((2, 12)), np.zeros((3, 0)), np.full((3, 12), .5), np.full((3, 12), np.nan)):
            with self.assertRaises(ValueError):
                phase2.paired_effect(bad, bad)
        with self.assertRaisesRegex(ValueError, "exact 30 repairs and 4 controls"):
            phase2.repair_comparisons({}, [])

    def test_missing_cohort_cannot_be_created_by_independent_audit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            class Forbidden:
                def verified_cohort(self, *args):
                    raise AssertionError("Independent audit must not create a real cohort")
            with self.assertRaisesRegex(ValueError, "do not construct"):
                phase2.audit_cohort(root, 1729, root / "collection", [], object(), Forbidden(), "decision_first")

    def test_partial_generic_stage_does_not_read_model_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = {"output": "partial", "args": {"data": "data/pool.jsonl"}}
            with patch.object(Path, "read_text", side_effect=AssertionError("Partial output access")):
                with self.assertRaisesRegex(ValueError, "partial outputs were not read"):
                    phase2.generic_review(root, stage, object(), {}, object(), {})


if __name__ == "__main__":
    unittest.main()
