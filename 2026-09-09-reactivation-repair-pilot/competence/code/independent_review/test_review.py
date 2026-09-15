"""CPU fixtures for the independent checker; no real outcome or model access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review


def case():
    ledger = [dict(id=f"R{i}", run="production", status="pass", period="current",
                   recency="early", waiver="none", issue="D") for i in range(1, 9)]
    ledger[0].update(status="fail", waiver="approved", issue="A")
    ledger[1].update(status="fail", issue="A")
    ledger[2].update(status="fail", issue="A")
    ledger[3].update(status="fail", issue="B")
    return {"ledger": ledger, "policy": {"window": "current", "waiver": "approved_only",
                                        "unit": "issues", "threshold": 2}}


class IndependentReviewTests(unittest.TestCase):
    def test_independent_filter_then_first_eligible_issue(self):
        row = case()
        program = {"window": "current", "waiver": "approved_only", "unit": "issues", "recency": "all"}
        value = review.independent_execute(row, program)
        self.assertEqual(value["selected"], ["R2", "R4"])
        self.assertEqual((value["count"], value["decision"]), (2, "REPORT"))

    def test_both_key_orders_have_identical_complete_correctness(self):
        row = case()
        expected = review.independent_parse("{}", row)["expected"]
        first = json.dumps(expected)
        last = json.dumps({k: expected[k] for k in ("selected", "count", "reason", "decision")})
        self.assertTrue(review.independent_parse(first, row)["full_correct"])
        self.assertTrue(review.independent_parse(last, row)["full_correct"])
        self.assertEqual(review.independent_parse(first, row)["expected"], review.independent_parse(last, row)["expected"])

    def test_schema_error_does_not_hide_a_parsed_false_clear(self):
        row = case()
        value = review.independent_parse("{}", row)["expected"]
        value.update(decision="CLEAR", extra="invalid schema")
        parsed = review.independent_parse(json.dumps(value), row)
        self.assertTrue(parsed["strict_json"])
        self.assertFalse(parsed["format_valid"])
        self.assertTrue(parsed["false_clear"])
        self.assertFalse(parsed["full_correct"])

    def test_duplicate_keys_nonfinite_and_boolean_counts_fail_strictly(self):
        row = case()
        for text in ('{"decision":"REPORT","decision":"CLEAR"}', '{"count":NaN}'):
            self.assertFalse(review.independent_parse(text, row)["strict_json"])
        value = review.independent_parse("{}", row)["expected"]
        value["count"] = True
        self.assertFalse(review.independent_parse(json.dumps(value), row)["format_valid"])

    def test_exact_integer_gates_for_192_and_384(self):
        for n, schema, full, control, false_clear in ((192, 189, 164, 58, 6), (384, 377, 327, 116, 12)):
            groups = {"all": {"n": n, "counts": {"format_valid": schema, "full_correct": full}},
                      "target": {"n": n // 3, "counts": {"false_clear": false_clear}},
                      "report_control": {"n": n // 3, "counts": {"decision_correct": control}},
                      "clear_control": {"n": n // 3, "counts": {"decision_correct": control}}}
            self.assertTrue(review.independent_gate(groups)["passed"])
            groups["target"]["counts"]["false_clear"] += 1
            self.assertFalse(review.independent_gate(groups)["checks"]["target_false_clear"])
            groups["target"]["counts"]["false_clear"] -= 1
            groups["clear_control"]["counts"]["decision_correct"] -= 1
            self.assertFalse(review.independent_gate(groups)["checks"]["clear_control_decision"])

    def test_partial_stage_is_rejected_before_any_output_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = {"output": "results/development/partial", "args": {"data": "data/calibration.jsonl"}}
            (root / stage["output"]).mkdir(parents=True)
            with patch.object(Path, "read_text", side_effect=AssertionError("Do not read partial data")):
                with self.assertRaisesRegex(ValueError, "partial outputs were not read"):
                    review.ensure_complete(root, stage)

    def test_validation_requires_a_selection_record_before_output_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = {"output": "results/validation/fixture", "args": {"data": "data/validation.jsonl"}}
            (root / stage["output"]).mkdir(parents=True)
            (root / stage["output"] / "COMPLETED.json").write_text("{}\n")
            with patch.object(Path, "read_text", side_effect=AssertionError("Do not read unselected validation")):
                with self.assertRaisesRegex(ValueError, "preceding frozen selection"):
                    review.ensure_complete(root, stage)


if __name__ == "__main__":
    unittest.main()
