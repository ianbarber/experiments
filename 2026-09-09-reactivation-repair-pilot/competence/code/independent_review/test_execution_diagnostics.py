import copy
import json
import unittest

import execution_diagnostics as ed
import review
from test_review import case


class ExecutionDiagnosticTests(unittest.TestCase):
    def test_distinguishes_membership_order_and_threshold_errors(self):
        source = case()
        gold = review.independent_parse("{}", source)["expected"]
        variants = []
        for action in ("gold", "order", "membership", "decision"):
            value = copy.deepcopy(gold)
            if action == "order":
                value["selected"].reverse()
            elif action == "membership":
                value["selected"][0] = "R3"
            elif action == "decision":
                value["decision"] = "CLEAR"
            scored = review.independent_parse(json.dumps(value), source)
            item = {"id": action, "stratum": "trigger_report", "family": "fixture", "independent": scored}
            variants.append(ed.components(item, source))
        self.assertEqual([r["selection_category"] for r in variants], ["exact_gold_order_and_membership", "same_gold_membership_wrong_order", "wrong_membership", "exact_gold_order_and_membership"])
        actual = ed.summarize(variants)
        self.assertEqual(actual["full_correct"], 1)
        self.assertEqual(actual["count_equals_gold"], 4)
        self.assertEqual(actual["wrong_membership_same_gold_count"], 1)
        self.assertEqual(actual["wrong_membership_different_gold_count"], 0)
        self.assertEqual(actual["decision_inconsistent_with_own_count_and_case_threshold"], 1)
        self.assertEqual(actual["exact_selection_count_but_wrong_decision"], 1)

    def test_schema_invalid_has_separate_denominator(self):
        source = case()
        item = {"id": "invalid", "stratum": "trigger_report", "family": "fixture", "independent": review.independent_parse("{}", source)}
        actual = ed.summarize([ed.components(item, source)])
        self.assertEqual(actual["n"], 1)
        self.assertEqual(actual["schema_valid_n"], 0)
        self.assertEqual(actual["selection_categories"], {"not_schema_valid": 1})


if __name__ == "__main__":
    unittest.main()
