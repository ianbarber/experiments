"""Finite synthetic parity against the separately frozen saved-data analyzer.

Only CPU array routines are called. No actual model-output file is opened.
"""
import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np

import factorial
import phase2
import review


class SeparateAnalysisParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = review.ROOT / "results/analysis_tools/analyze.py"
        if review.sha(path) != "ddd6ac505e7809aec5b29468da73428e0c3ecd0667d9a158dd3a4873bafcf7f4":
            raise ValueError("Separately frozen analyzer changed")
        spec = importlib.util.spec_from_file_location("separately_frozen_cpu_analyzer", path)
        cls.main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.main)

    def test_all_factorial_effects_and_paired_draws_match_on_random_binary_fixture(self):
        rows = [{"id": f"fixture-{i}", "stratum": factorial.STRATA[i % 3]} for i in range(384)]
        rng = np.random.default_rng(99)
        outputs = {}
        for recipe in review.RECIPE_NAMES:
            for seed in phase2.INSTALLATIONS:
                outputs[f"validation_{recipe}_i{seed}"] = [dict(row, decision_correct=bool(rng.integers(2)), full_correct=bool(rng.integers(2))) for row in rows]
        official = self.main.factorial_effects(outputs, rows, draws=211, seed=45)
        full, clear = factorial.shared_draws([r["stratum"] for r in rows], draws=211, seed=45)
        self.assertEqual(official["shared_draws_sha256"], hashlib.sha256(full.astype("<i8").tobytes()).hexdigest())
        for endpoint, metric, indices, subset in (
                ("clear_decision_accuracy", "decision_correct", clear, np.arange(2, 384, 3)),
                ("complete_correctness", "full_correct", full, np.arange(384))):
            values = np.asarray([[[r[metric] for r in outputs[f"validation_{recipe}_i{seed}"]]
                                  for recipe in review.RECIPE_NAMES] for seed in phase2.INSTALLATIONS])
            independent = factorial.summarize_effects(values, indices, subset)
            for item in (e for e in official["effects"] if e["endpoint"] == endpoint):
                actual = independent[item["contrast"]]["values"]
                for seed in phase2.INSTALLATIONS:
                    self.assertAlmostEqual(actual[str(seed)]["difference"], item["seed_effects"][str(seed)], places=15)
                expected = item["observed_seed_mean"]
                self.assertAlmostEqual(actual["observed_seed_mean"]["difference"], expected["effect"], places=15)
                np.testing.assert_allclose(actual["observed_seed_mean"]["ci95"], [expected["ci_low"], expected["ci_high"]], atol=1e-15, rtol=0)

    def test_complete_34_checkpoint_repair_fixture_matches_main_contrasts(self):
        rows = [{"id": f"fixture-{suite}-{stratum}-{i}", "suite": suite, "stratum": stratum}
                for suite, n in (("id", 64), ("heldout", 64), ("narrative", 128))
                for stratum in factorial.STRATA for i in range(n)]
        names = {f"{arm}_i{i}_s{s}" for arm in phase2.ARMS for i in phase2.INSTALLATIONS for s in phase2.SEEDS} | {f"{label}_{i}" for i in phase2.INSTALLATIONS for label in ("competent", "bad")}
        rng = np.random.default_rng(131)
        values = {name: [{**row, **dict(zip(review.METRICS, map(bool, rng.integers(0, 2, len(review.METRICS)))))} for row in rows] for name in sorted(names)}
        independent_rows = {name: [{"id": row["id"], "independent": {metric: row[metric] for metric in review.METRICS}} for row in data] for name, data in values.items()}
        actual = phase2.repair_comparisons(independent_rows, rows)
        expected = self.main.evaluation_contrasts(values, rows)
        self.assertEqual(actual["primary_ids"], expected["primary_ids"])
        self.assertEqual(len(actual["primary_cell_counts"]), 34)
        self.assertEqual(len(actual["control_preservation"]), 2 * 3 * 3 * 5 * 6)
        indexed = {(r["installation"], r["comparator"]): r for r in actual["contrasts"]}
        for item in expected["contrasts"]:
            got = indexed[item["installation"], item["comparator"]]
            self.assertEqual(got["per_seed"], item["seed_effects"])
            self.assertEqual(got["role"] == "primary", item["primary"])
            a, b = got["observed_seed_mean"], item["observed_seed_mean"]
            self.assertEqual(a["difference"], b["effect"])
            np.testing.assert_array_equal(a["ci95"], [b["ci_low"], b["ci_high"]])
        tampered = dict(independent_rows)
        tampered.pop(next(iter(tampered)))
        with self.assertRaises(ValueError):
            phase2.repair_comparisons(tampered, rows)
        tampered = dict(independent_rows)
        name = next(iter(tampered))
        tampered[name] = tampered[name][::-1]
        with self.assertRaisesRegex(ValueError, "pairing"):
            phase2.repair_comparisons(tampered, rows)


if __name__ == "__main__":
    unittest.main()
