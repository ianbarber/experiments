"""Finite synthetic CPU checks for independent factorial arithmetic."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import factorial


class FactorialReviewTests(unittest.TestCase):
    def test_final_requires_all_audits_before_any_stage_output_read(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(factorial.review, "frozen_program", return_value=(object(), {}, {})), \
                 patch.object(factorial.review, "scoped_stages", return_value=({"missing_stage": {}}, {})), \
                 patch.object(factorial.review, "contract_checker", side_effect=AssertionError("premature output access")):
                with self.assertRaisesRegex(ValueError, "every completed scoped audit"):
                    factorial.compute(root)

    def test_shared_draws_preserve_strata_and_exact_seed_traversal(self):
        strata = np.asarray(["clear", "trigger_report", "control_report"] * 4)
        full, clear = factorial.shared_draws(strata, draws=73, seed=1234)
        rng = np.random.default_rng(1234)
        expected = []
        for name in factorial.STRATA:
            indices = np.flatnonzero(strata == name)
            expected.append(indices[rng.integers(0, len(indices), size=(73, len(indices)))])
        np.testing.assert_array_equal(full, np.concatenate(expected, axis=1))
        np.testing.assert_array_equal(clear, expected[-1])
        for row in full:
            self.assertEqual({name: int((strata[row] == name).sum()) for name in factorial.STRATA},
                             dict.fromkeys(factorial.STRATA, 4))

    def test_identical_cells_have_zero_main_effects_and_interaction(self):
        rng = np.random.default_rng(2)
        cells = np.repeat(rng.integers(0, 2, size=(2, 1, 12)), 4, axis=1)
        indices = rng.integers(0, 12, size=(31, 12))
        result = factorial.summarize_effects(cells, indices, np.arange(12))
        for effect in result.values():
            for seed in effect["values"].values():
                self.assertEqual(seed, {"difference": 0., "ci95": [0., 0.]})

    def test_known_factorial_main_effect_and_interaction_orientation(self):
        cells = np.zeros((2, 4, 12))
        cells[:, 3, :] = 1
        actual = factorial.effect_cases(cells)
        np.testing.assert_array_equal(actual[:, 0], .5)
        np.testing.assert_array_equal(actual[:, 1], .5)
        np.testing.assert_array_equal(actual[:, 2], 1.)
        cells[:, 1, :] = 1
        actual = factorial.effect_cases(cells)
        np.testing.assert_array_equal(actual[:, 0], 1.)
        np.testing.assert_array_equal(actual[:, 1:], 0.)

    def test_point_estimate_does_not_depend_on_which_cases_a_draw_omits(self):
        cells = np.zeros((2, 4, 4))
        cells[:, 1, 3] = 1
        cells[:, 3, 3] = 1
        # Deliberately omit the only improved case from the synthetic draw.
        result = factorial.summarize_effects(cells, np.zeros((1, 4), dtype=int), np.arange(4))
        value = result["reweighting"]["values"]["observed_seed_mean"]
        self.assertEqual(value["difference"], .25)
        self.assertEqual(value["ci95"], [0., 0.])

    def test_case_scores_require_two_seeds_four_recipes_and_binary_values(self):
        for cells in (np.zeros((1, 4, 12)), np.zeros((2, 3, 12)), np.full((2, 4, 12), .5),
                      np.full((2, 4, 12), np.nan)):
            with self.assertRaises(ValueError):
                factorial.effect_cases(cells)


if __name__ == "__main__":
    unittest.main()
