import unittest

import numpy as np

from predify_experiments.paper_analysis import (
    _aligned_reference_indices,
    cross_validated_linear_svm,
    identity_grouped_folds,
    paired_interval,
    paired_intervals,
)


class PaperAnalysisTests(unittest.TestCase):
    def _separable_data(self):
        rng = np.random.default_rng(4)
        features = []
        categories = []
        identities = []
        for category in range(5):
            for identity_offset in range(5):
                identity = category * 10 + identity_offset
                for _ in range(2):
                    value = np.zeros(8, dtype=np.float32)
                    value[category] = 5.0
                    value += rng.normal(0, 0.05, size=8)
                    features.append(value)
                    categories.append(category)
                    identities.append(identity)
        return np.asarray(features), np.asarray(categories), np.asarray(identities)

    def test_grouped_folds_do_not_leak_object_identity(self):
        features, categories, identities = self._separable_data()
        del features
        folds = identity_grouped_folds(categories, identities, folds=5, seed=3)
        for train, test in folds:
            self.assertTrue(set(identities[train]).isdisjoint(identities[test]))

    def test_linear_svm_decodes_separable_categories(self):
        features, categories, identities = self._separable_data()
        predictions, correctness, accuracy = cross_validated_linear_svm(
            features,
            categories,
            identities,
            folds=5,
            seed=3,
        )
        np.testing.assert_array_equal(predictions, categories)
        self.assertTrue(correctness.all())
        self.assertEqual(accuracy, 1.0)

    def test_paired_interval_is_deterministic(self):
        values = np.full(20, 0.25)
        self.assertEqual(
            paired_interval(values, samples=100, seed=9, scale=100.0),
            (25.0, 25.0, 25.0),
        )

    def test_feedforward_reference_is_aligned_by_dataset_index(self):
        result = {"dataset_indices": np.asarray([30, 10])}
        reference = {"dataset_indices": np.asarray([10, 20, 30])}
        np.testing.assert_array_equal(
            _aligned_reference_indices(result, reference),
            np.asarray([2, 0]),
        )

    def test_batched_bootstrap_matches_direct_resampling(self):
        values = np.arange(12, dtype=float).reshape(6, 2)
        means, lows, highs = paired_intervals(
            values, samples=200, seed=17, scale=2.0
        )
        rng = np.random.default_rng(17)
        draws = rng.integers(0, len(values), size=(200, len(values)))
        distribution = values[draws].mean(axis=1) * 2.0
        expected_lows, expected_highs = np.quantile(
            distribution, (0.025, 0.975), axis=0
        )
        np.testing.assert_allclose(means, values.mean(axis=0) * 2.0)
        np.testing.assert_allclose(lows, expected_lows)
        np.testing.assert_allclose(highs, expected_highs)

    def test_missing_feedforward_reference_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing dataset index 99"):
            _aligned_reference_indices(
                {"dataset_indices": np.asarray([99])},
                {"dataset_indices": np.asarray([10])},
            )


if __name__ == "__main__":
    unittest.main()
