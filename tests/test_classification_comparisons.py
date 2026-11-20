import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from predify_experiments.classification_comparisons import (
    _index_results,
    paired_bootstrap,
    predify_correctness,
    timeline_rows,
)
from predify_experiments.vanilla_vgg_benchmark import (
    topk_correct,
    vanilla_result_filename,
)


class ClassificationComparisonTests(unittest.TestCase):
    def test_vanilla_topk_and_filename(self):
        logits = np.array([[0.1, 0.7, 0.2], [0.9, 0.08, 0.02]])
        labels = np.array([1, 2])
        np.testing.assert_array_equal(topk_correct(logits, labels, 1), [True, False])
        np.testing.assert_array_equal(topk_correct(logits, labels, 3), [True, True])
        self.assertEqual(vanilla_result_filename("none", 0), "Vanilla_VGG16_clean.pckl")
        self.assertIn("sigma0p5", vanilla_result_filename("predify_noise", 0.5))

    def test_paired_bootstrap_is_deterministic_and_paired(self):
        values = np.ones(20) * 0.25
        first = paired_bootstrap(values, samples=100, seed=7)
        second = paired_bootstrap(values, samples=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first, (25.0, 25.0, 25.0))

    def test_schema_v2_and_v3_common_timeline_loader(self):
        base = {
            "Out_list": np.array(
                [
                    [[[0.0, 1.0]], [[0.0, 1.0]]],
                    [[[0.0, 1.0]], [[1.0, 0.0]]],
                ]
            ).reshape(2, 2, 2),
            "y_list": [-1, 1],
            "idx_log_list": [9, 7],
        }
        for schema in (2, 3):
            with self.subTest(schema=schema):
                result = dict(base, schema_version=schema)
                correct, labels, indices = predify_correctness(result, 0, 1)
                np.testing.assert_array_equal(correct, [True])
                np.testing.assert_array_equal(labels, [1])
                np.testing.assert_array_equal(indices, [7])

    def test_timeline_accuracy_and_metadata_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vanilla_path = root / "vanilla.pckl"
            predify_path = root / "result_Prime.pckl"
            vanilla = {
                "run_metadata": {"degradation_mode": "none", "degradation_level": 0.0},
                "logits": np.array([[1.0, 0.0], [0.0, 1.0]]),
                "y_list": np.array([0, 0]),
                "idx_log_list": np.array([4, 6]),
            }
            predify = {
                "schema_version": 3,
                "run_metadata": {
                    "condition": "priming",
                    "degradation_mode": "none",
                    "degradation_level": 0.0,
                },
                "Out_list": np.array(
                    [
                        [[0.0, 1.0], [0.0, 1.0]],
                        [[1.0, 0.0], [1.0, 0.0]],
                        [[0.0, 1.0], [0.0, 1.0]],
                        [[1.0, 0.0], [1.0, 0.0]],
                    ]
                ),
                "y_list": [-1, 0, -1, 0],
                "idx_log_list": [4, 4, 6, 6],
            }
            with vanilla_path.open("wb") as handle:
                pickle.dump(vanilla, handle)
            with predify_path.open("wb") as handle:
                pickle.dump(predify, handle)
            vanilla_index = _index_results([root])
            # Remove the Predify entry accidentally indexed by the broad scan.
            vanilla_index = {("none", 0.0): vanilla_path}
            predify_index = _index_results([root], condition="priming")
            rows = timeline_rows(
                vanilla_index, predify_index, bootstrap_samples=100, seed=3
            )
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[0]["vanilla_top1"], 0.5)
            self.assertAlmostEqual(rows[0]["predify_top1"], 1.0)
            self.assertAlmostEqual(rows[0]["delta_top1_pp"], 50.0)

    def test_discovery_prefers_sigma_zero_clean_and_highest_nonzero_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            newer = root / "newer"
            newer.mkdir()

            def save(path, schema, mode, level):
                result = {
                    "schema_version": schema,
                    "run_metadata": {
                        "condition": "priming",
                        "degradation_mode": mode,
                        "degradation_level": level,
                    },
                }
                with path.open("wb") as handle:
                    pickle.dump(result, handle)

            generic_clean = root / "generic_Prime.pckl"
            sigma_clean = root / "predify_noise_sigma0_Prime.pckl"
            old_noise = root / "predify_noise_sigma0p5_Prime.pckl"
            new_noise = newer / "predify_noise_sigma0p5_Prime.pckl"
            save(generic_clean, 2, "none", 0.0)
            save(sigma_clean, 2, "predify_noise", 0.0)
            save(old_noise, 2, "predify_noise", 0.5)
            save(new_noise, 3, "predify_noise", 0.5)

            index = _index_results(
                [root, newer],
                condition="priming",
                prefer_highest_schema=True,
            )
            self.assertEqual(index[("none", 0.0)], sigma_clean.resolve())
            self.assertEqual(index[("predify_noise", 0.5)], new_noise.resolve())


if __name__ == "__main__":
    unittest.main()
