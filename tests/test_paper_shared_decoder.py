import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from predify_experiments.paper_shared_decoder import analyze_shared_decoder


LAYERS = ("conv4", "conv5", "fc1", "fc2")


def _result(indices, categories, *, carrier, identity, shift=0.0):
    categories = np.asarray(categories, dtype=np.int64)
    basis = np.eye(5, dtype=np.float32)[categories] * 5.0 + shift
    timeline = np.stack((basis, basis), axis=1)
    return {
        "paper_schema_version": 1,
        "run_metadata": {
            "degradation_mode": "predify_noise",
            "memory_carrier": carrier,
            "prime_identity": identity,
        },
        "dataset_indices": np.asarray(indices, dtype=np.int64),
        "category_labels": categories,
        "target_logits": np.zeros((len(categories), 2, 1000), dtype=np.float32),
        "target_representations": {layer: timeline.copy() for layer in LAYERS},
    }


def _dump(path: Path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle)


class SharedDecoderTests(unittest.TestCase):
    def test_fixed_decoder_uses_disjoint_training_images_and_writes_diagnostics(self):
        categories = np.repeat(np.arange(5), 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train"
            evaluation = root / "evaluation"
            output = root / "analysis"
            train.mkdir()
            evaluation.mkdir()
            _dump(
                train / "train.pckl",
                _result(range(100, 120), categories, carrier="none", identity="none"),
            )
            _dump(
                evaluation / "same.pckl",
                _result(range(20), categories, carrier="hebb_only", identity="same"),
            )
            _dump(
                evaluation / "different.pckl",
                _result(
                    range(20), categories, carrier="hebb_only", identity="different"
                ),
            )

            paths = analyze_shared_decoder(
                train,
                evaluation,
                output,
                seed=7,
                bootstrap_samples=100,
                max_iter=2_000,
            )

            self.assertTrue(all(path.exists() for path in paths.values()))
            diagnostics = paths["fit_diagnostics"].read_text(encoding="utf-8")
            self.assertIn("converged", diagnostics)
            self.assertEqual(diagnostics.count("\n"), 9)
            effects = paths["effects"].read_text(encoding="utf-8")
            self.assertEqual(effects.count("\n"), 9)
            self.assertIn("exact_mcnemar_two_sided_p", effects)
            self.assertIn(",0.0,0.0,0.0,False,0,0,1.0,1.0", effects)

    def test_training_and_evaluation_overlap_is_rejected(self):
        categories = np.arange(5)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train"
            evaluation = root / "evaluation"
            train.mkdir()
            evaluation.mkdir()
            value = _result(range(5), categories, carrier="none", identity="none")
            _dump(train / "train.pckl", value)
            _dump(evaluation / "eval.pckl", value)
            with self.assertRaisesRegex(ValueError, "image overlap"):
                analyze_shared_decoder(
                    train,
                    evaluation,
                    root / "output",
                    seed=7,
                    bootstrap_samples=10,
                )


if __name__ == "__main__":
    unittest.main()
