from pathlib import Path
import sys
import unittest


try:
    import torch

    LEGACY_DIR = Path(__file__).resolve().parents[1] / "Hebbian-CNN"
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    from OnlineSharpeningStats_Demo import OnlineSharpeningStats

    TORCH_IMPORT_ERROR = None
except Exception as error:
    torch = None
    OnlineSharpeningStats = None
    TORCH_IMPORT_ERROR = error


@unittest.skipIf(torch is None, f"PyTorch environment unavailable: {TORCH_IMPORT_ERROR}")
class OnlineSharpeningStatsTests(unittest.TestCase):
    def _finalize(self, features):
        stats = OnlineSharpeningStats(
            layer_names=["layer"],
            num_classes=2,
            device=torch.device("cpu"),
        )
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        stats.update({"layer": features}, labels)
        return stats.finalize()["layer"]

    def test_normalized_geometry_is_invariant_to_uniform_activation_scaling(self):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        )
        original = self._finalize(features)
        scaled = self._finalize(features * 3.5)

        torch.testing.assert_close(
            original["normalized_trace_var"],
            scaled["normalized_trace_var"],
            rtol=0.02,
            atol=2e-4,
        )
        torch.testing.assert_close(
            original["normalized_fisher_ratio"],
            scaled["normalized_fisher_ratio"],
            rtol=0.02,
            atol=2e-4,
        )
        self.assertTrue(
            torch.all(scaled["trace_var"] > original["trace_var"] * 5.0)
        )

    def test_normalized_trace_matches_pairwise_cosine_identity(self):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        )
        result = self._finalize(features)
        expected = (2.0 / 3.0) * (1.0 - result["within_cosine"])
        torch.testing.assert_close(
            result["normalized_trace_var"],
            expected,
            rtol=0.02,
            atol=2e-4,
        )

    def test_normalized_fisher_ratio_detects_class_separation(self):
        separated = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.2, 0.8],
            ]
        )
        collapsed = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
            ]
        )
        separated_result = self._finalize(separated)
        collapsed_result = self._finalize(collapsed)

        self.assertGreater(
            separated_result["normalized_fisher_ratio"].item(),
            collapsed_result["normalized_fisher_ratio"].item(),
        )


if __name__ == "__main__":
    unittest.main()
