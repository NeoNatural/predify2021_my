import unittest

from predify_experiments.recovery_calibration import (
    PCODER5_CANDIDATES,
    candidate_hyperparameters,
)


class RecoveryCalibrationTests(unittest.TestCase):
    def test_candidates_only_change_pcoder5(self):
        baseline_prefix = candidate_hyperparameters("baseline_ff0p8_er0p01")[:4]
        for name in PCODER5_CANDIDATES:
            with self.subTest(name=name):
                values = candidate_hyperparameters(name)
                self.assertEqual(len(values), 5)
                self.assertEqual(values[:4], baseline_prefix)
                self.assertEqual(values[4]["fbm"], 0.0)
                self.assertLessEqual(values[4]["ffm"] + values[4]["fbm"], 1.0)

    def test_unknown_candidate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown calibration candidate"):
            candidate_hyperparameters("missing")


if __name__ == "__main__":
    unittest.main()
