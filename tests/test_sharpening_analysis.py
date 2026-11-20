import unittest

import numpy as np

from predify_experiments.sharpening_analysis import (
    normalized_trace_var,
    summarize_pair,
)


class SharpeningAnalysisTests(unittest.TestCase):
    def test_legacy_normalized_trace_is_recovered_from_cosine(self):
        stats = {
            "count": np.array([5, 1]),
            "within_cosine": np.array([0.75, np.nan]),
        }
        actual = normalized_trace_var(stats)
        self.assertAlmostEqual(actual[0], 0.2)
        self.assertTrue(np.isnan(actual[1]))

    def test_pair_summary_uses_scale_invariant_variance_and_fisher(self):
        def result(activation, norm_trace, within, fisher):
            return {
                "repr_layer_names": ["layer"],
                "layer_metrics_arr": np.array(
                    [[[[0.0]]], [[[activation]]]], dtype=float
                ),
                "repr_stats_timeline": [
                    {
                        "layer": {
                            "count": np.array([5, 5]),
                            "within_cosine": np.array([within, within]),
                            "normalized_trace_var": np.array(
                                [norm_trace, norm_trace]
                            ),
                            "normalized_fisher_ratio": np.array(fisher),
                        }
                    }
                ],
            }

        prime = result(activation=8.0, norm_trace=0.1, within=0.9, fisher=3.0)
        control = result(activation=10.0, norm_trace=0.2, within=0.8, fisher=2.0)
        summary = summarize_pair(prime, control)["layer"]

        self.assertAlmostEqual(summary["activation_effect_pct"], -20.0)
        self.assertAlmostEqual(summary["normalized_trace_effect_pct"], -50.0)
        self.assertAlmostEqual(summary["within_cosine_delta_x1e3"], 100.0)
        self.assertAlmostEqual(summary["normalized_fisher_effect_pct"], 50.0)
        self.assertEqual(summary["complete_sharpening"], 1.0)


if __name__ == "__main__":
    unittest.main()
