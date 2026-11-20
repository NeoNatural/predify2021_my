import unittest

import numpy as np
import torch

from predify_experiments.fc_lateral_study_20261028 import (
    connection_model, dynamics, geometry, inner_folds, matched_controls, ridge_scores,
)


class LateralMechanismTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.rand(50, 16)+.1
        self.y = np.repeat(np.arange(5), 10)
        self.groups = np.array([f"c{c}_g{j//2}" for c in range(5) for j in range(10)])

    def test_connections_have_no_self_loops_and_preserve_symmetry(self):
        model = connection_model(self.x)
        for e in model["matrices"].values():
            self.assertTrue(torch.allclose(e, e.T, atol=1e-6))
            self.assertTrue((e >= 0).all())
            self.assertEqual(float(e.diagonal().abs().max()), 0.)
            self.assertLessEqual(float(e.sum(1).max()), 1.000001)

    def test_global_subtraction_has_expected_constant_input_equilibrium(self):
        x = torch.full((10, 8), 2.)
        model = connection_model(x)
        config = dict(rule="zero", inhibition="subtract", gain=0., strength=2.)
        r, diag = dynamics(x, model, config)
        self.assertTrue(diag["converged"])
        self.assertTrue(torch.allclose(r, x/3, atol=5e-5))

    def test_local_compensation_cancels_training_mean_drive(self):
        x = self.x.mean(0, keepdim=True).repeat(10, 1)
        model = connection_model(x)
        config = dict(rule="raw", inhibition="local", gain=.75, strength=0.)
        r, diag = dynamics(x, model, config)
        self.assertTrue(diag["converged"])
        self.assertTrue(torch.allclose(r, x, atol=1e-6))

    def test_scale_changes_activity_but_not_angular_geometry_or_standardized_readout(self):
        y = torch.as_tensor(self.y)
        base, scaled = geometry(self.x, y), geometry(self.x*.5, y)
        self.assertAlmostEqual(base["activation"]*.5, scaled["activation"], places=6)
        self.assertAlmostEqual(base["within"], scaled["within"], places=6)
        self.assertAlmostEqual(base["fisher"], scaled["fisher"], places=6)
        a = ridge_scores(self.x, self.x[:10], self.y, self.groups)
        b = ridge_scores(self.x*.5, self.x[:10]*.5, self.y, self.groups)
        self.assertTrue(np.allclose(a, b, atol=2e-5))

    def test_inner_split_keeps_concepts_and_classes_intact(self):
        fold = inner_folds(self.y, self.groups, 123)
        for f in range(2):
            self.assertFalse(set(self.groups[fold == f]) & set(self.groups[fold != f]))
            self.assertEqual(set(self.y[fold == f]), set(range(5)))

    def test_matched_scale_is_per_image_and_preserves_directions(self):
        target = torch.relu(self.x-.4)
        scaled, threshold, mismatch = matched_controls(self.x, target)
        self.assertTrue(torch.allclose(scaled.mean(1), target.mean(1), atol=1e-6))
        u = scaled/scaled.norm(dim=1, keepdim=True)
        v = self.x/self.x.norm(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(u, v, atol=1e-6))
        self.assertEqual(mismatch, 0.)
        self.assertTrue(torch.equal((threshold > 1e-7).sum(1), (target > 1e-7).sum(1)))


if __name__ == "__main__": unittest.main()
