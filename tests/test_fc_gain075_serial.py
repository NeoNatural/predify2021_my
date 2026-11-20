import unittest

import torch

from predify_experiments.fc_gain075_serial_20261028 import parameters, conditions, feature_keys
from predify_experiments.fc_serial_20261028 import SerialRunner, bridge


class StrongFC1Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(481)
        self.x = torch.relu(torch.randn(30, 160) - .4)
        self.w = torch.randn(160, 160) * .04
        self.b = torch.ones(160) * .1
        self.z = bridge(self.x, self.w, self.b)

    def runner(self, graph, x=None):
        x = self.x if x is None else x
        return SerialRunner(x, bridge(x, self.w, self.b), 20, *parameters(graph), self.w, self.b)

    def test_both_graphs_actually_change_fc1_with_fixed_fc2(self):
        a, b = self.runner('k32'), self.runner('k128')
        self.assertEqual(a.c1['gain'], .75)
        self.assertEqual(a.c2, b.c2)
        self.assertFalse(torch.allclose(a.stage('hard', 5)['fc1'], b.stage('hard', 5)['fc1']))
        for r in (a, b):
            s = r.stage('hard', 5)
            self.assertTrue(torch.equal(s['fc2_input'], bridge(s['fc1'], self.w, self.b)))

    def test_controls_match_current_strong_fc1_target_and_final_mean(self):
        for graph in ('k32', 'k128'):
            r = self.runner(graph)
            hard = r.stage('hard', 5)['fc1']
            gain = r.stage('gain_hard', 5)['fc1']
            self.assertTrue(torch.allclose(hard.mean(1), gain.mean(1), atol=1e-6))
            self.assertTrue(torch.allclose(gain / gain.norm(dim=1, keepdim=True), self.x / self.x.norm(dim=1, keepdim=True), atol=1e-6))
            for cal in ('frozen', 'matched'):
                target = r.output(dict(m1='hard', t1=5, m2='hard', t2=5, calibration=cal))
                for m1 in ('hard', 'gain_hard'):
                    ctrl = r.output(dict(m1=m1, t1=5, m2='gain_hard', t2=5, calibration=cal))
                    self.assertTrue(torch.allclose(target.mean(1), ctrl.mean(1), atol=1e-6))

    def test_feature_reuse_cannot_mix_the_two_graphs(self):
        for c in conditions():
            a, b = feature_keys('k32', c), feature_keys('k128', c)
            if c['m1'] in ('hard', 'gain_hard'):
                self.assertNotEqual(a['fc1_output'], b['fc1_output'])
                self.assertNotEqual(a['fc2_input'], b['fc2_input'])
            else:
                self.assertEqual(a['fc1_output'], b['fc1_output'])
            if c['m2'] == 'off':
                self.assertEqual(a['fc2_output'], a['fc2_input'])

    def test_matched_training_matrices_ignore_test_perturbations(self):
        changed = self.x.clone(); changed[20:] *= 10
        for graph in ('k32', 'k128'):
            a, b = self.runner(graph), self.runner(graph, changed)
            key = 'corr_' + graph
            self.assertTrue(torch.equal(a.model1['matrices'][key], b.model1['matrices'][key]))
            ma = a.side_model2(('hard', 5), 'matched')
            mb = b.side_model2(('hard', 5), 'matched')
            self.assertTrue(torch.equal(ma['scale'], mb['scale']))
            self.assertTrue(torch.equal(ma['matrices']['corr_k32'], mb['matrices']['corr_k32']))


if __name__ == '__main__':
    unittest.main()
