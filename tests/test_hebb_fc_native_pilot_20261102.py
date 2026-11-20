import unittest
import numpy as np
import torch
from analysis.hebb_fc_native_pilot_20261102 import RULES,apply_fc
from analysis.report_hebb_fc_native_pilot_20261102 import fast_single

class FCPilotTests(unittest.TestCase):
    def test_single_image_fast_update_matches_original_and_keeps_graph_fixed(self):
        torch.manual_seed(14);dim=23
        x=torch.relu(torch.randn(6,dim));x[0]=0
        e=torch.rand(dim,dim);e=(e+e.T)*.5;e.fill_diagonal_(0);e/=e.sum(1).max()
        model=dict(scale=torch.tensor(.71),matrices={'corr_k128':e,'corr_k32':e})
        before=e.clone()
        for method in RULES:
            for layer in ('fc1','fc2'):
                for sample in x:
                    z=sample[None,:]
                    expected=apply_fc(z,model,method,layer);actual=fast_single(z,model,method,layer)
                    self.assertTrue(torch.equal(actual,expected),(method,layer))
        self.assertTrue(torch.equal(e,before))

    def test_mean_std_threshold_differs_from_only_inhibition_when_graph_active(self):
        torch.manual_seed(4);dim=23;x=torch.rand(1,dim)
        e=torch.ones(dim,dim)-torch.eye(dim);e/=e.sum(1).max()
        model=dict(scale=torch.tensor(1.),matrices={'corr_k128':e,'corr_k32':e})
        active=fast_single(x,model,'mean_std','fc1');inhibited=fast_single(x,model,'inhibition_only','fc1')
        self.assertTrue(torch.isfinite(active).all())
        self.assertFalse(torch.equal(active,inhibited))

if __name__=='__main__':unittest.main()
