import unittest
from unittest.mock import patch

import torch

from predify_experiments import fc_sparse_lateral_20261028 as sparse
from predify_experiments.fc_finite_steps_20261028 import trajectories,add_effects


class FiniteStepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(113)
        self.x=torch.relu(torch.randn(20,32)-.4)
        self.model=sparse.connection_model(self.x)

    def test_requested_horizons_are_not_stopped_by_residual_tolerance(self):
        c=dict(rule='corr',graph='k32',mode='soft',gain=.25,budget=1.)
        with patch.object(sparse.core,'TOL',1e9):
            outputs=list(trajectories(self.x,self.model,c,steps=(1,2,20)))
        self.assertEqual([r[0] for r in outputs],[1,2,20])
        self.assertEqual(outputs[-1][2]['updates_applied'],20)

    def test_one_and_two_steps_match_explicit_equations(self):
        c=dict(rule='corr',graph='dense',mode='divide',gain=.75,budget=1.)
        got={s:r for s,r,diag in trajectories(self.x,self.model,c,steps=(1,2))}
        x=self.x/self.model['scale'];r=x.clone();e=self.model['matrices']['corr_dense']
        for step in (1,2):
            target=torch.relu(x+.75*(r@e.T))/(1+2*r.mean(1,keepdim=True))
            r=.5*r+.5*target
            self.assertTrue(torch.equal(got[step],r*self.model['scale']))

    def test_hard_threshold_retained_at_all_horizons_and_obeys_budget(self):
        c=dict(rule='corr',graph='k32',mode='hard',gain=.75,budget=.75)
        outputs=list(trajectories(self.x,self.model,c,steps=(1,2,3,99,100)))
        self.assertEqual(len(outputs),5)
        cap=torch.ceil((self.x>1e-7).sum(1)*.75)
        for step,r,diag in outputs:
            self.assertTrue(torch.isfinite(r).all())
            self.assertTrue(((r>1e-7).sum(1)<=cap).all())

    def test_nonstationarity_does_not_disqualify_sharpening(self):
        base=dict(activation=2.,within=1.,between=1.,fisher=1.,accuracy=.8,auc=.9)
        m=dict(activation=1.,within=.8,between=.96,fisher=1.2,accuracy=.8,auc=.9,
            zero_fraction=0.,max_support_ratio=1.,residual=.2,converged=False)
        add_effects(m,base)
        self.assertTrue(m['joint']);self.assertTrue(m['sparse_joint'])

    def test_no_lateral_and_scale_controls_are_time_invariant(self):
        for mode,factor in (('identity',1.),('scale',.5)):
            c=dict(rule='zero',graph='zero',mode=mode,gain=0.,budget=1.)
            for step,r,diag in trajectories(self.x,self.model,c,steps=(1,5,100)):
                self.assertTrue(torch.equal(r,self.x*factor))


if __name__=='__main__':unittest.main()
