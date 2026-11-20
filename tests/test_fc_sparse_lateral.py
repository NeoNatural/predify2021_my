import unittest

import numpy as np
import torch

from predify_experiments import fc_lateral_study_20261028 as core
from predify_experiments.fc_sparse_lateral_20261028 import (
    mutual_topk, connection_model, dynamics, adaptive_target,
)


class SparseLateralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.x=torch.relu(torch.randn(30,48)-.5)

    def test_mutual_graph_is_symmetric_and_degree_bounded(self):
        a=torch.rand(48,48);a=(a+a.T)*.5;a.fill_diagonal_(0)
        e,stats=mutual_topk(a,7)
        self.assertTrue(torch.equal(e,e.T))
        self.assertTrue((e>=0).all())
        self.assertEqual(float(e.diagonal().max()),0.)
        self.assertLessEqual(stats['degree_max'],7)
        self.assertLessEqual(float(e.sum(1).max()),1.000001)

    def test_threshold_targets_obey_variable_per_image_budgets(self):
        drive=torch.tensor([[8.,7.,6.,5.],[9.,3.,2.,1.],[4.,3.,2.,1.]])
        count=torch.tensor([2,1,0])
        for mode in ('soft','hard'):
            target,theta=adaptive_target(drive,count,mode)
            self.assertTrue(torch.equal((target>0).sum(1),count))
            self.assertTrue((target>=0).all())
            if mode=='hard': self.assertEqual(float(target[0,0]),8.)
            else: self.assertLess(float(target[0,0]),8.)

    def test_no_leaky_residuals_outside_output_budget(self):
        model=connection_model(self.x)
        for mode in ('soft','hard'):
            c=dict(rule='corr',graph='k32',mode=mode,gain=.75,budget=.75)
            r,diag=dynamics(self.x,model,c)
            budget=torch.ceil((self.x>1e-7).sum(1)*.75).long()
            self.assertTrue(((r>1e-7).sum(1)<=budget).all())

    def test_only_hard_threshold_full_budget_preserves_direction(self):
        model=connection_model(self.x)
        c=dict(rule='zero',graph='zero',mode='hard',gain=0.,budget=1.)
        r,diag=dynamics(self.x,model,c)
        self.assertTrue(diag['converged'])
        u=r/r.norm(dim=1,keepdim=True)
        v=self.x/self.x.norm(dim=1,keepdim=True)
        self.assertTrue(torch.allclose(u,v,atol=1e-6))

    def test_dense_reference_replays_original_dynamics(self):
        model=connection_model(self.x)
        c=dict(rule='corr',graph='dense',mode='divide',gain=.75,budget=1.)
        a,_=dynamics(self.x,model,c)
        old=core.connection_model(self.x)
        b,_=core.dynamics(self.x,old,dict(rule='corr',inhibition='divide',gain=.75,strength=2.))
        self.assertTrue(torch.equal(a,b))


if __name__=='__main__':unittest.main()
