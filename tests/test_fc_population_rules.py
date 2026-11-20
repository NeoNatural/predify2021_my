import unittest
from unittest.mock import patch
import torch
from predify_experiments.fc_population_rules_20261028 import trajectories
from predify_experiments.fc_finite_steps_20261028 import trajectories as previous


class PopulationRuleTests(unittest.TestCase):
    def model(self,d=16,dtype=torch.float32):
        E=(torch.ones(d,d,dtype=dtype)-torch.eye(d,dtype=dtype))/(d-1)
        return dict(scale=torch.tensor(1.,dtype=dtype),matrices={'corr_k128':E,'corr_k32':E})

    def test_quota_exactly_replays_previous_rule(self):
        torch.manual_seed(3);x=torch.relu(torch.randn(8,16)-.3);m=self.model()
        want=list(previous(x,m,dict(rule='corr',graph='k128',mode='hard',gain=.75,budget=1.),(1,5,20)))
        got=list(trajectories(x,m,dict(family='quota'),steps=(1,5,20)))
        for a,b in zip(want,got):self.assertTrue(torch.equal(a[1],b[1]))

    def test_new_rules_do_not_call_sorting_gate(self):
        x=torch.tensor([[2.,0.,0.,0.]])
        with patch('predify_experiments.fc_sparse_lateral_20261028.adaptive_target',side_effect=AssertionError('ranking called')):
            for c in [dict(family='moments',a=.5,b=.1,theta0=0.),dict(family='feedback',g=.5,rho=.5,theta0=0.)]:
                result=list(trajectories(x,self.model(4),c,steps=(5,)))[0][1]
                self.assertTrue(torch.isfinite(result).all())

    def test_activity_is_allowed_to_grow_without_quota(self):
        x=torch.zeros(1,16);x[0,0]=1
        for c in [dict(family='moments',a=.01,b=0.,theta0=0.),dict(family='feedback',g=.01,rho=.5,theta0=0.)]:
            r=list(trajectories(x,self.model(),c,steps=(1,)))[0][1]
            self.assertGreater(int((r>1e-7).sum()),int((x>1e-7).sum()))

    def test_moment_threshold_ignores_nonzero_count(self):
        x=torch.tensor([[0.,0.,1.,1.]],dtype=torch.float64)
        z=torch.tensor([[.5-(1/12)**.5]*3+[.5+(3/4)**.5]],dtype=torch.float64)
        self.assertNotEqual(int((x>0).sum()),int((z>0).sum()))
        c=dict(family='moments',a=.5,b=.25,theta0=0.)
        a=list(trajectories(x,self.model(4,torch.float64),c,gain=0,steps=(1,)))[0][2]['threshold_mean']
        b=list(trajectories(z,self.model(4,torch.float64),c,gain=0,steps=(1,)))[0][2]['threshold_mean']
        self.assertAlmostEqual(a,b,places=12)

    def test_feedback_state_is_reset_per_image_and_call(self):
        torch.manual_seed(5);x=torch.relu(torch.randn(7,16));m=self.model()
        c=dict(family='feedback',g=.5,rho=.25,theta0=0.)
        a=list(trajectories(x,m,c,steps=(5,)))[0][1]
        b=torch.cat([list(trajectories(v[None],m,c,steps=(5,)))[0][1] for v in x])
        self.assertTrue(torch.allclose(a,b,atol=1e-6))
        self.assertTrue(torch.equal(a,list(trajectories(x,m,c,steps=(5,)))[0][1]))

    def test_feedback_time_constant_affects_later_steps(self):
        x=torch.tensor([[3.,2.,1.,0.]])
        m=dict(scale=torch.tensor(1.),matrices={'corr_k128':torch.eye(4)*.1})
        a=list(trajectories(x,m,dict(family='feedback',g=1.,rho=.25,theta0=0.),steps=(1,2,5)))
        b=list(trajectories(x,m,dict(family='feedback',g=1.,rho=1.,theta0=0.),steps=(1,2,5)))
        self.assertTrue(torch.equal(a[0][1],b[0][1]))
        self.assertTrue(torch.equal(a[1][1],b[1][1]))
        self.assertFalse(torch.allclose(a[2][1],b[2][1]))


if __name__=='__main__':unittest.main()
