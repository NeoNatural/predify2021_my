import unittest

import torch

from predify_experiments.fc_serial_20261028 import bridge,mean_matched,SerialRunner,conditions,svm_confirmation
from predify_experiments.fc_finite_steps_20261028 import trajectories


class SerialFCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.x=torch.relu(torch.randn(20,16)-.3)
        self.w=torch.randn(16,16)*.2;self.b=torch.ones(16)*.2
        self.z=bridge(self.x,self.w,self.b)
        self.c1=dict(rule='corr',graph='k128',mode='hard',gain=.25,budget=1.)
        self.c2=dict(rule='corr',graph='k32',mode='hard',gain=.75,budget=1.)

    def runner(self,x=None,z=None):
        return SerialRunner(self.x if x is None else x,self.z if z is None else z,15,self.c1,self.c2,self.w,self.b)

    def test_bridge_is_original_affine_map_followed_by_relu(self):
        want=torch.relu(torch.nn.functional.linear(self.x,self.w,self.b))
        self.assertTrue(torch.equal(bridge(self.x,self.w,self.b),want))

    def test_fc1_change_really_propagates_to_fc2(self):
        runner=self.runner();stage=runner.stage('hard',5)
        self.assertFalse(torch.allclose(stage['fc1'],self.x))
        self.assertTrue(torch.equal(stage['fc2_input'],bridge(stage['fc1'],self.w,self.b)))
        self.assertFalse(torch.allclose(stage['fc2_input'],self.z))
        got=runner.output(dict(m1='hard',t1=5,m2='off',t2=0,calibration='none'))
        self.assertTrue(torch.equal(got,stage['fc2_input']))

    def test_fc2_only_replays_independent_processing(self):
        runner=self.runner()
        got=runner.output(dict(m1='off',t1=0,m2='hard',t2=5,calibration='frozen'))
        expected=list(trajectories(self.z,runner.model2,self.c2,steps=(5,)))[0][1]
        self.assertTrue(torch.equal(got,expected))

    def test_matched_second_model_never_uses_test_inputs(self):
        first=self.runner()
        x=self.x.clone();x[15:]*=20
        second=self.runner(x,bridge(x,self.w,self.b))
        a=first.side_model2(('hard',5),'matched')
        b=second.side_model2(('hard',5),'matched')
        self.assertTrue(torch.allclose(a['scale'],b['scale'],atol=1e-6))
        self.assertTrue(torch.allclose(a['matrices']['corr_k32'],b['matrices']['corr_k32'],atol=1e-6))

    def test_gain_controls_match_means_but_preserve_input_directions(self):
        target=self.x*.6
        r=mean_matched(self.x,target)
        self.assertTrue(torch.allclose(r.mean(1),target.mean(1),atol=1e-6))
        self.assertTrue(torch.allclose(r/r.norm(dim=1,keepdim=True),self.x/self.x.norm(dim=1,keepdim=True),atol=1e-6))

    def test_prespecified_factorial_and_confirmation_counts(self):
        rows=conditions()
        self.assertEqual(len(rows),57)
        self.assertEqual(sum(svm_confirmation(r['name']) for r in rows),11)
        hh=[r for r in rows if r['m1']=='hard' and r['m2']=='hard' and r['calibration']=='matched']
        self.assertEqual(len(hh),9)


if __name__=='__main__':unittest.main()
