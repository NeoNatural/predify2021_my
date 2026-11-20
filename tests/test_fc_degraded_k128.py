import unittest
import torch

from predify_experiments.fc_degraded_inputs_20261028 import seed_for,corrupt
from predify_experiments.fc_degraded_k128_20261028 import FrozenRunner,conditions
from predify_experiments.fc_gain075_serial_20261028 import parameters
from predify_experiments.fc_serial_20261028 import SerialRunner,fit_model,bridge


class DegradedFCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(242)
        self.x=torch.relu(torch.randn(30,160)-.4)
        self.w=torch.randn(160,160)*.04;self.b=torch.ones(160)*.1
        self.z=bridge(self.x,self.w,self.b)
        self.c1,self.c2=parameters('k128')
        self.m1=fit_model(self.x[:20],self.c1);self.m2=fit_model(self.z[:20],self.c2)

    def test_clean_frozen_runner_matches_previous_serial_computation(self):
        old=SerialRunner(self.x,self.z,20,self.c1,self.c2,self.w,self.b)
        new=FrozenRunner(self.x,self.z,self.m1,self.m2,self.w,self.b)
        for c in conditions():
            self.assertTrue(torch.equal(new.output(c),old.output(c)))

    def test_degraded_runner_cannot_refit_statistics(self):
        noisy=torch.relu(self.x+torch.randn_like(self.x)*.5)
        r=FrozenRunner(noisy,bridge(noisy,self.w,self.b),self.m1,self.m2,self.w,self.b)
        before=self.m1['matrices']['corr_k128'].clone();scale=self.m2['scale'].clone()
        r.output(next(c for c in conditions() if c['name']=='both_hard_hard_5_5_frozen'))
        self.assertTrue(torch.equal(before,r.model1['matrices']['corr_k128']))
        self.assertTrue(torch.equal(scale,r.model2['scale']))
        with self.assertRaises(AssertionError):r.side_model2(('hard',5),'matched')

    def test_gain_control_matches_degraded_not_clean_candidate(self):
        noisy=torch.relu(self.x+torch.randn_like(self.x)*.8)
        r=FrozenRunner(noisy,bridge(noisy,self.w,self.b),self.m1,self.m2,self.w,self.b)
        self.assertTrue(torch.allclose(r.stage('gain_hard',5)['fc1'].mean(1),r.stage('hard',5)['fc1'].mean(1),atol=1e-6))
        clean=FrozenRunner(self.x,self.z,self.m1,self.m2,self.w,self.b)
        self.assertFalse(torch.allclose(r.stage('gain_hard',5)['fc1'].mean(1),clean.stage('hard',5)['fc1'].mean(1)))

    def test_randomization_is_reproducible_per_image_not_shared_pattern(self):
        image=torch.zeros(3,64,64)
        a,_=corrupt(image,'things',1,0,'predify_noise',.5)
        b,_=corrupt(image,'things',2,0,'predify_noise',.5)
        c,_=corrupt(image,'things',1,0,'predify_noise',.5)
        self.assertTrue(torch.equal(a,c));self.assertFalse(torch.equal(a,b))
        self.assertNotEqual(seed_for('things',1,0,'predify_noise'),seed_for('things',1,1,'predify_noise'))
        self.assertAlmostEqual(float(a.std()),.5,delta=.02)

    def test_occlusion_opacity_replays_legacy_pixels(self):
        image=torch.randn(3,64,64)
        altered,diag=corrupt(image,'imagenet',5,1,'fractal_occlusion',.3)
        self.assertLess(float(diag[-1]),1e-6)
        self.assertGreaterEqual(diag[4],0);self.assertLessEqual(diag[4],1)
        self.assertFalse(torch.equal(altered,image))


if __name__=='__main__':unittest.main()
