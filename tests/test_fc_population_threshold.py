import unittest
import numpy as np
import torch
from predify_experiments import fc_population_threshold_20261028 as exp
from predify_experiments.fc_serial_20261028 import bridge,fit_model


class PopulationSerialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(54)
        self.x=torch.relu(torch.randn(30,160)-.4)
        self.w=torch.randn(160,160)*.04;self.b=torch.ones(160)*.1
        self.z=bridge(self.x,self.w,self.b)
        c1,c2=exp.strong.parameters('k128')
        self.m1=fit_model(self.x[:20],c1);self.m2=fit_model(self.z[:20],c2)

    def runner(self,family):
        c=dict(family=family,a=.25,b=.1,theta0=0.) if family=='moments' else dict(family=family,g=.25,rho=.5,theta0=0.)
        return exp.PopulationRunner(self.x,self.z,self.m1,self.m2,self.w,self.b,dict(fc1=c,fc2=c))

    def test_quota_replays_original_whole_cascade(self):
        old=exp.old.FrozenRunner(self.x,self.z,self.m1,self.m2,self.w,self.b)
        new=exp.PopulationRunner(self.x,self.z,self.m1,self.m2,self.w,self.b,dict(fc1=dict(family='quota'),fc2=dict(family='quota')))
        for c in exp.conditions():self.assertTrue(torch.equal(old.output(c),new.output(c)))

    def test_controls_match_actual_family_output_and_bridge(self):
        for family in ('moments','feedback'):
            r=self.runner(family)
            target=r.output(dict(m1='hard',t1=5,m2='hard',t2=5,calibration='frozen'))
            for c in exp.conditions():
                stage=r.stage(c['m1'],c['t1']);out=r.output(c)
                if c['m1']=='gain_hard':self.assertTrue(torch.allclose(stage['fc1'].mean(1),r.stage('hard',5)['fc1'].mean(1),atol=1e-6))
                if c['m2']=='gain_hard':self.assertTrue(torch.allclose(out.mean(1),target.mean(1),atol=1e-6))
                if c['m1']!='off':self.assertTrue(torch.equal(stage['fc2_input'],bridge(stage['fc1'],self.w,self.b)))

    def test_supplied_models_stay_fixed_and_matched_refit_rejected(self):
        E=self.m1['matrices']['corr_k128'].clone();s=self.m2['scale'].clone()
        for family in ('moments','feedback'):
            r=self.runner(family)
            for c in exp.conditions():r.output(c)
            with self.assertRaises(AssertionError):r.side_model2(('hard',5),'matched')
        self.assertTrue(torch.equal(E,self.m1['matrices']['corr_k128']))
        self.assertTrue(torch.equal(s,self.m2['scale']))

    def test_auxiliary_scope_contains_controls_but_not_sensitivity(self):
        keys=exp.auxiliary_keys()
        self.assertEqual(len(keys),10)
        for c in exp.conditions():
            f=exp.strong.feature_keys('k128',c)
            for stage in ('fc1_output','fc2_output'):
                self.assertEqual(f[stage] in keys,c['t1'] in (0,5))


if __name__=='__main__':unittest.main()
