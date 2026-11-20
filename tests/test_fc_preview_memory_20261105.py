import unittest
import torch
from predify_experiments.fc_preview_memory_20261105 import PreviewMemory,TrialEngine,side,bridge,DEFAULTS,METHODS


class PreviewMemoryTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(511)

    def test_explicit_outer_updates_and_zero_diagonal(self):
        p=torch.relu(torch.randn(2,17,dtype=torch.float64));m=PreviewMemory(17)
        expected=torch.zeros(17,17,dtype=torch.float64)
        for v in p:
            u=v/v.norm();h=torch.outer(u,u);h.fill_diagonal_(0)
            expected=.5*expected+.5*h;m.write(v)
            self.assertTrue(torch.allclose(m.dense(),expected,atol=1e-15,rtol=1e-14))
        self.assertEqual(m.coefficients,[.25,.5]);m.freeze()
        r=torch.rand(4,17,dtype=torch.float64)
        self.assertTrue(torch.allclose(m.apply(r),r@expected.T,atol=1e-15,rtol=1e-14))
        self.assertTrue(torch.equal(expected,expected.T));self.assertEqual(int(expected.diagonal().count_nonzero()),0)
        self.assertAlmostEqual(m.stats()['frobenius'],float(expected.norm()),places=12)
        self.assertEqual(m.stats()['nonzero_edges'],int(expected.count_nonzero()))

    def test_target_write_rejected_and_weights_unchanged_after_reads(self):
        m=PreviewMemory(17);m.write(torch.rand(17));m.freeze();old=m.fingerprint()
        with self.assertRaises(RuntimeError):m.write(torch.rand(17))
        for _ in range(5):m.apply(torch.rand(1,17))
        self.assertEqual(old,m.fingerprint());m.reset()
        self.assertEqual(m.writes,0);self.assertFalse(m.factors)

    def test_preview_tensor_does_not_alias_memory(self):
        p=torch.rand(17);m=PreviewMemory(17);m.write(p);old=m.fingerprint()
        p.zero_();self.assertEqual(old,m.fingerprint())

    def test_permutation_preserves_operator_and_spectrum(self):
        m=PreviewMemory(17)
        for _ in range(2):m.write(torch.rand(17,dtype=torch.float64))
        m.freeze();p=torch.randperm(17);q=m.permuted(p);w=m.dense()
        self.assertTrue(torch.allclose(q.dense(),w[p][:,p],atol=1e-15))
        self.assertTrue(torch.allclose(torch.linalg.eigvalsh(w),torch.linalg.eigvalsh(q.dense()),atol=1e-14))

    def test_side_is_independent_of_other_images_in_batch(self):
        m=PreviewMemory(17);m.write(torch.rand(17));m.freeze();x=torch.rand(5,17)
        a,_=side(x,m,1.)
        b=torch.cat([side(v[None],m,1.)[0] for v in x])
        self.assertTrue(torch.allclose(a,b,atol=2e-7,rtol=2e-6))


class TrialEngineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(521)
        self.a=torch.rand(1,17);self.w=torch.randn(13,17)*.15;self.b=torch.ones(13)*.1
        self.p1=torch.rand(2,17);self.p2=torch.rand(2,13)
        self.engine=TrialEngine(self.w,self.b,(.8,.6))
        self.perms=(torch.randperm(17),torch.randperm(13))

    def run_case(self,method='hebbian',p1=None,p2=None):
        return self.engine.run(self.a,self.p1 if p1 is None else p1,self.p2 if p2 is None else p2,method,self.perms)

    def test_no_preview_equals_only_inhibition(self):
        h,meta=self.run_case(p1=[],p2=[])
        i,im=self.run_case('inhibition')
        for k in h:self.assertTrue(torch.equal(h[k],i[k]))
        self.assertEqual(meta['writes'],[0,0]);self.assertEqual(im['writes'],[0,0])

    def test_controls_do_not_carry_preview_history(self):
        for method in ('vanilla','inhibition'):
            a,_=self.run_case(method);b,_=self.run_case(method,[],[])
            for k in a:self.assertTrue(torch.equal(a[k],b[k]))

    def test_order_reset_and_target_readonly(self):
        a,ma=self.run_case()
        self.engine.run(self.a*3,self.p1.flip(0),self.p2.flip(0),'shuffled',self.perms)
        b,mb=self.run_case()
        for k in a:self.assertTrue(torch.equal(a[k],b[k]))
        self.assertEqual(ma['before_target'],ma['after_target'])
        self.assertEqual(ma['before_target'],mb['before_target'])
        self.assertEqual(ma['writes'],[2,2]);self.assertEqual(len(ma['preview_boundaries']),2)
        self.assertNotEqual(ma['preview_boundaries'][0],ma['preview_boundaries'][1])

    def test_target_cannot_affect_preview_weights(self):
        _,ma=self.run_case()
        _,mb=self.engine.run(self.a*4,self.p1,self.p2,'hebbian',self.perms)
        self.assertEqual(ma['before_target'],mb['before_target'])

    def test_bridge_and_mean_controls_are_real_cascade(self):
        h,_=self.run_case()
        for method in METHODS:
            r,_=self.run_case(method)
            self.assertTrue(torch.equal(r['fc2_input'],bridge(r['fc1'],self.w,self.b)))
            if method in ('scale1','scale_both'):self.assertTrue(torch.allclose(r['fc1'].mean(),h['fc1'].mean(),atol=1e-7))
            if method in ('scale2','scale_both'):self.assertTrue(torch.allclose(r['fc2'].mean(),h['fc2'].mean(),atol=1e-7))


if __name__=='__main__':unittest.main()
