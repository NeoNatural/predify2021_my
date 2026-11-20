import unittest
import numpy as np
from predify_experiments.filtered_cv_20261027 import pair_statistics,encode_geometry,geometry_value,design_filter,conditions
from analysis.filtered_cv_20261027 import geometry_from_grams,auc_macro,weighted_rows,fit_svm,relative

def make_grams(x):
    # x[G,2,T,D], two images per concept
    values=[pair_statistics(v[0],v[1]) for v in x]
    raw=np.stack([v['sum_x'] for v in values],axis=1).astype(np.float64)
    norm=np.stack([v['sum_u'] for v in values],axis=1).astype(np.float64)
    return dict(graw=np.einsum('tid,tjd->tij',raw,raw),gnorm=np.einsum('tid,tjd->tij',norm,norm),
                **{out:np.stack([v[key] for v in values],axis=1) for out,key in [('sq','sum_sq'),('unorm','sum_unorm_sq'),('zeros','zeros')]})

class FilteredCVTests(unittest.TestCase):
    def test_filter_frequency_spec(self):
        k,f,db,m=design_filter();self.assertLessEqual(m['ripple_db'],.1);self.assertGreaterEqual(m['stop_attenuation_db'],60)
        np.testing.assert_allclose(k,k[::-1]);self.assertAlmostEqual(k.sum(),1)
    def test_only_planned_conditions(self):
        c=conditions();self.assertEqual(len(c),12);self.assertEqual({r['coeff'] for r in c},{0.,.05});self.assertFalse(any('neutral' in r['name'] for r in c))
    def test_lossless_encoding(self):
        a=np.random.default_rng(1).normal(size=(15,17)).astype(np.float32);v=encode_geometry({'x':a})
        self.assertTrue(np.array_equal(a.view(np.uint32),geometry_value(v,'x').view(np.uint32)))
    def test_full_map_geometry_matches_direct(self):
        x=np.random.default_rng(2).normal(size=(8,2,3,29)).astype(np.float32);cat=np.repeat(np.arange(4),2);g=make_grams(x)
        r=geometry_from_grams(**g,category=cat,weights=np.ones((1,8)))
        for t in range(3):
            raw=x[:,:,t].reshape(16,29).astype(float);u=raw/np.linalg.norm(raw,axis=1,keepdims=True)
            trace=[];nt=[];cos=[];means=[]
            for c in range(4):
                v=raw[np.repeat(cat,2)==c];z=u[np.repeat(cat,2)==c];trace.append(v.var(0).sum());nt.append(z.var(0).sum());means.append(z.mean(0))
                cos.append((z@z.T)[~np.eye(len(z),dtype=bool)].mean())
            means=np.array(means);between=np.square(means-means.mean(0)).sum(1).mean()
            for key,expected in [('raw_trace_variance',np.mean(trace)),('normalized_trace_variance',np.mean(nt)),('within_class_cosine',np.mean(cos)),('normalized_between_scatter',between),('normalized_fisher_ratio',between/np.mean(nt))]:
                self.assertAlmostEqual(r[key][0,t],expected,places=6)
    def test_concept_multiplicity_matches_explicit_replication(self):
        x=np.random.default_rng(4).normal(size=(8,2,2,13)).astype(np.float32);cat=np.repeat(np.arange(4),2);w=np.tile([2,0],4)
        a=geometry_from_grams(**make_grams(x),category=cat,weights=w)
        y=np.repeat(x,w,axis=0);b=geometry_from_grams(**make_grams(y),category=cat,weights=np.ones(8))
        for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-7)
    def test_scaling_invariant_normalized_geometry(self):
        x=np.random.default_rng(5).normal(size=(8,2,2,19)).astype(np.float32);cat=np.repeat(np.arange(4),2)
        a=geometry_from_grams(**make_grams(x),category=cat,weights=np.ones(8));b=geometry_from_grams(**make_grams(x*3),category=cat,weights=np.ones(8))
        for k in a:
            if k!='raw_trace_variance':np.testing.assert_allclose(a[k],b[k],atol=2e-7)
        np.testing.assert_allclose(b['raw_trace_variance'],9*a['raw_trace_variance'],rtol=1e-6)
    def test_zero_vectors_not_assumed_unit_length(self):
        x=np.zeros((8,2,2,7),np.float32);x[::2,:,0,0]=1;cat=np.repeat(np.arange(4),2)
        r=geometry_from_grams(**make_grams(x),category=cat,weights=np.ones(8))
        np.testing.assert_allclose(r['zero_vector_fraction'],[[.5,1.]])
        np.testing.assert_allclose(r['within_class_cosine'],[[1/6,0.]])
        self.assertTrue(np.isfinite(r['normalized_fisher_ratio']).all())
    def test_auc_manual_pair_ranking(self):
        y=np.repeat(np.arange(4),2);s=np.random.default_rng(6).normal(size=(8,4));macro,classes=auc_macro(y,s)
        expected=[]
        for c in range(4):
            pos=s[y==c,c];neg=s[y!=c,c];expected.append(np.mean((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg)))
        np.testing.assert_allclose(classes,expected);self.assertAlmostEqual(macro,np.mean(expected))
    def test_noisy_weight_preserves_target_total(self):
        x=np.zeros((4,64,15,4096),np.float32);clean,wc=weighted_rows(x,[0]);noisy,wn=weighted_rows(x,[1,2,3])
        self.assertEqual(wc.sum(),64);self.assertAlmostEqual(wn.sum(),64);self.assertEqual(noisy.shape[0],192)
    def test_weighted_scaler_fits_training_only(self):
        x=np.random.default_rng(7).normal(size=(20,12)).astype(np.float32);y=np.tile(np.arange(4),5);w=np.arange(1,21,dtype=float)
        head,attempts=fit_svm(x,y,w);np.testing.assert_allclose(head[0].mean_,np.average(x,axis=0,weights=w),atol=1e-6)
        before=head[0].mean_.copy();head.decision_function(np.ones((4,12),np.float32)*1e6);np.testing.assert_array_equal(head[0].mean_,before)
        self.assertTrue(attempts[-1]['converged'])
    def test_percentage_zero_baseline_is_nan(self):
        np.testing.assert_allclose(relative(np.array([2.,3.]),np.array([1.,0.])),[100.,np.nan],equal_nan=True)

if __name__=='__main__':unittest.main()
