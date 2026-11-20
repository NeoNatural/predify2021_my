import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from predify_experiments.native_categories_20261029 import group_statistics
import analysis.native_categories_20261029 as new
import analysis.filtered_cv_20261027 as old
NATIVE_FIT=new.fit_svm

class TestProjectionHead:
    """Test-only dimension reduction to exercise cache/routing without 35 costly fits."""
    def __init__(self,head):self.head=head
    def decision_function(self,x):return self.head.decision_function(x[:,:8])

def small_test_fit(x,y,w):
    head,attempts=NATIVE_FIT(x[:,:8].copy(),y,w)
    return TestProjectionHead(head),attempts

class NativeReadoutTests(unittest.TestCase):
    def tearDown(self):new.CATEGORIES=('animals','clothing','food','vehicles')

    def test_mvpa_identical_four_class_fit(self):
        x=np.random.default_rng(42).normal(size=(32,19)).astype('float32');y=np.tile(np.arange(4),8);w=np.full(32,1/3)
        a,_=new.fit_svm(x,y,w);b,_=old.fit_svm(x,y,w)
        np.testing.assert_array_equal(a.decision_function(x),b.decision_function(x))

    def test_five_class_fit_and_auc(self):
        new.CATEGORIES=tuple('abcde')
        x=np.random.default_rng(3).normal(size=(40,13)).astype('float32');y=np.tile(np.arange(5),8)
        head,_=new.fit_svm(x,y,np.ones(40));s=head.decision_function(x)
        self.assertEqual(s.shape,(40,5));a,v=new.auc_macro(y,s)
        self.assertEqual(len(v),5);self.assertAlmostEqual(a,np.mean(v))

    def test_mode_t9_weights_and_training_isolation(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(12,30,15,4096)).astype('float32');y=np.tile(np.arange(5),6);fold=np.tile(np.arange(5),6);w=np.full(30,1/3)
        for carrier,j in [('hebb',0),('pcoder',4),('joint',8)]:
            train,labels,weights,tr=new.t9_training_rows(x,y,fold,w,2,carrier)
            self.assertTrue(np.all(fold[tr]!=2));self.assertEqual(len(train),4*len(tr))
            np.testing.assert_array_equal(train[:len(tr)],x[j,tr,9])
            np.testing.assert_array_equal(labels,np.tile(y[tr],4));np.testing.assert_allclose(weights,1/12)
            self.assertAlmostEqual(weights.sum(),w[tr].sum())

    def test_four_image_five_class_geometry_matches_full_maps(self):
        new.CATEGORIES=tuple('abcde');rng=np.random.default_rng(7)
        x=rng.normal(size=(10,4,3,23)).astype('float32');cat=np.repeat(np.arange(5),2)
        stats=[group_statistics(list(v)) for v in x]
        raw=np.stack([v['sum_x'] for v in stats],axis=1).astype('float64');norm=np.stack([v['sum_u'] for v in stats],axis=1).astype('float64')
        values=dict(graw=np.einsum('tid,tjd->tij',raw,raw),gnorm=np.einsum('tid,tjd->tij',norm,norm),**{a:np.stack([v[b] for v in stats],axis=1) for a,b in [('sq','sum_sq'),('unorm','sum_unorm_sq'),('zeros','zeros')]})
        result=new.geometry_from_grams(**values,category=cat,weights=np.ones((1,10)))
        for t in range(3):
            a=x[:,:,t].reshape(40,23).astype('float64');u=a/np.linalg.norm(a,axis=1)[:,None]
            tr=[];nt=[];co=[];cent=[]
            for c in range(5):
                b=a[np.repeat(cat,4)==c];v=u[np.repeat(cat,4)==c];tr.append(b.var(0).sum());nt.append(v.var(0).sum());cent.append(v.mean(0));co.append((v@v.T)[~np.eye(len(v),dtype=bool)].mean())
            cent=np.array(cent);between=np.square(cent-cent.mean(0)).sum(1).mean()
            for k,expect in [('raw_trace_variance',np.mean(tr)),('normalized_trace_variance',np.mean(nt)),('within_class_cosine',np.mean(co)),('normalized_fisher_ratio',between/np.mean(nt))]:self.assertAlmostEqual(result[k][0,t],expect,places=6)

    def test_reference_training_uses_one_unique_row_per_target(self):
        # Reference row count/weight are independent of formal content/noise replication.
        rng=np.random.default_rng(23);x=rng.normal(size=(50,15,4096)).astype('float32');y=np.tile(np.arange(5),10);f=np.repeat(np.arange(5),10);tr=f!=3
        a,_=new.fit_svm(x[tr,9],y[tr],np.ones(tr.sum()));mean=a[0].mean_.copy()
        np.testing.assert_allclose(mean,x[tr,9].astype(float).mean(0),atol=1e-7)
        a.decision_function(x[~tr].reshape(-1,4096));np.testing.assert_array_equal(mean,a[0].mean_)

    def test_all_t9_heads_and_reference_reuse_across_noise(self):
        new.CATEGORIES=tuple('abcde');rng=np.random.default_rng(12);n=25
        ref=rng.normal(size=(n,15,4096)).astype('float32');y=np.tile(np.arange(5),5);fold=np.repeat(np.arange(5),5)
        x=np.stack([ref+np.float32(j/10) for j in range(12)])
        root=Path(__file__).resolve().parents[1]/'docs/native_categories_readouts_20261029/test_tmp';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp,patch.object(new,'OUT',Path(tmp)),patch.object(new,'fit_svm',side_effect=small_test_fit) as fit,new.threadpool_limits(limits=1):
            new.t9_decode('synthetic','clean',x,y,fold,np.ones(n),ref,y,fold)
            self.assertEqual(fit.call_count,20)
            noisy=np.tile(x,(1,3,1,1));ny=np.tile(y,3);nf=np.tile(fold,3)
            new.t9_decode('synthetic','noisy',noisy,ny,nf,np.full(3*n,1/3),ref,y,fold)
            self.assertEqual(fit.call_count,35)
            files=list(Path(tmp).rglob('*.pckl'));self.assertEqual(len(files),35)
            for kind in ('t9_mode_specific','t9_clean_reference'):
                with np.load(Path(tmp)/'synthetic/cv/noisy'/kind/'joint_exact.npz') as z:
                    self.assertEqual(z['scores'].shape,(75,15,5));self.assertTrue(np.isfinite(z['scores']).all())

if __name__=='__main__':unittest.main()
