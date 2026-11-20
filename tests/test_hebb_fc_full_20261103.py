import numpy as np
import torch
import json
from analysis.hebb_fc_full_replay_20261103 import batched, matched
from analysis.report_hebb_fc_native_pilot_20261102 import fast_single

def test_batched_rules_agree_with_single_sample():
    torch.manual_seed(41)
    x=torch.rand(6,32)
    E=torch.rand(32,32);E=(E+E.T)/2;E.fill_diagonal_(0);E/=E.sum(1).max()
    model={'scale':torch.tensor(.7),'matrices':{'corr_k128':E,'corr_k32':E}}
    for method in ('mean_std','inhibition_only'):
        for layer in ('fc1','fc2'):
            expected=torch.cat([fast_single(a[None],model,method,layer) for a in x])
            torch.testing.assert_close(batched(x,model,method,layer),expected,atol=2e-6,rtol=2e-6)

def test_mean_matched_for_each_sample():
    x=torch.tensor([[1.,2.,3.],[2.,4.,1.]])
    target=torch.tensor([[3.,2.,1.],[1.,1.,1.]])
    torch.testing.assert_close(matched(x,target).mean(1),target.mean(1))

def test_noise_first_indexing_matches_row_weights():
    n=7
    a=np.arange(4*n).reshape(4,n)
    assert np.array_equal(a[[1,2,3]].reshape(-1),np.arange(n,4*n))

def test_decode_fold_uses_outer_training_and_produces_expected_rows(tmp_path,monkeypatch):
    import analysis.hebb_fc_full_decode_20261103 as d
    class Head:
        def decision_function(self,x):
            return np.tile(np.arange(4,dtype=float),(len(x),1))
    def fit(x,y,w):
        assert set(y)==set(range(4)) and len(x)==len(y)==len(w)
        return Head(),[{'converged':True}]
    root=tmp_path/'out';native_root=tmp_path/'native';ds='imagenet';dest=root/ds/'folds/fold0';dest.mkdir(parents=True)
    source=native_root/ds;source.mkdir(parents=True)
    trials=[{'category_index':i%4,'fold':i//4,'dataset_index':i} for i in range(20)]
    (source/'manifest.json').write_text(json.dumps({'trials':trials,'categories':['a','b','c','d']}))
    (dest/'replay_complete.json').write_text(json.dumps({'test_rows':[0,1,2,3]}))
    x=np.lib.format.open_memmap(dest/'mean_std_fc2.npy',mode='w+',dtype='float32',shape=(12,4,20,15,4096));x.flush();del x
    x=np.lib.format.open_memmap(dest/'mean_std_reference_fc2.npy',mode='w+',dtype='float32',shape=(20,15,4096));x.flush();del x
    monkeypatch.setattr(d,'OUT',root);monkeypatch.setattr(d.native,'OUT',native_root)
    monkeypatch.setattr(d,'METHODS',('mean_std',));monkeypatch.setattr(d.native,'CARRIERS',('hebb',))
    monkeypatch.setattr(d.native,'conditions',lambda:[{'carrier':'hebb','name':f'hebb_{i}'} for i in range(4)])
    monkeypatch.setattr(d,'fit_svm',fit)
    d.decode(ds,0,workers=1)
    with np.load(dest/'scores/mean_std/noisy_hebb_0.npz') as z:
        assert z['mvpa'].shape==(12,15,4)
        np.testing.assert_array_equal(z['rows'],np.tile(np.arange(4),3))
        np.testing.assert_array_equal(z['seeds'],np.repeat([2921,2922,2923],4))
