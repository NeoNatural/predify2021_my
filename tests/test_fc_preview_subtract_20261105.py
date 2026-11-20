import torch
from predify_experiments.fc_preview_memory_20261105 import PreviewMemory,TrialEngine as Original,bridge
from predify_experiments.fc_preview_subtract_20261105 import TrialEngine,side,DEFAULTS

def fixture():
    torch.manual_seed(812);d=8
    return torch.rand(1,d,dtype=torch.float64),torch.rand(2,d,dtype=torch.float64),torch.randn(d,d,dtype=torch.float64),torch.rand(d,dtype=torch.float64)

def test_dense_subtract_equivalence_and_negative_net_inputs():
    x,p,_,_=fixture();m=PreviewMemory(8)
    for q in p:m.write(q)
    m.freeze();W=m.dense(torch.float64);r=x.clone();negative=False
    for _ in range(5):
        h=.75*(r@W.T);pre=x+h-4*h.mean(-1,keepdim=True);negative|=bool((pre<0).any());r=.5*r+.5*pre.clamp_min(0)
    a,trace=side(x,m,1.)
    assert negative and torch.allclose(a,r,atol=1e-13,rtol=1e-13)
    assert all(t['weight_sha256']==m.fingerprint() for t in trace)

def test_no_preview_identity_for_new_rule_and_pool_only():
    x,p,w,b=fixture();e=TrialEngine(w,b,(.7,1.3))
    for method in ('hebbian','pool_only'):
        r,m=e.run(x,[],[],method)
        assert torch.equal(r['fc1'],x) and torch.equal(r['fc2'],bridge(x,w,b)) and m['writes']==[0,0]

def test_memory_inhibition_is_distinct_from_absent_memory():
    x,p,w,b=fixture();e=TrialEngine(w,b,(1.,1.));r,m=e.run(x,p,p,'pool_only')
    assert r['fc1'].mean()<x.mean() and m['writes']==[2,2]
    assert m['before_target']==m['after_target']

def test_reference_inhibition_unchanged():
    x,p,w,b=fixture();e=TrialEngine(w,b,(1.,1.));a,_=e.run(x,p,p,'inhibition');b,_=Original(w,b,(1.,1.)).run(x,p,p,'inhibition')
    for k in a:assert torch.equal(a[k],b[k])

def test_real_bridge_and_scale_mean_controls():
    x,p,w,b=fixture();e=TrialEngine(w,b,(1.,1.));full,_=e.run(x,p,p,'hebbian')
    for method in ('scale1','scale2','scale_both'):
        r,_=e.run(x,p,p,method);assert torch.equal(r['fc2_input'],bridge(r['fc1'],w,b))
        if method in ('scale1','scale_both'):assert torch.allclose(r['fc1'].mean(),full['fc1'].mean())
        if method in ('scale2','scale_both'):assert torch.allclose(r['fc2'].mean(),full['fc2'].mean())

def test_reset_counterfactual_and_batch_independence():
    x,p,w,b=fixture();e=TrialEngine(w,b,(1.,1.));a,ma=e.run(x,p,p,'hebbian');_,mb=e.run(x*1.7,p,p,'hebbian')
    assert ma['before_target']==mb['before_target']
    e.run(x,[],[],'hebbian');again,mc=e.run(x,p,p,'hebbian')
    assert ma==mc and all(torch.equal(a[k],again[k]) for k in a)
    m=e.memories[0];batch=torch.cat([x,x*1.4]);v,_=side(batch,m,1.);s=torch.cat([side(q[None],m,1.)[0] for q in batch])
    assert torch.allclose(v,s,atol=1e-13,rtol=1e-13)
