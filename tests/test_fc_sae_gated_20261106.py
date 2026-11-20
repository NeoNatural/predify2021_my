import torch
from predify_experiments.fc_sae_gain_20261105 import Projection,GainProxy
from predify_experiments.fc_sae_gated_20261106 import TrialMemory,memory_from_preview,query,side,cascade

def proxy():return GainProxy(Projection(8,4,19),torch.zeros(4),1.,6,17).double().eval().requires_grad_(False)
CAL=dict(q_rms=.1,kappa0=.1)

def test_two_preview_writes_and_target_guard():
    torch.manual_seed(14);p=proxy();frames=torch.rand(3,2,8,dtype=torch.float64)
    m=memory_from_preview(p,frames,3,'cpu',torch.float64);z=p.encode(frames);u=z/z.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    W=.25*u[:,0,:,None]*u[:,0,None,:]+.5*u[:,1,:,None]*u[:,1,None,:];W.diagonal(dim1=-2,dim2=-1).zero_()
    assert torch.allclose(m.W,W,atol=1e-14) and m.writes==2
    try:m.write(z[:,0]);raise AssertionError('Target write succeeded')
    except RuntimeError:pass

def test_gated_signal_is_memory_energy_gradient():
    torch.manual_seed(12);p=proxy();m=memory_from_preview(p,torch.rand(2,2,8,dtype=torch.float64),2,'cpu',torch.float64)
    m.W=m.W.clone()
    x=torch.rand(2,8,dtype=torch.float64,requires_grad=True);z,h,q,_=query(p,x,m)
    grad=torch.autograd.grad(.5*(z*h).sum(),x)[0]
    assert torch.allclose(q/p.scale,grad,atol=1e-13)

def test_no_preview_identity_and_auxiliary_decoder_never_used():
    p=proxy();m=memory_from_preview(p,None,3,'cpu',torch.float64);x=torch.rand(3,8,dtype=torch.float64)
    def forbidden(*args):raise AssertionError('Reconstruction entered target')
    p.decode_projected=forbidden
    r,_=side(x,p,m,CAL,.03,.03);assert torch.equal(r,x)

def test_modulation_mean_balance_zero_support_and_pool_independent_of_epsilon():
    torch.manual_seed(23);p=proxy();x=torch.rand(3,8,dtype=torch.float64);x[:,::3]=0
    m=memory_from_preview(p,torch.rand(3,2,8,dtype=torch.float64),3,'cpu',torch.float64);saved=m.W.clone()
    a,ta=side(x,p,m,CAL,.03,.03,steps=1);b,tb=side(x,p,m,CAL,0.,.03,steps=1)
    assert abs(ta[0]['preclip_net_modulation'])<1e-14 and ta[0]['threshold']==tb[0]['threshold']
    assert torch.equal(saved,m.W) and a[x==0].count_nonzero()==0 and (a<=1.06*x+1e-14).all()
    assert (b<x).any()

def test_trial_reset_bridge_and_batch_independence():
    torch.manual_seed(29);p=[proxy(),proxy()];x=torch.rand(3,8,dtype=torch.float64)
    v=[torch.rand(3,2,8,dtype=torch.float64) for _ in p];w=torch.randn(8,8,dtype=torch.float64);b=torch.ones(8,dtype=torch.float64)
    cfg=dict(epsilon=.01,**{'lambda':.03},steps=5)
    a,ma=cascade(x,v,p,w,b,[CAL,CAL],cfg);cascade(x,[None,None],p,w,b,[CAL,CAL],cfg)
    c,mc=cascade(x,v,p,w,b,[CAL,CAL],cfg)
    assert ma==mc and all(torch.equal(a[k],c[k]) for k in a)
    assert torch.equal(a['fc2_input'],torch.relu(a['fc1']@w.T+b))
    solo,_=cascade(x[:1],[z[:1] for z in v],p,w,b,[CAL,CAL],cfg)
    assert all(torch.allclose(solo[k],a[k][:1],atol=1e-13) for k in a)
