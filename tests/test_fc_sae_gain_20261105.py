import math
import torch
from predify_experiments.fc_sae_gain_20261105 import Projection,GainProxy,Memory,TrialEngine,side,fwht,budget,model_hash

def proxy(d=8,p=4,m=6):return GainProxy(Projection(d,p,19),torch.zeros(p),1.,m,17).double().eval().requires_grad_(False)

def test_fast_projection_dense_equivalence_and_adjoint():
    H=torch.ones(1,1,dtype=torch.float64)
    for _ in range(3):H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    H/=math.sqrt(8);P=Projection(8,4,19).double();x=torch.randn(5,8,dtype=torch.float64);v=torch.randn(5,4,dtype=torch.float64)
    dense=H[P.rows]*P.signs[None,:]
    assert torch.allclose(P(x),x@dense.T,atol=1e-13)
    assert torch.allclose(P.adjoint(v),v@dense,atol=1e-13)
    assert torch.allclose(P(P.adjoint(v)),v,atol=1e-13)

def test_memory_write_read_matches_outer_formula_and_freeze():
    m=Memory(6,dtype=torch.float64);a=torch.rand(2,6,dtype=torch.float64)
    for x in a:m.write(x)
    m.freeze();u=a/a.norm(dim=1,keepdim=True);W=.25*u[0,:,None]*u[0,None,:]+.5*u[1,:,None]*u[1,None,:];W.fill_diagonal_(0)
    assert torch.allclose(m.W,W,atol=1e-13);q=torch.rand(3,6,dtype=torch.float64);assert torch.allclose(m.read(q),q@W.T)
    try:m.write(a[0]);raise AssertionError('Target write accepted')
    except RuntimeError:pass

def test_zero_memory_exact_identity_and_no_decoder_output_routing():
    p=proxy();m=Memory(6,dtype=torch.float64);m.freeze();x=torch.rand(3,8,dtype=torch.float64)
    def forbidden(*args):raise AssertionError('Auxiliary reconstruction entered target')
    p.decode_projected=forbidden
    a,_=side(x,p,m);assert torch.equal(a,x)

def test_multiplication_preserves_zeros_and_bounds_and_batch_independence():
    p=proxy();x=torch.rand(3,8,dtype=torch.float64);x[:,::2]=0;m=Memory(6,dtype=torch.float64)
    for z in p.encode(torch.rand(2,8,dtype=torch.float64)):m.write(z)
    m.freeze();a,_=side(x,p,m);b=torch.cat([side(v[None],p,m)[0] for v in x])
    assert torch.allclose(a,b,atol=1e-13) and torch.count_nonzero(a[x==0])==0 and (a<=1.75*x+1e-13).all()

def test_trial_reset_real_bridge_and_original_feature_dimensions():
    torch.manual_seed(27);p=[proxy(),proxy()];w=torch.randn(8,8,dtype=torch.float64);b=torch.rand(8,dtype=torch.float64);e=TrialEngine(w,b,p,(1.,1.))
    x=torch.rand(1,8,dtype=torch.float64);v=torch.rand(2,8,dtype=torch.float64)
    a,ma=e.run(x,v,v,'hebbian');e.run(x,[],[],'hebbian');c,mc=e.run(x,v,v,'hebbian')
    assert ma==mc and all(torch.equal(a[k],c[k]) for k in a)
    assert torch.equal(a['fc2_input'],torch.relu(a['fc1']@w.T+b)) and ma['proxy_before']==ma['proxy_after']

def test_actual_parameter_and_payload_budget():
    p=GainProxy(Projection(),torch.zeros(32),1.);m=Memory();b=budget(p,m)
    assert b['trainable']==2112 and b['floating_buffers']==33 and b['temporary_hebb_values']==4096
    assert b['tensor_payload_bytes']==29316 and b['tensor_payload_bytes']<8194*4
