import torch
import pytest
from predify_experiments.conv_fc_transfer_20261106 import (
    SpatialCodec,query,side,TransferOperator,new_memory,shuffle_memory)
from predify_experiments.fc_sae_gain_20261105 import GainProxy,Projection

CAL=dict(q_rms=.1,kappa0=.1)


def codec():
    return SpatialCodec(GainProxy(Projection(8,4,17),torch.zeros(4),1.,6,19).double().eval().requires_grad_(False))


def test_spatial_sae_query_is_energy_gradient():
    torch.manual_seed(5);c=codec();m=new_memory(6)
    for name in ('intra','trace_inter','inter'): setattr(m,name,getattr(m,name).double())
    for _ in range(2):m.write(c.encode(torch.rand(1,8,3,4,dtype=torch.float64)))
    m.freeze();x=torch.rand(1,8,3,4,dtype=torch.float64,requires_grad=True)
    q,k,z=query(x,m,c)
    h=(2/3)*torch.nn.functional.conv2d(z,m.intra[:,:,None,None])+(1/3)*torch.nn.functional.conv2d(z,m.inter,padding=1)
    grad=torch.autograd.grad(.5*(z*h).sum(),x)[0]
    assert torch.allclose(q/c.proxy.scale,grad,atol=1e-12)
    assert (k>=0).all()


def test_both_codecs_no_memory_are_native_identity_and_decoder_is_forbidden():
    for c,dim in [(SpatialCodec(),8),(codec(),6)]:
        m=new_memory(dim);op=TransferOperator(m,c,CAL,scale=.87)
        if c.proxy is not None:
            def forbidden(*a):raise AssertionError('Reconstruction entered forward path')
            c.proxy.decode_projected=forbidden
        x=torch.rand(1,8,3,4)
        assert torch.equal(op(x),x)


def test_budget_balance_pool_control_and_true_zeroing():
    torch.manual_seed(7);x=torch.rand(1,8,4,4);x[:,::2,0,:]=0
    c=SpatialCodec();m=new_memory(8)
    m.write(x);m.write(x*.7);m.freeze();before=m.fingerprint()
    a,ta=side(x,m,c,CAL,.03,.03,1,True)
    b,tb=side(x,m,c,CAL,0.,.03,1,True)
    assert abs(ta[0]['preclip_balance'])<1e-7
    assert ta[0]['threshold']==tb[0]['threshold']
    assert a[x==0].count_nonzero()==0 and (a<=1.06*x+1e-6).all()
    gone,_=side(x,m,c,CAL,0.,100.,5)
    assert gone.count_nonzero()<x.count_nonzero()
    assert before==m.fingerprint()
    with pytest.raises(RuntimeError):m.write(x)


def test_spatial_shuffle_preserves_budget_reciprocity_and_scale_profile():
    torch.manual_seed(9);m=new_memory(8);x=torch.rand(1,8,4,4)
    m.write(x);m.write(x);m.freeze()
    a=m.intra.clone();b=m.inter.clone();p=torch.randperm(8)
    shuffle_memory(m,p)
    assert torch.equal(m.intra,a[p][:,p]) and torch.equal(m.inter,b[p][:,p])
    assert torch.equal(m.intra,m.intra.T)
    op=TransferOperator(new_memory(8),SpatialCodec(),CAL,source='none',method='scale')
    op.reset([.123,.456])
    for desired in (.123,.456):assert abs(float(op(x).mean())-desired)<1e-7

