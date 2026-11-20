import numpy as np
import torch
from predify_experiments.fc_sae_proxy_20261105 import SparseProxy,training_statistics,fingerprint,objective

def model():return SparseProxy(7,3,np.ones(7,dtype=np.float32)*.4,.73,seed=37)

def test_zero_edit_is_exact_even_when_reconstruction_is_poor():
    m=model().eval();x=torch.tensor([[0.,1.,2.,.1,0.,3.,.7],[1e5,0.,.1,5.,3.,0.,2.]])
    before=fingerprint(m);hat,_=m(x);y,d=m.zero_edit(x)
    assert not torch.allclose(hat,x) and torch.equal(x,y) and torch.count_nonzero(d)==0 and before==fingerprint(m)

def test_weighted_training_only_statistics():
    x=np.array([[0.,2.],[4.,0.],[9.,9.]],np.float32);w=np.array([.25,.25,.5])
    c,s=training_statistics(x,w);assert np.allclose(c*s,np.average(x,axis=0,weights=w))
    assert np.isclose(s*s,np.average(np.mean(x.astype(float)**2,axis=1),weights=w))

def test_no_batch_population_normalization():
    m=model().double();x=torch.rand(5,7,dtype=torch.float64)
    a,z=m(x);b=torch.cat([m(v[None])[0] for v in x]);assert torch.allclose(a,b,atol=1e-13,rtol=1e-13)
    assert (z>=0).all() and torch.allclose(m.decoder.norm(dim=0),torch.ones(3,dtype=torch.float64),atol=1e-7)

def test_dictionary_constraint_and_training_gradients():
    m=model();x=torch.rand(11,7);w=torch.ones(11)/11;opt=torch.optim.Adam(m.parameters(),lr=.001)
    before=fingerprint(m);loss,_=objective(m,x,w,.1);loss.backward();m.project_dictionary_gradient();opt.step();m.constrain_dictionary()
    assert torch.isfinite(loss) and before!=fingerprint(m)
    assert torch.allclose(m.decoder.norm(dim=0),torch.ones(3),atol=1e-6)
