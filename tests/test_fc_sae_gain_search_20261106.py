import torch
from analysis.fc_sae_gain_search_20261106 import side
from predify_experiments.fc_sae_gain_20261105 import Projection,GainProxy

class ConstantDrive:
    def encode(self,x):return torch.ones(len(x),64,dtype=x.dtype,device=x.device)
    def gain_signal(self,h):return h.mean(-1,keepdim=True).expand(-1,8)*.1

def test_hard_update_removes_weak_tail_while_mixing_retains_it():
    x=torch.tensor([[0.,.001,.1,.2,.4,.6,1.,2.]],dtype=torch.float64);W=torch.eye(64,dtype=x.dtype)[None]
    cfg=dict(eta=.1,beta=4.,gamma=0.,alpha=1.,steps=5)
    hard=side(x,ConstantDrive(),W,1.,cfg);mixed=side(x,ConstantDrive(),W,1.,dict(cfg,alpha=.5))
    assert hard[0,1]==0 and mixed[0,1]>0 and hard[0,0]==mixed[0,0]==0

def test_calibrated_zero_memory_is_identity_and_does_not_mutate_weights():
    x=torch.rand(3,8,dtype=torch.float64);W=torch.zeros(3,64,64,dtype=x.dtype);saved=W.clone()
    cfg=dict(eta=.15,beta=12.,gamma=1.,alpha=1.,steps=5)
    assert torch.equal(side(x,ConstantDrive(),W,.01,cfg),x) and torch.equal(W,saved)

def test_batched_trial_memories_have_no_cross_trial_influence():
    torch.manual_seed(17);p=GainProxy(Projection(8,4,19),torch.zeros(4),1.,64).double().eval().requires_grad_(False)
    x=torch.rand(3,8,dtype=torch.float64);W=torch.rand(3,64,64,dtype=x.dtype)*.01;saved=W.clone()
    cfg=dict(eta=.05,beta=4.,gamma=.25,alpha=1.,steps=5)
    batch=side(x,p,W,.03,cfg);one=torch.cat([side(x[i:i+1],p,W[i:i+1],.03,cfg) for i in range(3)])
    assert torch.allclose(batch,one,atol=1e-13) and torch.equal(W,saved)
