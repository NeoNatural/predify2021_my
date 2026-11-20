"""Trial-local SAE gain with active-code gating and an independent population threshold."""
from __future__ import annotations
import torch
from predify_experiments.fc_sae_gain_20261105 import tensor_hash

class TrialMemory:
    def __init__(self,batch,dimension=64,device='cpu',dtype=torch.float32):
        self.W=torch.zeros(batch,dimension,dimension,device=device,dtype=dtype)
        self.phase='preview';self.writes=0
        assert self.W.count_nonzero()==0
    def write(self,z):
        if self.phase!='preview':raise RuntimeError('Target is read-only')
        assert z.shape==(len(self.W),self.W.shape[-1]) and (z>=0).all() and torch.isfinite(z).all()
        u=z/z.norm(dim=-1,keepdim=True).clamp_min(1e-8);outer=u[:,:,None]*u[:,None,:]
        outer.diagonal(dim1=-2,dim2=-1).zero_();self.W.mul_(.5).add_(outer,alpha=.5);self.writes+=1
    def freeze(self):self.phase='target'
    def read(self,z):
        if self.phase!='target':raise RuntimeError('Freeze before target')
        return torch.bmm(self.W,z.unsqueeze(-1)).squeeze(-1)
    def fingerprints(self):return [tensor_hash(v) for v in self.W]

def query(proxy,r,memory):
    z=proxy.encode(r);h=memory.read(z)
    q=proxy.gain_signal(h*(z>0))
    kappa=(z*h).sum(-1,keepdim=True)/(z.square().sum(-1,keepdim=True)+1e-12)
    return z,h,q,kappa

@torch.inference_mode()
def side(x,proxy,memory,calibration,epsilon,lam,steps=5):
    assert memory.phase=='target' and epsilon>=0 and lam>=0
    saved=memory.W.clone();r=x.clone();trace=[]
    c=calibration['q_rms'];k0=calibration['kappa0'];assert c>0 and k0>0
    for t in range(steps):
        z,h,q,kappa=query(proxy,r,memory)
        u=torch.tanh(q/c);center=(x*u).sum(-1,keepdim=True)/x.sum(-1,keepdim=True).clamp_min(1e-12)
        modulation=epsilon*(u-center)
        mu=r.mean(-1,keepdim=True);sigma=r.std(dim=-1,keepdim=True,correction=0)
        threshold=lam*(mu+sigma)*kappa/(kappa+k0)
        r=(x*(1+modulation)-threshold).clamp_min(0)
        assert torch.equal(saved,memory.W) and (r>=0).all() and torch.isfinite(r).all()
        assert r[x==0].count_nonzero()==0 and (r<=(1+2*epsilon)*x+2e-6).all()
        trace.append(dict(step=t+1,activation=float(r.mean()),threshold=float(threshold.mean()),
            kappa=float(kappa.mean()),mu=float(mu.mean()),sigma=float(sigma.mean()),
            modulation_rms=float(modulation.square().mean().sqrt()),
            modulation_min=float(modulation.min()),modulation_max=float(modulation.max()),
            preclip_net_modulation=float((x*modulation).sum()/x.sum().clamp_min(1e-12)),
            active_latents=float((z>0).float().sum(-1).mean()),
            nonzero=float((r>1e-7).float().mean())))
    return r,trace

@torch.inference_mode()
def memory_from_preview(proxy,preview,batch,device,dtype):
    m=TrialMemory(batch,len(proxy.A),device,dtype)
    if preview is not None:
        for frame in range(preview.shape[1]):m.write(proxy.encode(preview[:,frame]))
    m.freeze();return m

@torch.inference_mode()
def cascade(x,previews,proxies,w,b,calibration,config,method='full',permutations=None):
    assert method in ('full','pool_only','scale_both','shuffled')
    memories=[memory_from_preview(p,v,len(x),x.device,x.dtype) for p,v in zip(proxies,previews)]
    if method=='shuffled':
        assert permutations is not None
        for m,p in zip(memories,permutations):
            m.W=m.W.gather(1,p[:,:,None].expand(-1,-1,m.W.shape[-1])).gather(2,p[:,None,:].expand(-1,m.W.shape[-1],-1))
    before=[m.fingerprints() for m in memories]
    eps=0. if method=='pool_only' else config['epsilon'];lam=config['lambda']
    raw2=torch.relu(torch.nn.functional.linear(x,w,b))
    one,t1=side(x,proxies[0],memories[0],calibration[0],eps,lam,config['steps'])
    incoming=torch.relu(torch.nn.functional.linear(one,w,b))
    two,t2=side(incoming,proxies[1],memories[1],calibration[1],eps,lam,config['steps'])
    if method=='scale_both':
        one=x*(one.mean(-1,keepdim=True)/x.mean(-1,keepdim=True).clamp_min(1e-12))
        incoming=torch.relu(torch.nn.functional.linear(one,w,b))
        two=incoming*(two.mean(-1,keepdim=True)/incoming.mean(-1,keepdim=True).clamp_min(1e-12))
    after=[m.fingerprints() for m in memories];assert before==after
    if previews[0] is None:
        assert torch.equal(one,x) and torch.equal(two,raw2)
    return dict(fc1=one,fc2_input=incoming,fc2=two),dict(writes=[m.writes for m in memories],
        reset=True,before=before,after=after,traces=[t1,t2])
