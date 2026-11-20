"""Trial-local preview Hebbian memory; frozen target reads and independent FC cascade."""
from __future__ import annotations
import hashlib
import json
import torch

DEFAULTS=dict(rho=.5,gain=.75,division=2.,alpha=.5,mean_threshold=.25,std_threshold=.1,steps=5)
METHODS=('vanilla','inhibition','hebbian','scale1','scale2','scale_both','shuffled')


def digest_tensor(x):
    a=x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


class PreviewMemory:
    """Explicit write/read phases; the only persistent payload is preview outer factors."""
    def __init__(self,dimension,rho=.5):
        self.dimension=dimension;self.rho=rho;self.reset()

    def reset(self):
        self.factors=[];self.coefficients=[];self.phase='preview';self.writes=0

    def write(self,activity):
        if self.phase!='preview':raise RuntimeError('Target memory is read-only')
        z=activity.detach().reshape(-1).clone()
        if z.numel()!=self.dimension or not torch.isfinite(z).all() or (z<0).any():raise ValueError('Invalid preview')
        u=z/z.norm().clamp_min(1e-8)
        self.coefficients=[self.rho*c for c in self.coefficients]+[1-self.rho]
        self.factors.append(u);self.writes+=1

    def freeze(self):
        if self.phase!='preview':raise RuntimeError('Already frozen')
        self.phase='target'

    def fingerprint(self):
        payload=dict(d=self.dimension,rho=self.rho,writes=self.writes,coefficients=self.coefficients,
                     factors=[digest_tensor(v) for v in self.factors])
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

    def apply(self,r):
        if self.phase!='target':raise RuntimeError('Read requires target phase')
        if r.shape[-1]!=self.dimension:raise ValueError('Dimension mismatch')
        if not self.factors:return torch.zeros_like(r)
        u=torch.stack(self.factors).to(device=r.device,dtype=r.dtype)
        c=torch.tensor(self.coefficients,device=r.device,dtype=r.dtype)
        return ((r@u.T)*c)@u-r*(c[:,None]*u.square()).sum(0)

    def dense(self,dtype=None):
        if not self.factors:return torch.zeros(self.dimension,self.dimension,dtype=dtype or torch.float64)
        u=torch.stack(self.factors).to(dtype=dtype or self.factors[0].dtype)
        c=torch.tensor(self.coefficients,device=u.device,dtype=u.dtype)
        w=u.T@(c[:,None]*u);w.fill_diagonal_(0)
        return w

    def permuted(self,permutation):
        if self.phase!='target':raise RuntimeError('Freeze before permutation')
        if sorted(permutation.detach().cpu().tolist())!=list(range(self.dimension)):raise ValueError('Not a permutation')
        new=PreviewMemory(self.dimension,self.rho)
        new.factors=[v[permutation.to(v.device)].clone() for v in self.factors]
        new.coefficients=list(self.coefficients);new.writes=self.writes;new.freeze()
        return new

    def stats(self):
        if not self.factors:return dict(writes=0,frobenius=0.,nonzero_edges=0,degree_max=0,isolated=self.dimension,coefficients=[])
        u=torch.stack(self.factors).double();c=torch.tensor(self.coefficients,device=u.device,dtype=u.dtype)
        diagonal=(c[:,None]*u.square()).sum(0)
        norm2=(c[:,None]*c[None,:]*(u@u.T).square()).sum()-diagonal.square().sum()
        # At most two preview frames in this experiment; count union of positive off-diagonal edges.
        assert len(u)<=2
        masks=u>0;degree=torch.zeros(self.dimension,device=u.device,dtype=torch.long)
        for m in masks:degree+=m.long()*(m.sum()-1)
        if len(masks)==2:
            shared=masks[0]&masks[1];degree-=shared.long()*(shared.sum()-1)
        return dict(writes=self.writes,frobenius=float(norm2.clamp_min(0).sqrt()),nonzero_edges=int(degree.sum()),
                    degree_max=int(degree.max()),isolated=int((degree==0).sum()),coefficients=list(self.coefficients))


def side(activity,memory,scale,config=DEFAULTS,gain=None):
    if memory.phase!='target':raise RuntimeError('Freeze memory before target')
    before=memory.fingerprint();x=activity/scale;r=x.clone()
    gamma=config['gain'] if gain is None else gain
    trace=[]
    for step in range(config['steps']):
        boost=memory.apply(r) if gamma else torch.zeros_like(r)
        assert float(boost.min())>=-2e-5
        drive=x+gamma*boost.clamp_min(0)
        theta=config['mean_threshold']*drive.mean(-1,keepdim=True)+config['std_threshold']*drive.std(-1,keepdim=True,unbiased=False)
        gate=drive>theta
        target=torch.where(gate,drive,0.)/(1+config['division']*r.mean(-1,keepdim=True))
        r=torch.where(gate,(1-config['alpha'])*r+config['alpha']*target,0.)
        assert torch.isfinite(r).all() and before==memory.fingerprint(),'Memory changed during target'
        trace.append(dict(step=step+1,weight_sha256=before,theta_mean=float(theta.mean()),nonzero=float((r*scale>1e-7).float().mean())))
    return r*scale,trace


def bridge(a,w,b):return torch.relu(torch.nn.functional.linear(a,w,b))
def mean_match(a,target):return a*(target.mean(-1,keepdim=True)/a.mean(-1,keepdim=True).clamp_min(1e-8))


class TrialEngine:
    def __init__(self,w,b,scales,config=None):
        self.w,self.b=w,b;self.scales=scales;self.config=dict(DEFAULTS if config is None else config)
        self.memories=[PreviewMemory(w.shape[1],self.config['rho']),PreviewMemory(w.shape[0],self.config['rho'])]

    def run(self,a1,preview1,preview2,method,permutations=None):
        if method not in METHODS:raise ValueError(method)
        if len(preview1)!=len(preview2):raise ValueError('Unpaired preview layers')
        resets=[];boundaries=[]
        for m in self.memories:m.reset();resets.append(m.fingerprint());assert not m.factors and m.writes==0
        if method not in ('vanilla','inhibition'):
            for p1,p2 in zip(preview1,preview2):
                self.memories[0].write(p1);self.memories[1].write(p2)
                boundaries.append([m.fingerprint() for m in self.memories])
        for m in self.memories:m.freeze()
        source=self.memories
        if method=='shuffled':
            if permutations is None:raise ValueError('Explicit paired permutations required')
            source=[m.permuted(p) for m,p in zip(source,permutations)]
        frozen=[m.fingerprint() for m in source];traces=[]
        def apply(a,layer):
            r,t=side(a,source[layer],self.scales[layer],self.config)
            traces.append(dict(layer=layer+1,updates=t));return r
        a2=bridge(a1,self.w,self.b)
        if method=='vanilla':h1,z2,h2=a1,a2,a2
        else:
            canonical1=apply(a1,0);canonical_input=bridge(canonical1,self.w,self.b);canonical2=apply(canonical_input,1)
            h1=mean_match(a1,canonical1) if method in ('scale1','scale_both') else canonical1
            z2=bridge(h1,self.w,self.b)
            h2=mean_match(z2,canonical2) if method in ('scale2','scale_both') else (canonical2 if method!='scale1' else apply(z2,1))
            if method in ('scale1','scale_both'):assert torch.allclose(h1.mean(-1),canonical1.mean(-1),atol=2e-6,rtol=1e-5)
            if method in ('scale2','scale_both'):assert torch.allclose(h2.mean(-1),canonical2.mean(-1),atol=2e-6,rtol=1e-5)
        assert frozen==[m.fingerprint() for m in source]
        query=[]
        for layer,x in enumerate((a1/self.scales[0],a2/self.scales[1])):
            wx=source[layer].apply(x)
            query.append(dict(match=float((x*wx).sum()/x.square().sum().clamp_min(1e-12)),
                              relative_drive=float(self.config['gain']*wx.norm()/x.norm().clamp_min(1e-8))))
        incoming=z2/self.scales[1];q=source[1].apply(incoming)
        actual=dict(match=float((incoming*q).sum()/incoming.square().sum().clamp_min(1e-12)),
                    relative_drive=float(self.config['gain']*q.norm()/incoming.norm().clamp_min(1e-8)))
        return dict(fc1=h1,fc2_input=z2,fc2=h2),dict(reset_sha256=resets,preview_boundaries=boundaries,
            writes=[m.writes for m in source],before_target=frozen,after_target=[m.fingerprint() for m in source],
            memory=[m.stats() for m in source],query=query,actual_fc2_query=actual,traces=traces)
