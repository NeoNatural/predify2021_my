"""Tiny tied SAE as a Hebbian gain side path; physical FC responses remain the signal."""
from __future__ import annotations
import hashlib
import math
import torch
from torch import nn
from predify_experiments.fc_preview_memory_20261105 import bridge,mean_match,PreviewMemory,side as old_side,DEFAULTS as OLD

FIT=dict(dimension=4096,projection=32,latents=64,l1=.1,steps=1000,learning_rate=.001,minimum_learning_rate=.0001)
UPDATE=dict(rho=.5,eta=.75,beta=4.,alpha=.5,steps=5)
METHODS=('vanilla','inhibition','pool_only','hebbian','scale1','scale2','scale_both','shuffled')

def tensor_hash(x):
    a=x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()

def model_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(tensor_hash(v).encode())
    return h.hexdigest()

def fwht(x):
    n=x.shape[-1]
    if n<1 or n&(n-1):raise ValueError('Power-of-two dimension required')
    y=x;step=1
    while step<n:
        a=y.reshape(*x.shape[:-1],-1,2,step)
        left,right=a[...,0,:],a[...,1,:]
        y=torch.stack((left+right,left-right),dim=-2).reshape(x.shape);step*=2
    return y/math.sqrt(n)

class Projection(nn.Module):
    def __init__(self,dimension=4096,projected=32,seed=20261105):
        super().__init__();g=torch.Generator().manual_seed(seed);self.dimension=dimension;self.seed=seed
        self.register_buffer('signs',(2*torch.randint(0,2,(dimension,),generator=g)-1).to(torch.int8))
        self.register_buffer('rows',torch.randperm(dimension,generator=g)[:projected])
    def forward(self,x):return fwht(x*self.signs.to(x.dtype)).index_select(-1,self.rows)
    def adjoint(self,v):
        out=torch.zeros(*v.shape[:-1],self.dimension,device=v.device,dtype=v.dtype)
        out.index_copy_(-1,self.rows,v)
        return fwht(out)*self.signs.to(v.dtype)

class GainProxy(nn.Module):
    def __init__(self,projection,center,scale,latents=64,seed=20261105):
        super().__init__();self.projection=projection;g=torch.Generator().manual_seed(seed)
        self.A=nn.Parameter(torch.randn(latents,len(center),generator=g))
        with torch.no_grad():self.A.div_(self.A.norm(dim=1,keepdim=True))
        self.b=nn.Parameter(torch.zeros(latents))
        self.register_buffer('center',torch.as_tensor(center,dtype=torch.float32).clone())
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32).clone())
    def encode_projected(self,v):return torch.relu(torch.nn.functional.linear(v/self.scale-self.center,self.A,self.b))
    def encode(self,x):return self.encode_projected(self.projection(x))
    def decode_projected(self,z):return self.scale*(torch.nn.functional.linear(z,self.A.T)+self.center)
    def gain_signal(self,h):
        # Decode Hebbian drive, with no mean/bias/reconstructed FC response.
        return self.projection.adjoint(torch.nn.functional.linear(h,self.A.T))
    @torch.no_grad()
    def constrain(self):self.A.div_(self.A.norm(dim=1,keepdim=True).clamp_min(1e-12))
    @torch.no_grad()
    def project_gradient(self):
        if self.A.grad is not None:self.A.grad.sub_(self.A*(self.A*self.A.grad).sum(1,keepdim=True))

class Memory:
    def __init__(self,dimension=64,rho=.5,device='cpu',dtype=torch.float32):
        self.W=torch.zeros(dimension,dimension,device=device,dtype=dtype);self.rho=rho;self.reset()
    def reset(self):self.W.zero_();self.phase='preview';self.writes=0
    @torch.no_grad()
    def write(self,z):
        if self.phase!='preview':raise RuntimeError('Target is read-only')
        z=z.detach().flatten()
        if len(z)!=len(self.W) or not torch.isfinite(z).all() or (z<0).any():raise ValueError('Invalid preview code')
        u=z/z.norm().clamp_min(1e-8);outer=u[:,None]*u[None,:];outer.fill_diagonal_(0)
        self.W.mul_(self.rho).add_((1-self.rho)*outer);self.writes+=1
    def freeze(self):self.phase='target'
    def fingerprint(self):return hashlib.sha256(f'{self.rho}/{self.writes}/'.encode()+tensor_hash(self.W).encode()).hexdigest()
    def read(self,z):
        if self.phase!='target':raise RuntimeError('Freeze before target')
        return z@self.W.T
    def permuted(self,p):
        assert self.phase=='target' and sorted(p.cpu().tolist())==list(range(len(self.W)))
        m=Memory(len(self.W),self.rho,self.W.device,self.W.dtype);m.W.copy_(self.W[p][:,p]);m.writes=self.writes;m.freeze();return m

def side(x,proxy,memory,config=UPDATE,positive=True):
    if memory.phase!='target':raise RuntimeError('Target must read frozen memory')
    fingerprint=memory.fingerprint();r=x.clone();trace=[]
    for step in range(config['steps']):
        z=proxy.encode(r);h=memory.read(z);q=proxy.gain_signal(h)
        modulation=config['eta']*torch.tanh(q)
        increase=x*modulation.clamp_min(0);pool=config['beta']*increase.mean(-1,keepdim=True)
        gain=1+modulation if positive else torch.ones_like(modulation)
        pre=x*gain-pool;candidate=pre.clamp_min(0)
        r=(1-config['alpha'])*r+config['alpha']*candidate
        assert fingerprint==memory.fingerprint() and torch.isfinite(r).all() and (r>=0).all()
        assert torch.count_nonzero(r[x==0])==0
        assert (r<=x*(1+config['eta'])+1e-6).all()
        trace.append(dict(step=step+1,weight_sha256=fingerprint,gain_min=float(gain.min()),gain_max=float(gain.max()),
            gain_mean=float(gain.mean()),positive_increase_mean=float(increase.mean()),pool_mean=float(pool.mean()),
            activation_mean=float(r.mean()),negative_pre_fraction=float((pre<0).float().mean()),
            latent_match=float((z*h).sum()/z.square().sum().clamp_min(1e-12))))
    return r,trace

class TrialEngine:
    def __init__(self,w,b,proxies,reference_scales,config=None):
        self.w,self.b,self.proxies,self.reference_scales=w,b,proxies,reference_scales
        self.config=dict(UPDATE if config is None else config)
        self.memories=[Memory(len(p.A),self.config['rho'],w.device,w.dtype) for p in proxies]
    def run(self,a1,preview1,preview2,method,permutations=None):
        if method not in METHODS or len(preview1)!=len(preview2):raise ValueError('Invalid condition')
        proxy_before=[model_hash(p) for p in self.proxies];resets=[];boundaries=[]
        for m in self.memories:m.reset();resets.append(m.fingerprint());assert m.writes==0 and m.W.count_nonzero()==0
        if method not in ('vanilla','inhibition'):
            for p1,p2 in zip(preview1,preview2):
                for p,x,m in zip(self.proxies,(p1,p2),self.memories):m.write(p.encode(x))
                boundaries.append([m.fingerprint() for m in self.memories])
        for m in self.memories:m.freeze()
        source=self.memories
        if method=='shuffled':
            if permutations is None:raise ValueError('Paired 64-latent permutation required')
            source=[m.permuted(p) for m,p in zip(source,permutations)]
        frozen=[m.fingerprint() for m in source];traces=[]
        def apply(x,layer):
            if method=='inhibition':
                zero=PreviewMemory(x.shape[-1]);zero.freeze();v,t=old_side(x,zero,self.reference_scales[layer],OLD)
                traces.append(dict(layer=layer+1,rule='old_inhibition_reference',updates=t));return v
            v,t=side(x,self.proxies[layer],source[layer],self.config,positive=method!='pool_only')
            traces.append(dict(layer=layer+1,rule='memory_pool_only' if method=='pool_only' else 'multiplicative_gain',updates=t));return v
        raw2=bridge(a1,self.w,self.b)
        if method=='vanilla':h1,z2,h2=a1,raw2,raw2
        else:
            canonical1=apply(a1,0);canonical2=apply(bridge(canonical1,self.w,self.b),1)
            h1=mean_match(a1,canonical1) if method in ('scale1','scale_both') else canonical1
            z2=bridge(h1,self.w,self.b)
            h2=mean_match(z2,canonical2) if method in ('scale2','scale_both') else (apply(z2,1) if method=='scale1' else canonical2)
        assert frozen==[m.fingerprint() for m in source] and proxy_before==[model_hash(p) for p in self.proxies]
        for v in (h1,z2,h2):assert v.shape[-1]==4096 or v.shape[-1]==self.w.shape[0]
        return dict(fc1=h1,fc2_input=z2,fc2=h2),dict(reset_sha256=resets,preview_boundaries=boundaries,
            writes=[m.writes for m in source],before_target=frozen,after_target=[m.fingerprint() for m in source],
            proxy_before=proxy_before,proxy_after=[model_hash(p) for p in self.proxies],traces=traces)

def budget(proxy,memory):
    return dict(trainable=sum(p.numel() for p in proxy.parameters()),floating_buffers=sum(b.numel() for b in proxy.buffers() if b.is_floating_point()),
        projection_metadata_bytes=sum(b.numel()*b.element_size() for b in proxy.projection.buffers()),
        temporary_hebb_values=memory.W.numel(),tensor_payload_bytes=sum(p.numel()*p.element_size() for p in proxy.parameters())+
            sum(b.numel()*b.element_size() for b in proxy.buffers())+memory.W.numel()*memory.W.element_size())
