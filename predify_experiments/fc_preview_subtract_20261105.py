"""Independent read-only trial memory with subtractive population competition."""
from __future__ import annotations
import torch
from predify_experiments.fc_preview_memory_20261105 import PreviewMemory,bridge,mean_match,side as reference_side,DEFAULTS as REFERENCE

DEFAULTS=dict(rho=.5,gain=.75,beta=4.,alpha=.5,steps=5)
METHODS=('vanilla','inhibition','pool_only','hebbian','scale1','scale2','scale_both','shuffled')

def side(activity,memory,scale,config=DEFAULTS,positive=True):
    if memory.phase!='target':raise RuntimeError('Target must read frozen memory')
    fingerprint=memory.fingerprint();x=activity/scale;r=x.clone();trace=[]
    for t in range(config['steps']):
        raw=memory.apply(r);assert float(raw.min())>=-2e-5
        h=config['gain']*raw.clamp_min(0);pool=config['beta']*h.mean(-1,keepdim=True)
        pre=x+(h if positive else torch.zeros_like(h))-pool
        v=pre.clamp_min(0);r=(1-config['alpha'])*r+config['alpha']*v
        assert fingerprint==memory.fingerprint() and torch.isfinite(r).all() and (r>=0).all()
        trace.append(dict(step=t+1,weight_sha256=fingerprint,boost_mean=float(h.mean()),pool_mean=float(pool.mean()),
            pre_mean=float(pre.mean()),rectified_mean=float(v.mean()),state_mean=float(r.mean()),negative_fraction=float((pre<0).float().mean()),
            clipping_correction=float((-pre).clamp_min(0).mean())))
    # The zero operator is exactly the identity in physical units; avoid a
    # needless FP32 divide/multiply round trip in the no-preview control.
    return (activity if not memory.factors else r*scale),trace

class TrialEngine:
    def __init__(self,w,b,scales,config=None):
        self.w,self.b,self.scales=w,b,scales;self.config=dict(DEFAULTS if config is None else config)
        self.memories=[PreviewMemory(w.shape[1],self.config['rho']),PreviewMemory(w.shape[0],self.config['rho'])]

    def run(self,a1,preview1,preview2,method,permutations=None):
        if method not in METHODS:raise ValueError(method)
        if len(preview1)!=len(preview2):raise ValueError('Unpaired previews')
        resets=[];boundaries=[]
        for m in self.memories:m.reset();resets.append(m.fingerprint());assert not m.factors and m.writes==0
        if method not in ('vanilla','inhibition'):
            for p1,p2 in zip(preview1,preview2):
                self.memories[0].write(p1);self.memories[1].write(p2)
                boundaries.append([m.fingerprint() for m in self.memories])
        for m in self.memories:m.freeze()
        source=self.memories
        if method=='shuffled':
            if permutations is None:raise ValueError('Explicit paired shuffle required')
            source=[m.permuted(p) for m,p in zip(source,permutations)]
        frozen=[m.fingerprint() for m in source];traces=[]
        def apply(a,layer):
            if method=='inhibition':
                value,updates=reference_side(a,source[layer],self.scales[layer],REFERENCE)
                traces.append(dict(layer=layer+1,rule='previous_inhibition_reference',updates=updates))
            else:
                value,updates=side(a,source[layer],self.scales[layer],self.config,positive=method!='pool_only')
                traces.append(dict(layer=layer+1,rule='subtract_pool_only' if method=='pool_only' else 'subtract',updates=updates))
            return value
        raw2=bridge(a1,self.w,self.b)
        if method=='vanilla':h1,z2,h2=a1,raw2,raw2
        else:
            canonical1=apply(a1,0);canonical2=apply(bridge(canonical1,self.w,self.b),1)
            h1=mean_match(a1,canonical1) if method in ('scale1','scale_both') else canonical1
            z2=bridge(h1,self.w,self.b)
            h2=mean_match(z2,canonical2) if method in ('scale2','scale_both') else (apply(z2,1) if method=='scale1' else canonical2)
        assert frozen==[m.fingerprint() for m in source]
        return dict(fc1=h1,fc2_input=z2,fc2=h2),dict(reset_sha256=resets,preview_boundaries=boundaries,writes=[m.writes for m in source],
            before_target=frozen,after_target=[m.fingerprint() for m in source],traces=traces)
