"""FC-inspired gain redistribution and memory-matched subtraction on native conv maps."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from .conv_preview_memory_20261106 import DEFAULTS, PreviewMemory, inhibition

CONFIG = dict(epsilon=.03, lam=.03, steps=5, rho=1/3)


class SpatialCodec(nn.Module):
    """The proxy encodes each column; decoding is exclusively a gain control signal."""
    def __init__(self, proxy=None):
        super().__init__(); self.proxy=proxy

    def encode(self, x):
        if self.proxy is None: return x.clamp_min(0)
        return self.proxy.encode(x.movedim(1,-1)).movedim(-1,1).contiguous()

    def gain(self, h):
        if self.proxy is None: return h
        return self.proxy.gain_signal(h.movedim(1,-1)).movedim(-1,1).contiguous()


def query(r, memory, codec, rho=1/3):
    z=codec.encode(r)
    h=(1-rho)*F.conv2d(z,memory.intra[:,:,None,None])+rho*F.conv2d(z,memory.inter,padding=1)
    q=codec.gain(h*(z>0))
    kappa=(z*h).sum(1,keepdim=True)/(z.square().sum(1,keepdim=True)+1e-12)
    return q,kappa,z


def side(x, memory, codec, calibration, epsilon=.03, lam=.03, steps=5, trace=False):
    """Read a fixed memory while replacing the activity, never the native representation."""
    if min(epsilon,lam)<0 or epsilon>=.5: raise ValueError('Invalid control strengths')
    r=x;records=[]
    for t in range(steps):
        q,k,z=query(r,memory,codec)
        u=torch.tanh(q/calibration['q_rms'])
        center=(x*u).sum(1,keepdim=True)/x.sum(1,keepdim=True).clamp_min(1e-12)
        m=epsilon*(u-center)
        mu=inhibition(r);sd=(inhibition(r.square())-mu.square()).clamp_min(0).sqrt()
        tau=lam*(mu+sd)*k/(k+calibration['kappa0'])
        r=(x*(1+m)-tau).clamp_min(0)
        if trace:
            records.append(dict(step=t+1,mean=float(r.mean()),threshold=float(tau.mean()),
                kappa=float(k.mean()),modulation_rms=float(m.square().mean().sqrt()),
                preclip_balance=float((x*m).sum()/x.sum().clamp_min(1e-12)),
                balance_max_abs=float((x*m).sum(1).abs().max()),
                gain_min=float((1+m).min()),gain_max=float((1+m).max()),
                density=float((r>1e-7).float().mean()),active_code=float((z>0).float().mean())))
    return r,records


def shuffle_memory(memory, permutation):
    if memory.phase!='target': raise RuntimeError('Shuffle only after preview freeze')
    p=permutation
    memory.intra.copy_(memory.intra[p][:,p])
    memory.trace_inter.copy_(memory.trace_inter[p][:,p])
    memory.inter.copy_(memory.inter[p][:,p])


class TransferOperator(nn.Module):
    def __init__(self,memory,codec,calibration,scale=1.,source='rep4',method='full',config=None):
        super().__init__()
        if source not in ('none','rep4','prd5') or method not in ('clip','full','pool','shuffle','scale'):
            raise ValueError((source,method))
        self.memory=memory;self.codec=codec;self.calibration=calibration
        self.scale=float(scale);self.source=source;self.method=method
        self.config=dict(CONFIG if config is None else config)
        self.reset()

    def reset(self,profile=None):
        self.calls=0;self.last_trace=[];self.input_activity=None
        self.output_means=[];self.profile=profile;self.trace_enabled=False

    def forward(self,a):
        with torch.no_grad():
            native=a.detach().clamp_min(0);x=native/self.scale
            self.input_activity=native
            if self.method=='scale':
                if self.profile is None: raise RuntimeError('Full-method mean profile required')
                desired=float(self.profile[self.calls])
                out=native*(desired/native.mean().clamp_min(1e-12))
                if desired>0 and not native.count_nonzero(): raise RuntimeError('Cannot match positive mean to zero map')
                self.last_trace=[]
            elif self.method=='clip' or self.memory.writes==0:
                # Exact native identity in the absence of memory; no divide/multiply roundoff.
                out=native;self.last_trace=[]
            else:
                q=self.config
                r,self.last_trace=side(x,self.memory,self.codec,self.calibration,
                    0. if self.method=='pool' else q['epsilon'],q['lam'],q['steps'],self.trace_enabled)
                out=r*self.scale
            self.calls+=1;self.output_means.append(float(out.mean()))
            return out.requires_grad_(True)

    def write(self,activity):
        # Both sources use the same codec and rep4 scale. No proxy reconstruction.
        return self.memory.write(self.codec.encode(activity.detach().clamp_min(0)/self.scale))


def new_memory(channels,device='cpu'):
    return PreviewMemory(channels,device,dict(DEFAULTS))
