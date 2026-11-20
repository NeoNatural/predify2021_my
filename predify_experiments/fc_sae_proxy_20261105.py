"""Frozen auxiliary sparse autoencoder; never installs hooks or changes the backbone."""
from __future__ import annotations
import hashlib
import torch
from torch import nn

CONFIG=dict(width=512,l1=.1,steps=1000,learning_rate=.001,minimum_learning_rate=.0001,
    optimizer='Adam',seed=20261105,role_weight_target=.5,role_weight_preview=.5)

class SparseProxy(nn.Module):
    def __init__(self,dimension,width,center,scale,seed=20261105):
        super().__init__();gen=torch.Generator().manual_seed(seed)
        self.register_buffer('center',torch.as_tensor(center,dtype=torch.float32).clone())
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32).clone())
        self.decoder=nn.Parameter(torch.randn(dimension,width,generator=gen))
        with torch.no_grad():self.decoder.div_(self.decoder.norm(dim=0,keepdim=True))
        self.encoder=nn.Linear(dimension,width)
        with torch.no_grad():self.encoder.weight.copy_(self.decoder.T);self.encoder.bias.zero_()

    def encode(self,x):return torch.relu(self.encoder(x/self.scale-self.center))
    def decode(self,z):return self.scale*torch.nn.functional.linear(z,self.decoder,self.center)
    def forward(self,x):
        z=self.encode(x);return self.decode(z),z

    def zero_edit(self,x):
        # Compute the two decodes, then their difference. Do not substitute a
        # direct identity return; this exercises the proposed residual pathway.
        z=self.encode(x);base=self.decode(z);edited=self.decode(z.clone())
        delta=edited-base
        return torch.relu(x+delta),delta

    @torch.no_grad()
    def constrain_dictionary(self):self.decoder.div_(self.decoder.norm(dim=0,keepdim=True).clamp_min(1e-12))

    @torch.no_grad()
    def project_dictionary_gradient(self):
        if self.decoder.grad is not None:
            self.decoder.grad.sub_(self.decoder*(self.decoder*self.decoder.grad).sum(dim=0,keepdim=True))

def fingerprint(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode());a=value.detach().cpu().contiguous().numpy();h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()

def training_statistics(x,weights):
    import numpy as np
    a=np.asarray(x,np.float64);w=np.asarray(weights,np.float64)
    assert np.isfinite(a).all() and (a>=0).all() and (w>0).all() and abs(w.sum()-1)<1e-10
    scale=float(np.sqrt(np.sum(w*np.mean(a*a,axis=1))))
    assert scale>0
    center=np.average(a/scale,axis=0,weights=w)
    return center.astype(np.float32),scale

def objective(model,x,weights,l1):
    hat,z=model(x)
    per_error=.5*((hat-x)/model.scale).square().sum(-1)
    per_l1=z.sum(-1)
    return (weights*(per_error+l1*per_l1)).sum(),dict(reconstruction=(weights*per_error).sum(),
        sparsity=(weights*per_l1).sum(),active=(weights*(z>0).float().mean(-1)).sum())
