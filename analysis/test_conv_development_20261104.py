import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from analysis.conv_development_20261104 import Operator,PredTransform,candidates,split
q=candidates();assert len(q)==78 and len({x['id'] for x in q})==78
for ds in ('imagenet','things'):
 m,tr,dev=split(ds);assert all(t['fold'] in (1,2,3) for t in tr) and all(t['fold']==4 for t in dev)
g=dict(scale=torch.tensor(1.),e0=(torch.ones(4,4)-torch.eye(4))/3,el=torch.ones(4,4,3,3)/36)
x=torch.rand(2,4,5,5);x[:,:,0,:]=0
for family in ('add','gate','multiply'):
 cfg=dict(family=family,gain=.25,rho=1/3,lam=.5,a=.25,b=.1)
 op=Operator(g,cfg);y=op(x);assert torch.isfinite(y).all() and (y>=0).all();assert torch.allclose(y,torch.cat([op(v[None]) for v in x]),atol=1e-6)
 if family=='multiply':assert (y[x==0]==0).all()
 if family=='gate':assert (y==0).any()
base=torch.nn.Sequential(torch.nn.Conv2d(4,4,1),torch.nn.ReLU());p=PredTransform(base,Operator(g,dict(family='add',gain=.25,rho=1/3,lam=.5,a=0,b=0)));v=x.clone().requires_grad_();loss=p(v).square().mean();grad=torch.autograd.grad(loss,v)[0];assert torch.isfinite(grad).all() and grad.abs().sum()>0
print('PASS: 78 candidates, role isolation, nonnegativity, batch independence, zero-support preservation, gate zeros, prd gradient chain. No model extraction executed.')
