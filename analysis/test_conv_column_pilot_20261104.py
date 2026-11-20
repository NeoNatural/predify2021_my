import ast,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=ROOT/'analysis/conv_column_pilot_20261104.py';tree=ast.parse(p.read_text(encoding='utf-8-sig'))
ns=dict(torch=torch,F=F,np=np,OFFSETS=[(y,x) for y in (-1,0,1) for x in (-1,0,1) if (y,x)!=(0,0)])
for node in tree.body:
 if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('shift','inhibition','Lateral'):exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
lat=ns['Lateral']();g=torch.load(ROOT/'Hebbian-CNN/Log/conv_column_nonnegative_20261104/imagenet/graph.pt',weights_only=False);lat.graph=g
x=torch.randn(2,512,5,5);results={}
lat.mode='original';assert lat(x) is x;results['original_identity']=True
lat.mode='clip';assert torch.equal(lat(x),x.clamp_min(0));results['clip_exact']=True
for mode in ['inhibition','excitation','intra','full','shuffle','matched','intra_budget']:
 lat.mode=mode;y=lat(x);yb=torch.cat([lat(a[None]) for a in x]);assert torch.allclose(y,yb,atol=2e-6,rtol=2e-5);assert y.min()>=0 and torch.isfinite(y).all();results[mode+'_batch_independent']=True
 r=x.clamp_min(0);dense=F.conv2d(r,g['el'],padding=1);direct=torch.zeros_like(r)
 for dy,dx in ns['OFFSETS']:direct+=torch.einsum('ij,bjhw->bihw',g['el'][:,:,dy+1,dx+1],ns['shift'](r,dy,dx))
 assert torch.allclose(dense,direct,atol=2e-6,rtol=2e-5)
results['masked_conv_equals_offset_sum']=True
lat.mode='full';y=lat(x);prd=F.conv2d(y,torch.randn(3,512,1,1));grad=torch.autograd.grad(prd.square().mean(),y)[0];assert torch.isfinite(grad).all();results['final_rep_error_gradient']=True
for steps in [1,3,5,20]:lat.steps=steps;assert torch.isfinite(lat(x)).all()
results['finite_1_3_5_20_steps']=True
out=ROOT/'docs/conv_column_nonnegative_20261104/implementation_acceptance.json';out.write_text(json.dumps(results,indent=2));print(results)
