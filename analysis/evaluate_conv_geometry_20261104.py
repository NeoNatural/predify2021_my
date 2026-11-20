import sys,json,csv,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'Hebbian-CNN/Log/conv_column_nonnegative_20261104';D=ROOT/'docs/conv_column_nonnegative_20261104';rows=[]
torch.backends.cuda.matmul.allow_tf32=False
for ds in ('imagenet','things'):
 m=json.loads((O/ds/'manifest.json').read_text());trials=m['test'];y=torch.tensor([t['category_index'] for t in trials],device='cuda');n=len(trials)
 for seed in (0,2921):
  for ident in ('exact','same_concept','same_category','different_category'):
   for mode in ('clip','inhibition','full'):
    paths=[O/ds/f"{t['dataset_index']}_{seed}_{ident}_{mode}.npz" for t in trials]
    for layer in ('conv4','conv5','fc1','fc2'):
     arrays=[]
     for p in paths:
      with np.load(p) as z:arrays.append(z[layer])
     x=torch.from_numpy(np.stack(arrays)).cuda().transpose(0,1).contiguous();del arrays
     assert torch.isfinite(x).all()
     # FP64 Gram provides accurate, scale-invariant geometry without large centered arrays.
     g=torch.bmm(x.double(),x.double().transpose(1,2));diag=g.diagonal(dim1=1,dim2=2);norm=diag.sqrt();gn=g/(norm[:,:,None]*norm[:,None,:]).clamp_min(1e-30)
     out={};
     for name,gg in [('raw',g),('normalized',gn)]:
      tr=gg.diagonal(dim1=1,dim2=2).sum(1);pooled=torch.zeros(15,device='cuda',dtype=torch.float64)
      for c in y.unique():
       mask=y==c;pooled+=gg[:,mask][:,:,mask].sum((1,2))/mask.sum()
      within=(tr-pooled)/n;between=pooled/n-gg.sum((1,2))/(n*n)
      out[name+'_within']=within.cpu().numpy();out[name+'_between']=between.cpu().numpy();out[name+'_fisher']=(between/within.clamp_min(1e-20)).cpu().numpy()
     cos=[]
     for c in y.unique():
      mask=y==c;nc=int(mask.sum());sub=gn[:,mask][:,:,mask];cos.append((sub.sum((1,2))-sub.diagonal(dim1=1,dim2=2).sum(1))/(nc*(nc-1)))
     out['within_cosine']=torch.stack(cos).mean(0).cpu().numpy();out['mean']=x.mean((1,2)).cpu().numpy();out['zero_vectors']=(norm==0).sum(1).cpu().numpy();out['negative']=(x<0).float().mean((1,2)).cpu().numpy();out['zero_fraction']=(x==0).float().mean((1,2)).cpu().numpy()
     for t in range(15):rows.append(dict(dataset=ds,seed=seed,identity=ident,mode=mode,layer=layer,t=t,**{k:float(v[t]) for k,v in out.items()}))
     del x,g,gn
    with (D/'focused_geometry.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(ds,seed,ident,mode,len(rows),flush=True)
(D/'focused_analysis_status.json').write_text(json.dumps(dict(complete=True,rows=len(rows),expected=2880,method='FP64 GPU Gram; full CHW; three requested conditions')))
