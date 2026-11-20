"""Nonnegative PCoder4 column pilot; isolated outputs and resumable per-trial files."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,json,time,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from predify_experiments import native_categories_20261029 as native
from predify_experiments.supplementary import load
from predify_experiments.tripartite_20261027 import restore_state,set_init,gray_image
from predify_experiments.sequence import run_recurrent_sequence
from analysis.hebb_fc_native_pilot_20261102 import test_inputs
from torchvision.models import VGG16_Weights
OUT=ROOT/'Hebbian-CNN/Log/conv_column_nonnegative_20261104';OUT.mkdir(parents=True,exist_ok=True)
DOC=ROOT/'docs/conv_column_nonnegative_20261104';DOC.mkdir(parents=True,exist_ok=True)
MODES=['original','clip','inhibition','excitation','intra','full','shuffle','matched','intra_budget']
OFFSETS=[(y,x) for y in (-1,0,1) for x in (-1,0,1) if (y,x)!=(0,0)]
def writejson(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf8')
def shift(x,dy,dx):
 h,w=x.shape[-2:];return F.pad(x,(1,1,1,1))[...,1+dy:1+dy+h,1+dx:1+dx+w]
def inhibition(r):
 col=r.mean(1,keepdim=True);s=sum(shift(col,*o) for o in OFFSETS);n=sum(shift(torch.ones_like(col),*o) for o in OFFSETS)
 return .5*col+.5*s/n.clamp_min(1)
class Lateral(torch.nn.Module):
 def __init__(self):super().__init__();self.mode='original';self.graph=None;self.calls=0;self.steps=3
 def forward(self,a):
  self.calls+=1
  if self.mode=='original':return a
  with torch.no_grad():
   x=a.detach().clamp_min(0)
   if self.mode=='clip':return x.requires_grad_(True)
   g=self.graph;x=x/g['scale'];r=x
   e0,el=g['e0'],g['el']
   if self.mode=='shuffle':e0,el=g['s0'],g['sl']
   for _ in range(self.steps):
    d=x
    if self.mode!='inhibition':
     gain=.375 if self.mode=='intra_budget' else .25
     d=d+gain*F.conv2d(r,e0[:,:,None,None])
     if self.mode not in ('intra','intra_budget'):
      # Dense masked 3x3 reference computes exactly the sparse graph.
      d=d+.125*F.conv2d(r,el,padding=1)
    den=1 if self.mode=='excitation' else 1+.5*inhibition(r)
    r=.5*r+.5*d/den
   if self.mode=='matched':r=x*(r.mean((1,2,3),keepdim=True)/x.mean((1,2,3),keepdim=True).clamp_min(1e-12))
   out=r*g['scale'];assert torch.isfinite(out).all() and out.min()>=0
  return out.requires_grad_(True)
def setup():
 t,model,layers,rec,_=native.setup_model();rec.close()
 for l in layers:l.compute_enabled=False;l.update_enabled=False
 from predify_hebb_inject import HebbRepInjectPModule
 lat=Lateral().cuda();pc=model.pcoder4;pc.pmodule=HebbRepInjectPModule(pc.pmodule.base_pmodule,lat,pc)
 saved=load(native.OUT/'gray_snapshot.pckl');common=[{k:v.cuda() for k,v in item.items()} for item in saved['state']]
 return model,layers,lat,common
model,layers,lat,common=setup();grey=gray_image(torch);filt=native.FilteredInput(torch);transform=VGG16_Weights.IMAGENET1K_V1.transforms();cache={}
handles=[layers[i].register_forward_hook(lambda m,inp,out,k=k:cache.__setitem__(k,out.detach().clone())) for i,k in [(1,'fc1'),(0,'fc2')]]
def reset():
 model.reset();set_init(model,False);restore_state(model,common);model.input_mem=grey[None]
def sequence(target,pvs,collect=True):
 reset();features={k:[] for k in ('conv4','conv5','fc1','fc2')};before=lat.calls
 def after(t,out):
  if collect:
   for k,a in [('conv4',model.pcoder4.rep),('conv5',model.pcoder5.rep),('fc1',cache['fc1']),('fc2',cache['fc2'])]:features[k].append(a.detach().cpu().numpy().reshape(-1).copy())
 with torch.no_grad():
  for pv in pvs:run_recurrent_sequence(model,pv[None],2,hebbian_layers=layers,commit_update=False,preserve_model_state=True)
  run_recurrent_sequence(model,target[None],15,hebbian_layers=layers,commit_update=False,preserve_model_state=True,after_step=after)
 assert lat.calls-before==15+2*len(pvs)
 assert all(not l.compute_enabled and not l.update_enabled for l in layers)
 return {k:np.stack(v) for k,v in features.items()} if collect else None
def fit_graph(dataset,train,dest):
 path=dest/'graph.pt'
 if path.exists():return torch.load(path,map_location='cuda',weights_only=False)
 lat.mode='clip';acts=[];start=time.time()
 for j,t in enumerate(train):
  with np.load(native.OUT/dataset/'formal'/('stimuli_%s.npz'%t['dataset_index'])) as z:target=torch.from_numpy(z['target_0']).cuda()
  reset()
  with torch.no_grad():run_recurrent_sequence(model,target[None],10,hebbian_layers=layers,commit_update=False,preserve_model_state=True)
  acts.append(model.pcoder4.rep.detach().cpu())
  if j%32==0:print('FIT',dataset,j,len(train),flush=True)
 a=torch.cat(acts).cuda();scale=a.square().mean().sqrt().clamp_min(1e-8);a=a/scale
 # Position-wise centering across training images; no evaluation data.
 a=a-a.mean(0,keepdim=True);v=a.square().mean((0,2,3));delta=(.1*v[v>0].median()).clamp_min(1e-6)
 def corr(dy,dx):
  h,w=a.shape[-2:];ys=slice(max(0,-dy),min(h,h-dy));xs=slice(max(0,-dx),min(w,w-dx));yt=slice(max(0,dy),min(h,h+dy));xt=slice(max(0,dx),min(w,w+dx))
  aa=a[:,:,ys,xs].permute(1,0,2,3).reshape(512,-1);bb=a[:,:,yt,xt].permute(1,0,2,3).reshape(512,-1)
  return ((aa@bb.T/aa.shape[1])/torch.sqrt((aa.square().mean(1)+delta)[:,None]*(bb.square().mean(1)+delta)[None,:])).clamp_min(0)
 e0=corr(0,0);e0=(e0+e0.T)/2;e0.fill_diagonal_(0);e0/=e0.sum(1).max().clamp_min(1e-8)
 scores=torch.stack([corr(*o)*np.exp(-sum(v*v for v in o)/2) for o in OFFSETS],-1)
 mask=torch.zeros_like(scores.reshape(512,-1),dtype=torch.bool);mask.scatter_(1,scores.reshape(512,-1).topk(32,dim=1).indices,True);mask=mask.reshape(512,512,8)
 weights=torch.zeros_like(scores)
 for j,o in enumerate(OFFSETS):
  rev=OFFSETS.index((-o[0],-o[1]));weights[:,:,j]=scores[:,:,j]*mask[:,:,j]*mask[:,:,rev].T
 weights/=weights.sum((1,2)).max().clamp_min(1e-8)
 el=torch.zeros(512,512,3,3,device='cuda')
 for j,(dy,dx) in enumerate(OFFSETS):el[:,:,dy+1,dx+1]=weights[:,:,j]
 assert torch.allclose(e0,e0.T,atol=1e-6)
 for dy,dx in OFFSETS:assert torch.allclose(el[:,:,dy+1,dx+1],el[:,:,1-dy,1-dx].T,atol=1e-6)
 perm=torch.randperm(512,generator=torch.Generator().manual_seed(20261104)).cuda()
 g=dict(e0=e0,el=el,s0=e0[perm][:,perm],sl=el[perm][:,perm],scale=scale)
 torch.save({k:v.cpu() for k,v in g.items()},path);writejson(dest/'graph_info.json',dict(train_count=len(train),seconds=time.time()-start,scale=scale.item(),intra_nonzero=int((e0>0).sum()),inter_nonzero=int((el>0).sum()),max_inter_degree=int((el>0).sum((1,2,3)).max()),source='clean no-preview clip-only PCoder4 t9; train folds1-4'))
 return g
for dataset in ('imagenet','things'):
 dest=OUT/dataset;dest.mkdir(exist_ok=True);m=json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'));train=[t for t in m['trials'] if t['fold']!=0];test=[]
 for c in range(len(m['categories'])):
  pool=[t for t in m['trials'] if t['fold']==0 and t['category_index']==c]
  for concept in sorted({t['concept'] for t in pool})[:2]:test+=sorted([t for t in pool if t['concept']==concept],key=lambda t:t['dataset_index'])[:2]
 train_concepts={t['concept'] for t in train};train_bytes={m['images'][str(t['dataset_index'])]['sha256'] for t in train}
 for t in test:
  for idx in [t['dataset_index'],*t['previews'],*t['primes'].values()]:
   assert m['images'][str(idx)]['concept'] not in train_concepts
   assert m['images'][str(idx)]['sha256'] not in train_bytes
 writejson(dest/'manifest.json',dict(dataset=dataset,train=train,test=test,modes=MODES,fc_hebb=False,old_conv_hebb=False,steps=3,source_sha=hashlib.sha256((native.OUT/dataset/'manifest.json').read_bytes()).hexdigest()))
 lat.graph=fit_graph(dataset,train,dest)
 for trial in test:
  targets,pvs=test_inputs(dataset,trial,transform,filt,torch)
  for seed,target in targets.items():
   for ident,pv in pvs.items():
    for mode in MODES:
     path=dest/f"{trial['dataset_index']}_{seed}_{ident}_{mode}.npz"
     if path.exists():continue
     lat.mode=mode;torch.cuda.reset_peak_memory_stats();start=time.time();values=sequence(target,pv)
     # Full CHW float32 retained for exact geometry; compression is lossless.
     np.savez_compressed(path,**values,category=trial['category_index'],concept=trial['concept'],seconds=time.time()-start,peak_bytes=torch.cuda.max_memory_allocated())
     writejson(OUT/'status.json',dict(stage='pilot',dataset=dataset,target=trial['dataset_index'],seed=seed,identity=ident,mode=mode,files=len(list(dest.glob('*.npz')))))
     print('PILOT',dataset,trial['dataset_index'],seed,ident,mode,flush=True)
writejson(OUT/'status.json',dict(stage='extraction_complete',complete=True))
