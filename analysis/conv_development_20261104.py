"""Mechanism diagnosis and inner-fold coarse search. No extraction on import/default CLI."""
import sys,argparse,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from analysis import conv_development_core_20261104 as c
OUT=ROOT/'Hebbian-CNN/Log/conv_development_20261104'
def candidates():
 rows=[]
 for site in ('rep4','prd5'):
  for gain in (0,.125,.25,.375):
   for rho in ((0,) if gain==0 else (0,1/3)):
    for lam in (0,.5,1,2):rows.append(dict(site=site,family='add',gain=gain,rho=rho,lam=lam,a=0.,b=0.))
 for gain in (.125,.25):
  for lam in (.5,1):
   for a,b in ((.125,0),(.25,0),(.25,.1)):rows.append(dict(site='rep4',family='gate',gain=gain,rho=1/3,lam=lam,a=a,b=b))
   rows.append(dict(site='rep4',family='multiply',gain=gain,rho=1/3,lam=lam,a=0.,b=0.))
 # Gate-only controls distinguish threshold effects from learned excitation.
 for lam in (.5,1):
  for a,b in ((.125,0),(.25,0),(.25,.1)):rows.append(dict(site='rep4',family='gate',gain=0.,rho=0.,lam=lam,a=a,b=b))
 return [dict(id=f'c{i:03}',**r) for i,r in enumerate(rows)]
class Operator(torch.nn.Module):
 def __init__(self,g,cfg):super().__init__();self.g=g;self.cfg=cfg
 def forward(self,a):
  q=self.cfg;g=self.g;x=a.clamp_min(0)/g['scale'];r=x
  for _ in range(3):
   u=(1-q['rho'])*F.conv2d(r,g['e0'][:,:,None,None])+q['rho']*F.conv2d(r,g['el'],padding=1)
   d=x*(1+q['gain']*u/(1+u)) if q['family']=='multiply' else x+q['gain']*u
   out=.5*r+.5*d/(1+q['lam']*c.inhibition(r))
   if q['family']=='gate':
    mu=c.inhibition(d);sd=(c.inhibition(d.square())-mu.square()).clamp_min(0).sqrt();out=out*(d>q['a']*mu+q['b']*sd)
   r=out
  return r*g['scale']
class RepTransform(torch.nn.Module):
 def __init__(self,op):super().__init__();self.op=op
 def forward(self,a):
  with torch.no_grad():z=self.op(a)
  return z.requires_grad_(True)
class PredTransform(torch.nn.Module):
 def __init__(self,base,op):super().__init__();self.base=base;self.op=op
 def forward(self,a):return self.op(self.base(a)) # retain full error-gradient chain

def split(ds):
 m=json.loads((c.native.OUT/ds/'manifest.json').read_text(encoding='utf8'));train=[t for t in m['trials'] if t['fold'] in (1,2,3)];dev=[]
 for category in range(len(m['categories'])):
  pool=[t for t in m['trials'] if t['fold']==4 and t['category_index']==category]
  for concept in sorted({t['concept'] for t in pool})[:2]:dev+=sorted([t for t in pool if t['concept']==concept],key=lambda t:t['dataset_index'])[:2]
 roles=lambda ts:{i for t in ts for i in [t['dataset_index'],*t['previews'],*t['primes'].values()]}
 a,b=roles(train),roles(dev)
 for key in ('concept','sha256'):assert not {m['images'][str(i)][key] for i in a}&{m['images'][str(i)][key] for i in b},key
 return m,train,dev

def initialize():
 c.model,c.layers,c.lat,c.common=c.setup()
 for param in c.model.parameters():param.requires_grad_(False)
 c.grey=c.gray_image(torch);c.cache={}
 for i,k in [(1,'fc1'),(0,'fc2')]:c.layers[i].register_forward_hook(lambda m,inp,out,k=k:c.cache.__setitem__(k,out.detach().clone()))
 return c.model.pcoder4.pmodule,c.model.pcoder5.pmodule

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','diagnose','screen'],nargs='?',default='plan');p.add_argument('--dataset',choices=['imagenet','things'],default='imagenet');p.add_argument('--candidate',help='Explicit candidate ID required for screen');args=p.parse_args()
 configs=candidates();m,train,dev=split(args.dataset)
 if args.action=='plan':print(json.dumps(dict(candidates=configs,training=len(train),development=len(dev),held_out_fold0_unused=True),indent=2));return
 if args.action=='screen' and args.candidate not in {q['id'] for q in configs}:p.error('screen requires a valid --candidate cNNN')
 cfg=next((q for q in configs if q['id']==args.candidate),None);site=cfg['site'] if cfg else 'rep4';dest=OUT/args.dataset;dest.mkdir(parents=True,exist_ok=True)
 # Freeze training/development roles and source before any expensive work.
 spec=dict(train=train,development=dev,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fc_hebb=False,old_hebb=False)
 manifest=dest/'manifest.json'
 if manifest.exists():assert json.loads(manifest.read_text())==spec,'Source/split changed; use new output directory'
 else:c.writejson(manifest,spec)
 p4,p5=initialize();c.GRAPH_SITE=site;c.lat.mode='clip'
 graphdir=dest/site;graphdir.mkdir(exist_ok=True)
 if cfg:
  graph=c.fit_graph(args.dataset,train,graphdir);op=Operator(graph,cfg).cuda()
  if site=='rep4':p4.hebb=RepTransform(op)
  else:c.model.pcoder5.pmodule=PredTransform(p5,op)
 name=cfg['id'] if cfg else 'diagnose';folder=dest/name;folder.mkdir(exist_ok=True);c.writejson(folder/'config.json',cfg or dict(site='rep4',family='clip',diagnostic=True))
 transform=c.VGG16_Weights.IMAGENET1K_V1.transforms();filt=c.native.FilteredInput(torch)
 # Diagnostic stats include prediction states; raw original PCoder is retained for diagnosis.
 if args.action=='diagnose':c.lat.mode='original'
 for tr in dev:
  targets,pvs=c.test_inputs(args.dataset,tr,transform,filt,torch)
  for seed,target in targets.items():
   for ident in ('exact','different_category'):
    path=folder/f"{tr['dataset_index']}_{seed}_{ident}.npz"
    if path.exists():continue
    stats=[]
    def hook(mod,inp,out):
     values={}
     for k in (4,5):
      pc=getattr(c.model,f'pcoder{k}')
      for field in ('rep','prd'):
       a=getattr(pc,field).detach();s=a.abs().square().mean().sqrt().clamp_min(1e-12)
       values[f'{field}{k}']=[float(a.mean()),float(a.min()),float((a==0).float().mean()),float((a.abs()>1e-3*s).float().mean()),float(pc.prediction_error),*torch.quantile(a.flatten().float(),torch.tensor([.001,.01,.5,.99],device=a.device)).tolist()]
     stats.append(values)
    h=c.model.register_forward_hook(hook);start=time.time()
    try:v=c.sequence(target,pvs[ident])
    finally:h.remove()
    assert all(np.isfinite(a).all() for a in v.values())
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**v,diagnostics=json.dumps(stats),category=tr['category_index'],concept=tr['concept'],seconds=time.time()-start);tmp.replace(path)
    print(name,tr['dataset_index'],seed,ident,flush=True)
 c.writejson(folder/'complete.json',dict(complete=True,expected=len(dev)*4,files=len(list(folder.glob('*.npz')))))
if __name__=='__main__':main()
