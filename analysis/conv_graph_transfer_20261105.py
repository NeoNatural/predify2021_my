"""Transfer prd5-learned lateral graphs to rep4; six matched conditions."""
import sys,json,hashlib,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from analysis import conv_development_20261104 as d
from analysis import evaluate_conv_development_20261104 as ev
OUT=Path('E:/Predify_Experiments_20261105/conv_graph_transfer_20261105')
DOC=ROOT/'docs/conv_graph_transfer_20261105'
CASES={'gate_rep':'c066','gate_none':'c074','multiply_rep':'c067','multiply_none':'c001'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**kw):DOC.mkdir(exist_ok=True,parents=True);d.c.writejson(DOC/'status.json',kw)
def run():
 assert OUT.parent.is_dir(),'SSD missing; refuse fallback'
 OUT.mkdir(exist_ok=True)
 for ds in ('imagenet','things'):
  status(stage='preparing',dataset=ds,complete=False);m,train,dev=d.split(ds);dest=OUT/ds;dest.mkdir(exist_ok=True)
  old=d.OUT/ds;oldmanifest=json.loads((old/'manifest.json').read_text());assert oldmanifest['train']==train and oldmanifest['development']==dev
  graphs={s:torch.load(old/s/'graph.pt',map_location='cpu',weights_only=False) for s in ('rep4','prd5')}
  manifest=dict(dataset=ds,train=train,development=dev,fc_hebb=False,old_hebb=False,application='rep4',application_scale=float(graphs['rep4']['scale']),gain=.25,rho=1/3,lam=.5,alpha=.5,steps=3,gate_a=.25,gate_b=.1,source_hashes={str(p):sha(p) for p in [Path(__file__),ROOT/'analysis/conv_development_20261104.py',ROOT/'analysis/conv_development_core_20261104.py',old/'rep4/graph.pt',old/'prd5/graph.pt']},reused=CASES)
  mp=dest/'manifest.json'
  if mp.exists():assert json.loads(mp.read_text())==manifest
  else:d.c.writejson(mp,manifest)
  # Reuse exactly matched previous trajectories, without re-running or changing their data.
  for case,cid in CASES.items():
   q=dest/case;q.mkdir(exist_ok=True);src=old/cid;assert (src/'complete.json').exists()
   cfg=json.loads((src/'config.json').read_text());assert cfg['lam']==.5
   if 'none' not in case:assert cfg['gain']==.25 and abs(cfg['rho']-1/3)<1e-12
   if case.startswith('gate'):assert cfg['a']==.25 and cfg['b']==.1 and cfg['family']=='gate'
   for f in ('metrics.csv','priming_contrasts.csv'):shutil.copy2(src/f,q/f)
   d.c.writejson(q/'provenance.json',dict(raw_source=str(src.resolve()),source_config=cfg,metrics_sha=sha(src/'metrics.csv')))
  stats={}
  for name,g in graphs.items():
   stats[name]=dict(intra_edges=int((g['e0']>0).sum()),inter_edges=int((g['el']>0).sum()),intra_max_rowsum=float(g['e0'].sum(1).max()),inter_max_rowsum=float(g['el'].sum((1,2,3)).max()),training_scale=float(g['scale']))
  for key in ('e0','el'):
   a,b=graphs['rep4'][key],graphs['prd5'][key];stats[key+'_positive_edge_jaccard']=float(((a>0)&(b>0)).sum()/((a>0)|(b>0)).sum())
  d.c.writejson(dest/'graph_comparison.json',stats)
  p4,p5=d.initialize();c=d.c;original_p5=c.model.pcoder5.pmodule
  transform=c.VGG16_Weights.IMAGENET1K_V1.transforms();filt=c.native.FilteredInput(torch)
  for family in ('gate','multiply'):
   case=family+'_prd';q=dest/case;q.mkdir(exist_ok=True)
   if (q/'complete.json').exists() and (q/'priming_contrasts.csv').exists():continue
   g={k:v.cuda() for k,v in graphs['prd5'].items()};g['scale']=graphs['rep4']['scale'].cuda()
   cfg=dict(family=family,gain=.25,rho=1/3,lam=.5,a=.25 if family=='gate' else 0.,b=.1 if family=='gate' else 0.)
   op=d.Operator(g,cfg).cuda();p4.hebb=d.RepTransform(op)
   assert c.model.pcoder5.pmodule is original_p5
   assert g['scale'].item()==graphs['rep4']['scale'].item()
   d.c.writejson(q/'config.json',dict(**cfg,learn_from='prd5',apply_to='rep4',scale_from='rep4'))
   expected=[]
   for tr in dev:
    targets,pvs=c.test_inputs(ds,tr,transform,filt,torch)
    for seed,target in targets.items():
     for ident in ('exact','different_category'):
      f=q/f"{tr['dataset_index']}_{seed}_{ident}.npz";expected.append(f)
      if f.exists():continue
      status(stage='extracting',dataset=ds,case=case,target=tr['dataset_index'],seed=seed,identity=ident,complete=False)
      start=time.time();v=c.sequence(target,pvs[ident]);assert all(np.isfinite(a).all() for a in v.values())
      tmp=f.with_suffix('.tmp.npz');np.savez_compressed(tmp,**v,category=tr['category_index'],concept=tr['concept'],seconds=time.time()-start);tmp.replace(f)
      print(ds,case,tr['dataset_index'],seed,ident,flush=True)
   assert len(expected)==len(dev)*4 and all(f.exists() for f in expected)
   c.writejson(q/'complete.json',dict(complete=True,files=len(expected)))
   status(stage='evaluating',dataset=ds,case=case,complete=False)
   ev.OUT=OUT;sys.argv=['evaluate','--dataset',ds,'--candidate',case];ev.main()
 status(stage='extraction_evaluation_complete',complete=True)
if __name__=='__main__':run()
