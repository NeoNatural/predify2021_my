import sys,json,time,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.supplementary import load
OUT=ROOT/'Hebbian-CNN/Log/conv_column_nonnegative_20261104';DOC=ROOT/'docs/conv_column_nonnegative_20261104'
if '--wait' in sys.argv:
 while True:
  p=OUT/'status.json'
  try:
   if p.exists() and json.loads(p.read_text())['stage']=='extraction_complete':break
  except (ValueError,KeyError):pass
  time.sleep(15)
rows=[]
for dataset in ('imagenet','things'):
 dest=OUT/dataset
 if not (dest/'manifest.json').exists():continue
 m=json.loads((dest/'manifest.json').read_text());trials=m['test'];y=np.array([t['category_index'] for t in trials]);head=load(ROOT/'Hebbian-CNN/Log/native_categories_readouts_20261029'/dataset/'cv/reference_heads/fold0.pckl')['head']
 for seed in (0,2921):
  for ident in ('exact','same_concept','same_category','different_category'):
   for mode in m['modes']:
    paths=[dest/f"{t['dataset_index']}_{seed}_{ident}_{mode}.npz" for t in trials]
    if not all(p.exists() for p in paths):continue
    data=[np.load(p) for p in paths]
    for layer in ('conv4','conv5','fc1','fc2'):
     arrays=[d[layer] for d in data]
     for step in range(15):
      x=np.stack([a[step] for a in arrays]).astype(np.float64);norm=np.linalg.norm(x,axis=1);z=x/np.maximum(norm[:,None],1e-12)
      row=dict(dataset=dataset,seed=seed,identity=ident,mode=mode,layer=layer,t=step,mean=float(x.mean()),meanabs=float(np.abs(x).mean()),negative=float((x<0).mean()),zero_vectors=int((norm==0).sum()))
      for name,xx in [('raw',x),('normalized',z)]:
       global_mean=xx.mean(0);within=0.;between=0.
       for c in np.unique(y):
        a=xx[y==c];mu=a.mean(0);within+=np.square(a-mu).sum();between+=len(a)*np.square(mu-global_mean).sum()
       row[name+'_within']=within/len(y);row[name+'_between']=between/len(y);row[name+'_fisher']=between/max(within,1e-20)
      cs=[]
      for c in np.unique(y):
       zz=z[(y==c)&(norm>0)];n=len(zz)
       if n>1:cs.append((np.square(zz.sum(0)).sum()-n)/(n*(n-1)))
      row['within_cosine']=float(np.mean(cs)) if cs else float('nan')
      row['reference_accuracy']=float(np.mean(head.predict(x.astype(np.float32))==y)) if layer=='fc2' else ''
      rows.append(row)
    for d in data:d.close()
    print(dataset,seed,ident,mode,flush=True)
with (DOC/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
 if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(DOC/'analysis_status.json').write_text(json.dumps(dict(complete=True,metric_rows=len(rows),scope='full-map geometry and frozen reference accuracy; adapted heads and inferential statistics not run'),indent=2))
