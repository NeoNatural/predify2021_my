"""Evaluate completed development candidates using a training-only frozen reference head."""
import sys,json,csv,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.conv_development_20261104 import OUT,split,c
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['imagenet','things']);p.add_argument('--candidate',required=True);a=p.parse_args();m,train,dev=split(a.dataset);folder=OUT/a.dataset/a.candidate
 assert (folder/'complete.json').exists(),'Extraction incomplete'
 x=[];y=[]
 for tr in train:
  z=c.load(c.native.OUT/a.dataset/'formal'/f"reference_{tr['dataset_index']}.pckl");x.append(z['result']['fc2'][9]);y.append(tr['category_index'])
 # Never use historical fold0 head: its training includes the development fold4.
 head=make_pipeline(StandardScaler(),LinearSVC(C=1,dual='auto',max_iter=200000,random_state=20261104));head.fit(np.asarray(x),y)
 if head[-1].n_iter_>=200000:raise RuntimeError('Readout did not converge')
 rows=[];labels=np.array([t['category_index'] for t in dev]);features={}
 for seed in (0,2921):
  for ident in ('exact','different_category'):
   data=[]
   for tr in dev:
    with np.load(folder/f"{tr['dataset_index']}_{seed}_{ident}.npz") as z:data.append({k:z[k] for k in ('conv4','conv5','fc1','fc2')})
   for layer in ('conv4','conv5','fc1','fc2'):
    for t in range(15):
     v=np.array([d[layer][t] for d in data],dtype=np.float64);norm=np.linalg.norm(v,axis=1);z=v/np.maximum(norm[:,None],1e-12);mu=z.mean(0);w=b=0.
     for cat in np.unique(labels):
      q=z[labels==cat];cm=q.mean(0);w+=np.square(q-cm).sum();b+=len(q)*np.square(cm-mu).sum()
     row=dict(seed=seed,identity=ident,layer=layer,t=t,mean=float(v.mean()),zero_fraction=float((v==0).mean()),fisher=b/max(w,1e-20),within=w/len(v),between=b/len(v),zero_vectors=int((norm==0).sum()),accuracy=float(np.mean(head.predict(v.astype(np.float32))==labels)) if layer=='fc2' else '')
     rows.append(row)
 def write(p,rr):
  with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 write(folder/'metrics.csv',rows);contrasts=[]
 for seed in (0,2921):
  for layer in ('conv4','conv5','fc1','fc2'):
   for window,ts in [('early',range(5)),('late',range(10,15))]:
    pair={i:[r for r in rows if r['seed']==seed and r['layer']==layer and r['identity']==i and r['t'] in ts] for i in ('exact','different_category')}
    row=dict(seed=seed,layer=layer,window=window)
    for metric in ('mean','zero_fraction','fisher','within','between')+ (('accuracy',) if layer=='fc2' else ()):
     row[metric+'_exact_minus_different']=float(np.mean([r[metric] for r in pair['exact']])-np.mean([r[metric] for r in pair['different_category']]))
    contrasts.append(row)
 # Layer-specific columns must share a schema.
 for r in contrasts:r.setdefault('accuracy_exact_minus_different','')
 write(folder/'priming_contrasts.csv',contrasts)
 print('Saved metrics and contrasts; compare same seed/layer/window against c000 (clip) and matched inhibition candidates.')
if __name__=='__main__':main()
