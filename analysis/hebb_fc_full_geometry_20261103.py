"""Descriptive held-out category geometry within each outer-fold feature space."""
from pathlib import Path
import csv,json,sys,time,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.hebb_fc_full_20261103 import OUT,atomic_json
from analysis.hebb_fc_full_replay_20261103 import METHODS
from predify_experiments import native_categories_20261029 as native

def geometry(dataset,fold):
    dest=OUT/dataset/'folds'/f'fold{fold}';marker=json.loads((dest/'replay_complete.json').read_text(encoding='utf8'))
    m=json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'));trials=m['trials'];te=np.array(marker['test_rows'])
    y=np.array([trials[j]['category_index'] for j in te]);k=len(m['categories']);assert set(y)==set(range(k))
    off2=np.load(native.OUT/dataset/'features.npy',mmap_mode='r')
    transformed={method:(np.load(dest/f'{method}_test_fc1.npy',mmap_mode='r'),np.load(dest/f'{method}_fc2.npy',mmap_mode='r')) for method in METHODS}
    rows=[];start=time.monotonic()
    def calc(arr,method,layer,ci,si):
        assert arr.shape==(len(te),15,4096) and np.isfinite(arr).all()
        for t in range(15):
            X=arr[:,t].astype(np.float64);norm=np.linalg.norm(X,axis=1,keepdims=True);u=X/np.maximum(norm,1e-12)
            means=[];within=[]
            for c in range(k):
                z=u[y==c];center=z.mean(0);means.append(center);within.append(np.square(z-center).sum(1).mean())
            means=np.asarray(means);w=float(np.mean(within));between=float(np.square(means-means.mean(0)).sum(1).mean())
            rows.append(dict(dataset=dataset,fold=fold,method=method,layer=layer,condition=native.conditions()[ci]['name'],seed=native.NOISES[si],timestep=t,
                activation=float(X.mean()),density=float((X>1e-7).mean()),normalized_within=w,normalized_between=between,
                normalized_fisher=between/max(w,1e-12),n_images=len(te),n_concepts=len({trials[j]['concept'] for j in te})))
    for ci,c in enumerate(native.conditions()):
        for si,seed in enumerate(native.NOISES):
            base=np.empty((len(te),15,4096),np.float32);lookup={trials[j]['dataset_index']:p for p,j in enumerate(te)}
            for concept in sorted({trials[j]['concept'] for j in te}):
                with np.load(OUT/dataset/'base'/f"concept_{concept}_seed_{seed}_{c['name']}.npz") as z:
                    a=z['fc1'];ids=z['images']
                    for p,i in enumerate(ids):base[lookup[int(i)]]=a[p]
            calc(base,'off','fc1',ci,si);calc(np.asarray(off2[ci,si,te]),'off','fc2',ci,si)
            for method,(f1,f2) in transformed.items():
                calc(np.asarray(f1[ci,si]),method,'fc1',ci,si);calc(np.asarray(f2[ci,si,te]),method,'fc2',ci,si)
        print('GEOMETRY',dataset,fold,c['name'],'elapsed',round(time.monotonic()-start,1),flush=True)
    path=dest/'geometry.csv';tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tmp.replace(path)
    atomic_json(dest/'geometry_complete.json',{'dataset':dataset,'fold':fold,'rows':len(rows),
        'interpretation':'descriptive category geometry within held-out fold; includes multiple concepts per class; normalized feature vectors',
        'seconds':time.monotonic()-start})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=('imagenet','things'));p.add_argument('--fold',required=True,type=int,choices=range(5));a=p.parse_args();geometry(a.dataset,a.fold)
