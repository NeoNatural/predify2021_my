"""Condition-specific diagonal temporal MVPA with concept-grouped out-of-fold scores."""
from pathlib import Path
import sys,json,csv,hashlib,warnings,time,os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel,delayed
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DOC=ROOT/'docs/sliding_mvpa_20261013'
OUT=ROOT/'Hebbian-CNN/Log/sliding_mvpa_20261013'
SEED=20261013

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def folds_for(labels,groups):
    rng=np.random.default_rng(SEED)
    assignment=np.full(len(labels),-1,int)
    # Groups have a single superclass. Balance each class separately.
    for cl in np.unique(labels):
        gs=np.unique(groups[labels==cl]);rng.shuffle(gs)
        gs=sorted(gs,key=lambda g:-np.sum(groups==g))
        counts=np.zeros(5,int)
        for g in gs:
            ix=np.flatnonzero(groups==g)
            assert np.all(labels[ix]==cl)
            f=int(rng.choice(np.flatnonzero(counts==counts.min())))
            assignment[ix]=f;counts[f]+=len(ix)
    folds=[(np.flatnonzero(assignment!=f),np.flatnonzero(assignment==f)) for f in range(5)]
    seen=np.zeros(len(labels),int)
    for train,test in folds:
        assert not set(groups[train])&set(groups[test])
        assert set(labels[train])==set(labels)
        assert set(labels[test])==set(labels)
        seen[test]+=1
    assert np.all(seen==1)
    return folds

def fit_time(condition,t,x,y,folds,dest):
    file=dest/f'{condition}_t{t:02d}.npz'
    if file.exists():
        with np.load(file) as z:
            assert z['scores'].shape==(len(y),len(np.unique(y))) and np.isfinite(z['scores']).all()
        return
    started=time.monotonic();scores=np.full((len(y),len(np.unique(y))),np.nan);iterations=[]
    with threadpool_limits(limits=1):
        for tr,te in folds:
            for cap in (50000,200000):
                model=make_pipeline(StandardScaler(),LinearSVC(C=1.,dual='auto',max_iter=cap,random_state=SEED))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    model.fit(x[tr,t].astype(np.float32),y[tr])
                converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
                if converged:break
            if not converged:raise RuntimeError(f'Unconverged {condition} t{t}')
            assert np.array_equal(model.classes_,np.arange(scores.shape[1]))
            scores[te]=model.decision_function(x[te,t].astype(np.float32))
            iterations.append(int(model[-1].n_iter_))
    assert np.isfinite(scores).all()
    np.savez_compressed(file,scores=scores,iterations=iterations,seconds=time.monotonic()-started)
    print(f'{dest.name} {condition} t{t} complete {time.monotonic()-started:.1f}s',flush=True)

def run(dataset):
    import importlib
    module=importlib.import_module('analysis.'+('readout_init_20261012' if dataset=='imagenet' else 'things_readout_20261012'))
    d=module.data('test');y=d['labels'];groups=d['synsets'];folds=folds_for(y,groups)
    dest=OUT/dataset;dest.mkdir(parents=True,exist_ok=True)
    spec=dict(dataset=dataset,source_sha=sha(__file__),manifest_sha=sha(module.OUT/'manifest.json'),
              frozen_readout_sha=sha(module.OUT/'readouts.pckl'),keys=d['keys'],labels=y.tolist(),groups=groups.tolist(),
              folds=[{'train':a.tolist(),'test':b.tolist()} for a,b in folds],
              conditions=sorted(d['features']),times=list(range(15)),C=1.,seed=SEED,
              scope='Exploratory reanalysis of previously inspected test cohort. Condition-specific diagonal CV, grouped by concept. No temporal generalization.')
    fp=dest/'specification.json';encoded=json.loads(json.dumps(spec))
    if fp.exists():assert json.loads(fp.read_text())==encoded,'Specification changed; use a new run directory'
    else:fp.write_text(json.dumps(spec,indent=2),encoding='utf8')
    (dest/'source.py').write_bytes(Path(__file__).read_bytes())
    print(dataset,'images',len(set(k[0] for k in d['keys'])),'concepts',len(set(groups)),flush=True)
    Parallel(n_jobs=4,prefer='threads')(delayed(fit_time)(c,t,x,y,folds,dest) for c,x in d['features'].items() for t in range(15))
    rows=[];foldrows=[];noise=[];preds={}
    for c in d['features']:
        preds[c]=np.empty((len(y),15),int)
        for t in range(15):
            with np.load(dest/f'{c}_t{t:02d}.npz') as z:pr=z['scores'].argmax(1)
            preds[c][:,t]=pr;correct=pr==y
            rows.append(dict(dataset=dataset,condition=c,timestep=t,total_calls=int(d['calls'][c][0,t]),accuracy_pct=100*float(correct.mean()),balanced_accuracy_pct=100*float(np.mean([correct[y==cl].mean() for cl in np.unique(y)]))))
            for i,(_,te) in enumerate(folds):foldrows.append(dict(dataset=dataset,condition=c,timestep=t,fold=i,images=len(set(d['keys'][j][0] for j in te)),accuracy_pct=100*float(correct[te].mean())))
            for seed in sorted(set(k[1] for k in d['keys'])):
                ix=[k[1]==seed for k in d['keys']];noise.append(dict(dataset=dataset,condition=c,timestep=t,noise=seed,accuracy_pct=100*float(correct[ix].mean())))
    write(DOC/f'{dataset}_timeline.csv',rows);write(DOC/f'{dataset}_folds.csv',foldrows);write(DOC/f'{dataset}_noise.csv',noise)
    effects=[]
    for c in d['features']:
        if c=='carry_exact':continue
        for t in range(15):effects.append(dict(dataset=dataset,control=c,timestep=t,exact_minus_control_pp=100*float(np.mean((preds['carry_exact'][:,t]==y).astype(float)-(preds[c][:,t]==y).astype(float)))))
    write(DOC/f'{dataset}_contrasts_descriptive.csv',effects)
    audit=dict(images=len(set(k[0] for k in d['keys'])),concepts=len(set(groups)),folds=5,timepoints=15,conditions=len(d['features']),fits=5*15*len(d['features']),no_concept_overlap=True,one_oof_prediction_per_row=True,scaler_fit_within_training_fold=True,all_svm_converged=True,temporal_generalization=False,source_sha=sha(__file__),fold_class_counts=[np.bincount(y[b]).tolist() for a,b in folds])
    (DOC/f'{dataset}_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print('COMPLETE',dataset,audit,flush=True)

if __name__=='__main__':
    DOC.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    for dataset in sys.argv[1:] or ['imagenet','things']:run(dataset)
