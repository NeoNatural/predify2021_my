"""Fit outer-fold-only readouts on fold-specific Hebb-FC representations."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from pathlib import Path
import argparse,json,sys,time,warnings
import numpy as np
from joblib import Parallel,delayed
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.hebb_fc_full_20261103 import OUT,atomic_json
from analysis.hebb_fc_full_replay_20261103 import METHODS
from predify_experiments import native_categories_20261029 as native

def fit_svm(x,y,weights):
    """Identical estimator and retries; threads capped by process environment.

    Avoid old threadpoolctl's Windows OpenBLAS DLL introspection, which can
    receive a null version string after PyTorch has loaded its libraries.
    """
    assert x.dtype==np.float32 and set(y)==set(range(int(np.max(y))+1))
    attempts=[]
    for cap in (50000,200000):
        head=make_pipeline(StandardScaler(),LinearSVC(C=1.,dual='auto',max_iter=cap,random_state=native.SEED))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            head.fit(x,y,standardscaler__sample_weight=weights,linearsvc__sample_weight=weights)
        ok=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        attempts.append(dict(cap=cap,n_iter=int(head[-1].n_iter_),converged=ok))
        if ok:return head,attempts
    raise RuntimeError('Nonconvergence: stop without changing C')

def decode(dataset,fold,workers=8):
    dest=OUT/dataset/'folds'/f'fold{fold}'
    marker=json.loads((dest/'replay_complete.json').read_text(encoding='utf8'))
    m=json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'))
    trials=m['trials'];n=len(trials);k=len(m['categories'])
    y0=np.array([t['category_index'] for t in trials]);f0=np.array([t['fold'] for t in trials])
    tr0=np.flatnonzero(f0!=fold);te0=np.flatnonzero(f0==fold)
    assert marker['test_rows']==te0.tolist() and set(y0[tr0])==set(y0[te0])==set(range(k))
    started=time.monotonic()
    for method in METHODS:
        complete=dest/f'{method}_decode_complete.json'
        if complete.exists():continue
        x=np.load(dest/f'{method}_fc2.npy',mmap_mode='r');ref=np.load(dest/f'{method}_reference_fc2.npy',mmap_mode='r')
        assert x.shape==(12,4,n,15,4096) and ref.shape==(n,15,4096)
        output=dest/'scores'/method;output.mkdir(parents=True,exist_ok=True)
        head,attempts=fit_svm(np.asarray(ref[tr0,9]),y0[tr0],np.ones(len(tr0)))
        def t9_ref(test):
            return head.decision_function(test.reshape(-1,4096)).reshape(-1,15,k)
        native.atomic_npz(output/'reference_self.npz',scores=t9_ref(np.asarray(ref[te0])),y=y0[te0],rows=te0)
        reference_attempts=attempts
        for noise,si in [('clean',[0]),('noisy',[1,2,3])]:
            y=np.tile(y0,len(si));f=np.tile(f0,len(si));tr=np.flatnonzero(f!=fold);te=np.flatnonzero(f==fold)
            weights=np.ones(len(y))/len(si)
            for carrier in native.CARRIERS:
                cis=[j for j,c in enumerate(native.conditions()) if c['carrier']==carrier]
                pending=[]
                for ci in cis:
                    condition=native.conditions()[ci]['name']
                    score_path=output/f'{noise}_{condition}.npz'
                    meta_path=output/f'{noise}_{condition}.json'
                    if score_path.exists() and meta_path.exists():
                        with np.load(score_path) as saved:
                            assert np.array_equal(saved['y'],y[te]) and np.array_equal(saved['rows'],te%n)
                            assert np.array_equal(saved['seeds'],np.array([native.NOISES[si[j//n]] for j in te]))
                            for kind in ('mvpa','t9_mode_specific','t9_clean_reference'):
                                assert saved[kind].shape==(len(te),15,k) and np.isfinite(saved[kind]).all()
                        saved_meta=json.loads(meta_path.read_text(encoding='utf8'))
                        assert len(saved_meta['mvpa_attempts'])==15 and all(v[-1]['converged'] for v in saved_meta['mvpa_attempts'])
                        assert saved_meta['t9_mode_attempts'][-1]['converged'] and saved_meta['t9_reference_attempts'][-1]['converged']
                        print('DECODE RESUME VERIFIED',dataset,fold,method,noise,condition,flush=True)
                    else:pending.append(ci)
                if not pending:continue
                train=np.concatenate([np.asarray(x[ci,si,:,9]).reshape(-1,4096)[tr] for ci in cis])
                a_head,a_attempts=fit_svm(train,np.tile(y[tr],4),np.tile(weights[tr]/4,4))
                del train
                for ci in pending:
                    condition=native.conditions()[ci]['name']
                    a=np.asarray(x[ci,si]).reshape(-1,15,4096)
                    def fit_time(t):
                        h,attempts=fit_svm(a[tr,t],y[tr],weights[tr])
                        return h.decision_function(a[te,t]),attempts
                    # Separate workers avoid concurrent threadpoolctl changes and
                    # keep each BLAS/liblinear fit single-threaded.
                    computed=Parallel(n_jobs=workers,prefer='processes',max_nbytes='2M')(delayed(fit_time)(t) for t in range(15))
                    mvpa=np.stack([z[0] for z in computed],axis=1)
                    target=a[te].reshape(-1,4096)
                    mode=a_head.decision_function(target).reshape(-1,15,k)
                    refscore=head.decision_function(target).reshape(-1,15,k)
                    assert all(np.isfinite(v).all() for v in (mvpa,mode,refscore))
                    native.atomic_npz(output/f'{noise}_{condition}.npz',mvpa=mvpa,t9_mode_specific=mode,t9_clean_reference=refscore,
                                      y=y[te],rows=te%n,seeds=np.array([native.NOISES[si[j//n]] for j in te]))
                    atomic_json(output/f'{noise}_{condition}.json',{'mvpa_attempts':[z[1] for z in computed],
                        't9_mode_attempts':a_attempts,'t9_reference_attempts':reference_attempts})
                    print('DECODE',dataset,fold,method,noise,condition,'elapsed',round(time.monotonic()-started,1),flush=True)
        atomic_json(complete,{'dataset':dataset,'fold':fold,'method':method,'conditions':24,'readouts':['mvpa','t9_mode_specific','t9_clean_reference'],
            'train_rows':tr0.tolist(),'test_rows':te0.tolist(),'workers':workers,'seconds':time.monotonic()-started})
    atomic_json(dest/'decode_complete.json',{'dataset':dataset,'fold':fold,'methods':list(METHODS),'complete':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=('imagenet','things'));p.add_argument('--fold',required=True,type=int,choices=range(5));p.add_argument('--workers',type=int,default=8);a=p.parse_args();decode(a.dataset,a.fold,a.workers)
