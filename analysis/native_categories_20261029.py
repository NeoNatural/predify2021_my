"""CV readouts and full-map geometric statistics for the filtered protocol."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import argparse,csv,json,sys,time,warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from joblib import Parallel,delayed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.native_categories_20261029 import OUT,DOC,SEED,CATEGORIES,CARRIERS,IDENTITIES,LAYERS,NOISES,METRICS,conditions,atomic_npz,verify_freeze,geometry_value
from predify_experiments.supplementary import load,save
from predify_experiments.readout_init_20261012 import immutable_json,sha
WINDOWS=dict(early=list(range(5)),middle=list(range(5,10)),late=list(range(10,15)),full=list(range(15)),t9=[9])

def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def dataset_data(dataset):
    m=json.loads((OUT/dataset/'manifest.json').read_text(encoding='utf8'));trials=m['trials'];indices={t['dataset_index']:i for i,t in enumerate(trials)}
    n=len(trials)
    xp=OUT/dataset/'features.npy'
    x=np.lib.format.open_memmap(xp,mode='w+',dtype='float32',shape=(12,4,n,15,4096));x[:]=np.nan
    activation=np.full((12,4,n,15,4,7),np.nan,np.float64);legacy=activation.copy();noise=[];seen=set()
    for path in (OUT/dataset/'formal').glob('concept_*.pckl'):
        d=load(path);ci=[c['name'] for c in conditions()].index(d['condition']['name']);si=NOISES.index(d['seed'])
        key=d['concept'],si,ci;assert key not in seen;seen.add(key)
        for trial,r,ns in zip(d['trials'],d['results'],d['noise_stats']):
            j=indices[trial['dataset_index']];x[ci,si,j]=r['fc2'];activation[ci,si,j]=r['activation'];legacy[ci,si,j]=r['legacy_activation']
            assert r['target_W_frozen'] and r['fc2'].dtype==np.float32
            assert len(r['commits'])==(0 if d['condition']['carrier']=='pcoder' else 4)
            noise.append(dict(image=trial['dataset_index'],seed=d['seed'],condition=d['condition']['name'],target_hash=r['target_hash'],**ns))
    assert len(seen)==len(m['concept_folds'])*48 and all(np.isfinite(x[ci,si]).all() for ci in range(12) for si in range(4)) and np.isfinite(activation).all() and np.isfinite(legacy).all()
    for index in indices:
        for seed in NOISES:
            assert len({r['target_hash'] for r in noise if r['image']==index and r['seed']==seed})==1
    return dict(manifest=m,trials=trials,x=x,activation=activation,legacy=legacy,noise=noise)

def fit_svm(x,y,weights):
    assert x.dtype==np.float32 and set(y)==set(range(int(np.max(y))+1))
    attempts=[]
    with threadpool_limits(limits=1):
        for cap in (50000,200000):
            head=make_pipeline(StandardScaler(),LinearSVC(C=1.,dual='auto',max_iter=cap,random_state=SEED))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                head.fit(x,y,standardscaler__sample_weight=weights,linearsvc__sample_weight=weights)
            ok=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
            attempts.append(dict(cap=cap,n_iter=int(head[-1].n_iter_),converged=ok))
            if ok:return head,attempts
    raise RuntimeError('Nonconvergence: stop without changing C')

def auc_macro(y,scores):
    assert scores.shape==(len(y),len(CATEGORIES)) and set(y)==set(range(int(np.max(y))+1))
    vals=np.array([roc_auc_score(y==c,scores[:,c]) for c in range(len(CATEGORIES))])
    return float(vals.mean()),vals

def weighted_rows(x,si):
    # Noise first then image. Each target's total weight is one per condition.
    z=x[si].reshape(-1,15,4096).astype(np.float32)
    weights=np.ones(len(z),np.float64)/len(si)
    return z,weights

def cv_fit(dataset,noise,kind,condition,x,y,fold,weights):
    dest=OUT/dataset/'cv'/noise/kind;path=dest/f'{condition}.npz'
    if path.exists():
        with np.load(path) as z:assert z['scores'].shape==(len(y),15,len(CATEGORIES))
        return
    scores=np.full((len(y),15,len(CATEGORIES)),np.nan);fitrows=[]
    for f in range(5):
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
        assert kind=='mvpa'
        for t in range(15):
            modelpath=dest/f'{condition}_t{t}_fold{f}.pckl'
            if modelpath.exists():saved=load(modelpath);head=saved['head'];attempts=saved['attempts']
            else:
                head,attempts=fit_svm(x[tr,t],y[tr],weights[tr])
                save(modelpath,dict(head=head,attempts=attempts,train_rows=tr,test_rows=te,sample_weight_sum=float(weights[tr].sum()),source_sha=sha(__file__)))
            scores[te,t]=head.decision_function(x[te,t]);fitrows.append(dict(fold=f,timestep=t,attempts=attempts))
    assert np.isfinite(scores).all()
    atomic_npz(path,scores=scores,y=y,fold=fold,weights=weights,predictions=scores.argmax(-1))
    immutable_json(dest/f'{condition}_fits.json',fitrows)
    print('CV COMPLETE',dataset,noise,kind,condition,flush=True)


def geometry_from_grams(graw,gnorm,sq,unorm,zeros,category,weights):
    """Exact bootstrap of concept-additive statistics via Gram matrices.

    weights[B,G] are concept multiplicities, each concept contributes four images.
    All native classes have an equal total weight. Outputs[B,T].
    """
    weights=np.atleast_2d(weights).astype(np.float64);b=len(weights);t=graw.shape[0];rawtrace=[];normtrace=[];cos=[];centroid_raw=[];centroid_norm=[]
    for c in range(len(CATEGORIES)):
        idx=np.flatnonzero(category==c);w=weights[:,idx];n=4*w.sum(1)
        assert np.all(n>=2)
        r=np.einsum('bi,tij,bj->bt',w,graw[:,idx][:,:,idx],w,optimize=True)
        u=np.einsum('bi,tij,bj->bt',w,gnorm[:,idx][:,:,idx],w,optimize=True)
        ss=w@sq[:,idx].T;uu=w@unorm[:,idx].T
        rawtrace.append(np.maximum(ss/n[:,None]-r/n[:,None]**2,0.))
        normtrace.append(np.maximum(uu/n[:,None]-u/n[:,None]**2,0.))
        cos.append((u-uu)/(n[:,None]*(n[:,None]-1)))
        wc=np.zeros_like(weights);wc[:,idx]=w/n[:,None];centroid_raw.append(wc);centroid_norm.append(wc)
    # between = mean class-centroid squared norm - grand-centroid squared norm
    raw_cent_norm=np.zeros((b,t));norm_cent_norm=np.zeros((b,t))
    for wc in centroid_raw:
        raw_cent_norm+=np.einsum('bi,tij,bj->bt',wc,graw,wc,optimize=True)/len(CATEGORIES)
        norm_cent_norm+=np.einsum('bi,tij,bj->bt',wc,gnorm,wc,optimize=True)/len(CATEGORIES)
    grand=sum(centroid_raw)/len(CATEGORIES)
    rawbetween=np.maximum(raw_cent_norm-np.einsum('bi,tij,bj->bt',grand,graw,grand,optimize=True),0.)
    normbetween=np.maximum(norm_cent_norm-np.einsum('bi,tij,bj->bt',grand,gnorm,grand,optimize=True),0.)
    rawwithin=np.mean(rawtrace,axis=0);normwithin=np.mean(normtrace,axis=0)
    return dict(raw_trace_variance=rawwithin,normalized_trace_variance=normwithin,within_class_cosine=np.mean(cos,axis=0),
                normalized_between_scatter=normbetween,normalized_fisher_ratio=normbetween/np.maximum(normwithin,1e-8),
                zero_vector_fraction=(weights@zeros.T)/(4*weights.sum(1)[:,None]))

def gram_cache(dataset,seed,condition,layer,concepts):
    path=OUT/dataset/'grams'/f'{condition}_{seed}_{layer}.npz'
    with np.load(path) as z:
        assert np.array_equal(z['concepts'],concepts)
        return {k:z[k] for k in ('graw','gnorm','sq','unorm','zeros')}


def relative(a,b):
    a,b=np.broadcast_arrays(a,b);out=np.full(a.shape,np.nan);mask=np.abs(b)>=1e-12;out[mask]=100*(a[mask]/b[mask]-1);return out


def t9_training_rows(x,y,fold,weights,f,carrier):
    """One target total weight one, distributed across four preview contents."""
    tr=np.flatnonzero(fold!=f)
    ci=[j for j,c in enumerate(conditions()) if c['carrier']==carrier]
    train=np.concatenate([x[j,tr,9] for j in ci]);labels=np.tile(y[tr],4);w=np.tile(weights[tr]/4,4)
    assert len(ci)==4 and np.isclose(w.sum(),weights[tr].sum())
    return train,labels,w,tr


def reference_data(dataset,trials):
    x=[]
    for t in trials:
        d=load(OUT/dataset/'formal'/f"reference_{t['dataset_index']}.pckl")
        assert d['trial']==t and d['result']['commits']==[] and d['result']['target_W_frozen']
        x.append(d['result']['fc2'])
    return np.stack(x)


def t9_decode(dataset,noise,x,y,fold,w,ref,y0,fold0):
    nclasses=len(CATEGORIES)
    # Fit B only on unique clean reference targets, once per dataset/fold.
    rb=OUT/dataset/'cv'/'reference_heads';rb.mkdir(parents=True,exist_ok=True)
    selfscores=np.full((len(y0),15,nclasses),np.nan)
    bheads={}
    for f in range(5):
        tr=np.flatnonzero(fold0!=f);te=np.flatnonzero(fold0==f);path=rb/f'fold{f}.pckl'
        if path.exists():head=load(path)['head']
        else:
            head,attempts=fit_svm(ref[tr,9],y0[tr],np.ones(len(tr)))
            save(path,dict(head=head,attempts=attempts,train_rows=tr,test_rows=te,sample_weight_sum=float(len(tr)),training='unique clean no-memory reference t9',reference_count=len(y0),source_sha=sha(__file__)))
        bheads[f]=head;selfscores[te]=head.decision_function(ref[te].reshape(-1,4096)).reshape(len(te),15,nclasses)
    atomic_npz(rb/'self_oof.npz',scores=selfscores,y=y0,fold=fold0,predictions=selfscores.argmax(-1))
    for carrier in CARRIERS:
        adest=OUT/dataset/'cv'/noise/'t9_mode_specific';adest.mkdir(parents=True,exist_ok=True)
        aheads={}
        for f in range(5):
            path=adest/f'{carrier}_fold{f}.pckl'
            if path.exists():head=load(path)['head']
            else:
                train,labels,weights,tr=t9_training_rows(x,y,fold,w,f,carrier)
                head,attempts=fit_svm(train,labels,weights)
                save(path,dict(head=head,attempts=attempts,train_rows=tr,test_rows=np.flatnonzero(fold==f),sample_weight_sum=float(weights.sum()),training='four contents at t9; one carrier/noise',source_sha=sha(__file__)))
                del train,labels,weights
            aheads[f]=head
        for ci,c in enumerate(conditions()):
            if c['carrier']!=carrier:continue
            for kind,heads in [('t9_mode_specific',aheads),('t9_clean_reference',bheads)]:
                dest=OUT/dataset/'cv'/noise/kind;path=dest/f"{c['name']}.npz"
                if path.exists():continue
                scores=np.full((len(y),15,nclasses),np.nan)
                for f,head in heads.items():
                    te=np.flatnonzero(fold==f)
                    scores[te]=head.decision_function(x[ci,te].reshape(-1,4096)).reshape(len(te),15,nclasses)
                assert np.isfinite(scores).all()
                atomic_npz(path,scores=scores,y=y,fold=fold,weights=w,predictions=scores.argmax(-1))


def decoding(dataset):
    global CATEGORIES
    verify_freeze();d=dataset_data(dataset);m=d['manifest'];CATEGORIES=tuple(m['categories'])
    y0=np.array([t['category_index'] for t in d['trials']]);fold0=np.array([t['fold'] for t in d['trials']])
    ref=reference_data(dataset,d['trials'])
    immutable_json(OUT/dataset/'folds.json',dict(concept_folds=m['concept_folds'],image_folds=[dict(image=t['dataset_index'],concept=t['concept'],fold=t['fold']) for t in d['trials']],A_content_weight=.25,A_noisy_row_weight=1/12,B_unique_targets=len(y0),noisy_realization_weight=1/3,fold_mean='arithmetic',pooled_auc='not computed across models'))
    for noise,si in [('clean',[0]),('noisy',[1,2,3])]:
        xs=[]
        for ci in range(12):z,w=weighted_rows(d['x'][ci],si);xs.append(z)
        x=np.stack(xs);del xs
        y=np.tile(y0,len(si));fold=np.tile(fold0,len(si))
        with threadpool_limits(limits=1):
            t9_decode(dataset,noise,x,y,fold,w,ref,y0,fold0)
        with threadpool_limits(limits=1):
            Parallel(n_jobs=2,prefer='threads')(delayed(cv_fit)(dataset,noise,'mvpa',c['name'],x[ci],y,fold,w) for ci,c in enumerate(conditions()))
        del x
    summarize_decoding(dataset,d)


def summarize_decoding(dataset,d):
    rows=[];summaries=[];contrasts=[];windows=[];integrals=[];pernoise=[];base=DOC/dataset;n=len(d['trials'])
    for noise,si in [('clean',[0]),('noisy',[1,2,3])]:
        for kind in ('mvpa','t9_mode_specific','t9_clean_reference'):
            metrics={}
            for c in conditions():
                name=c['name']
                with np.load(OUT/dataset/'cv'/noise/kind/f'{name}.npz') as z:scores=z['scores'];y=z['y'];fold=z['fold']
                scorepath=OUT/dataset/'cv'/noise/kind/f'{name}.npz'
                with np.load(scorepath) as z:values={key:z[key] for key in z.files}
                values.update(image=np.tile([t['dataset_index'] for t in d['trials']],len(si)),concept=np.tile([t['concept'] for t in d['trials']],len(si)),noise_seed=np.repeat([NOISES[i] for i in si],n))
                atomic_npz(scorepath,**values)
                local={}
                for t in range(15):
                    accs=[];aucs=[]
                    for f in range(5):
                        mask=fold==f;acc=100*float(np.mean(scores[mask,t].argmax(-1)==y[mask]));auc,classes=auc_macro(y[mask],scores[mask,t]);accs.append(acc);aucs.append(auc)
                        rows.append(dict(dataset=dataset,noise=noise,readout=kind,condition=name,fold=f,timestep=t,accuracy_pct=acc,roc_auc=auc,**{f'auc_{CATEGORIES[k]}':float(v) for k,v in enumerate(classes)},**{f'acc_{category}':100*float(np.mean(scores[mask&(y==k),t].argmax(-1)==k)) for k,category in enumerate(CATEGORIES)}))
                        for sn,seed in enumerate([NOISES[i] for i in si]):
                            sub=mask&(np.arange(len(y))//n==sn);aa,_=auc_macro(y[sub],scores[sub,t])
                            pernoise.append(dict(dataset=dataset,noise=noise,seed=seed,readout=kind,condition=name,fold=f,timestep=t,accuracy_pct=100*float(np.mean(scores[sub,t].argmax(-1)==y[sub])),roc_auc=aa))
                    local[t]=dict(accuracy=np.array(accs),roc_auc=np.array(aucs))
                    pooled=100*float(np.mean(scores[:,t].argmax(-1)==y));assert abs(pooled-np.mean(accs))<1e-10
                    summaries.append(dict(dataset=dataset,noise=noise,readout=kind,condition=name,timestep=t,accuracy_fold_mean=float(np.mean(accs)),accuracy_fold_sd=float(np.std(accs,ddof=1)),accuracy_oof_pooled=pooled,roc_auc_fold_mean=float(np.mean(aucs)),roc_auc_fold_sd=float(np.std(aucs,ddof=1))))
                metrics[name]=local
                for metric in ('accuracy','roc_auc'):
                    v=np.trapz(np.stack([local[t][metric] for t in range(15)]),dx=1,axis=0)/14
                    integrals.append(dict(dataset=dataset,noise=noise,readout=kind,condition=name,metric=f'temporal_integral_{metric}',mean=float(v.mean()),fold_sd=float(v.std(ddof=1))))
            for carrier in CARRIERS:
                baseline=metrics[f'{carrier}_different_category']
                for identity in IDENTITIES[:-1]:
                    prime=metrics[f'{carrier}_{identity}']
                    da=np.stack([prime[t]['accuracy']-baseline[t]['accuracy'] for t in range(15)])
                    du=np.stack([prime[t]['roc_auc']-baseline[t]['roc_auc'] for t in range(15)])
                    for t in range(15):contrasts.append(dict(dataset=dataset,noise=noise,readout=kind,carrier=carrier,identity=identity,timestep=t,delta_accuracy_pp=float(da[t].mean()),delta_accuracy_fold_sd=float(da[t].std(ddof=1)),delta_roc_auc=float(du[t].mean()),delta_auc_fold_sd=float(du[t].std(ddof=1))))
                    for window,ts in WINDOWS.items():windows.append(dict(dataset=dataset,noise=noise,readout=kind,carrier=carrier,identity=identity,window=window,delta_accuracy_pp=float(da[ts].mean()),delta_accuracy_fold_sd=float(da[ts].mean(0).std(ddof=1)),delta_roc_auc=float(du[ts].mean()),delta_auc_fold_sd=float(du[ts].mean(0).std(ddof=1))))
    for name,value in [('cv_fold_metrics.csv',rows),('cv_summary.csv',summaries),('cv_noise_metrics.csv',pernoise),('cv_content_contrasts.csv',contrasts),('cv_window_contrasts.csv',windows),('temporal_integrals.csv',integrals),('input_noise.csv',d['noise'])]:write(base/name,value)
    path=OUT/dataset/'cv/reference_heads/self_oof.npz'
    with np.load(path) as z:scores=z['scores'];y=z['y'];fold=z['fold'];values={key:z[key] for key in z.files}
    values.update(image=np.array([t['dataset_index'] for t in d['trials']]),concept=np.array([t['concept'] for t in d['trials']]),noise_seed=np.zeros(n,int))
    atomic_npz(path,**values);reference_rows=[]
    for t in range(15):
        for f in range(5):
            mask=fold==f;auc,_=auc_macro(y[mask],scores[mask,t]);reference_rows.append(dict(dataset=dataset,fold=f,timestep=t,accuracy_pct=100*float(np.mean(scores[mask,t].argmax(-1)==y[mask])),roc_auc=auc))
    write(base/'reference_self_timeline.csv',reference_rows)

def representation(dataset):
    global CATEGORIES
    CATEGORIES=tuple(json.loads((OUT/dataset/'manifest.json').read_text(encoding='utf8'))['categories'])
    verify_freeze();d=dataset_data(dataset);trials=d['trials'];concepts=sorted({t['concept'] for t in trials});category=np.array([next(t['category_index'] for t in trials if t['concept']==c) for c in concepts]);folds=np.array([next(t['fold'] for t in trials if t['concept']==c) for c in concepts])
    rng=np.random.default_rng(SEED);boot=np.zeros((10000,len(concepts)),np.float64)
    for c in range(len(CATEGORIES)):
        idx=np.flatnonzero(category==c)
        for j,draw in enumerate(rng.choice(idx,size=(10000,len(idx)),replace=True)):boot[j]=boot[j]+np.bincount(draw,minlength=len(concepts))
    weights=np.concatenate([np.ones((1,len(concepts))),boot],axis=0)
    # Persist multiplicities to make all metric/baseline bootstrap pairings auditable.
    atomic_npz(OUT/dataset/'bootstrap_weights.npz',weights=boot,concepts=np.array(concepts),categories=category)
    timeline=[];foldrows=[];effects=[];windoweffects=[];act_timeline=[]
    for layer in LAYERS:
        li=LAYERS.index(layer)
        for noise,seeds in [('clean',[0]),('noisy',list(NOISES[1:]))]:
            for carrier in CARRIERS:
                allgeom={};allact={}
                for identity in IDENTITIES:
                    condition=f'{carrier}_{identity}';ci=[c['name'] for c in conditions()].index(condition)
                    cache=OUT/dataset/'representation_cache'/f'{condition}_{noise}_{layer}.npz'
                    if cache.exists():
                        with np.load(cache) as z:res={k:z[k] for k in z.files}
                        foldrows.extend(json.loads(cache.with_suffix('.json').read_text(encoding='utf8')))
                    else:
                        accum=None;current_foldrows=[]
                        for seed in seeds:
                            g=gram_cache(dataset,seed,condition,layer,concepts)
                            with threadpool_limits(limits=4):v=geometry_from_grams(**g,category=category,weights=weights)
                            if accum is None:accum=v
                            else:
                                for key in accum:accum[key]+=v[key]
                            for f in range(5):
                                w=(folds==f).astype(float)[None,:];fv=geometry_from_grams(**g,category=category,weights=w)
                                for t in range(15):current_foldrows.append(dict(dataset=dataset,noise=noise,seed=seed,carrier=carrier,identity=identity,layer=layer,fold=f,timestep=t,**{k:float(a[0,t]) for k,a in fv.items()}))
                        res={k:v/len(seeds) for k,v in accum.items()};immutable_json(cache.with_suffix('.json'),current_foldrows);atomic_npz(cache,**res);foldrows.extend(current_foldrows)
                    allgeom[identity]=res
                    # Per-concept activation means first, preserving both images and all paired seeds.
                    act={}
                    for label,arr in [('output',d['activation']),('legacy_x_tmp',d['legacy'])]:
                        for k,metric in enumerate(METRICS):
                            perconcept=np.stack([arr[ci,[NOISES.index(s) for s in seeds]][:,[j for j,t in enumerate(trials) if t['concept']==c],:,li,k].mean((0,1)) for c in concepts])
                            # Four equal-size classes -> concept mean equals class-balanced mean.
                            act[f'{label}_{metric}']=weights@perconcept/weights.sum(1)[:,None]
                    allact[identity]=act
                    for t in range(15):
                        timeline.append(dict(dataset=dataset,noise=noise,carrier=carrier,identity=identity,layer=layer,timestep=t,**{k:float(v[0,t]) for k,v in res.items()}))
                        act_timeline.append(dict(dataset=dataset,noise=noise,carrier=carrier,identity=identity,layer=layer,timestep=t,**{k:float(v[0,t]) for k,v in act.items()}))
                for identity in IDENTITIES[:-1]:
                    a,b=allgeom[identity],allgeom['different_category'];p,q=allact[identity],allact['different_category']
                    e={f'{k}_delta':a[k]-b[k] for k in a}
                    e.update(normalized_trace_effect_pct=relative(a['normalized_trace_variance'],b['normalized_trace_variance']),
                             within_cosine_delta_x1e3=1000*(a['within_class_cosine']-b['within_class_cosine']),
                             normalized_fisher_effect_pct=relative(a['normalized_fisher_ratio'],b['normalized_fisher_ratio']))
                    for key in p:e[f'{key}_delta']=p[key]-q[key];e[f'{key}_effect_pct']=relative(p[key],q[key])
                    e['activation_effect_pct']=e['legacy_x_tmp_mean_effect_pct']
                    e['complete_sharpening']=((e['activation_effect_pct']<0)&(e['normalized_trace_effect_pct']<0)&(e['normalized_fisher_effect_pct']>0)).astype(float)
                    e['output_complete_sharpening']=((e['output_mean_effect_pct']<0)&(e['normalized_trace_effect_pct']<0)&(e['normalized_fisher_effect_pct']>0)).astype(float)
                    for metric,values in e.items():
                        for t in range(15):
                            lo,hi=np.nanquantile(values[1:,t],[.025,.975]);effects.append(dict(dataset=dataset,noise=noise,carrier=carrier,identity=identity,layer=layer,timestep=t,metric=metric,value=float(values[0,t]),low=float(lo),high=float(hi)))
                        for window,ts in WINDOWS.items():
                            # Window mean of timepoint effects; percent at a window is additionally explicitly named below.
                            v=values[:,ts].mean(1);lo,hi=np.nanquantile(v[1:],[.025,.975]);windoweffects.append(dict(dataset=dataset,noise=noise,carrier=carrier,identity=identity,layer=layer,window=window,metric=metric,value=float(v[0]),low=float(lo),high=float(hi)))
                    print('REPRESENTATION',dataset,layer,noise,carrier,identity,flush=True)
        # Checkpoint completed layers without changing any estimates.
        write(DOC/dataset/'geometry_timeline.csv',timeline);write(DOC/dataset/'geometry_fold_diagnostics.csv',foldrows)
        write(DOC/dataset/'activation_timeline.csv',act_timeline);write(DOC/dataset/'representation_effects.csv',effects);write(DOC/dataset/'representation_window_effects.csv',windoweffects)
    print(dataset,'REPRESENTATION COMPLETE',flush=True)

