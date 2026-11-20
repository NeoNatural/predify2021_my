"""Frozen common readout and concept-grouped temporal MVPA for the 2026-10-27 study."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import argparse,csv,json,sys,time,warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from joblib import Parallel,delayed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.tripartite_20261027 import OUT,DOC,SEED,conditions,CATEGORIES,IDENTITIES
from predify_experiments.readout_init_20261012 import immutable_json,sha
from predify_experiments.supplementary import load,save
from analysis.sliding_mvpa_20261013 import folds_for


def write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def data(dataset,split):
    m=json.loads((OUT/dataset/'manifest.json').read_text())
    trials=m['dev'][:8] if split=='old_layout' else m[split]
    noises=m['noise_seeds']['dev'][:1] if split=='old_layout' else m['noise_seeds'][split]
    expected={(r['dataset_index'],s) for r in trials for s in noises}
    names={r['name'] for r in conditions()}
    if split=='old_layout':names={r['name'] for r in conditions(False) if r['init']=='grey' and r['identity'] in ('exact','neutral')}
    seen=set();values={c:{} for c in names};activations={c:{} for c in names};metadata={};noise_stats=[];preview_weights={}
    protected={r['dataset_index'] for s in ('train','dev','test') for r in m[s]}
    for p in sorted((OUT/dataset/split).glob('image_*.pckl')):
        d=load(p);t=d['trial'];key=t['dataset_index'],d['noise']
        assert key in expected and key not in seen;seen.add(key)
        assert {r['name'] for r in d['records']}==names
        assert len({r['target_hash'] for r in d['records']})==1
        metadata[key]=(t['category_index'],t['concept'])
        noise_stats.append(dict(image=key[0],noise=key[1],**d['noise_stats']))
        for r in d['records']:
            assert r['fc2'].shape==(15,4096) and np.isfinite(r['fc2']).all()
            assert r['target_W_frozen']
            weight_key=t['dataset_index'],r['name']
            if weight_key in preview_weights:assert preview_weights[weight_key]==r['weight_norms'],'Clean preview W depends on target noise'
            preview_weights[weight_key]=r['weight_norms']
            assert not ((set(r['preview_indices'])-{t['dataset_index'],None})&protected)
            if r['carrier'] in ('hebb','joint'):
                assert len(r['commit_events'])==4
                assert {e['phase'] for e in r['commit_events']}=={'preview_boundary_0','preview_boundary_1'}
            else:assert not r['commit_events']
            values[r['name']][key]=r['fc2'];activations[r['name']][key]=r['activation'][:,0]
    assert seen==expected,(dataset,split,'missing',len(expected-seen))
    keys=sorted(seen)
    return dict(keys=keys,y=np.array([metadata[k][0] for k in keys]),groups=np.array([metadata[k][1] for k in keys]),
        features={c:np.stack([v[k] for k in keys]) for c,v in values.items()},
        activation={c:np.stack([v[k] for k in keys]) for c,v in activations.items()},noise_stats=noise_stats)


def fit_svm(x,y,weights=None):
    attempts=[]
    with threadpool_limits(limits=1):
        for cap in (50000,200000):
            clf=make_pipeline(StandardScaler(),LinearSVC(C=1.,dual='auto',max_iter=cap,random_state=SEED))
            kw={} if weights is None else {'standardscaler__sample_weight':weights,'linearsvc__sample_weight':weights}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning);clf.fit(x,y,**kw)
            converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
            attempts.append(dict(cap=cap,n_iter=int(clf[-1].n_iter_),converged=converged))
            if converged:return clf,attempts
    raise RuntimeError('SVM did not converge; do not inspect confirmation to tune C')


def training_rows(d,m):
    counters={c:0 for c in range(4)};ordinals={}
    for row in m['train']:
        c=row['category_index'];ordinals[row['dataset_index']]=counters[c];counters[c]+=1
    xs=[];ys=[];weights=[];audit=[]
    for j,(index,noise) in enumerate(d['keys']):
        noise_ord=m['noise_seeds']['train'].index(noise)
        for t in range(15):
            identity=IDENTITIES[(ordinals[index]+noise_ord+t)%5]
            for c,w in [('none',1.),(f'pcoder_{identity}',1.),(f'hebb_{identity}',.5),
                        (f'hebb_weak_{identity}',.5),(f'joint_{identity}',.5),(f'joint_weak_{identity}',.5)]:
                xs.append(d['features'][c][j,t]);ys.append(d['y'][j]);weights.append(w)
                audit.append(dict(image=index,noise=noise,timestep=t,condition=c,weight=w))
    x=np.asarray(xs,np.float32);y=np.asarray(ys);weights=np.asarray(weights)
    for index in {k[0] for k in d['keys']}:
        assert sum(r['weight'] for r in audit if r['image']==index)==180
    return x,y,weights,audit


def fit(dataset):
    base=OUT/dataset;dest=DOC/dataset;dest.mkdir(parents=True,exist_ok=True)
    if (base/'freeze.json').exists():raise ValueError('Already frozen; refusing to refit')
    d=data(dataset,'train');m=json.loads((base/'manifest.json').read_text())
    x,y,w,audit=training_rows(d,m)
    print(dataset,'FIT',x.shape,'effective_rows',float(w.sum()),flush=True)
    head,attempts=fit_svm(x,y,w)
    save(base/'head.pckl',dict(head=head,attempts=attempts,training_keys=d['keys'],manifest_sha=sha(base/'manifest.json'),sampler='one identity per image/noise/time, both strengths half-weighted',source_sha=sha(__file__)))
    (base/('head_source_'+sha(__file__)[:12]+'.py')).write_bytes(Path(__file__).read_bytes())
    write(dest/'readout_training_rows.csv',audit)
    write(dest/'readout_fit.csv',attempts)
    print(dataset,'FIT DONE',attempts,flush=True)


def margin(scores,y):
    result=scores[np.arange(len(y)),y]
    wrong=scores.copy();wrong[np.arange(len(y)),y]=-np.inf
    return result-wrong.max(-1)


def common_scores(dataset,split):
    d=data(dataset,split);head=load(OUT/dataset/'head.pckl')['head']
    scores={c:head.decision_function(x.astype(np.float32).reshape(-1,4096)).reshape(len(d['y']),15,4) for c,x in d['features'].items()}
    return d,scores


def freeze(dataset):
    base=OUT/dataset;dest=DOC/dataset
    implementation=json.loads((OUT/'implementation.json').read_text())
    for name,expected in implementation['source_hashes'].items():assert sha(ROOT/name)==expected,('Implementation changed',name)
    d,scores=common_scores(dataset,'dev')
    assert load(base/'head.pckl')['manifest_sha']==sha(base/'manifest.json')
    rows=[]
    for c,s in scores.items():
        for t in range(15):rows.append(dict(condition=c,timestep=t,accuracy_pct=100*float(np.mean(s[:,t].argmax(-1)==d['y']))))
    write(dest/'development_common_readout.csv',rows)
    write(dest/'development_noise.csv',d['noise_stats'])
    final={c:next(r['accuracy_pct'] for r in rows if r['condition']==c and r['timestep']==14) for c in ('none','hebb_exact','pcoder_exact','joint_exact')}
    if max(final.values())<=25:raise RuntimeError('Severe four-class floor; development implementation requires inspection, no confirmation yet')
    ratios={c:float(np.mean(d['activation'][c][:,-1]/np.maximum(d['activation']['none'][:,-1],1e-8))) for c in ('hebb_exact','joint_exact')}
    if min(ratios.values())<.1:raise RuntimeError('Extreme channel suppression in development; inspect before confirmation')
    frozen=dict(head_sha=sha(base/'head.pckl'),manifest_sha=sha(base/'manifest.json'),protocol_sha=sha(DOC/'预先方案.md'),implementation_sha=sha(OUT/'implementation.json'),
        grey_sha=sha(OUT/'grey.json'),runner_sha=sha(ROOT/'predify_experiments/tripartite_20261027.py'),analysis_sha=sha(__file__),
        primary='concept-grouped time-specific MVPA; common frozen head is complementary inferential control',
        windows=dict(early=[0,4],middle=[5,9],late=[10,14],full=[0,14]),
        inferential_contrasts=['joint-pcoder','joint-hebb','pcoder-hebb','hebb-none'],
        inferential_endpoint='common head full t0-t14 mean accuracy, exact, strong; Holm across 8 tests',
        peak_inference='descriptive only; no retraining/selection uncertainty claimed',
        dev_floor_check=final,dev_activation_ratios=ratios,selection='none; all settings prespecified')
    immutable_json(base/'freeze.json',frozen)
    (base/'analysis_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(dataset,'FROZEN',final,ratios,flush=True)


def fit_cv_time(c,t,x,y,folds,dest):
    path=dest/f'{c}_t{t:02d}.npz'
    if path.exists():
        with np.load(path) as z:assert z['scores'].shape==(len(y),4)
        return
    scores=np.full((len(y),4),np.nan);iterations=[]
    for tr,te in folds:
        head,attempts=fit_svm(x[tr,t].astype(np.float32),y[tr]);scores[te]=head.decision_function(x[te,t].astype(np.float32))
        iterations.append(attempts[-1]['n_iter'])
    assert np.isfinite(scores).all()
    np.savez_compressed(path,scores=scores,iterations=iterations)


def summarize_scores(dataset,d,scores,method):
    dest=DOC/dataset;rows=[];category_rows=[];confusion=[];noise_rows=[];peaks=[]
    for c,s in scores.items():
        correct=s.argmax(-1)==d['y'][:,None]
        for t in range(15):
            rows.append(dict(dataset=dataset,method=method,condition=c,timestep=t,accuracy_pct=100*float(correct[:,t].mean()),
                mean_margin=float(margin(s[:,t],d['y']).mean()),mean_abs_fc2_activation=float(d['activation'][c][:,t].mean())))
            for category in range(4):
                mask=d['y']==category
                category_rows.append(dict(condition=c,timestep=t,category=CATEGORIES[category],accuracy_pct=100*float(correct[mask,t].mean())))
                for predicted in range(4):
                    confusion.append(dict(condition=c,timestep=t,true_category=category,predicted_category=predicted,
                        count=int(np.sum(mask&(s[:,t].argmax(-1)==predicted)))))
            for noise in sorted({k[1] for k in d['keys']}):
                mask=np.array([k[1]==noise for k in d['keys']]);noise_rows.append(dict(condition=c,timestep=t,noise=noise,accuracy_pct=100*float(correct[mask,t].mean())))
        for name,a,b in [('early',0,5),('middle',5,10),('late',10,15),('full',0,15)]:
            curve=correct[:,a:b].mean(0)*100;peak=int(np.argmax(curve))+a
            peaks.append(dict(condition=c,window=name,average_accuracy_pct=float(curve.mean()),peak_accuracy_pct=float(curve.max()),first_peak_timestep=peak,
                inference='descriptive; selection/refitting uncertainty not included'))
    write(dest/f'{method}_timeline.csv',rows);write(dest/f'{method}_categories.csv',category_rows)
    write(dest/f'{method}_confusion.csv',confusion);write(dest/f'{method}_noise.csv',noise_rows);write(dest/f'{method}_peaks.csv',peaks)


def contrast_definitions():
    result=[]
    for strength in ('strong','weak'):
        prefix='' if strength=='strong' else 'weak_'
        for identity in IDENTITIES:
            h=f'hebb_{prefix}{identity}';j=f'joint_{prefix}{identity}';p=f'pcoder_{identity}'
            for name,a,b in [('joint-pcoder',j,p),('joint-hebb',j,h),('pcoder-hebb',p,h),('hebb-none',h,'none'),('pcoder-none',p,'none'),('joint-none',j,'none')]:
                result.append(dict(strength=strength,identity=identity,contrast=name,a=a,b=b))
        for carrier in ('hebb','joint','pcoder'):
            if carrier=='pcoder' and strength=='weak':continue
            cp='' if carrier=='pcoder' else prefix
            for identity in IDENTITIES[1:]:
                result.append(dict(strength=strength,identity='exact',contrast=f'{carrier}_exact-minus-{identity}',a=f'{carrier}_{cp}exact',b=f'{carrier}_{cp}{identity}'))
        for carrier in ('hebb','joint','pcoder'):
            if carrier=='pcoder' and strength=='weak':continue
            cp='' if carrier=='pcoder' else prefix
            for control in ('none_random','none_feedforward'):
                result.append(dict(strength=strength,identity='exact',contrast=f'{carrier}-minus-{control}',a=f'{carrier}_{cp}exact',b=control))
    for control in ('none_random','none_feedforward'):
        result.append(dict(strength='strong',identity='exact',contrast=f'grey-minus-{control}',a='none',b=control))
    return result


def cluster_stats(delta,d,draws=10000):
    # Delta can already average over all prespecified times; noise remains within concept.
    groups=np.unique(d['groups']);clusters=np.array([delta[d['groups']==g].mean() for g in groups])
    labels=np.array([d['y'][np.flatnonzero(d['groups']==g)[0]] for g in groups])
    rng=np.random.default_rng(SEED);boot=np.zeros(draws)
    for category in range(4):
        values=clusters[labels==category];ix=rng.integers(len(values),size=(draws,len(values)))
        boot+=values[ix].mean(1)/4
    low,high=np.quantile(boot,[.025,.975])
    signs=np.random.default_rng(SEED).integers(0,2,size=(20000,len(groups)))*2-1
    distribution=signs@clusters
    p=(np.sum(np.abs(distribution)>=abs(clusters.sum())-1e-9)+1)/20001
    return dict(delta_pp=float(clusters.mean()),low_pp=float(low),high_pp=float(high),concepts=len(groups),signflip_p=float(p))


def effects(dataset,d,scores,method):
    correct={c:(s.argmax(-1)==d['y'][:,None]).astype(float) for c,s in scores.items()}
    rows=[];windows=[];interaction=[];interaction_windows=[];errors=[]
    for definition in contrast_definitions():
        a,b=definition['a'],definition['b'];delta=(correct[a]-correct[b])*100
        for t in range(15):
            rows.append(dict(**definition,timestep=t,delta_pp=float(delta[:,t].mean()),
                margin_delta=float(np.mean(margin(scores[a][:,t],d['y'])-margin(scores[b][:,t],d['y']))),
                wins=int(np.sum((correct[a][:,t]==1)&(correct[b][:,t]==0))),losses=int(np.sum((correct[a][:,t]==0)&(correct[b][:,t]==1)))))
        for name,a0,b0 in [('early',0,5),('middle',5,10),('late',10,15),('full',0,15)]:
            trial_delta=delta[:,a0:b0].mean(1)
            stats=cluster_stats(trial_delta,d) if method=='common' else dict(delta_pp=float(trial_delta.mean()),low_pp=float('nan'),high_pp=float('nan'),concepts=len(set(d['groups'])),signflip_p=float('nan'))
            bynoise={str(n):float(trial_delta[[k[1]==n for k in d['keys']]].mean()) for n in sorted({k[1] for k in d['keys']})}
            acurve=correct[a][:,a0:b0].mean(0)*100;bcurve=correct[b][:,a0:b0].mean(0)*100
            at=int(np.argmax(acurve));bt=int(np.argmax(bcurve))
            windows.append(dict(dataset=dataset,method=method,**definition,window=name,**stats,per_noise=json.dumps(bynoise),
                peak_height_delta_pp=float(acurve[at]-bcurve[bt]),a_peak_time=at+a0,b_peak_time=bt+a0,
                peak_time_delta_steps=at-bt,same_time_at_a_peak_delta_pp=float(acurve[at]-bcurve[at]),peak_inference='descriptive only',
                inferential=bool(method=='common' and definition['strength']=='strong' and definition['identity']=='exact' and name=='full' and definition['contrast'] in ('joint-pcoder','joint-hebb','pcoder-hebb','hebb-none'))))
    for strength in ('strong','weak'):
        prefix='' if strength=='strong' else 'weak_'
        for identity in IDENTITIES:
            values=100*(correct[f'joint_{prefix}{identity}']-correct[f'pcoder_{identity}']-correct[f'hebb_{prefix}{identity}']+correct['none'])
            for t in range(15):interaction.append(dict(strength=strength,identity=identity,timestep=t,interaction_pp=float(values[:,t].mean()),inference='descriptive'))
            for name,a0,b0 in [('early',0,5),('middle',5,10),('late',10,15),('full',0,15)]:
                averaged=values[:,a0:b0].mean(1)
                stats=cluster_stats(averaged,d) if method=='common' else dict(delta_pp=float(averaged.mean()),low_pp=float('nan'),high_pp=float('nan'),concepts=len(set(d['groups'])),signflip_p=float('nan'))
                interaction_windows.append(dict(dataset=dataset,method=method,strength=strength,identity=identity,window=name,**stats,
                    secondary_inferential=bool(method=='common' and identity=='exact' and name=='full')))
    dest=DOC/dataset
    write(dest/f'{method}_contrasts.csv',rows);write(dest/f'{method}_window_contrasts.csv',windows);write(dest/f'{method}_interaction.csv',interaction)
    write(dest/f'{method}_interaction_windows.csv',interaction_windows)


def analyze(dataset):
    base=OUT/dataset;dest=DOC/dataset;dest.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((base/'freeze.json').read_text());assert sha(base/'head.pckl')==frozen['head_sha']
    assert sha(DOC/'预先方案.md')==frozen['protocol_sha'],'Protocol changed after freeze'
    assert sha(OUT/'implementation.json')==frozen['implementation_sha'],'Implementation changed after freeze'
    assert sha(__file__)==frozen['analysis_sha'],'Analysis changed after freeze; preserve this study and use a new directory'
    d,scores=common_scores(dataset,'test')
    save(base/'common_scores.pckl',dict(keys=d['keys'],y=d['y'],groups=d['groups'],scores=scores))
    summarize_scores(dataset,d,scores,'common');effects(dataset,d,scores,'common')
    write(dest/'test_noise_statistics.csv',d['noise_stats'])
    folds=folds_for(d['y'],d['groups']);cvdir=base/'mvpa';cvdir.mkdir(parents=True,exist_ok=True)
    immutable_json(cvdir/'folds.json',dict(keys=d['keys'],groups=d['groups'].tolist(),
        folds=[dict(train=a.tolist(),test=b.tolist()) for a,b in folds],seed=20261013,scope='fixed concept-grouped diagonal CV; no peak significance'))
    print(dataset,'MVPA',len(d['keys']),'rows',len(set(d['groups'])),'concepts',flush=True)
    with threadpool_limits(limits=1):
        Parallel(n_jobs=4,prefer='threads')(delayed(fit_cv_time)(c,t,x,d['y'],folds,cvdir) for c,x in d['features'].items() for t in range(15))
    mvpa={}
    for c in d['features']:
        mvpa[c]=np.stack([np.load(cvdir/f'{c}_t{t:02d}.npz')['scores'] for t in range(15)],axis=1)
    save(base/'mvpa_scores.pckl',dict(keys=d['keys'],y=d['y'],groups=d['groups'],scores=mvpa))
    summarize_scores(dataset,d,mvpa,'mvpa');effects(dataset,d,mvpa,'mvpa')
    audit=dict(images=len(set(k[0] for k in d['keys'])),concepts=len(set(d['groups'])),noise_realizations=3,
        conditions=28,timepoints=15,svm_fits=2100,scaler_fit_within_fold=True,concept_overlap_between_folds=False,
        target_concepts_disjoint_from_frozen_head_training=True,paired_target_hashes=True,target_W_frozen=True,
        all_svm_converged=True,peak_significance_claimed=False)
    immutable_json(dest/'analysis_audit.json',audit)
    print(dataset,'ANALYSIS COMPLETE',audit,flush=True)


def old(dataset):
    d,old_scores=common_scores(dataset,'old_layout');new,new_scores=common_scores(dataset,'dev')
    lookup={k:i for i,k in enumerate(new['keys'])};ix=[lookup[k] for k in d['keys']]
    new_scores={c:s[ix] for c,s in new_scores.items() if c in old_scores}
    rows=[];effects_rows=[]
    for layout,scores in [('old_34px',old_scores),('new_64px',new_scores)]:
        correct={c:s.argmax(-1)==d['y'][:,None] for c,s in scores.items()}
        for c,pred in correct.items():
            for t in range(15):rows.append(dict(layout=layout,condition=c,timestep=t,images=len(d['keys']),accuracy_pct=100*float(pred[:,t].mean()),
                scope='same development targets and frozen NEW-layout readout; descriptive layout/readout sensitivity'))
        for identity in ('exact','neutral'):
            for name,a,b in [('joint-pcoder','joint_'+identity,'pcoder_'+identity),('joint-hebb','joint_'+identity,'hebb_'+identity),('pcoder-hebb','pcoder_'+identity,'hebb_'+identity),('hebb-none','hebb_'+identity,'none')]:
                for t in range(15):effects_rows.append(dict(layout=layout,identity=identity,contrast=name,timestep=t,
                    delta_pp=100*float(np.mean(correct[a][:,t].astype(float)-correct[b][:,t].astype(float)))))
    write(DOC/dataset/'old_layout_diagnostic.csv',rows);write(DOC/dataset/'old_layout_pairwise.csv',effects_rows)


def holm():
    rows=[]
    for dataset in ('imagenet','things'):
        with (DOC/dataset/'common_window_contrasts.csv').open(encoding='utf-8-sig') as f:
            rows.extend(r for r in csv.DictReader(f) if r['inferential']=='True')
    assert len(rows)==8
    order=np.argsort([float(r['signflip_p']) for r in rows]);adjusted=np.empty(8);running=0.
    for rank,index in enumerate(order):
        running=max(running,(8-rank)*float(rows[index]['signflip_p']));adjusted[index]=min(running,1.)
    for r,p in zip(rows,adjusted):r['holm_p']=float(p)
    write(DOC/'primary_common_readout_holm.csv',rows)
    print('PRIMARY',json.dumps(rows,indent=2),flush=True)
    interaction=[]
    for dataset in ('imagenet','things'):
        with (DOC/dataset/'common_interaction_windows.csv').open(encoding='utf-8-sig') as f:
            interaction.extend(r for r in csv.DictReader(f) if r['secondary_inferential']=='True')
    assert len(interaction)==4
    order=np.argsort([float(r['signflip_p']) for r in interaction]);running=0.
    for rank,index in enumerate(order):
        running=max(running,(4-rank)*float(interaction[index]['signflip_p']));interaction[index]['holm_p']=min(running,1.)
    write(DOC/'secondary_interaction_holm.csv',interaction)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('fit','freeze','analyze','holm','old'));p.add_argument('--dataset',choices=('imagenet','things'),default='imagenet');a=p.parse_args()
    if a.action=='holm':holm()
    else:globals()[a.action](a.dataset)


if __name__=='__main__':main()
