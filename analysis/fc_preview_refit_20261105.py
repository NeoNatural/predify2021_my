"""Training-fold-only readout supplement; frozen pilot responses and mechanism unchanged."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
import argparse
import hashlib
import json
import pickle
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis import fc_preview_pilot_20261105 as exp
from predify_experiments.fc_preview_memory_20261105 import TrialEngine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

OUT=ROOT/'Hebbian-CNN/Log/fc_preview_readout_refit_20261105'
METHODS=exp.METHODS+('inhib_scale',)

def protocol():
    old=exp.read(exp.OUT/'protocol.json');exp.validate(old)
    p=dict(version=1,source_sha256=exp.sha(__file__),pilot_protocol_sha256=exp.sha(exp.OUT/'protocol.json'),
        purpose='readout refitting only; existing default clean pilot test responses unchanged',
        training='formal fold!=0 targets and their own previews only; no offline Hebbian graph',
        C=1,max_iter=30000,seed=exp.SEED,shared_history_weights='1/3 per target/history, total mass unchanged',
        separate_heads='auxiliary remaining-linear-decodability diagnostic, not common readout priming contrast',
        no_search=True,no_new_test_trials=True,parameters=old['parameters'],datasets={})
    for d in ('imagenet','things'):
        exp.verify(exp.OUT/d/'inputs');exp.verify(exp.OUT/d/'pilot')
        p['datasets'][d]={k:exp.sha(exp.OUT/d/k) for k in ('inputs/complete.json','pilot/complete.json','reference_svm.pckl','nuisance.json')}
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'protocol.json'
    if path.exists():assert exp.read(path)==p,'Readout supplement source or inputs changed'
    else:
        exp.write_json(path,p);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return old,p

def marker(dest,names):
    exp.write_json(dest/'complete.json',dict(protocol_sha256=exp.sha(OUT/'protocol.json'),files={n:exp.sha(dest/n) for n in names}))

def verify(dest):
    p=exp.read(dest/'complete.json');assert p['protocol_sha256']==exp.sha(OUT/'protocol.json')
    for n,h in p['files'].items():assert exp.sha(dest/n)==h

def extract(d,old):
    """Encode current-trial training previews, verifying cached stimuli against fresh rendering."""
    dest=OUT/d/'inputs'
    if (dest/'complete.json').exists():verify(dest);return
    from torchvision.models import vgg16,VGG16_Weights
    from PIL import Image
    m=exp.read(exp.native.OUT/d/'manifest.json');trials=[t for t in m['trials'] if t['fold']!=0]
    raw,y,concept,info=exp.training(d,old)
    held={c for c,f in m['concept_folds'].items() if f==0}|{t['concept'] for t in m['development']}
    forbidden={r['sha256'] for r in m['images'].values() if r['concept'] in held}
    roles=[]
    for t in trials:
        for i in set([t['dataset_index'],*t['previews'],t['primes']['exact'],t['primes']['different_category']]):
            r=m['images'][str(i)]
            assert r['concept'] not in held and r['sha256'] not in forbidden
            assert m['concept_folds'][r['concept']]==t['fold']
            roles.append(dict(target=t['dataset_index'],image=i,concept=r['concept'],sha256=r['sha256']))
    model=vgg16(weights=None);state=torch.load(exp.CHECKPOINT,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);del state
    model.eval().requires_grad_(False).cuda();activations={}
    def model_sha():
        h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
        return h.hexdigest()
    before=model_sha()
    def record(k):
        def hook(module,args,out):activations[k]=out.detach()
        return hook
    hooks=[model.classifier[i].register_forward_hook(record(k)) for i,k in ((1,'fc1'),(4,'fc2'))]
    transform=VGG16_Weights.IMAGENET1K_V1.transforms();f=exp.native.FilteredInput(torch)
    # Cache CPU image tensors; memory does not persist in TrialEngine across trials.
    images={};encoded={};hashes=[];data={k:[] for k in ('preview_fc1','preview_fc2')};start=time.monotonic()
    def image(i):
        if i not in images:
            r=m['images'][str(i)];assert exp.sha(r['path'])==r['sha256']
            with Image.open(r['path']) as im:images[i]=transform(im.convert('RGB'))
        return images[i].cuda()
    with torch.inference_mode():
        for j,t in enumerate(trials):
            path=exp.native.OUT/d/'formal'/f"stimuli_{t['dataset_index']}.npz"
            with np.load(path) as z:
                target=f.canvas([(image(t['dataset_index']),'center')]);targetsha=exp.array_sha(target.cpu().numpy())
                assert targetsha==info['training_target_hashes'][str(t['dataset_index'])]
                assert exp.array_sha(z['target_0'])==targetsha
                vv={k:[] for k in ('fc1','fc2')};fh={}
                for hist,key in (('exact','exact'),('different','different_category')):
                    ids=list(t['previews']);ids[t['slot']]=t['primes'][key];zs=[];hs=[]
                    for frame in range(2):
                        canvas=f.canvas([(image(i),side) for i,side in zip(ids[2*frame:2*frame+2],('left','right'))],True)
                        h=exp.array_sha(canvas.cpu().numpy());assert h==exp.array_sha(z[f'{key}_preview{frame}'])
                        if h not in encoded:
                            model(canvas[None]);encoded[h]={k:v[0].cpu().numpy().copy() for k,v in activations.items()}
                        hs.append(h);zs.append(encoded[h])
                    fh[hist]=hs
                    for k in vv:vv[k].append(np.stack([v[k] for v in zs]))
                for k in vv:data['preview_'+k].append(np.stack(vv[k]))
                assert fh['exact'][1-t['slot']//2]==fh['different'][1-t['slot']//2]
                hashes.append(dict(target=t['dataset_index'],stimulus_sha256=exp.sha(path),target_sha256=targetsha,preview_sha256=fh))
            if (j+1)%32==0:print('TRAIN PREVIEWS',d,j+1,'/',len(trials),flush=True)
    after=model_sha();assert before==after
    for h in hooks:h.remove()
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'features.npz',target_fc1=raw['fc1'],target_fc2=raw['fc2'],
        **{k:np.stack(v) for k,v in data.items()},y=y,concept=concept,row=np.array([t['dataset_index'] for t in trials]))
    exp.write_json(dest/'audit.json',dict(passed=True,roles=roles,inputs=hashes,unique_frames=len(encoded),rows=len(trials),
        heldout_concepts_absent=True,heldout_visual_bytes_absent=True,all_preview_roles_same_fold=True,
        cached_stimuli_equal_current_rendering=True,backbone_before=before,backbone_after=after,seconds=time.monotonic()-start))
    marker(dest,['features.npz','audit.json']);del model;torch.cuda.empty_cache()

def add_inhib_scale(reps):
    for h in exp.HISTORIES:
        for s in exp.STAGES:
            base=reps[f'inhibition__{h}__{s}'];ref=reps[f'hebbian__{h}__{s}']
            reps[f'inhib_scale__{h}__{s}']=base*(ref.mean(1,keepdims=True)/np.maximum(base.mean(1,keepdims=True),1e-8))

def transform(d,old):
    dest=OUT/d/'representations'
    if (dest/'complete.json').exists():verify(dest);return
    verify(OUT/d/'inputs')
    with np.load(OUT/d/'inputs/features.npz') as z:a={k:z[k] for k in z.files}
    ns=exp.nuisance(d,old);w,b=exp.load_bridge();engine=TrialEngine(w,b,(ns['fc1']['scale'],ns['fc2']['scale']))
    reps={f'{m}__{h}__{s}':[] for m in exp.METHODS for h in exp.HISTORIES for s in exp.STAGES}
    audit=[];bridge_error=0.;start=time.monotonic()
    with torch.inference_mode():
        for j,target in enumerate(a['row']):
            x=torch.from_numpy(a['target_fc1'][j:j+1]).cuda();ps=exp.permutations(d,int(target))
            for h,idx in (('exact',0),('different',1),('none',None)):
                p1=[] if idx is None else torch.from_numpy(a['preview_fc1'][j,idx]).cuda()
                p2=[] if idx is None else torch.from_numpy(a['preview_fc2'][j,idx]).cuda()
                for method in exp.METHODS:
                    r,meta=engine.run(x,p1,p2,method,ps)
                    assert meta['before_target']==meta['after_target']
                    assert meta['writes']==([0,0] if idx is None or method in ('vanilla','inhibition') else [2,2])
                    for stage,v in r.items():reps[f'{method}__{h}__{stage}'].append(v[0].cpu().numpy())
                    if method=='vanilla':bridge_error=max(bridge_error,float(np.abs(r['fc2'][0].cpu().numpy()-a['target_fc2'][j]).max()))
                    audit.append(dict(target=int(target),method=method,history=h,reset=meta['reset_sha256'],writes=meta['writes'],
                        before_target=meta['before_target'],after_target=meta['after_target']))
            if (j+1)%32==0:print('TRAIN CASCADES',d,j+1,'/',len(a['row']),flush=True)
    reps={k:np.stack(v) for k,v in reps.items()};add_inhib_scale(reps)
    assert np.allclose(reps['vanilla__exact__fc2'],a['target_fc2'],atol=1e-4,rtol=2e-5),bridge_error
    for m in ('vanilla','inhibition'):
        for s in exp.STAGES:assert all(np.array_equal(reps[f'{m}__exact__{s}'],reps[f'{m}__{h}__{s}']) for h in exp.HISTORIES)
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'features.npz',**reps,y=a['y'],concept=a['concept'],row=a['row'])
    exp.write_json(dest/'audit.json',dict(passed=True,condition_trials=len(audit),records=audit,bridge_raw_cache_max_error=bridge_error,
        input_marker_sha256=exp.sha(OUT/d/'inputs/complete.json'),seconds=time.monotonic()-start,
        inhib_scale='posthoc per-stage diagnostic only, not a physical cascade'))
    marker(dest,['features.npz','audit.json'])

def pool_histories(xs,y,groups):
    """Equivalent weighted objective; coalesce identical history matrices without raising effective C."""
    unique={}
    for x in xs:
        h=exp.array_sha(x)
        if h in unique:unique[h][1]+=1
        else:unique[h]=[x,1]
    blocks=list(unique.values());w=exp.weights(y,groups)
    return np.concatenate([x for x,n in blocks]),np.tile(y,len(blocks)),np.concatenate([w*n/len(xs) for x,n in blocks])

def signature(x,y,w):
    h=hashlib.sha256()
    for a in (x,y,w):h.update(str((a.shape,str(a.dtype))).encode());h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def metrics(scores,y):
    k=scores.shape[1];pred=scores.argmax(1)
    return dict(ba=float(np.mean([(pred[y==c]==c).mean() for c in range(k)])),
        auc=float(np.mean([roc_auc_score(y==c,scores[:,c]) for c in range(k)])),
        recalls=[float((pred[y==c]==c).mean()) for c in range(k)])

def fit(d,old):
    dest=OUT/d/'readouts'
    if (dest/'complete.json').exists():verify(dest);return
    verify(OUT/d/'representations');exp.verify(exp.OUT/d/'pilot')
    with np.load(OUT/d/'representations/features.npz') as z:
        y,groups,rows=z['y'],z['concept'],z['row'];train={k:z[k] for k in z.files if '__' in k}
    with np.load(exp.OUT/d/'pilot/representations.npz') as z:
        test={k:z[k] for k in z.files if '__' in k};ty,tgroups,trows=z['y'],z['concept'],z['row']
    add_inhib_scale(test)
    assert set(groups).isdisjoint(set(tgroups)) and set(rows).isdisjoint(set(trows))
    assert rows.tolist()==old['datasets'][d]['training_rows']
    with (exp.OUT/d/'reference_svm.pckl').open('rb') as f:fixed=pickle.load(f)
    heads={};mapping={};summary=[];scorebank={};fitinfo=[];start=time.monotonic()
    def head_for(x,yy,ww,key):
        sig=signature(x,yy,ww);mapping[key]=sig
        assert abs(ww.sum()-len(y))<1e-7
        if sig not in heads:
            head=make_pipeline(StandardScaler(),LinearSVC(C=1,dual='auto',max_iter=30000,random_state=exp.SEED))
            with threadpool_limits(limits=2),warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                head.fit(x,yy,standardscaler__sample_weight=ww,linearsvc__sample_weight=ww)
            assert int(head[-1].n_iter_)<30000;heads[sig]=head
            fitinfo.append(dict(signature=sig,rows=len(yy),weight_mass=float(ww.sum()),n_iter=int(head[-1].n_iter_)))
            print('FIT',d,len(heads),key,'seconds',round(time.monotonic()-start,1),flush=True)
        return heads[sig]
    with threadpool_limits(limits=2):
        for method in METHODS:
            for stage in exp.STAGES:
                xs=[train[f'{method}__{h}__{stage}'] for h in exp.HISTORIES]
                x,yy,ww=pool_histories(xs,y,groups)
                shared=head_for(x,yy,ww,f'shared__{method}__{stage}')
                original=fixed['fc1' if stage=='fc1' else 'fc2']
                for h in exp.HISTORIES:
                    key=f'{method}__{h}__{stage}';a=test[key]
                    separate=head_for(train[key],y,exp.weights(y,groups),'separate__'+key)
                    scores=dict(fixed=original.decision_function(a),shared=shared.decision_function(a),separate=separate.decision_function(a),
                        scaler_only=original[-1].decision_function(shared[0].transform(a)))
                    for kind,s in scores.items():
                        scorebank[f'{kind}__{key}']=s
                        summary.append(dict(dataset=d,method=method,history=h,stage=stage,readout=kind,**metrics(s,ty)))
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'models.pckl').open('wb') as f:pickle.dump(dict(heads=heads,mapping=mapping),f,protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(dest/'test_scores.npz',**scorebank,y=ty,concept=tgroups,row=trows)
    exp.write_json(dest/'metrics.json',summary)
    exp.write_json(dest/'audit.json',dict(passed=True,train_rows=rows.tolist(),test_rows=trows.tolist(),train_count=len(y),test_count=len(ty),
        concepts_disjoint=True,test_not_used_for_fit=True,C=1,parameter_search=False,shared_total_sample_weight=len(y),
        models=len(heads),model_aliases=len(mapping),fit_info=fitinfo,training_representation_marker_sha256=exp.sha(OUT/d/'representations/complete.json'),
        seconds=time.monotonic()-start))
    marker(dest,['models.pckl','test_scores.npz','metrics.json','audit.json']);exp.validate(old)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','worker']);parser.add_argument('--dataset',choices=['imagenet','things']);args=parser.parse_args()
    old,p=protocol();exp.setup()
    if args.action=='worker':
        assert args.dataset;extract(args.dataset,old);transform(args.dataset,old);fit(args.dataset,old)
        protocol();print('DONE',args.dataset,flush=True)
