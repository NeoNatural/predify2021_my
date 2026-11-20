"""Single-beta subtractive clean pilot and training-only SVM adaptation; no original-model edits."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis import fc_preview_pilot_20261105 as exp
from analysis import fc_preview_refit_20261105 as previous
from predify_experiments.fc_preview_subtract_20261105 import TrialEngine,DEFAULTS,METHODS

OUT=ROOT/'Hebbian-CNN/Log/fc_preview_subtract_20261105'
DOC=ROOT/'docs/fc_preview_subtract_20261105'
ALL_METHODS=METHODS+('inhib_scale',)

def protocol():
    old=exp.read(exp.OUT/'protocol.json');exp.validate(old);previous.protocol()
    deps=['predify_experiments/fc_preview_subtract_20261105.py','predify_experiments/fc_preview_memory_20261105.py',
        'analysis/fc_preview_pilot_20261105.py','analysis/fc_preview_refit_20261105.py']
    p=dict(source_sha256=exp.sha(__file__),dependencies={k:exp.sha(ROOT/k) for k in deps},
        pilot_protocol_sha256=exp.sha(exp.OUT/'protocol.json'),refit_protocol_sha256=exp.sha(previous.OUT/'protocol.json'),
        parameters=DEFAULTS,methods=ALL_METHODS,histories=exp.HISTORIES,stages=exp.STAGES,
        beta_choice='4 from existing legacy convolution default; prespecified, not fitted/selected on test or training performance',
        subtract='h=.75 W r; v=ReLU(x+h-4 mean(h)); r_next=.5r+.5v, five steps; no extra gate or division',
        pool_only='retain current-preview W to read inhibitory pool; omit local +h; same closed-loop five-step rule',
        inhibition='unchanged previous no-memory gate+division reference, not a subtract-family ablation',
        no_preview='W=0: subtract full and pool-only equal vanilla; do not confuse with previous no-preview inhibition baseline',
        svm='unchanged weighted StandardScaler + LinearSVC C1 dual auto max_iter30000; fixed vanilla and shared-history plus auxiliary separate heads',
        adapted_stages=['fc1','fc2'],fc2_input_readout='fixed vanilla FC2 head diagnostic only',
        no_search=True,no_SAE=True,clean_only=True,datasets={})
    for d in ('imagenet','things'):
        exp.verify(exp.OUT/d/'pilot');previous.verify(previous.OUT/d/'inputs')
        p['datasets'][d]=dict(test_marker=exp.sha(exp.OUT/d/'pilot/complete.json'),
            train_marker=exp.sha(previous.OUT/d/'inputs/complete.json'),fixed_svm=exp.sha(exp.OUT/d/'reference_svm.pckl'),
            training_rows=old['datasets'][d]['training_rows'])
    # JSON round-trip makes tuple/list types identical for frozen comparisons.
    import json
    p=json.loads(json.dumps(p));OUT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert exp.read(OUT/'protocol.json')==p,'Subtractive protocol changed'
    else:
        exp.write_json(OUT/'protocol.json',p);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (OUT/'memory_snapshot.py').write_bytes((ROOT/deps[0]).read_bytes())
    return old,p

def marker(dest,names):
    exp.write_json(dest/'complete.json',dict(protocol_sha256=exp.sha(OUT/'protocol.json'),files={n:exp.sha(dest/n) for n in names}))

def verify(dest):
    p=exp.read(dest/'complete.json');assert p['protocol_sha256']==exp.sha(OUT/'protocol.json')
    for n,h in p['files'].items():assert exp.sha(dest/n)==h

def generate(d,scope,old):
    dest=OUT/d/scope
    if (dest/'complete.json').exists():verify(dest);return
    if scope=='pilot':a=exp.features(d)
    else:
        previous.verify(previous.OUT/d/'inputs')
        with np.load(previous.OUT/d/'inputs/features.npz') as z:a={k:z[k] for k in z.files}
        assert a['row'].tolist()==old['datasets'][d]['training_rows']
    ns=exp.nuisance(d,old);w,b=exp.load_bridge();engine=TrialEngine(w,b,(ns['fc1']['scale'],ns['fc2']['scale']))
    reps={f'{m}__{h}__{s}':[] for m in METHODS for h in exp.HISTORIES for s in exp.STAGES}
    records=[];count=0;steps=0;start=time.monotonic();zero=exp.PreviewMemory(4096).fingerprint();references={}
    with torch.inference_mode():
        for j,target in enumerate(a['row']):
            x=torch.from_numpy(a['target_fc1'][j:j+1]).cuda();ps=exp.permutations(d,int(target))
            for history,idx in (('exact',0),('different',1),('none',None)):
                p1=[] if idx is None else torch.from_numpy(a['preview_fc1'][j,idx]).cuda()
                p2=[] if idx is None else torch.from_numpy(a['preview_fc2'][j,idx]).cuda()
                for method in METHODS:
                    r,meta=engine.run(x,p1,p2,method,ps);assert meta['reset_sha256']==[zero,zero]
                    assert meta['before_target']==meta['after_target']
                    assert meta['writes']==([0,0] if idx is None or method in ('vanilla','inhibition') else [2,2])
                    for trace in meta['traces']:
                        assert len(trace['updates'])==5
                        for step in trace['updates']:assert step['weight_sha256']==meta['before_target'][trace['layer']-1];steps+=1
                    for s,v in r.items():reps[f'{method}__{history}__{s}'].append(v[0].cpu().numpy())
                    count+=1
                    if scope=='pilot':
                        meta.update(target=int(target),method=method,history=history);records.append(meta)
                        if method=='hebbian':references[j,history]={s:v.clone() for s,v in r.items()}
            if (j+1)%32==0 or scope=='pilot':print('GENERATE',d,scope,j+1,'/',len(a['row']),flush=True)
        if scope=='pilot':
            # Reverse-order replay after intervening histories and controls.
            for j in reversed(range(len(a['row']))):
                for h,idx in (('none',None),('different',1),('exact',0)):
                    x=torch.from_numpy(a['target_fc1'][j:j+1]).cuda()
                    r,meta=engine.run(x,[] if idx is None else torch.from_numpy(a['preview_fc1'][j,idx]).cuda(),
                        [] if idx is None else torch.from_numpy(a['preview_fc2'][j,idx]).cuda(),'hebbian',exp.permutations(d,int(a['row'][j])))
                    for s in exp.STAGES:assert torch.equal(r[s],references[j,h][s])
            # Same real 4096-D factors as old pilot, FP64 explicit matrices independently evaluated.
            j=0;x=torch.from_numpy(a['target_fc1'][j:j+1]).cuda().double()
            p1=torch.from_numpy(a['preview_fc1'][j,0]).cuda().double();p2=torch.from_numpy(a['preview_fc2'][j,0]).cuda().double()
            ed=TrialEngine(w.double(),b.double(),engine.scales);r,meta=ed.run(x,p1,p2,'hebbian')
            def dense(v,m,scale):
                W=torch.zeros(4096,4096,dtype=torch.float64,device='cuda')
                for u,c in zip(m.factors,m.coefficients):q=u[:,None]*u[None,:];q.fill_diagonal_(0);W+=c*q
                z=v/scale;state=z.clone()
                for _ in range(5):h=.75*(state@W.T);state=.5*state+.5*(z+h-4*h.mean(-1,keepdim=True)).clamp_min(0)
                return state*scale
            one=dense(x,ed.memories[0],engine.scales[0]);two=dense(torch.relu(torch.nn.functional.linear(one,w.double(),b.double())),ed.memories[1],engine.scales[1])
            dense_error=max(float((one-r['fc1']).abs().max()),float((two-r['fc2']).abs().max()));assert dense_error<2e-10
    reps={k:np.stack(v) for k,v in reps.items()};previous.add_inhib_scale(reps)
    for s in exp.STAGES:
        for m in ('hebbian','pool_only','shuffled'):
            assert np.array_equal(reps[f'{m}__none__{s}'],reps[f'vanilla__none__{s}'])
        assert np.array_equal(reps[f'inhibition__exact__{s}'],reps[f'inhibition__different__{s}'])
    if scope=='pilot':
        with np.load(exp.OUT/d/'pilot/representations.npz') as z:
            for m in ('vanilla','inhibition'):
                for h in exp.HISTORIES:
                    for s in exp.STAGES:assert np.array_equal(reps[f'{m}__{h}__{s}'],z[f'{m}__{h}__{s}'])
        old_records=exp.read(exp.OUT/d/'pilot/trial_audit.json')
        old_memory={(r['target'],r['history']):r['before_target'] for r in old_records if r['method']=='hebbian'}
        for r in records:
            if r['method']=='hebbian':assert r['before_target']==old_memory[r['target'],r['history']]
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'representations.npz',**reps,row=a['row'],y=a['y'],concept=a['concept'])
    exp.write_json(dest/'audit.json',dict(passed=True,dataset=d,scope=scope,rows=len(a['row']),condition_trials=count,checked_updates=steps,
        reset=True,target_readonly=True,preview_writes_only=True,no_preview_identity=True,
        original_reference_replay_exact=scope=='pilot',same_preview_weights_as_previous=scope=='pilot',
        reverse_order_replay=scope=='pilot',dense_fp64_error=dense_error if scope=='pilot' else None,
        seconds=time.monotonic()-start,records=records))
    marker(dest,['representations.npz','audit.json']);exp.validate(old)

def fit(d,old):
    dest=OUT/d/'readouts';dest.mkdir(parents=True,exist_ok=True);(dest/'heads').mkdir(exist_ok=True)
    if (dest/'complete.json').exists():verify(dest);return
    verify(OUT/d/'training');verify(OUT/d/'pilot')
    with np.load(OUT/d/'training/representations.npz') as z:train={k:z[k] for k in z.files if '__' in k};y,groups,rows=z['y'],z['concept'],z['row']
    with np.load(OUT/d/'pilot/representations.npz') as z:test={k:z[k] for k in z.files if '__' in k};ty,tgroups,trows=z['y'],z['concept'],z['row']
    assert rows.tolist()==old['datasets'][d]['training_rows'] and set(groups).isdisjoint(tgroups) and set(rows).isdisjoint(trows)
    with (exp.OUT/d/'reference_svm.pckl').open('rb') as f:fixed=pickle.load(f)
    bank={};mapping={};summary=[];scores={};start=time.monotonic();info=[]
    def head_for(x,yy,ww,key):
        sig=previous.signature(x,yy,ww);mapping[key]=sig
        assert abs(ww.sum()-len(y))<1e-7
        if sig not in bank:
            path=dest/'heads'/f'{sig}.pckl';meta=path.with_suffix('.json')
            if meta.exists():
                note=exp.read(meta);assert note['protocol_sha256']==exp.sha(OUT/'protocol.json') and note['sha256']==exp.sha(path)
                with path.open('rb') as f:head=pickle.load(f)
            else:
                head=make_pipeline(StandardScaler(),LinearSVC(C=1,dual='auto',max_iter=30000,random_state=exp.SEED))
                with threadpool_limits(limits=2),warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    head.fit(x,yy,standardscaler__sample_weight=ww,linearsvc__sample_weight=ww)
                assert head[-1].n_iter_<30000
                with path.open('wb') as f:pickle.dump(head,f,protocol=pickle.HIGHEST_PROTOCOL)
                exp.write_json(meta,dict(protocol_sha256=exp.sha(OUT/'protocol.json'),sha256=exp.sha(path)))
            bank[sig]=head;info.append(dict(signature=sig,rows=len(yy),weight_mass=float(ww.sum()),n_iter=int(head[-1].n_iter_)))
            print('FIT',d,len(bank),key,'seconds',round(time.monotonic()-start,1),flush=True)
        return bank[sig]
    with threadpool_limits(limits=2):
        for m in ALL_METHODS:
            for s in exp.STAGES:
                original=fixed['fc1' if s=='fc1' else 'fc2']
                if s!='fc2_input':
                    pooled=previous.pool_histories([train[f'{m}__{h}__{s}'] for h in exp.HISTORIES],y,groups)
                    shared=head_for(*pooled,f'shared__{m}__{s}')
                for h in exp.HISTORIES:
                    key=f'{m}__{h}__{s}';x=test[key];ss=dict(fixed=original.decision_function(x))
                    if s!='fc2_input':
                        separate=head_for(train[key],y,exp.weights(y,groups),'separate__'+key)
                        ss.update(shared=shared.decision_function(x),separate=separate.decision_function(x),
                            scaler_only=original[-1].decision_function(shared[0].transform(x)))
                    for kind,v in ss.items():
                        scores[f'{kind}__{key}']=v;summary.append(dict(dataset=d,method=m,history=h,stage=s,readout=kind,**previous.metrics(v,ty)))
    with (dest/'models.pckl').open('wb') as f:pickle.dump(dict(heads=bank,mapping=mapping),f,protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(dest/'scores.npz',**scores,row=trows,y=ty,concept=tgroups)
    exp.write_json(dest/'metrics.json',summary)
    exp.write_json(dest/'audit.json',dict(passed=True,train_count=len(y),test_count=len(ty),train_rows=rows.tolist(),test_rows=trows.tolist(),
        concepts_disjoint=True,test_not_used_for_fit=True,C=1,no_search=True,shared_history_weight_mass=len(y),
        models=len(bank),mapping=mapping,fit_info=info,training_marker_sha256=exp.sha(OUT/d/'training/complete.json'),seconds=time.monotonic()-start))
    marker(dest,['models.pckl','scores.npz','metrics.json','audit.json']);exp.validate(old)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','worker']);p.add_argument('--dataset',choices=['imagenet','things']);args=p.parse_args()
    old,_=protocol();exp.setup()
    if args.action=='worker':
        assert args.dataset;generate(args.dataset,'pilot',old);generate(args.dataset,'training',old);fit(args.dataset,old)
        protocol();print('DONE',args.dataset,flush=True)
