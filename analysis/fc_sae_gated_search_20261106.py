"""Six-point train-only selection, then a frozen clean pilot and original-FC SVMs."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
import sys,argparse,pickle,warnings,time
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
from analysis import fc_preview_refit_20261105 as refit
from analysis import fc_sae_gain_pilot_20261105 as first
from analysis.fc_sae_gain_diagnosis_20261106 import load_models
from analysis.report_fc_preview_pilot_20261105 import characterize,mean
from predify_experiments.fc_sae_gain_20261105 import model_hash,Memory
from predify_experiments.fc_sae_gated_20261106 import memory_from_preview,query,cascade

OUT=ROOT/'Hebbian-CNN/Log/fc_sae_gated_six_20261106'
DOC=ROOT/'docs/fc_sae_gated_six_20261106'
GRID=[dict(name=f'eps{e}_lambda{l}',epsilon=e,**{'lambda':l},steps=5) for e in (0.,.01,.03) for l in (.01,.03)]
HIST=('exact','different','none');LAYERS=('fc1','fc2');METHODS=('full','pool_only','scale_both','shuffled')

def protocol():
    old=exp.read(exp.OUT/'protocol.json');exp.validate(old)
    deps=['predify_experiments/fc_sae_gated_20261106.py','analysis/fc_sae_gain_diagnosis_20261106.py',
          'analysis/fc_preview_pilot_20261105.py','analysis/fc_preview_refit_20261105.py']
    p=dict(source_sha256=exp.sha(__file__),dependencies={f:exp.sha(ROOT/f) for f in deps},grid=GRID,methods=METHODS,
        calibration='pooled training Exact/Different vanilla queries: active-unit RMS gated q; mean strictly-positive kappa; constants frozen per dataset/layer',
        update='u=tanh(P^T A^T[(z>0)*Wz]/c); m=epsilon*(u-sum(x*u)/sum(x)); tau=lambda*(mean(r)+std_population(r))*kappa/(kappa+kappa0); next=ReLU[x*(1+m)-tau]; five steps',
        selection='all 8 train dataset/layer/history cells: suppression 1..15%, fixed BA loss<=1pp, mean true-class margin>=95% vanilla. Maximize worst Fisher change, then mean Fisher, then worst margin retention. Certification additionally requires all Fisher changes>=0.',
        fallback='if no feasible point, freeze least guard-violating diagnostic point; never claim joint success or select using pilot',
        caveat='training-side screen; proxies and fixed vanilla SVM already fitted on these folds, not inner cross-validation',
        svm='original4096-D FC only; fixed vanilla; shared adapted StandardScaler+LinearSVC C1, three histories 1/3 weight, max_iter30000; numerical convergence retry100000 only if needed',
        targets_readonly=True,trial_reset=True,preview_only_write=True,clean_only=True,proxy_frozen=True,
        source_protocol_sha256=exp.sha(exp.OUT/'protocol.json'),datasets={})
    for d in ('imagenet','things'):
        refit.verify(refit.OUT/d/'inputs');first.verify(first.OUT/d/'proxy')
        p['datasets'][d]=dict(training_inputs=exp.sha(refit.OUT/d/'inputs/complete.json'),
            pilot_inputs=exp.sha(exp.OUT/d/'inputs/complete.json'),proxies=exp.sha(first.OUT/d/'proxy/complete.json'),
            fixed_svm=exp.sha(exp.OUT/d/'reference_svm.pckl'))
    import json
    p=json.loads(json.dumps(p));OUT.mkdir(exist_ok=True,parents=True);DOC.mkdir(exist_ok=True,parents=True)
    if (OUT/'protocol.json').exists():assert exp.read(OUT/'protocol.json')==p,'Frozen protocol differs'
    else:exp.write_json(OUT/'protocol.json',p)
    return old,p

def data(d,scope):
    if scope=='pilot':return exp.features(d)
    with np.load(refit.OUT/d/'inputs/features.npz') as f:return {k:f[k] for k in f.files}

def gpu(a):return torch.from_numpy(a).cuda()

@torch.inference_mode()
def run(d,a,models,w,b,cal,cfg,method='full',audit=False):
    before=[model_hash(p) for p in models];reps={};records=[]
    x=gpu(a['target_fc1']);ps=[torch.stack([exp.permutations(d,int(i),64)[k] for i in a['row']]).cuda().long() for k in range(2)]
    for h,idx in (('exact',0),('different',1),('none',None)):
        previews=[None,None] if idx is None else [gpu(a['preview_'+l][:,idx]) for l in LAYERS]
        rr,meta=cascade(x,previews,models,w,b,cal,cfg,method,ps)
        assert meta['writes']==([0,0] if idx is None else [2,2]) and meta['before']==meta['after']
        for l,v in rr.items():reps[h,l]=v.cpu().numpy()
        if audit:records.append(dict(history=h,method=method,rows=a['row'].tolist(),**meta))
    assert before==[model_hash(p) for p in models]
    return reps,records

def resources(d,a,w,b):
    with (exp.OUT/d/'reference_svm.pckl').open('rb') as f:heads=pickle.load(f)
    with np.load(exp.OUT/d/'nuisance_arrays.npz') as f:maps={l:f[l+'_quartiles'] for l in LAYERS}
    with torch.inference_mode():base=dict(fc1=a['target_fc1'],fc2=torch.relu(torch.nn.functional.linear(gpu(a['target_fc1']),w,b)).cpu().numpy())
    return heads,maps,base

def stats(a,reps,heads,maps,base):
    result={};per={}
    for h in HIST:
        for l in LAYERS:
            result[h,l],per[h,l]=characterize(reps[h,l],heads[l].decision_function(reps[h,l]),a['y'],a['concept'],maps[l],base[l])
    return result,per

def changes(d,name,a,ss,per):
    rows=[]
    for h in ('exact','different'):
        for l in LAYERS:
            s=ss[h,l];base=ss['none',l];bm=base['margin']
            rows.append(dict(dataset=d,name=name,history=h,layer=l,
                suppression=mean(100*(1-per[h,l]['activation']/per['none',l]['activation']),a['y'],a['concept']),
                exact_different_suppression=mean(100*(1-per['exact',l]['activation']/per['different',l]['activation']),a['y'],a['concept']),
                fisher_change=100*(s['fisher']/base['fisher']-1),within_change=100*(s['within']/base['within']-1),
                between_change=100*(s['between']/base['between']-1),preferred_pp=100*(s['preferred_share']-base['preferred_share']),
                weak_share_pp=100*(s['weak_share']-base['weak_share']),nonzero_pp=100*(s['nonzero']-base['nonzero']),
                fixed_ba=s['svm_accuracy'],ba_change_pp=100*(s['svm_accuracy']-base['svm_accuracy']),
                fixed_margin=s['margin'],vanilla_fixed_margin=bm,margin_change=s['margin']-bm,
                margin_retention=s['margin']/bm if bm>1e-12 else None,cosine=s['cosine_to_vanilla']))
    return rows

@torch.inference_mode()
def calibration(d,a,models,w,b):
    xs=[gpu(a['target_fc1'])];xs.append(torch.relu(torch.nn.functional.linear(xs[0],w,b)));out=[]
    for k,(p,x) in enumerate(zip(models,xs)):
        qq=[];kk=[]
        for idx in (0,1):
            m=memory_from_preview(p,gpu(a['preview_'+LAYERS[k]][:,idx]),len(x),'cuda',x.dtype)
            z,h,q,kappa=query(p,x,m);qq.append(q[x>0]);kk.append(kappa.flatten())
        q=torch.cat(qq);kappa=torch.cat(kk);positive=kappa[kappa>0]
        out.append(dict(q_rms=float(q.square().mean().sqrt().clamp_min(1e-8)),kappa0=float(positive.mean().clamp_min(1e-8)),
            targets=len(x),histories=['exact','different'],positive_kappa_count=len(positive)))
    return out

def screen():
    old,p=protocol();exp.setup();w,b=exp.load_bridge();rows=[];cal={}
    for d in ('imagenet','things'):
        models=load_models(d);a=data(d,'training');assert a['row'].tolist()==old['datasets'][d]['training_rows']
        cal[d]=calibration(d,a,models,w,b);heads,maps,base=resources(d,a,w,b)
        for cfg in GRID:
            reps,_=run(d,a,models,w,b,cal[d],cfg);ss,per=stats(a,reps,heads,maps,base);rr=changes(d,cfg['name'],a,ss,per);rows.extend(rr)
            print('SCREEN',d,cfg['name'],'suppression',[round(r['suppression'],3) for r in rr],
                'Fisher',[round(r['fisher_change'],3) for r in rr],'margin',[round(r['margin_retention'],4) for r in rr],flush=True)
    rank=[]
    for cfg in GRID:
        rr=[r for r in rows if r['name']==cfg['name']]
        violation=sum(max(1-r['suppression'],0)+max(r['suppression']-15,0)+max(-1-r['ba_change_pp'],0)+100*max(.95-r['margin_retention'],0) for r in rr)
        rank.append(dict(name=cfg['name'],eligible=violation==0,guard_violation=violation,
            min_fisher=min(r['fisher_change'] for r in rr),mean_fisher=float(np.mean([r['fisher_change'] for r in rr])),
            min_margin_retention=min(r['margin_retention'] for r in rr)))
    candidates=[r for r in rank if r['eligible']]
    best=max(candidates,key=lambda r:(r['min_fisher'],r['mean_fisher'],r['min_margin_retention'])) if candidates else min(rank,key=lambda r:(r['guard_violation'],-r['min_fisher'],-r['mean_fisher']))
    cfg=next(c for c in GRID if c['name']==best['name'])
    selection=dict(protocol_sha256=exp.sha(OUT/'protocol.json'),ranking=rank,config=cfg,
        certified=best['eligible'] and best['min_fisher']>=0,training_only=True,heldout_used=False)
    exp.write_json(OUT/'calibration.json',cal);exp.write_json(OUT/'training_screen.json',rows)
    exp.write_json(OUT/'selection.json',selection);exp.validate(old);print('FROZEN',selection,flush=True)

def svm_metrics(scores,y,groups):
    margin=scores[np.arange(len(y)),y]-np.where(np.arange(scores.shape[1])[None,:]==y[:,None],-np.inf,scores).max(1)
    correct=scores.argmax(1)==y;ww=exp.weights(y,groups)
    return dict(**refit.metrics(scores,y),margin=mean(margin,y,groups),
        correct_only_margin=float(np.average(margin[correct],weights=ww[correct])) if correct.any() else None),margin

def fit_head(dest,sig,tx,yy,ww):
    path=dest/(sig+'.pckl');note=path.with_suffix('.json')
    if note.exists():
        info=exp.read(note);assert info['training_signature']==sig and info['sha256']==exp.sha(path)
        with path.open('rb') as f:head=pickle.load(f)
        return head,info
    cap=30000;attempts=[]
    while True:
        head=make_pipeline(StandardScaler(),LinearSVC(C=1,dual='auto',max_iter=cap,random_state=exp.SEED))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);head.fit(tx,yy,standardscaler__sample_weight=ww,linearsvc__sample_weight=ww)
            assert head[-1].n_iter_<cap;break
        except ConvergenceWarning:
            attempts.append(dict(max_iter=cap,converged=False))
            if cap>=100000:raise
            cap=100000
    assert head[-1].n_features_in_==4096
    with path.open('wb') as f:pickle.dump(head,f)
    info=dict(training_signature=sig,sha256=exp.sha(path),n_iter=int(head[-1].n_iter_),max_iter=cap,previous_attempts=attempts,C=1)
    exp.write_json(note,info);return head,info

def worker(d):
    start=time.monotonic();old,p=protocol();exp.setup();selection=exp.read(OUT/'selection.json');assert selection['protocol_sha256']==exp.sha(OUT/'protocol.json')
    cfg=selection['config'];cal=exp.read(OUT/'calibration.json')[d];frozen=exp.sha(OUT/'selection.json');models=load_models(d);state=[model_hash(p) for p in models]
    w,b=exp.load_bridge();tr=data(d,'training');te=data(d,'pilot')
    assert set(tr['concept']).isdisjoint(te['concept']) and set(tr['row']).isdisjoint(te['row'])
    heads,maps,base=resources(d,te,w,b);dest=OUT/d;dest.mkdir(exist_ok=True);hd=dest/'heads';hd.mkdir(exist_ok=True)
    summary=[];contrast=[];readouts=[];perrows=[];audit_records=[];fitinfo=[];stored={};scorebank={};bank={}
    for method in METHODS:
        trainrep,_=run(d,tr,models,w,b,cal,cfg,method)
        reps,records=run(d,te,models,w,b,cal,cfg,method,audit=True);audit_records.extend(records)
        ss,per=stats(te,reps,heads,maps,base);contrast.extend(changes(d,method,te,ss,per))
        for h in HIST:
            for l in ('fc1','fc2_input','fc2'):stored[f'{method}__{h}__{l}']=reps[h,l]
            for l in LAYERS:
                summary.append(dict(dataset=d,method=method,history=h,layer=l,**ss[h,l]))
                for j,row in enumerate(te['row']):perrows.append(dict(dataset=d,method=method,history=h,layer=l,target=int(row),
                    concept=str(te['concept'][j]),category=int(te['y'][j]),activation=float(per[h,l]['activation'][j]),
                    preferred_share=float(per[h,l]['preferred_share'][j]),fixed_margin=float(per[h,l]['margin'][j]),fixed_correct=int(per[h,l]['correct'][j])))
        np.savez_compressed(dest/(method+'_training.npz'),**{f'{h}__{l}':v for (h,l),v in trainrep.items()},row=tr['row'],y=tr['y'],concept=tr['concept'])
        for l in LAYERS:
            tx,yy,ww=refit.pool_histories([trainrep[h,l] for h in HIST],tr['y'],tr['concept']);sig=refit.signature(tx,yy,ww)
            if sig not in bank:bank[sig]=fit_head(hd,sig,tx,yy,ww)
            shared,info=bank[sig];fitinfo.append(dict(method=method,layer=l,**info))
            print('SVM',d,method,l,'iterations',info['n_iter'],'elapsed',round(time.monotonic()-start,1),flush=True)
            for kind,clf in (('fixed',heads[l]),('shared',shared)):
                none=clf.decision_function(reps['none',l]);none_mm,_=svm_metrics(none,te['y'],te['concept'])
                for h in HIST:
                    scores=clf.decision_function(reps[h,l]);mm,margins=svm_metrics(scores,te['y'],te['concept'])
                    readouts.append(dict(dataset=d,method=method,history=h,layer=l,kind=kind,**mm,
                        margin_change_vs_none=mm['margin']-none_mm['margin'],prediction_agreement_vs_none=float(np.mean(scores.argmax(1)==none.argmax(1))),
                        max_score_change_vs_none=float(np.abs(scores-none).max())))
                    scorebank[f'{kind}__{method}__{h}__{l}']=scores
                    for j,row in enumerate(te['row']):perrows.append(dict(dataset=d,method=method,history=h,layer=l,target=int(row),
                        concept=str(te['concept'][j]),category=int(te['y'][j]),readout=kind,prediction=int(scores[j].argmax()),margin=float(margins[j])))
    mean_error=max(float(np.abs(stored[f'scale_both__{h}__{l}'].mean(1)-stored[f'full__{h}__{l}'].mean(1)).max()) for h in HIST for l in LAYERS);assert mean_error<2e-6
    # Reverse and singleton replay ensure no inter-trial memory, including after all controls.
    rev={k:v[::-1].copy() for k,v in te.items()};reverse,_=run(d,rev,models,w,b,cal,cfg)
    order_error=max(float(np.abs(reverse[h,l][::-1]-stored[f'full__{h}__{l}']).max()) for h in HIST for l in LAYERS);assert order_error<3e-5
    tiny={k:v[:1] for k,v in te.items()};solo,_=run(d,tiny,models,w,b,cal,cfg)
    batch_error=max(float(np.abs(solo[h,l]-stored[f'full__{h}__{l}'][:1]).max()) for h in HIST for l in LAYERS);assert batch_error<3e-5
    with torch.inference_mode():
        bridge_error=max(float(np.abs(torch.relu(torch.nn.functional.linear(gpu(stored[f'{m}__{h}__fc1']),w,b)).cpu().numpy()-stored[f'{m}__{h}__fc2_input']).max()) for m in METHODS for h in HIST)
    assert bridge_error<2e-5 and state==[model_hash(p) for p in models] and exp.sha(OUT/'selection.json')==frozen
    # Independent preview outer-product acceptance on every actual pilot trial and layer.
    preview_error=0.
    with torch.inference_mode():
        for idx in (0,1):
            for p,l in zip(models,LAYERS):
                previews=gpu(te['preview_'+l][:,idx]);mm=memory_from_preview(p,previews,len(te['row']),'cuda',torch.float32)
                for j in range(len(te['row'])):
                    legacy=Memory(64,device='cuda')
                    for z in p.encode(previews[j]):legacy.write(z)
                    legacy.freeze();preview_error=max(preview_error,float((legacy.W-mm.W[j]).abs().max()))
                    try:mm.write(p.encode(previews[:,0]));raise AssertionError('Target write allowed')
                    except RuntimeError:pass
    assert preview_error<1e-6
    audit=dict(passed=True,dataset=d,protocol_sha256=exp.sha(OUT/'protocol.json'),selection_sha256=frozen,calibration_sha256=exp.sha(OUT/'calibration.json'),
        training_only_selection=True,clean_only=True,train=len(tr['row']),test=len(te['row']),concepts_disjoint=True,
        trial_reset=True,preview_only_write=True,target_readonly=True,no_preview_identity=True,proxy_unchanged=True,
        proxy_hashes=state,original_FC_dimensions=4096,scale_mean_error=mean_error,order_error=order_error,batch_error=batch_error,
        bridge_error=bridge_error,preview_write_error=preview_error,fit_info=fitinfo,records=audit_records,seconds=time.monotonic()-start)
    for name,rows in (('summary',summary),('contrasts',contrast),('readouts',readouts),('per_target',perrows)):exp.write_json(dest/(name+'.json'),rows)
    np.savez_compressed(dest/'representations.npz',**stored,row=te['row'],y=te['y'],concept=te['concept'])
    np.savez_compressed(dest/'scores.npz',**scorebank,row=te['row'],y=te['y'],concept=te['concept'])
    exp.write_json(dest/'audit.json',audit);exp.validate(old)
    files=[f for f in dest.iterdir() if f.is_file() and f.name!='complete.json']
    exp.write_json(dest/'complete.json',dict(passed=True,protocol_sha256=exp.sha(OUT/'protocol.json'),files={f.name:exp.sha(f) for f in files}))
    print('DONE',d,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('screen','worker'));parser.add_argument('--dataset',choices=('imagenet','things'));args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.action=='screen':screen()
        else:assert args.dataset;worker(args.dataset)
