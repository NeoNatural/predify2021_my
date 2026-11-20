"""Twelve-point training-only screen; frozen SAE, original FC readouts, no backbone edits."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
import sys,pickle,itertools,warnings,time
from pathlib import Path
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.fc_sae_gain_diagnosis_20261106 import OUT,DOC,load_models,memory,read
from analysis import fc_preview_pilot_20261105 as exp
from analysis import fc_preview_refit_20261105 as refit
from analysis import fc_sae_gain_pilot_20261105 as pilot
from analysis.report_fc_preview_pilot_20261105 import characterize,mean
from predify_experiments.fc_sae_gain_20261105 import model_hash,Memory

GRID=[dict(name=f'e{eta}_b{beta}_g{gamma}',eta=eta,beta=beta,gamma=gamma,alpha=1.,steps=5)
      for eta,beta,gamma in itertools.product((.05,.15),(4.,12.),(0.,.25,1.))]
BASE=dict(name='original',eta=.75,beta=4.,gamma=1.,alpha=.5,steps=5)
HIST=('exact','different','none');LAYERS=('fc1','fc2');METHODS=('full','pool_only','scale1','scale2','scale_both','shuffled')

def features(d,scope):
    if scope=='pilot':return exp.features(d)
    refit.verify(refit.OUT/d/'inputs')
    with np.load(refit.OUT/d/'inputs/features.npz') as f:return {k:f[k] for k in f.files}

@torch.inference_mode()
def side(x,p,W,c,config,gamma=None):
    frozen=W.clone();r=x.clone();gamma=config['gamma'] if gamma is None else gamma
    for _ in range(config['steps']):
        _,_,q=read(p,r,W);m=config['eta']*torch.tanh(q/c)
        tau=config['beta']*(x*m.clamp_min(0)).mean(-1,keepdim=True)
        candidate=(x*(1+gamma*m)-tau).clamp_min(0)
        r=(1-config['alpha'])*r+config['alpha']*candidate
        assert torch.equal(W,frozen) and (r>=0).all() and torch.count_nonzero(r[x==0])==0
    return r

@torch.inference_mode()
def prepare(d,a,models,w,b):
    xx=torch.from_numpy(a['target_fc1']).cuda();raw2=torch.relu(torch.nn.functional.linear(xx,w,b))
    memories={}
    for h,idx in (('exact',0),('different',1),('none',None)):
        memories[h]=[]
        for layer,p in zip(LAYERS,models):
            W=torch.zeros(len(xx),64,64,device='cuda') if idx is None else memory(p,torch.from_numpy(a['preview_'+layer][:,idx]).cuda())
            # Check actual legacy lifecycle and exact write formula for the first trial.
            single=Memory(64,device='cuda')
            assert single.W.count_nonzero()==0 and single.writes==0
            if idx is not None:
                for z in p.encode(torch.from_numpy(a['preview_'+layer][0,idx]).cuda()):single.write(z)
            single.freeze();assert torch.allclose(single.W,W[0],atol=1e-7,rtol=1e-6)
            try:single.write(torch.ones(64,device='cuda'));raise AssertionError('Target write accepted')
            except RuntimeError:pass
            memories[h].append(W)
    ps=[torch.stack([p[k] for p in [exp.permutations(d,int(row),64) for row in a['row']]]).cuda().long() for k in range(2)]
    shuffled={h:[W.gather(1,ps[k][:,:,None].expand(-1,-1,64)).gather(2,ps[k][:,None,:].expand(-1,64,-1))
        for k,W in enumerate(memories[h])] for h in HIST}
    return xx,raw2,memories,shuffled

@torch.inference_mode()
def cascade(prepared,models,w,b,cal,config,method='full'):
    x,raw2,memories,shuffled=prepared;reps={}
    for h in HIST:
        W=(shuffled if method=='shuffled' else memories)[h]
        gamma=0. if method=='pool_only' else config['gamma']
        one=side(x,models[0],W[0],cal[0],config,gamma)
        twoin=torch.relu(torch.nn.functional.linear(one,w,b))
        two=side(twoin,models[1],W[1],cal[1],config,gamma)
        if method in ('scale1','scale_both'):
            one=x*(one.mean(-1,keepdim=True)/x.mean(-1,keepdim=True))
            twoin=torch.relu(torch.nn.functional.linear(one,w,b))
            if method=='scale1':two=side(twoin,models[1],W[1],cal[1],config,gamma)
        if method in ('scale2','scale_both'):two=twoin*(two.mean(-1,keepdim=True)/twoin.mean(-1,keepdim=True))
        reps[h,'fc1']=one.cpu().numpy();reps[h,'fc2']=two.cpu().numpy();reps[h,'fc2_input']=twoin.cpu().numpy()
    for l,base in (('fc1',x),('fc2',raw2)):
        assert np.allclose(reps['none',l],base.cpu().numpy(),atol=2e-6,rtol=2e-6)
    return reps

def characterize_all(d,a,reps,heads,maps,base):
    stats={};pers={}
    for (h,l),v in reps.items():
        if l=='fc2_input':continue
        ss,per=characterize(v,heads[l].decision_function(v),a['y'],a['concept'],maps[l],base[l]);stats[h,l]=ss;pers[h,l]=per
    return stats,pers

def contrasts(d,name,stats,pers,a,basestats):
    rows=[]
    for h in ('exact','different'):
        for l in LAYERS:
            ss=stats[h,l];bs=basestats[l];left=pers[h,l];none=pers['none',l]
            rows.append(dict(dataset=d,name=name,history=h,layer=l,
                suppression=mean(100*(1-left['activation']/none['activation']),a['y'],a['concept']),
                fisher_change=100*(ss['fisher']/bs['fisher']-1),ba_change_pp=100*(ss['svm_accuracy']-bs['svm_accuracy']),
                nonzero_change_pp=100*(ss['nonzero']-bs['nonzero']),preferred_pp=100*(ss['preferred_share']-bs['preferred_share']),
                margin_change=ss['margin']-bs['margin'],fixed_ba=100*ss['svm_accuracy'],fixed_auc=ss['svm_auc']))
    return rows

def main():
    exp.setup();old=exp.read(exp.OUT/'protocol.json');exp.validate(old);OUT.mkdir(exist_ok=True,parents=True);DOC.mkdir(exist_ok=True,parents=True)
    protocol=dict(grid=GRID,base=BASE,calibration='one active-unit RMS of q per layer/dataset, pooled training Exact/Different on vanilla queries; fixed afterwards',
        selection='all eight training dataset/layer/history cells: suppression 1..15%, fixed BA drop<=1pp. Among eligible maximize worst-cell Fisher change then mean Fisher. No candidate certified unless all eight Fisher changes >=0.',
        caveat='training screen with proxies and fixed SVM already fitted on these training folds; not independent cross-validation',
        proxy_frozen=True,no_backbone_edits=True,heldout_used_for_selection=False,source_sha256=exp.sha(__file__),
        dependencies={str(p.relative_to(ROOT)):exp.sha(p) for p in (ROOT/'analysis/fc_sae_gain_diagnosis_20261106.py',ROOT/'predify_experiments/fc_sae_gain_20261105.py')})
    exp.write_json(OUT/'search_protocol.json',protocol);w,b=exp.load_bridge();bundles={};search=[];diagnostics=[]
    for d in ('imagenet','things'):
        models=load_models(d);state=[model_hash(p) for p in models];a=features(d,'training');assert a['row'].tolist()==old['datasets'][d]['training_rows']
        prepared=prepare(d,a,models,w,b);x,raw2,mems,_=prepared
        with (exp.OUT/d/'reference_svm.pckl').open('rb') as f:heads=pickle.load(f)
        with np.load(exp.OUT/d/'nuisance_arrays.npz') as f:maps={l:f[l+'_quartiles'] for l in LAYERS}
        base={l:v.cpu().numpy() for l,v in zip(LAYERS,(x,raw2))};cal=[]
        with torch.inference_mode():
            for k,xx in enumerate((x,raw2)):
                qs=[read(models[k],xx,mems[h][k])[2][xx>0] for h in ('exact','different')]
                cal.append(float(torch.cat(qs).square().mean().sqrt().clamp_min(1e-8)))
        original=cascade(prepared,models,w,b,[1.,1.],BASE)
        pilot.verify(pilot.OUT/d/'training')
        with np.load(pilot.OUT/d/'training/representations.npz') as f:
            replay=max(float(np.abs(original[h,l]-f[f'hebbian__{h}__{l}']).max()) for h in HIST for l in LAYERS)
        assert replay<3e-5,replay
        bs,_=characterize_all(d,a,original,heads,maps,base);basestats={l:bs['none',l] for l in LAYERS}
        bundles[d]=(models,state,a,prepared,heads,maps,base,cal,basestats)
        diagnostics.append(dict(dataset=d,calibration=cal,old_replay_max_error=replay,training_targets=len(x)))
        for cfg in GRID:
            reps=cascade(prepared,models,w,b,cal,cfg);ss,pp=characterize_all(d,a,reps,heads,maps,base)
            rows=contrasts(d,cfg['name'],ss,pp,a,basestats);search.extend(rows)
            print('SCREEN',d,cfg['name'],'min Fisher',min(r['fisher_change'] for r in rows),'suppression',[round(r['suppression'],2) for r in rows],flush=True)
        assert state==[model_hash(p) for p in models]
    exp.write_json(OUT/'training_screen.json',search);exp.write_json(DOC/'training_screen.json',search);exp.write_json(OUT/'calibration.json',diagnostics)
    ranked=[]
    for cfg in GRID:
        rows=[r for r in search if r['name']==cfg['name']]
        eligible=all(1<=r['suppression']<=15 and r['ba_change_pp']>=-1 for r in rows)
        ranked.append(dict(name=cfg['name'],eligible=eligible,min_fisher=min(r['fisher_change'] for r in rows),mean_fisher=float(np.mean([r['fisher_change'] for r in rows]))))
    eligible=[r for r in ranked if r['eligible']]
    selection=dict(ranking=ranked,selection_training_only=True,certified=False,config=None)
    if eligible:
        best=max(eligible,key=lambda r:(r['min_fisher'],r['mean_fisher']));selection.update(config=next(c for c in GRID if c['name']==best['name']),certified=best['min_fisher']>=0)
    exp.write_json(OUT/'selection.json',selection);exp.write_json(DOC/'selection.json',selection)
    print('SELECTION',selection,flush=True)
    # One frozen choice is tested once, even if training screen failed certification;
    # then called diagnostic only. Never choose another point based on pilot output.
    if selection['config'] is None:return
    cfg=selection['config'];all_metrics=[];readouts=[];audits=[]
    for d,bundle in bundles.items():
        models,state,tr,prep,heads,maps,base,cal,bs=bundle;te=features(d,'pilot');tp=prepare(d,te,models,w,b)
        assert set(tr['concept']).isdisjoint(te['concept'])
        tbase={l:v.cpu().numpy() for l,v in zip(LAYERS,tp[:2])}
        tbs,_=characterize_all(d,te,{(h,l):v for h in HIST for l,v in tbase.items()},heads,maps,tbase);tbs={l:tbs['none',l] for l in LAYERS}
        dest=OUT/d;dest.mkdir(exist_ok=True);stored={};max_meanerror=0.;fitted=[];headbank={}
        for method in METHODS:
            trainrep=cascade(prep,models,w,b,cal,cfg,method);testrep=cascade(tp,models,w,b,cal,cfg,method)
            ss,pp=characterize_all(d,te,testrep,heads,maps,tbase);rr=contrasts(d,method,ss,pp,te,tbs)
            for r in rr:
                l=r['layer'];r['exact_different_suppression']=mean(100*(1-pp['exact',l]['activation']/pp['different',l]['activation']),te['y'],te['concept'])
                r['cosine']=ss[r['history'],l]['cosine_to_vanilla']
            all_metrics.extend(rr)
            for (h,l),v in testrep.items():stored[f'{method}__{h}__{l}']=v
            for l in LAYERS:
                tx,yy,ww=refit.pool_histories([trainrep[h,l] for h in HIST],tr['y'],tr['concept'])
                sig=refit.signature(tx,yy,ww);headpath=dest/f'{method}_{l}_shared.pckl'
                if sig in headbank:head=headbank[sig]
                elif headpath.exists():
                    with headpath.open('rb') as f:head=pickle.load(f)
                    assert np.max(np.abs(head[0].mean_-np.average(tx.astype(np.float64),axis=0,weights=ww)))<1e-9
                    headbank[sig]=head
                else:
                    head=make_pipeline(StandardScaler(),LinearSVC(C=1,dual='auto',max_iter=30000,random_state=exp.SEED))
                    with threadpool_limits(limits=2),warnings.catch_warnings():
                        warnings.simplefilter('error',ConvergenceWarning);head.fit(tx,yy,standardscaler__sample_weight=ww,linearsvc__sample_weight=ww)
                    headbank[sig]=head
                assert head[-1].n_features_in_==4096 and head[-1].n_iter_<30000
                with headpath.open('wb') as f:pickle.dump(head,f)
                exp.write_json(headpath.with_suffix('.json'),dict(training_signature=sig,sha256=exp.sha(headpath)))
                fitted.append(dict(method=method,layer=l,n_iter=int(head[-1].n_iter_),training_signature=sig))
                for h in HIST:
                    for kind,clf in (('fixed',heads[l]),('shared',head)):
                        scores=clf.decision_function(testrep[h,l]);mm=refit.metrics(scores,te['y'])
                        readouts.append(dict(dataset=d,method=method,layer=l,history=h,kind=kind,**mm))
                print('REFIT',d,method,l,flush=True)
        for h in HIST:
            for method,l in (('scale1','fc1'),('scale2','fc2'),('scale_both','fc1'),('scale_both','fc2')):
                err=float(np.abs(stored[f'{method}__{h}__{l}'].mean(1)-stored[f'full__{h}__{l}'].mean(1)).max());max_meanerror=max(max_meanerror,err)
        assert max_meanerror<2e-6
        # Independent per-trial replay detects cross-trial interference.
        for h in HIST:
            tiny=(tp[0][:1],tp[1][:1],{k:[v[:1] for v in vv] for k,vv in tp[2].items()},{k:[v[:1] for v in vv] for k,vv in tp[3].items()})
            check=cascade(tiny,models,w,b,cal,cfg)
            assert max(float(np.abs(check[h,l]-stored[f'full__{h}__{l}'][:1]).max()) for l in LAYERS)<3e-5
        assert state==[model_hash(p) for p in models]
        np.savez_compressed(dest/'representations.npz',**stored,row=te['row'],y=te['y'],concept=te['concept'])
        audits.append(dict(dataset=d,passed=True,scale_mean_error=max_meanerror,train=len(tr['row']),test=len(te['row']),
            train_test_concepts_disjoint=True,target_weights_unchanged=True,preview_writes_only=True,no_preview_identity=True,
            proxy_state_unchanged=True,shared_heads=fitted))
    exp.validate(old);exp.write_json(OUT/'pilot_metrics.json',all_metrics);exp.write_json(DOC/'pilot_metrics.json',all_metrics)
    exp.write_json(OUT/'readouts.json',readouts);exp.write_json(DOC/'readouts.json',readouts)
    exp.write_json(OUT/'audit.json',dict(passed=True,datasets=audits,no_original_model_edits=True,heldout_selection=False))
    print('DONE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
