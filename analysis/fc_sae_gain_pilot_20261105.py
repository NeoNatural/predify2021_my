"""Tiny gain-only SAE pilot: one configuration, training folds only, physical FC readouts."""
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
from analysis import fc_sae_acceptance_20261105 as source_sae
from predify_experiments.fc_sae_gain_20261105 import TrialEngine,GainProxy,Projection,Memory,FIT,UPDATE,METHODS,budget,model_hash

OUT=ROOT/'Hebbian-CNN/Log/fc_sae_gain_pilot_20261105'
DOC=ROOT/'docs/fc_sae_gain_pilot_20261105'
PLAN=ROOT/'docs/fc_sae_gain_only_20261105/设计修订与参数预算.md'
ALL_METHODS=METHODS+('inhib_scale',)

def protocol():
    old=exp.read(exp.OUT/'protocol.json');exp.validate(old);previous.protocol()
    deps=['predify_experiments/fc_sae_gain_20261105.py','predify_experiments/fc_preview_memory_20261105.py',
        'analysis/fc_preview_pilot_20261105.py','analysis/fc_preview_refit_20261105.py','analysis/fc_sae_acceptance_20261105.py']
    p=dict(source_sha256=exp.sha(__file__),dependencies={k:exp.sha(ROOT/k) for k in deps},
        pilot_protocol_sha256=exp.sha(exp.OUT/'protocol.json'),refit_protocol_sha256=exp.sha(previous.OUT/'protocol.json'),
        parameters=UPDATE,fit=FIT,plan_sha256=exp.sha(PLAN),methods=ALL_METHODS,histories=exp.HISTORIES,stages=exp.STAGES,
        choices='eta .75, beta4, alpha .5 and five steps prespecified from existing update defaults; no outcome-based selection',
        update='z=E(P r), h=Wz z, q=P^T A^T h, modulation=.75 tanh(q); increase=x max(modulation,0); pool=4 mean(increase); candidate=ReLU(x*(1+modulation)-pool); next=.5r+.5candidate',
        pool_only='retain current-preview W and its implied pool; set physical gain=1; closed-loop five-step rule',
        inhibition='unchanged previous no-memory gate+division reference, not a subtract-family ablation',
        no_preview='Wz=0: gain1 and pool0; full/pool-only equal vanilla; old inhibition is a separate reference',
        svm='unchanged weighted StandardScaler + LinearSVC C1 dual auto max_iter30000; fixed vanilla and shared-history plus auxiliary separate heads',
        adapted_stages=['fc1','fc2'],fc2_input_readout='fixed vanilla FC2 head diagnostic only',
        no_search=True,SAE_gain_only=True,no_reconstruction_or_latent_SVM=True,clean_only=True,
        fit_roles='training folds1-4 vanilla target and unique preview; .5 total weight each; final1000 step; no labels in SAE loss',datasets={})
    for d in ('imagenet','things'):
        exp.verify(exp.OUT/d/'pilot');previous.verify(previous.OUT/d/'inputs')
        p['datasets'][d]=dict(test_marker=exp.sha(exp.OUT/d/'pilot/complete.json'),
            train_marker=exp.sha(previous.OUT/d/'inputs/complete.json'),fixed_svm=exp.sha(exp.OUT/d/'reference_svm.pckl'),
            training_rows=old['datasets'][d]['training_rows'])
    # JSON round-trip makes tuple/list types identical for frozen comparisons.
    import json
    p=json.loads(json.dumps(p));OUT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert exp.read(OUT/'protocol.json')==p,'Gain-only protocol changed'
    else:
        exp.write_json(OUT/'protocol.json',p);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (OUT/'memory_snapshot.py').write_bytes((ROOT/deps[0]).read_bytes())
    return old,p

def fit_proxies(d,old):
    data,metadata,provenance=source_sae.data(d,old);models=[]
    dest=OUT/d/'proxy';dest.mkdir(parents=True,exist_ok=True)
    exp.write_json(dest/'data_provenance.json',provenance)
    for layer,l in enumerate(('fc1','fc2')):
        path=dest/(l+'.pth');note_path=dest/(l+'.json');seed=exp.SEED+(100 if d=='things' else 0)+layer
        projection=Projection(FIT['dimension'],FIT['projection'],seed+1000).cuda()
        with torch.inference_mode():v=projection(torch.from_numpy(data[l]['fit']).cuda())
        values=v.cpu().numpy().astype(np.float64);weights=data[l]['fit_weights']
        scale=float(np.sqrt(np.average(np.mean(values**2,axis=1),weights=weights)))
        center=(np.average(values,axis=0,weights=weights)/scale).astype(np.float32)
        model=GainProxy(projection,center,scale,FIT['latents'],seed+2000).cuda()
        if path.exists():
            note=exp.read(note_path);assert note['protocol_sha256']==exp.sha(OUT/'protocol.json') and note['checkpoint_sha256']==exp.sha(path)
            model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True),strict=True)
        else:
            model.train();ww=torch.tensor(weights,dtype=torch.float32,device='cuda')
            opt=torch.optim.Adam(model.parameters(),lr=FIT['learning_rate'])
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,FIT['steps'],eta_min=FIT['minimum_learning_rate'])
            history=[];start=time.monotonic()
            for step in range(FIT['steps']):
                opt.zero_grad(set_to_none=True);z=model.encode_projected(v);hat=model.decode_projected(z)
                mse=.5*((hat-v)/model.scale).square().sum(-1);l1=z.sum(-1);loss=(ww*(mse+FIT['l1']*l1)).sum()
                assert torch.isfinite(loss);loss.backward();model.project_gradient();opt.step();model.constrain();scheduler.step()
                if step==0 or (step+1)%100==0:
                    row=dict(step=step+1,loss=float(loss.detach()),reconstruction=float((ww*mse).sum().detach()),
                        code_l1=float((ww*l1).sum().detach()),active=float((ww*(z>0).float().sum(-1)).sum().detach()))
                    history.append(row);print('TINY SAE',d,l,row,flush=True)
            torch.save({k:t.detach().cpu() for k,t in model.state_dict().items()},path)
            exp.write_json(note_path,dict(protocol_sha256=exp.sha(OUT/'protocol.json'),checkpoint_sha256=exp.sha(path),
                seed=seed,scale=scale,center_sha256=exp.array_sha(center),fit_array_sha256=exp.array_sha(data[l]['fit']),
                fit_weights_sha256=exp.array_sha(weights),final_step_only=True,no_test_during_fit=True,history=history,seconds=time.monotonic()-start))
        model.eval().requires_grad_(False);assert np.array_equal(model.center.cpu().numpy(),center)
        bg=budget(model,Memory(64,device='cuda'));assert bg==dict(trainable=2112,floating_buffers=33,projection_metadata_bytes=4352,temporary_hebb_values=4096,tensor_payload_bytes=29316)
        assert bg['tensor_payload_bytes']<8194*4
        with torch.inference_mode():
            orthogonality_error=float((model.projection(model.projection.adjoint(torch.eye(32,device='cuda')))-torch.eye(32,device='cuda')).abs().max())
            assert orthogonality_error<1e-6
            aux={}
            for name in ('train','train_preview','development','development_preview'):
                pv=model.projection(torch.from_numpy(data[l][name]).cuda());zz=model.encode_projected(pv);hat=model.decode_projected(zz)
                aux[name]=dict(projected_mse=float((hat-pv).square().mean()),active_latents=float((zz>0).float().sum(-1).mean()))
        exp.write_json(dest/(l+'_acceptance.json'),dict(passed=True,budget=bg,state_sha256=model_hash(model),auxiliary_monitor_only=aux,
            projection_row_orthogonality_max_error=orthogonality_error,
            FC_reconstruction_not_computed=True,latent_or_reconstruction_readout_not_fitted=True))
        models.append(model)
    marker(dest,['data_provenance.json','fc1.pth','fc2.pth','fc1.json','fc2.json','fc1_acceptance.json','fc2_acceptance.json'])
    return models

def marker(dest,names):
    exp.write_json(dest/'complete.json',dict(protocol_sha256=exp.sha(OUT/'protocol.json'),files={n:exp.sha(dest/n) for n in names}))

def verify(dest):
    p=exp.read(dest/'complete.json');assert p['protocol_sha256']==exp.sha(OUT/'protocol.json')
    for n,h in p['files'].items():assert exp.sha(dest/n)==h

def generate(d,scope,old,models):
    dest=OUT/d/scope
    if (dest/'complete.json').exists():verify(dest);return
    if scope=='pilot':a=exp.features(d)
    else:
        previous.verify(previous.OUT/d/'inputs')
        with np.load(previous.OUT/d/'inputs/features.npz') as z:a={k:z[k] for k in z.files}
        assert a['row'].tolist()==old['datasets'][d]['training_rows']
    ns=exp.nuisance(d,old);w,b=exp.load_bridge();engine=TrialEngine(w,b,models,(ns['fc1']['scale'],ns['fc2']['scale']))
    reps={f'{m}__{h}__{s}':[] for m in METHODS for h in exp.HISTORIES for s in exp.STAGES}
    records=[];count=0;steps=0;start=time.monotonic();zero=Memory(64,device='cuda').fingerprint();references={}
    reference_zero=exp.PreviewMemory(4096).fingerprint();accepted=dict(passed=True,target_write_rejected=True,counterfactual_target_weight_invariant=True)
    with torch.inference_mode():
        for j,target in enumerate(a['row']):
            x=torch.from_numpy(a['target_fc1'][j:j+1]).cuda();ps=exp.permutations(d,int(target),64)
            for history,idx in (('exact',0),('different',1),('none',None)):
                p1=[] if idx is None else torch.from_numpy(a['preview_fc1'][j,idx]).cuda()
                p2=[] if idx is None else torch.from_numpy(a['preview_fc2'][j,idx]).cuda()
                for method in METHODS:
                    r,meta=engine.run(x,p1,p2,method,ps);assert meta['reset_sha256']==[zero,zero]
                    assert meta['before_target']==meta['after_target']
                    assert meta['proxy_before']==meta['proxy_after']
                    assert meta['writes']==([0,0] if idx is None or method in ('vanilla','inhibition') else [2,2])
                    for trace in meta['traces']:
                        assert len(trace['updates'])==5
                        expected=reference_zero if method=='inhibition' else meta['before_target'][trace['layer']-1]
                        for step in trace['updates']:assert step['weight_sha256']==expected;steps+=1
                    for s,v in r.items():reps[f'{method}__{history}__{s}'].append(v[0].cpu().numpy())
                    count+=1
                    if scope=='pilot':
                        meta.update(target=int(target),method=method,history=history);records.append(meta)
                        if method=='hebbian':
                            references[j,history]={s:v.clone() for s,v in r.items()}
                            if idx is not None:
                                old_weights=meta['before_target']
                                _,counter=engine.run(x*1.7,p1,p2,method,ps);assert counter['before_target']==old_weights
                                for memory in engine.memories:
                                    assert memory.W.norm()>0
                                    try:memory.write(torch.ones(64,device='cuda'));raise AssertionError('Target write accepted')
                                    except RuntimeError:pass
            if (j+1)%32==0 or scope=='pilot':print('GENERATE',d,scope,j+1,'/',len(a['row']),flush=True)
        if scope=='pilot':
            # Reverse-order replay after intervening histories and controls.
            for j in reversed(range(len(a['row']))):
                for h,idx in (('none',None),('different',1),('exact',0)):
                    x=torch.from_numpy(a['target_fc1'][j:j+1]).cuda()
                    r,meta=engine.run(x,[] if idx is None else torch.from_numpy(a['preview_fc1'][j,idx]).cuda(),
                        [] if idx is None else torch.from_numpy(a['preview_fc2'][j,idx]).cuda(),'hebbian',exp.permutations(d,int(a['row'][j]),64))
                    for s in exp.STAGES:assert torch.equal(r[s],references[j,h][s])
            # Independent FP64 preview outer writes and physical multiplicative cascade.
            import copy
            j=0;x=torch.from_numpy(a['target_fc1'][j:j+1]).cuda().double()
            p1=torch.from_numpy(a['preview_fc1'][j,0]).cuda().double();p2=torch.from_numpy(a['preview_fc2'][j,0]).cuda().double()
            dp=[copy.deepcopy(p).double() for p in models];ed=TrialEngine(w.double(),b.double(),dp,engine.reference_scales);r,meta=ed.run(x,p1,p2,'hebbian')
            def independent(v,proxy,preview):
                z=proxy.encode(preview);u=z/z.norm(dim=1,keepdim=True).clamp_min(1e-8)
                W=.25*u[0,:,None]*u[0,None,:]+.5*u[1,:,None]*u[1,None,:];W.fill_diagonal_(0);state=v.clone()
                for _ in range(5):
                    code=proxy.encode(state);latent=code@W.T;q=proxy.projection.adjoint(latent@proxy.A)
                    mod=.75*torch.tanh(q);pool=4*(v*mod.clamp_min(0)).mean(-1,keepdim=True)
                    state=.5*state+.5*(v*(1+mod)-pool).clamp_min(0)
                return state
            one=independent(x,dp[0],p1);two=independent(torch.relu(torch.nn.functional.linear(one,w.double(),b.double())),dp[1],p2)
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
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'representations.npz',**reps,row=a['row'],y=a['y'],concept=a['concept'])
    exp.write_json(dest/'audit.json',dict(passed=True,dataset=d,scope=scope,rows=len(a['row']),condition_trials=count,checked_updates=steps,
        reset=True,target_readonly=True,preview_writes_only=True,no_preview_identity=True,
        original_reference_replay_exact=scope=='pilot',same_vanilla_preview_inputs_as_previous=True,
        reverse_order_replay=scope=='pilot',dense_fp64_error=dense_error if scope=='pilot' else None,
        proxy_state_invariant=True,acceptance=accepted,original_FC_dimensions_only=True,
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
                    key=f'{m}__{h}__{s}';x=test[key];assert x.shape[1]==4096 and train[key].shape[1]==4096
                    ss=dict(fixed=original.decision_function(x))
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
        assert args.dataset;models=fit_proxies(args.dataset,old)
        generate(args.dataset,'pilot',old,models);generate(args.dataset,'training',old,models);fit(args.dataset,old)
        protocol();print('DONE',args.dataset,flush=True)
