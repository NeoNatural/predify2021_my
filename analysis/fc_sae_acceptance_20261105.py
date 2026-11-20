"""One-config, training-fold-only SAE feasibility acceptance; no Hebbian or backbone editing."""
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
from predify_experiments.fc_sae_proxy_20261105 import SparseProxy,CONFIG,fingerprint,training_statistics,objective

OUT=ROOT/'Hebbian-CNN/Log/fc_sae_acceptance_20261105'
DOC=ROOT/'docs/fc_sae_acceptance_20261105'
LAYERS=('fc1','fc2')

def protocol():
    old=exp.read(exp.OUT/'protocol.json');exp.validate(old);previous.protocol()
    deps=['predify_experiments/fc_sae_proxy_20261105.py','analysis/fc_preview_pilot_20261105.py','analysis/fc_preview_refit_20261105.py']
    p=dict(source_sha256=exp.sha(__file__),dependencies={k:exp.sha(ROOT/k) for k in deps},configuration=CONFIG,
        training='formal folds1-4 vanilla targets and unique preview frames; 1/2 total weight per role; no labels in SAE loss',
        heldout='formal fold0 targets primary; existing development targets secondary; development preview reconstruction only',
        optimization='1000 full-batch Adam updates; cosine lr .001 to .0001; final step only; no validation-based checkpoint selection',
        model='4096 -> 512 ReLU -> signed affine 4096 decoder; unit L2 decoder columns; fixed train-only mean and scalar RMS',
        loss='weighted [0.5 sum(((reconstruction-x)/RMS)^2) + .1 sum(nonnegative code)]',
        readouts='fixed original vanilla C1 SVM; train-target-only adapted raw/rectified reconstruction, latent and signed residual C1 SVMs',
        no_Hebbian=True,no_SAE_installation=True,no_original_model_changes=True,no_search=True,no_per_image_topk=True,
        datasets={})
    for d in ('imagenet','things'):
        previous.verify(previous.OUT/d/'inputs');exp.verify(exp.OUT/d/'inputs')
        p['datasets'][d]=dict(pilot_protocol_sha256=exp.sha(exp.OUT/'protocol.json'),
            train_input_marker_sha256=exp.sha(previous.OUT/d/'inputs/complete.json'),
            development_input_marker_sha256=exp.sha(exp.OUT/d/'inputs/complete.json'),
            raw_cache_sha256=old['datasets'][d]['training_cache_sha256'],fixed_svm_sha256=exp.sha(exp.OUT/d/'reference_svm.pckl'))
    OUT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert exp.read(OUT/'protocol.json')==p,'Frozen SAE configuration or sources changed'
    else:
        exp.write_json(OUT/'protocol.json',p);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (OUT/'model_snapshot.py').write_bytes((ROOT/deps[0]).read_bytes())
    return old,p

def marker(dest,names):exp.write_json(dest/'complete.json',dict(protocol_sha256=exp.sha(OUT/'protocol.json'),files={n:exp.sha(dest/n) for n in names}))
def verify(dest):
    p=exp.read(dest/'complete.json');assert p['protocol_sha256']==exp.sha(OUT/'protocol.json')
    for n,h in p['files'].items():assert exp.sha(dest/n)==h

def unique_previews(a,records):
    hashes=[];values={l:[] for l in LAYERS};seen={}
    assert len(records)==len(a['row'])
    for j,r in enumerate(records):
        assert r['target']==int(a['row'][j])
        for k,h in enumerate(('exact','different')):
            for frame,sha in enumerate(r['preview_sha256'][h]):
                if sha in seen:
                    for l in LAYERS:assert np.array_equal(values[l][seen[sha]],a['preview_'+l][j,k,frame])
                else:
                    seen[sha]=len(hashes);hashes.append(sha)
                    for l in LAYERS:values[l].append(a['preview_'+l][j,k,frame].copy())
    return {l:np.stack(v) for l,v in values.items()},hashes

def data(d,old):
    spec=old['datasets'][d];m=exp.read(exp.native.OUT/d/'manifest.json')
    with np.load(previous.OUT/d/'inputs/features.npz') as z:tr={k:z[k] for k in z.files}
    roles=exp.read(previous.OUT/d/'inputs/audit.json');preview,ph=unique_previews(tr,roles['inputs'])
    assert len(ph)==roles['unique_frames'] and tr['row'].tolist()==spec['training_rows']
    dev=exp.features(d);devnotes=exp.read(exp.OUT/d/'inputs/provenance.json');dp,dh=unique_previews(dev,devnotes['inputs'])
    with np.load(spec['training_cache']) as z:
        assert z['images'].tolist()==[t['dataset_index'] for t in m['trials']]
        ix=[j for j,t in enumerate(m['trials']) if t['fold']==0]
        held={l:z[l][ix].copy() for l in LAYERS}
    ht=[m['trials'][j] for j in ix]
    metadata=dict(train=dict(y=tr['y'],concept=tr['concept'],row=tr['row']),
        formal0=dict(y=np.array([t['category_index'] for t in ht]),concept=np.array([t['concept'] for t in ht]),row=np.array([t['dataset_index'] for t in ht])),
        development=dict(y=dev['y'],concept=dev['concept'],row=dev['row']))
    trainconcepts=set(metadata['train']['concept']);forbidden=set(metadata['formal0']['concept'])|set(metadata['development']['concept'])
    assert trainconcepts.isdisjoint(forbidden)
    for r in roles['roles']:assert r['concept'] not in forbidden
    train_visuals={r['sha256'] for r in roles['roles']}
    held_visuals={m['images'][str(i)]['sha256'] for i in metadata['formal0']['row']}
    dev_visuals={r['sha256'] for r in devnotes['visual_roles']}
    assert train_visuals.isdisjoint(held_visuals|dev_visuals) and set(ph).isdisjoint(dh)
    result={}
    for l in LAYERS:
        target=tr['target_'+l];mix=np.concatenate([target,preview[l]])
        w=np.concatenate([np.full(len(target),.5/len(target)),np.full(len(preview[l]),.5/len(preview[l]))])
        result[l]=dict(fit=mix,fit_weights=w,train=target,train_preview=preview[l],formal0=held[l],development=dev['target_'+l],development_preview=dp[l])
    provenance=dict(dataset=d,train_target_rows=tr['row'].tolist(),formal0_rows=metadata['formal0']['row'].tolist(),
        development_rows=dev['row'].tolist(),unique_train_preview_hashes=ph,unique_development_preview_hashes=dh,
        train_role_counts=dict(target=len(tr['row']),preview=len(ph)),role_total_weights=dict(target=.5,preview=.5),
        no_heldout_concepts_or_visuals_in_fit=True,no_labels_in_SAE_objective=True)
    return result,metadata,provenance

def fit_proxy(d,layer,a):
    dest=OUT/d/layer;dest.mkdir(parents=True,exist_ok=True);path=dest/'model.pth'
    center,scale=training_statistics(a['fit'],a['fit_weights']);seed=CONFIG['seed']+(0 if d=='imagenet' else 100)+LAYERS.index(layer)
    model=SparseProxy(4096,CONFIG['width'],center,scale,seed).cuda()
    if path.exists():
        note=exp.read(dest/'training.json');assert note['protocol_sha256']==exp.sha(OUT/'protocol.json') and note['checkpoint_sha256']==exp.sha(path)
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True),strict=True)
    else:
        x=torch.from_numpy(a['fit']).cuda();weights=torch.tensor(a['fit_weights'],dtype=torch.float32,device='cuda')
        opt=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate']);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=CONFIG['steps'],eta_min=CONFIG['minimum_learning_rate'])
        history=[];start=time.monotonic();model.train()
        for step in range(CONFIG['steps']):
            opt.zero_grad(set_to_none=True);loss,components=objective(model,x,weights,CONFIG['l1']);assert torch.isfinite(loss)
            loss.backward();model.project_dictionary_gradient();opt.step();model.constrain_dictionary();schedule.step()
            if step==0 or (step+1)%100==0:
                row=dict(step=step+1,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in components.items()})
                history.append(row);print('SAE FIT',d,layer,row,'seconds',round(time.monotonic()-start,1),flush=True)
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
        exp.write_json(dest/'training.json',dict(protocol_sha256=exp.sha(OUT/'protocol.json'),checkpoint_sha256=exp.sha(path),
            seed=seed,configuration=CONFIG,fit_count=len(a['fit']),scale=float(scale),center_sha256=exp.array_sha(center),
            fit_array_sha256=exp.array_sha(a['fit']),fit_weights_sha256=exp.array_sha(a['fit_weights']),
            history=history,final_step_used=True,no_heldout_evaluation_during_optimization=True,seconds=time.monotonic()-start))
        del x,weights,opt,schedule
    model.eval().requires_grad_(False)
    assert np.array_equal(model.center.cpu().numpy(),center) and np.isclose(float(model.scale),scale,rtol=1e-6)
    assert float((model.decoder.norm(dim=0)-1).abs().max())<1e-5
    return model

def reconstruct_metrics(x,hat,z,reference,weights):
    a=x.astype(np.float64);b=hat.astype(np.float64);w=weights/weights.sum();delta=b-a
    sq=np.square(delta).sum(1);var=np.square(a-reference).sum(1);norm=np.linalg.norm(a,axis=1)
    return dict(mse=float(np.average(sq/a.shape[1],weights=w)),fvu=float(np.average(sq,weights=w)/max(np.average(var,weights=w),1e-12)),
        relative_l2=float(np.sqrt(np.average(sq,weights=w)/max(np.average(np.square(a).sum(1),weights=w),1e-12))),
        cosine=float(np.average((a*b).sum(1)/np.maximum(norm*np.linalg.norm(b,axis=1),1e-12),weights=w)),
        mean_activity=float(np.average(b.mean(1),weights=w)),input_mean=float(np.average(a.mean(1),weights=w)),
        negative_fraction=float(np.mean(b<0)),latent_active_mean=float(np.mean((z>0).sum(1))),
        latent_active_p10=float(np.quantile((z>0).sum(1),.1)),latent_active_p90=float(np.quantile((z>0).sum(1),.9)),
        latent_unused_fraction=float(np.mean(~(z>0).any(0))))

def evaluate(d,layer,model,a,metadata):
    dest=OUT/d/layer;before=fingerprint(model);arrays={};metrics=[];start=time.monotonic()
    target_center=a['train'].astype(np.float64).mean(0);preview_center=a['train_preview'].astype(np.float64).mean(0)
    with torch.inference_mode():
        for split in ('train','train_preview','formal0','development','development_preview'):
            x=torch.from_numpy(a[split]).cuda();hat,z=model(x);zero,delta=model.zero_edit(x)
            assert torch.equal(zero,x) and torch.count_nonzero(delta)==0
            h=hat.cpu().numpy();zz=z.cpu().numpy();clipped=np.maximum(h,0)
            for k,value in dict(raw=x.cpu().numpy(),reconstruction=h,rectified=clipped,latent=zz,residual=a[split]-h,zero_edit=zero.cpu().numpy()).items():arrays[f'{split}__{k}']=value
            weights=exp.weights(metadata[split]['y'],metadata[split]['concept']) if split in metadata else np.ones(len(x))
            center=preview_center if 'preview' in split else target_center
            for variant,v in (('reconstruction',h),('rectified',clipped)):
                metrics.append(dict(dataset=d,layer=layer,split=split,variant=variant,n=len(x),
                    **reconstruct_metrics(a[split],v,zz,center,weights)))
    assert before==fingerprint(model)
    with (exp.OUT/d/'reference_svm.pckl').open('rb') as f:fixed=pickle.load(f)[layer]
    heads={};readouts=[];scores={};headinfo={};scalew=exp.weights(metadata['train']['y'],metadata['train']['concept']);labels=metadata['train']['y']
    for variant in ('reconstruction','rectified','latent','residual'):
        head=make_pipeline(StandardScaler(),LinearSVC(C=1,dual='auto',max_iter=30000,random_state=exp.SEED))
        with threadpool_limits(limits=2),warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            head.fit(arrays['train__'+variant],labels,standardscaler__sample_weight=scalew,linearsvc__sample_weight=scalew)
        assert head[-1].n_iter_<30000;heads[variant]=head;headinfo[variant]=int(head[-1].n_iter_)
    for split in ('train','formal0','development'):
        y=metadata[split]['y'];rawscores=fixed.decision_function(arrays[split+'__raw']);rawpred=rawscores.argmax(1)
        for variant in ('raw','zero_edit','reconstruction','rectified','latent','residual'):
            kinds=['fixed'] if variant in ('raw','zero_edit') else (['fixed','adapted'] if variant in ('reconstruction','rectified') else ['adapted'])
            for kind in kinds:
                v=fixed.decision_function(arrays[split+'__'+variant]) if kind=='fixed' else heads[variant].decision_function(arrays[split+'__'+variant])
                if variant=='zero_edit':assert np.array_equal(v,rawscores)
                scores[f'{split}__{variant}__{kind}']=v
                readouts.append(dict(dataset=d,layer=layer,split=split,variant=variant,readout=kind,**previous.metrics(v,y),
                    agreement_with_fixed_raw=float(np.mean(v.argmax(1)==rawpred)),
                    fixed_score_rmse=float(np.sqrt(np.mean((v-rawscores)**2))) if kind=='fixed' else None))
    np.savez_compressed(dest/'representations.npz',**arrays)
    np.savez_compressed(dest/'scores.npz',**scores)
    with (dest/'readouts.pckl').open('wb') as f:pickle.dump(heads,f,protocol=pickle.HIGHEST_PROTOCOL)
    exp.write_json(dest/'reconstruction_metrics.json',metrics);exp.write_json(dest/'readout_metrics.json',readouts)
    exp.write_json(dest/'evaluation_audit.json',dict(passed=True,state_before=before,state_after=fingerprint(model),zero_edit_max_abs_error=0.,
        zero_edit_delta_nonzero=0,zero_edit_fixed_scores_identical=True,readout_training_rows=metadata['train']['row'].tolist(),
        heldout_not_used_for_fitting=True,no_labels_in_SAE_objective=True,readout_n_iter=headinfo,
        dictionary_unit_norm_max_error=float((model.decoder.norm(dim=0)-1).abs().max()),seconds=time.monotonic()-start))
    marker(dest,['model.pth','training.json','representations.npz','scores.npz','readouts.pckl','reconstruction_metrics.json','readout_metrics.json','evaluation_audit.json'])
    print('SAE EVALUATED',d,layer,flush=True)

def bridge_acceptance(d,models,a,metadata):
    w,b=exp.load_bridge();result={}
    before={l:fingerprint(m) for l,m in models.items()}
    with torch.inference_mode():
        for split in ('train','formal0','development'):
            x=torch.from_numpy(a['fc1'][split]).cuda();one,_=models['fc1'].zero_edit(x)
            baseline=torch.relu(torch.nn.functional.linear(x,w,b));incoming=torch.relu(torch.nn.functional.linear(one,w,b))
            two,_=models['fc2'].zero_edit(incoming)
            assert torch.equal(x,one) and torch.equal(baseline,two)
            cached=a['fc2'][split];error=float(np.abs(cached-baseline.cpu().numpy()).max())
            assert np.allclose(cached,baseline.cpu().numpy(),atol=1e-4,rtol=2e-5)
            result[split]=dict(zero_edit_serial_max_error=0.,cached_vanilla_fc2_bridge_max_error=error,rows=len(x))
    assert before=={l:fingerprint(m) for l,m in models.items()}
    exp.write_json(OUT/d/'bridge_acceptance.json',dict(passed=True,relative_to_same_original_bridge=True,results=result,states_unchanged=True))

def worker(d,old):
    a,metadata,provenance=data(d,old);dest=OUT/d;dest.mkdir(parents=True,exist_ok=True)
    exp.write_json(dest/'data_provenance.json',provenance)
    np.savez_compressed(dest/'target_metadata.npz',**{f'{s}__{k}':v for s,rs in metadata.items() for k,v in rs.items()})
    models={}
    for layer in LAYERS:
        model=fit_proxy(d,layer,a[layer]);models[layer]=model
        if (OUT/d/layer/'complete.json').exists():verify(OUT/d/layer)
        else:evaluate(d,layer,model,a[layer],metadata)
    bridge_acceptance(d,models,a,metadata);exp.validate(old);protocol()
    marker(dest,['data_provenance.json','target_metadata.npz','bridge_acceptance.json','fc1/complete.json','fc2/complete.json'])
    print('SAE DONE',d,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','worker']);p.add_argument('--dataset',choices=['imagenet','things']);args=p.parse_args()
    old,_=protocol();exp.setup()
    if args.action=='worker':
        assert args.dataset
        with threadpool_limits(limits=2):worker(args.dataset,old)
