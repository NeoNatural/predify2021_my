"""Default-only clean pilot: trial-local preview weights; no changes to the original model."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(_key,'2')
import argparse
from collections import Counter
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.fc_preview_memory_20261105 import PreviewMemory,TrialEngine,DEFAULTS,METHODS,side,bridge,digest_tensor
from predify_experiments import native_categories_20261029 as native
from predify_experiments.fc_lateral_vanilla_20261028 import sha,write_json

OUT=ROOT/'Hebbian-CNN/Log/fc_preview_temporary_pilot_20261105'
DOC=ROOT/'docs/fc_preview_temporary_20261105'
PLAN=DOC/'独立FC临时Hebbian实验方案.md'
CHECKPOINT=Path(r'C:\Users\liang\.cache\torch\hub\checkpoints\vgg16-397923af.pth')
HISTORIES=('exact','different','none')
STAGES=('fc1','fc2_input','fc2')
SEED=20261105


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def setup():
    torch.set_num_threads(2);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)


def selected(m):
    result=[]
    for cat in range(len(m['categories'])):
        concepts=sorted({t['concept'] for t in m['development'] if t['category_index']==cat})[:2]
        for concept in concepts:
            result.extend(sorted([t for t in m['development'] if t['concept']==concept],key=lambda t:t['dataset_index'])[:2])
    return result


def freeze():
    manifests={d:read(native.OUT/d/'manifest.json') for d in ('imagenet','things')}
    deps=['predify_experiments/fc_preview_memory_20261105.py','predify_experiments/native_categories_20261029.py']
    original=['Hebbian-CNN/Hebbian_VGG_Lib.py','Hebbian-CNN/Hebb_Priming_Utils.py',
              'predify2021/model_factory/get_model.py','predify2021/model_factory/pvgg16_shared.py']
    p=dict(version=1,source_sha256=sha(__file__),dependencies={s:sha(ROOT/s) for s in deps},
        original_sources={s:sha(ROOT/s) for s in original},plan_sha256=sha(PLAN),checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),
        parameters=DEFAULTS,methods=list(METHODS),histories=list(HISTORIES),target='clean only',
        training='formal fold!=0 vanilla forward activations only; generic class/concept weighted C1 SVM and RMS; no offline E',
        memory='two preview-only normalized off-diagonal outer writes; rho .5; target frozen; every trial resets',
        preview='unmodified vanilla FC1/FC2 encoding of current real two-frame four-object previews; no PCoder or conv Hebb',
        permutation='one fixed deterministic trial/layer permutation shared by Exact and Different; none stays zero',
        no_search=True,no_adapted_svm=True,no_noise=True,no_other_horizons=True,
        datasets={})
    for d,m in manifests.items():
        trials=selected(m);train=[t for t in m['trials'] if t['fold']!=0]
        old=ROOT/'Hebbian-CNN/Log/hebb_fc_native_pilot_20261102'/d
        p['datasets'][d]=dict(manifest_sha256=sha(native.OUT/d/'manifest.json'),categories=m['categories'],trials=trials,
            training_rows=[t['dataset_index'] for t in train],training_cache=str(old/'vanilla_training_fc.npz'),
            training_cache_sha256=sha(old/'vanilla_training_fc.npz'),training_info_sha256=sha(old/'graph_training.json'),
            selected_concepts={c:len({t['concept'] for t in trials if t['category_index']==k}) for k,c in enumerate(m['categories'])},
            slot_counts=dict(Counter(str(t['slot']) for t in trials)))
    OUT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==p,'Frozen pilot changed'
    else:
        write_json(path,p);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (OUT/'memory_snapshot.py').write_bytes((ROOT/deps[0]).read_bytes())
        (OUT/'plan_snapshot.md').write_bytes(PLAN.read_bytes())
    return p


def validate(p):
    assert sha(__file__)==p['source_sha256'] and sha(CHECKPOINT)==p['checkpoint_sha256']
    assert sha(PLAN)==p['plan_sha256']
    for k,h in (p['dependencies']|p['original_sources']).items():assert sha(ROOT/k)==h,k
    for d,s in p['datasets'].items():
        assert sha(native.OUT/d/'manifest.json')==s['manifest_sha256']
        assert sha(s['training_cache'])==s['training_cache_sha256']


def marker(dest,names):
    write_json(dest/'complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),files={name:sha(dest/name) for name in names}))


def verify(dest):
    m=read(dest/'complete.json');assert m['protocol_sha256']==sha(OUT/'protocol.json')
    for name,value in m['files'].items():assert sha(dest/name)==value


def training(d,p):
    m=read(native.OUT/d/'manifest.json');s=p['datasets'][d]
    info=read(Path(s['training_cache']).parent/'graph_training.json')
    assert sha(Path(s['training_cache']).parent/'graph_training.json')==s['training_info_sha256']
    assert info['files']['vanilla_training_fc.npz']==s['training_cache_sha256']
    with np.load(s['training_cache']) as z:
        assert z['images'].tolist()==[t['dataset_index'] for t in m['trials']]
        ix=[j for j,t in enumerate(m['trials']) if t['fold']!=0]
        a={k:z[k][ix].copy() for k in ('fc1','fc2')}
    ts=[m['trials'][i] for i in ix]
    assert [t['dataset_index'] for t in ts]==s['training_rows']
    return a,np.array([t['category_index'] for t in ts]),np.array([t['concept'] for t in ts]),info


def load_bridge():
    weights=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    w,b=(weights[f'classifier.3.{k}'].to('cuda') for k in ('weight','bias'));del weights
    return w,b


def extract(d):
    p=freeze();validate(p);setup();dest=OUT/d/'inputs'
    if (dest/'complete.json').exists():verify(dest);print('INPUTS EXIST',d,flush=True);return
    from PIL import Image
    from torchvision.models import vgg16,VGG16_Weights
    m=read(native.OUT/d/'manifest.json');spec=p['datasets'][d];trials=spec['trials']
    train,_,_,oldinfo=training(d,p)
    formal_concepts={t['concept'] for t in m['trials']};formal_hashes={m['images'][str(t['dataset_index'])]['sha256'] for t in m['trials']}
    roles=[]
    for t in trials:
        assert t['concept'] not in formal_concepts
        assert t['primes']['exact']==t['dataset_index']
        assert m['images'][str(t['primes']['different_category'])]['category']!=t['category_index']
        for i in set([t['dataset_index'],*t['previews'],t['primes']['exact'],t['primes']['different_category']]):
            r=m['images'][str(i)]
            assert r['concept'] not in formal_concepts and r['sha256'] not in formal_hashes
            roles.append(dict(target=t['dataset_index'],image=i,concept=r['concept'],sha256=r['sha256']))
    model=vgg16(weights=None)
    weights=torch.load(CHECKPOINT,map_location='cpu',weights_only=True);model.load_state_dict(weights,strict=True);del weights
    model.eval().requires_grad_(False).cuda();cache={}
    def state_sha():
        h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    model_before=state_sha()
    def record(name):
        def hook(module,args,out):cache[name]=out.detach().clone()
        return hook
    hooks=[model.classifier[i].register_forward_hook(record(k)) for i,k in [(1,'fc1'),(4,'fc2')]]
    transform=VGG16_Weights.IMAGENET1K_V1.transforms();f=native.FilteredInput(torch);image_cache={}
    def image(i):
        if i not in image_cache:
            r=m['images'][str(i)];assert sha(r['path'])==r['sha256']
            with Image.open(r['path']) as im:image_cache[i]=transform(im.convert('RGB')).cuda()
        return image_cache[i]
    def encode(canvas):
        model(canvas[None]);return {k:v[0].cpu().numpy() for k,v in cache.items()}
    arrays={k:[] for k in ('target_fc1','target_fc2','preview_fc1','preview_fc2','aligned_different_fc1','aligned_different_fc2')}
    hashes=[];replay=[];start=time.monotonic()
    with torch.inference_mode():
        # Replay three allowed training targets: compare the old vanilla-only cache,
        # never load graph weights and never use PCoder reference representations.
        for j,i in enumerate(spec['training_rows'][:3]):
            canvas=f.canvas([(image(i),'center')]);v=encode(canvas)
            assert array_sha(canvas.cpu().numpy())==oldinfo['training_target_hashes'][str(i)]
            errors={k:float(np.abs(v[k]-train[k][j]).max()) for k in ('fc1','fc2')}
            assert all(np.allclose(v[k],train[k][j],atol=1e-4,rtol=2e-5) for k in ('fc1','fc2')),errors
            replay.append(dict(image=i,errors=errors))
        bridge_error=0.
        for t in trials:
            i=t['dataset_index'];target=f.canvas([(image(i),'center')]);v=encode(target)
            for k in ('fc1','fc2'):arrays['target_'+k].append(v[k])
            predicted=bridge(torch.from_numpy(v['fc1'])[None].cuda(),model.classifier[3].weight,model.classifier[3].bias)[0].cpu().numpy()
            error=float(np.abs(predicted-v['fc2']).max());bridge_error=max(bridge_error,error)
            assert np.allclose(predicted,v['fc2'],atol=1e-6,rtol=1e-6)
            frame_hashes={};values={k:[] for k in ('fc1','fc2')}
            for ident,key in (('exact','exact'),('different','different_category')):
                ids=list(t['previews']);ids[t['slot']]=t['primes'][key]
                fs=[];zs=[]
                for stage in range(2):
                    canvas=f.canvas([(image(j),side) for j,side in zip(ids[stage*2:stage*2+2],('left','right'))],True)
                    fs.append(array_sha(canvas.cpu().numpy()));zs.append(encode(canvas))
                frame_hashes[ident]=fs
                for k in values:values[k].append(np.stack([z[k] for z in zs]))
            changed=t['slot']//2
            assert frame_hashes['exact'][1-changed]==frame_hashes['different'][1-changed]
            assert frame_hashes['exact'][changed]!=frame_hashes['different'][changed]
            for k in values:arrays['preview_'+k].append(np.stack(values[k]))
            aligned=encode(f.canvas([(image(t['primes']['different_category']),'center')]))
            for k in aligned:arrays['aligned_different_'+k].append(aligned[k])
            hashes.append(dict(target=i,target_sha256=array_sha(target.cpu().numpy()),preview_sha256=frame_hashes))
            print('EXTRACT',d,i,len(hashes),'/',len(trials),flush=True)
    model_after=state_sha();assert model_before==model_after
    for hook in hooks:hook.remove()
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'features.npz',**{k:np.stack(v) for k,v in arrays.items()},
        y=np.array([t['category_index'] for t in trials]),concept=np.array([t['concept'] for t in trials]),row=np.array([t['dataset_index'] for t in trials]))
    write_json(dest/'provenance.json',dict(dataset=d,rows=len(trials),visual_roles=roles,inputs=hashes,
        training_replay=replay,bridge_max_abs_error=bridge_error,checkpoint_model_state_before=model_before,checkpoint_model_state_after=model_after,
        categories=spec['categories'],formal_development_concepts_disjoint=True,formal_development_visual_bytes_disjoint=True,seconds=time.monotonic()-start))
    marker(dest,['features.npz','provenance.json']);validate(p)
    print('INPUTS DONE',d,'seconds',round(time.monotonic()-start,1),flush=True)


def weights(y,groups):
    w=np.zeros(len(y),np.float64)
    for c in np.unique(y):
        ids=np.flatnonzero(y==c);gs,ns=np.unique(groups[ids],return_counts=True);counts=dict(zip(gs,ns))
        w[ids]=[1/(len(gs)*counts[g]) for g in groups[ids]]
    return w*len(y)/w.sum()


def nuisance(d,p,fit=False):
    train,y,groups,_=training(d,p);result={};dest=OUT/d
    for layer in ('fc1','fc2'):
        x=train[layer];u=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
        centers=np.stack([u[y==c].mean(0) for c in range(len(p['datasets'][d]['categories']))])
        pref=centers-(centers.sum(0)-centers)/(len(centers)-1)
        rank=np.argsort(pref,axis=1,kind='stable');quartiles=np.empty_like(rank)
        for c in range(len(centers)):quartiles[c,rank[c]]=np.arange(x.shape[1])*4//x.shape[1]
        result[layer]=dict(scale=float(np.sqrt(np.mean(x.astype(np.float64)**2))),quartiles=quartiles,preference=pref)
    if fit:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        from sklearn.exceptions import ConvergenceWarning
        from threadpoolctl import threadpool_limits
        path=dest/'reference_svm.pckl'
        if path.exists():
            note=read(dest/'nuisance.json');assert sha(path)==note['svm_sha256'];assert note['training_rows']==p['datasets'][d]['training_rows']
            with path.open('rb') as f:models=pickle.load(f)
        else:
            models={};w=weights(y,groups)
            with threadpool_limits(limits=2),warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                for layer in ('fc1','fc2'):
                    head=make_pipeline(StandardScaler(),LinearSVC(C=1,dual='auto',max_iter=30000,random_state=SEED))
                    head.fit(train[layer],y,standardscaler__sample_weight=w,linearsvc__sample_weight=w)
                    assert int(head[-1].n_iter_)<head[-1].max_iter;models[layer]=head
            dest.mkdir(exist_ok=True,parents=True)
            with path.open('wb') as f:pickle.dump(models,f,protocol=pickle.HIGHEST_PROTOCOL)
            np.savez_compressed(dest/'nuisance_arrays.npz',**{f'{l}_{k}':v for l,rs in result.items() for k,v in rs.items()})
            write_json(dest/'nuisance.json',dict(training_rows=p['datasets'][d]['training_rows'],held_out_formal_fold=0,
                scales={k:v['scale'] for k,v in result.items()},svm_sha256=sha(path),arrays_sha256=sha(dest/'nuisance_arrays.npz'),
                svm_n_iter={l:int(h[-1].n_iter_) for l,h in models.items()},no_training_side_weights=True))
        return result,models
    return result


def permutations(d,target,dimension=4096):
    return tuple(torch.from_numpy(np.random.default_rng(int.from_bytes(hashlib.sha256(f'{SEED}/{d}/{target}/{l}'.encode()).digest()[:8],'little')).permutation(dimension)).cuda() for l in (1,2))


def features(d):
    verify(OUT/d/'inputs')
    with np.load(OUT/d/'inputs/features.npz') as z:return {k:z[k] for k in z.files}


def acceptance(d):
    p=freeze();validate(p);setup();a=features(d);ns=nuisance(d,p);w,b=load_bridge()
    engine=TrialEngine(w,b,(ns['fc1']['scale'],ns['fc2']['scale']))
    record=dict(dataset=d,passed=True,trial_count=len(a['row']),no_preview_equal_inhibition=True,
        preview_changes_weights=True,target_cannot_write=True,target_cannot_change_learned_weights=True,
        target_weights_constant=True,order_independent=True,batch_independent=True,dense_factor_equivalent=True,
        dense_operator_max_error=0.,dense_cascade_max_error=0.,batch_max_error=0.,centered_memory_diagnostic=[])
    references={}
    with torch.inference_mode():
        for i,target in enumerate(a['row']):
            x=torch.from_numpy(a['target_fc1'][i:i+1]).cuda();ps=permutations(d,int(target))
            cases=[]
            for h,idx in (('exact',0),('different',1),('none',None)):
                p1=[] if idx is None else torch.from_numpy(a['preview_fc1'][i,idx]).cuda()
                p2=[] if idx is None else torch.from_numpy(a['preview_fc2'][i,idx]).cuda()
                r,m=engine.run(x,p1,p2,'hebbian',ps)
                assert m['writes']==([0,0] if idx is None else [2,2])
                assert m['before_target']==m['after_target']
                references[i,h]={k:v.clone() for k,v in r.items()};cases.append(m)
                if idx is not None:
                    assert m['preview_boundaries'][0]!=m['preview_boundaries'][1]
                    assert all(q['frobenius']>0 for q in m['memory'])
                    _,changed=engine.run(x*1.7,p1,p2,'hebbian',ps)
                    assert changed['before_target']==m['before_target']
                    for memory in engine.memories:
                        old=memory.fingerprint()
                        try:memory.write(x);raise AssertionError('Target write was accepted')
                        except RuntimeError:pass
                        assert memory.fingerprint()==old
            assert cases[0]['before_target']!=cases[1]['before_target']
            for method in ('vanilla','inhibition'):
                values=[]
                for idx in (0,1,None):
                    r,m=engine.run(x,[] if idx is None else torch.from_numpy(a['preview_fc1'][i,idx]).cuda(),
                        [] if idx is None else torch.from_numpy(a['preview_fc2'][i,idx]).cuda(),method,ps)
                    assert m['writes']==[0,0];values.append(r)
                for k in STAGES:assert all(torch.equal(values[0][k],z[k]) for z in values)
                if method=='inhibition':
                    for k in STAGES:assert torch.equal(values[0][k],references[i,'none'][k])
        # Reverse complete trial order after interference from unrelated trials.
        for i in reversed(range(len(a['row']))):
            for h,idx in (('none',None),('different',1),('exact',0)):
                r,m=engine.run(torch.from_numpy(a['target_fc1'][i:i+1]).cuda(),
                    [] if idx is None else torch.from_numpy(a['preview_fc1'][i,idx]).cuda(),
                    [] if idx is None else torch.from_numpy(a['preview_fc2'][i,idx]).cuda(),'hebbian',permutations(d,int(a['row'][i])))
                for k in STAGES:assert torch.equal(r[k],references[i,h][k])
        # Real 4096-D factors and explicit dense matrices, independently in FP64.
        u1,u2=(torch.from_numpy(a['preview_'+l][0,0]).cuda().double() for l in ('fc1','fc2'))
        xd=torch.from_numpy(a['target_fc1'][0:1]).cuda().double();ed=TrialEngine(w.double(),b.double(),engine.scales)
        r,meta=ed.run(xd,u1,u2,'hebbian')
        dense=[]
        for memory in ed.memories:
            W=torch.zeros(memory.dimension,memory.dimension,device='cuda',dtype=torch.float64)
            for u,c in zip(memory.factors,memory.coefficients):
                v=u[:,None]*u[None,:];v.fill_diagonal_(0);W+=c*v
            q=torch.from_numpy(a['target_fc1'][0:1]).cuda().double()
            err=float((memory.apply(q)-q@W.T).abs().max());record['dense_operator_max_error']=max(record['dense_operator_max_error'],err)
            assert err<2e-11 and torch.equal(W,W.T) and W.diagonal().count_nonzero()==0
            dense.append(W)
        def independent_side(v,W,s):
            x=v/s;r=x.clone()
            for _ in range(5):
                d=x+.75*(r@W.T);theta=.25*d.mean(1,keepdim=True)+.1*torch.sqrt(((d-d.mean(1,keepdim=True))**2).mean(1,keepdim=True))
                r=torch.where(d>theta,.5*r+.5*d/(1+2*r.mean(1,keepdim=True)),0.)
            return r*s
        one=independent_side(xd,dense[0],engine.scales[0]);two=independent_side(torch.relu(one@w.double().T+b.double()),dense[1],engine.scales[1])
        err=max(float((one-r['fc1']).abs().max()),float((two-r['fc2']).abs().max()))
        record['dense_cascade_max_error']=err;assert err<2e-10
        del dense
        # A batch shares the same preview memory, but pools activity within each image.
        memory=ed.memories[0];xb=torch.from_numpy(a['target_fc1'][:3]).cuda().double()
        rb,_=side(xb,memory,engine.scales[0]);rs=torch.cat([side(x[None],memory,engine.scales[0])[0] for x in xb])
        err=float((rb-rs).abs().max());record['batch_max_error']=err;assert err<2e-10
        for i in range(len(a['row'])):
            probe=[]
            for layer in ('fc1','fc2'):
                query=torch.from_numpy(a['target_'+layer][i]).cuda()
                vals=[]
                for v in (a['target_'+layer][i],a['aligned_different_'+layer][i]):
                    m=PreviewMemory(4096);m.write(torch.from_numpy(v).cuda());m.freeze()
                    vals.append(float((query*m.apply(query)).sum()/query.square().sum().clamp_min(1e-12)))
                probe.append(dict(layer=layer,exact=vals[0],different=vals[1]))
            record['centered_memory_diagnostic'].append(dict(target=int(a['row'][i]),layers=probe))
    path=OUT/d/'state_acceptance.json';write_json(path,record);validate(p)
    print('STATE ACCEPTANCE PASSED',d,{k:v for k,v in record.items() if k!='centered_memory_diagnostic'},flush=True)


def run(d):
    p=freeze();validate(p);setup();dest=OUT/d/'pilot'
    if (dest/'complete.json').exists():verify(dest);print('PILOT EXISTS',d,flush=True);return
    accept=read(OUT/d/'state_acceptance.json');assert accept['passed'] and accept['trial_count']==len(p['datasets'][d]['trials'])
    ns,models=nuisance(d,p,fit=True);a=features(d);w,b=load_bridge()
    engine=TrialEngine(w,b,(ns['fc1']['scale'],ns['fc2']['scale']))
    reps={};meta=[];factors={};start=time.monotonic()
    with torch.inference_mode():
        for i,target in enumerate(a['row']):
            x=torch.from_numpy(a['target_fc1'][i:i+1]).cuda();ps=permutations(d,int(target))
            for history,idx in (('exact',0),('different',1),('none',None)):
                p1=[] if idx is None else torch.from_numpy(a['preview_fc1'][i,idx]).cuda()
                p2=[] if idx is None else torch.from_numpy(a['preview_fc2'][i,idx]).cuda()
                for method in METHODS:
                    r,m=engine.run(x,p1,p2,method,ps)
                    assert m['before_target']==m['after_target']
                    for stage,v in r.items():reps.setdefault(f'{method}__{history}__{stage}',[]).append(v[0].cpu().numpy())
                    if method=='hebbian':
                        for layer,memory in enumerate(engine.memories):
                            factors[f'{int(target)}__{history}__fc{layer+1}']=torch.stack(memory.factors).cpu().numpy() if memory.factors else np.empty((0,4096),np.float32)
                    m.update(target=int(target),method=method,history=history);meta.append(m)
            print('PILOT TRIAL',d,i+1,'/',len(a['row']),flush=True)
    reps={k:np.stack(v) for k,v in reps.items()};dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'representations.npz',**reps,row=a['row'],y=a['y'],concept=a['concept'])
    np.savez_compressed(dest/'preview_factors.npz',**factors)
    scores={}
    for key,values in reps.items():scores[key]=models['fc1' if key.endswith('__fc1') else 'fc2'].decision_function(values)
    np.savez_compressed(dest/'scores.npz',**scores,row=a['row'],y=a['y'])
    write_json(dest/'trial_audit.json',meta)
    write_json(dest/'run.json',dict(dataset=d,parameters=DEFAULTS,rows=len(a['row']),methods=list(METHODS),histories=list(HISTORIES),
        physical_target_trials=len(meta),state_acceptance_sha256=sha(OUT/d/'state_acceptance.json'),
        nuisance_sha256=sha(OUT/d/'nuisance.json'),input_marker_sha256=sha(OUT/d/'inputs/complete.json'),
        seconds=time.monotonic()-start,no_parameter_search=True))
    marker(dest,['representations.npz','preview_factors.npz','scores.npz','trial_audit.json','run.json'])
    validate(p);print('PILOT DONE',d,len(meta),'target trials','seconds',round(time.monotonic()-start,1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','extract','accept','run'))
    parser.add_argument('--dataset',choices=('imagenet','things'));args=parser.parse_args()
    if args.action=='prepare':print('FROZEN',freeze()['parameters'],flush=True)
    else:
        for dataset in ([args.dataset] if args.dataset else ['imagenet','things']):
            {'extract':extract,'accept':acceptance,'run':run}[args.action](dataset)
