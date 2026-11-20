"""Native-channel versus spatial-SAE transfer of the final FC gain/subtraction rule."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'4')
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis import conv_preview_temporary_20261106 as previous
from analysis import conv_development_20261104 as old
from predify_experiments.sequence import run_recurrent_sequence,reset_hebbian_state
from predify_experiments.conv_fc_transfer_20261106 import (
    CONFIG,SpatialCodec,TransferOperator,query,new_memory,shuffle_memory)
from predify_experiments.fc_sae_gain_20261105 import Projection,GainProxy,model_hash

OUT=Path('E:/Predify_Experiments_20261105/conv_fc_transfer_20261106')
DOC=ROOT/'docs/conv_fc_transfer_20261106'
writejson=previous.writejson;sha=previous.sha;immutable=previous.immutable
BRANCHES=('direct','sae');SOURCES=('rep4','prd5');METHODS=('full','pool','shuffle','scale')
CASES=['clip']+[f'{b}_{s}_{m}' for b in BRANCHES for s in SOURCES for m in METHODS]


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def status(**kw):writejson(DOC/'status.json',dict(updated=time.time(),complete=False,**kw))
def atomic(p,**arrays):old.c.native.atomic_npz(p,**arrays)


def training_subset(train):
    selected=[]
    for fold in (1,2,3):
        for cat in sorted({t['category_index'] for t in train}):
            candidates=[t for t in train if t['fold']==fold and t['category_index']==cat]
            concept=min(t['concept'] for t in candidates)
            selected+=sorted([t for t in candidates if t['concept']==concept],key=lambda t:t['dataset_index'])[:2]
    return selected


def prepare():
    assert OUT.parent.is_dir(),'E: missing; no fallback'
    files=[Path(__file__),ROOT/'predify_experiments/conv_fc_transfer_20261106.py',
        ROOT/'predify_experiments/conv_preview_memory_20261106.py',
        ROOT/'predify_experiments/fc_sae_gain_20261105.py',
        ROOT/'analysis/conv_preview_temporary_20261106.py',
        ROOT/'analysis/report_conv_fc_transfer_20261106.py']
    protocol=dict(version=1,config=CONFIG,cases=CASES,clean_only=True,fc_hebb=False,legacy_hebb=False,
        constraint='trial reset; preview 2+2; two endpoint writes; target 15 steps read only, no decay; PCoder carry within trial',
        memory='same spatial coactivation and decay/eta .5/.5 as corrected convolution pilot; all within-column nonself candidates; neighbor mutual top32',
        source='rep4 or prd5 endpoint; both use one shared codec per dataset and the same original rep4 scale',
        direct='512 native channels; SAE branch uses 64 codes per spatial column; no spatial flattening before memory',
        sae='fixed512->32 signed Hadamard projection; tied64-latent SAE; 2112 learned parameters; training-fold only; frozen; gain signal only',
        fit='deterministic 24/30 training targets, folds1-3; first concept per category per fold, two images; equal target/preview role weight; 32 uniform spatial samples per frame; 1000 steps',
        calibration='q RMS on positive native query units; positive kappa mean across training positions; pooled exact/different t0,t9; separate branch/source/dataset, frozen before development',
        read='q=codec.adjoint[(z>0)*Wz]; kappa=channel_sum(z*Wz)/channel_sum(z²); m=epsilon*(tanh(q/c)-sum_channel(x*tanh(q/c))/sum_channel(x))',
        inhibition='local mean+population_std from same-column and valid8-neighbor column mixture; tau=lambda*(mu+sd)*kappa/(kappa+kappa0); next=ReLU[x*(1+m)-tau], five steps',
        controls='pool: epsilon0, same matching rule; shuffle: channel/code permutation only after preview2; scale: replay full 19-step global rep4 mean profile into an independent recurrent sequence',
        none='two gray frames, 2+2 steps; no writes; one exact shared baseline after identity acceptance',
        no_search=True,parameters='FC final epsilon .03 lambda .03 steps5 transferred unchanged; training fits proxy and calibration only',
        readout='frozen original FC1 and FC2 StandardScaler+LinearSVC C1, clean t9 reference folds1-3; no latent/reconstruction SVM',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    immutable(OUT/'protocol.json',protocol);immutable(DOC/'protocol.json',protocol)
    for ds in ('imagenet','things'):
        m,train,dev=old.split(ds);oldspec=read(previous.OUT/ds/'manifest.json')
        assert train==oldspec['training'] and dev==oldspec['development']
        selected=training_subset(train)
        spec=dict(dataset=ds,categories=m['categories'],training=train,calibration_training=selected,
            development=dev,scale=oldspec['scale'],previous_manifest_sha256=sha(previous.OUT/ds/'manifest.json'),
            protocol_sha256=sha(OUT/'protocol.json'),expected_development_sequences=len(dev)*(2*len(CASES)+1))
        immutable(OUT/ds/'manifest.json',spec);immutable(DOC/f'{ds}_manifest.json',spec)
        print('PLAN',ds,len(selected),'fit targets;',len(dev),'development;',spec['expected_development_sequences'],'sequences',flush=True)


def initialize():
    p4,p5=old.initialize();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    transform=old.c.VGG16_Weights.IMAGENET1K_V1.transforms();filt=old.c.native.FilteredInput(torch)
    return p4,p5,transform,filt


def baseline_training_trial(target,previews,scale):
    c=old.c;c.reset();reset_hebbian_state(c.layers)
    result={k:[] for k in ('rep4','prd5','query')}
    with torch.no_grad():
        for pv in previews:
            run_recurrent_sequence(c.model,pv[None],2,hebbian_layers=c.layers,commit_update=False,preserve_model_state=True)
            for k,a in [('rep4',c.model.pcoder4.rep),('prd5',c.model.pcoder5.prd)]:
                result[k].append(a.detach().clamp_min(0).cpu().numpy()[0].copy()/scale)
        def after(t,_):
            if t in (0,9):result['query'].append(c.model.pcoder4.rep.detach().cpu().numpy()[0].copy()/scale)
        run_recurrent_sequence(c.model,target[None],15,hebbian_layers=c.layers,commit_update=False,preserve_model_state=True,after_step=after)
    return {k:np.stack(v) for k,v in result.items()}


def collect_training(ds,p4,transform,filt):
    spec=read(OUT/ds/'manifest.json');dest=OUT/ds/'training';dest.mkdir(exist_ok=True)
    memory=new_memory(512,'cuda');p4.hebb=TransferOperator(memory,SpatialCodec(),dict(q_rms=1.,kappa0=1.),source='none',method='clip')
    for j,tr in enumerate(spec['calibration_training']):
        file=dest/f"{tr['dataset_index']}.npz";note=file.with_suffix('.json')
        if file.exists() and note.exists():assert read(note)['sha256']==sha(file);continue
        status(stage='training_inputs',dataset=ds,trial=j+1,total=len(spec['calibration_training']))
        targets,pvs=old.c.test_inputs(ds,tr,transform,filt,torch);values=[];start=time.time()
        for ident in ('exact','different_category'):values.append(baseline_training_trial(targets[0],pvs[ident],spec['scale']))
        atomic(file,**{k:np.stack([v[k] for v in values]) for k in values[0]})
        writejson(note,dict(sha256=sha(file),target=tr['dataset_index'],concept=tr['concept'],seconds=time.time()-start))
        print('TRAIN INPUT',ds,j+1,len(spec['calibration_training']),flush=True)


def fit_proxy(ds):
    spec=read(OUT/ds/'manifest.json');dest=OUT/ds/'proxy';dest.mkdir(exist_ok=True)
    file=dest/'spatial_sae.pt';note=dest/'spatial_sae.json';seed=20261106+(100 if ds=='things' else 0)
    projection=Projection(512,32,seed).cuda()
    if file.exists():
        info=read(note);assert info['sha256']==sha(file)
        proxy=GainProxy(projection,torch.tensor(info['center']),info['scale'],64,seed+1).cuda()
        proxy.load_state_dict(torch.load(file,map_location='cuda',weights_only=True))
        return proxy.eval().requires_grad_(False)
    rng=np.random.default_rng(seed);roles={'preview':[],'target':[]}
    for tr in spec['calibration_training']:
        with np.load(OUT/ds/'training'/f"{tr['dataset_index']}.npz") as z:
            for role,frames in [('preview',z['rep4'].reshape(-1,512,*z['rep4'].shape[-2:])),('target',z['query'][:,1])]:
                for a in frames:
                    flat=a.reshape(512,-1).T;idx=rng.choice(len(flat),32,replace=False)
                    with torch.no_grad():v=projection(torch.from_numpy(flat[idx].copy()).cuda()).cpu().numpy()
                    roles[role].append(v)
    parts=[np.concatenate(roles[r]) for r in ('preview','target')]
    v=torch.from_numpy(np.concatenate(parts)).cuda()
    weights=torch.cat([torch.full((len(a),),.5/len(a),device='cuda') for a in parts])
    scale=float((weights*v.square().mean(1)).sum().sqrt().clamp_min(1e-8))
    center=(weights[:,None]*v/scale).sum(0)
    proxy=GainProxy(projection,center,scale,64,seed+1).cuda()
    opt=torch.optim.Adam(proxy.parameters(),lr=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1000,eta_min=.0001)
    history=[]
    for step in range(1000):
        opt.zero_grad(set_to_none=True);z=proxy.encode_projected(v);hat=proxy.decode_projected(z)
        mse=.5*((hat-v)/scale).square().sum(1);loss=(weights*(mse+.1*z.sum(1))).sum()
        assert torch.isfinite(loss);loss.backward();proxy.project_gradient();opt.step();proxy.constrain();sched.step()
        if (step+1)%200==0:
            rec=dict(step=step+1,loss=float(loss.detach()),active=float((weights*(z>0).float().sum(1)).sum()))
            history.append(rec);print('SAE',ds,rec,flush=True)
    proxy.eval().requires_grad_(False)
    torch.save({k:v.detach().cpu() for k,v in proxy.state_dict().items()},file)
    with torch.no_grad():
        z=proxy.encode_projected(v);hat=proxy.decode_projected(z)
        fvu=float((weights*((hat-v)/scale).square().sum(1)).sum()/(weights*((v/scale-center).square().sum(1))).sum().clamp_min(1e-12))
    writejson(note,dict(sha256=sha(file),center=center.cpu().tolist(),scale=scale,history=history,
        trainable=sum(p.numel() for p in proxy.parameters()),projected_fvu=fvu,proxy_hash=model_hash(proxy),
        role_counts={r:len(a) for r,a in zip(('preview','target'),parts)},training_targets=[t['dataset_index'] for t in spec['calibration_training']]))
    assert sum(p.numel() for p in proxy.parameters())==2112
    return proxy


def calibrate(ds,proxy):
    path=OUT/ds/'calibration.json'
    if path.exists():return read(path)
    spec=read(OUT/ds/'manifest.json');result={}
    for branch in BRANCHES:
        codec=SpatialCodec(proxy if branch=='sae' else None)
        for source in SOURCES:
            sq=count=ks=kn=0.;memory=new_memory(64 if branch=='sae' else 512,'cuda')
            with torch.no_grad():
                for j,tr in enumerate(spec['calibration_training']):
                    with np.load(OUT/ds/'training'/f"{tr['dataset_index']}.npz") as z:
                        for ident in range(2):
                            memory.reset()
                            for frame in z[source][ident]:memory.write(codec.encode(torch.from_numpy(frame[None]).cuda()))
                            memory.freeze()
                            for frame in z['query'][ident]:
                                x=torch.from_numpy(frame[None]).cuda();q,k,_=query(x,memory,codec)
                                a=q[x>0];sq+=float(a.double().square().sum());count+=a.numel()
                                a=k[k>0];ks+=float(a.double().sum());kn+=a.numel()
                result[f'{branch}_{source}']=dict(q_rms=max((sq/max(count,1))**.5,1e-8),
                    kappa0=max(ks/max(kn,1),1e-8),query_units=int(count),positive_kappa_count=int(kn))
                print('CALIBRATION',ds,branch,source,result[f'{branch}_{source}'],flush=True)
    writejson(path,result);writejson(DOC/f'{ds}_calibration.json',result);return result


def fit_heads(ds):
    from predify_experiments.supplementary import load,save
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    path=OUT/ds/'fixed_heads.pckl'
    if path.exists():return load(path)
    spec=read(OUT/ds/'manifest.json');cache=OUT/ds/'fixed_head_training.npz'
    if not cache.exists():
        c=old.c;x={'fc1':[],'fc2':[]};y=[]
        op=TransferOperator(new_memory(512,'cuda'),SpatialCodec(),dict(q_rms=1.,kappa0=1.),
            source='none',method='clip');c.model.pcoder4.pmodule.hebb=op
        with torch.no_grad():
            for j,tr in enumerate(spec['training']):
                with np.load(c.native.OUT/ds/'formal'/f"stimuli_{tr['dataset_index']}.npz") as z:
                    target=torch.from_numpy(z['target_0']).cuda()
                c.reset();reset_hebbian_state(c.layers);op.reset()
                run_recurrent_sequence(c.model,target[None],10,hebbian_layers=c.layers,
                    commit_update=False,preserve_model_state=True)
                for l in x:x[l].append(c.cache[l].detach().cpu().numpy().reshape(-1).copy())
                y.append(tr['category_index'])
                if (j+1)%32==0:print('HEAD INPUT',ds,j+1,len(spec['training']),flush=True)
        atomic(cache,**{l:np.stack(v) for l,v in x.items()},labels=np.array(y))
    with np.load(cache) as z:x={l:z[l] for l in ('fc1','fc2')};y=z['labels']
    heads={}
    for l in x:
        h=make_pipeline(StandardScaler(),LinearSVC(C=1,dual='auto',max_iter=200000,random_state=20261104))
        h.fit(np.asarray(x[l]),y);assert h[-1].n_iter_<200000;heads[l]=h
    save(path,heads);writejson(OUT/ds/'fixed_heads.json',dict(sha256=sha(path),
        input_sha256=sha(cache),source='fresh aligned clip no-preview t9 FC1/FC2, legacy Hebbian off',
        training=[t['dataset_index'] for t in spec['training']]))
    return heads


def run_trial(target,previews,op,permutation=None,profile=None,none=False):
    c=old.c;m=op.memory;m.reset();op.reset(profile);c.reset();reset_hebbian_state(c.layers)
    audit=dict(reset_zero=not bool(m.intra.count_nonzero() or m.trace_inter.count_nonzero()),preview=[],method=op.method)
    features={l:[] for l in previous.LAYERS};features['conv4_input']=[];trace=[]
    with torch.no_grad():
        for pv in previews:
            before=m.fingerprint()
            run_recurrent_sequence(c.model,pv[None],2,hebbian_layers=c.layers,commit_update=False,preserve_model_state=True)
            assert before==m.fingerprint()
            if not none and op.method not in ('clip','scale'):
                source=c.model.pcoder4.rep if op.source=='rep4' else c.model.pcoder5.prd
                info=op.write(source)
            else:info=m.stats()
            audit['preview'].append(info)
        m.freeze();audit['written_weight']=m.fingerprint()
        if op.method=='shuffle' and not none:shuffle_memory(m,permutation)
        before=m.fingerprint();versions=tuple(x._version for x in (m.intra,m.trace_inter,m.inter))
        op.trace_enabled=True
        def after(t,_):
            assert versions==tuple(x._version for x in (m.intra,m.trace_inter,m.inter))
            for l,v in [('conv4',c.model.pcoder4.rep),('conv5',c.model.pcoder5.rep),
                        ('fc1',c.cache['fc1']),('fc2',c.cache['fc2']),('conv4_input',op.input_activity)]:
                a=v.detach().cpu().numpy().reshape(-1).copy();assert np.isfinite(a).all();features[l].append(a)
            trace.append(op.last_trace)
        run_recurrent_sequence(c.model,target[None],15,hebbian_layers=c.layers,commit_update=False,preserve_model_state=True,after_step=after)
    audit.update(before_target=before,after_target=m.fingerprint(),writes=m.writes,
        means=op.output_means,trace=trace,graph=m.stats())
    assert audit['reset_zero'] and before==audit['after_target'] and op.calls==19
    assert m.writes==(0 if none or op.method in ('clip','scale') else 2)
    assert all(not l.compute_enabled and not l.update_enabled for l in c.layers)
    return {l:np.stack(v) for l,v in features.items()},audit


def compact(values):
    result=previous.compact_group(values)
    # Original intervention input, with exact full-map geometry and no raw-map payload.
    layer='conv4_input';vv=[v[layer] for v in values]
    for name,func in [('mean',lambda a:a.mean(1,dtype=np.float64)),
        ('mean_abs',lambda a:np.abs(a).mean(1,dtype=np.float64)),
        ('density',lambda a:(a!=0).mean(1)),('negative',lambda a:(a<0).mean(1))]:
        result[layer+'_'+name]=np.array([func(v) for v in vv])
    gg=[];nn=[]
    for t in range(15):
        x=np.array([v[t] for v in vv],dtype=np.float64);n=np.linalg.norm(x,axis=1)
        z=x/np.maximum(n[:,None],1e-12);gg.append(z@z.T);nn.append(n)
    result[layer+'_gram']=np.stack(gg);result[layer+'_norm']=np.stack(nn,axis=1)
    return result


def make_op(branch,source,method,proxy,cal,spec):
    return TransferOperator(new_memory(64 if branch=='sae' else 512,'cuda'),
        SpatialCodec(proxy if branch=='sae' else None),cal.get(f'{branch}_{source}',dict(q_rms=1.,kappa0=1.)),
        spec['scale'],source,method)


def smoke(ds,p4,p5,transform,filt,proxy,cal):
    spec=read(OUT/ds/'manifest.json');tr=spec['development'][0]
    targets,pvs=old.c.test_inputs(ds,tr,transform,filt,torch);checks={};gray=[old.c.grey]*2
    for branch in BRANCHES:
        for source in SOURCES:
            op=make_op(branch,source,'full',proxy,cal,spec);p4.hebb=op
            first,a=run_trial(targets[0],pvs['exact'],op)
            _,b=run_trial(targets[0],pvs['different_category'],op)
            again,c=run_trial(targets[0],pvs['exact'],op)
            assert all(np.array_equal(first[l],again[l]) for l in first)
            assert a['before_target']==c['before_target'] and a['before_target']!=b['before_target']
            none,n=run_trial(targets[0],gray,op,none=True)
            base=make_op('direct','none','clip',proxy,cal,spec);p4.hebb=base
            reference,_=run_trial(targets[0],gray,base,none=True)
            assert all(np.array_equal(none[l],reference[l]) for l in none)
            scaled=make_op(branch,source,'scale',proxy,cal,spec);p4.hebb=scaled
            _,s=run_trial(targets[0],pvs['exact'],scaled,profile=a['means'])
            error=max(abs(x-y) for x,y in zip(a['means'],s['means']));assert error<2e-6
            assert old.c.model.pcoder5.pmodule is p5
            checks[f'{branch}_{source}']=dict(exact=a,different=b,repeat_bitwise=True,no_history_identity=True,scale_error=error)
            print('SMOKE',ds,branch,source,flush=True)
    writejson(DOC/f'{ds}_smoke.json',checks)


def extract(ds,p4,p5,transform,filt,proxy,cal):
    spec=read(OUT/ds/'manifest.json');proxy_before=model_hash(proxy)
    immutable(OUT/ds/'freeze.json',dict(protocol=sha(OUT/'protocol.json'),proxy=model_hash(proxy),
        calibration=sha(OUT/ds/'calibration.json'),heads=sha(OUT/ds/'fixed_heads.pckl'),config=CONFIG))
    for case in CASES:
        branch,source,method=('direct','none','clip') if case=='clip' else case.split('_')
        histories=('none','exact','different_category') if case=='clip' else ('exact','different_category')
        folder=OUT/ds/'development'/case;folder.mkdir(parents=True,exist_ok=True)
        for history in histories:
            file=folder/f'{history}.npz';note=file.with_suffix('.json')
            if file.exists() and note.exists():assert read(note)['sha256']==sha(file);continue
            start=time.time();values=[];audits=[]
            op=make_op(branch,source,method,proxy,cal,spec);p4.hebb=op
            fullmeta=read(OUT/ds/'development'/f'{branch}_{source}_full'/f'{history}.json') if method=='scale' else None
            for j,tr in enumerate(spec['development']):
                status(stage='development',dataset=ds,case=case,history=history,trial=j+1,total=len(spec['development']))
                targets,pvs=old.c.test_inputs(ds,tr,transform,filt,torch)
                previews=[old.c.grey]*2 if history=='none' else pvs[history]
                dim=64 if branch=='sae' else 512
                perm=torch.randperm(dim,generator=torch.Generator().manual_seed(20261106+tr['dataset_index'])).cuda()
                profile=fullmeta['trials'][j]['means'] if fullmeta else None
                v,a=run_trial(targets[0],previews,op,perm,profile,history=='none')
                a['target']=tr['dataset_index'];values.append(v);audits.append(a)
                if method=='shuffle':
                    f=read(OUT/ds/'development'/f'{branch}_{source}_full'/f'{history}.json')['trials'][j]
                    assert a['written_weight']==f['written_weight'],'Shuffle changed preview learning'
                if method=='scale':assert max(abs(x-y) for x,y in zip(a['means'],profile))<2e-6
                assert old.c.model.pcoder5.pmodule is p5
            arrays=compact(values);arrays.update(indices=np.array([t['dataset_index'] for t in spec['development']]),
                labels=np.array([t['category_index'] for t in spec['development']]))
            atomic(file,**arrays);assert model_hash(proxy)==proxy_before
            writejson(note,dict(sha256=sha(file),trials=audits,seconds=time.time()-start,proxy_frozen=True))
            print('GROUP',ds,case,history,round(time.time()-start,2),flush=True)
    writejson(OUT/ds/'complete.json',dict(complete=True,groups=35,sequences=spec['expected_development_sequences'],proxy_hash=proxy_before))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','fit','smoke','run'))
    parser.add_argument('--dataset',choices=('imagenet','things','both'),default='both');args=parser.parse_args()
    prepare()
    if args.action=='prepare':return
    p4,p5,transform,filt=initialize()
    for ds in (('imagenet','things') if args.dataset=='both' else (args.dataset,)):
        collect_training(ds,p4,transform,filt);proxy=fit_proxy(ds);cal=calibrate(ds,proxy);fit_heads(ds)
        if args.action=='fit':continue
        if not (DOC/f'{ds}_smoke.json').exists():smoke(ds,p4,p5,transform,filt,proxy,cal)
        if args.action=='run':extract(ds,p4,p5,transform,filt,proxy,cal)
    status(stage='extraction_complete' if args.action=='run' else args.action+'_complete')


if __name__=='__main__':main()
