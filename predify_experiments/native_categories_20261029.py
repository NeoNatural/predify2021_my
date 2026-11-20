"""Exploratory filtered-input, concept/preview-disjoint CV protocol."""
from __future__ import annotations
import argparse, hashlib, json, random, time, shutil
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from .supplementary import ROOT, load, save, digest
from .readout_init_20261012 import immutable_json, sha
from .tripartite_20261027 import CATEGORIES, SEED, setup_model, state_snapshot, restore_state, set_init, gray_image
from .paper_stimuli import imagenet_supercategory

OLD=ROOT/'Hebbian-CNN/Log/tripartite_20261027'
OUT=ROOT/'Hebbian-CNN/Log/native_categories_readouts_20261029'
DOC=ROOT/'docs/native_categories_readouts_20261029'
PLAN=ROOT/'docs/native_categories_readouts_20261029/执行方案.md'
IDENTITIES=('exact','same_concept','same_category','different_category')
CARRIERS=('hebb','pcoder','joint')
LAYERS=('conv4','conv5','fc1','fc2')
NOISES=(0,2921,2922,2923)
SEED=20261029
METRICS=('mean','mean_abs','nonzero','max','q25','q50','q75')

def conditions():
    return [dict(name=f'{c}_{i}',carrier=c,identity=i,coeff=.05 if c!='pcoder' else 0.) for c in CARRIERS for i in IDENTITIES]

def atomic_npz(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**values)
    temp.replace(path)

def encode_geometry(values):
    """Lossless byte shuffle before ZIP; FP32 bit patterns are unchanged."""
    return {k+'_bytes' if a.ndim==2 and a.dtype==np.float32 else k:
            np.ascontiguousarray(a.view(np.uint8).reshape(*a.shape,4).transpose(2,0,1)) if a.ndim==2 and a.dtype==np.float32 else a for k,a in values.items()}

def geometry_value(z,key):
    if key in z:return z[key]
    a=z[key+'_bytes'];return np.ascontiguousarray(a.transpose(1,2,0)).view(np.float32).reshape(a.shape[1:])

def design_filter():
    from scipy.signal import kaiserord, firwin, freqz
    nyquist=1/7;pass_edge=.8*nyquist
    taps,beta=kaiserord(70,(nyquist-pass_edge)/.5);taps+=1-taps%2
    kernel=firwin(taps,(nyquist+pass_edge)/2,window=('kaiser',beta),fs=1.)
    freq,response=freqz(kernel,worN=131072,fs=1.);db=20*np.log10(np.maximum(np.abs(response),1e-15))
    ripple=float(np.ptp(db[freq<=pass_edge]));stop=float(-np.max(db[freq>=nyquist]))
    assert ripple<=.1 and stop>=60,(ripple,stop)
    audit=dict(taps=taps,beta=beta,pass_edge=pass_edge,stop_edge=nyquist,ripple_db=ripple,stop_attenuation_db=stop,
               boundary='reflect at each 224 object boundary',sampling='bilinear align_corners=False; pixel centers (j+.5)*3.5-.5',
               original_resize='torchvision VGG16 IMAGENET1K_V1: PIL bilinear resize shorter edge256 (PIL antialias), center crop224, ToTensor/Normalize',
               preview_contrast=.5,extra_gaussian_sigma_input=3.5,extra_gaussian_sigma_output=1.,gaussian_radius=14,
               finite_filter_tolerance='Specified frequency response, not ideal infinite band limitation')
    return kernel,freq,db,audit

class FilteredInput:
    def __init__(self,torch,device='cuda'):
        self.torch=torch;self.device=device;k,_,_,_=design_filter()
        self.kernel=torch.tensor(k,dtype=torch.float32,device=device)
        v=torch.arange(-14,15,dtype=torch.float64,device=device);g=torch.exp(-.5*(v/3.5)**2);self.gaussian=(g/g.sum()).float()
    def separable(self,x,k):
        F=self.torch.nn.functional;c=x.shape[0];r=len(k)//2
        y=F.conv2d(F.pad(x[None],(r,r,0,0),mode='reflect'),k.view(1,1,1,-1).repeat(c,1,1,1),groups=c)
        return F.conv2d(F.pad(y,(0,0,r,r),mode='reflect'),k.view(1,1,-1,1).repeat(c,1,1,1),groups=c)[0]
    def tile(self,x,preview=False):
        if preview:
            mean=x.mean((-2,-1),keepdim=True);x=mean+.5*(x-mean);x=self.separable(x,self.gaussian)
        x=self.separable(x,self.kernel)
        return self.torch.nn.functional.interpolate(x[None],size=(64,64),mode='bilinear',align_corners=False,antialias=False)[0]
    def canvas(self,objects,preview=False):
        mean=self.torch.tensor([.485,.456,.406],device=self.device)[:,None,None];std=self.torch.tensor([.229,.224,.225],device=self.device)[:,None,None]
        canvas=((.5-mean)/std).expand(3,224,224).clone()
        for x,side in objects:
            x0={'left':5,'center':80,'right':155}[side];canvas[:,80:144,x0:x0+64]=self.tile(x,preview)
        return canvas

def input_acceptance():
    import torch
    from scipy.signal import freqz
    torch.set_num_threads(4);f=FilteredInput(torch,'cpu');k,freq,db,meta=design_filter()
    x=torch.ones(3,224,224)*.4
    constant_error=float((f.tile(x)-.4).abs().max());assert constant_error<2e-6
    rows=[];axis=torch.arange(224);grid=axis[None,:].expand(224,-1)
    for frequency in (.04,.08,.1142857,1/7,.18,.25,.4):
        g=torch.cos(2*torch.pi*frequency*grid).repeat(3,1,1)
        central=f.tile(g);preview=f.tile(g,True)
        # Exclude reflection edge transients from measured interior alias energy.
        # Contrast reduction preserves each finite grating's channel mean.
        # Remove DC before measuring residual oscillatory/alias energy.
        a=central[:,28:36,28:36];b=preview[:,28:36,28:36]
        central_rms=float((a-a.mean()).square().mean().sqrt());preview_rms=float((b-b.mean()).square().mean().sqrt())
        if frequency>=1/7:assert central_rms<.002
        assert preview_rms<central_rms
        rows.append(dict(frequency=frequency,central_ac_rms=central_rms,preview_ac_rms=preview_rms,central_mean=float(a.mean()),preview_mean=float(b.mean())))
    v=torch.randn(3,224,224,generator=torch.Generator().manual_seed(SEED));m=v.mean((-2,-1),keepdim=True);contrast=m+.5*(v-m)
    assert torch.allclose(contrast.mean((-2,-1)),v.mean((-2,-1)),atol=1e-7)
    assert torch.allclose(contrast.std((-2,-1)),.5*v.std((-2,-1)),atol=1e-7)
    # Verify actual FP32 GPU filtering against CPU reference without labels.
    cuda=FilteredInput(torch,'cuda');gpu_error=float((cuda.tile(v.cuda()).cpu()-f.tile(v)).abs().max());assert gpu_error<2e-5
    board=((torch.arange(224)[:,None]+torch.arange(224)[None,:])%2).float().repeat(3,1,1)*2-1
    board_rms=float(f.tile(board)[:,28:36,28:36].square().mean().sqrt());assert board_rms<1e-5
    meta.update(constant_max_error=constant_error,gpu_cpu_max_error=gpu_error,gratings=rows,checkerboard_rms=board_rms,contrast_formula_accepted=True,passed=True)
    immutable_json(DOC/'input_acceptance.json',meta);atomic_npz(OUT/'filter.npz',kernel=k,frequency=freq,response_db=db)
    print('INPUT ACCEPTED',meta,flush=True)

def activation_stats(x):
    import torch
    v=x.flatten().float();q=torch.quantile(v,torch.tensor([.25,.5,.75],device=v.device))
    return torch.stack([v.mean(),v.abs().mean(),(v!=0).float().mean(),v.max(),*q]).cpu().numpy()

def pair_statistics(a,b):
    # Pair of complete flattened FP32 outputs. Scalar sumsq is sum_j(sum_i x_ij^2),
    # algebraically sufficient for trace variance; no diagonal covariance is used.
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    norms_a=np.sqrt(np.einsum('td,td->t',a.astype(np.float64),a.astype(np.float64)))
    norms_b=np.sqrt(np.einsum('td,td->t',b.astype(np.float64),b.astype(np.float64)))
    u=a/np.maximum(norms_a[:,None],1e-8);v=b/np.maximum(norms_b[:,None],1e-8)
    return dict(sum_x=a+b,sum_u=(u+v).astype(np.float32),sum_sq=norms_a**2+norms_b**2,
                sum_unorm_sq=np.einsum('td,td->t',u,u)+np.einsum('td,td->t',v,v),zeros=(norms_a==0).astype(int)+(norms_b==0).astype(int))

def gray_snapshot(torch,model,layers):
    from .sequence import reset_hebbian_state,run_recurrent_sequence
    model.reset();reset_hebbian_state(layers);set_init(model,False)
    grey=gray_image(torch)
    with torch.no_grad():run_recurrent_sequence(model,grey[None],22,hebbian_layers=layers,commit_update=False)
    current=state_snapshot(model)
    old=json.loads((OLD/'imagenet/test/grey_state_hash.json').read_text(encoding='utf8'));assert [digest(s['rep']) for s in current]==old['rep_sha'],'Grey state differs from original'
    path=OUT/'gray_snapshot.pckl'
    cpu=[{k:v.cpu() for k,v in s.items()} for s in current]
    hashes=[{k:digest(v) for k,v in s.items()} for s in cpu]
    if path.exists():
        saved=load(path);assert hashes==saved['hashes']
        current=[{k:v.cuda() for k,v in s.items()} for s in saved['state']]
    else:save(path,dict(state=cpu,hashes=hashes,steps=22,original_rep_hash_matched=True))
    restore_state(model,current)
    model.pcoder1.rep.add_(1.)
    assert digest(current[0]['rep'])==hashes[0]['rep'],'Restore aliases cached state'
    restore_state(model,current)
    immutable_json(DOC/'gray_acceptance.json',dict(hashes=hashes,original_rep_hash_matched=True,deep_copy_verified=True,steps=22))
    return grey,current

class Recorder:
    def __init__(self,layers):
        self.cache={};self.handles=[]
        for name,layer in zip(('fc2','fc1','conv5','conv4'),layers):
            def hook(module,inputs,output,name=name):self.cache[name]=(output.detach().clone(),module.x_tmp.detach().clone())
            self.handles.append(layer.register_forward_hook(hook))
    def close(self):
        for h in self.handles:h.remove()

def verify_freeze():
    f=json.loads((OUT/'freeze.json').read_text(encoding='utf8'))
    for p,h in f['sources'].items():assert sha(ROOT/p)==h,('Frozen source changed',p)
    for d,h in f['manifests'].items():assert sha(OUT/d/'manifest.json')==h
    assert sha(PLAN)==f['plan_sha']
    assert sha(OUT/'filter.npz')==f['filter_sha'] and sha(OUT/'gray_snapshot.pckl')==f['gray_sha']
    for p,h in f['acceptance_sha'].items():assert sha(DOC/p)==h
    for p,h in f['weight_hashes'].items():assert sha(p)==h

def extract(dataset,smoke=False):
    from PIL import Image
    from torchvision.models import VGG16_Weights
    from .sequence import reset_hebbian_state,run_recurrent_sequence
    if not smoke:verify_freeze()
    already=OUT/dataset/('smoke' if smoke else 'formal')/'complete.json'
    if already.exists():
        print('EXTRACTION ALREADY COMPLETE',dataset,already,flush=True)
        return
    m=json.loads((OUT/dataset/'manifest.json').read_text(encoding='utf8'));torch,model,layers,oldrec,config=setup_model();oldrec.close()
    immutable_json(OUT/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,torchvision=__import__('torchvision').__version__,
        scipy=__import__('scipy').__version__,sklearn=__import__('sklearn').__version__,device=torch.cuda.get_device_name(),dtype='model and output FP32; sums/Gram/scalars validated against FP64',
        deterministic_algorithms=True,noise_generator='CUDA torch.Generator, per image/seed, sigma .5 in normalized 224 object coordinates',
        geometry_storage='Lossless FP32 byte shuffle + ZIP; concept-pair sum_x/sum_u; scalar sum_sq equals sum over all feature squared sums'))
    recorder=Recorder(layers);transform=VGG16_Weights.IMAGENET1K_V1.transforms();f=FilteredInput(torch)
    grey,common=gray_snapshot(torch,model,layers);dest=OUT/dataset/('smoke' if smoke else 'formal');dest.mkdir(parents=True,exist_ok=True)
    events=[];phase=['idle']
    for index,layer in zip((5,4),layers[2:]):
        original=layer.commit_update
        def monitored(original=original,index=index):events.append((phase[0],index));return original()
        layer.commit_update=monitored
    def restore():restore_state(model,common);model.input_mem=grey[None]
    def image(index):
        r=m['images'][str(index)];assert sha(r['path'])==r['sha256']
        with Image.open(r['path']) as im:return transform(im.convert('RGB')).cuda()
    def run(trial,c,target,previews,full=True):
        events.clear();reset_hebbian_state(layers);model.reset();set_init(model,False);restore()
        for layer in layers[2:]:
            layer.coeff=c['coeff'];layer.decay=.5;layer.inh_c=4.;layer.sparse_thres=.9;layer.update_enabled=c['carrier'] in ('hebb','joint')
        boundary=[]
        for stage,preview in enumerate(previews):
            if stage and c['carrier']=='hebb':restore()
            phase[0]=f'preview_{stage}'
            run_recurrent_sequence(model,preview[None],2,hebbian_layers=layers,commit_update=c['carrier'] in ('hebb','joint'),preserve_model_state=True)
            boundary.append([float(l.boost_weight.norm()) for l in layers[2:]])
        if c['carrier']=='hebb':restore()
        before=[l.boost_weight.detach().clone() for l in layers[2:]];phase[0]='target'
        outputs={name:[] for name in LAYERS};acts=[];legacy=[];errors=[]
        def after(t,logits):
            vals=[];old=[]
            for name in LAYERS:
                value,tmp=recorder.cache[name];vals.append(activation_stats(value));old.append(activation_stats(tmp))
                outputs[name].append(value.flatten().float().cpu().numpy())
            acts.append(vals);legacy.append(old);errors.append([float(getattr(model,f'pcoder{k}').prediction_error) for k in range(1,6)])
        run_recurrent_sequence(model,target[None],15,hebbian_layers=layers,commit_update=False,preserve_model_state=True,after_step=after)
        assert all(torch.equal(a,l.boost_weight) for a,l in zip(before,layers[2:])), 'Target W changed'
        assert all(not l.boost_weight.count_nonzero() for l in layers[:2])
        expected=[(f'preview_{s}',i) for s in range(2) for i in (5,4)] if c['carrier'] in ('hebb','joint') else []
        assert events==expected
        if c['carrier']=='pcoder':assert all(not v.count_nonzero() for v in before)
        outputs={n:np.stack(v) for n,v in outputs.items()};acts=np.asarray(acts);legacy=np.asarray(legacy)
        assert all(np.isfinite(v).all() for v in outputs.values()) and np.isfinite(acts).all() and np.isfinite(legacy).all()
        return dict(fc2=outputs['fc2'],activation=acts,legacy_activation=legacy,internal_error=np.asarray(errors),target_hash=digest(target),
                    commits=list(events),boundary_norms=boundary,target_W_frozen=True,outputs=outputs,
                    output_min={n:float(v.min()) for n,v in outputs.items()})
    pairs=defaultdict(list)
    for trial in m['development'] if smoke else m['trials']:pairs[trial['concept']].append(trial)
    concepts=sorted(pairs)
    if smoke:
        concepts=[next(c for c in concepts if pairs[c][0]['category_index']==i) for i in range(len(m['categories']))]
    started=time.perf_counter();completed=0
    try:
        with torch.no_grad():
            # A complete condition is reduced to its exact Gram sufficient statistics
            # before proceeding, bounding storage independently of 48 conditions.
            for seed in NOISES[:1] if smoke else NOISES:
                for c in conditions():
                    for concept in concepts:
                        pair=pairs[concept];assert len(pair)==4
                        if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError('Less than 8 GiB available')
                        path=dest/f"concept_{concept}_seed_{seed}_{c['name']}.pckl"
                        if path.exists():continue
                        stimuli=[];previews=[]
                        for trial in pair:
                            idx=trial['dataset_index'];sp=dest/f'stimuli_{idx}.npz'
                            if not sp.exists():
                                im=image(idx);targets={0:f.canvas([(im,'center')])}
                                for sn in NOISES[1:]:
                                    g=torch.Generator(device='cuda').manual_seed((SEED+sn*1000003+idx*1009)%2147483647)
                                    noisy=im+torch.randn(im.shape,device='cuda',generator=g)*.5
                                    targets[sn]=f.canvas([(noisy,'center')])
                                tiles={i:image(i) for i in set(trial['previews'])|set(trial['primes'].values())}
                                ps={}
                                for identity in IDENTITIES:
                                    indices=list(trial['previews']);indices[trial['slot']]=trial['primes'][identity]
                                    ps[identity]=[f.canvas([(tiles[i],side) for i,side in zip(indices[2*s:2*s+2],('left','right'))],True) for s in range(2)]
                                values={f'target_{sn}':v.cpu().numpy() for sn,v in targets.items()}
                                values.update({f'{identity}_preview{s}':v.cpu().numpy() for identity,pv in ps.items() for s,v in enumerate(pv)})
                                atomic_npz(sp,**values)
                            with np.load(sp) as z:
                                targets={sn:torch.tensor(z[f'target_{sn}'],device='cuda') for sn in NOISES}
                                ps={i:[torch.tensor(z[f'{i}_preview{s}'],device='cuda') for s in range(2)] for i in IDENTITIES}
                            stimuli.append(targets);previews.append(ps)
                            rp=dest/f'reference_{idx}.pckl'
                            if not rp.exists():
                                ref=run(trial,dict(carrier='reference',coeff=0.),targets[0],[])
                                ref.pop('outputs');assert ref['commits']==[]
                                save(rp,dict(trial=trial,result=ref))
                        results=[run(t,c,stimuli[j][seed],previews[j][c['identity']]) for j,t in enumerate(pair)]
                        sufficient={n:group_statistics([r['outputs'][n] for r in results]) for n in LAYERS}
                        geom=dest/f"geometry_{concept}_seed_{seed}_{c['name']}.npz"
                        atomic_npz(geom,**encode_geometry({f'{n}_{key}':v for n,values in sufficient.items() for key,v in values.items()}))
                        if concept==concepts[0] and seed==0 and c['identity']=='exact':
                            atomic_npz(dest/f"diagnostic_{c['name']}.npz",**{f'image{j}_{n}':r['outputs'][n] for j,r in enumerate(results) for n in LAYERS})
                        for r in results:r.pop('outputs')
                        noise_stats=[]
                        for j,t in enumerate(pair):
                            target=stimuli[j][seed][:,80:144,80:144];clean=stimuli[j][0][:,80:144,80:144];delta=target-clean
                            noise_stats.append(dict(sd=float(delta.std(unbiased=False)),rmse=float(delta.square().mean().sqrt()),snr_db=None if seed==0 else float(10*torch.log10(clean.square().mean()/delta.square().mean()))))
                        save(path,dict(concept=concept,seed=seed,condition=c,trials=pair,results=results,noise_stats=noise_stats,geometry_file=geom.name,source_sha=sha(__file__)))
                        completed+=4
                        if smoke and c['identity']=='exact':
                            zero=dict(c,coeff=0.);control=run(pair[0],zero,stimuli[0][0],previews[0]['exact'])
                            if c['carrier']=='joint':
                                other=run(pair[0],dict(c,carrier='pcoder',coeff=0.),stimuli[0][0],previews[0]['exact'])
                                assert np.array_equal(control['fc2'],other['fc2'])
                            if c['carrier']=='hebb':
                                other=run(pair[0],zero,stimuli[0][0],previews[0]['different_category'])
                                assert np.array_equal(control['fc2'],other['fc2'])
                                reference=load(dest/f"reference_{pair[0]['dataset_index']}.pckl")['result']
                                assert np.array_equal(control['fc2'],reference['fc2'])
                        print(f"{dataset} {dest.name} {concept} seed={seed} {c['name']} new={completed} elapsed={time.perf_counter()-started:.1f}s",flush=True)
                    if not smoke:reduce_geometry(dataset,seed,c['name'],concepts,torch)
    finally:recorder.close()
    immutable_json(dest/'complete.json',dict(dataset=dataset,concepts=concepts,seeds=NOISES[:1] if smoke else NOISES,sequences=len(concepts)*4*(1 if smoke else 4)*12,reference_sequences=len(concepts)*4,elapsed_seconds=time.perf_counter()-started))


def reduce_geometry(dataset,seed,condition,concepts,torch):
    """Lossless sufficient-statistic reduction; keep FP64 dot products and scalars."""
    formal=OUT/dataset/'formal';paths=[formal/f'geometry_{c}_seed_{seed}_{condition}.npz' for c in concepts]
    destinations=[OUT/dataset/'grams'/f'{condition}_{seed}_{layer}.npz' for layer in LAYERS]
    if all(p.exists() for p in destinations):
        for p in paths:
            if p.exists():p.unlink()
        return
    assert all(p.exists() for p in paths),'Incomplete condition reduction inputs'
    work=OUT/dataset/'reduction_workspace';work.mkdir(parents=True,exist_ok=True)
    for layer,dest in zip(LAYERS,destinations):
        if dest.exists():continue
        with np.load(paths[0]) as z:shape=geometry_value(z,f'{layer}_sum_x').shape
        raw=np.lib.format.open_memmap(work/'raw.npy',mode='w+',dtype='float32',shape=(len(concepts),*shape))
        norm=np.lib.format.open_memmap(work/'norm.npy',mode='w+',dtype='float32',shape=(len(concepts),*shape))
        sq=[];unorm=[];zeros=[]
        for j,p in enumerate(paths):
            with np.load(p) as z:
                raw[j]=geometry_value(z,f'{layer}_sum_x');norm[j]=geometry_value(z,f'{layer}_sum_u')
                sq.append(z[f'{layer}_sum_sq']);unorm.append(z[f'{layer}_sum_unorm_sq']);zeros.append(z[f'{layer}_zeros'])
        gr=[];gu=[]
        for t in range(15):
            a=torch.tensor(np.asarray(raw[:,t]),dtype=torch.float64,device='cuda');b=torch.tensor(np.asarray(norm[:,t]),dtype=torch.float64,device='cuda')
            gr.append((a@a.T).cpu().numpy());gu.append((b@b.T).cpu().numpy())
        values=dict(graw=np.array(gr),gnorm=np.array(gu),sq=np.stack(sq,axis=1),unorm=np.stack(unorm,axis=1),zeros=np.stack(zeros,axis=1),concepts=np.array(concepts),images_per_concept=np.array(4))
        assert all(np.isfinite(v).all() for k,v in values.items() if k!='concepts')
        atomic_npz(dest,**values)
        del raw,norm,a,b
        (work/'raw.npy').unlink();(work/'norm.npy').unlink()
    # Delete only generated intermediates after all four Gram files pass a reopen check.
    for p in destinations:
        with np.load(p) as z:assert z['graw'].shape==(15,len(concepts),len(concepts)) and np.array_equal(z['concepts'],concepts)
    for p in paths:p.unlink()

def group_statistics(arrays):
    pairs=[pair_statistics(arrays[i],arrays[i+1]) for i in range(0,len(arrays),2)]
    return {k:sum(v[k] for v in pairs) for k in pairs[0]}
