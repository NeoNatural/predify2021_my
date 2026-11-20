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
OUT=ROOT/'Hebbian-CNN/Log/tripartite_filtered_cv_20261027'
DOC=ROOT/'docs/tripartite_filtered_cv_20261027'
PLAN=ROOT/'docs/tripartite_filtered_plan_20261027/实验计划.md'
IDENTITIES=('exact','same_concept','same_category','different_category')
CARRIERS=('hebb','pcoder','joint')
LAYERS=('conv4','conv5','fc1','fc2')
NOISES=(0,2721,2722,2723)
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

def prepare():
    from torchvision.datasets import ImageNet
    from .things_readout_data import DATA_ROOT
    DOC.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if not (DOC/'预先方案.md').exists():shutil.copyfile(PLAN,DOC/'预先方案.md')
    assert sha(PLAN)==sha(DOC/'预先方案.md')
    old={d:json.loads((OLD/d/'manifest.json').read_text(encoding='utf8')) for d in ('imagenet','things')}
    im=ImageNet('C:/Users/liang/Documents/ImageNet','val')
    pools={d:defaultdict(list) for d in old}
    paths={'imagenet':{i:Path(p) for i,(p,_) in enumerate(im.samples)},
           'things':{int(i):DATA_ROOT/'images'/r['relative_path'] for i,r in old['things']['images'].items()}}
    labels={'imagenet':im.targets,'things':{int(i):r['imagenet_label'] for i,r in old['things']['images'].items()}}
    for d in old:
        for i in paths[d]:
            l=int(labels[d][i])
            if imagenet_supercategory(l) in CATEGORIES:pools[d][l].append(i)
    assignments={};rng=random.Random(SEED)
    for cat in range(4):
        concepts=sorted({t['concept'] for t in old['imagenet']['test'] if t['category_index']==cat});rng.shuffle(concepts)
        assert len(concepts)==8
        for l,f in zip(concepts,[0,0,1,1,2,2,3,4]):assignments[l]=f
    target_concepts=set(assignments);preview_only=[]
    for cat in CATEGORIES:
        candidates=sorted(l for l in pools['things'] if l not in target_concepts and imagenet_supercategory(l)==cat and pools['imagenet'][l])
        rng.shuffle(candidates)
        for f in range(5):
            count=sum(imagenet_supercategory(l)==cat and fold==f for l,fold in assignments.items())
            for _ in range(max(0,2-count)):
                if not candidates:raise RuntimeError(f'Inventory gap: {cat}, fold {f}; no relaxation permitted')
                l=candidates.pop();assignments[l]=f;preview_only.append(l)
    audits={}
    for d in old:
        protected={t['dataset_index'] for s in ('train','dev','test') for t in old[d][s]}
        reserves={l:[i for i in pools[d][l] if i not in protected] for l in assignments}
        gaps=[l for l in assignments if not reserves[l]]
        if gaps:raise RuntimeError(f'{d}: preview inventory gap {gaps}; no relaxation permitted')
        trials=[];roles=[]
        for j,t in enumerate(old[d]['test']):
            index=t['dataset_index'];l=t['concept'];cat=t['category_index'];f=assignments[l]
            local=random.Random(SEED+index)
            cp={c:[label for label in assignments if assignments[label]==f and imagenet_supercategory(label)==c] for c in CATEGORIES}
            other=[c for c in CATEGORIES if c!=CATEGORIES[cat]]
            cats=other+[other[(j//4)%3]];used=[];background=[]
            for c in cats:
                choice=local.choice([v for v in cp[c] if v not in used]);used.append(choice);background.append(local.choice(reserves[choice]))
            local.shuffle(background);slot=2+(j//4+j%4)%2
            same=local.choice([v for v in cp[CATEGORIES[cat]] if v!=l])
            primes=dict(exact=index,same_concept=local.choice(reserves[l]),same_category=local.choice(reserves[same]),different_category=background[slot])
            trial=dict(dataset_index=index,concept=l,category_index=cat,fold=f,previews=background,slot=slot,primes=primes,previously_viewed=True)
            trials.append(trial)
            for identity in IDENTITIES:
                indices=list(background);indices[slot]=primes[identity]
                for pos,i in enumerate(indices):
                    label=int(labels[d][i]);assert assignments[label]==f
                    assert i not in protected or (identity=='exact' and i==index)
                    roles.append(dict(target=index,identity=identity,position=pos,role='prime' if pos==slot else 'distractor',image=i,concept=label,fold=f))
            roles.append(dict(target=index,identity='all',position='center',role='target',image=index,concept=l,fold=f))
        used_ids=sorted({r['image'] for r in roles})
        historical=set(old[d]['historical_seen'])|protected
        for s in ('train','dev','test'):
            for t in old[d][s]:historical.update(t['previews']);historical.update(i for i in t['primes'].values() if i is not None)
        images={str(i):dict(path=str(paths[d][i]),concept=int(labels[d][i]),category=CATEGORIES.index(imagenet_supercategory(labels[d][i])),sha256=sha(paths[d][i]),historical=i in historical) for i in used_ids}
        # Byte-level independence supplements concept ownership.
        bytefold=defaultdict(set)
        for r in roles:bytefold[images[str(r['image'])]['sha256']].add(r['fold'])
        assert all(len(v)==1 for v in bytefold.values()),'Identical image bytes cross folds'
        counts=Counter(t['fold'] for t in trials);assert [counts[f] for f in range(5)]==[16,16,16,8,8]
        for f in range(5):
            train={r['concept'] for r in roles if r['fold']!=f};test={r['concept'] for r in roles if r['fold']==f};assert not train&test
        m=dict(dataset=d,seed=SEED,trials=trials,images=images,visual_roles=roles,concept_folds=assignments,preview_only=preview_only,
               noises=NOISES,conditions=conditions(),old_manifest_sha=sha(OLD/d/'manifest.json'),plan_sha=sha(PLAN),scope='Previously viewed 64-target exploratory repeated-measures cohort; all visual concepts disjoint between CV train/test')
        immutable_json(OUT/d/'manifest.json',m)
        audits[d]=dict(targets=len(trials),target_concepts=len(target_concepts),fold_sizes=dict(counts),preview_only_concepts=preview_only,role_rows=len(roles),
                       visual_concept_overlap_per_fold=[0]*5,visual_byte_overlap_per_fold=[0]*5,other_target_images_in_preview=0,images=len(images),preview_inventory_gaps=[])
    immutable_json(DOC/'inventory_acceptance.json',audits)
    print('INVENTORY ACCEPTED',audits,flush=True)

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
    for name,expected in f['sources'].items():assert sha(ROOT/name)==expected,('Frozen source changed',name)
    for d in ('imagenet','things'):assert sha(OUT/d/'manifest.json')==f['manifests'][d]
    assert sha(PLAN)==f['plan_sha'] and sha(OUT/'filter.npz')==f['filter_sha']
    assert sha(OUT/'gray_snapshot.pckl')==f['gray_sha']
    for p,expected in f['weight_hashes'].items():assert sha(p)==expected,('Weight changed',p)

def extract(dataset,smoke=False):
    from PIL import Image
    from torchvision.models import VGG16_Weights
    from .sequence import reset_hebbian_state,run_recurrent_sequence
    if not smoke:verify_freeze()
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
    for trial in m['trials']:pairs[trial['concept']].append(trial)
    concepts=sorted(pairs)
    if smoke:concepts=concepts[:1]
    started=time.perf_counter();completed=0
    try:
        with torch.no_grad():
            for concept in concepts:
                pair=pairs[concept];assert len(pair)==2
                if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError('Less than 8 GiB available: preserve outputs and stop for lossless storage review')
                stimuli=[];previews=[]
                for trial in pair:
                    idx=trial['dataset_index'];im=image(idx);clean=f.canvas([(im,'center')]);targets={0:clean}
                    for seed in NOISES[1:]:
                        g=torch.Generator(device='cuda').manual_seed((SEED+seed*1000003+idx*1009)%2147483647)
                        noisy=im+torch.randn(im.shape,device='cuda',generator=g)*.5;targets[seed]=f.canvas([(noisy,'center')])
                    tiles={i:image(i) for i in set(trial['previews'])|set(trial['primes'].values())}
                    ps={}
                    for identity in IDENTITIES:
                        indices=list(trial['previews']);indices[trial['slot']]=trial['primes'][identity]
                        ps[identity]=[f.canvas([(tiles[i],side) for i,side in zip(indices[2*s:2*s+2],('left','right'))],True) for s in range(2)]
                    stimuli.append(targets);previews.append(ps)
                    if not (dest/f'stimuli_{idx}.npz').exists():
                        values={f'target_{seed}':v.cpu().numpy() for seed,v in targets.items()}
                        values.update({f'{identity}_preview{s}':v.cpu().numpy() for identity,pv in ps.items() for s,v in enumerate(pv)})
                        atomic_npz(dest/f'stimuli_{idx}.npz',**values)
                for seed in NOISES[:1] if smoke else NOISES:
                    for c in conditions():
                        name=c['name'];path=dest/f'concept_{concept}_seed_{seed}_{name}.pckl'
                        if path.exists():continue
                        results=[run(t,c,stimuli[j][seed],previews[j][c['identity']]) for j,t in enumerate(pair)]
                        sufficient={n:pair_statistics(results[0]['outputs'][n],results[1]['outputs'][n]) for n in LAYERS}
                        geom=dest/f'geometry_{concept}_seed_{seed}_{name}.npz'
                        atomic_npz(geom,**encode_geometry({f'{n}_{key}':v for n,values in sufficient.items() for key,v in values.items()}))
                        # Preserve full map diagnostic examples; ordinary files retain exact concept sufficient statistics.
                        if concept==concepts[0] and seed==0 and c['identity']=='exact':
                            atomic_npz(dest/f'diagnostic_{name}.npz',**{f'image{j}_{n}':r['outputs'][n] for j,r in enumerate(results) for n in LAYERS})
                        for r in results:r.pop('outputs')
                        noise_stats=[]
                        for j,t in enumerate(pair):
                            target=stimuli[j][seed][:,80:144,80:144];clean=stimuli[j][0][:,80:144,80:144];delta=target-clean
                            noise_stats.append(dict(sd=float(delta.std(unbiased=False)),rmse=float(delta.square().mean().sqrt()),
                                                   snr_db=None if seed==0 else float(10*torch.log10(clean.square().mean()/delta.square().mean()))))
                        save(path,dict(concept=concept,seed=seed,condition=c,trials=pair,results=results,noise_stats=noise_stats,geometry_file=geom.name,source_sha=sha(__file__)))
                        completed+=2
                        if smoke and c['identity']=='exact':
                            zero=dict(c,coeff=0.);control=run(pair[0],zero,stimuli[0][0],previews[0]['exact'])
                            if c['carrier']=='joint':
                                p=run(pair[0],dict(c,carrier='pcoder',coeff=0.),stimuli[0][0],previews[0]['exact'])
                                assert np.array_equal(control['fc2'],p['fc2'])
                            if c['carrier']=='hebb':
                                other=run(pair[0],zero,stimuli[0][0],previews[0]['different_category']);assert np.array_equal(control['fc2'],other['fc2'])
                        print(f'{dataset} {dest.name} concept={concept} seed={seed} condition={name} new_sequences={completed} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    finally:recorder.close()
    immutable_json(dest/'complete.json',dict(dataset=dataset,concepts=concepts,seeds=NOISES[:1] if smoke else NOISES,sequences=len(concepts)*2*(1 if smoke else 4)*12,elapsed_seconds=time.perf_counter()-started))

def freeze():
    for p in ('inventory_acceptance.json','input_acceptance.json','gray_acceptance.json','unit_acceptance.json'):
        assert (DOC/p).exists(),p
    for d in ('imagenet','things'):assert (OUT/d/'smoke/complete.json').exists()
    files=['predify_experiments/filtered_cv_20261027.py','analysis/filtered_cv_20261027.py','predify_experiments/tripartite_20261027.py',
           'predify_experiments/sequence.py','predify_experiments/hebb_vgg_priming.py','Hebbian-CNN/OnlineSharpeningStats_Demo.py',
           'predify_experiments/readout_init_20261012.py','Hebbian-CNN/Hebb_Priming_Utils.py','configs/paper_memory_carrier_calibrated.toml',
           'Hebbian-CNN/Hebbian_VGG_Lib.py','Hebbian-CNN/predify_hebb_inject.py','predify2021/model_factory/get_model.py','predify2021/model_factory/pvgg16_shared.py']
    weights=list((ROOT/'predify2021/model_factory').glob('pvgg16*pmodule.pth'))
    weights+=list((Path.home()/'.cache/torch/hub/checkpoints').glob('vgg16*.pth'))
    assert len(weights)>=6,'Missing weight provenance'
    values=dict(sources={p:sha(ROOT/p) for p in files},manifests={d:sha(OUT/d/'manifest.json') for d in ('imagenet','things')},
                plan_sha=sha(PLAN),filter_sha=sha(OUT/'filter.npz'),gray_sha=sha(OUT/'gray_snapshot.pckl'),frozen_at=time.time(),
                exploratory=True,baseline='different_category within carrier/noise/time/layer',bootstrap=10000,cv_fits=3620,
                weight_hashes={str(p):sha(p) for p in weights})
    immutable_json(OUT/'freeze.json',values)
    for p in files:
        dest=OUT/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dest)
    print('FROZEN',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','inputs','smoke','extract','freeze'));p.add_argument('--dataset',choices=('imagenet','things'),default='imagenet');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='inputs':input_acceptance()
    elif a.action=='freeze':freeze()
    else:extract(a.dataset,a.action=='smoke')
