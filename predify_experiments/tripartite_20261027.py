"""Four-category, common-grey-state Hebbian/PCoder experiment (2026-10-27)."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np

from .supplementary import ROOT, load, save, digest
from .readout_init_20261012 import immutable_json, sha
from .paper_stimuli import imagenet_supercategory, make_paper_canvas

OUT=ROOT/'Hebbian-CNN/Log/tripartite_20261027'
DOC=ROOT/'docs/tripartite_20261027'
SEED=20261027
CATEGORIES=('animals','clothing','food','vehicles')
IDENTITIES=('exact','same_concept','same_category','different_category','neutral')
STEPS=15
NOISES={'train':(2701,2702,2703),'dev':(2711,2712,2713),'test':(2721,2722,2723)}


def conditions(weak=True):
    rows=[dict(name='none',carrier='none',identity='exact',coeff=0.,init='grey')]
    for carrier in ('hebb','pcoder','joint'):
        for identity in IDENTITIES:
            rows.append(dict(name=f'{carrier}_{identity}',carrier=carrier,identity=identity,
                             coeff=.05 if carrier in ('hebb','joint') else 0.,init='grey'))
    if weak:
        for carrier in ('hebb','joint'):
            for identity in IDENTITIES:
                rows.append(dict(name=f'{carrier}_weak_{identity}',carrier=carrier,identity=identity,coeff=.0125,init='grey'))
    rows.extend([dict(name='none_random',carrier='none',identity='exact',coeff=0.,init='random'),
                 dict(name='none_feedforward',carrier='none',identity='exact',coeff=0.,init='feedforward')])
    return rows


def state_snapshot(model):
    return [{field:getattr(getattr(model,f'pcoder{k}'),field).detach().clone()
             for field in ('rep','prd','grd','prediction_error')} for k in range(1,6)]


def restore_state(model,state):
    for k,values in enumerate(state,1):
        pc=getattr(model,f'pcoder{k}')
        for field,value in values.items():setattr(pc,field,value.detach().clone())


def make_canvas(images,old=False):
    return make_paper_canvas(images,object_scale=.15 if old else 64/224,
                             horizontal_offset=.175 if old else 1/3,background_value=.5)


def historical_imagenet():
    seen=set()
    old=load(ROOT/'Hebbian-CNN/Log/supplementary_20261009/manifest.pckl')
    seen.update(old['excluded_previous_images']);seen.update(old['confirmation_indices'])
    for t in old['exploration']:
        seen.add(t['dataset_index']);seen.update(t['previews']);seen.update(i for i in t['primes'].values() if i is not None)
    for folder in ('stage_b','stage_d'):
        for p in (ROOT/'Hebbian-CNN/Log/supplementary_20261009'/folder).glob('*.pckl'):
            d=load(p);seen.add(d['trial']['dataset_index'])
            for r in d['records']:seen.update(i for i in r['preview_indices'] if i is not None)
    m=json.loads((ROOT/'Hebbian-CNN/Log/readout_init_20261012/manifest.json').read_text())
    for s in ('train','dev','test'):
        for t in m[s]:
            seen.add(t['dataset_index']);seen.update(t['previews']);seen.update(i for i in t['primes'].values() if i is not None)
    return seen


def prepare():
    from torchvision.datasets import ImageNet
    from .things_readout_data import inventory
    records,things_labels,old,audit=inventory()
    things_seen=set(old)
    for name in ('things_readout_20261012','things_readout_20261012_unbalanced_preview_pilot'):
        m=json.loads((ROOT/'Hebbian-CNN/Log'/name/'manifest.json').read_text())
        for s in ('train','dev','test'):
            for t in m[s]:
                things_seen.add(t['dataset_index']);things_seen.update(t['previews']);things_seen.update(i for i in t['primes'].values() if i is not None)
    im=ImageNet('C:/Users/liang/Documents/ImageNet','val')
    im_seen=historical_imagenet();im_labels=im.targets
    pools={d:defaultdict(list) for d in ('things','imagenet')}
    for i,label in enumerate(things_labels):
        if imagenet_supercategory(label) in CATEGORIES:pools['things'][int(label)].append(i)
    for i,label in enumerate(im_labels):
        if imagenet_supercategory(label) in CATEGORIES:pools['imagenet'][int(label)].append(i)
    rng=random.Random(SEED);group_split={s:[] for s in ('train','dev','test')}
    inventory_rows=[]
    for category in CATEGORIES:
        available=[l for l,p in pools['things'].items() if imagenet_supercategory(l)==category and len(p)>=3]
        fresh=[l for l in available if sum(i not in things_seen for i in pools['things'][l])>=2 and sum(i not in im_seen for i in pools['imagenet'][l])>=2]
        rng.shuffle(fresh)
        if len(fresh)<8:raise ValueError(f'{category}: insufficient fresh confirmation concepts')
        group_split['test'].extend(fresh[:8])
        remaining=[l for l in available if l not in fresh[:8]];rng.shuffle(remaining)
        if len(remaining)<14:raise ValueError(f'{category}: need 14 non-test concepts')
        group_split['dev'].extend(remaining[:4]);group_split['train'].extend(remaining[4:14])
        inventory_rows.append(dict(category=category,eligible_concepts=len(available),fresh_test_concepts=len(fresh),
            fresh_images=sum(i not in things_seen for l in available for i in pools['things'][l])))
    for dataset,labels,seen in [('imagenet',im_labels,im_seen),('things',things_labels,things_seen)]:
        selections={};preview_reserve={}
        for split,groups in group_split.items():
            chosen={c:[] for c in CATEGORIES}
            for label in groups:
                pool=list(pools[dataset][label]);rng.shuffle(pool)
                candidates=[i for i in pool if i not in seen] if split=='test' else pool
                if len(candidates)<2:raise ValueError('Insufficient images')
                chosen[imagenet_supercategory(label)].extend(candidates[:2])
                preview_reserve[label]=next(i for i in pool if i not in candidates[:2])
            selections[split]=[chosen[c][j] for j in range(len(next(iter(chosen.values())))) for c in CATEGORIES]
        protected={i for values in selections.values() for i in values}
        result={}
        for split,selected in selections.items():
            groups=set(group_split[split])
            cp={c:[] for c in CATEGORIES};lp={}
            for label in groups:
                lp[label]=[i for i in pools[dataset][label] if i not in protected]
                cp[imagenet_supercategory(label)].extend(lp[label])
            trials=[]
            for ordinal,index in enumerate(selected):
                label=int(labels[index]);category=imagenet_supercategory(label)
                local=random.Random(SEED+index);others=[c for c in CATEGORIES if c!=category]
                preview=[local.choice(cp[c]) for c in others]
                extra=others[(ordinal//4)%3]
                extra_candidates=[i for i in cp[extra] if i not in preview and int(labels[i]) not in [int(labels[j]) for j in preview if imagenet_supercategory(labels[j])==extra]]
                preview.append(local.choice(extra_candidates));local.shuffle(preview)
                slot=2+(ordinal//4+ordinal%4)%2
                trials.append(dict(dataset_index=index,concept=label,category_index=CATEGORIES.index(category),
                    previews=preview,slot=slot,primes=dict(exact=index,same_concept=local.choice(lp[label]),
                    same_category=local.choice([i for i in cp[category] if int(labels[i])!=label]),
                    different_category=preview[slot],neutral=None)))
            result[split]=trials
        manifest=dict(dataset=dataset,seed=SEED,categories=CATEGORIES,**result,concept_splits=group_split,
            historical_seen=sorted(seen),noise_seeds=NOISES,conditions=conditions(),steps=STEPS,
            scope='Concept-disjoint readout train/dev/test, including preview concept pools; confirmation images exclude all known historical targets and previews; old images allowed only in train/development.')
        if dataset=='things':manifest['images']=records
        immutable_json(OUT/dataset/'manifest.json',json.loads(json.dumps(manifest)))
    immutable_json(DOC/'inventory.json',dict(things_inventory=audit,stock=inventory_rows,
        concepts=group_split,split_images_per_dataset=dict(train=80,dev=32,test=64),
        pairwise_concept_overlap=0,paired_concept_quotas_across_datasets=True))
    print('PREPARED',inventory_rows,flush=True)


def setup_model():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from .config import load_experiment_config
    from .paper_parafoveal import _add_legacy_module_path,_FeatureRecorder
    from .hebb_vgg_priming import _build_model
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    _add_legacy_module_path(ROOT)
    config=load_experiment_config(ROOT/'configs/paper_memory_carrier_calibrated.toml').with_overrides(imagenet_root='C:/Users/liang/Documents/ImageNet',device='cuda')
    config.validate_for_run();torch.manual_seed(SEED)
    model,layers,_=_build_model(config,torch.device('cuda'),torch)
    for layer in layers:layer.update_enabled=False
    for layer in layers[:2]:layer.compute_enabled=False
    return torch,model,layers,_FeatureRecorder(model,layers,torch),config


def set_init(model,random_init):
    model.random_init=random_init
    for k in range(1,6):getattr(model,f'pcoder{k}').random_init=random_init


def gray_image(torch):
    mean=torch.tensor([.485,.456,.406],device='cuda')[:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[:,None,None]
    return ((.5-mean)/std).expand(3,224,224).clone()


def grey_calibration():
    torch,model,layers,recorder,config=setup_model()
    from .sequence import reset_hebbian_state
    grey=gray_image(torch);trace=[];stable=0;selected=None
    with torch.no_grad():
        reset_hebbian_state(layers);model.reset();set_init(model,False)
        previous=None
        for step in range(1,129):
            model(grey[None] if step==1 else None)
            current=[getattr(model,f'pcoder{k}').rep.detach().clone() for k in range(1,6)]
            changes=[float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-8)) for a,b in zip(current,previous)] if previous is not None else [1.]*5
            trace.append(dict(step=step,relative_changes=changes));previous=current
            stable=stable+1 if max(changes)<1e-4 else 0
            if stable>=3:selected=step;break
    if selected is None:raise RuntimeError('Grey state did not stabilize within fixed cap; inspect stability, not classification')
    immutable_json(OUT/'grey.json',dict(steps=selected,criterion='all 5 relative PCoder rep changes <1e-4 for 3 consecutive steps',
        init='feedforward',trace=trace,selection_uses_target_labels=False))
    recorder.close();print('GREY_STEPS',selected,flush=True)


def extract(dataset,split,smoke=False,old=False):
    torch,model,layers,recorder,config=setup_model()
    from .sequence import reset_hebbian_state,run_recurrent_sequence
    from Hebb_Priming_Utils import apply_input_degradation
    from torchvision.models import VGG16_Weights
    from torchvision.datasets import ImageNet
    from .things_readout_data import ThingsImages
    manifest=json.loads((OUT/dataset/'manifest.json').read_text())
    dataset_obj=ThingsImages(manifest,VGG16_Weights.IMAGENET1K_V1.transforms()) if dataset=='things' else ImageNet(str(config.paths.imagenet_root),'val',transform=VGG16_Weights.IMAGENET1K_V1.transforms())
    freeze=OUT/dataset/'freeze.json'
    if split=='test':
        frozen=json.loads(freeze.read_text());assert frozen['manifest_sha']==sha(OUT/dataset/'manifest.json')
        assert frozen['head_sha']==sha(OUT/dataset/'head.pckl')
    selected_conditions=conditions()
    if old:selected_conditions=[r for r in conditions(False) if r['init']=='grey' and r['identity'] in ('exact','neutral')]
    directory=OUT/dataset/('smoke' if smoke else 'old_layout' if old else split)
    directory.mkdir(parents=True,exist_ok=True)
    source_hash=sha(__file__)
    immutable_json(directory/'provenance.json',dict(source_sha=source_hash,manifest_sha=sha(OUT/dataset/'manifest.json'),
        grey_sha=sha(OUT/'grey.json'),torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(),
        weights={p.name:sha(p) for p in config.paths.pcoder_weight_dir.glob('pvgg16*pmodule.pth')},
        FC_dynamic_update=False,FC_compute_bypass='after ReLU, zero weights/cut=0 is identity',old_layout=old))
    (directory/'source.py').write_bytes(Path(__file__).read_bytes())
    grey=gray_image(torch);set_init(model,False);reset_hebbian_state(layers)
    grey_steps=json.loads((OUT/'grey.json').read_text())['steps']
    with torch.no_grad():run_recurrent_sequence(model,grey[None],grey_steps,hebbian_layers=layers,commit_update=False)
    common=state_snapshot(model)
    immutable_json(directory/'grey_state_hash.json',dict(rep_sha=[digest(x['rep']) for x in common]))
    commit_events=[];phase=['idle']
    for layer_index,layer in zip((5,4),layers[2:]):
        original_commit=layer.commit_update
        def monitored_commit(original=original_commit,index=layer_index):
            commit_events.append(dict(layer=index,phase=phase[0]))
            return original()
        layer.commit_update=monitored_commit
    def restore():restore_state(model,common);model.input_mem=grey[None]
    def execute(trial,condition,target,images,noise,keep_content=None):
        commit_events.clear();phase[0]='initialization'
        reset_hebbian_state(layers);model.reset();set_init(model,False)
        for layer in layers[2:]:
            layer.coeff=condition['coeff'];layer.decay=.5;layer.inh_c=4;layer.sparse_thres=.9
            layer.update_enabled=condition['carrier'] in ('hebb','joint')
        identity=condition['identity'] if keep_content is None else keep_content
        previews=list(trial['previews']);previews[trial['slot']]=trial['primes'][identity]
        updates=[];boundary_norms=[]
        initial_w=[x.boost_weight.detach().clone() for x in layers[2:]]
        if condition['init']=='grey':
            restore()
            for stage in range(2):
                phase[0]=f'preview_boundary_{stage}'
                if stage>0 and condition['carrier'] not in ('pcoder','joint'):restore()
                preview=make_canvas([(images[i],side) for i,side in zip(previews[stage*2:stage*2+2],('left','right')) if i is not None],old)
                run_recurrent_sequence(model,preview[None],2,hebbian_layers=layers,
                    commit_update=condition['carrier'] in ('hebb','joint'),preserve_model_state=True)
                updates.append([int(x.update_enabled) for x in layers[2:]])
                boundary_norms.append([float(x.boost_weight.norm()) for x in layers[2:]])
            if condition['carrier'] not in ('pcoder','joint'):restore()
        else:
            set_init(model,condition['init']=='random');torch.manual_seed((SEED+trial['dataset_index']*1009)%2147483647)
        before=[x.boost_weight.detach().clone() for x in layers[2:]]
        features,logits,acts,errors=[],[],[],[]
        def after(t,output):
            snap=recorder.snapshot();features.append(snap['fc2']);logits.append(output.detach().cpu().numpy()[0])
            acts.append([float(x.x_tmp.detach().abs().mean()) for x in layers]);errors.append([float(getattr(model,f'pcoder{k}').prediction_error) for k in range(1,6)])
        start=time.perf_counter()
        phase[0]='target'
        run_recurrent_sequence(model,target[None],STEPS,hebbian_layers=layers,commit_update=False,
            preserve_model_state=condition['init']=='grey',after_step=after)
        for a,layer in zip(before,layers[2:]):assert torch.equal(a,layer.boost_weight),'Target modified W'
        assert all(not x.boost_weight.count_nonzero() for x in layers[:2]),'FC W changed'
        if condition['carrier'] in ('none','pcoder'):assert all(not x.count_nonzero() for x in before)
        expected=[dict(layer=index,phase=f'preview_boundary_{stage}') for stage in range(2) for index in (5,4)] if condition['carrier'] in ('hebb','joint') else []
        assert commit_events==expected,('Unexpected Hebbian commits',commit_events)
        return dict(**condition,fc2=np.stack(features),logits=np.stack(logits),activation=np.array(acts),internal_error=np.array(errors),
            target_hash=digest(target),preview_indices=previews,weight_norms=boundary_norms,target_W_frozen=True,
            commit_events=list(commit_events),target_seconds=time.perf_counter()-start)
    trials=manifest[split][:4] if smoke else manifest[split][:8] if old else manifest[split]
    noises=manifest['noise_seeds'][split][:1] if smoke or old else manifest['noise_seeds'][split]
    started=time.perf_counter()
    try:
        with torch.no_grad():
            for ordinal,trial in enumerate(trials):
                index=trial['dataset_index']
                missing=[s for s in noises if not (directory/f'image_{index}_seed_{s}.pckl').exists()]
                if not missing:continue
                image=dataset_obj[index][0].cuda()
                indices=set(trial['previews'])|{i for i in trial['primes'].values() if i is not None}
                images={i:dataset_obj[i][0].cuda() for i in indices}
                clean=make_canvas([(image,'center')],old)
                for noise in missing:
                    noise_seed=(SEED+noise*1000003+index*1009)%2147483647
                    damaged=apply_input_degradation(image,mode='predify_noise',level=.5,seed=noise_seed)
                    target=make_canvas([(damaged,'center')],old)
                    records=[execute(trial,c,target,images,noise) for c in selected_conditions]
                    if smoke:
                        for identity in IDENTITIES:
                            for carrier,baseline in [('hebb','none'),('joint',f'pcoder_{identity}')]:
                                zero=dict(name=f'zero_{carrier}_{identity}',carrier=carrier,identity=identity,coeff=0.,init='grey')
                                r=execute(trial,zero,target,images,noise)
                                anchor=next(x for x in records if x['name']==baseline)
                                assert np.array_equal(r['fc2'],anchor['fc2']) and np.array_equal(r['logits'],anchor['logits'])
                        anchor=next(x for x in records if x['name']=='none')
                        for identity in IDENTITIES:
                            r=execute(trial,selected_conditions[0],target,images,noise,keep_content=identity)
                            assert np.array_equal(r['fc2'],anchor['fc2']) and np.array_equal(r['logits'],anchor['logits'])
                    assert len({r['target_hash'] for r in records})==1
                    assert all(np.isfinite(r['fc2']).all() and np.isfinite(r['logits']).all() for r in records)
                    delta=target-clean;active=(delta!=0)
                    x0=95 if old else 80;x1=129 if old else 144
                    noise_stats=dict(canvas_std=float(delta.std()),tile_std=float(delta[:,x0:x1,x0:x1].std()),
                                     tile_rmse=float(delta[:,x0:x1,x0:x1].square().mean().sqrt()))
                    save(directory/f'image_{index}_seed_{noise}.pckl',dict(trial=trial,noise=noise,noise_seed=noise_seed,
                         records=records,noise_stats=noise_stats,source_sha=source_hash))
                    # Persist actual normalized tensors, not only presentation metadata.
                    np.savez_compressed(directory/f'stimulus_{index}_seed_{noise}.npz',target=target.cpu().numpy(),clean=clean.cpu().numpy())
                preview_canvases={}
                for identity in IDENTITIES:
                    indices=list(trial['previews']);indices[trial['slot']]=trial['primes'][identity]
                    for stage in range(2):
                        preview_canvases[f'{identity}_preview{stage}']=make_canvas([(images[i],side) for i,side in zip(indices[stage*2:stage*2+2],('left','right')) if i is not None],old).cpu().numpy()
                np.savez_compressed(directory/f'previews_{index}.npz',**preview_canvases)
                print(f'{dataset} {directory.name} {ordinal+1}/{len(trials)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    finally:recorder.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','grey','smoke','train','dev','test','old'))
    p.add_argument('--dataset',choices=('imagenet','things'),default='imagenet');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    if a.action=='prepare':prepare()
    elif a.action=='grey':grey_calibration()
    else:extract(a.dataset,'dev' if a.action in ('smoke','old') else a.action,smoke=a.action=='smoke',old=a.action=='old')


if __name__=='__main__':main()
