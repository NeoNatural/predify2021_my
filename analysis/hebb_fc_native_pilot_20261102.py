"""Fresh FC graph fit and small integration pilot on current native-category data."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from pathlib import Path
import argparse,hashlib,json,time,sys
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments import native_categories_20261029 as native
from predify_experiments.supplementary import load,save,digest
from predify_experiments.readout_init_20261012 import sha,immutable_json

OUT=ROOT/'Hebbian-CNN/Log/hebb_fc_native_pilot_20261102'
DOC=ROOT/'docs/hebb_fc_native_pilot_20261102'
STEPS=5
RULES={
 'quota':dict(family='quota'),
 'mean':dict(family='moments',a=.25,b=0.,theta0=0.),
 'mean_std':dict(family='moments',a=.25,b=.1,theta0=0.),
 'feedback':dict(family='feedback',g=.25,rho=1.,theta0=0.),
 'inhibition_only':dict(family='moments',a=.25,b=.1,theta0=0.),
}
IDENTITIES=('exact','same_concept','same_category','different_category')

def prepare():
    native.verify_freeze();OUT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    result={}
    for dataset in ('imagenet','things'):
        source=native.OUT/dataset/'manifest.json';m=json.loads(source.read_text(encoding='utf8'))
        train=m['trials'];dev=m['development'];train_concepts={t['concept'] for t in train}
        train_hashes={m['images'][str(t['dataset_index'])]['sha256'] for t in train}
        assert not train_concepts&{t['concept'] for t in dev}
        selected=[]
        for k,c in enumerate(m['categories']):
            candidates=sorted({t['concept'] for t in dev if t['category_index']==k})
            assert candidates
            concept=candidates[0]
            chosen=sorted([t for t in dev if t['concept']==concept],key=lambda t:t['dataset_index'])[:2]
            assert len(chosen)==2;selected.extend(chosen)
        roles=[]
        for t in selected:
            ids=[t['dataset_index'],*t['previews'],*t['primes'].values()]
            for i in ids:
                r=m['images'][str(i)];assert r['concept'] not in train_concepts and r['sha256'] not in train_hashes
                roles.append(dict(target=t['dataset_index'],image=i,concept=r['concept'],sha256=r['sha256']))
        assert len({t['dataset_index'] for t in selected})==2*len(m['categories'])
        for t in selected:
            assert m['images'][str(t['dataset_index'])]['sha256'] not in train_hashes
        spec=dict(dataset=dataset,categories=m['categories'],current_manifest_sha256=sha(source),
          training=dict(source='current study formal clean target canvas; labels not used in FC graph estimation',images=[t['dataset_index'] for t in train],concepts=sorted(train_concepts),count=len(train)),
          evaluation=dict(source='current study development subset; first concept alphabetically and two lowest image indices per category; no outcome selection',trials=selected,visual_roles=roles),
          overlap=dict(train_vs_test_concepts=0,train_targets_vs_test_visual_bytes=0),
          prior_study_use='algorithm and hyperparameter constants only; no previous split, graph weights, or previous FC experiment data',
          readout='current native experiment clean-reference fold 0 head; frozen and fitted on current formal training concepts; descriptive transfer only',
          rule_configs=RULES,fc_steps=STEPS,gain=.75,fc1_graph='corr_k128',fc2_graph='corr_k32',
          noise_seeds=[0,2921],carriers=list(native.CARRIERS),identities=list(IDENTITIES),
          missing_same_category='THINGS plant development contains one concept; that control is omitted for its 2 targets',
          timing='2+2 preview, target t0-t14; every network step performs five FC1 and five FC2 side updates; FC state resets each network step; learned E matrices fixed',
          analysis='pilot, no fitting or hyperparameter selection on development outcomes; pair target/noise/carrier/identity across FC settings; no inferential claims')
        (OUT/dataset).mkdir(exist_ok=True)
        immutable_json(OUT/dataset/'pilot_manifest.json',spec)
        result[dataset]=dict(categories=len(m['categories']),train=len(train),test=len(selected),test_concepts=len({t['concept'] for t in selected}),visual_roles=len(roles))
    print('DESIGN FROZEN',json.dumps(result),flush=True)

def train(dataset):
    import torch
    from torchvision.models import vgg16,VGG16_Weights
    from predify_experiments.fc_lateral_study_20261028 import connection_model
    from predify_experiments.fc_sparse_lateral_20261028 import mutual_topk
    native.verify_freeze();spec=json.loads((OUT/dataset/'pilot_manifest.json').read_text(encoding='utf8'));dest=OUT/dataset
    marker=dest/'graph_training.json'
    if marker.exists():
        info=json.loads(marker.read_text(encoding='utf8'));assert info['pilot_manifest_sha256']==sha(dest/'pilot_manifest.json')
        for name,value in info['files'].items():assert sha(dest/name)==value
        print('GRAPHS ALREADY TRAINED',dataset,flush=True);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.manual_seed(native.SEED)
    device=torch.device('cuda');model=vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval().to(device);cache={}
    handles=[model.classifier[i].register_forward_hook(lambda m,inp,out,k=k:cache.__setitem__(k,out.detach().clone())) for i,k in [(1,'fc1'),(4,'fc2')]]
    x1=[];x2=[];hashes={};ids=spec['training']['images'];start=time.monotonic()
    with torch.inference_mode():
        for off in range(0,len(ids),12):
            group=ids[off:off+12];images=[]
            for index in group:
                path=native.OUT/dataset/'formal'/f'stimuli_{index}.npz';ref=load(native.OUT/dataset/'formal'/f'reference_{index}.pckl')
                with np.load(path) as z:array=z['target_0']
                h=hashlib.sha256(array.tobytes()).hexdigest();assert h==ref['result']['target_hash'];hashes[str(index)]=h;images.append(array)
            model(torch.from_numpy(np.stack(images)).to(device));x1.append(cache['fc1'].cpu());x2.append(cache['fc2'].cpu())
            if off%120==0:print('GRAPH TRAINING INPUT',dataset,min(off+12,len(ids)),'/',len(ids),flush=True)
    for h in handles:h.remove()
    del model;torch.cuda.empty_cache()
    activations={'fc1':torch.cat(x1).to(device),'fc2':torch.cat(x2).to(device)}
    assert all(v.shape==(len(ids),4096) and torch.isfinite(v).all() for v in activations.values())
    arrays={};stats={}
    for layer,k in [('fc1',128),('fc2',32)]:
        model=connection_model(activations[layer]);graph,info=mutual_topk(model['matrices']['corr'],k)
        assert torch.equal(graph,graph.T) and graph.diagonal().count_nonzero()==0 and graph.min()>=0
        arrays['E1' if layer=='fc1' else 'E2']=graph.cpu().numpy();arrays['scale1' if layer=='fc1' else 'scale2']=np.array(float(model['scale']))
        stats[layer]=dict(k=k,**info,scale=float(model['scale']),variance_floor=model['correlation_variance_floor'])
        del model,graph;torch.cuda.empty_cache()
    native.atomic_npz(dest/'graphs.npz',**arrays)
    native.atomic_npz(dest/'vanilla_training_fc.npz',fc1=activations['fc1'].cpu().numpy(),fc2=activations['fc2'].cpu().numpy(),images=np.array(ids))
    info=dict(dataset=dataset,pilot_manifest_sha256=sha(dest/'pilot_manifest.json'),training_target_hashes=hashes,statistics=stats,
              source='new unlabeled corr graph from current clean target canvases; no graph from prior FC study',
              files={f:sha(dest/f) for f in ('graphs.npz','vanilla_training_fc.npz')},seconds=time.monotonic()-start)
    immutable_json(marker,info);print('GRAPHS TRAINED',dataset,stats,flush=True)

def test_inputs(dataset,trial,transform,filtered,torch):
    from PIL import Image
    m=json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'));index=trial['dataset_index']
    def image(i):
        r=m['images'][str(i)];assert sha(r['path'])==r['sha256']
        with Image.open(r['path']) as im:return transform(im.convert('RGB')).cuda()
    original=image(index);targets={0:filtered.canvas([(original,'center')])}
    g=torch.Generator(device='cuda').manual_seed((native.SEED+2921*1000003+index*1009)%2147483647)
    targets[2921]=filtered.canvas([(original+torch.randn(original.shape,device='cuda',generator=g)*.5,'center')])
    allimages={i:image(i) for i in set(trial['previews'])|set(trial['primes'].values())}
    previews={}
    identities=[i for i in IDENTITIES if i!='same_category' or trial['development_same_category_available']]
    for ident in identities:
        a=list(trial['previews']);a[trial['slot']]=trial['primes'][ident]
        previews[ident]=[filtered.canvas([(allimages[i],side) for i,side in zip(a[2*j:2*j+2],('left','right'))],True) for j in range(2)]
    return targets,previews

def apply_fc(x,model,rule,layer):
    from predify_experiments.fc_population_rules_20261028 import trajectories
    gain=0. if rule=='inhibition_only' else .75
    return next(trajectories(x,model,RULES[rule],layer,gain=gain,steps=(STEPS,)))[1]

def extract(dataset):
    import torch
    from torchvision.models import VGG16_Weights
    from predify_experiments.sequence import reset_hebbian_state,run_recurrent_sequence
    from predify_experiments.tripartite_20261027 import restore_state,set_init,gray_image
    spec=json.loads((OUT/dataset/'pilot_manifest.json').read_text(encoding='utf8'));dest=OUT/dataset
    graphinfo=json.loads((dest/'graph_training.json').read_text(encoding='utf8'))
    assert graphinfo['pilot_manifest_sha256']==sha(dest/'pilot_manifest.json') and graphinfo['files']['graphs.npz']==sha(dest/'graphs.npz')
    native.verify_freeze();torch,model,layers,rec,_=native.setup_model();rec.close()
    # Match the current Predify experiment's cuDNN default; FC matmul remains FP32.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    with np.load(dest/'graphs.npz') as z:
        m1=dict(scale=torch.tensor(float(z['scale1']),device='cuda'),matrices={'corr_k128':torch.tensor(z['E1'],device='cuda')})
        m2=dict(scale=torch.tensor(float(z['scale2']),device='cuda'),matrices={'corr_k32':torch.tensor(z['E2'],device='cuda')})
    saved=load(native.OUT/'gray_snapshot.pckl');common=[{k:v.cuda() for k,v in item.items()} for item in saved['state']];grey=gray_image(torch)
    filt=native.FilteredInput(torch);transform=VGG16_Weights.IMAGENET1K_V1.transforms();cache={};phase=['preview'];mode=['off']
    def fc_hook(layer):
        def hook(module,inputs,output):
            value=output
            if mode[0]!='off':value=apply_fc(output,m1 if layer=='fc1' else m2,mode[0],layer)
            cache[layer]=value.detach().clone()
            return value
        return hook
    hooks=[layers[1].register_forward_hook(fc_hook('fc1')),layers[0].register_forward_hook(fc_hook('fc2'))]
    def restore():restore_state(model,common);model.input_mem=grey[None]
    def run(target,previews,carrier,fc_rule):
        mode[0]=fc_rule;reset_hebbian_state(layers);model.reset();set_init(model,False);restore()
        for layer in layers[2:]:
            layer.coeff=.05 if carrier!='pcoder' else 0.;layer.decay=.5;layer.inh_c=4.;layer.sparse_thres=.9;layer.update_enabled=carrier in ('hebb','joint')
        phase[0]='preview'
        for stage,preview in enumerate(previews):
            if stage and carrier=='hebb':restore()
            run_recurrent_sequence(model,preview[None],2,hebbian_layers=layers,commit_update=carrier in ('hebb','joint'),preserve_model_state=True)
        if carrier=='hebb':restore()
        before=[l.boost_weight.detach().clone() for l in layers[2:]];fc1=[];fc2=[];pred=[];reps=[]
        phase[0]='target'
        def after(t,logits):
            fc1.append(cache['fc1'].cpu().numpy()[0]);fc2.append(cache['fc2'].cpu().numpy()[0]);pred.append(logits.cpu().numpy()[0])
            reps.append([digest(getattr(model,f'pcoder{k}').rep) for k in (4,5)])
        run_recurrent_sequence(model,target[None],15,hebbian_layers=layers,commit_update=False,preserve_model_state=True,after_step=after)
        assert all(torch.equal(a,l.boost_weight) for a,l in zip(before,layers[2:]))
        return dict(fc1=np.stack(fc1),fc2=np.stack(fc2),logits=np.stack(pred),pcoder_hashes=reps,
                    target_hash=digest(target),W_hashes=[digest(x) for x in before])
    try:
        # PCoders internally enable autograd to compute prediction-error gradients.
        with torch.no_grad():
            # Validate the true classifier hook before collecting the full pilot.
            trial=spec['evaluation']['trials'][0];targets,previews=test_inputs(dataset,trial,transform,filt,torch)
            base_path=dest/'baseline'/f"{trial['dataset_index']}_0_joint_exact.pckl"
            if base_path.exists():base=load(base_path)['values']
            else:
                base=run(targets[0],previews['exact'],'joint','off')
                save(base_path,dict(trial=trial,seed=0,carrier='joint',identity='exact',values=base))
            checks={}
            for rule in RULES:
                online=run(targets[0],previews['exact'],'joint',rule)
                assert online['pcoder_hashes']==base['pcoder_hashes'] and online['W_hashes']==base['W_hashes']
                x=torch.tensor(base['fc1'],device='cuda')
                # Use the same single-image multiplication shape as the live model.
                f1=torch.cat([apply_fc(x[t:t+1],m1,rule,'fc1') for t in range(15)])
                bridge=torch.cat([torch.relu(model.backbone.classifier.pipe[3](f1[t:t+1])) for t in range(15)])
                f2=torch.cat([apply_fc(bridge[t:t+1],m2,rule,'fc2') for t in range(15)])
                diff1=float(np.max(np.abs(online['fc1']-f1.cpu().numpy())));diff2=float(np.max(np.abs(online['fc2']-f2.cpu().numpy())))
                assert np.array_equal(online['fc1'],f1.cpu().numpy()) and np.array_equal(online['fc2'],f2.cpu().numpy()),(rule,diff1,diff2)
                checks[rule]=dict(fc1_max_abs=diff1,fc2_max_abs=diff2,pcoder_and_preview_weights_exact=True)
            immutable_json(dest/'integration_acceptance.json',dict(dataset=dataset,trial=trial['dataset_index'],checks=checks,passed=True,
                interpretation='FC is downstream of all PCoders; matching hashes show no recurrent feedback from classifier to PCoder on this trial; replay uses the identical single-image multiplication shape.'))
            prior=native.OUT/dataset/'smoke'/f"concept_{trial['concept']}_seed_0_joint_exact.pckl"
            if prior.exists():
                old=load(prior);ix=next(i for i,t in enumerate(old['trials']) if t['dataset_index']==trial['dataset_index']);reference=old['results'][ix]
                error=float(np.max(np.abs(reference['fc2']-base['fc2'])));assert reference['target_hash']==base['target_hash'] and error<1e-6,error
                immutable_json(dest/'baseline_reference_check.json',dict(passed=True,target=trial['dataset_index'],max_fc2_abs_error=error,prior_record=str(prior)))
            print('IN-MODEL AND REPLAY ACCEPTED',dataset,checks,flush=True)
            for trial in spec['evaluation']['trials']:
                targets,previews=test_inputs(dataset,trial,transform,filt,torch)
                for seed,target in targets.items():
                    for carrier in native.CARRIERS:
                        for ident,pv in previews.items():
                            path=dest/'baseline'/f"{trial['dataset_index']}_{seed}_{carrier}_{ident}.pckl"
                            if path.exists():continue
                            values=run(target,pv,carrier,'off');save(path,dict(trial=trial,seed=seed,carrier=carrier,identity=ident,values=values))
                            print('PILOT BASE',dataset,trial['dataset_index'],seed,carrier,ident,flush=True)
    finally:
        for h in hooks:h.remove()
    immutable_json(dest/'baseline_complete.json',dict(dataset=dataset,files=len(list((dest/'baseline').glob('*.pckl'))),pilot_manifest_sha256=sha(dest/'pilot_manifest.json'),passed=True))
    print('BASELINE AND INTEGRATION ACCEPTED',dataset,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','train','extract'));p.add_argument('--dataset',choices=('imagenet','things'));a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        for d in ([a.dataset] if a.dataset else ['imagenet','things']):globals()[a.action](d)
