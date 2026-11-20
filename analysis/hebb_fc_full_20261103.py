"""Concept-fold Hebb-FC study: reproducible fold graphs and Predify FC1 extraction.

All output is separate from the frozen native and pilot experiments.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
from pathlib import Path
import argparse, json, sys, time
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predify_experiments import native_categories_20261029 as native
from predify_experiments.supplementary import load, digest
from predify_experiments.readout_init_20261012 import sha

OUT = ROOT / 'Hebbian-CNN/Log/hebb_fc_full_20261103'
DOC = ROOT / 'docs/hebb_fc_full_20261103'
PILOT = ROOT / 'Hebbian-CNN/Log/hebb_fc_native_pilot_20261102'

def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf8')
    tmp.replace(path)

def prepare():
    native.verify_freeze()
    details = {}
    for dataset in ('imagenet', 'things'):
        m = json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'))
        trials = m['trials']
        for f in range(5):
            train = {t['concept'] for t in trials if t['fold'] != f}
            test = {t['concept'] for t in trials if t['fold'] == f}
            assert not train & test
            for t in trials:
                for idx in [t['dataset_index'], *t['previews'], *t['primes'].values()]:
                    r = m['images'][str(idx)]
                    assert m['concept_folds'][r['concept']] == t['fold']
        with np.load(PILOT/dataset/'vanilla_training_fc.npz') as z:
            assert list(z['images']) == [t['dataset_index'] for t in trials]
            assert z['fc1'].shape == z['fc2'].shape == (len(trials), 4096)
        details[dataset] = {'targets': len(trials), 'concepts': len(m['concept_folds']),
                            'classes': m['categories'], 'folds': 5,
                            'manifest_sha256': sha(native.OUT/dataset/'manifest.json'),
                            'training_activations_sha256': sha(PILOT/dataset/'vanilla_training_fc.npz')}
    plan = {'datasets': details, 'graph_training': 'For each outer fold, unlabeled clean native formal targets in the other four folds only; no previous FC weights.',
        'graph': 'FC1 corr mutual k128, FC2 corr mutual k32, RMS scaling, gain 0.75; five local updates; mean+std threshold a=.25 b=.1.',
        'controls': ['off', 'inhibition_only', 'mean_matched_scale_both'],
        'stimuli': 'All four preview identities, three carriers, clean plus three fixed noise realizations, 15 target steps, plus one clean no-preview reference per target.',
        'readouts': ['condition-and-time MVPA', 'carrier-and-noise t9, four preview identities equally weighted', 'clean no-preview t9 reference'],
        'isolation': 'Every visual role belongs to the same fold as its target; graph and decoder training exclude the complete test fold. Features for every training and test row use the same outer-fold graph.',
        'status': 'exploratory; historical concept/image reuse documented in native study historical audit; no claim of independent novelty',
        'precision': 'Predify is extracted at single-image execution shape; FC replay will specify and validate its numerical batching against single-image execution.'}
    path = OUT/'design.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf8')) == plan
    else: atomic_json(path, plan)
    print('DESIGN VERIFIED', json.dumps(details), flush=True)

def graphs(dataset):
    import torch
    from predify_experiments.fc_lateral_study_20261028 import connection_model
    from predify_experiments.fc_sparse_lateral_20261028 import mutual_topk
    prepare(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    m = json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'))
    fold = np.array([t['fold'] for t in m['trials']])
    with np.load(PILOT/dataset/'vanilla_training_fc.npz') as z:
        acts = {k: z[k] for k in ('fc1','fc2')}
    for f in range(5):
        dest=OUT/dataset/'graphs'/f'fold{f}.npz'
        meta=dest.with_suffix('.json')
        if meta.exists():
            assert json.loads(meta.read_text(encoding='utf8'))['sha256']==sha(dest)
            continue
        arrays={}; stats={}
        for layer,k in [('fc1',128),('fc2',32)]:
            x=torch.tensor(acts[layer][fold!=f],device='cuda')
            model=connection_model(x); graph,info=mutual_topk(model['matrices']['corr'],k)
            assert torch.equal(graph,graph.T) and graph.diagonal().count_nonzero()==0 and info['degree_max']<=k
            arrays['E1' if layer=='fc1' else 'E2']=graph.cpu().numpy()
            arrays['scale1' if layer=='fc1' else 'scale2']=np.array(float(model['scale']))
            stats[layer]=dict(k=k,scale=float(model['scale']),**info)
            del x,model,graph;torch.cuda.empty_cache()
        native.atomic_npz(dest,**arrays)
        atomic_json(meta,dict(dataset=dataset,fold=f,training_indices=np.flatnonzero(fold!=f).tolist(),
                              test_indices=np.flatnonzero(fold==f).tolist(),statistics=stats,sha256=sha(dest)))
        print('GRAPH COMPLETE',dataset,f,flush=True)

def extract(dataset):
    import torch
    from predify_experiments.sequence import reset_hebbian_state,run_recurrent_sequence
    from predify_experiments.tripartite_20261027 import restore_state,set_init,gray_image
    prepare()
    m=json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'))
    torch,model,layers,rec,_=native.setup_model();rec.close()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    saved=load(native.OUT/'gray_snapshot.pckl')
    common=[{k:v.cuda() for k,v in item.items()} for item in saved['state']]
    grey=gray_image(torch);cache={}
    handles=[layers[1].register_forward_hook(lambda module,inputs,output:cache.__setitem__('fc1',output.detach().clone())),
             layers[0].register_forward_hook(lambda module,inputs,output:cache.__setitem__('fc2',output.detach().clone()))]
    def restore():
        restore_state(model,common);model.input_mem=grey[None]
    def run(target,previews,carrier):
        reset_hebbian_state(layers);model.reset();set_init(model,False);restore()
        for layer in layers[2:]:
            layer.coeff=.05 if carrier in ('hebb','joint') else 0.
            layer.decay=.5;layer.inh_c=4.;layer.sparse_thres=.9
            layer.update_enabled=carrier in ('hebb','joint')
        for stage,preview in enumerate(previews):
            if stage and carrier=='hebb':restore()
            run_recurrent_sequence(model,preview[None],2,hebbian_layers=layers,commit_update=carrier in ('hebb','joint'),preserve_model_state=True)
        if carrier=='hebb':restore()
        before=[l.boost_weight.detach().clone() for l in layers[2:]]
        fc1=[];fc2=[]
        def after(t,logits):
            fc1.append(cache['fc1'].cpu().numpy()[0]);fc2.append(cache['fc2'].cpu().numpy()[0])
        run_recurrent_sequence(model,target[None],15,hebbian_layers=layers,commit_update=False,preserve_model_state=True,after_step=after)
        assert all(torch.equal(a,l.boost_weight) for a,l in zip(before,layers[2:]))
        return np.stack(fc1),np.stack(fc2)
    pairs=defaultdict(list)
    for trial in m['trials']:pairs[trial['concept']].append(trial)
    start=time.monotonic();done=0
    try:
        with torch.no_grad():
            for concept in sorted(pairs):
                trials=pairs[concept];assert len(trials)==4
                for trial in trials:
                    idx=trial['dataset_index'];dest=OUT/dataset/'reference'/f'{idx}.npz'
                    if dest.exists():continue
                    with np.load(native.OUT/dataset/'formal'/f'stimuli_{idx}.npz') as z:
                        target=torch.tensor(z['target_0'],device='cuda')
                    old=load(native.OUT/dataset/'formal'/f'reference_{idx}.pckl')['result']
                    assert digest(target)==old['target_hash']
                    x,y=run(target,[],'reference')
                    assert np.array_equal(y,old['fc2']),('reference mismatch',dataset,idx,float(np.max(np.abs(y-old['fc2']))))
                    native.atomic_npz(dest,fc1=x)
                for seed in native.NOISES:
                    for c in native.conditions():
                        dest=OUT/dataset/'base'/f"concept_{concept}_seed_{seed}_{c['name']}.npz"
                        if dest.exists():continue
                        old=load(native.OUT/dataset/'formal'/f"concept_{concept}_seed_{seed}_{c['name']}.pckl")
                        assert old['trials']==trials
                        xs=[]
                        for trial,previous in zip(trials,old['results']):
                            idx=trial['dataset_index']
                            with np.load(native.OUT/dataset/'formal'/f'stimuli_{idx}.npz') as z:
                                target=torch.tensor(z[f'target_{seed}'],device='cuda')
                                pv=[torch.tensor(z[f"{c['identity']}_preview{s}"],device='cuda') for s in range(2)]
                            assert digest(target)==previous['target_hash']
                            x,y=run(target,pv,c['carrier'])
                            assert np.array_equal(y,previous['fc2']),('baseline mismatch',dataset,idx,seed,c['name'],float(np.max(np.abs(y-previous['fc2']))))
                            xs.append(x)
                        native.atomic_npz(dest,fc1=np.stack(xs),images=np.array([t['dataset_index'] for t in trials]))
                        done+=4
                        print('EXTRACT',dataset,concept,seed,c['name'],'new',done,'seconds',round(time.monotonic()-start,1),flush=True)
    finally:
        for h in handles:h.remove()
    atomic_json(OUT/dataset/'extraction_complete.json',dict(dataset=dataset,concepts=len(pairs),sequences=len(m['trials'])*48,references=len(m['trials']),baseline_fc2_bit_identical=True))
    print('EXTRACTION COMPLETE',dataset,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','graphs','extract'))
    p.add_argument('--dataset',choices=('imagenet','things'));a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        for d in ([a.dataset] if a.dataset else ['imagenet','things']):globals()[a.action](d)
