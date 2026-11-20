"""Paired concept-level activation contrasts after completed fold-specific replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from pathlib import Path
import argparse,csv,json,sys,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.hebb_fc_full_20261103 import OUT as EXP,atomic_json
from analysis.hebb_fc_full_replay_20261103 import batched,matched
from predify_experiments import native_categories_20261029 as native
from predify_experiments.readout_init_20261012 import sha

OUT=ROOT/'Hebbian-CNN/Log/preview_suppression_comparison_20261103'
DOC=ROOT/'docs/preview_suppression_comparison_20261103'
METHODS=('off','mean_std','inhibition_only','scale_both')
LAYERS=('fc1','fc2')
CONDITIONS=native.conditions()
IDENTITIES=('exact','same_concept','same_category')
BOOT_SEED=20261103;N_BOOT=10000

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def extract(dataset):
    import torch
    native.verify_freeze();marker=EXP/dataset/'extraction_complete.json';assert marker.exists()
    m=json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'));trials=m['trials'];n=len(trials)
    ix={t['dataset_index']:j for j,t in enumerate(trials)};fold=np.array([t['fold'] for t in trials]);pairs={}
    for t in trials:pairs.setdefault(t['concept'],[]).append(t)
    assert all(len(v)==4 and len({q['fold'] for q in v})==1 for v in pairs.values())
    dest=OUT/dataset;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        meta=json.loads((dest/'complete.json').read_text(encoding='utf8'));assert sha(dest/'activations.npz')==meta['sha256'];return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    state=torch.load(Path.home()/'.cache/torch/hub/checkpoints/vgg16-397923af.pth',map_location='cpu',weights_only=True)
    w=state['classifier.3.weight'].cuda();b=state['classifier.3.bias'].cuda();del state
    source2=np.load(native.OUT/dataset/'features.npy',mmap_mode='r')
    # (method,layer,condition,seed,image,time); FP64 reduction of original FP32 vectors.
    result=np.full((4,2,12,4,n,15),np.nan,dtype=np.float64);start=time.monotonic()
    with torch.no_grad():
        for f in range(5):
            graph=EXP/dataset/'graphs'/f'fold{f}.npz';meta=json.loads(graph.with_suffix('.json').read_text(encoding='utf8'))
            assert sha(graph)==meta['sha256'] and all(fold[i]!=f for i in meta['training_indices'])
            with np.load(graph) as z:
                m1={'scale':torch.tensor(float(z['scale1']),device='cuda'),'matrices':{'corr_k128':torch.tensor(z['E1'],device='cuda')}}
                m2={'scale':torch.tensor(float(z['scale2']),device='cuda'),'matrices':{'corr_k32':torch.tensor(z['E2'],device='cuda')}}
            for ci,c in enumerate(CONDITIONS):
                for si,seed in enumerate(native.NOISES):
                    for concept in sorted(k for k,v in pairs.items() if v[0]['fold']==f):
                        with np.load(EXP/dataset/'base'/f"concept_{concept}_seed_{seed}_{c['name']}.npz") as z:
                            ids=z['images'];v=z['fc1'];assert v.shape==(4,15,4096)
                        rows=np.array([ix[int(i)] for i in ids]);assert all(fold[j]==f for j in rows)
                        x=torch.tensor(v.reshape(-1,4096),device='cuda');base2=np.asarray(source2[ci,si,rows])
                        y=batched(x,m1,'mean_std','fc1');i=batched(x,m1,'inhibition_only','fc1');s=matched(x,y)
                        bridge=lambda a:torch.relu(torch.nn.functional.linear(a,w,b))
                        q=batched(bridge(y),m2,'mean_std','fc2');iq=batched(bridge(i),m2,'inhibition_only','fc2');sq=matched(bridge(s),q)
                        def means(a):return a.double().mean(1).cpu().numpy().reshape(4,15)
                        for j,(a1,a2) in enumerate([(x,None),(y,q),(i,iq),(s,sq)]):
                            result[j,0,ci,si,rows]=means(a1)
                            result[j,1,ci,si,rows]=base2.astype(np.float64).mean(axis=-1) if a2 is None else means(a2)
                print('ACTIVATION',dataset,'fold',f,c['name'],'elapsed',round(time.monotonic()-start,1),flush=True)
            del m1,m2;torch.cuda.empty_cache()
    assert np.isfinite(result).all()
    # Every saved mean must reproduce the original held-out fold aggregate.
    with (ROOT/'docs/hebb_fc_full_20261103/fold_geometry_timeline.csv').open(encoding='utf-8-sig',newline='') as fp:
        r=[v for v in csv.DictReader(fp) if v['dataset']==dataset]
    maxerr=0.
    for v in r:
        f=int(v['fold']);mi=METHODS.index(v['method']);li=LAYERS.index(v['layer']);ci=next(j for j,c in enumerate(CONDITIONS) if c['name']==v['condition']);si=native.NOISES.index(int(v['seed']));t=int(v['timestep'])
        err=abs(float(result[mi,li,ci,si,fold==f,t].mean())-float(v['activation']));maxerr=max(maxerr,err)
    assert maxerr<1e-8,(dataset,maxerr)
    native.atomic_npz(dest/'activations.npz',activation=result,images=np.array([t['dataset_index'] for t in trials]))
    atomic_json(dest/'complete.json',{'dataset':dataset,'sha256':sha(dest/'activations.npz'),'full_study_aggregate_max_abs_error':maxerr,
       'fold_graphs_verified':True,'n_images':n,'concepts':len(pairs),'elapsed_seconds':time.monotonic()-start})
    print('ACTIVATION VERIFIED',dataset,'max error',maxerr,flush=True)

def analyze(dataset):
    m=json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'));trials=m['trials'];n=len(trials)
    meta=json.loads((OUT/dataset/'complete.json').read_text(encoding='utf8'));assert sha(OUT/dataset/'activations.npz')==meta['sha256']
    with np.load(OUT/dataset/'activations.npz') as z:
        act=z['activation'];assert np.array_equal(z['images'],[t['dataset_index'] for t in trials])
    concepts=sorted({t['concept'] for t in trials});group=[];strata=[]
    for concept in concepts:
        ii=np.array([j for j,t in enumerate(trials) if t['concept']==concept]);assert len(ii)==4
        assert len({(trials[j]['category_index'],trials[j]['fold']) for j in ii})==1
        group.append(ii);strata.append((trials[ii[0]]['category_index'],trials[ii[0]]['fold']))
    strata=sorted(set(strata));group_index=[]
    for cat,f in strata:
        a=np.array([i for i,ids in enumerate(group) if trials[ids[0]]['category_index']==cat and trials[ids[0]]['fold']==f]);assert len(a)==4;group_index.append(a)
    assert len(strata)==len(m['categories'])*5 and len(group)==4*len(strata)
    # Bootstrap multiplicities: stratified by category and fixed outer fold;
    # same resampled concepts are used for all methods, seeds and timepoints.
    rng=np.random.default_rng(BOOT_SEED);weights=np.zeros((N_BOOT,len(group)),dtype=np.float64)
    rr=np.arange(N_BOOT)[:,None]
    for a in group_index:
        draws=rng.choice(a,size=(N_BOOT,len(a)),replace=True)
        np.add.at(weights,(np.broadcast_to(rr,draws.shape),draws),1.)
    weights/=len(group);assert np.allclose(weights.sum(1),1.)
    intervals=[];timelines=[];vectors=[];keys=[]
    for noise,si in [('clean',[0]),('noisy',[1,2,3])]:
        for carrier in native.CARRIERS:
            base=next(j for j,c in enumerate(CONDITIONS) if c['name']==carrier+'_different_category')
            for layer in LAYERS:
                li=LAYERS.index(layer)
                for ident in IDENTITIES:
                    ci=next(j for j,c in enumerate(CONDITIONS) if c['name']==carrier+'_'+ident)
                    for mi,method in enumerate(METHODS):
                        # Seed x image x time; paired image and seed comparisons.
                        A=act[mi,li,ci,si]-act[mi,li,base,si]
                        entries=[('priming_minus_different',A)]
                        if mi:
                            # Difference of paired preview effects (active - Off).
                            entries.append(('interaction_vs_off',A-(act[0,li,ci,si]-act[0,li,base,si])))
                        for contrast,data in entries:
                            per_concept=np.stack([data[:,ii,:].mean(axis=(0,1)) for ii in group])
                            assert per_concept.shape==(len(group),15)
                            for t in range(15):
                                timelines.append(dict(dataset=dataset,noise=noise,carrier=carrier,layer=layer,identity=ident,method=method,contrast=contrast,timestep=t,value=float(per_concept[:,t].mean())))
                            for window,tt in [('early',range(5)),('late',range(10,15))]:
                                vectors.append(per_concept[:,tt].mean(1));keys.append(dict(dataset=dataset,noise=noise,carrier=carrier,layer=layer,identity=ident,method=method,contrast=contrast,window=window))
    matrix=np.stack(vectors,axis=1);assert matrix.shape[0]==len(group)
    for start in range(0,matrix.shape[1],36):
        q=matrix[:,start:start+36];boot=weights@q;lo,hi=np.quantile(boot,[.025,.975],axis=0)
        for j in range(q.shape[1]):intervals.append(dict(**keys[start+j],value=float(q[:,j].mean()),ci_low=float(lo[j]),ci_high=float(hi[j]),n_concepts=len(group),bootstrap=N_BOOT))
    write(DOC/dataset/'timecourse.csv',timelines);write(DOC/dataset/'windows.csv',intervals)
    atomic_json(DOC/dataset/'analysis_audit.json',{'passed':True,'source_activation_sha256':meta['sha256'],'concepts':len(group),'strata':len(strata),'concepts_per_stratum':4,
       'bootstrap_seed':BOOT_SEED,'replicates':N_BOOT,'paired_across_conditions':True,'fixed_outer_folds_and_fixed_graphs':True,
       'time_rows':len(timelines),'window_rows':len(intervals)})
    print('ANALYSIS COMPLETE',dataset,'timeline',len(timelines),'windows',len(intervals),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('extract','analyze'));p.add_argument('--dataset',choices=('imagenet','things'));a=p.parse_args()
    for d in ([a.dataset] if a.dataset else ['imagenet','things']):globals()[a.action](d)
