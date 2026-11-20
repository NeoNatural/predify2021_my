"""Audit persisted stimuli, CV outputs, counts, hashes and representation tables."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.filtered_cv_20261027 import OUT,DOC,NOISES,LAYERS,IDENTITIES,conditions,verify_freeze
from predify_experiments.readout_init_20261012 import sha
from predify_experiments.supplementary import load
from analysis.filtered_cv_20261027 import write
import csv

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    verify_freeze();report=dict(passed=True,datasets={});total=0;totalfits=0
    software=json.loads((OUT/'software_details.json').read_text(encoding='utf8'))
    for path,expected in software['predify_sources'].items():assert sha(path)==expected
    from PIL import Image
    from torchvision.models import VGG16_Weights
    transform=VGG16_Weights.IMAGENET1K_V1.transforms()
    for dataset in ('imagenet','things'):
        base=OUT/dataset;formal=base/'formal';m=json.loads((base/'manifest.json').read_text(encoding='utf8'))
        stimuli={};contrast=[];source_stats={}
        for index,r in m['images'].items():
            assert sha(r['path'])==r['sha256']
            with Image.open(r['path']) as im:x=transform(im.convert('RGB')).numpy()
            source_stats[int(index)]=(x.mean((1,2)),x.std((1,2)))
        for trial in m['trials']:
            index=trial['dataset_index']
            with np.load(formal/f'stimuli_{index}.npz') as z:
                stimuli[index]={seed:__import__('hashlib').sha256(z[f'target_{seed}'].tobytes()).hexdigest() for seed in NOISES}
                assert len(z.files)==12
                for identity in IDENTITIES:
                    ids=list(trial['previews']);ids[trial['slot']]=trial['primes'][identity]
                    for stage in range(2):
                        canvas=z[f'{identity}_preview{stage}'];assert np.isfinite(canvas).all()
                        for side,x0 in enumerate((5,155)):
                            tile=canvas[:,80:144,x0:x0+64];source_mean,source_sd=source_stats[ids[2*stage+side]]
                            contrast.append(dict(dataset=dataset,target=index,identity=identity,stage=stage,side=side,image=ids[2*stage+side],
                                channel_mean=json.dumps(tile.mean((1,2)).tolist()),channel_sd=json.dumps(tile.std((1,2)).tolist()),mean_channel_sd=float(tile.std((1,2)).mean()),
                                source_224_channel_mean=json.dumps(source_mean.tolist()),source_224_channel_sd=json.dumps(source_sd.tolist()),
                                contrast_only_channel_sd=json.dumps((.5*source_sd).tolist()),final_vs_source_sd_ratio=json.dumps(np.divide(tile.std((1,2)),source_sd,out=np.full(3,np.nan),where=source_sd>1e-12).tolist())))
        count=0;mins={n:float('inf') for n in LAYERS};maxdiff={n:0. for n in LAYERS};geometry_index=[];preview_weights={}
        for path in formal.glob('concept_*.pckl'):
            d=load(path);assert d['source_sha']==sha(ROOT/'predify_experiments/filtered_cv_20261027.py')
            assert (formal/d['geometry_file']).exists()
            assert len(d['trials'])==2
            geometry_index.append(dict(concept=d['concept'],seed=d['seed'],condition=d['condition']['name'],n=2,geometry_file=d['geometry_file'],
                conv4_dimension=401408,conv5_dimension=100352,fc1_dimension=4096,fc2_dimension=4096))
            for trial,r in zip(d['trials'],d['results']):
                assert r['target_hash']==stimuli[trial['dataset_index']][d['seed']]
                assert r['target_W_frozen'] and r['fc2'].shape==(15,4096) and r['fc2'].dtype==np.float32
                assert np.isfinite(r['fc2']).all() and np.isfinite(r['activation']).all()
                expected=[(f'preview_{s}',i) for s in range(2) for i in (5,4)] if d['condition']['carrier']!='pcoder' else []
                assert r['commits']==expected
                key=trial['dataset_index'],d['condition']['name']
                if key in preview_weights:assert preview_weights[key]==r['boundary_norms'],'Preview memory depends on target noise'
                preview_weights[key]=r['boundary_norms']
                for li,name in enumerate(LAYERS):
                    mins[name]=min(mins[name],r['output_min'][name]);maxdiff[name]=max(maxdiff[name],float(np.max(np.abs(r['activation'][:,li,0]-r['legacy_activation'][:,li,0]))))
                count+=1
        assert count==3072;total+=count
        models=list((base/'cv').rglob('*.pckl'));assert len(models)==1810;totalfits+=len(models)
        capmax=0;iters=0
        for p in models:
            d=load(p);assert d['attempts'][-1]['converged'];assert not set(d['train_rows'])&set(d['test_rows'])
            assert d['source_sha']==sha(ROOT/'analysis/filtered_cv_20261027.py')
            repeats=1 if 'clean' in p.parts else 3
            assert abs(d['sample_weight_sum']-len(d['train_rows'])/repeats)<1e-8
            capmax=max(capmax,d['attempts'][-1]['cap']);iters=max(iters,d['attempts'][-1]['n_iter'])
        for path in (base/'cv').rglob('*.npz'):
            with np.load(path) as z:
                assert np.isfinite(z['scores']).all();assert set(z['fold'])==set(range(5))
                for f in range(5):assert set(z['y'][z['fold']==f])==set(range(4))
        metadata=[]
        for noise,seeds in [('clean',[0]),('noisy',list(NOISES[1:]))]:
            for sn,seed in enumerate(seeds):
                for j,t in enumerate(m['trials']):metadata.append(dict(dataset=dataset,noise=noise,row=sn*64+j,image=t['dataset_index'],concept=t['concept'],category=t['category_index'],fold=t['fold'],seed=seed))
        write(DOC/dataset/'prediction_row_metadata.csv',metadata)
        boot=np.load(base/'bootstrap_weights.npz');assert boot['weights'].shape==(10000,32)
        for c in range(4):assert np.all(boot['weights'][:,boot['categories']==c].sum(1)==8)
        assert len(list((formal).glob('geometry_*.npz')))==1536
        for name,expected in [('cv_summary.csv',720),('cv_fold_metrics.csv',3600),('geometry_timeline.csv',1440),('geometry_fold_diagnostics.csv',14400),('activation_timeline.csv',1440)]:
            assert len(read(DOC/dataset/name))==expected,(dataset,name)
        report['datasets'][dataset]=dict(sequences=count,stimulus_archives=64,geometry_archives=1536,cv_models=len(models),max_solver_iterations=iters,
             max_solver_cap=capmax,output_min=mins,max_output_vs_legacy_mean_difference=maxdiff,all_target_tensor_hashes_match=True,
             target_W_frozen=True,complete_visual_concept_isolation=True,bootstrap_concepts_per_class=8,bootstrap_replicates=10000)
        write(DOC/dataset/'final_preview_contrast.csv',contrast)
        write(DOC/dataset/'geometry_index.csv',geometry_index)
    assert total==6144 and totalfits==3620
    report.update(sequences=total,timepoints=total*15,svm_fits=totalfits,all_frozen_sources_unchanged=True,
        note='Marginal descriptive bootstrap intervals only; no classifier-refit permutation inference or between-carrier comparisons')
    (DOC/'final_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
