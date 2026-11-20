"""Fold-specific FC replay; one outer-fold representation space at a time."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.hebb_fc_full_20261103 import OUT, PILOT, atomic_json
from analysis.report_hebb_fc_native_pilot_20261102 import fast_single
from predify_experiments import native_categories_20261029 as native
from predify_experiments.readout_init_20261012 import sha

METHODS=('mean_std','inhibition_only','scale_both')

def batched(x,model,method,layer):
    scale=model['scale'];E=model['matrices']['corr_'+('k128' if layer=='fc1' else 'k32')]
    q=x/scale;r=q.clone()
    for _ in range(5):
        mean=r.mean(1,keepdim=True)
        drive=q if method=='inhibition_only' else q+.75*(r@E.T)
        theta=.25*drive.mean(1,keepdim=True)+.1*drive.std(1,keepdim=True,unbiased=False)
        target=torch.where(drive>theta,drive,0.)/(1+2*mean)
        r=torch.where(target>0,.5*r+.5*target,0.)
    return r*scale

def matched(x,y):
    return x*(y.mean(1)/x.mean(1).clamp_min(1e-8))[:,None]

def replay(dataset,fold,batch_size=64):
    native.verify_freeze()
    assert (OUT/dataset/'extraction_complete.json').exists(), 'Predify extraction is incomplete'
    m=json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'));trials=m['trials'];n=len(trials)
    assert len({t['dataset_index'] for t in trials})==n
    row={t['dataset_index']:j for j,t in enumerate(trials)}
    graph=OUT/dataset/'graphs'/f'fold{fold}.npz';info=json.loads(graph.with_suffix('.json').read_text(encoding='utf8'))
    assert sha(graph)==info['sha256'] and all(trials[i]['fold']!=fold for i in info['training_indices'])
    dest=OUT/dataset/'folds'/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'replay_complete.json').exists():
        print('REPLAY ALREADY COMPLETE',dataset,fold,flush=True);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    with np.load(graph) as z:
        m1=dict(scale=torch.tensor(float(z['scale1']),device='cuda'),matrices={'corr_k128':torch.tensor(z['E1'],device='cuda')})
        m2=dict(scale=torch.tensor(float(z['scale2']),device='cuda'),matrices={'corr_k32':torch.tensor(z['E2'],device='cuda')})
    state=torch.load(Path.home()/'.cache/torch/hub/checkpoints/vgg16-397923af.pth',map_location='cpu',weights_only=True)
    w=state['classifier.3.weight'].cuda();b=state['classifier.3.bias'].cuda();del state
    def forward(x):
        y1=batched(x,m1,'mean_std','fc1');i1=batched(x,m1,'inhibition_only','fc1')
        entry=torch.relu(torch.nn.functional.linear(y1,w,b));ientry=torch.relu(torch.nn.functional.linear(i1,w,b))
        y2=batched(entry,m2,'mean_std','fc2');i2=batched(ientry,m2,'inhibition_only','fc2')
        s1=matched(x,y1);sentry=torch.relu(torch.nn.functional.linear(s1,w,b));s2=matched(sentry,y2)
        return {'mean_std':(y1,y2),'inhibition_only':(i1,i2),'scale_both':(s1,s2)}
    # Cached fold arrays are incomplete until replay_complete.json is written.
    shape=(12,4,n,15,4096)
    features={method:np.lib.format.open_memmap(dest/f'{method}_fc2.npy',mode='w+',dtype='float32',shape=shape) for method in METHODS}
    test=np.array([j for j,t in enumerate(trials) if t['fold']==fold]);testrow={j:i for i,j in enumerate(test)}
    geometry={method:np.lib.format.open_memmap(dest/f'{method}_test_fc1.npy',mode='w+',dtype='float32',shape=(12,4,len(test),15,4096)) for method in METHODS}
    refs={method:np.lib.format.open_memmap(dest/f'{method}_reference_fc2.npy',mode='w+',dtype='float32',shape=(n,15,4096)) for method in METHODS}
    maxerr={};checked=False;started=time.monotonic()
    def process(a):
        nonlocal checked
        flat=np.asarray(a,dtype='float32').reshape(-1,4096);outputs={k:([],[]) for k in METHODS}
        for start in range(0,len(flat),batch_size):
            x=torch.tensor(flat[start:start+batch_size],device='cuda')
            result=forward(x)
            if not checked and len(x)>1:
                # Numerical gate sensitivity is measured, never assumed bit-identical.
                checks={}
                for j in (0,min(1,len(x)-1),min(14,len(x)-1)):
                    q=x[j:j+1]
                    for method in ('mean_std','inhibition_only'):
                        p=fast_single(q,m1,method,'fc1');entry=torch.relu(torch.nn.functional.linear(p,w,b));v=fast_single(entry,m2,method,'fc2')
                        for layer,actual,expected in [('fc1',result[method][0][j:j+1],p),('fc2',result[method][1][j:j+1],v)]:
                            key=method+'_'+layer;err=float((actual-expected).abs().max());checks[key]=max(checks.get(key,0.),err)
                maxerr.update(checks);assert max(checks.values())<.001,checks
                checked=True
            for method,(a1,a2) in result.items():
                outputs[method][0].append(a1.cpu().numpy());outputs[method][1].append(a2.cpu().numpy())
        return {k:(np.concatenate(v[0]).reshape(*a.shape),np.concatenate(v[1]).reshape(*a.shape)) for k,v in outputs.items()}
    with torch.no_grad():
        for j,t in enumerate(trials):
            with np.load(OUT/dataset/'reference'/f"{t['dataset_index']}.npz") as z:a=z['fc1']
            for method,(_,v) in process(a).items():refs[method][j]=v
        for ci,c in enumerate(native.conditions()):
            for si,seed in enumerate(native.NOISES):
                for concept in sorted(m['concept_folds']):
                    with np.load(OUT/dataset/'base'/f"concept_{concept}_seed_{seed}_{c['name']}.npz") as z:
                        a=z['fc1'];ids=z['images']
                    rows=np.array([row[int(i)] for i in ids]);results=process(a)
                    for method,(f1,f2) in results.items():
                        features[method][ci,si,rows]=f2
                        for p,r in enumerate(rows):
                            if r in testrow:geometry[method][ci,si,testrow[r]]=f1[p]
                print('REPLAY',dataset,'fold',fold,'condition',c['name'],'seed',seed,'elapsed',round(time.monotonic()-started,1),flush=True)
    for group in (features,geometry,refs):
        for a in group.values():a.flush()
    assert checked
    atomic_json(dest/'replay_complete.json',{'dataset':dataset,'fold':fold,'batch_size':batch_size,'single_image_comparison_max_abs':maxerr,
        'comparison_scope':'three reference timepoints, each rule and both layers; batched output is not claimed bit-identical',
        'graph_sha256':sha(graph),'test_rows':test.tolist(),'shape':shape,'methods':list(METHODS),'seconds':time.monotonic()-started})
    print('REPLAY COMPLETE',dataset,fold,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=('imagenet','things'));p.add_argument('--fold',type=int,required=True,choices=range(5));p.add_argument('--batch',type=int,default=64);a=p.parse_args();replay(a.dataset,a.fold,a.batch)
