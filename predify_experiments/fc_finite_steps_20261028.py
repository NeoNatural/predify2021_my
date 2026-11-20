"""Fixed-horizon evaluation, including previously nonstationary hard thresholds."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from . import fc_sparse_lateral_20261028 as sparse
from . import fc_lateral_study_20261028 as core
from .fc_lateral_vanilla_20261028 import ROOT,sha,write_json

DATA=sparse.DATA
OUT=ROOT/"Hebbian-CNN/Log/fc_finite_steps_20261028"
DOC=ROOT/"docs/fc_finite_steps_20261028"
STEPS=(1,2,3,5,10,20,50,99,100)
SVM_STEPS=(1,5,20,100)
DEVICE=core.DEVICE
EPS=sparse.EPS


def trajectories(original,model,config,steps=STEPS):
    """Exactly max(steps) updates; residual is recorded but never stops inference."""
    steps=tuple(sorted(set(steps)))
    if config['mode'] in ('identity','scale'):
        r=original if config['mode']=='identity' else original*.5
        for step in steps:
            yield step,r.clone(),dict(residual=0.,support_switch_units_mean=0.,updates_applied=0)
        return
    x=original/model['scale'];r=x.clone()
    e=model['matrices'].get(config['rule']+'_'+config['graph'])
    count=torch.ceil((original>EPS).sum(1)*config['budget']).long()
    for step in range(1,max(steps)+1):
        drive=x+config['gain']*(r@e.T) if config['gain'] else x
        if config['mode']=='divide':target=torch.relu(drive)
        else:target,_=sparse.adaptive_target(drive,count,config['mode'])
        target=target/(1+2*r.mean(1,keepdim=True))
        updated=(1-core.ALPHA)*r+core.ALPHA*target
        if config['mode'] in ('soft','hard'):updated=torch.where(target>0,updated,0.)
        if step in steps:
            residual=float(((updated-r).norm(dim=1)/(x.norm(dim=1)+1e-8)).max())
            switches=float(((updated*model['scale']>EPS)!=(r*model['scale']>EPS)).sum(1).float().mean())
            yield step,updated*model['scale'],dict(residual=residual,support_switch_units_mean=switches,updates_applied=step)
        r=updated


def config_id(configs,**values):
    found=[c['id'] for c in configs if all(c[k]==v for k,v in values.items())]
    assert len(found)==1,values
    return found[0]


def raw_geometry(r,y):
    centers=torch.stack([r[y==c].mean(0) for c in range(5)])
    w=torch.stack([(r[y==c]-centers[c]).square().sum(1).mean() for c in range(5)]).mean()
    b=(centers-centers.mean(0)).square().sum(1).mean()
    return dict(raw_within=float(w),raw_between=float(b),raw_fisher=float(b/w.clamp_min(1e-12)))


def evaluate(r,original,yt):
    m=sparse.measurements(r,original,yt,False)
    m.update(raw_geometry(r,yt))
    m['per_class_activation']=[float(r[yt==c].mean()) for c in range(5)]
    return m


def add_effects(m,base):
    m['effects']=core.effects(m,base)
    m['joint']=bool(m['activation']<base['activation']*(1-1e-6)
        and m['within']<base['within']*(1-1e-6) and m['fisher']>base['fisher']*(1+1e-6)
        and m['zero_fraction']==0)
    m['sparse_joint']=bool(m['joint'] and m['max_support_ratio']<=1.00001)


def freeze():
    protocol=dict(version=1,exploratory_reuse=True,steps=STEPS,svm_steps=SVM_STEPS,configs=sparse.configurations(),
        scope="all 66 fixed configurations, including all 24 Hebbian hard-threshold settings; no new hyperparameter or horizon selection",
        inference="exact fixed updates; no early stop and no stationary/fixed-point eligibility test; nonfinite values are numerical errors, not ordinary nonstationarity",
        main_pairs="parameters from round2 inner-selected soft retained per outer fold; matched hard changes only threshold mode; paired no-Hebb controls use same budget; no re-selection from outer curves",
        readout="same fixed ridge refitted train-only at every configuration/horizon; frozen original SVM throughout; refit C1 SVM for predeclared soft/hard pairs and paired no-Hebb at1,5,20,100; angular ridge for soft/hard diagnostic",
        selection="none in this study; report full fixed-grid outer results descriptively; prior configurations and reused cohort are exploratory, not independent confirmation",
        convergence="record update residual and support switches diagnostically only; no filter or ranking uses them",
        source_sha256=sha(__file__),dependencies={p:sha(ROOT/p) for p in [
            "predify_experiments/fc_sparse_lateral_20261028.py","predify_experiments/fc_lateral_study_20261028.py",
            "predify_experiments/fc_lateral_vanilla_20261028.py"]},
        original_model_sources=sparse.freeze()['original_model_sources'],
        inputs={d:{f:sha(DATA/d/f) for f in ['manifest.json','activations.npz','baseline_models.pckl']} for d in ['imagenet','things']},
        old_selections={f'{d}/{l}/{f}':sha(sparse.OUT/d/l/f'fold{f}'/'selection.json')
            for d in ['imagenet','things'] for l in ['fc1','fc2'] for f in range(5)})
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==json.loads(json.dumps(protocol)),'Frozen source changed'
    else:
        write_json(path,protocol);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return protocol


def run(dataset,layer):
    protocol=freeze()
    torch.set_num_threads(4);torch.manual_seed(core.SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    configs=sparse.configurations()
    with np.load(DATA/dataset/'activations.npz') as z:
        x=torch.as_tensor(z[layer],device=DEVICE);y,groups,folds=z['category'],z['concept'],z['fold']
    yt=torch.as_tensor(y,device=DEVICE)
    with (DATA/dataset/'baseline_models.pckl').open('rb') as f:heads=pickle.load(f)['models'][layer]
    started=time.monotonic()
    for f in range(5):
        dest=OUT/dataset/layer/f'fold{f}'
        if (dest/'complete.json').exists():print('EXISTS',dataset,layer,f,flush=True);continue
        train,test=np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
        assert not set(groups[train])&set(groups[test])
        order=np.concatenate([train,test]);n=len(train)
        model=sparse.connection_model(x[train])
        previous=json.loads((sparse.OUT/dataset/layer/f'fold{f}'/'selection.json').read_text(encoding='utf-8'))
        old=previous['choices']['selected']['config']
        assert old['mode']=='soft'
        soft_id=config_id(configs,rule=old['rule'],graph=old['graph'],mode='soft',gain=old['gain'],budget=old['budget'])
        hard_id=config_id(configs,rule=old['rule'],graph=old['graph'],mode='hard',gain=old['gain'],budget=old['budget'])
        no_soft=config_id(configs,rule='zero',mode='soft',gain=0.,budget=old['budget'])
        no_hard=config_id(configs,rule='zero',mode='hard',gain=0.,budget=old['budget'])
        roles={soft_id:'prior_soft',hard_id:'matched_hard',no_soft:'paired_nohebb_soft',no_hard:'paired_nohebb_hard'}
        histories=[]
        for c in configs:
            if c['mode']!='hard':continue
            rs=[r for r in previous['candidates'] if r['config']['id']==c['id']]
            histories.append(dict(config_id=c['id'],family=c['family'],
                previous_nonconverged_inner_runs=sum(not r['metrics']['converged'] for r in rs)))
        write_json(dest/'design.json',dict(train_rows=train.tolist(),test_rows=test.tolist(),
            roles=roles,prior_soft=old,hard_history=histories))
        baseline=evaluate(x[test],x[test],yt[test])
        baseline_scores=core.ridge_scores(x[train],x[test],y[train],groups[train])
        baseline.update(core.classification(baseline_scores,y[test]))
        baseline['frozen_svm']=core.classification(heads[f].decision_function(x[test].cpu().numpy()),y[test])
        grid=[];pairs=[];saved={};models={};all_scores={};checkpoints={}
        for c in configs:
            role=roles.get(c['id'])
            for step,r,diag in trajectories(x[order],model,c):
                assert torch.isfinite(r).all() and bool((r>=0).all())
                a,b=r[:n],r[n:]
                m=evaluate(b,x[test],yt[test]);scores=core.ridge_scores(a,b,y[train],groups[train])
                m.update(core.classification(scores,y[test]),**diag)
                m['frozen_svm']=core.classification(heads[f].decision_function(b.cpu().numpy()),y[test])
                add_effects(m,baseline)
                grid.append(dict(config=c,step=step,metrics=m))
                all_scores[f'c{c["id"]}_t{step}']=scores
                if role:
                    p=dict(role=role,config=c,step=step,metrics=dict(m))
                    if step in SVM_STEPS:
                        s,h=core.svm_fit(a,b,y[train],groups[train])
                        p['metrics']['refit_svm']=core.classification(s,y[test])
                        models[f'{role}_t{step}']=h;all_scores[f'{role}_svm_t{step}']=s
                    if role in ('prior_soft','matched_hard'):
                        an=a/a.norm(dim=1,keepdim=True).clamp_min(1e-8)
                        bn=b/b.norm(dim=1,keepdim=True).clamp_min(1e-8)
                        p['metrics']['angular_ridge']=core.classification(core.ridge_scores(an,bn,y[train],groups[train]),y[test])
                        saved[f'{role}_t{step}']=b.cpu().numpy()
                    pairs.append(p)
                if c['id']==0:
                    assert np.allclose(scores,baseline_scores,atol=2e-4)
                if c['id']==1:
                    assert abs(m['within']/baseline['within']-1)<2e-5
                    assert abs(m['fisher']/baseline['fisher']-1)<2e-5
                if c['mode'] in ('soft','hard'):
                    assert m['max_support_ratio']<=1.00001
            if c['id']%12==11:print('GRID',dataset,layer,f,c['id']+1,'/66','seconds',round(time.monotonic()-started,1),flush=True)
        # Do not infer a temporal oscillation from only even checkpoints.
        for role in ('prior_soft','matched_hard'):
            a=saved[f'{role}_t99'];b=saved[f'{role}_t100']
            checkpoints[role]=dict(relative_l2_99_100_mean=float(np.mean(np.linalg.norm(b-a,axis=1)/np.maximum(np.linalg.norm(x[test].cpu().numpy(),axis=1),1e-8))),
                support_switch_units_99_100_mean=float(np.mean(((a>EPS)!=(b>EPS)).sum(1))))
        assert len(grid)==66*len(STEPS) and len(pairs)==4*len(STEPS)
        hard_ids={r['config']['id'] for r in grid if r['config']['mode']=='hard' and r['config']['gain']>0}
        assert len(hard_ids)==24
        write_json(dest/'results.json',dict(dataset=dataset,layer=layer,fold=f,baseline=baseline,
            grid=grid,pairs=pairs,late_step_diagnostics=checkpoints,graph_statistics=model['graph_statistics']))
        np.savez_compressed(dest/'pair_representations.npz',row=test,y=y[test],concept=groups[test],**saved)
        np.savez_compressed(dest/'readout_scores.npz',row=test,y=y[test],**all_scores)
        with (dest/'svm_models.pckl').open('wb') as handle:pickle.dump(models,handle,protocol=pickle.HIGHEST_PROTOCOL)
        write_json(dest/'complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
            files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},seconds=time.monotonic()-started))
        for role in ('prior_soft','matched_hard'):
            line=[p for p in pairs if p['role']==role and p['step'] in (1,5,20,100)]
            print('PAIR',dataset,layer,f,role,[(p['step'],{k:round(v,4) for k,v in p['metrics']['effects'].items()}) for p in line],flush=True)
        print('COMPLETE',dataset,layer,f,'seconds',round(time.monotonic()-started,1),flush=True)
        del model
    for p,h in protocol['original_model_sources'].items():assert sha(ROOT/p)==h


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('freeze','run'))
    p.add_argument('--dataset',choices=('imagenet','things'));p.add_argument('--layer',choices=('fc1','fc2'))
    a=p.parse_args()
    if a.action=='freeze':print('FROZEN',len(freeze()['configs']),'configurations',len(STEPS),'horizons',flush=True)
    else:
        with torch.inference_mode():
            for d in ([a.dataset] if a.dataset else ['imagenet','things']):
                for l in ([a.layer] if a.layer else ['fc1','fc2']):run(d,l)


if __name__=='__main__':main()
