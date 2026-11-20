"""Rank-free population thresholds and clean nested calibration; backbone untouched."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import time
import numpy as np
import torch
from . import fc_degraded_k128_20261028 as old
from . import fc_lateral_study_20261028 as core
from . import fc_finite_steps_20261028 as finite
from .fc_lateral_vanilla_20261028 import ROOT,sha,write_json

OUT=ROOT/'Hebbian-CNN/Log/fc_population_threshold_20261028'
DOC=ROOT/'docs/fc_population_threshold_20261028'
FAMILIES=('moments','feedback')
TIMES=(1,5,20)
read=old.read


def candidates(family):
    if family=='moments':
        return [dict(family=family,a=a,b=b,theta0=0.) for a in (.25,.5,1.) for b in (0.,.1,.25,.5)]
    if family=='feedback':
        return [dict(family=family,g=g,rho=rho,theta0=0.) for g in (.25,.5,1.,2.,4.) for rho in (.25,.5,1.)]
    raise ValueError(family)


def trajectories(original,model,config,layer='fc1',gain=.75,steps=TIMES):
    """Only quota uses counts/sorting; the new paths pool all units per image."""
    if config['family']=='quota':
        c=dict(rule='corr',graph='k128' if layer=='fc1' else 'k32',mode='hard',gain=gain,budget=1.)
        yield from finite.trajectories(original,model,c,steps)
        return
    steps=tuple(sorted(set(steps)))
    x=original/model['scale'];r=x.clone()
    E=model['matrices']['corr_'+('k128' if layer=='fc1' else 'k32')]
    pool=x.mean(1,keepdim=True)
    for t in range(1,max(steps)+1):
        mean=r.mean(1,keepdim=True)
        drive=x+gain*(r@E.T) if gain else x
        if config['family']=='moments':
            theta=config['theta0']+config['a']*drive.mean(1,keepdim=True)+config['b']*drive.std(1,keepdim=True,unbiased=False)
            next_pool=pool
        elif config['family']=='feedback':
            theta=config['theta0']+config['g']*pool
            next_pool=(1-config['rho'])*pool+config['rho']*mean
        else:raise ValueError(config)
        target=torch.where(drive>theta,drive,0.)/(1+2*mean)
        updated=.5*r+.5*target
        updated=torch.where(target>0,updated,0.)
        if t in steps:
            assert torch.isfinite(updated).all()
            yield t,updated*model['scale'],dict(updates_applied=t,threshold_mean=float(theta.mean()),
                threshold_min=float(theta.min()),threshold_max=float(theta.max()),
                feedback_pool_mean=float(pool.mean()),
                residual=float(((updated-r).norm(dim=1)/x.norm(dim=1).clamp_min(1e-8)).max()),
                support_switch_units_mean=float(((updated*model['scale']>1e-7)!=(r*model['scale']>1e-7)).sum(1).float().mean()))
        r,pool=updated,next_pool


def geometry(r,y):
    return core.geometry(r,torch.as_tensor(y,device=r.device),False)


def score(m,reference):
    if m['zero_fraction']>0 or min(m['activation'],m['within'],m['fisher'],reference['activation'],reference['within'],reference['fisher'])<=1e-12:
        return -1e6
    return min(math.log(reference['activation']/m['activation']),
               math.log(reference['within']/m['within']),math.log(m['fisher']/reference['fisher']))


def rank(rows):
    for r in rows:
        r['all_inner_joint']=all(v['score']>0 for v in r['inner'])
        r['mean_score']=float(np.mean([v['score'] for v in r['inner']]))
    return max(rows,key=lambda r:(r['all_inner_joint'],r['mean_score']))


def freeze():
    old.validate(read(old.OUT/'protocol.json'))
    protocol=dict(version=1,source_sha256=sha(__file__),old_protocol_sha256=sha(old.OUT/'protocol.json'),
        grids={f:candidates(f) for f in FAMILIES},times=list(TIMES),selection_step=5,
        moment_rule='theta=theta0+a*mean(drive)+b*population_std(drive), all units within each image; theta0=0',
        feedback_rule='I0=mean(normalized actual input); theta_t=theta0+g*I_t; I_next=(1-rho)*I_t+rho*mean(r_t); per-image reset; theta0=0',
        fixed='clean outer-fold saved E1 corr/k128 E2 corr/k32 and RMS scales; gamma .75, division2, mixture .5; hard veto',
        selection='two concept-disjoint inner folds of clean outer training only; inner E/RMS fit only inner clean training; final outer E/RMS loaded unchanged from old study',
        fc1_score='minimum log improvement of activation, W, Fisher; prefer joint in both inner folds then mean score',
        fc2_score='same score for both local actual input and total vanilla FC2; minimum of all six terms; selected FC1 precedes original bridge',
        no_svm_selection=True,no_density_guard=True,no_fixed_point_filter=True,
        side_files={f'{d}/{f}':sha(old.OUT/d/f'fold{f}'/'clean_side_models.npz') for d in ('imagenet','things') for f in range(5)})
    OUT.mkdir(exist_ok=True);DOC.mkdir(exist_ok=True)
    p=OUT/'calibration_protocol.json'
    if p.exists():assert read(p)==protocol,'Frozen calibration changed'
    else:
        write_json(p,protocol);(OUT/'rules_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return protocol


def select(dataset,folds):
    protocol=freeze()
    original=old.validate(read(old.OUT/'protocol.json'))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weights=torch.load(original['checkpoint'],map_location='cpu',weights_only=True)
    w,b=(weights[f'classifier.3.{k}'].to(core.DEVICE) for k in ('weight','bias'));del weights
    with np.load(old.serial.DATA/dataset/'activations.npz') as z:
        x1,x2=(torch.as_tensor(z[k],device=core.DEVICE) for k in ('fc1','fc2'))
        y,groups,outer=z['category'],z['concept'],z['fold']
    for fold in folds:
        dest=OUT/dataset/f'fold{fold}';path=dest/'selection.json'
        if path.exists():
            assert read(path)['calibration_protocol_sha256']==sha(OUT/'calibration_protocol.json')
            print('SELECTION EXISTS',dataset,fold,flush=True);continue
        started=time.monotonic();tr=np.flatnonzero(outer!=fold)
        inner=core.inner_folds(y[tr],groups[tr],core.SEED+fold)
        pieces=[]
        for f in range(2):
            it,iv=tr[inner!=f],tr[inner==f]
            assert not set(groups[it])&set(groups[iv])
            m1=old.serial.fit_model(x1[it],old.strong.parameters('k128')[0])
            m2=old.serial.fit_model(x2[it],old.strong.parameters('k128')[1])
            pieces.append(dict(train_rows=it.tolist(),validation_rows=iv.tolist(),m1=m1,m2=m2,
                input1=x1[iv],vanilla2=x2[iv],labels=y[iv],
                base1=geometry(x1[iv],y[iv]),base2=geometry(x2[iv],y[iv])))
        results={}
        with torch.inference_mode():
            for family in FAMILIES:
                grid1=[]
                for config in candidates(family):
                    records=[]
                    for i,p in enumerate(pieces):
                        r=list(trajectories(p['input1'],p['m1'],config,'fc1',steps=(5,)))[0][1]
                        metric=geometry(r,p['labels'])
                        records.append(dict(inner_fold=i,score=score(metric,p['base1']),metrics=metric))
                    grid1.append(dict(config=config,inner=records))
                best1=rank(grid1)
                incoming=[]
                for p in pieces:
                    r=list(trajectories(p['input1'],p['m1'],best1['config'],'fc1',steps=(5,)))[0][1]
                    z=old.serial.bridge(r,w,b)
                    incoming.append((z,geometry(z,p['labels'])))
                grid2=[]
                for config in candidates(family):
                    records=[]
                    for i,p in enumerate(pieces):
                        z,reference=incoming[i]
                        r=list(trajectories(z,p['m2'],config,'fc2',steps=(5,)))[0][1]
                        metric=geometry(r,p['labels'])
                        local,total=score(metric,reference),score(metric,p['base2'])
                        records.append(dict(inner_fold=i,score=min(local,total),local_score=local,total_score=total,metrics=metric))
                    grid2.append(dict(config=config,inner=records))
                best2=rank(grid2)
                results[family]=dict(fc1=best1['config'],fc2=best2['config'],fc1_all_inner_joint=best1['all_inner_joint'],
                    fc2_all_inner_joint=best2['all_inner_joint'],fc1_grid=grid1,fc2_grid=grid2)
                print('SELECTED',dataset,fold,family,best1['config'],best2['config'],
                      'inner_joint',best1['all_inner_joint'],best2['all_inner_joint'],flush=True)
        write_json(path,dict(dataset=dataset,fold=fold,train_rows=tr.tolist(),
            inner_folds=[{k:p[k] for k in ('train_rows','validation_rows')} for p in pieces],
            results=results,calibration_protocol_sha256=sha(OUT/'calibration_protocol.json'),seconds=time.monotonic()-started))
        print('SELECTION DONE',dataset,fold,'seconds',round(time.monotonic()-started,1),flush=True)
    assert sha(__file__)==protocol['source_sha256']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('freeze','select'))
    p.add_argument('--dataset',choices=('imagenet','things'));p.add_argument('--folds',type=int,nargs='+',choices=range(5),default=list(range(5)))
    a=p.parse_args()
    if a.action=='freeze':freeze();print('FROZEN',sum(len(candidates(f)) for f in FAMILIES),'candidate rules per layer')
    else:
        for d in ([a.dataset] if a.dataset else ['imagenet','things']):select(d,a.folds)
