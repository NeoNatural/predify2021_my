"""Frozen clean Hebbian graphs, rank-free thresholds, paired corruptions and SVM."""
from __future__ import annotations
import argparse
import copy
import pickle
from pathlib import Path
import time
import numpy as np
import torch
from . import fc_population_rules_20261028 as rules
from . import fc_degraded_k128_20261028 as old
from . import fc_degraded_inputs_20261028 as inputs
from . import fc_gain075_serial_20261028 as strong
from . import fc_serial_20261028 as serial
from . import fc_finite_steps_20261028 as finite
from . import fc_lateral_study_20261028 as core
from .fc_lateral_vanilla_20261028 import ROOT,sha,write_json

OUT,DOC=rules.OUT,rules.DOC
read=old.read
conditions=old.conditions
STAGES=old.STAGES
BASE_KEYS=('fc1__off_0','fc2_input__off_0')
FAMILIES=('moments','moments_std','feedback')


def selected_config(selection,family):
    if family!='moments_std':return {k:selection['results'][family][k] for k in ('fc1','fc2')}
    # Positive-b supplementary comparison fixed before any outer evaluation.
    source=selection['results']['moments']
    one=rules.rank(copy.deepcopy([r for r in source['fc1_grid'] if r['config']['b']>0]))
    # FC2 grid was calibrated downstream of the unconstrained FC1. Recalibrate
    # this supplementary pair using the original inner clean training only.
    return dict(fc1=one['config'],fc2=None)


def auxiliary_keys():
    keys=set(BASE_KEYS)
    for c in conditions():
        if c['t1'] not in (0,5):continue
        f=strong.feature_keys('k128',c)
        keys.update(f[k] for k in ('fc1_output','fc2_output'))
    assert len(keys)==10
    return keys


class PopulationRunner(old.FrozenRunner):
    def __init__(self,x1,x2,model1,model2,w,b,config):
        super().__init__(x1,x2,model1,model2,w,b)
        self.config=config;self.diagnostics={}

    def stage(self,mode,t):
        key=(mode,t)
        if key in self.stages:return self.stages[key]
        if mode=='gain_hard':
            r1=serial.mean_matched(self.x1,self.stage('hard',5)['fc1'])
            self.stages[key]=dict(fc1=r1,fc2_input=serial.bridge(r1,self.w,self.b))
        else:
            assert mode in ('hard','inhib_hard')
            for step,r1,diag in rules.trajectories(self.x1,self.model1,self.config['fc1'],'fc1',
                    gain=0. if mode=='inhib_hard' else .75,steps=(5,) if mode=='inhib_hard' else rules.TIMES):
                self.stages[(mode,step)]=dict(fc1=r1,fc2_input=serial.bridge(r1,self.w,self.b))
                self.diagnostics[f'fc1/{mode}/{step}']=diag
        return self.stages[key]

    def output(self,c):
        key=(c['m1'],c['t1']);z=self.stage(*key)['fc2_input']
        mode,t,cal=c['m2'],c['t2'],c['calibration']
        if mode=='off':return z
        assert cal=='frozen'
        if mode=='gain_hard':
            return serial.mean_matched(z,self.output(dict(m1='hard',t1=5,m2='hard',t2=5,calibration='frozen')))
        cache=(key,mode,t,cal)
        if cache not in self.outputs:
            model=self.side_model2(key,cal)
            for step,r,diag in rules.trajectories(z,model,self.config['fc2'],'fc2',
                    gain=0. if mode=='inhib_hard' else .75,steps=(5,) if mode=='inhib_hard' else rules.TIMES):
                self.outputs[(key,mode,step,cal)]=r
                self.diagnostics[f'fc2/{key}/{mode}/{step}']=diag
        return self.outputs[cache]


def freeze():
    rules.freeze();old.validate(read(old.OUT/'protocol.json'))
    p=dict(version=1,source_sha256=sha(__file__),calibration_protocol_sha256=sha(OUT/'calibration_protocol.json'),
        old_protocol_sha256=sha(old.OUT/'protocol.json'),conditions=conditions(),families=list(FAMILIES),
        primary_svm='method-specific clean-training StandardScaler + LinearSVC C1; all 19 feature keys, all times, frozen clean-to-degraded transfer',
        compatibility_svm='original clean vanilla C1 SVM; diagnostic only',
        auxiliary_svm='degraded C1 refit only 5-step FC1/final FC2 and controls; each repeat separately; baseline models reused exactly',
        auxiliary_feature_keys=sorted(auxiliary_keys()),baseline_reuse='same-fold old clean/refit vanilla features and heads only',
        quota_comparison='reuse all old artifacts unchanged; no quota refits',
        fixed='exact saved old clean E1/E2/RMS loaded, never recalibrated from degraded or propagated inputs',
        controls='inhibition-only same selected threshold parameters with gain0; per-image mean-matched scaling to current-family/current-input true 5-step output',
        statistics='three corruption repeats averaged within each concept-disjoint outer fold, then five folds equally',
        selection='clean nested geometry only; all steps retained irrespective of fixed-point convergence; no density eligibility',
        positive_std_comparison='additional moments_std family: restrict b>0 in the same frozen grid, choose FC1 on same inner folds and recalibrate FC2 downstream of that FC1 using only those inner folds',
        input_protocol_sha256=sha(inputs.INPUTS/'protocol.json'))
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==p,'Frozen experiment changed'
    else:write_json(path,p);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return p


def validate(p):
    assert sha(__file__)==p['source_sha256']
    assert sha(OUT/'calibration_protocol.json')==p['calibration_protocol_sha256']
    rules.freeze()
    assert sha(old.OUT/'protocol.json')==p['old_protocol_sha256']
    return old.validate(read(old.OUT/'protocol.json'))


def supplement(selection,folder,clean1,clean2,y,w,b):
    path=folder/'positive_std_selection.json'
    if path.exists():
        s=read(path);assert s['parent_sha256']==sha(folder/'selection.json');return s['config']
    config=selected_config(selection,'moments_std');records=[];pieces=[]
    with torch.inference_mode():
        for i,p in enumerate(selection['inner_folds']):
            it,iv=np.array(p['train_rows']),np.array(p['validation_rows'])
            m1=serial.fit_model(clean1[it],strong.parameters('k128')[0])
            m2=serial.fit_model(clean2[it],strong.parameters('k128')[1])
            r=list(rules.trajectories(clean1[iv],m1,config['fc1'],'fc1',steps=(5,)))[0][1]
            z=serial.bridge(r,w,b)
            pieces.append((z,m2,y[iv],rules.geometry(z,y[iv]),rules.geometry(clean2[iv],y[iv])))
        for c in rules.candidates('moments'):
            if c['b']<=0:continue
            inner=[]
            for i,(z,m2,labels,local,total) in enumerate(pieces):
                r=list(rules.trajectories(z,m2,c,'fc2',steps=(5,)))[0][1];m=rules.geometry(r,labels)
                inner.append(dict(inner_fold=i,score=min(rules.score(m,local),rules.score(m,total)),metrics=m))
            records.append(dict(config=c,inner=inner))
    best=rules.rank(records);config['fc2']=best['config']
    write_json(path,dict(parent_sha256=sha(folder/'selection.json'),config=config,fc2_grid=records,fc2_all_inner_joint=best['all_inner_joint']))
    return config


def phase(dataset,fold,label,x1,x2,n,y,groups,yt,model1,model2,w3,b3,w6,b6,fine,heads,
          clean1,clean2,primary_models,clean_features=None,old_features=None,old_scores=None,refit=False,config=None,refit_base=None):
    started=time.monotonic()
    runner=PopulationRunner(x1,x2,model1,model2,w3,b3,config)
    metrics,features,scores,models,paired={},{},{},{},{}
    previous_error=0.
    def characterize(key,r,layer):
        nonlocal previous_error
        if key in metrics:return copy.deepcopy(metrics[key])
        test=r[n:];reference=x1[n:] if layer=='fc1' else x2[n:]
        m=finite.evaluate(test,reference,yt);m.update(core.geometry(test,yt,detailed=True))
        a=test.cpu().numpy()
        if key in primary_models:
            primary=primary_models[key]
        else:
            assert not refit
            _,primary=core.svm_fit(r[:n],test,y[:n],groups[:n])
        primary_score=primary.decision_function(a)
        m.update(core.classification(primary_score,y[n:]))
        m['vanilla_svm']=core.classification(heads[layer][fold].decision_function(a),y[n:])
        if not refit:models[key]=primary
        if refit and key in auxiliary_keys():
            if key in refit_base:
                secondary=refit_base[key];secondary_score=secondary.decision_function(a)
            else:secondary_score,secondary=core.svm_fit(r[:n],test,y[:n],groups[:n])
            m['degraded_svm']=core.classification(secondary_score,y[n:])
            scores[key+'__degraded']=secondary_score;models[key]=secondary
        else:
            m['degraded_svm']=None
        if key in models:assert int(models[key][-1].n_iter_)<models[key][-1].max_iter
        scores[key+'__clean']=primary_score
        scores[key+'__vanilla']=heads[layer][fold].decision_function(a)
        clean=clean1[n:] if layer=='fc1' else clean2[n:]
        target=test if clean_features is None else torch.as_tensor(clean_features[key],device=serial.DEVICE)
        diagnostic,values=old.paired_diagnostics(test,clean,target,yt)
        m.update(diagnostic);paired[key]=values
        m['original_imagenet_top1_pct']=None
        if dataset=='imagenet' and layer=='fc2':
            logits=test@w6.T+b6
            m['original_imagenet_top1_pct']=100*float((logits.argmax(1)==fine).float().mean())
        if not refit and old_features is not None and key in old_features.files:
            assert np.array_equal(a,old_features[key])
            error=float(np.abs(primary_score-old_scores[key]).max())
            previous_error=max(previous_error,error);assert error<1e-8
        features[key]=a;metrics[key]=copy.deepcopy(m)
        if len(features)%5==0:
            print('FEATURE',dataset,fold,label,len(features),'/19','seconds',round(time.monotonic()-started,1),flush=True)
        return m
    base1=characterize('fc1__off_0',x1,'fc1')
    base2=characterize('fc2_input__off_0',x2,'fc2')
    results={}
    for c in conditions():
        stage=runner.stage(c['m1'],c['t1']);r=runner.output(c);keys=strong.feature_keys('k128',c)
        vectors=dict(fc1_output=stage['fc1'],fc2_input=stage['fc2_input'],fc2_output=r)
        ms={}
        for which in ('fc1_output','fc2_input','fc2_output'):
            layer='fc1' if which=='fc1_output' else 'fc2'
            m=characterize(keys[which],vectors[which],layer)
            serial.safe_effects(m,base1 if layer=='fc1' else base2);ms[which]=m
        local=copy.deepcopy(ms['fc2_output'])
        local.update(finite.evaluate(r[n:],stage['fc2_input'][n:],yt));serial.safe_effects(local,ms['fc2_input'])
        ms['fc2_local']=local
        if c['m2']=='gain_hard':
            target=runner.output(dict(m1='hard',t1=5,m2='hard',t2=5,calibration='frozen'))
            assert torch.allclose(r.mean(1),target.mean(1),atol=2e-6,rtol=1e-5)
        if c['m1']=='gain_hard':
            assert torch.allclose(stage['fc1'].mean(1),runner.stage('hard',5)['fc1'].mean(1),atol=2e-6,rtol=1e-5)
        results[c['name']]=dict(condition=c,feature_keys=keys,metrics=ms)
    assert len(features)==19
    return dict(results=results,dynamics=runner.diagnostics,config=config,previous_clean_svm_error=previous_error,seconds=time.monotonic()-started),features,scores,models,paired



def save_phase(dest,data,features,scores,models,paired,te,y,concepts,protocol):
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'results.json',data)
    np.savez_compressed(dest/'representations.npz',row=te,y=y,concept=concepts,**features)
    np.savez_compressed(dest/'scores.npz',row=te,y=y,**scores)
    np.savez_compressed(dest/'paired_diagnostics.npz',row=te,**paired)
    with (dest/'svm_models.pckl').open('wb') as f:pickle.dump(models,f,protocol=pickle.HIGHEST_PROTOCOL)
    write_json(dest/'complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
        files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='complete.json'}))


def run(dataset,folds):
    protocol=freeze();original=validate(protocol)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weights=torch.load(original['checkpoint'],map_location='cpu',weights_only=True)
    w3,b3=(weights[f'classifier.3.{k}'].to(core.DEVICE) for k in ('weight','bias'))
    w6,b6=(weights[f'classifier.6.{k}'].to(core.DEVICE) for k in ('weight','bias'));del weights
    with np.load(serial.DATA/dataset/'activations.npz') as z:
        clean1,clean2=(torch.as_tensor(z[k],device=core.DEVICE) for k in ('fc1','fc2'))
        y,concepts,outer=z['category'],z['concept'],z['fold']
    with (serial.DATA/dataset/'baseline_models.pckl').open('rb') as f:heads=pickle.load(f)['models']
    fine=np.array([r['imagenet_label'] for r in read(serial.DATA/dataset/'manifest.json')['rows']])
    for fold in folds:
        folder=OUT/dataset/f'fold{fold}';selection=read(folder/'selection.json')
        assert selection['calibration_protocol_sha256']==protocol['calibration_protocol_sha256']
        tr,te=np.flatnonzero(outer!=fold),np.flatnonzero(outer==fold)
        assert selection['train_rows']==tr.tolist() and not set(concepts[tr])&set(concepts[te])
        order=np.r_[tr,te];n=len(tr);yt=torch.as_tensor(y[te],device=core.DEVICE)
        a,b=clean1[order],clean2[order]
        previous=old.OUT/dataset/f'fold{fold}';side=previous/'clean_side_models.npz'
        assert sha(side)==read(OUT/'calibration_protocol.json')['side_files'][f'{dataset}/{fold}']
        with np.load(side) as z:
            assert np.array_equal(z['train_rows'],tr)
            m1=dict(scale=torch.tensor(float(z['scale1']),device=core.DEVICE),matrices={'corr_k128':torch.as_tensor(z['E1'],device=core.DEVICE)})
            m2=dict(scale=torch.tensor(float(z['scale2']),device=core.DEVICE),matrices={'corr_k32':torch.as_tensor(z['E2'],device=core.DEVICE)})
        with (previous/'clean/svm_models.pckl').open('rb') as f:baseline_clean={k:v for k,v in pickle.load(f).items() if k in BASE_KEYS}
        assert len(baseline_clean)==2
        for family in FAMILIES:
            dest=folder/family
            if (dest/'complete.json').exists():old.verify_marker(dest);print('EXISTS',dataset,fold,family,flush=True);continue
            config=supplement(selection,folder,clean1,clean2,y,w3,b3) if family=='moments_std' else selected_config(selection,family)
            clean_dest=dest/'clean'
            if (clean_dest/'complete.json').exists():
                old.verify_marker(clean_dest)
                with (clean_dest/'svm_models.pckl').open('rb') as f:primary=pickle.load(f)
                with np.load(clean_dest/'representations.npz') as z:clean_features={k:z[k] for k in z.files if k not in ('row','y','concept')}
            else:
                with torch.inference_mode():
                    data,clean_features,scores,primary,paired=phase(dataset,fold,'clean',a,b,n,y[order],concepts[order],yt,m1,m2,w3,b3,w6,b6,
                        torch.as_tensor(fine[te],device=core.DEVICE),heads,a,b,baseline_clean,config=config)
                data.update(dataset=dataset,fold=fold,family=family,mode='none',repeat=-1,level=0.,train_rows=tr.tolist(),test_rows=te.tolist(),clean_svm_reused=2,clean_svm_new=17)
                save_phase(clean_dest,data,clean_features,scores,primary,paired,te,y[te],concepts[te],protocol)
                del scores,paired
            for mode,level in inputs.DEGRADATIONS:
                for repeat in inputs.REPEATS:
                    target=dest/mode/f'repeat{repeat}'
                    if (target/'complete.json').exists():old.verify_marker(target);continue
                    cache=inputs.INPUTS/dataset/mode/f'repeat{repeat}';old.verify_marker(cache)
                    with np.load(cache/'activations.npz') as z:
                        assert np.array_equal(z['category'],y) and np.array_equal(z['fold'],outer) and np.array_equal(z['concept'],concepts)
                        x1,x2=(torch.as_tensor(z[k][order],device=core.DEVICE) for k in ('fc1','fc2'))
                    with (previous/mode/f'repeat{repeat}'/'svm_models.pckl').open('rb') as f:refit_base={k:v for k,v in pickle.load(f).items() if k in BASE_KEYS}
                    with torch.inference_mode():
                        data,features,scores,secondary,paired=phase(dataset,fold,f'{family}/{mode}/{repeat}',x1,x2,n,y[order],concepts[order],yt,m1,m2,w3,b3,w6,b6,
                            torch.as_tensor(fine[te],device=core.DEVICE),heads,a,b,primary,clean_features,refit=True,config=config,refit_base=refit_base)
                    data.update(dataset=dataset,fold=fold,family=family,mode=mode,level=level,repeat=repeat,train_rows=tr.tolist(),test_rows=te.tolist(),
                        clean_complete_sha256=sha(clean_dest/'complete.json'),input_complete_sha256=sha(cache/'complete.json'),refit_new=8,refit_reused=2)
                    save_phase(target,data,features,scores,secondary,paired,te,y[te],concepts[te],protocol)
                    print('PHASE DONE',dataset,fold,family,mode,repeat,'seconds',round(data['seconds'],1),flush=True)
                    del x1,x2,features,scores,secondary,paired
            write_json(dest/'complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),selection_sha256=sha(folder/'selection.json'),side_sha256=sha(side),
                files={'clean/complete.json':sha(clean_dest/'complete.json'),**{f'{m}/repeat{r}/complete.json':sha(dest/m/f'repeat{r}'/'complete.json') for m,_ in inputs.DEGRADATIONS for r in inputs.REPEATS}}))
            print('FAMILY DONE',dataset,fold,family,flush=True)
            del primary,clean_features
    validate(protocol);print('SHARD DONE',dataset,folds,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('freeze','run'))
    p.add_argument('--dataset',choices=('imagenet','things'));p.add_argument('--folds',type=int,nargs='+',default=list(range(5)))
    a=p.parse_args()
    if a.action=='freeze':freeze();print('FROZEN')
    else:
        for d in ([a.dataset] if a.dataset else ['imagenet','things']):run(d,a.folds)
