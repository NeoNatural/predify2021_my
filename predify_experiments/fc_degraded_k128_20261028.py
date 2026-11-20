"""Clean-learned frozen lateral k128->k32 on paired degraded inputs."""
from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path
import time

import numpy as np
import torch

from . import fc_degraded_inputs_20261028 as inputs
from . import fc_gain075_serial_20261028 as strong
from . import fc_serial_20261028 as serial
from . import fc_finite_steps_20261028 as finite
from . import fc_lateral_study_20261028 as core
from .fc_lateral_vanilla_20261028 import ROOT,sha,write_json

OUT,DOC=inputs.OUT,inputs.DOC
read=inputs.read
STAGES=strong.STAGES


def conditions():
    result=[dict(name='baseline',m1='off',t1=0,m2='off',t2=0,calibration='none')]
    for t in (1,5,20):
        result.append(dict(name=f'both_hard_hard_{t}_{t}_frozen',m1='hard',t1=t,m2='hard',t2=t,calibration='frozen'))
    result.append(dict(name='inhibition_only_hard_5_frozen',m1='inhib_hard',t1=5,m2='inhib_hard',t2=5,calibration='frozen'))
    for name,m1,m2 in (('gain1','gain_hard','hard'),('gain2','hard','gain_hard'),('gain_both','gain_hard','gain_hard')):
        result.append(dict(name=f'{name}_hard_5_frozen',m1=m1,t1=5,m2=m2,t2=5,calibration='frozen'))
    assert len(result)==8
    return result


class FrozenRunner(serial.SerialRunner):
    """This API accepts only supplied clean-trained models, never fits on its inputs."""
    def __init__(self,x1,x2,model1,model2,w,b):
        self.x1,self.x2=x1,x2
        self.c1,self.c2=strong.parameters('k128')
        self.model1,self.model2=model1,model2
        self.w,self.b=w,b
        self.stages={('off',0):dict(fc1=x1,fc2_input=x2)}
        self.models2,self.outputs={},{}

    def side_model2(self,key,calibration):
        assert calibration=='frozen','Degraded inputs must never recalibrate lateral statistics'
        return self.model2


def freeze():
    old=read(strong.OUT/'protocol.json');strong.validate(old)
    protocol=dict(version=1,source_sha256=sha(__file__),
        clean_study_protocol_sha256=sha(strong.OUT/'protocol.json'),
        input_protocol_sha256=sha(inputs.INPUTS/'protocol.json'),
        dependencies={p:sha(ROOT/p) for p in ('predify_experiments/fc_degraded_inputs_20261028.py',
            'predify_experiments/fc_serial_20261028.py','predify_experiments/fc_gain075_serial_20261028.py',
            'predify_experiments/fc_lateral_study_20261028.py','predify_experiments/fc_finite_steps_20261028.py',
            'Hebbian-CNN/Hebb_Priming_Utils.py')},
        conditions=conditions(),fc1=strong.parameters('k128')[0],fc2=strong.parameters('k128')[1],
        lateral_estimation='only original clean vanilla training FC1/FC2; weights and RMS scales fixed across all degradation types and repeats',
        primary_svm='method-specific clean-training C1 SVM -> degraded held-out concepts; reuse exact old clean models when features agree',
        compatibility_svm='original clean vanilla SVM -> every degraded processing condition',
        secondary_svm='same C1 pipeline refitted on degraded training concepts, separately per repeat; no test fit; same number of training images',
        repeats=list(inputs.REPEATS),degradations=[list(d) for d in inputs.DEGRADATIONS],
        primary_goal='within-degradation activation down + normalized W down + Fisher up; each layer separately',
        clean_reference='paired cosine to original clean vanilla and same-method clean representations; diagnostics, not hard eligibility',
        selection='no tuning, no new threshold mechanism, no fixed-point filter, no ranking by SVM',
        clean_complete_hashes={f'{d}/{f}':sha(strong.OUT/d/f'fold{f}'/'complete.json') for d in ('imagenet','things') for f in range(5)})
    p=OUT/'protocol.json'
    if p.exists():assert read(p)==protocol,'Frozen degraded experiment changed'
    else:
        write_json(p,protocol);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    validate(protocol)
    return protocol


def validate(protocol):
    assert sha(__file__)==protocol['source_sha256']
    assert sha(inputs.INPUTS/'protocol.json')==protocol['input_protocol_sha256']
    for p,h in protocol['dependencies'].items():assert sha(ROOT/p)==h
    assert sha(strong.OUT/'protocol.json')==protocol['clean_study_protocol_sha256']
    return strong.validate(read(strong.OUT/'protocol.json'))


def verify_marker(dest,name='complete.json'):
    marker=read(dest/name)
    for p,h in marker['files'].items():assert sha(dest/p)==h
    return marker


def paired_diagnostics(r,clean,method_clean,yt):
    def cosine(a,b):
        return (a*b).sum(1)/(a.norm(dim=1)*b.norm(dim=1)).clamp_min(1e-12)
    original_cos=cosine(r,clean)
    method_cos=cosine(r,method_clean)
    raw_relative=(r-clean).norm(dim=1)/clean.norm(dim=1).clamp_min(1e-8)
    equal_class=lambda a:float(torch.stack([a[yt==c].mean() for c in range(5)]).mean())
    return dict(cosine_to_clean_vanilla=equal_class(original_cos),cosine_to_clean_same_method=equal_class(method_cos),
        relative_l2_to_clean_vanilla=equal_class(raw_relative),
        per_class_clean_cosine=[float(original_cos[yt==c].mean()) for c in range(5)]),torch.stack((original_cos,method_cos,raw_relative),1).cpu().numpy()


def phase(dataset,fold,label,x1,x2,n,y,groups,yt,model1,model2,w3,b3,w6,b6,fine,heads,
          clean1,clean2,primary_models,clean_features=None,old_features=None,old_scores=None,refit=False):
    started=time.monotonic()
    runner=FrozenRunner(x1,x2,model1,model2,w3,b3)
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
        models[key]=primary
        if refit:
            secondary_score,secondary=core.svm_fit(r[:n],test,y[:n],groups[:n])
            m['degraded_svm']=core.classification(secondary_score,y[n:])
            scores[key+'__degraded']=secondary_score;models[key]=secondary
        else:
            m['degraded_svm']=None
        assert int(models[key][-1].n_iter_)<models[key][-1].max_iter
        scores[key+'__clean']=primary_score
        scores[key+'__vanilla']=heads[layer][fold].decision_function(a)
        clean=clean1[n:] if layer=='fc1' else clean2[n:]
        target=test if clean_features is None else torch.as_tensor(clean_features[key],device=serial.DEVICE)
        diagnostic,values=paired_diagnostics(test,clean,target,yt)
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
            finite.add_effects(m,base1 if layer=='fc1' else base2);ms[which]=m
        local=copy.deepcopy(ms['fc2_output'])
        local.update(finite.evaluate(r[n:],stage['fc2_input'][n:],yt));finite.add_effects(local,ms['fc2_input'])
        ms['fc2_local']=local
        if c['m2'] in ('hard','inhib_hard'):assert local['max_support_ratio']<=1.00001
        if c['m2']=='gain_hard':
            target=runner.output(dict(m1='hard',t1=5,m2='hard',t2=5,calibration='frozen'))
            assert torch.allclose(r.mean(1),target.mean(1),atol=2e-6,rtol=1e-5)
        if c['m1']=='gain_hard':
            assert torch.allclose(stage['fc1'].mean(1),runner.stage('hard',5)['fc1'].mean(1),atol=2e-6,rtol=1e-5)
        results[c['name']]=dict(condition=c,feature_keys=keys,metrics=ms)
    assert len(features)==19
    return dict(results=results,previous_clean_svm_error=previous_error,seconds=time.monotonic()-started),features,scores,models,paired


def save_phase(dest,data,features,scores,models,paired,te,y,concepts,protocol):
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'results.json',data)
    np.savez_compressed(dest/'representations.npz',row=te,y=y,concept=concepts,**features)
    np.savez_compressed(dest/'scores.npz',row=te,y=y,**scores)
    np.savez_compressed(dest/'paired_diagnostics.npz',row=te,**paired)
    with (dest/'svm_models.pckl').open('wb') as f:pickle.dump(models,f,protocol=pickle.HIGHEST_PROTOCOL)
    write_json(dest/'complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
        files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))


def run(dataset,requested_folds):
    protocol=freeze();original=validate(protocol)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weights=torch.load(original['checkpoint'],map_location='cpu',weights_only=True)
    w3,b3=(weights[f'classifier.3.{k}'].to(serial.DEVICE) for k in ('weight','bias'))
    w6,b6=(weights[f'classifier.6.{k}'].to(serial.DEVICE) for k in ('weight','bias'));del weights
    with np.load(serial.DATA/dataset/'activations.npz') as z:
        clean1,clean2=(torch.as_tensor(z[k],device=serial.DEVICE) for k in ('fc1','fc2'))
        y,concepts,folds=z['category'],z['concept'],z['fold']
    with (serial.DATA/dataset/'baseline_models.pckl').open('rb') as f:heads=pickle.load(f)['models']
    manifest=read(serial.DATA/dataset/'manifest.json')
    fine=np.array([r['imagenet_label'] for r in manifest['rows']])
    for fold in requested_folds:
        dest=OUT/dataset/f'fold{fold}'
        if (dest/'complete.json').exists():
            verify_marker(dest);print('EXISTS FOLD',dataset,fold,flush=True);continue
        tr,te=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        assert not set(concepts[tr])&set(concepts[te])
        order=np.concatenate((tr,te));n=len(tr)
        a,b=clean1[order],clean2[order];yt=torch.as_tensor(y[te],device=serial.DEVICE)
        c1,c2=strong.parameters('k128')
        model1,model2=serial.fit_model(clean1[tr],c1),serial.fit_model(clean2[tr],c2)
        old=strong.OUT/dataset/f'fold{fold}'
        assert sha(old/'complete.json')==protocol['clean_complete_hashes'][f'{dataset}/{fold}'];verify_marker(old)
        clean_dest=dest/'clean'
        if (clean_dest/'complete.json').exists():
            verify_marker(clean_dest)
            with (clean_dest/'svm_models.pckl').open('rb') as f:primary_models=pickle.load(f)
            with np.load(clean_dest/'representations.npz') as z:
                clean_features={k:z[k] for k in z.files if k not in ('row','y','concept')}
        else:
            with (old/'svm_models.pckl').open('rb') as f:old_models=pickle.load(f)
            with np.load(old/'representations.npz') as old_features,np.load(old/'scores.npz') as old_scores,torch.inference_mode():
                data,clean_features,scores,primary_models,paired=phase(dataset,fold,'clean',a,b,n,y[order],concepts[order],yt,
                    model1,model2,w3,b3,w6,b6,torch.as_tensor(fine[te],device=serial.DEVICE),heads,a,b,old_models,
                    old_features=old_features,old_scores=old_scores)
            data.update(dataset=dataset,fold=fold,mode='none',repeat=-1,train_rows=tr.tolist(),test_rows=te.tolist(),
                clean_svm_reused=sum(k in old_models for k in primary_models),clean_svm_new=sum(k not in old_models for k in primary_models))
            save_phase(clean_dest,data,clean_features,scores,primary_models,paired,te,y[te],concepts[te],protocol)
            del old_models,scores,paired
        side=dest/'clean_side_models.npz'
        if side.exists():
            with np.load(side) as saved:
                assert np.array_equal(saved['E1'],model1['matrices']['corr_k128'].cpu().numpy())
                assert np.array_equal(saved['E2'],model2['matrices']['corr_k32'].cpu().numpy())
                assert np.array_equal(saved['train_rows'],tr)
        else:
            np.savez_compressed(side,E1=model1['matrices']['corr_k128'].cpu().numpy(),
                E2=model2['matrices']['corr_k32'].cpu().numpy(),scale1=float(model1['scale']),scale2=float(model2['scale']),train_rows=tr)
        for mode,level in inputs.DEGRADATIONS:
            for repeat in inputs.REPEATS:
                phase_dest=dest/mode/f'repeat{repeat}'
                if (phase_dest/'complete.json').exists():
                    verify_marker(phase_dest);print('EXISTS PHASE',dataset,fold,mode,repeat,flush=True);continue
                cache=inputs.INPUTS/dataset/mode/f'repeat{repeat}'
                marker=verify_marker(cache)
                assert marker['protocol_sha256']==protocol['input_protocol_sha256']
                with np.load(cache/'activations.npz') as z:
                    assert np.array_equal(z['category'],y) and np.array_equal(z['fold'],folds) and np.array_equal(z['concept'],concepts)
                    x1,x2=(torch.as_tensor(z[k][order],device=serial.DEVICE) for k in ('fc1','fc2'))
                with torch.inference_mode():
                    data,features,scores,secondary_models,paired=phase(dataset,fold,f'{mode}/{repeat}',x1,x2,n,y[order],concepts[order],yt,
                        model1,model2,w3,b3,w6,b6,torch.as_tensor(fine[te],device=serial.DEVICE),heads,a,b,primary_models,clean_features,refit=True)
                data.update(dataset=dataset,fold=fold,mode=mode,level=level,repeat=repeat,train_rows=tr.tolist(),test_rows=te.tolist(),
                    clean_complete_sha256=sha(clean_dest/'complete.json'),input_complete_sha256=sha(cache/'complete.json'))
                save_phase(phase_dest,data,features,scores,secondary_models,paired,te,y[te],concepts[te],protocol)
                print('PHASE DONE',dataset,fold,mode,repeat,'seconds',round(data['seconds'],1),flush=True)
                del x1,x2,features,scores,secondary_models,paired
        write_json(dest/'complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
            files={'clean_side_models.npz':sha(dest/'clean_side_models.npz'),'clean/complete.json':sha(clean_dest/'complete.json'),
                   **{f'{mode}/repeat{r}/complete.json':sha(dest/mode/f'repeat{r}'/'complete.json') for mode,_ in inputs.DEGRADATIONS for r in inputs.REPEATS}}))
        print('FOLD DONE',dataset,fold,flush=True)
    validate(protocol);print('SHARD DONE',dataset,requested_folds,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('freeze','run'))
    p.add_argument('--dataset',choices=('imagenet','things'));p.add_argument('--folds',type=int,nargs='+',choices=range(5),default=list(range(5)))
    args=p.parse_args()
    if args.action=='freeze':print('FROZEN',len(freeze()['conditions']),'conditions')
    else:
        for d in ([args.dataset] if args.dataset else ['imagenet','things']):run(d,args.folds)
