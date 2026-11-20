"""True serial FC1-lateral -> frozen VGG bridge -> FC2-lateral evaluation."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from . import fc_finite_steps_20261028 as finite
from . import fc_sparse_lateral_20261028 as sparse
from . import fc_lateral_study_20261028 as core
from .fc_lateral_vanilla_20261028 import ROOT,sha,write_json

DATA=finite.DATA
OUT=ROOT/'Hebbian-CNN/Log/fc_serial_20261028'
DOC=ROOT/'docs/fc_serial_20261028'
TIMES=(1,5,20)
DEVICE=core.DEVICE


def conditions():
    rows=[]
    def add(name,m1='off',t1=0,m2='off',t2=0,calibration='none',group='main'):
        rows.append(dict(name=name,m1=m1,t1=t1,m2=m2,t2=t2,calibration=calibration,group=group))
    add('baseline')
    for mode in ('soft','hard'):
        for t in TIMES:
            add(f'fc1_{mode}_{t}',mode,t)
            add(f'fc2_{mode}_{t}',m2=mode,t2=t,calibration='frozen')
    for t1 in TIMES:
        for t2 in TIMES:
            modes=[('hard','hard')] if t1!=t2 else [(a,b) for a in ('soft','hard') for b in ('soft','hard')]
            for m1,m2 in modes:
                for cal in ('frozen','matched'):
                    add(f'both_{m1}_{m2}_{t1}_{t2}_{cal}',m1,t1,m2,t2,cal)
    for mode in ('soft','hard'):
        add(f'inhibition_only_{mode}_5',f'inhib_{mode}',5,f'inhib_{mode}',5,'matched','control')
    for cal in ('frozen','matched'):
        add(f'gain1_hard_5_{cal}','gain_hard',5,'hard',5,cal,'control')
        add(f'gain2_hard_5_{cal}','hard',5,'gain_hard',5,cal,'control')
        add(f'gain_both_hard_5_{cal}','gain_hard',5,'gain_hard',5,cal,'control')
    for i,r in enumerate(rows):r['id']=i
    assert len(rows)==57 and len({r['name'] for r in rows})==57
    return rows


def svm_confirmation(name):
    if name in ('fc1_hard_5','fc2_hard_5','inhibition_only_hard_5'):return True
    return name in {f'both_hard_hard_{t}_{t}_{cal}' for t in TIMES for cal in ('frozen','matched')} | {f'gain_both_hard_5_{cal}' for cal in ('frozen','matched')}


def bridge(fc1,weight,bias):
    return torch.relu(fc1@weight.T+bias)


def mean_matched(original,target):
    return original*(target.mean(1)/original.mean(1).clamp_min(1e-8))[:,None]


def fit_model(train,config):
    m=sparse.connection_model(train)
    key=config['rule']+'_'+config['graph']
    m['matrices']={key:m['matrices'][key]}
    m['graph_statistics']={key:m['graph_statistics'][key]}
    return m


class SerialRunner:
    def __init__(self,x1,x2,n,config1,config2,w,b):
        self.x1,self.x2,self.n=x1,x2,n
        self.c1,self.c2=config1,config2
        self.w,self.b=w,b
        self.model1=fit_model(x1[:n],config1)
        self.model2=fit_model(x2[:n],config2)
        self.stages={('off',0):dict(fc1=x1,fc2_input=x2)}
        self.models2={}
        self.outputs={}

    def stage(self,mode,t):
        key=(mode,t)
        if key in self.stages:return self.stages[key]
        if mode=='gain_hard':
            r1=mean_matched(self.x1,self.stage('hard',5)['fc1'])
            self.stages[key]=dict(fc1=r1,fc2_input=bridge(r1,self.w,self.b))
        else:
            c=dict(self.c1,mode=mode.removeprefix('inhib_'))
            if mode.startswith('inhib_'):c['gain']=0.
            steps=(5,) if mode.startswith('inhib_') else TIMES
            for step,r1,diag in finite.trajectories(self.x1,self.model1,c,steps):
                self.stages[(mode,step)]=dict(fc1=r1,fc2_input=bridge(r1,self.w,self.b))
        return self.stages[key]

    def side_model2(self,key,calibration):
        if calibration=='frozen':return self.model2
        assert calibration=='matched'
        if key not in self.models2:
            self.models2[key]=fit_model(self.stage(*key)['fc2_input'][:self.n],self.c2)
        return self.models2[key]

    def output(self,condition):
        key=(condition['m1'],condition['t1'])
        z=self.stage(*key)['fc2_input']
        mode,t2,cal=condition['m2'],condition['t2'],condition['calibration']
        if mode=='off':return z
        if mode=='gain_hard':
            canonical=dict(m1='hard',t1=5,m2='hard',t2=5,calibration=cal)
            return mean_matched(z,self.output(canonical))
        cache=(key,mode,t2,cal)
        if cache not in self.outputs:
            model=self.side_model2(key,cal)
            c=dict(self.c2,mode=mode.removeprefix('inhib_'))
            if mode.startswith('inhib_'):c['gain']=0.
            steps=(5,) if mode.startswith('inhib_') else TIMES
            for step,r,diag in finite.trajectories(z,model,c,steps):self.outputs[(key,mode,step,cal)]=r
        return self.outputs[cache]


def safe_effects(m,base):
    if min(base['activation'],base['within'],base['fisher'])>1e-12:
        finite.add_effects(m,base)
    else:
        m['effects']={k:None for k in ('activation_pct','within_pct','between_pct','fisher_pct')}
        m['effects'].update(accuracy_pp=100*(m['accuracy']-base['accuracy']),auc_delta=m['auc']-base['auc'])
        m['joint']=False;m['sparse_joint']=False
    if m.get('collapsed',False):m['joint']=False;m['sparse_joint']=False


def freeze(checkpoint):
    protocol=dict(version=1,exploratory_reuse=True,conditions=conditions(),times=TIMES,
        architecture='FC1 post-ReLU -> lateral1 finite updates -> unmodified classifier.3 linear + ReLU -> lateral2 finite updates; eval dropout is identity; sequential without FC2-to-FC1 feedback',
        calibration=dict(frozen='all FC2 lateral moments and scale learned from vanilla FC2 training cache',
            matched='FC2 lateral weights and scale re-estimated on actual FC1-modified bridge training inputs only'),
        selection='no new parameter/horizon selection; prior fold-specific correlation graph/gain/budget retained; only soft vs hard and specified horizons varied',
        readout='train-only ridge for FC1, propagated FC2 input and final FC2; frozen vanilla SVM all final outputs; refit C1 SVM on eleven prespecified final-output comparisons; angular ridge diagnostic only',
        interpretation='end-to-end FC2 changes vs original cache and local FC2 changes vs propagated input separately; FC2 support budget uses its actual incoming activity count, not the old cache count',
        baseline='original cached FC2 used for FC1-off baseline after verifying full bridge reconstruction within FP32 tolerance',
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),
        dependencies={p:sha(ROOT/p) for p in ['predify_experiments/fc_finite_steps_20261028.py','predify_experiments/fc_sparse_lateral_20261028.py',
            'predify_experiments/fc_lateral_study_20261028.py','predify_experiments/fc_lateral_vanilla_20261028.py']},
        original_model_sources=sparse.freeze()['original_model_sources'],
        inputs={d:{p:sha(DATA/d/p) for p in ['manifest.json','activations.npz','baseline_models.pckl']} for d in ('imagenet','things')},
        old_selections={f'{d}/{l}/{f}':sha(sparse.OUT/d/l/f'fold{f}'/'selection.json') for d in ('imagenet','things') for l in ('fc1','fc2') for f in range(5)})
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==json.loads(json.dumps(protocol)),'Frozen source changed'
    else:
        write_json(path,protocol);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return protocol


def run_fold(dataset,fold,x1,x2,y,groups,folds,fine_labels,w3,b3,w6,b6,heads,protocol):
    dest=OUT/dataset/f'fold{fold}'
    if (dest/'complete.json').exists():print('EXISTS',dataset,fold,flush=True);return
    started=time.monotonic()
    train,test=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
    assert not set(groups[train])&set(groups[test])
    order=np.concatenate([train,test]);n=len(train);yt=torch.as_tensor(y[test],device=DEVICE)
    c1=json.loads((sparse.OUT/dataset/'fc1'/f'fold{fold}'/'selection.json').read_text(encoding='utf-8'))['choices']['selected']['config']
    c2=json.loads((sparse.OUT/dataset/'fc2'/f'fold{fold}'/'selection.json').read_text(encoding='utf-8'))['choices']['selected']['config']
    runner=SerialRunner(x1[order],x2[order],n,c1,c2,w3,b3)
    stage_metrics={};results={};features={};scores_saved={};fitted={}

    def characterize(r,reference,layer):
        a,b=r[:n],r[n:]
        m=finite.evaluate(b,reference[n:],yt)
        m.update(core.geometry(b,yt,detailed=True))
        m['collapsed']=bool(m['effective_dimension']<1e-6)
        s=core.ridge_scores(a,b,y[train],groups[train]);m.update(core.classification(s,y[test]))
        m['frozen_svm']=core.classification(heads[layer][fold].decision_function(b.cpu().numpy()),y[test])
        return m,s

    base1,_=characterize(runner.x1,runner.x1,'fc1')
    base2,_=characterize(runner.x2,runner.x2,'fc2')
    def describe_stage(key):
        if key not in stage_metrics:
            stage=runner.stage(*key)
            m1,s1=characterize(stage['fc1'],runner.x1,'fc1');safe_effects(m1,base1)
            mi,si=characterize(stage['fc2_input'],runner.x2,'fc2');safe_effects(mi,base2)
            stage_metrics[key]=dict(fc1=m1,fc2_input=mi)
            tag=f'{key[0]}_{key[1]}'
            scores_saved[f'fc1_{tag}']=s1;scores_saved[f'input2_{tag}']=si
        return stage_metrics[key]

    for condition in conditions():
        name=condition['name'];key=(condition['m1'],condition['t1'])
        r=runner.output(condition);stage=runner.stage(*key);smeta=describe_stage(key)
        m,s=characterize(r,runner.x2,'fc2');safe_effects(m,base2)
        local=finite.evaluate(r[n:],stage['fc2_input'][n:],yt)
        local.update(accuracy=m['accuracy'],auc=m['auc']);safe_effects(local,smeta['fc2_input'])
        m['local_fc2']=local
        m['fc1']=smeta['fc1'];m['propagated_fc2']=smeta['fc2_input']
        m['combined_fc_mean_pct']=100*((m['activation']+smeta['fc1']['activation'])/(base2['activation']+base1['activation'])-1)
        if condition['m2'] in ('soft','hard','inhib_soft','inhib_hard'):assert local['max_support_ratio']<=1.00001
        if svm_confirmation(name):
            sc,h=core.svm_fit(r[:n],r[n:],y[train],groups[train]);m['refit_svm']=core.classification(sc,y[test])
            fitted[name]=h;scores_saved[name+'_svm']=sc
        if name=='baseline':m['refit_svm']=base2['frozen_svm']
        un=r/r.norm(dim=1,keepdim=True).clamp_min(1e-8)
        m['angular_ridge']=core.classification(core.ridge_scores(un[:n],un[n:],y[train],groups[train]),y[test])
        logits=r[n:]@w6.T+b6
        if dataset=='imagenet':
            labels=torch.as_tensor(fine_labels[test],device=DEVICE)
            m['original_imagenet_top1_pct']=100*float((logits.argmax(1)==labels).float().mean())
            m['original_imagenet_top5_pct']=100*float((logits.topk(5,dim=1).indices==labels[:,None]).any(1).float().mean())
        else:m['original_imagenet_top1_pct']=None;m['original_imagenet_top5_pct']=None
        assert torch.isfinite(r).all() and bool((r>=0).all())
        results[name]=dict(condition=condition,metrics=m);features[name]=r[n:].cpu().numpy();scores_saved[name]=s
        if condition['id']%12==11:print('CONDITION',dataset,fold,condition['id']+1,'/57','seconds',round(time.monotonic()-started,1),flush=True)
    stages={}
    for key,stage in runner.stages.items():
        tag=f'{key[0]}_{key[1]}'
        stages['fc1_'+tag]=stage['fc1'][n:].cpu().numpy()
        stages['input2_'+tag]=stage['fc2_input'][n:].cpu().numpy()
    assert np.array_equal(features['baseline'],x2[test].cpu().numpy())
    for cal in ('frozen','matched'):
        canonical=features[f'both_hard_hard_5_5_{cal}']
        for ctrl in (f'gain2_hard_5_{cal}',f'gain_both_hard_5_{cal}'):
            assert np.allclose(features[ctrl].mean(1),canonical.mean(1),atol=2e-6)
    assert len(results)==57
    write_json(dest/'results.json',dict(dataset=dataset,fold=fold,train_rows=train.tolist(),test_rows=test.tolist(),
        params1=c1,params2=c2,baseline_fc1=base1,baseline_fc2=base2,results=results,
        matched_models={str(k):dict(scale=float(v['scale']),graph_statistics=v['graph_statistics']) for k,v in runner.models2.items()}))
    np.savez_compressed(dest/'final_fc2.npz',row=test,y=y[test],concept=groups[test],**features)
    np.savez_compressed(dest/'intermediate_stages.npz',row=test,y=y[test],**stages)
    np.savez_compressed(dest/'readout_scores.npz',row=test,y=y[test],**scores_saved)
    with (dest/'svm_models.pckl').open('wb') as handle:pickle.dump(fitted,handle,protocol=pickle.HIGHEST_PROTOCOL)
    write_json(dest/'complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},seconds=time.monotonic()-started))
    for name in ('fc1_hard_5','fc2_hard_5','both_hard_hard_5_5_frozen','both_hard_hard_5_5_matched'):
        print('RESULT',dataset,fold,name,results[name]['metrics']['effects'],'seconds',round(time.monotonic()-started,1),flush=True)


def run(dataset,checkpoint):
    protocol=freeze(checkpoint)
    torch.set_num_threads(4);torch.manual_seed(core.SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    w3,b3=(state[f'classifier.3.{k}'].to(DEVICE) for k in ('weight','bias'))
    w6,b6=(state[f'classifier.6.{k}'].to(DEVICE) for k in ('weight','bias'))
    del state
    with np.load(DATA/dataset/'activations.npz') as z:
        x1=torch.as_tensor(z['fc1'],device=DEVICE);x2=torch.as_tensor(z['fc2'],device=DEVICE)
        logits_ref=torch.as_tensor(z['logits'],device=DEVICE)
        y,groups,folds=z['category'],z['concept'],z['fold']
    bridge_error=float((bridge(x1,w3,b3)-x2).abs().max())
    head_error=float((x2@w6.T+b6-logits_ref).abs().max())
    assert bridge_error<1e-4 and head_error<1e-4
    write_json(OUT/dataset/'bridge_check.json',dict(bridge_max_error=bridge_error,head_max_error=head_error,
        checkpoint_sha256=sha(checkpoint),eval_dropout_identity=True))
    manifest=json.loads((DATA/dataset/'manifest.json').read_text(encoding='utf-8'))
    fine_labels=np.array([r['imagenet_label'] for r in manifest['rows']],dtype=np.int64)
    with (DATA/dataset/'baseline_models.pckl').open('rb') as f:heads=pickle.load(f)['models']
    for fold in range(5):run_fold(dataset,fold,x1,x2,y,groups,folds,fine_labels,w3,b3,w6,b6,heads,protocol)
    for p,h in protocol['original_model_sources'].items():assert sha(ROOT/p)==h
    assert sha(checkpoint)==protocol['checkpoint_sha256']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'));parser.add_argument('--dataset',choices=('imagenet','things'))
    args=parser.parse_args();checkpoint=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
    if args.action=='freeze':print('FROZEN',len(freeze(checkpoint)['conditions']),'serial conditions',flush=True)
    else:
        with torch.inference_mode():
            for d in ([args.dataset] if args.dataset else ['imagenet','things']):run(d,checkpoint)


if __name__=='__main__':main()
