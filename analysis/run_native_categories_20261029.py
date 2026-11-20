"""Study-local resumable stages; source and data gates precede formal extraction."""
from pathlib import Path
import argparse,json,sys,time,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments import native_categories_20261029 as study
from predify_experiments.readout_init_20261012 import sha,immutable_json

def freeze():
    assert not (study.OUT/'freeze.json').exists()
    for name in ('data_acceptance.json','input_acceptance.json','gray_acceptance.json','unit_acceptance.json'):
        d=json.loads((study.DOC/name).read_text(encoding='utf8'))
        accepted=(d.get('deep_copy_verified') is True and d.get('original_rep_hash_matched') is True and d.get('steps')==22) if name=='gray_acceptance.json' else d.get('passed') is True
        assert accepted,name
    for dataset in ('imagenet','things'):assert (study.OUT/dataset/'smoke/complete.json').exists()
    files=['predify_experiments/native_categories_20261029.py','analysis/native_categories_20261029.py','analysis/prepare_native_20261029.py','analysis/run_native_categories_20261029.py','analysis/report_native_categories_20261029.py','analysis/qc_native_20261029.py',
           'predify_experiments/tripartite_20261027.py','predify_experiments/filtered_cv_20261027.py','analysis/filtered_cv_20261027.py','predify_experiments/sequence.py','predify_experiments/hebb_vgg_priming.py','Hebbian-CNN/OnlineSharpeningStats_Demo.py','predify_experiments/readout_init_20261012.py','Hebbian-CNN/Hebb_Priming_Utils.py','configs/paper_memory_carrier_calibrated.toml','Hebbian-CNN/Hebbian_VGG_Lib.py','Hebbian-CNN/predify_hebb_inject.py','predify2021/model_factory/get_model.py','predify2021/model_factory/pvgg16_shared.py','analysis/plot_adaptive_20261028.py']
    weights=list((ROOT/'predify2021/model_factory').glob('pvgg16*pmodule.pth'))+list((Path.home()/'.cache/torch/hub/checkpoints').glob('vgg16*.pth'))
    assert len(weights)>=6
    files+=['predify_experiments/config.py','predify_experiments/paper_parafoveal.py','predify_experiments/paper_stimuli.py','predify_experiments/supplementary.py','analysis/refresh_2d_20261028.py','analysis/report_filtered_cv_20261027.py','tests/test_native_categories_20261029.py']
    value=dict(frozen_at=time.time(),sources={p:sha(ROOT/p) for p in files},manifests={d:sha(study.OUT/d/'manifest.json') for d in ('imagenet','things')},plan_sha=sha(study.PLAN),filter_sha=sha(study.OUT/'filter.npz'),gray_sha=sha(study.OUT/'gray_snapshot.pckl'),acceptance_sha={p:sha(study.DOC/p) for p in ('data_acceptance.json','input_acceptance.json','gray_acceptance.json','unit_acceptance.json')},weight_hashes={str(p):sha(p) for p in weights},target_sequences=35280,mvpa_fits=3600,t9A_fits=60,t9B_fits=10)
    for p in files:
        target=study.OUT/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,target)
    immutable_json(study.OUT/'freeze.json',value)

def pipeline():
    study.verify_freeze();path=study.OUT/'pipeline_status.json'
    state=json.loads(path.read_text(encoding='utf8')) if path.exists() else dict(started=time.time(),completed=[],active=None)
    stages=[]
    for d in ('imagenet','things'):stages.append((d+' extract',['analysis/run_native_categories_20261029.py','extract','--dataset',d]))
    for d in ('imagenet','things'):
        for a in ('decoding','representation'):stages.append((d+' '+a,['analysis/run_native_categories_20261029.py',a,'--dataset',d]))
    stages.append(('report',['analysis/report_native_categories_20261029.py']))
    for name,args in stages:
        if name in state['completed']:continue
        state.update(active=name,error=None);path.write_text(json.dumps(state,indent=2),encoding='utf8')
        with (study.OUT/(name.replace(' ','_')+'.log')).open('a',encoding='utf8') as log:
            p=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=p.pid;path.write_text(json.dumps(state,indent=2),encoding='utf8');code=p.wait()
        if code:
            state.update(active=None,error=dict(stage=name,exit_code=code));path.write_text(json.dumps(state,indent=2),encoding='utf8');raise RuntimeError(state['error'])
        state['completed'].append(name);state['active']=None;path.write_text(json.dumps(state,indent=2),encoding='utf8')
    state.update(active=None,finished=time.time());path.write_text(json.dumps(state,indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('inputs','smoke','freeze','extract','decoding','representation','pipeline'));p.add_argument('--dataset',choices=('imagenet','things'),default='imagenet');a=p.parse_args()
    if a.action=='inputs':study.input_acceptance()
    elif a.action in ('smoke','extract'):study.extract(a.dataset,a.action=='smoke')
    elif a.action=='freeze':freeze()
    elif a.action=='pipeline':pipeline()
    else:
        from analysis import native_categories_20261029 as analysis
        getattr(analysis,a.action)(a.dataset)
