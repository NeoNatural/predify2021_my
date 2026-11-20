"""Resumable execution; stage logs stay within this study directory."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.filtered_cv_20261027 import OUT,DOC

def main():
    path=OUT/'pipeline_status.json'
    state=json.loads(path.read_text(encoding='utf8')) if path.exists() else dict(started=time.time(),completed=[],active=None)
    stages=[]
    for dataset in ('imagenet','things'):stages.append((f'{dataset} extract',['-m','predify_experiments.filtered_cv_20261027','extract','--dataset',dataset]))
    for dataset in ('imagenet','things'):
        for action in ('decoding','representation'):stages.append((f'{dataset} {action}',['analysis/filtered_cv_20261027.py',action,'--dataset',dataset]))
    stages.extend([('audit',['analysis/audit_filtered_cv_20261027.py']),('report',['analysis/report_filtered_cv_20261027.py'])])
    for name,args in stages:
        if name in state['completed']:continue
        state.update(active=name,error=None);path.write_text(json.dumps(state,indent=2),encoding='utf8')
        print('STAGE',name,flush=True)
        log=OUT/(name.replace(' ','_')+'.log')
        with log.open('a',encoding='utf8') as f:
            p=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            state['child_pid']=p.pid;path.write_text(json.dumps(state,indent=2),encoding='utf8');code=p.wait()
        if code:
            state.update(error=dict(stage=name,exit_code=code,log=str(log)),active=None);path.write_text(json.dumps(state,indent=2),encoding='utf8')
            raise SystemExit(f'{name} failed; see {log}')
        state['completed'].append(name);state['active']=None;path.write_text(json.dumps(state,indent=2),encoding='utf8');print('DONE',name,flush=True)
    state.update(finished=time.time(),active=None);path.write_text(json.dumps(state,indent=2),encoding='utf8');print('PIPELINE COMPLETE',flush=True)

if __name__=='__main__':main()
