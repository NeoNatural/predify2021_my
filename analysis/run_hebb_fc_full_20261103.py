"""Restartable unattended execution of the complete FC experiment.

Scratch feature arrays are discarded only after fold scores and geometry finish;
Predify base data, learned graphs, scores, and metrics are retained.
"""
from pathlib import Path
import json,sys,time,traceback,contextlib,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.hebb_fc_full_20261103 import OUT,atomic_json
from analysis.hebb_fc_full_replay_20261103 import METHODS

LOG=OUT/'run.log';STATUS=OUT/'status.json'
def log(s):
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a',encoding='utf8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S ')+s+'\n')
def stage(label,fn,*args,**kwargs):
    atomic_json(STATUS,{'state':'running','stage':label,'updated':time.time()});log('START '+label)
    with LOG.open('a',encoding='utf8') as f:
        with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):fn(*args,**kwargs)
    gc.collect()
    import torch
    if torch.cuda.is_available():torch.cuda.empty_cache()
    log('DONE '+label)
def main():
    try:
        from analysis.hebb_fc_full_20261103 import graphs,extract
        from analysis.hebb_fc_full_replay_20261103 import replay
        from analysis.hebb_fc_full_decode_20261103 import decode
        from analysis.hebb_fc_full_geometry_20261103 import geometry
        from analysis.hebb_fc_full_report_20261103 import report
        for d in ('imagenet','things'):stage(d+' fold graph fitting',graphs,d)
        for d in ('imagenet','things'):
            if not (OUT/d/'extraction_complete.json').exists():
                stage(d+' Predify extraction',extract,d)
        for d in ('imagenet','things'):
            for fold in range(5):
                dest=OUT/d/'folds'/f'fold{fold}'
                if not (dest/'replay_complete.json').exists():stage(f'{d} fold{fold} FC replay',replay,d,fold)
                if not (dest/'decode_complete.json').exists():stage(f'{d} fold{fold} readouts',decode,d,fold,workers=8)
                if not (dest/'geometry_complete.json').exists():stage(f'{d} fold{fold} geometry',geometry,d,fold)
                # Only known scratch files created by this experiment are removed.
                released=dest/'scratch_released.json'
                if not released.exists():
                    removed=[]
                    for method in METHODS:
                        for suffix in ('fc2.npy','test_fc1.npy','reference_fc2.npy'):
                            path=dest/f'{method}_{suffix}'
                            if path.exists():path.unlink();removed.append(path.name)
                    atomic_json(released,{'reason':'fold scores and geometry completed; base trajectories and fold graphs retained','files':removed})
                    log(f'{d} fold{fold} released scratch {len(removed)} files')
        stage('final audit and report',report)
        atomic_json(STATUS,{'state':'complete','updated':time.time()});log('COMPLETE')
    except BaseException as e:
        log('FAILED '+repr(e)+'\n'+traceback.format_exc())
        atomic_json(STATUS,{'state':'failed','error':repr(e),'updated':time.time()})
        raise
if __name__=='__main__':main()
