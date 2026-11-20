"""Resume the prespecified study; each dependent phase stops on failure."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Hebbian-CNN/Log/tripartite_20261027'
status=dict(started=time.time(),completed=[],active=None)


def run(label,*arguments):
    status['active']=label
    (OUT/'pipeline_status.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    print('PHASE',label,flush=True)
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[name]='1'
    subprocess.run([sys.executable,*arguments],cwd=ROOT,env=env,check=True)
    status['completed'].append(label)


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-training',action='store_true');args=p.parse_args()
    if args.after_training:
        for dataset in ('imagenet','things'):
            assert len(list((OUT/dataset/'train').glob('image_*.pckl')))==240,'Training still incomplete; do not start a duplicate GPU job'
    else:
        for dataset in ('imagenet','things'):
            run(dataset+' train','-m','predify_experiments.tripartite_20261027','train','--dataset',dataset)
    for dataset in ('imagenet','things'):
        if not (OUT/dataset/'head.pckl').exists():
            run(dataset+' fit','analysis/tripartite_20261027.py','fit','--dataset',dataset)
    for dataset in ('imagenet','things'):
        run(dataset+' dev','-m','predify_experiments.tripartite_20261027','dev','--dataset',dataset)
        if not (OUT/dataset/'freeze.json').exists():
            run(dataset+' freeze','analysis/tripartite_20261027.py','freeze','--dataset',dataset)
    for dataset in ('imagenet','things'):
        run(dataset+' test','-m','predify_experiments.tripartite_20261027','test','--dataset',dataset)
    for dataset in ('imagenet','things'):
        run(dataset+' old layout','-m','predify_experiments.tripartite_20261027','old','--dataset',dataset)
        run(dataset+' old diagnostics','analysis/tripartite_20261027.py','old','--dataset',dataset)
        run(dataset+' analysis','analysis/tripartite_20261027.py','analyze','--dataset',dataset)
    run('multiplicity correction','analysis/tripartite_20261027.py','holm')
    run('final audit','analysis/audit_tripartite_20261027.py')
    run('figures','analysis/plot_tripartite_20261027.py')
    run('report','analysis/report_tripartite_20261027.py')
    status['active']=None;status['finished']=time.time()
    (OUT/'pipeline_status.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    print('PIPELINE COMPLETE',flush=True)


if __name__=='__main__':main()
