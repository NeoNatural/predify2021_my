"""Execution-only supplement: frozen fits/statistics, finer process scheduling."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,json,time,inspect,hashlib,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.native_categories_20261029 import OUT,DOC,verify_freeze,conditions,NOISES
from predify_experiments.supplementary import load,save
from predify_experiments.readout_init_20261012 import sha
WORK=OUT/'parallel_execution_20261030'

def model_job(task):
    dataset,noise,ci,f,t,probe=task
    from analysis import native_categories_20261029 as a
    m=json.loads((OUT/dataset/'manifest.json').read_text(encoding='utf8'));a.CATEGORIES=tuple(m['categories'])
    n=len(m['trials']);si=[0] if noise=='clean' else [1,2,3]
    y=np.tile([v['category_index'] for v in m['trials']],len(si));fold=np.tile([v['fold'] for v in m['trials']],len(si));w=np.ones(len(y))/len(si)
    tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
    name=conditions()[ci]['name'];path=OUT/dataset/'cv'/noise/'mvpa'/f'{name}_t{t}_fold{f}.pckl'
    if path.exists() and not probe:return dict(skipped=True,path=str(path))
    x=np.load(OUT/dataset/'features.npy',mmap_mode='r')
    def take(rows):return np.asarray(x[ci,np.asarray(si)[rows//n],rows%n,t,:],dtype=np.float32)
    start=time.time();head,attempts=a.fit_svm(take(tr),y[tr],w[tr])
    if probe:
        old=load(path)['head'];z=take(te)
        np.testing.assert_allclose(head[0].mean_,old[0].mean_,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(head[0].var_,old[0].var_,atol=1e-12,rtol=1e-12)
        repeat,_=a.fit_svm(take(tr),y[tr],w[tr])
        with a.threadpool_limits(limits=1):s=head.decision_function(z);r=old.decision_function(z);again=repeat.decision_function(z)
        repeat_error=float(np.max(np.abs(s-again)));assert repeat_error<1e-10,repeat_error
        error=float(np.max(np.abs(s-r)))
        assert np.array_equal(s.argmax(-1),r.argmax(-1))
        new_auc,_=a.auc_macro(y[te],s);old_auc,_=a.auc_macro(y[te],r)
        return dict(probe=True,path=str(path),max_score_error=error,predictions_identical=True,auc_delta=new_auc-old_auc,isolated_repeat_score_error=repeat_error,training_scaler_identical=True,elapsed=time.time()-start,note='Existing fits used concurrent threads; liblinear approximate solutions need not be bit-identical after process isolation. C, tolerance, cap, seed, weights and data are unchanged.')
    save(path,dict(head=head,attempts=attempts,train_rows=tr,test_rows=te,sample_weight_sum=float(w[tr].sum()),source_sha=sha(a.__file__),execution_driver_sha=sha(__file__)))
    return dict(path=str(path),elapsed=time.time()-start,pid=os.getpid())

def representation_job(dataset,layer):
    from analysis import native_categories_20261029 as a
    layer_index=a.LAYERS.index(layer);dest=WORK/'representation'/layer;dest.mkdir(parents=True,exist_ok=True)
    m=json.loads((OUT/dataset/'manifest.json').read_text(encoding='utf8'))
    # The frozen method sees exactly the requested original layer in position zero.
    activation=np.load(WORK/'activation.npy',mmap_mode='r')[:,:,:,:,[layer_index],:]
    legacy=np.load(WORK/'legacy.npy',mmap_mode='r')[:,:,:,:,[layer_index],:]
    data=dict(manifest=m,trials=m['trials'],activation=activation,legacy=legacy)
    a.dataset_data=lambda _:data;a.LAYERS=(layer,);a.DOC=dest
    atomic=a.atomic_npz
    def routed(path,**values):
        target=dest/'bootstrap_weights.npz' if Path(path).name=='bootstrap_weights.npz' else path
        return atomic(target,**values)
    a.atomic_npz=routed
    started=time.time()
    print('REPRESENTATION WORKER',layer,os.getpid(),flush=True)
    with a.threadpool_limits(limits=4):a.representation(dataset)
    return dict(layer=layer,pid=os.getpid(),elapsed=time.time()-started)

def plot_job(task):
    dataset,kind=task
    import analysis.report_native_categories_20261029 as r
    if not hasattr(r,'_parallel_original_figures_source'):
        r._parallel_original_figures_source=inspect.getsource(r.figures)
    r.KINDS=('mvpa','t9_mode_specific','t9_clean_reference')
    s=r.style;s.SAVED=[];s.AXES=[];s.DEST=DOC/'figures';s.DOC=DOC;s.OUT=OUT
    s.DEST.mkdir(parents=True,exist_ok=True);started=time.time()
    if kind in r.KINDS:
        # Partition the unchanged plotting function before its representation section.
        source=r._parallel_original_figures_source
        source=source[:source.index('    activation=')]
        source=source.replace("for dataset in ('imagenet','things'):",f"for dataset in ({dataset!r},):")
        r.KINDS=(kind,);exec(compile(source,str(Path(__file__))+'::readout_partition','exec'),r.__dict__)
        r.figures()
    else:
        noise=kind.removeprefix('representation_');s.DS=(dataset,);s.NS=(noise,)
        activation={dataset:r.read(DOC/dataset/'activation_timeline.csv')}
        geometry={dataset:r.read(DOC/dataset/'geometry_timeline.csv')}
        effects={dataset:r.read(DOC/dataset/'representation_effects.csv')}
        s.representation_figures(activation,geometry,effects)
    dest=WORK/'figures';dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'{dataset}_{kind}.json';path.write_text(json.dumps(dict(figures=s.SAVED,axes=s.AXES),indent=2),encoding='utf8')
    return dict(task=task,manifest=str(path),figures=len(s.SAVED),elapsed=time.time()-started,pid=os.getpid())

def state_save(state):
    (OUT/'pipeline_status.json').write_text(json.dumps(state,indent=2),encoding='utf8')

def main():
    verify_freeze();WORK.mkdir(parents=True,exist_ok=True)
    state=json.loads((OUT/'pipeline_status.json').read_text(encoding='utf8'))
    state.update(active='things decoding',child_pid=os.getpid(),execution_supplement=str(Path(__file__)),parallel_workers=8,error=None)
    state_save(state);audit=dict(started=time.time(),driver_sha=sha(__file__),frozen_sources_unchanged=True,model_workers=8,representation_workers=4,representation_threads_per_worker=4,plot_workers=6,probes=[],models=[],representation=[],plots=[])
    def checkpoint():(WORK/'execution_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    from analysis import native_categories_20261029 as a
    dataset='things';m=json.loads((OUT/dataset/'manifest.json').read_text(encoding='utf8'));a.CATEGORIES=tuple(m['categories'])
    tasks=[]
    for noise in ('clean','noisy'):
        for ci,c in enumerate(conditions()):
            for f in range(5):
                for t in range(15):
                    path=OUT/dataset/'cv'/noise/'mvpa'/f"{c['name']}_t{t}_fold{f}.pckl"
                    if not path.exists():tasks.append((dataset,noise,ci,f,t,False))
    audit['missing_models_at_resume']=len(tasks);checkpoint()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for v in pool.map(model_job,[(dataset,'clean',0,0,0,True),(dataset,'noisy',4,0,0,True)]):
            audit['probes'].append(v);checkpoint();print('SERIAL/PARALLEL PROBE',v,flush=True)
        futures=[pool.submit(model_job,t) for t in tasks]
        for f in as_completed(futures):
            v=f.result();audit['models'].append(v);checkpoint();print('MODEL COMPLETE',len(audit['models']),'/',len(tasks),v,flush=True)
    features=np.load(OUT/dataset/'features.npy',mmap_mode='r');n=len(m['trials'])
    for noise,si in [('clean',[0]),('noisy',[1,2,3])]:
        y=np.tile([t['category_index'] for t in m['trials']],len(si));fold=np.tile([t['fold'] for t in m['trials']],len(si))
        for ci,c in enumerate(conditions()):
            if (OUT/dataset/'cv'/noise/'mvpa'/f"{c['name']}.npz").exists():continue
            x,w=a.weighted_rows(features[ci],si)
            with a.threadpool_limits(limits=1):a.cv_fit(dataset,noise,'mvpa',c['name'],x,y,fold,w)
            del x
    del features
    # Original validation and summary code, with all completed original heads reused.
    data=a.dataset_data(dataset)
    with a.threadpool_limits(limits=4):a.summarize_decoding(dataset,data)
    if 'things decoding' not in state['completed']:state['completed'].append('things decoding')
    state.update(active='things representation',parallel_workers=4,threads_per_worker=4);state_save(state)
    np.save(WORK/'activation.npy',data['activation']);np.save(WORK/'legacy.npy',data['legacy']);del data
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(representation_job,dataset,layer) for layer in ('conv4','conv5','fc1','fc2')]
        for f in as_completed(futures):
            audit['representation'].append(f.result());checkpoint();print('LAYER COMPLETE',audit['representation'][-1],flush=True)
    # Merge disjoint rows in the original frozen layer order.
    from analysis.prepare_native_20261029 import read,write
    names=('geometry_timeline.csv','geometry_fold_diagnostics.csv','activation_timeline.csv','representation_effects.csv','representation_window_effects.csv')
    for name in names:
        rows=[]
        for layer in ('conv4','conv5','fc1','fc2'):rows.extend(read(WORK/'representation'/layer/dataset/name))
        write(DOC/dataset/name,rows)
    for layer in ('conv5','fc1','fc2'):
        with np.load(WORK/'representation/conv4/bootstrap_weights.npz') as x,np.load(WORK/'representation'/layer/'bootstrap_weights.npz') as y:
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    shutil.copyfile(WORK/'representation/conv4/bootstrap_weights.npz',OUT/dataset/'bootstrap_weights.npz')
    if 'things representation' not in state['completed']:state['completed'].append('things representation')
    state.update(active='report audit',parallel_workers=1);state_save(state)
    import analysis.report_native_categories_20261029 as report
    checked=report.audit();audit['final_data_audit_passed']=checked['passed'];checkpoint()
    state.update(active='parallel figures',parallel_workers=6);state_save(state)
    jobs=[(d,k) for d in ('imagenet','things') for k in ('mvpa','t9_mode_specific','t9_clean_reference','representation_clean','representation_noisy')]
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(plot_job,t) for t in jobs]
        for f in as_completed(futures):
            audit['plots'].append(f.result());checkpoint();print('PLOT GROUP COMPLETE',audit['plots'][-1],flush=True)
    allfig=[];axes=[]
    for v in audit['plots']:
        d=json.loads(Path(v['manifest']).read_text(encoding='utf8'));allfig.extend(d['figures']);axes.extend(d['axes'])
    assert len({v['name'] for v in allfig})==len(allfig)==156
    for v in allfig:
        for ext in ('png','svg'):assert (DOC/'figures'/f"{v['name']}.{ext}").stat().st_size>0
    (DOC/'figure_manifest.json').write_text(json.dumps(dict(figures=allfig,axes=axes),indent=2),encoding='utf8')
    report.report(checked)
    audit.update(finished=time.time(),all_figures=len(allfig));checkpoint()
    state['completed'].append('report');state.update(active=None,finished=time.time(),parallel_workers=0);state_save(state)
    print('PIPELINE COMPLETE',DOC/'实验结果.md',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        path=OUT/'pipeline_status.json'
        state=json.loads(path.read_text(encoding='utf8'));state.update(error=dict(type=type(e).__name__,message=str(e)));state_save(state)
        raise
