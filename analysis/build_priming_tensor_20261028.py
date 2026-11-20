"""Build a linked high-dimensional visualization from fixed result tables only."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/tripartite_filtered_cv_20261027'
DEST=Path('C:/Users/liang/.codex/visualizations/2026/10/09/01a11e35-009a-7660-893d-f61aebd2ce0e/priming-tensor-explorer.html')
METRICS=[
    ('output_mean_effect_pct','Output激活变化','Output激活','%'),
    ('legacy_x_tmp_mean_effect_pct','Legacy x_tmp激活变化','Legacy激活','%'),
    ('normalized_trace_effect_pct','Normalized trace变化','Normalized trace','%'),
    ('within_cosine_delta_x1e3','Within-class cosine差值','Within cosine','×10⁻³'),
    ('normalized_fisher_effect_pct','Normalized Fisher变化','Normalized Fisher','%'),
    ('normalized_between_scatter_delta','Normalized between scatter差值','Between scatter','Δ'),
    ('raw_trace_variance_delta','Raw trace variance差值（诊断）','Raw trace variance','Δ'),
]
DATASETS=('imagenet','things');NOISES=('clean','noisy');CARRIERS=('hebb','pcoder','joint');IDENTITIES=('exact','same_concept','same_category');LAYERS=('conv4','conv5','fc1','fc2');WINDOWS=('early','middle','late','full')

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    contexts=[(d,n,c,i) for d in DATASETS for n in NOISES for c in CARRIERS for i in IDENTITIES]
    rep=np.full((7,36,4,15,3),np.nan);windows=np.full((7,36,4,4,3),np.nan)
    readout=np.full((2,12,3,15,4),np.nan);rw=np.full((2,12,3,4,4),np.nan);sources={}
    for d in DATASETS:
        for filename,target,timekey in [('representation_effects.csv',rep,'timestep'),('representation_window_effects.csv',windows,'window')]:
            path=DOC/d/filename;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            for r in read(path):
                names=[m[0] for m in METRICS]
                if r['metric'] not in names or (timekey=='window' and r['window'] not in WINDOWS):continue
                mi=names.index(r['metric']);row=contexts.index((d,r['noise'],r['carrier'],r['identity']));layer=LAYERS.index(r['layer']);t=int(r['timestep']) if timekey=='timestep' else WINDOWS.index(r['window'])
                target[mi,row,layer,t]=[float(r[k]) for k in ('value','low','high')]
        for filename,target,timekey in [('cv_content_contrasts.csv',readout,'timestep'),('cv_window_contrasts.csv',rw,'window')]:
            path=DOC/d/filename;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            for r in read(path):
                if timekey=='window' and r['window'] not in WINDOWS:continue
                mi=('t9','mvpa').index(r['readout']);ci=(DATASETS.index(d)*2+NOISES.index(r['noise']))*3+CARRIERS.index(r['carrier']);i=IDENTITIES.index(r['identity']);t=int(r['timestep']) if timekey=='timestep' else WINDOWS.index(r['window'])
                target[mi,ci,i,t]=[float(r[k]) for k in ('delta_accuracy_pp','delta_accuracy_fold_sd','delta_roc_auc','delta_auc_fold_sd')]
    assert all(np.isfinite(a).all() for a in (rep,windows,readout,rw))
    assert (rep[...,1]<=rep[...,2]).all()
    def quantize(a):
        if isinstance(a,list):return [quantize(v) for v in a]
        return float(format(a,'.6g'))
    data=dict(metrics=[dict(key=k,name=n,short=s,unit=u) for k,n,s,u in METRICS],rep=quantize(rep.tolist()),windows=quantize(windows.tolist()),readout=quantize(readout.tolist()),readoutWindows=quantize(rw.tolist()))
    text=(ROOT/'analysis/priming_tensor_template.html').read_text(encoding='utf8').replace('__PRIMING_TENSOR_DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')))
    assert len(text.encode('utf8'))<1_000_000
    DEST.parent.mkdir(parents=True,exist_ok=True);DEST.write_text(text,encoding='utf8')
    audit=dict(fragment=str(DEST),bytes=len(text.encode('utf8')),representation_shape=list(rep.shape),window_shape=list(windows.shape),readout_shape=list(readout.shape),rounding_significant_digits=6,
        original_results_unchanged=True,source_hashes=sources,window_intervals='Existing concept-bootstrap window intervals; never averaged per-timepoint CI bounds',rows=contexts)
    (DOC/'high_dimensional_visualization.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('source_hashes','rows')},ensure_ascii=False))

if __name__=='__main__':main()
