"""Descriptive reports for diagonal temporal MVPA; no CV-naive significance tests."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs/sliding_mvpa_20261013';O=ROOT/'Hebbian-CNN/Log/sliding_mvpa_20261013'
NAMES={'none_random':'No memory: random','none_feedforward':'No memory: feedforward','carry_exact':'Exact image','carry_same_synset':'Same concept, different image','carry_same_category':'Same superclass','carry_different_category':'Different superclass','carry_neutral':'Neutral preview'}
COLORS=['#777777','#6A4C93','#2176AE','#E69F00','#B85C9E','#009E73','#875C35']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
fig,axes=plt.subplots(1,2,figsize=(15,6),sharey=True)
summary=[];checks={};inputhashes={}
for ax,ds,old in zip(axes,['imagenet','things'],['supplementary_20261012','things_readout_20261012']):
    a=pd.read_csv(D/f'{ds}_timeline.csv');audit=json.loads((D/f'{ds}_audit.json').read_text());spec=json.loads((O/ds/'specification.json').read_text())
    assert digest(ROOT/'analysis/sliding_mvpa_20261013.py')==spec['source_sha']
    raw=ROOT/'Hebbian-CNN/Log'/('readout_init_20261012' if ds=='imagenet' else 'things_readout_20261012')
    assert digest(raw/'manifest.json')==spec['manifest_sha']
    assert digest(raw/'readouts.pckl')==spec['frozen_readout_sha']
    inputhashes[ds]={p.name:digest(p) for p in sorted((raw/'test').glob('*.pckl'))}
    iterations=[];seconds=[]
    y=np.array(spec['labels']);classes=len(np.unique(y))
    for c in NAMES:
        for t in range(15):
            with np.load(O/ds/f'{c}_t{t:02d}.npz') as z:
                assert z['scores'].shape==(len(y),classes) and np.isfinite(z['scores']).all()
                assert len(z['iterations'])==5 and np.all(z['iterations']<200000)
                # Any exact cap would indicate a convergence problem, irrespective of warning capture.
                assert not np.any(np.isin(z['iterations'],[50000,200000]))
                acc=100*np.mean(z['scores'].argmax(1)==y)
                assert abs(acc-a[(a.condition==c)&(a.timestep==t)].accuracy_pct.iloc[0])<1e-8
                iterations.extend(z['iterations'].tolist());seconds.append(float(z['seconds']))
    checks[ds]=dict(**audit,max_svm_iterations=max(iterations),completed_score_files=105,source_and_frozen_readout_unchanged=True,all_score_files_recomputed=True)
    for (c,label),color in zip(NAMES.items(),COLORS):
        v=a[a.condition==c].sort_values('timestep');ax.plot(v.timestep,v.accuracy_pct,label=label,color=color,marker='o',markersize=3,linewidth=1.8)
    ax.axhline(100/classes,color='#999999',ls=':',label='Chance' if ds=='imagenet' else None)
    ax.set(title=f'{ds.upper()} ({classes} classes, N={audit["images"]})',xlabel='Target timestep (train t, test t)',ylim=(0,100),xticks=range(0,15,2));ax.grid(alpha=.18)
    frozen=pd.read_csv(ROOT/'docs'/old/'test_timeline.csv')
    for c in NAMES:
        row={'dataset':ds,'condition':c}
        for t in [0,1,3,9,14]:row[f't{t}']=float(a[(a.condition==c)&(a.timestep==t)].accuracy_pct.iloc[0])
        summary.append(row)
    comparison=[]
    for h in ['matched_t9','pooled']:
        for c in NAMES:
            for t in range(15):
                v=float(a[(a.condition==c)&(a.timestep==t)].accuracy_pct.iloc[0]);f=float(frozen[(frozen.readout==h)&(frozen.condition==c)&(frozen.timestep==t)].accuracy_pct.iloc[0])
                comparison.append(dict(dataset=ds,condition=c,timestep=t,frozen_head=h,sliding_oof_pct=v,frozen_test_pct=f,difference_pp=v-f))
    pd.DataFrame(comparison).to_csv(D/f'{ds}_vs_frozen_descriptive.csv',index=False,encoding='utf-8-sig')
axes[0].set_ylabel('Out-of-fold accuracy (%)')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=10)
fig.suptitle('Condition-specific temporal decoding: 5-fold concept-grouped CV',fontsize=15)
fig.tight_layout(rect=(0,.14,1,.95));fig.savefig(D/'sliding_accuracy.png',dpi=180);plt.close(fig)
pd.DataFrame(summary).to_csv(D/'selected_times.csv',index=False,encoding='utf-8-sig')
(D/'audit.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
(O/'input_hashes.json').write_text(json.dumps(inputhashes,indent=2),encoding='utf8')
print(pd.DataFrame(summary).round(2).to_string(index=False))
for ds in ['imagenet','things']:
    a=pd.read_csv(D/f'{ds}_contrasts_descriptive.csv')
    print(ds,a[a.timestep.isin([0,3,9,14])].round(3).to_string(index=False))
print('AUDIT PASSED',checks)
