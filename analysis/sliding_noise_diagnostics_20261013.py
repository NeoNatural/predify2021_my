from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs/sliding_mvpa_20261013'
O=ROOT/'Hebbian-CNN/Log/sliding_mvpa_20261013'
controls={'none_feedforward':'Feedforward no memory','carry_same_synset':'Same concept','carry_same_category':'Same superclass','carry_different_category':'Different superclass','carry_neutral':'Neutral preview'}
rows=[];cats=[]
fig,axes=plt.subplots(2,5,figsize=(19,8),sharex=True,sharey=True)
for di,ds in enumerate(['imagenet','things']):
 spec=json.loads((O/ds/'specification.json').read_text())
 y=np.array(spec['labels']);keys=np.array(spec['keys']);seeds=np.unique(keys[:,1])
 names=['animals','clothing','food','plants','vehicles'] if ds=='imagenet' else ['animals','clothing','food','vehicles']
 pred={c:np.stack([np.load(O/ds/f'{c}_t{t:02d}.npz')['scores'].argmax(1) for t in range(15)],axis=1) for c in spec['conditions']}
 for c in pred:
  for cat in range(len(names)):
   for t in range(15):
    m=y==cat;cats.append(dict(dataset=ds,condition=c,category=names[cat],timestep=t,images=len(np.unique(keys[m,0])),accuracy_pct=100*float(np.mean(pred[c][m,t]==y[m]))))
 for ci,(c,label) in enumerate(controls.items()):
  delta=100*((pred['carry_exact']==y[:,None]).astype(float)-(pred[c]==y[:,None]).astype(float))
  values=np.stack([delta[keys[:,1]==seed].mean(0) for seed in seeds])
  for t in range(15):
   row=dict(dataset=ds,control=c,timestep=t,delta_pp=float(delta[:,t].mean()),positive_noises=int(sum(values[:,t]>0)),negative_noises=int(sum(values[:,t]<0)),zero_noises=int(sum(values[:,t]==0)))
   for j,seed in enumerate(seeds):row[f'noise_{j+1}_delta_pp']=float(values[j,t])
   rows.append(row)
  ax=axes[di,ci]
  for j in range(3):ax.plot(range(15),values[j],alpha=.35,lw=1,label=f'Noise {j+1}')
  ax.plot(range(15),delta.mean(0),color='#152C3A',lw=2.3,label='Mean')
  ax.axhline(0,color='gray',ls=':',lw=1);ax.grid(alpha=.15)
  ax.set_title(label);ax.set_xticks([0,3,6,9,12,14])
  if ci==0:ax.set_ylabel(ds.upper()+'\nExact minus control (pp)')
  if di==1:ax.set_xlabel('Target timestep')
fig.suptitle('Noise = 0.5: paired temporal contrasts across three noise realizations',fontsize=16)
h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4)
fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(D/'noise_temporal_contrasts.png',dpi=170)
pd.DataFrame(rows).to_csv(D/'noise_contrast_stability.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(cats).to_csv(D/'category_timelines.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(rows).query('timestep == 1').round(2).to_string(index=False))
