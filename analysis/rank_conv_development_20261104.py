import sys,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.conv_development_20261104 import OUT,candidates
D=ROOT/'docs/conv_development_20261104';rows=[]
for ds in ('imagenet','things'):
 for cfg in candidates():
  p=OUT/ds/cfg['id']/'metrics.csv'
  if not p.exists():continue
  x=pd.read_csv(p)
  for seed in (0,2921):
   for window,ts in [('early',range(5)),('late',range(10,15))]:
    y=x[(x.seed==seed)&x.t.isin(ts)];fc=y[y.layer=='fc2'];cv=y[y.layer=='conv4'];r=dict(dataset=ds,seed=seed,window=window,**cfg,accuracy=fc.accuracy.mean(),mean=cv['mean'].mean(),zero_fraction=cv.zero_fraction.mean())
    for layer,z in [('conv4',cv),('fc2',fc)]:
     q=z.groupby('identity').mean(numeric_only=True)
     for metric in ('mean','fisher','within','between')+(('accuracy',) if layer=='fc2' else ()):
      r[layer+'_'+metric+'_priming']=q.loc['exact',metric]-q.loc['different_category',metric]
    rows.append(r)
a=pd.DataFrame(rows);out=[]
for _,r in a.iterrows():
 b=a[(a.dataset==r.dataset)&(a.seed==r.seed)&(a.window==r.window)&(a.id=='c000')]
 if b.empty:continue
 b=b.iloc[0];r=r.copy();r['accuracy_loss_pp']=100*(b.accuracy-r.accuracy);r['mean_change_pct']=100*(r['mean']/b['mean']-1);r['density_change_pct']=100*((1-r.zero_fraction)/(1-b.zero_fraction)-1)
 r['fidelity_pass']=r.accuracy_loss_pp<=(2 if r.seed==0 else 3)
 r['activity_pass']=(r.mean_change_pct<=-5) or (r.density_change_pct<=-10)
 r['repetition_direction_pass']=(r.conv4_mean_priming<0) and (r.fc2_fisher_priming>0) and (r.fc2_accuracy_priming>0)
 r['continue_candidate']=r.fidelity_pass and r.activity_pass and r.repetition_direction_pass;out.append(r)
z=pd.DataFrame(out);z.to_csv(D/'candidate_comparison.csv',index=False)
s=z[z.window=='early'].groupby('id')[['fidelity_pass','activity_pass','repetition_direction_pass','continue_candidate']].sum();s.to_csv(D/'candidate_screen_counts.csv')
(D/'selection_notes.md').write_text('开发筛选：candidate_comparison.csv保留全部结果；通过数量不是显著性检验。每候选早期应有两个数据集×两输入共4格。仅零值稀疏度；各门槛单独报告，不以加权总分掩盖读出损失。下一步需逐内容绝对ACC及仅抑制对照复核，再冻结确认。',encoding='utf8')
print(s.to_string())
