from pathlib import Path
import pandas as pd,numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def markdown(self,index=True,**kwargs):
 a=self.reset_index() if index else self
 return '| '+' | '.join(map(str,a.columns))+' |\n| '+' | '.join(['---']*len(a.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in a.itertuples(index=False,name=None))
pd.DataFrame.to_markdown=markdown
D=Path('docs/conv_column_nonnegative_20261104');g=pd.read_csv(D/'focused_geometry.csv');r=pd.read_csv(D/'readout_per_target.csv')
assert len(g)==2880
for a in (g,r):a['window']=np.where(a.t<5,'early',np.where(a.t<10,'middle','late'))
metrics=['mean','raw_within','raw_between','normalized_within','normalized_between','normalized_fisher','within_cosine','zero_fraction']
s=g.groupby(['dataset','seed','window','layer','mode'])[metrics].mean();comparisons=[]
for (ds,seed,win,layer),a in s.groupby(level=[0,1,2,3]):
 f=a.xs('full',level='mode').iloc[0]
 for base in ('clip','inhibition'):
  b=a.xs(base,level='mode').iloc[0];row=dict(dataset=ds,seed=seed,window=win,layer=layer,baseline=base)
  for k in metrics:row[k+'_delta']=f[k]-b[k];row[k+'_pct']=100*(f[k]/b[k]-1) if b[k]!=0 else np.nan
  comparisons.append(row)
c=pd.DataFrame(comparisons);c.to_csv(D/'geometry_contrasts.csv',index=False)
acc=r.groupby(['dataset','seed','window','mode']).correct.mean().unstack('mode')*100;acc['full_minus_clip']=acc.full-acc['clip'];acc['full_minus_inhibition']=acc.full-acc.inhibition
print('ACC\n',acc[['clip','inhibition','full','full_minus_clip','full_minus_inhibition']].round(3).to_string())
print('GEOMETRY early\n',c[c.window=='early'][['dataset','seed','layer','baseline','mean_pct','normalized_within_pct','normalized_between_pct','normalized_fisher_pct','within_cosine_delta','zero_fraction_delta']].round(4).to_string(index=False))
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
for ax,(ds,seed) in zip(axes.flat,[(d,s) for d in ('imagenet','things') for s in (0,2921)]):
 for mode,label,col in [('clip','Clip only','#555555'),('inhibition','Inhibition only','#4595d0'),('full','Hebbian + inhibition','#d25930')]:
  z=r[(r.dataset==ds)&(r.seed==seed)&(r['mode']==mode)].groupby('t').correct.mean()*100;ax.plot(z.index,z.values,'o-',ms=3,label=label,color=col)
 ax.set_title(ds+' / '+('clean' if seed==0 else 'noisy'));ax.set_ylabel('Frozen-reference accuracy (%)');ax.set_xlabel('Target step');ax.grid(alpha=.2);ax.axvspan(0,4,alpha=.05,color='black')
axes[0,0].legend(fontsize=8);fig.suptitle('Pilot: mean over four preview contents; no confidence intervals');fig.tight_layout();fig.savefig(D/'readout_comparison.png',dpi=180);plt.close(fig)
report=['# 卷积 Hebbian＋侧向抑制 pilot 评估','', '提取清单核对：ImageNet 1152、THINGS 1440 条，无缺失；本次补做三主条件的2880行完整CHW几何统计及所有九组的逐目标读出。原自动分析未完成，不复用其完成状态。','', '比较定义：full=柱内＋柱间Hebbian＋除法抑制；clip=仅截断；inhibition=仅截断＋同强度除法抑制。各组FC Hebbian关闭。四预览内容等权；early=t0–t4，late=t10–t14。下表为冻结clean t9参考头，不是重新拟合的可解码性。','', '## 分类读出（百分比；差值为百分点）','',acc[['clip','inhibition','full','full_minus_clip','full_minus_inhibition']].round(3).to_markdown(),'', '## 早期几何相对变化（%）','',c[c.window=='early'][['dataset','seed','layer','baseline','mean_pct','normalized_within_pct','normalized_between_pct','normalized_fisher_pct']].round(3).to_markdown(index=False),'','## 解释边界','','每数据集仅一个留出折、每类2概念×2图，ImageNet16图/THINGS20图，noisy仅一种实现。时间、预览及噪声重复不是独立样本；不作显著性宣称。几何为测试小样本描述，没有置信区间。冻结参考读出的变化包含分布迁移；适配读出尚未拟合。PCoder4直接受干预，PCoder5/FC1/FC2为下游效应。','', '归一化类内散度和余弦改善不能单独证明类别分离，因为类间散度也可能收缩。无硬门控，柱间连接稀疏不代表响应必然更稀疏。', '', '文件：geometry_contrasts.csv、focused_geometry.csv、readout_windows.csv、readout_content_windows.csv、priming_accuracy_pp.csv、readout_per_target.csv。']
(D/'效果评估.md').write_text('\n'.join(report),encoding='utf8')
print('zero_vectors',g.zero_vectors.max(),'nonfinite',not np.isfinite(g.select_dtypes('number')).all().all())
