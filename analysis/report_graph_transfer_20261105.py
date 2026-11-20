from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=Path('E:/Predify_Experiments_20261105/conv_graph_transfer_20261105');DOC=ROOT/'docs/conv_graph_transfer_20261105'
rows=[]
for ds in ('imagenet','things'):
 for case in ('gate_rep','gate_prd','gate_none','multiply_rep','multiply_prd','multiply_none','clip'):
  p=OUT/ds/case/'metrics.csv' if case!='clip' else Path('E:/Predify_Experiments_20261105/conv_development_20261104')/ds/'c000/metrics.csv'
  assert p.exists(),p
  a=pd.read_csv(p);assert len(a)==240
  a['dataset']=ds;a['case']=case;a['window']=np.where(a.t<5,'early',np.where(a.t<10,'middle','late'));rows.append(a)
x=pd.concat(rows,ignore_index=True);x.to_csv(DOC/'all_metrics.csv',index=False)
summary=[]
for (ds,case,seed,win),a in x.groupby(['dataset','case','seed','window']):
 r=dict(dataset=ds,case=case,seed=seed,window=win)
 for layer in ('conv4','conv5','fc1','fc2'):
  v=a[a.layer==layer];q=v.groupby('identity').mean(numeric_only=True)
  for key in ('mean','zero_fraction','fisher','within','between')+(('accuracy',) if layer=='fc2' else ()):
   r[layer+'_'+key]=v[key].mean();r[layer+'_'+key+'_priming']=q.loc['exact',key]-q.loc['different_category',key]
 summary.append(r)
s=pd.DataFrame(summary);s.to_csv(DOC/'window_metrics.csv',index=False);comparisons=[]
for _,r in s[s['case']!='clip'].iterrows():
 for base in ['clip',r['case'].split('_')[0]+'_none',r['case'].split('_')[0]+'_rep']:
  b=s[(s.dataset==r.dataset)&(s['case']==base)&(s.seed==r.seed)&(s.window==r.window)].iloc[0]
  z=dict(dataset=r.dataset,case=r['case'],baseline=base,seed=int(r.seed),window=r.window)
  for layer in ('conv4','conv5','fc1','fc2'):
   for metric in ('mean','fisher','within','between'):
    k=layer+'_'+metric;z[k+'_pct']=100*(r[k]/b[k]-1);z[k+'_priming_change']=r[k+'_priming']-b[k+'_priming']
   z[layer+'_density_pct']=100*((1-r[layer+'_zero_fraction'])/(1-b[layer+'_zero_fraction'])-1)
  z['accuracy_delta_pp']=100*(r.fc2_accuracy-b.fc2_accuracy);z['priming_accuracy_change_pp']=100*(r.fc2_accuracy_priming-b.fc2_accuracy_priming)
  comparisons.append(z)
cs=pd.DataFrame(comparisons);cs.to_csv(DOC/'paired_comparisons.csv',index=False)
print('EARLY vs clip');print(cs[(cs.window=='early')&(cs.baseline=='clip')][['dataset','case','seed','accuracy_delta_pp','conv4_mean_pct','conv4_density_pct','conv4_fisher_pct','fc2_fisher_pct']].round(5).to_string(index=False))
print('PRIMING EARLY');print(s[(s.window=='early')&(s['case'].str.contains('prd|rep|none'))][['dataset','case','seed','conv4_mean_priming','conv4_within_priming','fc2_mean_priming','fc2_within_priming','fc2_fisher_priming','fc2_accuracy_priming']].round(7).to_string(index=False))
print('PRD vs REP');print(cs[(cs.window=='early')&cs['case'].str.endswith('prd')&cs.baseline.str.endswith('rep')][['dataset','case','seed','accuracy_delta_pp','conv4_mean_pct','conv4_density_pct','conv4_fisher_pct','fc2_fisher_pct','priming_accuracy_change_pp']].round(5).to_string(index=False))
def table(a):return '| '+' | '.join(a.columns)+' |\n| '+' | '.join(['---']*len(a.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in a.itertuples(index=False,name=None))
report=['# 预测建图→表征调制配对实验','', '门控/乘性×rep4图/prd5图/无Hebbian，共六组；全部作用于rep4。两数据集16/20开发图，各clean/noisy×exact/different。FC Hebbian关闭。应用尺度统一为训练rep4 RMS；gamma=.25，rho=1/3，lambda=.5，alpha=.5，K=3。门控a=.25,b=.1。连接图沿用同训练折、同t9的冻结估计；原始权重和数据划分不变。四组对照复用完全匹配的历史轨迹，新增两组真实递归轨迹；无权重重训或基于结果调参。','', '以下均为小样本开发描述，不能作为独立验证。固定参考头只用内层训练概念；未拟合适配读出。密度指精确非零比例；Fisher与类内/类间散度在单位范数特征上计算。','', '## 早期相对仅截断','',table(cs[(cs.window=='early')&(cs.baseline=='clip')][['dataset','case','seed','accuracy_delta_pp','conv4_mean_pct','conv4_density_pct','conv4_fisher_pct','fc2_fisher_pct']].round(5)),'','## 图来源的直接比较（prd图减rep图）','',table(cs[(cs.window=='early')&cs['case'].str.endswith('prd')&cs.baseline.str.endswith('rep')][['dataset','case','seed','accuracy_delta_pp','conv4_mean_pct','conv4_density_pct','conv4_fisher_pct','fc2_fisher_pct','priming_accuracy_change_pp']].round(5)),'','## 早期Exact−Different','',table(s[s.window=='early'][['dataset','case','seed','conv4_mean_priming','fc2_mean_priming','fc2_within_priming','fc2_fisher_priming','fc2_accuracy_priming']].round(7))]
aggregate=cs[(cs.window=='early')&(cs.baseline=='clip')].groupby('case')[['accuracy_delta_pp','conv4_mean_pct','conv4_density_pct']].mean().round(4).reset_index()
conclusion=['## 结论与筛选','',
'本次全部完成。新增288条完整15步目标轨迹，复用576条参数匹配轨迹；训练/开发划分核对一致，新增轨迹均通过有限数值检查，旧FC/卷积Hebbian关闭。新轨迹保存于E盘；复用组的provenance.json指向保留的原始轨迹。', '',
'下表为目标t0–4、两数据集×clean/noisy四格等权平均；准确率为固定FC2参考分类头相对仅截断的百分点变化。非零密度变化是相对百分比，不是零比例的百分点。时间步与噪声条件不是独立样本。', '',table(aggregate),'',
'1. 门控prd图具有可继续考察的读出/稀疏折中，但没有建立图来源优势：相对rep图，THINGS准确率提高0.5/1.0个百分点，ImageNet为0/−0.625个百分点；平均激活提高0.31%–0.44%，非零密度提高0.90%–1.59%。无Hebbian门控仍给出更强的平均激活下降与稀疏化。',
'2. 乘性prd图与rep图的平均读出损失相同，但prd图激活更高、稀疏化更弱；FC2 Fisher相对rep图在四格全部略低。暂不优先推进这一替换。',
'3. 没有稳定repetition suppression：早期六组在四格中conv4和FC2的Exact−Different平均激活均为正。整体激活下降不能解释为重复抑制。晚期效应非常小，不能用单个符号变化认定成功。',
'4. sharpening证据混合：早期FC2 Fisher的Exact−Different为正，但单位范数类内散度也均增加，尚未形成更紧凑的类内表征。门控prd图相对无Hebbian的FC2 Fisher priming差值在四格均下降，类内散度priming差值在四格均增加。',
'5. 门控prd图的早期Exact−Different读出差为ImageNet 0/0个百分点、THINGS +5/+7个百分点；相对无Hebbian的额外改善为0/+1.25、0/+3个百分点。作用集中在THINGS，晚期六组的读出差全部为0，不能称为跨数据集稳定的重复锐化。乘性prd图的读出priming差为ImageNet −1.25/−2.5、THINGS +2/+3个百分点，也不稳定。',
'6. 建议保留无Hebbian门控为强对照，门控prd图仅作为待复验候选；当前结果不支持扩大乘性prd图实验，也不支持宣称已同时实现读出保真、重复抑制和重复锐化。下一步应针对重复特异机制提出修改，再预先确定参数与独立验证集。本轮未自动开展新搜索。', '',
'图来源对比是整套建图结果的对比：相同候选连接掩码、正相关估计和归一化规则下，有效正边数仍不同。ImageNet/THINGS的柱内正边Jaccard约0.667/0.664，柱间约0.474/0.448；不能把差异完全归因于预测的语义纯净度。', '',
'窗口：early=t0–4，middle=t5–9，late=t10–14。全部窗口和逐步数据见window_metrics.csv、paired_comparisons.csv、all_metrics.csv。读出保真在这里用固定头准确率衡量，不等同于逐样本logit一致。', '']
report[2:2]=conclusion
(DOC/'配对实验评估.md').write_text('\n'.join(report),encoding='utf8');(DOC/'analysis_complete.json').write_text(json.dumps(dict(complete=True,metric_rows=len(x),window_rows=len(s),comparison_rows=len(cs)),indent=2))
