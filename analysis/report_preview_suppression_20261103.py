"""Audit and present paired priming effects, independently of the slide deck."""
from pathlib import Path
from collections import defaultdict
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.preview_suppression_comparison_20261103 import DOC,OUT,METHODS,IDENTITIES
from predify_experiments import native_categories_20261029 as native

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
data={d:read(DOC/d/'timecourse.csv') for d in ('imagenet','things')}
windows={d:read(DOC/d/'windows.csv') for d in ('imagenet','things')}
geom=read(ROOT/'docs/hebb_fc_full_20261103/fold_geometry_timeline.csv')
source={(r['dataset'],int(r['fold']),r['method'],r['layer'],r['condition'],int(r['seed']),int(r['timestep'])):float(r['activation']) for r in geom}
tc={d:{(r['noise'],r['carrier'],r['layer'],r['identity'],r['method'],r['contrast'],int(r['timestep'])):float(r['value']) for r in data[d]} for d in data}
win={d:{(r['noise'],r['carrier'],r['layer'],r['identity'],r['method'],r['contrast'],r['window']):r for r in windows[d]} for d in windows}
errors=[]
for d,rows in data.items():
    for r in rows:
        seeds=(0,) if r['noise']=='clean' else (2921,2922,2923);t=int(r['timestep']);carrier=r['carrier'];identity=r['identity'];method=r['method'];layer=r['layer']
        def contrast(method):
            return np.mean([source[d,f,method,layer,carrier+'_'+identity,seed,t]-source[d,f,method,layer,carrier+'_different_category',seed,t] for f in range(5) for seed in seeds])
        expected=contrast(method)
        if r['contrast']=='interaction_vs_off':expected-=contrast('off')
        errors.append(abs(expected-float(r['value'])))
assert len(errors)==7560 and max(errors)<1e-12,(len(errors),max(errors))
for d in windows:
    assert len(windows[d])==504
    for r in windows[d]:
        assert float(r['ci_low'])<=float(r['ci_high']) and np.isfinite([float(r['value']),float(r['ci_low']),float(r['ci_high'])]).all()

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
colors={'exact':'#111111','same_concept':'#52bfe1','same_category':'#267df4'}
labels={'exact':'Exact','same_concept':'Same concept','same_category':'Same category'}
carriers={'hebb':'Hebbian-only','pcoder':'PCoder carry-only','joint':'Joint'}
ctrl={'mean_std':('Hebb-FC','#267df4'),'inhibition_only':('Inhibition','#52bfe1'),'scale_both':('Matched scale','#d66d2c')}
figdir=DOC/'figures';figdir.mkdir(parents=True,exist_ok=True)
manifest=[]
with PdfPages(DOC/'激活差值比较图册.pdf') as pdf:
 for d in ('imagenet','things'):
  for layer in ('fc1','fc2'):
   for noise in ('clean','noisy'):
    fig,axes=plt.subplots(1,3,figsize=(14,5.8),sharey=True);t=np.arange(15)
    allv=[]
    for ax,(carrier,clabel) in zip(axes,carriers.items()):
     for ident in IDENTITIES:
      for method,style in [('off','--'),('mean_std','-')]:
       y=np.array([tc[d][noise,carrier,layer,ident,method,'priming_minus_different',i] for i in t])*1000;allv.extend(y)
       ax.plot(t,y,color=colors[ident],linestyle=style,lw=1.5,label=labels[ident]+' / '+('Off' if method=='off' else 'Hebb-FC'))
     ax.axhline(0,color='#888888',lw=.9);ax.axvspan(-.2,4.5,color='#eeeeee',alpha=.55,zorder=-2);ax.axvspan(9.5,14.2,color='#f5f5f5',alpha=.75,zorder=-2)
     ax.set_title(clabel);ax.set_xlabel('Target timestep');ax.set_xlim(0,14);ax.set_xticks([0,2,4,6,8,10,12,14]);ax.grid(alpha=.16)
    axes[0].set_ylabel('Priming - different category (× 10⁻³ activation)')
    lim=max(max(abs(np.array(allv)))*1.12,1e-6);axes[0].set_ylim(-lim,lim)
    handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,ncol=3,loc='lower center',bbox_to_anchor=(.5,.08),frameon=False,fontsize=9)
    fig.suptitle(f'{d.upper()}  {layer.upper()}  {noise}: preview effect',fontsize=15)
    fig.text(.5,.03,'Solid: Hebb-FC; dashed: Off. Negative means lower activation than different-category. Grey: early / late windows.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.19,1,.92));name=f'{d}_{layer}_{noise}_preview_effect';fig.savefig(figdir/f'{name}.png',dpi=170);fig.savefig(figdir/f'{name}.svg');pdf.savefig(fig);plt.close(fig);manifest.append(name)
    fig,axes=plt.subplots(3,3,figsize=(14,9),sharex=True,sharey=True);allv=[]
    for r,ident in enumerate(IDENTITIES):
     for c,(carrier,clabel) in enumerate(carriers.items()):
      ax=axes[r,c]
      for method,(label,color) in ctrl.items():
       y=np.array([tc[d][noise,carrier,layer,ident,method,'interaction_vs_off',i] for i in t])*1000;allv.extend(y)
       # Matched scale is drawn dashed because its activation matches Hebb-FC by construction.
       ax.plot(t,y,color=color,linestyle='--' if method=='scale_both' else '-',lw=1.6,label=label)
      ax.axhline(0,color='#888888',lw=.8);ax.axvspan(-.2,4.5,color='#eeeeee',alpha=.45,zorder=-2);ax.set_xlim(0,14);ax.grid(alpha=.15)
      if r==0:ax.set_title(clabel)
      if c==0:ax.set_ylabel(labels[ident]+'\nDifference of differences\n(× 10⁻³ activation)')
      if r==2:ax.set_xlabel('Target timestep')
    lim=max(max(abs(np.array(allv)))*1.12,1e-6);axes[0,0].set_ylim(-lim,lim)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='lower center',bbox_to_anchor=(.5,.06),frameon=False)
    fig.suptitle(f'{d.upper()}  {layer.upper()}  {noise}: FC effect on preview contrast',fontsize=15)
    fig.text(.5,.025,'(Priming − different category) under each FC setting, minus the same contrast under Off.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.95));name=f'{d}_{layer}_{noise}_interaction';fig.savefig(figdir/f'{name}.png',dpi=170);fig.savefig(figdir/f'{name}.svg');pdf.savefig(fig);plt.close(fig);manifest.append(name)

def val(r):return float(r['value'])*1000
def ci(r):return f"{val(r):+.3f} [{float(r['ci_low'])*1000:+.3f}, {float(r['ci_high'])*1000:+.3f}]"
lines=['# Priming 相对 different-category 的激活比较','','比较目标是看 priming 是否使目标平均激活低于 non-priming，并进一步看 Hebb-FC 是否改变这一差值。这里的 non-priming 操作定义是 **different-category preview**，它仍然是有预览的对照。','','对每个数据集、概念折、目标图、噪声实现、记忆模式、层和时间点，分别计算：','','- 预览效应：`priming − different-category`。负值表示该 priming 的平均激活更低。','- FC 差值之差：`(priming − different-category)_FC − (priming − different-category)_Off`。负值只表示FC使预览效应更低，不保证预览效应本身为负。','','窗口为 early t0–t4、late t10–t14。以下数值乘以 1000 以方便阅读，单位为平均激活的原始单位 ×10⁻³，**不是百分比或百分点**。方括号为 10,000 次类别×外层折分层的概念配对 bootstrap 边际 95% 区间。','','## 示例：Joint、noisy、exact','','此处三列为三个不同估计，不应混为一谈。全部模式、预览粒度及对照见各数据集的 CSV。','','| 数据集 | 层 | 窗口 | Off 的预览效应 | Hebb-FC 的预览效应 | Hebb-FC 差值之差 |','|---|---|---|---:|---:|---:|']
for d in ('imagenet','things'):
 for layer in ('fc1','fc2'):
  for window in ('early','late'):
   o=win[d]['noisy','joint',layer,'exact','off','priming_minus_different',window]
   h=win[d]['noisy','joint',layer,'exact','mean_std','priming_minus_different',window]
   v=win[d]['noisy','joint',layer,'exact','mean_std','interaction_vs_off',window]
   lines.append(f'| {d} | {layer} | {window} | {ci(o)} | {ci(h)} | {ci(v)} |')
lines+=['','FC1 的这个示例中，Hebb-FC 的差值之差为负，但 Hebb-FC 下的 exact 预览效应仍为正：其含义是与 Off 相比，正向激活差缩小，并非观察到负向 suppression。FC2 的符号随数据集与窗口变化，部分区间跨过零。',
 '', '## 对照与解释', '', 'Matched scale 在每张目标、每个时间点和每层都将均值匹配至 Hebb-FC，因此它的平均激活及由其组成的预览差值、差值之差与 Hebb-FC 按设计相同（仅有浮点误差）。这是事后逐样本匹配，不能把两条均值曲线相同当作独立复现，也不能仅凭平均激活区分侧向连接与整体缩放。Inhibition-only 的相应估计保存在完整表中。',
 '', '每个类别×外层折的4个概念为重采样层；每个概念内的4张图、三个噪声实现和全部时间点保持配对。区间以已拟合、固定的FC图和折划分为条件，没有在每次bootstrap中重训FC图，也未校正多重比较；预览角色可能共享概念。它们适合探索稳定性，不是全流程重训或严格独立验证的区间。部分概念和图片与先前FC研究有历史重叠。',
 '', '该分析没有修改或重训Predify网络、FC连接或slides。各折逐目标激活由已保存Predify状态重放FC得到，并已对原有逐折平均激活逐项复核。',
 '', '[全部图册](激活差值比较图册.pdf) · [ImageNet 时程](imagenet/timecourse.csv) · [ImageNet 窗口与区间](imagenet/windows.csv) · [THINGS 时程](things/timecourse.csv) · [THINGS 窗口与区间](things/windows.csv) · [数值审计](审计.json)']
for d in ('imagenet','things'):
 lines+=['',f'![{d} FC2 noisy preview effect](figures/{d}_fc2_noisy_preview_effect.png)',f'![{d} FC2 noisy interaction](figures/{d}_fc2_noisy_interaction.png)']
(DOC/'激活抑制比较.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(DOC/'审计.json').write_text(json.dumps({'passed':True,'timeline_comparison_count':len(errors),'timeline_max_abs_error':max(errors),'figure_count':len(manifest),'figures':manifest,'interval_rows':{d:len(windows[d]) for d in windows},'interval_scope':'stratified concept bootstrap conditional on fixed fitted fold-specific graphs; marginal, exploratory'},ensure_ascii=False,indent=2),encoding='utf8')
print('REPORT COMPLETE',len(manifest),'figures, max independent error',max(errors),flush=True)
