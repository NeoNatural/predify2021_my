"""Build a source-linked visual report. Reads results only; no model execution."""
from pathlib import Path
import csv, hashlib, html, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'docs/priming_sharpen_20261112'
LC=SRC/'learning_curve_20261113'
OUT=ROOT/'docs/hebbian_week_20261112_14'
OUT.mkdir(exist_ok=True)
(OUT/'figures').mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','figure.facecolor':'white',
 'axes.titlepad':14,'axes.labelpad':9,'savefig.facecolor':'white'})
BLUE='#2369a1'; GOLD='#c47c27'; GREEN='#238678'; RED='#bd4f50'; GRAY='#657381'
DS=['things','imagenet']; DN={'things':'THINGS','imagenet':'ImageNet'}
HIST=['exact','same_concept','different']; HN=['同图','同概念异图','异类']
ARMS=['trainable','extended_frozen']; AN=['可训练编码器','冻结编码器＋延长训练']
sources={}
def read(name):
 p=LC/(name+'.csv'); sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
tables={n:read(n) for n in ['learning_curve','native_test','frozen_budget_test','fc2_encoder_control','fc2_encoder_control_gates','fc2_ba_history_audit','fc2_gate_swap_effects']}
def row(name,**kw):
 rs=[r for r in tables[name] if all(r.get(k)==str(v) for k,v in kw.items())]
 assert len(rs)==1,(name,kw,len(rs))
 return rs[0]
def val(r):return float(r['estimate'])
def fc(ds,metric,comparison='absolute',arm='trainable',hist='exact',ref=''):
 return row('fc2_encoder_control',dataset=ds,metric=metric,comparison=comparison,condition=arm,history=hist,reference_history=ref,seed='mean')
def effect(ds,name):return row('fc2_gate_swap_effects',dataset=ds,effect=name,seed='mean')
def grid(ax,axis='y'):
 ax.grid(axis=axis,color='#e6ebef',linewidth=.7);ax.set_axisbelow(True)
def finish(fig,name):
 fig.savefig(OUT/'figures'/f'{name}.svg',bbox_inches='tight')
 fig.savefig(OUT/'figures'/f'{name}.png',dpi=165,bbox_inches='tight');plt.close(fig)
def points(ax,rows,labels,colors=None):
 for i,r in enumerate(rows):
  c=(colors or [BLUE]*len(rows))[i];lo,hi=float(r['ci_low']),float(r['ci_high']);v=val(r)
  ax.plot([lo,hi],[i,i],color=c,lw=2);ax.plot(v,i,'o',color=c,ms=6)
  ax.text(1.02,i,f'{v:+.2f} [{lo:+.2f}, {hi:+.2f}]',transform=ax.get_yaxis_transform(),va='center',fontsize=9)
 ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.axvline(0,color=GRAY,lw=.8);grid(ax,'x')

# 1. Equal-gate intervention: observed values transcribed from the archived report.
fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for j,ds in enumerate(DS):
 sup=[.1476,.0735,.0735] if ds=='things' else [.2046,.0640,.0640]
 fisher=[4.5087,-.0671,-.0651] if ds=='things' else [9.3036,-.0540,-.0474]
 for i,(a,y,label) in enumerate(zip(axs[:,j],[sup,fisher],['活动下降（%）','Fisher 相对增益（%）'])):
  a.bar(range(3),y,color=[BLUE,GOLD,GRAY],width=.55);a.axhline(0,color=GRAY,lw=.8)
  a.set_xticks(range(3),['逐物体匹配门','同一目标共用门值','仅四层抑制']);a.set_ylabel(label);grid(a)
  for x,v in enumerate(y):a.annotate(f'{v:.3f}',(x,v),xytext=(0,6 if v>=0 else -16),textcoords='offset points',ha='center')
  a.margins(y=.25)
 axs[0,j].set_title(DN[ds]+'｜同图对同概念异图')
finish(fig,'01_equal_gate')

# 2. Initial familiarity generalization and memory controls, separate cohorts.
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,ds in zip(axs,DS):
 vals=[.968,.578,.736,.532,.496,.5] if ds=='things' else [.919,.518,.891,.572,.508,.5]
 labels=['训练图\n新回路','验证图\n新回路','新图\n原匹配','新图\n真实记忆','新图\n打乱记忆','新图\n清空记忆']
 ax.bar(range(6),vals,color=[GRAY,GRAY,BLUE,GOLD,GREEN,GREEN],width=.65)
 ax.axhline(.5,color=RED,ls='--',lw=1);ax.set_ylim(0,1.08);ax.set_xticks(range(6),labels,fontsize=9)
 ax.set_title(DN[ds]+'｜同图 / 同概念异图辨别');ax.set_ylabel('重复辨别 AUC（同图与同概念异图）');grid(ax)
 for x,v in enumerate(vals):ax.text(x,v+.025,f'{v:.3f}',ha='center',fontsize=10)
finish(fig,'02_memory_generalization')

# 3. Mid/late windows from independent images, explicitly point estimates.
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,ds in zip(axs,DS):
 old,new=([5.05,5.43],[-.25,.49]) if ds=='things' else ([12.05,12.25],[2.19,2.09])
 for y,c,m,l in [(old,BLUE,'o','逐物体匹配门'),(new,GOLD,'s','初版熟悉度门')]:
  ax.plot([0,1],y,color=c,marker=m,label=l,lw=2)
  for x,v in enumerate(y):ax.annotate(f'{v:+.2f}%',(x,v),xytext=(0,9),textcoords='offset points',ha='center')
 ax.set_xticks([0,1],['中期 t5–9','晚期 t10–14']);ax.set_xlim(-.3,1.3);ax.set_ylim(-1.5,15)
 ax.set_ylabel('同图对同概念异图 Fisher 增益（%）');ax.set_title(DN[ds]);ax.axhline(0,color=GRAY,lw=.8);grid(ax);ax.legend(loc='upper center',fontsize=9)
finish(fig,'03_mid_late')

# 4. Diagnostics / architectural changes.
fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for ax,ds in zip(axs,DS):
 vals=[.737,.651,.665,.5775,.5694,.5688,.5394,.5494] if ds=='things' else [.809,.609,.6367,.5176,.5283,.5283,.5244,.5537]
 labs=['原 Conv5\n局部','64 维\n局部','加配对训练\n局部','原全图\n熟悉度','加配对训练\n全图','左右\n平均','左右\n竞争','两帧×左右\n四路竞争']
 ax.plot(range(3),vals[:3],'o-',color=BLUE,label='局部编码诊断')
 ax.plot(range(3,8),vals[3:],'s',color=GOLD,label='全图回路读出')
 for x,v in enumerate(vals):ax.annotate(f'{v:.3f}',(x,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
 ax.set_xticks(range(8),labs,fontsize=8);ax.set_ylim(.47,.88);ax.axhline(.5,color=RED,ls='--',lw=1)
 ax.set_title(DN[ds]+'｜验证折诊断');ax.set_ylabel('同图 / 同概念异图 AUC');grid(ax);ax.legend(fontsize=9)
finish(fig,'04_diagnosis')

# 5. Learning curves, raw test query versus shared train/validation query.
fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for ax,ds in zip(axs,DS):
 for split,col,mark,label in [('train',GRAY,'^','训练（共用目标反应）'),('validation',GOLD,'s','验证（共用目标反应）'),('native',BLUE,'o','测试（各条件自身反应）')]:
  rs=[row('native_test',dataset=ds,arm='trainable_encoder',level=l,negative='same_concept') if split=='native' else row('learning_curve',dataset=ds,arm='trainable_encoder',level=l,negative='same_concept',split=split) for l in ['2','8','16']]
  y=[val(r) for r in rs];ax.plot([1,4,8],y,color=col,marker=mark,label=label,lw=2)
  if split=='native':ax.fill_between([1,4,8],[float(r['ci_low']) for r in rs],[float(r['ci_high']) for r in rs],color=col,alpha=.12)
  offset=9 if split=='train' or (split=='native' and ds=='imagenet') or (split=='validation' and ds=='things') else -18
  for x,v in zip([1,4,8],y):ax.annotate(f'{v:.3f}',(x,v),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=9)
 ax.set_xticks([1,4,8],['1×\n80 图' if ds=='things' else '1×\n64 图','4×\n320 图' if ds=='things' else '4×\n256 图','8×\n640 图' if ds=='things' else '8×\n512 图'])
 ax.set_ylim(.48,1.07);ax.set_title(DN[ds]);ax.set_ylabel('同图 / 同概念异图 AUC');ax.axhline(.5,color=RED,ls='--',lw=1);grid(ax);ax.legend(loc='center right',fontsize=9)
finish(fig,'05_learning_curve')

# 6. Budget-controlled encoder comparison.
fig,axs=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
for ax,ds in zip(axs,DS):
 for j,(neg,col,lab) in enumerate([('same_concept',BLUE,'同图 / 同概念异图'),('different',GOLD,'同图 / 异类')]):
  rs=[row('frozen_budget_test',dataset=ds,negative=neg,arm=a,reference='') for a in ['original_frozen','extended_frozen','trainable_encoder']]
  xx=np.arange(3)+(j-.5)*.12;ax.plot(xx,[val(r) for r in rs],marker=['o','s'][j],color=col,label=lab)
  for x,r in zip(xx,rs):
   ax.plot([x,x],[float(r['ci_low']),float(r['ci_high'])],color=col,lw=1);ax.annotate(f'{val(r):.3f}',(x,val(r)),xytext=(0,11),textcoords='offset points',ha='center')
 ax.set_xticks(range(3),['冻结编码\n最多 64 轮','冻结编码\n最多 192 轮','可训练编码\n最多 64 轮']);ax.set_ylim(.53,.86);ax.set_title(DN[ds]+'｜8× 规模');ax.set_ylabel('重复辨别 AUC（测试图）');grid(ax);ax.legend(fontsize=9)
finish(fig,'06_training_budget')

# 7. Gate strength by history.
fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for ax,ds in zip(axs,DS):
 for arm,c,m,l in zip(ARMS,[BLUE,GOLD],['o','s'],AN):
  y=[val(row('fc2_encoder_control_gates',dataset=ds,scope='test',history=h,metric='mean_gate',condition=arm,reference='',seed='mean')) for h in HIST]
  ax.plot(range(3),y,marker=m,color=c,label=l,lw=2)
  for x,v in enumerate(y):ax.annotate(f'{v:.3f}',(x,v),xytext=(0,10 if arm=='extended_frozen' else -18),textcoords='offset points',ha='center')
 ax.set_xticks(range(3),HN);ax.set_ylim(.2,.95);ax.set_title(DN[ds]);ax.set_ylabel('平均门值（0–1）');grid(ax);ax.legend(fontsize=9)
finish(fig,'07_gates')

# 8. Absolute geometry: use exact CSV, independent y-axes clearly labeled.
fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for i,ds in enumerate(DS):
 for j,(met,lab) in enumerate([('within','类内散度'),('between','类间散度'),('fisher','Fisher＝类间 / 类内')]):
  ax=axs[i,j]
  for arm,c,m,l in zip(ARMS,[BLUE,GOLD],['o','s'],AN):
   y=[val(fc(ds,met,arm=arm,hist=h)) for h in HIST];ax.plot(range(3),y,marker=m,color=c,label=l,lw=1.8)
  ax.set_xticks(range(3),HN,fontsize=9);ax.set_title(DN[ds]+'｜'+lab);ax.set_ylabel('绝对值（局部纵轴）');grid(ax);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
  if i==0 and j==0:ax.legend(fontsize=8)
finish(fig,'08_absolute_geometry')

# 9. Geometry contrasts show separability versus compression.
fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for i,ds in enumerate(DS):
 for j,ref in enumerate(['same_concept','different']):
  ax=axs[i,j]
  for k,(arm,c,l) in enumerate(zip(ARMS,[BLUE,GOLD],AN)):
   rs=[fc(ds,m,'history_contrast',arm,ref=ref) for m in ['within','between','fisher']];xx=np.arange(3)+(k-.5)*.25
   ax.bar(xx,[val(r) for r in rs],width=.24,color=c,label=l)
   for x,r in zip(xx,rs):ax.plot([x,x],[float(r['ci_low']),float(r['ci_high'])],color='#344453',lw=1)
  ax.set_xticks(range(3),['类内变化','类间变化','Fisher 变化']);ax.set_title(DN[ds]+'｜同图对'+HN[HIST.index(ref)]);ax.set_ylabel('相对变化（%）');ax.axhline(0,color=GRAY,lw=.8);grid(ax)
  if i==0 and j==0:ax.legend(fontsize=9)
finish(fig,'09_geometry_contrasts')

# 10. Readout, show absolute BA and margin rather than infer from Fisher.
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for i,ds in enumerate(DS):
 for j,(met,lab) in enumerate([('ba','绝对平衡准确率 BA（%）'),('margin','绝对正确类别的得分优势')]):
  ax=axs[i,j]
  for arm,c,m,l in zip(ARMS,[BLUE,GOLD],['o','s'],AN):
   y=[val(fc(ds,met,arm=arm,hist=h)) for h in HIST];ax.plot(range(3),y,color=c,marker=m,label=l)
   for x,v in enumerate(y):
    other=val(fc(ds,met,arm=ARMS[1-ARMS.index(arm)],hist=HIST[x]))
    ax.annotate(f'{v:.2f}',(x,v),xytext=(0,10 if v>other else -18),textcoords='offset points',ha='center',fontsize=9)
  ax.set_xticks(range(3),HN);ax.set_ylabel(lab);ax.set_title(DN[ds]);grid(ax);ax.margins(y=.35)
  if i==0 and j==0:ax.legend(fontsize=8)
finish(fig,'10_readout')

# 11. Exact / Different accounting waterfall, no invented confidence intervals.
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for ax,ds in zip(axs,DS):
 start=val(effect(ds,'trainable_native'));n=val(effect(ds,'exact_contribution'));d=val(effect(ds,'different_contribution'));end=val(effect(ds,'frozen_native'))
 assert abs(start+n+d-end)<1e-10
 ax.bar(0,start,color=BLUE,width=.6);ax.bar(1,n,bottom=start,color=GREEN,width=.6);ax.bar(2,-d,bottom=end,color=RED,width=.6);ax.bar(3,end,color=GOLD,width=.6)
 for x,y in [(0,start),(1,start+n),(2,end)]:ax.plot([x+.3,x+.7],[y,y],color=GRAY,lw=1)
 for x,v,y in [(0,start,start),(1,n,start+n),(2,d,start+n),(3,end,end)]:ax.text(x,y+.3,f'{v:+.2f}' if x in [1,2] else f'{v:.2f}%',ha='center')
 ax.set_xticks(range(4),['可训练组','同图项贡献','异类项贡献','冻结组']);ax.set_ylim(0,14);ax.set_title(DN[ds]);ax.set_ylabel('同图对异类 Fisher 增益（%）');grid(ax)
finish(fig,'11_fisher_accounting')

# 12. Uncertainty of components; confidence ranges may exclude the original estimate.
fig,axs=plt.subplots(2,1,figsize=(12,8),layout='constrained')
names=['different_contribution','mean_strength_path','saturation_path','remaining_distribution','remaining_assignment']
labels=['异类门总贡献','整体强度（压顶后缩放）','保均值压顶','等均值：类条件分布','等均值：类内 target 分配']
for ax,ds in zip(axs,DS):
 points(ax,[effect(ds,n) for n in names],labels,[GRAY,BLUE,GREEN,GOLD,GOLD]);ax.set_xlim(-6.4,2);ax.set_xlabel('Fisher 历史增益变化（百分点）');ax.set_title(DN[ds],loc='left')
finish(fig,'12_strength_components')

# 13. Correct the absolute-vs-difference interpretation.
fig,axs=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
for hist,c,m,lab in [('exact',BLUE,'o','同图'),('same_concept',GOLD,'s','同概念异图')]:
 y=[val(row('fc2_ba_history_audit',dataset='things',history=hist,reference_history='',comparison='absolute',condition=k,seed='mean')) for k in ['1x','8x']]
 axs[0].plot([0,1],y,color=c,marker=m,label=lab,lw=2)
 for x,v in enumerate(y):axs[0].annotate(f'{v:.2f}%',(x,v),xytext=(0,10),textcoords='offset points',ha='center')
axs[0].set_xticks([0,1],['1× 数据','8× 数据']);axs[0].set_xlim(-.3,1.3);axs[0].set_ylim(71.8,74.2);axs[0].set_ylabel('绝对 BA（%）');axs[0].set_title('THINGS｜两种历史的绝对 BA 都增加');axs[0].legend();grid(axs[0])
rr=[row('fc2_ba_history_audit',dataset='things',history=h,reference_history='',comparison='same_history_difference',condition='8x',seed='mean') for h in ['exact','same_concept']]
rr.append(row('fc2_ba_history_audit',dataset='things',history='exact',reference_history='same_concept',comparison='history_contrast_difference',condition='8x',seed='mean'))
for i,r in enumerate(rr):
 axs[1].plot([float(r['ci_low']),float(r['ci_high'])],[i,i],color=[BLUE,GOLD,RED][i],lw=2);axs[1].plot(val(r),i,'o',color=[BLUE,GOLD,RED][i]);axs[1].text(val(r),i-.13,f'{val(r):+.2f}',ha='center')
axs[1].set_yticks(range(3),['同图 BA 变化','同概念异图 BA 变化','重复效应的变化']);axs[1].invert_yaxis();axs[1].margins(y=.3);axs[1].axvline(0,color=GRAY,lw=.8);axs[1].set_xlabel('8× 减 1×（百分点）');axs[1].set_title('两种预览条件的正确率相差多少？');grid(axs[1],'x')
finish(fig,'13_ba_correction')

# A standalone, self-contained, printable report with SVG charts and exact tables.
parts=[]
def p(s):parts.append(s)
def section(n,title):p(f'<section id="s{n}"><h2>{title}</h2>')
def end():p('</section>')
def para(s):p('<p>'+s+'</p>')
def fig(name,caption):
 svg=(OUT/'figures'/f'{name}.svg').read_text(encoding='utf-8');svg=svg[svg.index('<svg'):]
 # Each embedded Matplotlib SVG has its own ID namespace, including clip paths.
 ids=re.findall(r'\bid="([^"]+)"',svg)
 for ident in sorted(set(ids),key=len,reverse=True):
  replacement=name+'-'+ident
  svg=svg.replace('id="'+ident+'"','id="'+replacement+'"').replace('url(#'+ident+')','url(#'+replacement+')').replace('href="#'+ident+'"','href="#'+replacement+'"')
 p(f'<figure role="img" aria-label="{html.escape(caption)}">{svg}<figcaption>{caption}</figcaption></figure>')
def table(headers,rows):
 p('<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>')
 for rr in rows:p('<tr>'+''.join('<td>'+str(x)+'</td>' for x in rr)+'</tr>')
 p('</tbody></table></div>')
def source(label,path):
 pp=SRC/path;sources[str(pp.relative_to(ROOT))]=hashlib.sha256(pp.read_bytes()).hexdigest()
 para(f'<span class="source">来源：<a href="../priming_sharpen_20261112/{path}">{label}</a></span>')

p('<header><div class="eyebrow">实验覆盖：2026.11.12 周四 — 11.14 周六 · 11.18 释义修订</div><h1>从重复抑制到类别可分性：<br>三天实验的证据与边界</h1><p class="lead">同一张图片再次出现时，网络的反应变弱了。它还能把物体类别分得更清楚吗？</p><p>本报告回顾 11 月 12–14 日完成的实验。11 月 18 日根据读者批注补充了指标算法、短时记忆的实现、各图实验条件和现有数据的扩样空间。</p></header>')
section(1,'重复出现的图像会引起什么变化？')
table(['问题','目前证据','结论边界'],[
 ['仅抑制能否产生重复效应？','能。共用门值后，同图仍比同概念异图引起更低的平均反应，ImageNet 低约 0.064%，THINGS 低约 0.074%。','重复抑制自身没有带来稳定类别可分性改善。'],
 ['Fisher 上升来自哪里？','终端 FC2 语义重分配主要拉大类间距离，类内变化较小。','部分 THINGS 批量类内收缩，但不能概括为普遍类内压缩。'],
 ['新门真的读取预览吗？','清空记忆后门值为零，打乱记忆使辨别接近机会水平。','依赖记忆不等于同图辨别足够强，更不等于已证明生物实现。'],
 ['扩样有效吗？','8× 对 1×，两数据集同图 / 同概念异图测试 AUC 都提高约 0.06。','样本量与概念覆盖同时增加，8× 仍有明显训练—验证差距。'],
 ['编码器必须训练吗？','ImageNet 有额外收益；THINGS 延长训练的冻结编码达到相近辨别水平。','相近点估计、区间含零不是等效性证明，结论依赖当前预算。'],
 ['为什么冻结组同图相对异类的 Fisher 优势更小？','主要是异类也得到更强门值，异类绝对 Fisher 增加。','同图自身并未显示可靠变差，必须分开同图自身的变化与重复效应的变化。']])
end()
section(2,'三天的实验分别回答了什么问题？')
table(['日期 / 阶段','完成的工作','关键发现 / 推进理由'],[
 ['11/12 · 方向与读出','比较用预览和目标共同决定增益方向、仅用目标决定方向，以及减小调节幅度或合并多层类别证据。','早期 Fisher 改善常伴随 BA 或 margin 损失。预训练语义证据更稳定，但个别分类翻错仍存在。'],
 ['11/12 · 更严格的重复对照','增加同概念异图；比较两种方向；对每张目标图共用门值。','仅由目标决定方向更稳。共用门值几乎移除额外 Fisher 预览条件差，却保留原四层重复抑制。'],
 ['11/12 · 网络内熟悉度','全图 Conv5 → 512→64 编码 → 快速抑制记忆 → 群体响应差；清空、打乱、固定目标。','确实依赖预览记忆，但训练—验证差大，新图同图辨别弱。'],
 ['11/12 · 定位信息损失','局部编码、空间读写、平滑定标、跨视图续训、左右分区、两帧分开及竞争读出。','局部信号有损失；多项局部改动没有稳定传到全图门和 FC2。'],
 ['11/13 · 扩样与预算','两数据集 × 3 规模 × 2 编码训练方式 × 3 种子，共 36 个主训练运行；补充 8× 冻结组延长训练。','扩样有独立测试增益。控制训练预算后，编码器训练的必要性随数据集不同。'],
 ['11/13 · 接回同一 FC2','8× 可训练与延长冻结，验证集定标；报告每种预览条件绝对量及预览条件差。','同图对异图的可分性差异主要体现在类间，读出 BA 结论较弱。'],
 ['11/14 · 机制分解','异类门双向交换、Fisher 分子分母分解、保均值压顶、等均值缩放、类内置换。','整体门控强度的变化分别解释 THINGS 异类门总效应的约 83%、ImageNet 的约 68%。剩余分布项还需逐类控制。']])
para('以上日期依据归档批次和报告记录组织，同一天内按研究问题排列，不把顺序当作精确运行时间。早期搜索反复使用过的批量，后续应视为开发数据。各阶段图像清单不同，不能跨图直接把点估计差当作方法收益。')
source('初期搜索与四层复核','进展与结果.md');end()
section(3,'一次试次中，网络识别了什么？')
para('设想目标是一张猫的照片。目标出现前，屏幕先呈现两帧预览图像，每帧各有两个物体。实验者只替换第二帧中的一个物体，其余三个物体和目标照片都配对保持一致。如果这个位置出现同一张猫照片，称为同图重复；换成同一细粒度概念的另一张照片，称为同概念异图；换成另一大类的物体，称为异类。后文的“预览条件”指目标出现前看过哪一种刺激。')
p('<h3>一次试次可以回答两个不同的问题</h3>')
table(['科学问题','从哪里取得分数？','怎样评估？'],[
 ['目标这张照片刚才出现过吗？','旧方法逐一比较目标与四个预览物体，给出匹配分数。新回路根据预览留下的短时记忆，给出一个熟悉度分数。','正例是同图试次，负例分别取同概念异图或异类试次。例如两张都是猫，只有同一张照片算重复。图 2、4、5、6 衡量这种重复辨别。'],
 ['从神经群体活动能读出目标属于哪一大类吗？','把目标引起的 FC2 群体活动交给固定的线性支持向量机（SVM）。解码器对动物、植物等每个大类各给一个分数。','两张猫照片在这里都属于动物。比较不同预览条件下的分类表现，可以检验预览是否改变了 FC2 中可供解码的类别信息。图 10 和图 13 使用这一读出。']])
para('两种任务都可以画 ROC 曲线。改变判定阈值，把命中率对虚警率作图，曲线下面积就是 AUC。等价地，AUC = P(正例得分 > 负例得分) + ½ P(两者同分)。0.5 对应随机排序。举例来说，AUC 为 0.75 表示随机各取一个正例和负例，约四分之三的比较把正例排在前面（平局计半）。它描述排序，单凭这个值无法得出某个阈值下的正确率。')
p('<h3>怎样读懂同一个类别解码器的三个指标？</h3>')
table(['指标与公式','用一个例子理解','对什么变化敏感？'],[
 ['平衡准确率 BA = (1/K) Σₖ (第 k 类判对图数 / 第 k 类总图数)。K 是大类数。','若两类各自的识别率为 90% 和 50%，BA 就是 70%，两类权重相同。','关心最终判对了多少。类别样本数不同，大类也不会因图多而占更大比重。'],
 ['真实类别的得分优势 margin = sᵧ − maxₖ≠ᵧ sₖ。s 是固定解码器给出的分数，y 是正确类别。','动物图的动物得分 1.3，最强竞争类得分 1.1，优势为 +0.2。动物得分增至 1.5，优势变为 +0.4，两次都判对。','可以在 BA 不变时增加。负值表示有其他类得分更高。先逐图计算，再按报告的类别和概念权重汇总，单位为解码器分数。'],
 ['分类宏 AUC = (1/K) Σₖ AUCₖ。对第 k 类，用同一列得分 sₖ 比较本类图与其他类图。','例如看动物得分能否把动物图排在非动物图之前，对每个大类都做一次，再取等权平均。','关心同一个类别分数在不同图片之间的排序。margin 比较的是同一张图的不同类别分数。']])
p('<h3>怎样判断重复抑制和类别表征锐化？</h3>')
para('重复抑制看群体平均反应，抑制量 = 100 × (1 − 同图平均活动 / 对照平均活动)。类别可分性看群体活动模式：先把每张图的 FC2 活动向量归一化，再计算同一类别各图到类别中心的平均平方距离（类内散度 W），以及各类别中心到总体中心的平均平方距离（类间散度 B）。Fisher 比值为 B/W。类内收缩和类间拉开都能提高这个比值。这里用这两项实测距离描述群体表征的变化；单个神经元调谐曲线是否变窄，需要另外测量。')
para('“某一条件的绝对水平”是直接读出该条件自身的数值。例如只看同图试次，比较扩样前后 BA。“重复效应”要在同一模型里再减去对照条件的 BA。“重复效应的变化”则把两个模型的重复效应再相减。对于 Fisher，先算相对增益 100 × (同图 Fisher / 对照 Fisher − 1)，再比较两个模型。图 11 和图 13 分别用真实数据演示这两种算法。')
p('<h3>哪些连接实际用了 Hebbian 计算？</h3>')
para('有两处短时关联计算。原四层抑制通路利用预览期间的共同活动形成试次内的临时连接。后来增加的熟悉度回路也按共同活动写入记忆。以后者为例，某空间位置上，编码单元 j 的反应为 zⱼ，熟悉度单元 i 的参考反应为 hᵢ。每帧的写入量为 Bᵢⱼ = 各位置 hᵢ × zⱼ 的平均值。试次开始时记忆清零，两帧后 M = 0.5 B₁ + B₂。两个单元在预览时越常同时活跃，这条连接的短时抑制作用越强。目标出现时只读取 M，比较施加抑制前后的群体反应，得到熟悉度分数。这个按前后单元共同活动写入的步骤就是 Hebbian 型计算。')
para('还有两个不同时间尺度的过程。跨试次的编码权重和熟悉度读出由带重复标签的训练调整。FC2 的类别增益方向则来自预训练分类头的类别证据和训练折校准的类别原型。因而本实验观察到的类别间分离，是短时记忆调节已有类别表征后的结果；局部突触学习单独能产生多少类别分离，需看仅保留抑制的对照。')
end()
section(4,'重复抑制与类别分离分别来自哪一步？')
fig('01_equal_gate','图 1｜对同一批目标照片，利用记录下来的活动重算三种条件。上排是活动抑制，下排是 Fisher 相对增益。图示为点估计，完整配对区间见来源。THINGS 80 图，ImageNet 64 图。')
table(['图 1 的三个条件','对同一张目标图具体做什么？','这个比较能回答什么？'],[
 ['逐物体匹配门','对同图预览和同概念异图预览分别匹配。两次得到的门值各自决定 FC2 活动重新分配的幅度。','测量整个模型在两种预览后的反应差。'],
 ['同一目标共用门值','对目标 i，令两次门值都等于 (gᵢ同图 + gᵢ异图)/2。例如原值 0.9 和 0.3，就都用 0.6；下一张目标另算自己的均值。两次仍用各自的四层活动和由目标确定的方向。','保留这张图平均受到的调节，只消除两种预览之间的门控强度差。这个控制在每张目标图内部进行。'],
 ['仅四层抑制','直接读取四层网络经过自身抑制后的 FC2 活动，跳过额外的类别增益。','给出原网络本身能产生的重复抑制和类别分离。']])
para('THINGS 上，把同一目标两次的门值设为相等，Fisher 增益从 +4.51% 变为 −0.067%，而活动仍下降约 0.074%。ImageNet 也呈现相同趋势。这使我们能把原四层保留的重复抑制，与门控强度差带来的额外类别优势区分开。')
table(['初期路线','冻结复核中的表现','怎样解释'],[
 ['预览同时决定方向','首轮样例方向 ImageNet 对灰屏 Fisher +75.2%，margin −0.254。','巨大的 Fisher 增益也可能伴随读出恶化。'],
 ['目标证据残差','ImageNet 对灰屏 Fisher +31.42%，BA −3.125 pp。','连续分数改善不能代替最终分类正确率。'],
 ['预训练语义增益','四层首批 Fisher +6.49% 至 +13.39%，主要为类间拉开。','语义先验有效，但不是抑制机制自然产生的类别结构。'],
['减小增益强度，同时考察另一层提供的类别信息','“混合”是指把当前活动模式向类别证据指示的模式移动多少。把混合上限从 0.60 降到 0.30，活动更接近原反应。另一次实验让 FC1 和原分类头各提供一份类别概率，两者预测不一致时减弱调节。','新图上仍可见类别分离。ImageNet 的分类宏 AUC 没有稳定提高。'],
 ['预览中究竟哪个物体和目标相符？','旧方法将两帧中的四个物体分别裁出、移到中心，用同一网络编码，再与目标比较。越匹配的物体，对预览类别判断的权重越大。','这一步给出“调多强”，还能提供一份来自预览的类别证据。'],
 ['预览和目标要一起决定调节方向吗？','对每个类别，把目标概率与预览概率相乘、开方，再归一化。只有两边都支持的类别才得到较大权重。','固定参数后，两数据集分别比较同图与灰屏、同图与异类，共四个 BA 变化均为正。只有 THINGS 对异类的区间严格高于零。另一批量增加同概念异图对照后，保留同一门值、仅由目标决定方向，在两数据集都得到更高的同图 Fisher 和得分优势。']])
source('同概念异图与同一目标共用门值','同类异图复核.md');end()
section(5,'短时记忆能认出刚刚看过的图片吗？')
fig('02_memory_generalization','图 2｜任务均为同图对同概念异图。训练、验证和新图是不同划分；新图内部原门、真实、打乱、清空为配对对照。虚线为机会水平 0.5。')
table(['图 2 从左到右','每根柱测量什么？'],[
 ['① 训练图 · 新回路','在曾用于学习长期连接的图像上，用熟悉度分数区分同图和同概念异图。'],['② 验证图 · 新回路','换用验证图，考察学到的辨别能力能留下多少。验证集还用于选择检查点。'],
 ['③ 新图 · 逐物体匹配','在独立新图上，用四个物体分别与目标匹配得到的连续分数辨别重复。'],['④ 新图 · 真实记忆','同一批新图，用两帧实际写入的全图短时连接，让目标读取自己的试次记忆。'],
 ['⑤ 新图 · 打乱记忆','目标活动相同，将记忆矩阵的行和列独立打乱。保留连接值及总量，破坏连接与单元的对应。图中取八次置换平均。'],
 ['⑥ 新图 · 清空记忆','将短时连接全设为零，全部熟悉度分数相同。按平局计半，AUC 正好是 0.5。']])
para('六根柱都在问“这一张目标图片刚才出现过吗”。其中后四根共用同一批新图，可直接比较。真实记忆让 THINGS 从打乱时的 0.496 升到 0.532，ImageNet 从 0.508 升到 0.572。连接与单元之间的对应关系保留了一部分图片身份信息。逐物体匹配分别达到 0.736 和 0.891，它能单独比较每个预览物体；全图回路把四个物体和背景的活动汇入记忆，辨别同概念的两张照片更困难。空间汇总是一个可能原因，两种方案还同时改变了编码方式。图 4 进一步检查信息在哪一步减少。')
para('图 3 使用同样的两种门控方式，观察量换成下游 FC2 的类别分离。蓝线按逐物体相似程度调节，橙线按全图短时记忆引起的反应下降调节。两者都由当前目标决定类别增益方向，因此可以比较门控方式如何影响中期与晚期的群体表征。')
fig('03_mid_late','图 3｜初版熟悉度门的独立新图复核：THINGS 40 图，ImageNet 32 图。曲线连接两个窗口的均值，不是逐步动力学曲线。新门四个窗口的 Fisher 区间均含零，原门均高于零。')
para('中晚期检查没有显示“等得更久就恢复原门效果”。新图的 t9 重复抑制仍为正，但有预览对照下类内、类间及 Fisher 的区间均含零。后续扩样、冻结编码和门值交换主要使用 t9，尚没有对应的中晚期群体复核，不能把初版时间窗结论直接外推到 8× 模型。')
source('熟悉度回路、记忆对照与中晚期','熟悉度回路复核.md');end()
section(6,'同一物体换了位置，网络在哪里丢失了信息？')
fig('04_diagnosis','图 4｜同一验证折的诊断点估计。蓝色为使用空间区域信息的局部探针，橙色为全图回路，两种任务输入条件不同。局部点之间连线只表示处理阶段。')
para('Conv5 是第五卷积层。它在每个空间位置用 512 个特征通道描述局部图像，这是原网络的结构。熟悉度模型在每个位置使用同一组线性权重，把这 512 个数映射到设计者选定的 64 个编码单元，整流后至多保留反应最强的 12 个，再归一化。空间位置仍保留。64 是回路的设计参数；它也决定了短时连接矩阵的大小为 64 × 64。')
table(['图 4 从左到右','实验者取什么反应、怎样算分？'],[
 ['① 原 Conv5 局部','已知预览中替换物体的位置，取该区域与中央目标的 512 通道活动，分别作空间平均，以余弦相似度检验同一张图能否匹配。'],['② 64 维局部','在相同位置作同一比较，只把活动换成压缩后的 64 维编码。ImageNet AUC 从 0.809 降至 0.609。'],
 ['③ 配对训练后的局部编码','训练时额外要求中央目标与外围同图的局部编码相近，并区分同概念下不同图片；用相同局部相似度再测一次。'],
 ['④ 原全图熟悉度','让原回路用整个预览画面写入一份短时记忆，再让整个目标画面读出熟悉度。此时回路不读取实验者标记的物体位置。'],
 ['⑤ 配对训练后的全图熟悉度','把③训练后的编码器用于④的全图记忆读写，检验局部识别收益是否传到实际回路。'],
 ['⑥ 左右平均','左右半幅各存一份记忆，目标分别读出两个分数，取算术平均。'],['⑦ 左右竞争','仍读同样的两份记忆，采用两者中较高的熟悉度分数。'],
 ['⑧ 两帧、左右共四路竞争','两帧分开保存，每一帧再分左右，目标分别读四份记忆，取四个分数中最高的一个。']])
table(['诊断问题','具体改变与观察','怎样理解'],[
 ['写入和读取的位置是否重要？','从预览中只取替换物体所在区域写入，或删除这一区域，再让目标全图或中央部分读取。删除替换区域后，中央读取时对异类的辨别变差。','相关物体确实提供信息。仅保留这个区域时，对同概念异图的改善仍不确定。'],
 ['把熟悉度转换成门值时损失了什么？','原门设有上限 1。两个较大的熟悉度分数都可能变成 1，丢失先后顺序。用平滑曲线逐渐接近上限，THINGS 的同图辨别 AUC 提高约 0.029。','把新门值用于 FC2 后，同图相对同概念异图的 Fisher 增益反而减少约 0.73 个百分点。分数排序与实际施加的增益强度衡量不同环节。'],
 ['局部训练是否有助于整幅图的辨别？','配对训练后，THINGS 的局部 AUC 从 0.651 升到 0.665，全图从 0.578 降到 0.569。ImageNet 通过验证折筛选，但在两批旧图上，FC2 的改善未稳定超过原模型。','模型学会匹配指定位置的物体，整幅预览仍包含其他物体。THINGS 的结果止于验证折。'],
 ['取最强反应能排除干扰吗？','THINGS 左右取最大时 AUC 为 0.539，低于全图的 0.578。四路取最大还提高了异图试次中原本就偏高的分数。','候选越多，干扰物也越有机会碰巧得到高分。应同时观察同图命中与异图虚警。']])
source('编码、读写和定标诊断','熟悉度回路诊断与验证方案.md');source('配对续训','共享编码器续训复核.md');source('空间与帧分区','分区记忆读出复核.md');end()
section(7,'增加训练图像和物体种类后，重复辨别有所改善')
para('同一张目标照片，在不同预览之后可能引起略有不同的 Conv5 活动。读取记忆时需要输入一个目标活动模式，旧图例把它叫作“查询”。训练和验证时，研究者取同图条件的目标活动，依次读取三份不同预览形成的记忆。这样，三次辨别分数的差异只可能来自预览记忆。测试时，各条件用自己实际产生的目标活动读取自己的记忆，更接近完整模型的工作方式。两种读法都用熟悉度分数判断图片是否重复；下游类别解码另见图 10。')
fig('05_learning_curve','图 5｜可训练编码器的三个随机种子的均值。测试阴影为配对概念重采样的 95% 区间。训练和验证时三种预览共用一个目标反应；测试时各用各自的目标反应。两种读法都观察到扩样收益。')
table(['数据集','训练图像 / 概念，1×→4×→8×','验证','测试','8×−1× 测试 AUC'],[
 ['THINGS','80/10 → 320/40 → 640/80','120 图 / 30 概念','108 图 / 30 概念','+0.061 [0.034, 0.088]'],
 ['ImageNet','64/8 → 256/32 → 512/64','96 图 / 24 概念','96 图 / 24 概念','+0.060 [0.029, 0.093]']])
para('扩样降低了训练上的近乎完美记忆，同时提高留出辨别。THINGS 的主要收益出现在 1×→4×，4×→8× 的增量区间含零。ImageNet 的 4×→8× 仍有正增量。训练—验证 AUC 差仍约 0.27–0.29，因此过拟合被缓解而未消失。每个概念仍为八张训练图，本轮同时增加了图像数和概念覆盖，不能把两者拆开归因。')
p('<h3>还有多少扩样空间？</h3>')
para('这里的“概念”指大类里的细分物体，例如植物大类中的盆景与灌木。训练时每个大类从 2 个概念增到 8 个、再到 16 个，每个概念始终取 8 张图。因而 8 倍数据同时增加了图像数量和物体种类。若只给原有概念多加照片，检验的将是认识同一种物体的更多外观。')
para('按本地候选清单清点，排除本轮训练、验证和测试已用概念及图像哈希，要求每个新增概念至少有 8 张可读且不重复的图片。THINGS 还可用动物 60、服饰 49、食物 191、植物 1、交通工具 3 个概念；ImageNet 分别有动物 369、服饰 25、食物 14、交通工具 36 个。依照现有各大类等量抽样，THINGS 每类最多再加 1 个（共 40 张），ImageNet 每类可再加 14 个（共 448 张）。这是沿用当前类别审核及训练划分规则的库存清点，历史开发批次的曝光情况若另作排除，还需重新计数。')
para('可优先检验两条扩样路径：一组给已有概念增加不同照片，另一组以相同新增图像数加入新概念。THINGS 若要大幅扩充第二组，需先补充植物和交通工具的合格概念。数据增强可先单独检验预览位置或对比度扰动，使同一目标在不同位置、亮度下仍能被认出。把正负例的位置和亮度分布配平，并保留物体身份，才能排除网络借助呈现条件猜答案。')
para('目前训练已使用验证损失早停和权重衰减（0.001）。另一项单因素比较可适度加强权重衰减，或固定每批同概念异图的比例，使训练持续包含难区分的图片对。各方案共用训练预算，先比较验证集的重复辨别 AUC，再用未参与选择的图像检验类别解码。这些是后续候选，本文没有相应的效果估计。')
fig('06_training_budget','图 6｜8× 规模的三个随机种子的均值和 95% 区间。比较冻结随机编码器、延长训练的冻结编码器、可训练编码器。编码器被冻结时，熟悉度连接和读出仍训练。')
table(['图 6 的三个训练条件','哪些参数可以改变？','能回答什么问题？'],[
 ['冻结编码，最多 64 轮','512→64 的权重取该种子的随机初值并固定。熟悉度单元的长期连接、抑制强度和群体读出等参数继续学习。','固定的视觉编码配上学到的熟悉度读出，能达到什么水平？'],
 ['冻结编码，最多 192 轮','同一随机编码、同一批训练图、相同优化规则；把允许的最长训练时间延长，按验证损失选检查点。','前一组的差距有多少来自读出训练时间不足？'],
 ['可训练编码，最多 64 轮','从配对的随机初值出发，同时更新编码器和熟悉度回路的长期参数。三组在每个试次都重新写入预览短时记忆。','给编码器学习机会后，原图重复与相似异图是否更容易分开？']])
para('只比较相同 64 轮会低估冻结组，因为它尚未收敛。最多 192 轮后，THINGS 的同概念异图辨别达到 0.626，与可训练的 0.625 接近；ImageNet 为 0.609 对 0.656，仍支持训练编码器的额外收益。两组的异类辨别都仍由可训练编码器占优。延长组仍到达预算上限，因此不把当前差距视为最终能力上限。')
source('扩样、隔离协议与训练预算','learning_curve_20261113/熟悉度学习曲线.md');end()
section(8,'熟悉度如何改变 FC2 的类别表征？')
p('<p><a href="timecourse_20261119/index.html"><b>11 月 19 日补图 · 三种预览条件的完整时间过程</b></a>　比较 THINGS 与 ImageNet 在第 0–14 步的 FC2 激活、类内与类间散度、Fisher、固定解码及逐步 MVPA。</p>')
p('<p><a href="timecourse_20261119/fc2_ablation/lateral/fresh_replication/index.html#gain"><b>新旧 FC2 增益的直接对照</b></a>　在同一批留出照片上比较显式类别增益与稀疏侧向增益，先看同图相对异类预览的解码、MVPA 和 Fisher 时间曲线。</p>')
p('<p><a href="timecourse_20261119/preview_components/index.html"><b>同概念异图为什么在早期占优？</b></a>　用训练折估计类别共有成分，比较三种预览下的照片残差，并在同一批目标上检验 PCoder 状态重置的影响。</p>')
para('此前的四层网络一直包含第二全连接层 FC2，周四也已把门用于调节这一层。图 5、6 暂时只评估扩样后的熟悉度回路。此处把两组回路产生的分数各自在验证集换算成门值 g，再作用于同一目标的 FC2 活动，检验前面观察到的辨别能力能否影响下游类别表征。')
para('具体做法是先取四层抑制后的 FC2 活动。其平均幅度乘以 (1 − 0.003g)，各单元的相对活动按 0.45g 的比例向目标自身给出的类别活动模式调整。当 g = 0 时原样输出。随后用同一个训练好的线性解码器读出物体类别。比较两种编码器时，各预览条件使用相同的原始 FC2 活动、类别方向和解码权重。')
fig('07_gates','图 7｜同一 8× 测试批量的平均门值。冻结组对异类给出更高门强度；同图门值差小。蓝色可训练，橙色延长训练冻结，后续图保持此配色。')
fig('08_absolute_geometry','图 8｜三种预览条件下的类内散度、类间散度和 Fisher，三个随机种子的均值。纵轴使用局部范围以显示小变化，面板间纵轴不可直接比较。完整数值见下方展开表。')
fig('09_geometry_contrasts','图 9｜同图相对两种历史的几何变化，竖线为 95% 区间。类内下降有利于紧凑，类间上升有利于分离。Fisher 的主要增益来自类间拉开。')
para('THINGS 同图对同概念异图的类内变化接近零或略增，仍有 Fisher 增益。ImageNet 类内略降，但幅度远小于类间增加。这更适合描述为类别可分性增强。冻结组相对异类的 Fisher 优势较小，并不等于冻结组同图表征本身更差。')
fig('10_readout','图 10｜同一固定 SVM 的绝对 BA 与 margin，三个随机种子的均值，局部纵轴。点估计接近时标签分置曲线两侧。图中未绘绝对值区间，组间推断使用配对差区间。')
para('在同图预览条件下，两编码器组的 Fisher、margin、宏 AUC 和 BA 差异均未给出稳定的非零证据。THINGS 同图 BA 为 73.06% 对 72.50%；ImageNet 为 71.88% 对 72.57%。这与同图相对异类的 Fisher 优势存在组间差异并不矛盾。')
p('<details><summary>展开：全部历史的绝对活动、几何与固定读出</summary>')
rows=[]
for ds in DS:
 for h,hn in zip(HIST,HN):
  for arm,an in zip(ARMS,AN):rows.append([DN[ds],hn,an]+[f'{val(fc(ds,m,arm=arm,hist=h)):.{6 if m in ["mean","within","between","fisher"] else 3}f}' for m in ['mean','within','between','fisher','margin','auc','ba']])
table(['数据','历史','编码器','活动','类内','类间','Fisher','margin','宏 AUC %','BA %'],rows);p('</details>')
p('<details><summary>展开：重复抑制、预览条件差与配对不确定性</summary>')
rows=[]
for ds in DS:
 for ref in ['same_concept','different']:
  for arm,an in zip(ARMS,AN):
   rs=[fc(ds,m,'history_contrast',arm,ref=ref) for m in ['suppression','within','between','fisher','margin','auc','ba']]
   rows.append([DN[ds],HN[HIST.index(ref)],an]+[f'{val(r):+.3f} [{float(r["ci_low"]):+.3f}, {float(r["ci_high"]):+.3f}]' for r in rs])
table(['数据','同图的对照','编码器','抑制 %','类内 %','类间 %','Fisher %','margin 差','宏 AUC pp','BA pp'],rows);p('</details>')
source('冻结与可训练编码器 FC2 对照','learning_curve_20261113/冻结编码器FC2对照.md');end()
section(9,'为什么冻结编码器后，同图相对异类的优势变小？')
fig('11_fisher_accounting','图 11｜从可训练组到冻结组的 Fisher 相对增益变化。同图项与异类项的贡献按两种替换顺序平均，逐种子计算后汇总；不能用绝对 Fisher 的组均值直接替代。正贡献抵消部分下降。')
table(['图 11 的四个位置','具体怎么算？'],[
 ['左端 · 可训练组','起点 H = 100 × (同图 Fisher / 异类 Fisher − 1)。THINGS 为 9.13%，即同图比异类条件高 9.13%。'],
 ['第二柱 · 同图项','只把分子换成冻结组的同图 Fisher，分母保持给定值。分别用两组的异类分母算一次，取平均。THINGS 的这项贡献为 +0.21 个百分点。'],
 ['第三柱 · 异类项','只替换分母，分别用两组的同图分子算一次再平均。冻结组的异类 Fisher 较高，因此 THINGS 的重复优势减少 2.54 个百分点。中间两柱是同一组比较的算术分解。'],
 ['右端 · 冻结组','终点是冻结组自己的同图相对异类增益。THINGS 从 9.13% 加上两项变化，得到约 6.80%。计算在各随机种子内完成，展示值经过四舍五入。']])
table(['数据','同图项贡献 pp','异类项贡献 pp','净变化 pp','冻结组保留同图门值，只换入可训练组的异类门值（百分点）'],[
 [DN[ds]]+[f'{val(effect(ds,n)):+.2f} [{float(effect(ds,n)["ci_low"]):+.2f}, {float(effect(ds,n)["ci_high"]):+.2f}]' for n in ['exact_contribution','different_contribution','net_change','restored_on_frozen']] for ds in DS])
para('冻结组的异类 Fisher 增加，令同图 / 异类的比值变小。同图的贡献点估计反而略正。双向交换只换异类门，其他特征、方向和预算固定，结果支持异类门向量足以传递这部分效应。由于固定终端算子对同一输入和门是确定的，交换本身不能进一步证明究竟是均值、饱和还是门分给哪些目标的作用。')
fig('12_strength_components','图 12｜冻结减可训练的贡献及 95% 区间。每条横线是同一概念块重采样的配对区间。点可能位于百分位区间之外，原样保留而不人为修正。')
para('<b>整体强度项占异类门总贡献约 82.8%（THINGS）、67.5%（ImageNet）。</b>路径先把两组保均值压顶到 0.95，再把冻结组缩放至可训练组全批均值。这是路径依赖的贡献比例，分母不是总历史变化。压顶区间含零，不能据此证明饱和毫无作用。')
para('ImageNet 等均值后的总残余为 −1.54 pp [−3.26, +0.56]。其中类内置换后保留的分布项为 −0.59 pp [−0.77, −0.19]，门值在同类目标间的分配项为 −0.95 pp [−2.76, +1.03]。分布项仍包含逐类均值差、触发率及正门形状，不能直接称为触发率效应。总残余区间含零，分布项的非零证据不等于总残余已经稳定。')
source('门值交换与分解','learning_curve_20261113/异类门值交换与Fisher分解.md');end()
section(10,'两种条件的正确率都提高了，为什么同图优势反而缩小？')
fig('13_ba_correction','图 13｜THINGS 可训练编码器的 1× 与 8× 比较。右图为配对差及 95% 区间。这不是冻结与可训练编码器的比较。')
table(['THINGS 的平衡准确率','1 倍训练规模','8 倍训练规模','8 倍减 1 倍'],[
 ['同图预览后识别目标','72.78%','73.06%','+0.28 个百分点'],
 ['同概念异图预览后识别目标','72.22%','73.61%','+1.39 个百分点'],
 ['同图减同概念异图（先在各规模内作差）','约 +0.56 个百分点','约 −0.56 个百分点','−1.11 个百分点（用未舍入数计算）']])
para('旧报告的 −1.11 pp 指 [同图 BA − 同概念异图 BA] 在两个数据规模之间的变化。实际同图 BA 增加约 +0.28 pp，同概念异图增加约 +1.39 pp，因此两者差距缩小。这正是本周解读所有 Fisher 与读出结果时需要保持的区分。')
end()
section(11,'下一步最小验证：三个新增对照，尚未执行')
para('继续复用缓存，沿用原四层网络、同图门值和固定解码器，由当前目标决定 FC2 调节方向。先让每个类别的平均门值相等，再分别改变门开启的试次比例，以及这些试次间门值强弱的分布。T / F 在此指已完成压顶、全批等均值后的可训练 / 冻结门。')
table(['条件','逐类均值 μ','触发比例 p','门开启时的相对强弱分布 q','状态'],[
 ['A：可训练基线','μT','pT','qT','已有'],['B：冻结基线','μF','pF','qF','已有'],
 ['C：逐类均值匹配','μT','pF','qF','拟新增'],['D：覆盖交换','μT','pF','qT','拟新增'],['E：形状交换','μT','pT','qF','拟新增']])
para('因为 μ = p × 门开启时的平均强度，固定 μ 后提高触发比例必须降低正门平均幅度。因此 D 相对 A 应解释为“扩大覆盖并补偿正门均值”。C−B 检查逐类强度，A/D/E/C 构成覆盖×形状交叉对照，再计算交互。所有条件共用类内随机分配和概念块重采样。')
para('先检查构造值能否保持在允许范围内，不能悄悄截断并破坏均值匹配。若需要统一降强度，应单列相对原结果的桥接变化。报告绝对类内、类间、Fisher、重复抑制、BA、margin 和配对区间；有限样本无法区分时明确保留不确定性，不继续扩大参数搜索。')
end()
section(12,'证据强弱与可追溯性')
table(['需要保留的限制','对结论的影响'],[
 ['早期进行过多轮、大量配置搜索','筛选成功不等于新样本成功，必须区分开发结果、冻结后复核和事后解释。'],
 ['扩样测试图像隔离，概念有历史曝光重叠','可评价本轮训练—验证—测试隔离下的泛化，不能称所有测试概念从未见于研究过程。'],
 ['冻结 FC2 比较和门交换复用已查看测试批量','是机制复核，不能作为额外独立确认。'],
 ['三个随机种子的均值、4,000 次类内概念块重采样','区间主要反映当前概念抽样，不完整覆盖随机初始化和未来数据分布的不确定性。'],
 ['定标与干预不同','模型阈值、斜率只用验证集；缓存等均值等操作利用测试批量统计量，只用于检验机制的模拟对照，不是部署定标。'],
 ['如何解读跨过零的区间','区间内同时包含改善与变差的可能。判断两种方法是否足够接近，需要预先规定多大的差别可以接受。'],
 ['终端 FC2 与早期 Conv4 原通道实验不同','本周主线是真实四层轨迹上的终端 FC2 操作。未获得对应 8× 模型 Conv4 原通道 rep4 中晚期类别结果，不能将终端收益写成卷积通道普遍 sharpening。'],
 ['在线接线只抽查少量 trial','抽查试次中，在线活动与离线重算的最大绝对差约为 10⁻⁶ 量级，支持缓存回放接线一致性，不替代群体效应复核。']])
para('本报告图 1–4 采用归档 Markdown 表格中已经报告的点估计，其配对区间可查阅原始实验报告。图 5–13 直接读取结果 CSV。本次修订还依据候选图像清单核查了可用于扩样的概念数。报告内的来源链接指向原文，数据文件指纹见 source_manifest.json，绘图脚本保存于 analysis/build_hebbian_week_report_20261114.py。')
p('<details><summary>展开：机器可核查的数据来源</summary>')
table(['文件','SHA-256'],[[html.escape(k),v] for k,v in sources.items()]);p('</details>');end()
css='''*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#233241;font:16px/1.8 "Microsoft YaHei",sans-serif}main{max-width:1120px;margin:32px auto;background:white;padding:54px 64px}header{padding-bottom:32px;border-bottom:3px solid #2369a1}h1{font-size:36px;line-height:1.45;margin:16px 0}h2{font-size:25px;line-height:1.5;margin:8px 0 18px}p{margin:14px 0}.lead{font-size:19px;color:#536779}.eyebrow{font-size:13px;letter-spacing:1px;color:#64798c}section{padding:36px 0;border-bottom:1px solid #dce4eb}figure{margin:24px 0}figure svg{width:100%;height:auto}figcaption,.source{font-size:13px;color:#586c7c}a{color:#2369a1;text-decoration-thickness:1px;text-underline-offset:3px}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.7;margin:16px 0}th{text-align:left;background:#eef3f7;color:#29465d}td,th{padding:12px 14px;border-bottom:1px solid #dce4eb;vertical-align:top}td:first-child{font-weight:600;min-width:90px}.flow{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:18px 0}.flow div{flex:1;background:#edf3f7;padding:16px;text-align:center;border-radius:6px}.flow span{color:#2369a1;font-size:24px}details{margin:20px 0}summary{cursor:pointer;color:#2369a1;padding:12px;background:#f2f6f9}nav{display:flex;flex-wrap:wrap;gap:8px;padding:22px 0}nav a{font-size:13px;padding:5px 10px;background:#edf3f7;text-decoration:none;border-radius:4px}@media(max-width:700px){main{padding:24px 18px;margin:0}h1{font-size:27px}h2{font-size:21px}.flow{flex-wrap:wrap}.flow div{min-width:120px}table{font-size:13px}}@media print{body{background:white}main{max-width:none;margin:0;padding:0}nav{display:none}section{break-before:auto}figure,table tr{break-inside:avoid}h2{break-after:avoid}details{display:block}details>summary{display:none}details>*{display:block}}'''
nav='<nav aria-label="章节目录">'+''.join(f'<a href="#s{i}">{i:02d} {t}</a>' for i,t in enumerate(['结论','实验地图','指标释义','门差对照','熟悉度回路','诊断路线','扩样与预算','FC2 结果','门值归因','BA 更正','下一步','证据边界'],1))+'</nav>'
doc='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hebbian 实验图解报告｜2026.11.12–14</title><style>'+css+'</style></head><body><main>'+nav+'\n'.join(parts)+'</main></body></html>'
(OUT/'report.html').write_text(doc,encoding='utf-8')
(OUT/'source_manifest.json').write_text(json.dumps({'period':['2026-11-12','2026-11-14'],'source_sha256':sources,'figures':13,'new_experiments':False},ensure_ascii=False,indent=2),encoding='utf-8')
assert doc.count('<figure ')==13
assert doc.count('<section ')==12
for ds in DS:
 assert abs(sum(val(effect(ds,k)) for k in ['saturation_path','mean_strength_path','equal_mean_unsaturated'])-val(effect(ds,'different_contribution')))<1e-10
 assert abs(sum(val(effect(ds,k)) for k in ['remaining_distribution','remaining_assignment'])-val(effect(ds,'equal_mean_unsaturated')))<1e-10
print('Report:',OUT/'report.html')
print('13 figures; 12 sections; source hashes recorded; additive decompositions verified.')
