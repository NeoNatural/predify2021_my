"""Data-dependent presentation axes and three-carrier representation atlases."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FuncFormatter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.filtered_cv_20261027 import OUT,DOC,verify_freeze
from analysis.report_filtered_cv_20261027 import LABELS,COLORS,TEXT
from analysis.refresh_2d_20261028 import MARKERS

DEST=DOC/'figures_2d_v3';DS=('imagenet','things');NS=('clean','noisy');CS=('hebb','pcoder','joint');IDS=('exact','same_concept','same_category','different_category');LS=('conv4','conv5','fc1','fc2')
EFFECTS=[('normalized_trace_effect_pct','Normalized trace change (%)'),('within_cosine_delta_x1e3','Within cosine difference (x1e3)'),('normalized_fisher_effect_pct','Normalized Fisher change (%)')]
RAW=[('normalized_trace_variance','Normalized trace variance'),('within_class_cosine','Within-class cosine'),('normalized_fisher_ratio','Normalized Fisher ratio'),('raw_trace_variance','Raw trace variance (diagnostic)'),('normalized_between_scatter','Normalized between scatter')]
SAVED=[];AXES=[];SOURCES={};VERTICES=0

def read(path):
    SOURCES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def bounds(values,reference=None,physical=None):
    a=np.asarray(values,float).ravel();a=a[np.isfinite(a)];assert len(a)
    lo,hi=float(a.min()),float(a.max());span=max(hi-lo,abs(hi)*1e-5,1e-8)
    if reference is not None:lo=min(lo,reference);hi=max(hi,reference)
    pad=max(span*.05,(physical[1]-physical[0])*.005 if physical else 0,1e-10)
    quantum=10.**np.floor(np.log10(max(pad/4,1e-12)))
    lower=np.floor((lo-pad)/quantum)*quantum;upper=np.ceil((hi+pad)/quantum)*quantum
    if physical:lower=max(physical[0],lower);upper=min(physical[1],upper)
    if upper<=lower:upper=lower+quantum
    assert lower<=a.min() and upper>=a.max()
    return float(lower),float(upper)

class Axis:
    def __init__(self,values,reference=None,physical=None,compress=False):
        a=np.asarray(values,float).ravel();a=a[np.isfinite(a)]
        self.lo,self.hi=bounds(a,reference,physical);self.reference=reference;self.observed=[float(a.min()),float(a.max())]
        qlo,qhi=np.quantile(a,[.1,.9]);span=max(float(np.ptp(a)),1e-10)
        middle=(qlo+qhi)/2;half=max((qhi-qlo)*.56,span*.1)
        qlo=max(self.lo,middle-half);qhi=min(self.hi,middle+half)
        self.focus=[float(qlo),float(qhi)];self.knots=np.array([self.lo,qlo,qhi,self.hi],float)
        # Remove degenerate knots and derive a limited display budget for sparse tails.
        if not self.lo<qlo<qhi<self.hi:self.knots=np.array([self.lo,self.hi]);self.slopes=np.ones(1)
        else:
            widths=np.diff(self.knots);self.slopes=np.ones(3);central=widths[1]
            if compress:
                for j in (0,2):
                    if widths[j]>.18*(self.hi-self.lo):self.slopes[j]=np.clip(.25*central/widths[j],.12,.5)
        self.mapped=np.r_[self.knots[0],self.knots[0]+np.cumsum(np.diff(self.knots)*self.slopes)]
        self.compressed=[dict(low=float(self.knots[i]),high=float(self.knots[i+1]),ratio=float(s)) for i,s in enumerate(self.slopes) if s<1]
        grid=np.linspace(self.lo,self.hi,101);assert (np.diff(self.forward(grid))>0).all();np.testing.assert_allclose(self.inverse(self.forward(grid)),grid,atol=max(1e-10,(self.hi-self.lo)*1e-12))
    def forward(self,y):
        y=np.asarray(y,float);i=np.clip(np.searchsorted(self.knots,y,side='right')-1,0,len(self.slopes)-1)
        return self.mapped[i]+(y-self.knots[i])*self.slopes[i]
    def inverse(self,y):
        y=np.asarray(y,float);i=np.clip(np.searchsorted(self.mapped,y,side='right')-1,0,len(self.slopes)-1)
        return self.knots[i]+(y-self.mapped[i])/self.slopes[i]
    def meta(self):return dict(limits=[self.lo,self.hi],observed=self.observed,focus=self.focus,compressed_segments=self.compressed)
    def apply(self,ax,unit='',record=None):
        if self.compressed:ax.set_yscale('function',functions=(self.forward,self.inverse))
        ax.set_ylim(self.lo,self.hi)
        if self.compressed:
            # Greedily retain readable true-value ticks in display coordinates.
            cand=[]
            if self.reference is not None and self.lo<=self.reference<=self.hi:cand.append(float(self.reference))
            cand+=list(MaxNLocator(nbins=5).tick_values(*self.focus));cand+=list(self.focus)+[self.lo,self.hi]
            for s in self.compressed:cand+=list(MaxNLocator(nbins=3).tick_values(s['low'],s['high']))
            selected=[];gap=.075*(self.mapped[-1]-self.mapped[0])
            for v in cand:
                if self.lo<=v<=self.hi and not any(abs(self.forward(v)-self.forward(x))<gap for x in selected):selected.append(float(v))
            ax.set_yticks(sorted(selected))
            for j,s in enumerate(self.compressed):
                ax.axhspan(s['low'],s['high'],color='#dedede',alpha=.45,zorder=-5)
                text=f"{s['low']:.3g}-{s['high']:.3g}{unit}: x{s['ratio']:.2f}"
                at=self.inverse((self.forward(s['low'])+self.forward(s['high']))/2)
                ax.text(.985,at,text,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=7.8,color='#555555')
        else:ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,p:f'{y:.4g}'))
        if self.reference is not None and self.lo<=self.reference<=self.hi:ax.axhline(self.reference,color='#777777',ls='--',lw=.8,zorder=-1)
        if record is not None:AXES.append(dict(**record,**self.meta()))

def curve(ax,x,y,identity,label=None,band=None):
    global VERTICES
    x=np.asarray(x);y=np.asarray(y);a,=ax.plot(x,y,color=COLORS[identity],marker=MARKERS[identity],lw=1.3,markersize=3.2,label=label or TEXT[identity])
    assert np.array_equal(a.get_xdata(),x) and np.array_equal(a.get_ydata(),y);VERTICES+=len(x)
    if band is not None:ax.fill_between(x,band[0],band[1],color=COLORS[identity],alpha=.10)

def finish(fig,axs,title,subtitle,footer):
    h,l=axs.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.938),ncol=4,frameon=False,fontsize=9)
    fig.suptitle(title,y=.994,fontsize=14);fig.text(.5,.963,subtitle,ha='center',fontsize=9);fig.text(.5,.009,footer,ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.035,1,.93))

def save(fig,name,category):
    DEST.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):fig.savefig(DEST/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);SAVED.append(dict(name=name,category=category))

def readout_figures(summary,styles=None):
    for dataset in DS:
        for method in ('t9','mvpa'):
            for measure,field,label,reference,physical in [('acc_cv','accuracy_fold_mean','Accuracy (%)',25.,(0.,100.)),('acc_oof','accuracy_oof_pooled','Pooled OOF accuracy (%)',25.,(0.,100.)),('auc','roc_auc_fold_mean','Macro OVR ROC-AUC',.5,(0.,1.))]:
                ys={}
                for carrier in CS:
                    for noise in NS:
                        for identity in IDS:
                            rows=sorted([r for r in summary[dataset] if r['readout']==method and r['noise']==noise and r['condition']==f'{carrier}_{identity}'],key=lambda r:int(r['timestep']))
                            assert len(rows)==15;ys[carrier,noise,identity]=np.array([float(r[field]) for r in rows])
                for style in ('adaptive','linear','late','exact_delta','exact_delta_late'):
                    if styles is not None and style not in styles:continue
                    delta=style.startswith('exact');late=style in ('late','exact_delta_late');ts=np.arange(5,15) if late else np.arange(15)
                    series={(c,n,i):(ys[c,n,'exact']-ys[c,n,i] if delta else v)[ts] for (c,n,i),v in ys.items() if not delta or i!='exact'}
                    full=np.concatenate(list(series.values()));ref=0. if delta else None if late else reference
                    spec=Axis(full,ref,None if delta else physical,compress=style=='adaptive')
                    fig,axs=plt.subplots(3,2,figsize=(12.8,10.5),sharex=True,sharey=True)
                    filename=f'{dataset}_{method}_{measure}_{style}'
                    for ri,c in enumerate(CS):
                        for ni,n in enumerate(NS):
                            ax=axs[ri,ni]
                            for i in IDS[1:] if delta else IDS:curve(ax,ts,series[c,n,i],i,'Exact - '+TEXT[i] if delta else None)
                            spec.apply(ax,'%' if measure!='auc' and not delta else '',dict(figure=filename,panel=f'{c}_{n}',measure=measure,style=style))
                            ax.set_title(f'{LABELS[c]} | {n}',fontsize=11);ax.grid(alpha=.16);ax.set_xticks(ts);ax.tick_params(labelsize=9);ax.set_xlim(ts[0]-.15,ts[-1]+.15)
                            if ni==0:ax.set_ylabel(('Exact - other (pp)' if measure!='auc' else 'Exact - other (delta AUC)') if delta else label)
                            if ri==2:ax.set_xlabel('Target timestep')
                    estimator='5-fold arithmetic mean' if measure!='acc_oof' else 'Pooled OOF correct proportion; N=64 clean / 192 noisy'
                    head='Fold-specific t9 heads frozen across target time' if method=='t9' else 'Time/condition-specific cross-validated heads'
                    foot='Axes are data-dependent per figure; all original vertices retained. No smoothing or significance inferred from separation.'
                    if spec.compressed:foot='Grey spans are data-derived compression regions; ticks show true values. Visual slopes are not comparable rates.'
                    finish(fig,axs,f'{dataset.upper()} | {method.upper()} | {measure} | {style.replace("_"," ")}',estimator+' | '+head,foot)
                    save(fig,filename,'readout')
        print(dataset,'READOUT FIGURES COMPLETE',flush=True)

def representation_figures(activ,geom,effects):
    for dataset in DS:
        for noise in NS:
            specifications=[('output_raw','Mean output (raw units)','activation','output_mean'),('legacy_raw','Mean legacy x_tmp (raw units)','activation','legacy_x_tmp_mean'),
                ('output_delta','Output - different (raw units)','effect','output_mean_delta'),('output_pct','Output change vs different (%)','effect','output_mean_effect_pct'),
                ('legacy_pct','Legacy change vs different (%)','effect','legacy_x_tmp_mean_effect_pct')]
            specifications += [(k,label,'geometry',k) for k,label in RAW]
            specifications += [(k,label,'effect',k) for k,label in EFFECTS]
            for key,label,kind,field in specifications:
                fig,axs=plt.subplots(4,3,figsize=(15,12),sharex=True,sharey='row');filename=f'{dataset}_{noise}_{key}_three_modes'
                for li,layer in enumerate(LS):
                    src=activ[dataset] if kind=='activation' else geom[dataset] if kind=='geometry' else effects[dataset]
                    rows=[r for r in src if r['noise']==noise and r['layer']==layer and (kind!='effect' or r['metric']==field)]
                    vals=[float(r[k]) for r in rows for k in (('value','low','high') if kind=='effect' else (field,))]
                    spec=Axis(vals,0. if kind=='effect' else None)
                    for ci,carrier in enumerate(CS):
                        ax=axs[li,ci]
                        for identity in IDS[:-1] if kind=='effect' else IDS:
                            selected=sorted([r for r in rows if r['carrier']==carrier and r['identity']==identity],key=lambda r:int(r['timestep']))
                            assert len(selected)==15;y=[float(r['value'] if kind=='effect' else r[field]) for r in selected]
                            band=([float(r['low']) for r in selected],[float(r['high']) for r in selected]) if kind=='effect' else None
                            curve(ax,np.arange(15),y,identity,band=band)
                        spec.apply(ax,record=dict(figure=filename,panel=f'{layer}_{carrier}',measure=field,style='linear_auto'))
                        ax.set_title(f'{LABELS[carrier]} | {layer.upper()}',fontsize=10);ax.set_xticks([0,2,4,6,8,10,12,14]);ax.tick_params(labelsize=9);ax.grid(alpha=.16)
                        if ci==0:ax.set_ylabel(label,fontsize=9)
                        if li==3:ax.set_xlabel('Target timestep')
                sub='Priming minus own different-category baseline; marginal 95% concept-bootstrap bands' if kind=='effect' else 'Actual metric values; local linear limits per layer, shared across all 3 modes in each row'
                finish(fig,axs,f'{dataset.upper()} | {noise} | {label} | ALL THREE MEMORY MODES',sub,'Raw axes need not start at zero. Overlapping means do not establish identical full representations; no between-carrier tests.')
                save(fig,filename,'representation')
            # Directly readable FC2 overview: 3 measures x all 3 memory modes.
            for variant in ('raw','effects'):
                fields=RAW[:3] if variant=='raw' else EFFECTS;fig,axs=plt.subplots(3,3,figsize=(15,10),sharex=True,sharey='row');filename=f'{dataset}_{noise}_fc2_sharpening_{variant}_three_modes'
                for ri,(field,label) in enumerate(fields):
                    rows=[r for r in (geom[dataset] if variant=='raw' else effects[dataset]) if r['layer']=='fc2' and r['noise']==noise and (variant=='raw' or r['metric']==field)]
                    spec=Axis([float(r[k]) for r in rows for k in ((field,) if variant=='raw' else ('value','low','high'))],0. if variant=='effects' else None)
                    for ci,carrier in enumerate(CS):
                        ax=axs[ri,ci]
                        for identity in IDS if variant=='raw' else IDS[:-1]:
                            s=sorted([r for r in rows if r['carrier']==carrier and r['identity']==identity],key=lambda r:int(r['timestep']))
                            curve(ax,range(15),[float(r[field] if variant=='raw' else r['value']) for r in s],identity,band=None if variant=='raw' else ([float(r['low']) for r in s],[float(r['high']) for r in s]))
                        spec.apply(ax,record=dict(figure=filename,panel=f'{field}_{carrier}',measure=field,style='linear_auto'));ax.set_title(LABELS[carrier],fontsize=11);ax.grid(alpha=.16);ax.set_xticks([0,2,4,6,8,10,12,14]);ax.tick_params(labelsize=9)
                        if ci==0:ax.set_ylabel(label,fontsize=9)
                        if ri==2:ax.set_xlabel('Target timestep')
                finish(fig,axs,f'{dataset.upper()} | {noise} | FC2 sharpening {variant} | ALL THREE MODES','Same numeric y-axis within each metric row; all original timepoints','Intervals, where shown, are marginal descriptive concept-bootstrap intervals; no new significance tests.')
                save(fig,filename,'fc2_overview')
            print(dataset,noise,'THREE-MODE REPRESENTATION FIGURES COMPLETE',flush=True)

def link(name,label):
    p=(DEST/name).resolve().as_posix();return f'[{label}](<{p}.png>) · [SVG](<{p}.svg>)'

def zoom_representations(activ,effects):
    for dataset in DS:
        for noise in NS:
            for carrier in CS:
                for kind in ('activation','sharpening'):
                    fields=[('output_mean','Actual output mean'),('output_mean_delta','Output - different (raw units)'),('output_mean_effect_pct','Output change (%)')] if kind=='activation' else EFFECTS
                    fig,axs=plt.subplots(4,3,figsize=(15,12),sharex=True);filename=f'{dataset}_{noise}_{carrier}_{kind}_local_zoom'
                    for li,layer in enumerate(LS):
                        for fi,(field,label) in enumerate(fields):
                            raw=field=='output_mean';source=activ[dataset] if raw else effects[dataset]
                            rows=[r for r in source if r['noise']==noise and r['carrier']==carrier and r['layer']==layer and (raw or r['metric']==field)]
                            vals=[float(r[k]) for r in rows for k in ((field,) if raw else ('value','low','high'))];spec=Axis(vals,None if raw else 0.)
                            ax=axs[li,fi]
                            for identity in IDS if raw else IDS[:-1]:
                                s=sorted([r for r in rows if r['identity']==identity],key=lambda r:int(r['timestep']))
                                curve(ax,range(15),[float(r[field] if raw else r['value']) for r in s],identity,band=None if raw else ([float(r['low']) for r in s],[float(r['high']) for r in s]))
                            spec.apply(ax,record=dict(figure=filename,panel=f'{layer}_{field}',measure=field,style='independent_linear_zoom'))
                            ax.set_title(f'{layer.upper()} | {label}',fontsize=10);ax.set_xticks([0,2,4,6,8,10,12,14]);ax.tick_params(labelsize=9);ax.grid(alpha=.16)
                            if fi==0:ax.set_ylabel(layer.upper())
                            if li==3:ax.set_xlabel('Target timestep')
                    finish(fig,axs,f'{dataset.upper()} | {noise} | {LABELS[carrier]} | {kind}: LOCAL DATA-BASED AXES','Each panel uses its own data and intervals; compare numeric axis labels, not visual heights','All 4 layers; means and paired effects are distinct. Bands, where present, are marginal descriptive 95% intervals.')
                    save(fig,filename,'mode_specific_zoom')

def gallery():
    audit=json.loads((DOC/'activation_overlap_audit.json').read_text(encoding='utf8'));overlap=read(DOC/'activation_overlap_audit.csv')
    lines=['# 激活重叠核查与自适应纵轴图册','',
        '本轮仅核查保存的实验数据并更新绘图，不修改模型、实验条件、读出或统计估计。全部激活与sharpening主图在同一图中展示Hebbian-only、PCoder carry-only、Joint三种模式。',
        '', '## 为什么平均激活看起来重叠','',
        '逐图记录重算的组均值与结果表一致（最大误差小于2×10⁻¹⁶）；保存的完整诊断特征图在CPU上独立计算的均值也与记录吻合。不同内容并非同一份数据；所检查的全部配对FC2完整时间轨迹均不完全相同。Hebbian/Joint的预览后记忆权重非零，PCoder-only权重为零。',
        '', '均值确实很接近：下表是相对各自different-category基线的最大组均值百分比差，遍历全部层、内容、噪声与t0–t14。逐图正负变化在组平均中存在抵消；完整向量的变化可能明显大于单个全局均值的变化。原先从0起的纵轴进一步掩盖了这些小差异。',
        '', '| 数据集 | 模式 | 最大绝对组均值变化% | t5–t14最大绝对变化% |','|---|---|---:|---:|']
    for d in DS:
        for c in CS:
            r=[x for x in overlap if x['dataset']==d and x['carrier']==c]
            lines.append(f'| {d} | {LABELS[c]} | {max(float(x["max_abs_group_effect_pct"]) for x in r):.3f} | {max(float(x["late_max_abs_group_effect_pct"]) for x in r):.3f} |')
    lines+=['', '卷积Hebbian模块对每通道整张特征图施加同一缩放，因此全输出的平均值在数学上应接近缩放后的通道均值x_tmp；FC旁路时二者相同。二者近似相等本身不说明记录错误。上述核查支持“曲线不是被复制”，但不把均值接近直接解释成记忆没有作用，也不据此确认某种生物机制。',
        '', f'[逐图/完整特征图核查](<{(DOC/"activation_overlap_audit.json").resolve().as_posix()}>) · [分层差异和抵消统计](<{(DOC/"activation_overlap_audit.csv").resolve().as_posix()}>)',
        '', '## ACC与AUC：每张图的数据决定坐标','',
        '每个数据集×读出×估计量独立计算坐标范围：上限是实际最大值加小幅留白，不再固定100%/1。全时程保留机会水平25%/0.5；后期放大和差值图按对应数据取范围。图内不同记忆模式与噪声面板共用同一坐标，不同图之间的数值范围可不同。',
        '', '压缩段由该图全部观测值的10%–90%分位主体范围及两端长度确定，留白和压缩比例也随该图数据变化。仅足够长的尾部区间压缩；灰底及真实端点/倍率直接标在图上。实际时间点和数值不变，无平滑、插值或峰值选择。非线性坐标下斜率不是变化速率，视觉分离不是统计显著。',
        '', 'ACC仍分五折算术均值与OOF正确比例；AUC仅用逐折macro OVR平均，不混合跨折margin计算pooled AUC。clean为64个目标、noisy为64×3个预测。']
    for d in DS:
        lines+=['',f'### {d.upper()} 读出','', '| 读出/估计量 | 数据驱动压缩 | 线性核对 | 后期线性 | Exact差值 | 后期差值 |','|---|---|---|---|---|---|']
        for method in ('t9','mvpa'):
            for measure,label in [('acc_cv','ACC五折均值'),('acc_oof','ACC OOF'),('auc','ROC-AUC五折均值')]:
                lines.append(f'| {method} / {label} | '+' | '.join(link(f'{d}_{method}_{measure}_{s}',t) for s,t in [('adaptive','自适应压缩'),('linear','线性'),('late','t5–t14'),('exact_delta','Exact−其他'),('exact_delta_late','t5–t14差值')])+' |')
    lines+=['','## 激活与sharpening：每张图都包含三种记忆模式','',
        '原始激活采用每层数据的局部线性范围；基线中心化原始差值及百分比变化另列，并保留已有bootstrap区间。原始量图不添加从差值区间借来的“置信带”。四层图每行一层、每列一种记忆模式；每行内三列范围相同。']
    for d in DS:
        for n in NS:
            lines+=['',f'### {d.upper()} / {n}','', '| 内容 | 三种模式与四层 |','|---|---|']
            for key,label in [('output_raw','原始output mean（局部纵轴）'),('legacy_raw','原始legacy x_tmp'),('output_delta','output−different原始差值与CI'),('output_pct','output相对变化%与CI'),('legacy_pct','legacy相对变化%与CI')]:
                lines.append(f'| {label} | {link(f"{d}_{n}_{key}_three_modes",label)} |')
            for field,label in RAW+EFFECTS:lines.append(f'| {label} | {link(f"{d}_{n}_{field}_three_modes",label)} |')
            lines+=['',link(f'{d}_{n}_fc2_sharpening_raw_three_modes','FC2三项原始指标×三种模式'),' ',link(f'{d}_{n}_fc2_sharpening_effects_three_modes','FC2三项效果及CI×三种模式')]
            lines+=['','各模式独立放大版（每面板按本身数据及区间缩放，不能按跨图视觉高度比较幅度）：','', '| 模式 | 激活：原始mean/原始差值/百分比 | 四层sharpening效果及区间 |','|---|---|---|']
            for c in CS:lines.append(f'| {LABELS[c]} | {link(f"{d}_{n}_{c}_activation_local_zoom","激活独立放大")} | {link(f"{d}_{n}_{c}_sharpening_local_zoom","sharpening独立放大")} |')
    lines+=['','## 直接展示：ImageNet noisy，三种记忆模式','']
    for name,label in [('imagenet_t9_acc_cv_adaptive','ACC自适应坐标'),('imagenet_t9_auc_adaptive','AUC自适应坐标'),('imagenet_noisy_output_delta_three_modes','四层激活基线中心化差值'),('imagenet_noisy_fc2_sharpening_effects_three_modes','三种记忆模式的FC2 sharpening')]:
        lines+=['',f'![{label}](<{(DEST/(name+".png")).resolve().as_posix()}>)']
    lines+=['',f'[每张图的范围与压缩参数](<{(DOC/"adaptive_axes_20261028.json").resolve().as_posix()}>)']
    (DOC/'自适应坐标与三种记忆图册.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def main():
    verify_freeze();summary={d:read(DOC/d/'cv_summary.csv') for d in DS};activ={d:read(DOC/d/'activation_timeline.csv') for d in DS};geom={d:read(DOC/d/'geometry_timeline.csv') for d in DS};effects={d:read(DOC/d/'representation_effects.csv') for d in DS}
    readout_figures(summary);representation_figures(activ,geom,effects);zoom_representations(activ,effects)
    (DOC/'adaptive_axes_20261028.json').write_text(json.dumps(AXES,indent=2),encoding='utf8');gallery()
    for p,h in SOURCES.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    result=dict(passed=True,figures=SAVED,figure_count=len(SAVED),axes_panels=len(AXES),original_vertices_checked=VERTICES,source_hashes=SOURCES,models_and_original_results_unchanged=True)
    (DOC/'adaptive_plot_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('ADAPTIVE PLOTS COMPLETE',len(SAVED),flush=True)

if __name__=='__main__':main()
