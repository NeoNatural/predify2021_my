"""Presentation-only refresh: original vertices, honest CV/OOF labels, raw representation plots."""
from pathlib import Path
import csv,json,math,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.filtered_cv_20261027 import OUT,DOC,verify_freeze
from analysis.report_filtered_cv_20261027 import COLORS,TEXT,LABELS

DEST=DOC/'figures_2d_v2'
DATASETS=('imagenet','things');NOISES=('clean','noisy');CARRIERS=('hebb','pcoder','joint')
IDENTITIES=('exact','same_concept','same_category','different_category');LAYERS=('conv4','conv5','fc1','fc2')
MARKERS=dict(exact='o',same_concept='s',same_category='^',different_category='D')
GEOM=[('normalized_trace_variance','Normalized trace variance'),('within_class_cosine','Within-class cosine'),('normalized_fisher_ratio','Normalized Fisher ratio')]
EFFECTS=[('normalized_trace_effect_pct','Normalized trace change (%)'),('within_cosine_delta_x1e3','Within cosine difference (x1e3)'),('normalized_fisher_effect_pct','Normalized Fisher change (%)')]
SAVED=[];SOURCE_HASHES={};POINT_CHECKS=0

def read(path):
    SOURCE_HASHES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def forward(y):
    y=np.asarray(y,dtype=float)
    return np.where(y<10,y,np.where(y<=45,10+.25*(y-10),18.75+(y-45)))

def inverse(y):
    y=np.asarray(y,dtype=float)
    return np.where(y<10,y,np.where(y<=18.75,10+4*(y-10),45+(y-18.75)))

def line(ax,x,y,identity,label=None):
    global POINT_CHECKS
    x=np.array(x);y=np.array(y)
    artist,=ax.plot(x,y,color=COLORS[identity],marker=MARKERS[identity],markersize=3.4,markeredgewidth=.6,lw=1.4,label=label or TEXT[identity])
    assert np.array_equal(artist.get_xdata(),x) and np.array_equal(artist.get_ydata(),y)
    POINT_CHECKS+=len(x)
    return artist

def save(fig,name):
    DEST.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):fig.savefig(DEST/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);SAVED.append(name)

def finish(fig,axs,title,subtitle,footer,legend=True):
    if legend:
        handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.935),ncol=4,fontsize=10,frameon=False)
    fig.suptitle(title,fontsize=15,y=.994);fig.text(.5,.961,subtitle,ha='center',fontsize=10)
    fig.text(.5,.012,footer,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.93))

def validate_accuracy(summary):
    checked=0;max_error=0.
    for dataset in DATASETS:
        for noise,n in [('clean',64),('noisy',192)]:
            for method in ('t9','mvpa'):
                for carrier in CARRIERS:
                    for identity in IDENTITIES:
                        condition=f'{carrier}_{identity}'
                        with np.load(OUT/dataset/'cv'/noise/method/f'{condition}.npz') as z:
                            scores=z['scores'];y=z['y'];fold=z['fold']
                        assert len(y)==n
                        correct=scores.argmax(-1)==y[:,None]
                        pooled=100*correct.mean(0)
                        mean=np.mean([100*correct[fold==f].mean(0) for f in range(5)],axis=0)
                        rows=sorted([r for r in summary[dataset] if r['noise']==noise and r['readout']==method and r['condition']==condition],key=lambda r:int(r['timestep']))
                        assert len(rows)==15
                        for field,expected in [('accuracy_fold_mean',mean),('accuracy_oof_pooled',pooled)]:
                            err=float(np.max(np.abs(np.array([float(r[field]) for r in rows])-expected)));max_error=max(max_error,err);assert err<1e-10
                        checked+=15
    a=np.linspace(0,100,1001);assert np.all(np.diff(forward(a))>0);np.testing.assert_allclose(inverse(forward(a)),a,atol=1e-12)
    assert abs((forward(45)-forward(10))/35-.25)<1e-12
    assert abs((forward(75)-forward(45))/30-1)<1e-12
    return dict(score_timepoint_checks=checked,max_accuracy_reconstruction_error=max_error,clean_predictions=64,noisy_predictions=192,
                compression_ratio=.25,compressed_true_interval=[10,45],continuous=True,inverse_roundtrip_passed=True)

def accuracy_figures(summary):
    for method in ('t9','mvpa'):
        allrows=[r for d in DATASETS for r in summary[d] if r['readout']==method]
        late=[float(r[field]) for r in allrows if int(r['timestep'])>=5 for field in ('accuracy_fold_mean','accuracy_oof_pooled')]
        late_limits=[5*math.floor((min(late)-2)/5),min(100,5*math.ceil((max(late)+2)/5))]
        for agg,field in [('cvmean','accuracy_fold_mean'),('oof','accuracy_oof_pooled')]:
            def series(dataset,noise,carrier,identity):
                rows=sorted([r for r in summary[dataset] if r['readout']==method and r['noise']==noise and r['condition']==f'{carrier}_{identity}'],key=lambda r:int(r['timestep']))
                return np.array([float(r[field]) for r in rows])
            differences=[series(d,n,c,'exact')-series(d,n,c,i) for d in DATASETS for n in NOISES for c in CARRIERS for i in IDENTITIES[1:]]
            delta_limit=math.ceil(max(np.abs(v).max() for v in differences)*1.08*2)/2 or .5
            delta_late_limit=math.ceil(max(np.abs(v[5:]).max() for v in differences)*1.12*2)/2 or .5
            for dataset in DATASETS:
                for style in ('compressed','linear','late_linear','exact_differences','exact_differences_late'):
                    fig,axs=plt.subplots(3,2,figsize=(12.8,10.5),sharex=True,sharey=True)
                    difference=style.startswith('exact_');late_view=style.endswith('late') or style=='late_linear';ts=np.arange(5,15) if late_view else np.arange(15)
                    for ri,carrier in enumerate(CARRIERS):
                        for col,noise in enumerate(NOISES):
                            ax=axs[ri,col]
                            for identity in IDENTITIES[1:] if difference else IDENTITIES:
                                y=series(dataset,noise,carrier,identity)
                                if difference:y=series(dataset,noise,carrier,'exact')-y
                                line(ax,ts,y[ts],identity,('Exact - '+TEXT[identity]) if difference else TEXT[identity])
                            if difference:
                                bound=delta_late_limit if late_view else delta_limit;ax.set_ylim(-bound,bound);ax.axhline(0,color='#777777',ls='--',lw=.8)
                            elif style=='compressed':
                                ax.set_yscale('function',functions=(forward,inverse));ax.set_ylim(10,100);ax.set_yticks([10,25,45,55,65,75,85,95,100]);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
                                ax.axhspan(10,45,color='#e7e7e7',alpha=.6,zorder=-5);ax.text(.98,14, '10-45%: height x1/4',transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=8,color='#555555')
                                ax.axhline(25,color='#777777',ls='--',lw=.9)
                            else:
                                ax.set_ylim(*(late_limits if late_view else [0,100]))
                                if not late_view:ax.axhline(25,color='#777777',ls='--',lw=.9)
                            ax.set_xlim(ts[0]-.15,ts[-1]+.15);ax.set_xticks(ts);ax.tick_params(labelsize=9);ax.grid(alpha=.16);ax.set_title(f'{LABELS[carrier]} | {noise}',fontsize=11)
                            if col==0:ax.set_ylabel('Exact - other accuracy (pp)' if difference else 'Accuracy (%)',fontsize=10)
                            if ri==2:ax.set_xlabel('Target timestep',fontsize=10)
                    estimator='Arithmetic mean of 5 folds' if agg=='cvmean' else 'Pooled OOF correct / N x 100%; N=64 clean, 192 noisy'
                    readout='Fold-specific heads trained at t9, frozen across t0-t14' if method=='t9' else 'Condition/time-specific 5-fold MVPA'
                    footer='Original observed timesteps; straight segments; no smoothing or peak selection. Descriptive differences are not significance.'
                    if style=='compressed':footer='Continuous compressed y-axis: 10-45% at 1/4 height; above 45% linear. Visual slopes are not rates; separation is not significance.'
                    finish(fig,axs,f'{dataset.upper()} | {method.upper()} | {style.replace("_"," ")}',estimator+' | '+readout,footer)
                    save(fig,f'{dataset}_{method}_{agg}_{style}')

def raw_activation_figures(activation):
    for field,label,suffix in [('output_mean','Mean module output (raw units)','output_activation_raw'),('legacy_x_tmp_mean','Mean legacy x_tmp (raw units)','legacy_activation_raw')]:
        limits={l:(0,max(float(r[field]) for d in DATASETS for r in activation[d] if r['layer']==l)*1.08) for l in LAYERS}
        for dataset in DATASETS:
            for carrier in CARRIERS:
                fig,axs=plt.subplots(4,2,figsize=(12.8,12.5),sharex=True,sharey='row')
                for li,layer in enumerate(LAYERS):
                    for ni,noise in enumerate(NOISES):
                        ax=axs[li,ni]
                        for identity in IDENTITIES:
                            rows=sorted([r for r in activation[dataset] if r['carrier']==carrier and r['identity']==identity and r['noise']==noise and r['layer']==layer],key=lambda r:int(r['timestep']))
                            line(ax,np.arange(15),[float(r[field]) for r in rows],identity)
                        ax.set_ylim(*limits[layer]);ax.set_title(f'{layer.upper()} | {noise}',fontsize=11);ax.set_xticks(range(15));ax.tick_params(labelsize=9);ax.grid(alpha=.16)
                        if ni==0:ax.set_ylabel(label,fontsize=10)
                        if li==3:ax.set_xlabel('Target timestep')
                finish(fig,axs,f'{dataset.upper()} | {LABELS[carrier]} | {label}', 'Absolute group means; 4 layers; all 4 preview contents; clean once, noisy averaged over 3 seeds', 'Raw means have no invented confidence bands. Relative-effect bootstrap intervals are provided separately.')
                save(fig,f'{dataset}_{carrier}_{suffix}')

def representation_figures(geometry,effects):
    for dataset in DATASETS:
        for carrier in CARRIERS:
            for mode in ('raw','effects','diagnostics'):
                fields=GEOM if mode=='raw' else EFFECTS if mode=='effects' else [('raw_trace_variance','Raw trace variance'),('normalized_between_scatter','Normalized between scatter')]
                ncols=len(fields)*2;fig,axs=plt.subplots(4,ncols,figsize=(3.7*ncols,12.8),sharex=True)
                for li,layer in enumerate(LAYERS):
                    for ni,noise in enumerate(NOISES):
                        for fi,(field,label) in enumerate(fields):
                            ax=axs[li,ni*len(fields)+fi]
                            for identity in IDENTITIES[:-1] if mode=='effects' else IDENTITIES:
                                source=effects[dataset] if mode=='effects' else geometry[dataset]
                                rows=sorted([r for r in source if r['carrier']==carrier and r['identity']==identity and r['noise']==noise and r['layer']==layer and (mode!='effects' or r['metric']==field)],key=lambda r:int(r['timestep']))
                                y=[float(r['value'] if mode=='effects' else r[field]) for r in rows];line(ax,np.arange(15),y,identity)
                                if mode=='effects':ax.fill_between(range(15),[float(r['low']) for r in rows],[float(r['high']) for r in rows],color=COLORS[identity],alpha=.10)
                            if mode=='effects':ax.axhline(0,color='#777777',ls='--',lw=.7)
                            ax.set_title(f'{noise} | {label}',fontsize=10);ax.set_xticks([0,2,4,6,8,10,12,14]);ax.tick_params(labelsize=9);ax.grid(alpha=.14)
                            if fi==0 and ni==0:ax.set_ylabel(layer.upper(),fontsize=12)
                            if li==3:ax.set_xlabel('Target timestep',fontsize=10)
                            # Pair clean/noisy domains at each layer/metric, using both datasets and all carriers.
                            globalrows=[r for d in DATASETS for r in (effects[d] if mode=='effects' else geometry[d]) if r['layer']==layer and (mode!='effects' or r['metric']==field)]
                            vals=[float(r[k]) for r in globalrows for k in (('value','low','high') if mode=='effects' else (field,))];lo=min(vals);hi=max(vals);pad=max((hi-lo)*.07,1e-8)
                            ax.set_ylim(lo-pad,hi+pad)
                subtitle='Priming - different category; marginal descriptive 95% concept-bootstrap intervals' if mode=='effects' else 'Raw geometric metric values; not percentage effects; all preview contents'
                footer='Full flattened representations. No smoothing, peak selection, or new hypothesis tests. Activity reduction alone is not sharpening.'
                finish(fig,axs,f'{dataset.upper()} | {LABELS[carrier]} | Sharpening {mode}',subtitle,footer)
                save(fig,f'{dataset}_{carrier}_sharpening_{mode}_all_layers')

def sharpening_example():
    dataset='imagenet';carrier='joint';layer='fc2';rows=read(DOC/dataset/'representation_effects.csv')
    fig,axs=plt.subplots(3,2,figsize=(12.8,10.5),sharex=True)
    for mi,(metric,label) in enumerate(EFFECTS):
        values=[float(r[k]) for r in rows if r['layer']==layer and r['metric']==metric for k in ('value','low','high')]
        lo=min(values);hi=max(values);pad=max((hi-lo)*.07,1e-8)
        for ni,noise in enumerate(NOISES):
            ax=axs[mi,ni]
            for identity in IDENTITIES[:-1]:
                selected=sorted([r for r in rows if r['carrier']==carrier and r['noise']==noise and r['identity']==identity and r['layer']==layer and r['metric']==metric],key=lambda r:int(r['timestep']))
                line(ax,np.arange(15),[float(r['value']) for r in selected],identity)
                ax.fill_between(range(15),[float(r['low']) for r in selected],[float(r['high']) for r in selected],color=COLORS[identity],alpha=.12)
            ax.set_ylim(lo-pad,hi+pad);ax.axhline(0,color='#777777',lw=.7,ls='--');ax.grid(alpha=.16);ax.set_title(noise);ax.set_xticks(range(15));ax.tick_params(labelsize=9)
            if ni==0:ax.set_ylabel(label)
            if mi==2:ax.set_xlabel('Target timestep')
    finish(fig,axs,'IMAGENET | Joint | FC2 sharpening effects: enlarged example','Each priming content minus different category; original 15 timesteps','Shading: marginal descriptive 95% concept-bootstrap intervals; not simultaneous significance bands.')
    save(fig,'imagenet_joint_fc2_sharpening_effects_zoom')

def local_link(name,label=None,ext='png'):
    result=f'[{label or name}](<{(DEST/(name+"."+ext)).resolve().as_posix()}>)'
    if ext=='png':result+=f' · [SVG](<{(DEST/(name+".svg")).resolve().as_posix()}>)'
    return result

def gallery():
    lines=['# 本轮2D绘图更新：ACC、激活值与sharpening','',
      '本图册使用同一批已完成实验结果。只更新展示、补充描述性差值，不重新拟合模型或新增显著性检验。',
      '', '## 统计口径与纵轴','',
      '本轮是五折交叉验证、四种内容（Exact、同概念、同大类、different category）。没有neutral的实验数据，故不补画第五条曲线。t9读出是每折训练一次后冻结的头；每张目标只由未见过其概念/预览概念的该折模型预测。',
      '', '主图标注CV fold mean，即五折准确率的算术均值。OOF补充图按全部预测正确比例×100%汇总：clean每点64张目标；noisy每点64×3=192个预测。两者因测试折大小16/16/16/8/8而可能不同，不把CV均值改称pooled ACC。',
      '', '压缩版为连续变换：10%–45%的显示高度压缩到线性的1/4，45%以上保持线性，10%–100%范围在所有数据集/面板一致。实际曲线高于75%，因此保留到100%，不截去数据。灰底标注压缩段，虚线为25%机会水平，刻度均为真实准确率。',
      '', '同时提供全程线性版、t5–t14线性放大版、Exact−各其他内容（pp）及后期差值放大版。每种版式跨两数据集、全部载体和噪声使用共同范围；所有原始时间点保留，直接以直线和固定标记连接。重合点不抖动。非线性轴视觉斜率不能用于比较变化速率，曲线分离不等于显著。',
      '', '## ACC图','']
    for dataset in DATASETS:
        lines+=['',f'### {dataset.upper()}','', '| 读出 / 汇总 | 压缩轴 | 全程线性 | t5–t14线性 | Exact差值 | 后期差值 |','|---|---|---|---|---|---|']
        for method in ('t9','mvpa'):
            for agg,label in [('cvmean','五折均值（主）'),('oof','OOF pooled（补充）')]:
                lines.append('| '+method+' / '+label+' | '+' | '.join(local_link(f'{dataset}_{method}_{agg}_{s}',s) for s in ('compressed','linear','late_linear','exact_differences','exact_differences_late'))+' |')
        name=f'{dataset}_t9_cvmean_compressed';lines+=['',f'![{dataset} t9压缩轴主图](<{(DEST/(name+".png")).resolve().as_posix()}>)']
    lines+=['','## 激活值：原始mean与历史x_tmp','',
      '以下是实际激活均值（不是相对变化百分比，也不是mean(abs)的替代名称），每图包含Conv4、Conv5、FC1、FC2及clean/noisy。原始均值不套用差值的bootstrap区间；旧版相对different-category的效果与区间仍单独保留。']
    for dataset in DATASETS:
        lines+=['',f'### {dataset.upper()} 激活','', '| 载体 | 模块输出原始mean | 历史x_tmp原始mean | 输出相对变化与CI |','|---|---|---|---|']
        for carrier in CARRIERS:
            old=DOC/'figures'/f'{dataset}_{carrier}_output_activation.png'
            lines.append(f'| {LABELS[carrier]} | {local_link(f"{dataset}_{carrier}_output_activation_raw","四层output mean")} | {local_link(f"{dataset}_{carrier}_legacy_activation_raw","四层legacy mean")} | [相对变化](<{old.resolve().as_posix()}>) |')
    lines+=['','## Sharpening：四层原始值、效果与区间','',
      'raw图给normalized trace variance、within-class cosine、normalized Fisher ratio实际值；effects图给相对different-category的normalized trace%、cosine差值×1000及Fisher%，阴影为边际描述性95%概念bootstrap区间。diagnostics另给raw trace variance与normalized between scatter。全部图包含四层；指标单位分列，不将激活降低单独称作sharpening。',
      '', '| 数据集 | 载体 | 三项原始指标 | 三项效果与CI | 两项诊断 |','|---|---|---|---|---|']
    for dataset in DATASETS:
        for carrier in CARRIERS:
            lines.append(f'| {dataset} | {LABELS[carrier]} | '+ ' | '.join(local_link(f'{dataset}_{carrier}_sharpening_{mode}_all_layers',label) for mode,label in [('raw','原始值'),('effects','效果与区间'),('diagnostics','诊断值')])+' |')
    lines+=['','### 逐层放大：sharpening效果与区间','', '| 数据集 | 载体 | Conv4 | Conv5 | FC1 | FC2 |','|---|---|---|---|---|---|']
    for dataset in DATASETS:
        for carrier in CARRIERS:
            links=[f'[{layer.upper()}](<{(DOC/"figures"/f"{dataset}_{carrier}_{layer}_sharpening.png").resolve().as_posix()}>)' for layer in LAYERS]
            lines.append(f'| {dataset} | {LABELS[carrier]} | '+' | '.join(links)+' |')
    lines+=['', '## 直接查看示例：ImageNet Joint','']
    for name,label in [('imagenet_joint_output_activation_raw','四层实际输出激活'),('imagenet_joint_sharpening_effects_all_layers','四层sharpening效果与区间')]:
        lines+=['',f'![{label}](<{(DEST/(name+".png")).resolve().as_posix()}>)']
    example=DEST/'imagenet_joint_fc2_sharpening_effects_zoom.png'
    lines+=['',f'![FC2 sharpening放大示例](<{example.resolve().as_posix()}>)']
    lines+=['','所有新图同时保存PNG和同名SVG，SVG适合放大查看密集面板。', '', f'[原始完整实验报告](<{(DOC/"实验结果.md").resolve().as_posix()}>)']
    (DOC/'2D绘图更新与图册.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def main():
    verify_freeze()
    summary={d:read(DOC/d/'cv_summary.csv') for d in DATASETS};activation={d:read(DOC/d/'activation_timeline.csv') for d in DATASETS}
    geometry={d:read(DOC/d/'geometry_timeline.csv') for d in DATASETS};effects={d:read(DOC/d/'representation_effects.csv') for d in DATASETS}
    audit=validate_accuracy(summary);accuracy_figures(summary);print('ACC FIGURES COMPLETE',flush=True)
    raw_activation_figures(activation);print('RAW ACTIVATION FIGURES COMPLETE',flush=True)
    representation_figures(geometry,effects);sharpening_example();print('SHARPENING FIGURES COMPLETE',flush=True);gallery()
    for path,expected in SOURCE_HASHES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    audit.update(passed=True,plots=len(SAVED),formats=['png','svg'],actual_vertices_checked=POINT_CHECKS,source_hashes=SOURCE_HASHES,scientific_sources_unchanged=True,identities=IDENTITIES,generated_names=SAVED)
    (DOC/'plot_2d_refresh_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in audit.items() if k not in ('source_hashes','generated_names')}),flush=True)

if __name__=='__main__':main()
