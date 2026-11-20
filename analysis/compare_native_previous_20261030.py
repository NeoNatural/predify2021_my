"""Descriptive early-rise/peak comparison; no model refitting or new significance tests."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import csv,json,sys,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OLD=ROOT/'docs/tripartite_filtered_cv_20261027'
NEW=ROOT/'docs/native_categories_readouts_20261029'
OUT=NEW/'comparison_20261030'
CS=('hebb','pcoder','joint');IDS=('exact','same_concept','same_category','different_category')
METRICS={'acc':'accuracy_fold_mean','auc':'roc_auc_fold_mean'}
COLORS={'exact':'#ce4446','same_concept':'#d78c06','same_category':'#218b7c','different_category':'#5b697c'}
LABELS={'hebb':'Hebbian-only','pcoder':'PCoder carry-only','joint':'Joint'}
TEXT={'exact':'Exact','same_concept':'Same concept','same_category':'Same category','different_category':'Different category'}
METHODS={'mvpa':('old_mvpa','new_mvpa'),'t9A':('old_t9','new_t9_mode_specific'),'t9B':('old_t9','new_t9_clean_reference')}

def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def profile(a,tolerance):
    a=np.asarray(a,float);early=a[:5];peak=float(a.max());ep=float(early.max())
    times=np.flatnonzero(np.isclose(a,peak,atol=1e-10,rtol=0));et=np.flatnonzero(np.isclose(early,ep,atol=1e-10,rtol=0))
    near=np.flatnonzero(a>=peak-tolerance-1e-10);diff=np.diff(early)
    return dict(t0=float(a[0]),t1=float(a[1]),t2=float(a[2]),t3=float(a[3]),t4=float(a[4]),
        rise_01=float(a[1]-a[0]),rise_02=float(a[2]-a[0]),rise_04=float(a[4]-a[0]),
        early_ols_slope=float(np.polyfit(np.arange(5),early,1)[0]),early_down_steps=int(np.sum(diff<-1e-10)),
        early_backtracking=float(np.abs(diff).sum()-abs(a[4]-a[0])),early_mean=float(early.mean()),
        early_peak=ep,early_peak_first=int(et[0]),early_peak_all=';'.join(map(str,et)),
        full_peak=peak,full_peak_first=int(times[0]),full_peak_all=';'.join(map(str,times)),
        nearpeak_tolerance=tolerance,nearpeak_first=int(near[0]),nearpeak_last=int(near[-1]),nearpeak_all=';'.join(map(str,near)),
        late_mean=float(a[10:].mean()),t14=float(a[-1]),peak_minus_late=float(peak-a[10:].mean()))

def load_curves():
    curves={};sources={}
    for version,base in [('old',OLD),('new',NEW)]:
        for dataset in ('imagenet','things'):
            for name in ('cv_summary.csv','cv_content_contrasts.csv','cv_fold_metrics.csv'):
                sources[str(base/dataset/name)]=sha(base/dataset/name)
            rows=read(base/dataset/'cv_summary.csv')
            for r in rows:
                key=(dataset,version+'_'+r['readout'],r['noise'],r['condition'])
                if key not in curves:curves[key]={m:np.full(15,np.nan) for m in ('acc','auc','acc_oof')}
                t=int(r['timestep'])
                for metric,field in METRICS.items():curves[key][metric][t]=float(r[field])
                curves[key]['acc_oof'][t]=float(r['accuracy_oof_pooled'])
            for r in read(base/dataset/'cv_content_contrasts.csv'):
                key=(dataset,version+'_'+r['readout'],r['noise'],r['carrier']+'_'+r['identity']);t=int(r['timestep'])
                b=curves[(dataset,version+'_'+r['readout'],r['noise'],r['carrier']+'_different_category')]
                for metric,field in [('acc','delta_accuracy_pp'),('auc','delta_roc_auc')]:
                    assert abs(curves[key][metric][t]-b[metric][t]-float(r[field]))<1e-10
    assert len(curves)==240
    for r in curves.values():
        for a in r.values():assert np.isfinite(a).all()
    return curves,sources

def metrics(curves):
    absolute=[];effects=[];absolute_lookup={};effect_lookup={}
    for (dataset,method,noise,condition),arrays in curves.items():
        carrier=next(c for c in CS if condition.startswith(c+'_'));identity=condition[len(carrier)+1:]
        baseline=curves[dataset,method,noise,carrier+'_different_category']
        for metric,a in arrays.items():
            tol=.005 if metric=='auc' else .5
            p=profile(a,tol);meta=dict(dataset=dataset,method=method,noise=noise,carrier=carrier,identity=identity,metric=metric)
            absolute.append(dict(**meta,**p));absolute_lookup[dataset,method,noise,carrier,identity,metric]=p
            if identity=='different_category':continue
            delta=a-baseline[metric];v=profile(delta,tol)
            v.update(own_peak_height_difference=float(a.max()-baseline[metric].max()),
                contrast_at_condition_peak=float(delta[p['full_peak_first']]),
                baseline_at_condition_peak=float(baseline[metric][p['full_peak_first']]),
                condition_peak_time=p['full_peak_first'],baseline_peak_time=int(np.argmax(baseline[metric])))
            effects.append(dict(**meta,**v));effect_lookup[dataset,method,noise,carrier,identity,metric]=v
    changes=[]
    for dataset in ('imagenet','things'):
        for family,(old,new) in METHODS.items():
            for noise in ('clean','noisy'):
                for c in CS:
                    for i in IDS[:-1]:
                        for metric in ('acc','auc','acc_oof'):
                            a=effect_lookup[dataset,old,noise,c,i,metric];b=effect_lookup[dataset,new,noise,c,i,metric]
                            changes.append(dict(dataset=dataset,family=family,old_method=old,new_method=new,noise=noise,carrier=c,identity=i,metric=metric,
                                **{f'old_{key}':a[key] for key in ('t0','t1','t2','early_mean','early_peak','early_peak_first','late_mean','full_peak','full_peak_first')},
                                **{f'new_{key}':b[key] for key in ('t0','t1','t2','early_mean','early_peak','early_peak_first','late_mean','full_peak','full_peak_first')},
                                **{f'new_minus_old_{key}':b[key]-a[key] for key in ('early_mean','early_peak','late_mean')}))
    write(OUT/'absolute_rise_and_peaks.csv',absolute);write(OUT/'within_mode_content_effects.csv',effects);write(OUT/'old_new_comparison.csv',changes)
    return absolute_lookup,effect_lookup,changes

def plot_job(task):
    dataset,family,metric,kind=task
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    with (OUT/'curves.json').open(encoding='utf8') as f:raw=json.load(f)
    curves={tuple(r['key']):r['arrays'] for r in raw};old,new=METHODS[family]
    # Shared linear range within a figure; rates are never compared on compressed axes.
    fig,axs=plt.subplots(3,2,figsize=(12.8,9.5),sharex=True,sharey=True)
    ts=np.arange(5) if kind=='early_absolute' else np.arange(15);allvals=[]
    for ri,c in enumerate(CS):
        for ni,noise in enumerate(('clean','noisy')):
            ax=axs[ri,ni]
            for method,style,marker in [(old,'--','o'),(new,'-','s')]:
                baseline=np.asarray(curves[dataset,method,noise,c+'_different_category'][metric])
                for identity in IDS[:-1] if kind=='content_delta' else IDS:
                    y=np.asarray(curves[dataset,method,noise,c+'_'+identity][metric]);y=y-baseline if kind=='content_delta' else y
                    ax.plot(ts,y[ts],color=COLORS[identity],linestyle=style,marker=marker,markersize=3,linewidth=1.5,alpha=.7 if method==old else 1.)
                    allvals.extend(y[ts])
            ax.set_title(f'{LABELS[c]} | {noise}');ax.grid(alpha=.18);ax.set_xticks(ts if len(ts)==5 else [0,1,2,3,4,6,8,10,12,14])
            if kind=='content_delta':ax.axhline(0,color='#888888',linewidth=.8)
            if len(ts)==15:
                ax.axvspan(-.1,4.1,color='#e9f2fa',alpha=.55,zorder=-5);ax.axvspan(9.9,14.1,color='#eeeeee',alpha=.4,zorder=-5)
            if ri==2:ax.set_xlabel('Target timestep')
            if ni==0:ax.set_ylabel(('Delta ACC (pp)' if metric=='acc' else 'Delta macro AUC') if kind=='content_delta' else ('Accuracy (%)' if metric=='acc' else 'Macro OVR ROC-AUC'))
    lo,hi=min(allvals),max(allvals);pad=max((hi-lo)*.08,.25 if metric=='acc' else .001)
    for ax in axs.flat:ax.set_ylim(lo-pad,hi+pad)
    heads=[Line2D([0],[0],color=COLORS[i],marker='o',label=TEXT[i]) for i in IDS[:-1] if kind=='content_delta'] if kind=='content_delta' else [Line2D([0],[0],color=COLORS[i],marker='o',label=TEXT[i]) for i in IDS]
    heads += [Line2D([0],[0],color='black',linestyle='--',label='Previous filtered CV'),Line2D([0],[0],color='black',linestyle='-',label='Current native categories')]
    fig.legend(handles=heads,loc='lower center',ncol=3,frameon=False,fontsize=9)
    subtitle='Same MVPA algorithm; different cohorts/tasks' if family=='mvpa' else 'Previous pooled t9 vs current '+('mode-specific t9-A' if family=='t9A' else 'clean-reference t9-B')+'; readout training also differs'
    fig.suptitle(f'{dataset.upper()} | {family} | {metric.upper()} | {kind.replace("_"," ")}\n{subtitle}',fontsize=13)
    fig.text(.5,.10,'Descriptive comparison. Shared linear axes; original vertices only. Blue: t0-t4; grey: t10-t14. No significance claims.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.13,1,.93));name='_'.join(task)
    for ext in ('png','svg'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig);return name

def fmt(x,places=2):return f'{x:+.{places}f}'

def report(absolute,effects):
    def p(dataset,method,noise,c,i,metric='acc'):return effects[dataset,method,noise,c,i,metric]
    def a(dataset,method,noise,c,i,metric='acc'):return absolute[dataset,method,noise,c,i,metric]
    lines=['# 新旧实验比较：早期上升、效应峰值与末期保持','','本报告的“旧版”指最近的外周退化／抗混叠filtered CV实验；“新版”指原生类别扩样实验。重点是预定early t0–t4，峰值描述其次，late t10–t14仅作持续性辅助。所有比较均为描述性，不新增显著性星号、不重拟合分类器、不移动原有统计窗口。',
        '', '## 主要判断','',
        '1. **重复出现的主要形态是初始／早期占优，随后基线追上；不能把它统一称为更大的上升斜率或更高终态。** 某条件t0已经较高时，t0→t4增幅反而可以更小。效应峰值常在t0/t1，绝对准确率峰值却在后面；两种“峰”必须分开。',
        '2. **THINGS中旧版精确重复的大幅MVPA早期增益没有在本轮等幅复现；同大类预览的早期效应保留得更明显。** 例如noisy联合模式exact早期均值由+6.33降至+0.18 pp、早期峰由+9.17降至+2.58 pp；same-category早期峰仍为+10.17 pp（旧版+10.83）。这不是一个所有内容同步缩小的结果。',
        '3. **ImageNet不是全面减弱。** noisy联合模式exact的MVPA早期均值由+0.50升至+2.17 pp，早期峰由+5.42@t1变为+5.83@t0；同概念效应峰由+9.58@t4变为+10.52@t0。当前更明显的是效应提前出现，而非证明终态大幅抬升。',
        '4. **读出依赖必须保留。** 新THINGS noisy exact在MVPA中的早期均值接近零（PCoder +0.07、Joint +0.18 pp），但固定t9-A为+2.32、+2.73 pp，t9-B为+2.45、+2.55 pp。不能因此说“exact完全无作用”，也不能由固定头的改善直接推论最优条件内可解码性提高。',
        '5. **准确率与ROC-AUC不是同一个结论。** THINGS noisy联合same-category的MVPA早期ΔACC由+6.42变为+4.85 pp，但早期ΔAUC由+0.0012变为+0.0300；不能把全部证据简写为“语义效应变弱”。',
        '', '## 可比性和定义','',
        '| 项目 | 旧版filtered CV | 新版native |','|---|---|---|',
        '| 正式目标 | 两数据集各64图、32概念，每概念2图 | ImageNet320图/80概念；THINGS400图/100概念，每概念4图 |',
        '| 类别 | 两数据集均4类，THINGS严格ImageNet概念映射 | ImageNet4类；THINGS原生5类，取消概念配对 |',
        '| 样本与折 | 已查看目标；五折16/16/16/8/8 | 新图扩样；THINGS正式概念全新；五折均衡 |',
        '| MVPA | 条件×时间×fold独立拟合 | 同一算法，仅标签、样本和fold适配 |',
        '| t9 | 模式×内容共12条件合并拟合一个头/噪声/fold | A：每模式、噪声独立，四内容合并；B：唯一无记忆clean参考 |',
        '| 输入／模型流程 | 外周退化、同一抗混叠、2+2+15、22步灰屏、target W冻结 | 保持上述规则；抽样和噪声种子更新 |',
        '', '旧→新不是同一批样本的配对再测，也不是只改变样本数的单因素实验；THINGS绝对ACC峰高下降尤其不能直接解释为模型退步。AUC机会水平虽都为0.5，也不保证任务难度可比。原始两个数据集不作跨数据集绝对成绩排名。',
        '', '主要ACC口径统一为五折算术均值。旧版折大小不等，已另算OOF pooled敏感性结果；新版二者一致。所有内容效应先减去同版本、同模式、同噪声、同读出的different-category基线。t0为第一次target更新后的状态，不是preview之前。MVPA的时程代表各时点独立可解码性；固定t9时程代表同一边界的时间泛化，不将模型step换算成人类毫秒。',
        '', '峰值均取原始15点或固定early五点，不平滑；并列峰取首次时点并保存全部并列时点。峰高为事后描述，不做峰值显著性检验。“early效应峰”是max(条件−基线)，不等于max(条件)−max(基线)。完整CSV同时保存这两个量和条件自身峰时的同刻差值。',
        '', '## 1．先看早期上升：noisy Exact的绝对ACC','',
        '下表不把更大增量解释成更快：起点越低，剩余提升空间通常越大。新版THINGS的类别数和队列也已变化。',
        '', '| 数据集 | 读出／模式 | 旧 t0→t1→t2→t4 | 新 t0→t1→t2→t4 | 旧／新t0→t4增量 pp |','|---|---|---|---|---|']
    for dataset in ('imagenet','things'):
        for family,(old,new) in [('mvpa',METHODS['mvpa']),('t9-A',METHODS['t9A'])]:
            for c in CS:
                x=a(dataset,old,'noisy',c,'exact');y=a(dataset,new,'noisy',c,'exact')
                seq=lambda v:' → '.join(f'{v[f"t{t}"]:.2f}' for t in (0,1,2,4))
                lines.append(f'| {dataset} | {family}/{LABELS[c]} | {seq(x)} | {seq(y)} | {x["rise_04"]:+.2f} / {y["rise_04"]:+.2f} |')
    lines+=['','ImageNet新版noisy t9-A PCoder exact的t0→t4增量为+21.46 pp，different基线为+33.33 pp；但exact在t0已高12.50 pp，随后基线追上。这种形态更适合称“早期读出提前占优”，而不是“priming令上升斜率更大”。',
        '', '## 2．主要比较：MVPA早期内容效应、峰时和末期辅助','',
        '主表为noisy，clean完整对应表见下一节及CSV。表中峰为early t0–t4最大配对ΔACC；末期为t10–t14配对均值。',
        '', '| 数据集 | 模式 | 内容 | 旧／新early均值 pp | 旧early峰 pp@t | 新early峰 pp@t | 旧／新late均值 pp |','|---|---|---|---|---|---|---|']
    for dataset in ('imagenet','things'):
        for c in CS:
            for i in IDS[:-1]:
                x=p(dataset,'old_mvpa','noisy',c,i);y=p(dataset,'new_mvpa','noisy',c,i)
                lines.append(f'| {dataset} | {LABELS[c]} | {TEXT[i]} | {x["early_mean"]:+.2f} / {y["early_mean"]:+.2f} | {x["early_peak"]:+.2f}@{x["early_peak_first"]} | {y["early_peak"]:+.2f}@{y["early_peak_first"]} | {x["late_mean"]:+.2f} / {y["late_mean"]:+.2f} |')
    lines+=['','新版noisy PCoder末期三种内容效应几乎回到零；联合模式末期也多为约±1 pp内。它支持“早期时程差异往往不转化为明显末期增益”的描述，不能把接近零写成已证明条件等价。',
        '', '### clean敏感性核对：MVPA early均值','',
        '| 数据集 | 模式 | Exact旧→新 pp | 同概念旧→新 pp | 同大类旧→新 pp |','|---|---|---:|---:|---:|']
    for dataset in ('imagenet','things'):
        for c in CS:
            values=[]
            for i in IDS[:-1]:
                x=p(dataset,'old_mvpa','clean',c,i);y=p(dataset,'new_mvpa','clean',c,i);values.append(f'{x["early_mean"]:+.2f} → {y["early_mean"]:+.2f}')
            lines.append('| '+' | '.join([dataset,LABELS[c]]+values)+' |')
    lines+=['','## 3．绝对准确率峰与效应峰分开解释','',
        '| 数据集／noisy MVPA | 内容 | 旧绝对峰%@t | 新绝对峰%@t | 旧／新峰高−末期均值 pp |','|---|---|---|---|---|']
    for dataset in ('imagenet','things'):
        for c in CS:
            for i in ('exact','different_category'):
                x=a(dataset,'old_mvpa','noisy',c,i);y=a(dataset,'new_mvpa','noisy',c,i)
                lines.append(f'| {dataset}/{LABELS[c]} | {TEXT[i]} | {x["full_peak"]:.2f}@{x["full_peak_first"]} | {y["full_peak"]:.2f}@{y["full_peak_first"]} | {x["peak_minus_late"]:.2f} / {y["peak_minus_late"]:.2f} |')
    lines+=['','THINGS noisy PCoder exact旧版91.67%@t5，随后末期均值84.83%，峰高高出末期6.83 pp；新版峰80.42%@t10，末期80.20%，仅相差0.22 pp。前者更像单点较高后回落，后者更接近平台内的小幅差异。不能只凭“t5变t10”判断新版变慢；两者任务不同，且首个精确最高点对离散正确数很敏感。',
        '', '完整表另外给出距峰0.5 pp内的全部ACC时点（AUC为0.005带宽），仅用于显示平台，不当作预设统计阈值或真实收敛时间。',
        '', '旧版折大小不等。五折算术均值与OOF汇总之间，绝对曲线early均值最大相差2.06 pp、峰高最大相差2.50 pp，个别曲线的首个最高点相差8步；新版两种口径相同至浮点误差。主表坚持原主要口径，但这些敏感性结果进一步限制了对单点峰时的速度解释，见fold_vs_oof_sensitivity.csv。',
        '', '## 4．固定t9：旧共同头、新A、新B','',
        '下面均是noisy条件的early内容效应“均值 / 早期峰@t”。旧共同头到新A/B同时改变了训练范围，不能把两者差异单独归因于记忆机制。A与B在同一新版队列上，则明确反映读出训练域依赖。',
        '', '| 数据集 | 模式／内容 | 旧共同t9 | 新t9-A | 新t9-B |','|---|---|---|---|---|']
    for dataset in ('imagenet','things'):
        for c in CS:
            for i in IDS[:-1]:
                vals=[p(dataset,k,'noisy',c,i) for k in ('old_t9','new_t9_mode_specific','new_t9_clean_reference')]
                lines.append('| '+' | '.join([dataset,LABELS[c]+'/'+TEXT[i]]+[f'{v["early_mean"]:+.2f} / {v["early_peak"]:+.2f}@{v["early_peak_first"]}' for v in vals])+' |')
    lines+=['','THINGS noisy联合same-category：A的效应峰为+12.00 pp@t0，B为+3.58 pp@t1；MVPA为+10.17 pp@t0。这说明同大类的早期效应在适配读出和逐时刻MVPA中均可见，但通过clean参考边界迁移时明显减小。B下较迟／较低的读出不能直接解释成类别信息本身丢失或神经状态真正变慢。',
        '', '## 5．ROC-AUC核对：不只看离散正确率','',
        '| 数据集／noisy MVPA | 模式／内容 | 旧／新early ΔAUC | 旧／新early ΔAUC峰@t | 旧／新late ΔAUC |','|---|---|---|---|---|']
    for dataset in ('imagenet','things'):
        for c in ('pcoder','joint'):
            for i in ('exact','same_category'):
                x=p(dataset,'old_mvpa','noisy',c,i,'auc');y=p(dataset,'new_mvpa','noisy',c,i,'auc')
                lines.append(f'| {dataset} | {LABELS[c]}/{TEXT[i]} | {x["early_mean"]:+.4f} / {y["early_mean"]:+.4f} | {x["early_peak"]:+.4f}@{x["early_peak_first"]} / {y["early_peak"]:+.4f}@{y["early_peak_first"]} | {x["late_mean"]:+.4f} / {y["late_mean"]:+.4f} |')
    lines+=['','例如旧THINGS noisy PCoder exact虽然early ΔACC为+2.25 pp，early ΔAUC却为−0.0263；新版对应为+0.07 pp和+0.0085。旧ACC瞬时峰不能单独作为“类别信息提升”的证据。',
        '', '## 图与可重算文件','',
        '[完整比较图册](比较图册.md)；[全部绝对曲线的上升与峰值](absolute_rise_and_peaks.csv)；[全部内容差值及效应峰](within_mode_content_effects.csv)；[逐项旧新比较](old_new_comparison.csv)；[来源与核验](comparison_audit.json)。',
        '', '下图均用线性坐标，保留全部原始点。旧版虚线、新版实线；蓝底为early，灰底为late。无平滑、插值或峰值显著性标记。']
    for dataset in ('imagenet','things'):
        lines+=['',f'![{dataset} MVPA早期绝对上升](figures/{dataset}_mvpa_acc_early_absolute.png)','',f'![{dataset} MVPA内容效应时程](figures/{dataset}_mvpa_acc_content_delta.png)']
    lines+=['','本比较不涉及记忆模式之间的配对检验、排名或协同推断。若要检验“早期峰是否真实改变”，需在同一刺激/类别任务上做读出与抽样因素的受控桥接，并包含完整分类器重拟合及峰选择的成组重采样；本轮没有把描述性差异写成这类确认性结论。',
        '', '更早的未滤波三区实验还改变了目标噪声路径、训练集合、neutral/强度条件及共同头权重，因此未与本报告主表合并。旧版选择依据见[最近旧版报告](../../tripartite_filtered_cv_20261027/实验结果.md)，更早t9背景见[三区t9补充](../../tripartite_t9_20261027/补充结果.md)。']
    (OUT/'新旧结果比较.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    curves,sources=load_curves();absolute,effects,changes=metrics(curves)
    (OUT/'curves.json').write_text(json.dumps([dict(key=list(k),arrays={m:v.tolist() for m,v in a.items()}) for k,a in curves.items()]),encoding='utf8')
    jobs=[(d,family,metric,kind) for d in ('imagenet','things') for metric in ('acc','auc') for family in METHODS for kind in ('absolute','content_delta')]
    jobs += [(d,'mvpa',metric,'early_absolute') for d in ('imagenet','things') for metric in ('acc','auc')]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for name in pool.map(plot_job,jobs):print('FIGURE',name,flush=True)
    report(absolute,effects)
    gallery=['# 新旧比较图册','','全部采用线性坐标；旧虚线、新实线。t9比较同时改变读出训练范围，不能解释成纯记忆机制变化。']
    for d in ('imagenet','things'):
        gallery+=['',f'## {d.upper()}','','| 读出比较 | 指标 | 绝对时程 | 相对different差值 | early绝对放大 |','|---|---|---|---|---|']
        for family in METHODS:
            for metric in ('acc','auc'):
                link=lambda kind:f'[PNG](figures/{d}_{family}_{metric}_{kind}.png) · [SVG](figures/{d}_{family}_{metric}_{kind}.svg)'
                gallery.append('| '+' | '.join([family,metric,link('absolute'),link('content_delta'),link('early_absolute') if family=='mvpa' else '见绝对时程t0–t4'])+' |')
    (OUT/'比较图册.md').write_text('\n'.join(gallery)+'\n',encoding='utf8')
    sensitivity=[]
    for key in absolute:
        dataset,method,noise,c,i,metric=key
        if metric!='acc':continue
        r=absolute[key];o=absolute[dataset,method,noise,c,i,'acc_oof']
        sensitivity.append(dict(dataset=dataset,method=method,noise=noise,carrier=c,identity=i,early_fold_minus_oof=r['early_mean']-o['early_mean'],peak_fold_minus_oof=r['full_peak']-o['full_peak'],first_peak_shift=r['full_peak_first']-o['full_peak_first']))
    write(OUT/'fold_vs_oof_sensitivity.csv',sensitivity)
    test=profile([1,3,3,2,2]+[2]*10,.5)
    assert test['early_peak_first']==1 and test['early_peak_all']=='1;2' and test['early_down_steps']==1 and test['rise_04']==1
    audit=dict(passed=True,source_sha256=sources,script_sha256=sha(__file__),absolute_curve_count=len(curves),metric_profiles=len(absolute),effect_profiles=len(effects),cross_version_rows=len(changes),figures=len(jobs),source_contrasts_recomputed=True,peak_tie_rise_selfcheck_passed=True,classification_refits=0,peak_inference='descriptive only; no significance testing',early=[0,4],late=[10,14],acc_nearpeak_tolerance_pp=.5,auc_nearpeak_tolerance=.005,finished=time.time())
    (OUT/'comparison_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print('COMPARISON COMPLETE',OUT/'新旧结果比较.md',flush=True)

if __name__=='__main__':main()
