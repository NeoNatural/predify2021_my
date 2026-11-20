"""Evaluate clean recurrent convolution transfer; no parameter selection on development."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from predify_experiments.supplementary import load
from analysis.conv_fc_transfer_20261106 import OUT,DOC,CASES,BRANCHES,SOURCES,METHODS,read,sha,writejson
from analysis.report_conv_preview_temporary_20261106 import geometry,bootstrap,table

LAYERS=('conv4','conv5','fc1','fc2','conv4_input')
WINDOWS={'early':slice(0,5),'middle':slice(5,10),'late':slice(10,15)}


def main():
    metrics=[];audit=[];graphs=[];traces=[];intervals=[];allsets={};readouts=[]
    for ds in ('imagenet','things'):
        spec=read(OUT/ds/'manifest.json');assert read(OUT/ds/'complete.json')['complete']
        labels=np.array([t['category_index'] for t in spec['development']]);concepts=np.array([t['concept'] for t in spec['development']])
        assert set(concepts).isdisjoint(t['concept'] for t in spec['training'])
        heads=load(OUT/ds/'fixed_heads.pckl');assert sha(OUT/ds/'fixed_heads.pckl')==read(OUT/ds/'fixed_heads.json')['sha256']
        for case in CASES:
            for hist in (('none','exact','different_category') if case=='clip' else ('exact','different_category')):
                file=OUT/ds/'development'/case/f'{hist}.npz';info=read(file.with_suffix('.json'));assert sha(file)==info['sha256']
                with np.load(file) as z:values={k:z[k] for k in z.files}
                assert np.array_equal(values['labels'],labels)
                assert np.array_equal(values['indices'],[t['dataset_index'] for t in spec['development']])
                for layer in ('fc1','fc2'):
                    a=values[layer];n,t,d=a.shape;scores=heads[layer].decision_function(a.reshape(n*t,d)).reshape(n,t,-1)
                    pred=scores.argmax(2);correct=pred==labels[:,None]
                    other=scores.copy();other[np.arange(n)[:,None],np.arange(t)[None,:],labels[:,None]]=-np.inf
                    margin=scores[np.arange(n)[:,None],np.arange(t)[None,:],labels[:,None]]-other.max(2)
                    values[layer+'_accuracy']=correct.astype(float);values[layer+'_margin']=margin
                    for win,sl in WINDOWS.items():
                        ss=scores[:,sl].mean(1);mm=margin[:,sl].mean(1)
                        readouts.append(dict(dataset=ds,case=case,history=hist,layer=layer,window=win,
                            time_average_accuracy=float(correct[:,sl].mean()),
                            window_mean_score_BA=balanced_accuracy_score(labels,ss.argmax(1)),
                            macro_auc=float(np.mean([roc_auc_score(labels==k,ss[:,k]) for k in np.unique(labels)])),
                            margin=float(mm.mean())))
                allsets[ds,case,hist]=values
                for tr in info['trials']:
                    assert tr['reset_zero'] and tr['before_target']==tr['after_target']
                    expected=0 if case=='clip' or case.endswith('_scale') else 2
                    assert tr['writes']==expected
                    audit.append(dict(dataset=ds,case=case,history=hist,target=tr['target'],writes=tr['writes'],frozen=True))
                    graphs.append(dict(dataset=ds,case=case,history=hist,target=tr['target'],**tr['graph']))
                    for t,steps in enumerate(tr['trace']):
                        for step in steps:traces.append(dict(dataset=ds,case=case,history=hist,target=tr['target'],t=t,**step))
                for layer in LAYERS:
                    for t in range(15):
                        metrics.append(dict(dataset=ds,case=case,history=hist,layer=layer,t=t,
                            mean=float(values[layer+'_mean'][:,t].mean()),density=float(values[layer+'_density'][:,t].mean()),
                            **geometry(values[layer+'_gram'][t],labels),
                            accuracy=float(values[layer+'_accuracy'][:,t].mean()) if layer.startswith('fc') else np.nan,
                            margin=float(values[layer+'_margin'][:,t].mean()) if layer.startswith('fc') else np.nan))
        for case in CASES:
            for win,sl in WINDOWS.items():
                for layer in ('conv4','fc1','fc2'):
                    for metric in ('mean','density')+(('accuracy','margin') if layer.startswith('fc') else ()):
                        key=layer+'_'+metric
                        for control in ('different_category','none'):
                            e=allsets[ds,case,'exact'][key][:,sl].mean(1)
                            c=allsets[ds,case,control if control!='none' or case=='clip' else 'exact'][key][:,sl].mean(1) if control!='none' else allsets[ds,'clip','none'][key][:,sl].mean(1)
                            v=100*(1-e/np.maximum(c,1e-12)) if metric=='mean' else e-c
                            lo,hi=bootstrap(v,concepts)
                            intervals.append(dict(dataset=ds,case=case,window=win,layer=layer,metric=metric,
                                control=control,effect=float(v.mean()),ci_low=lo,ci_high=hi))
    frame=pd.DataFrame(metrics);frame['window']=frame.t.map(lambda t:'early' if t<5 else 'middle' if t<10 else 'late')
    window=frame.groupby(['dataset','case','history','layer','window'],as_index=False).mean(numeric_only=True)
    comparison=[]
    for row in window.itertuples(index=False):
        ref=window[(window.dataset==row.dataset)&(window.case=='clip')&(window.history==row.history)&(window.layer==row.layer)&(window.window==row.window)].iloc[0]
        comparison.append(dict(dataset=row.dataset,case=row.case,history=row.history,layer=row.layer,window=row.window,
            mean_change_pct=100*(row.mean/ref['mean']-1),density_change_pp=100*(row.density-ref['density']),
            fisher_change_pct=100*(row.fisher/ref['fisher']-1),within_change_pct=100*(row.within/ref['within']-1),
            accuracy_change_pp=100*(row.accuracy-ref['accuracy']),margin_change=row.margin-ref['margin']))
    contrasts=[]
    for ds in ('imagenet','things'):
        for case in CASES:
            for win,sl in WINDOWS.items():
                for layer in LAYERS:
                    base=window[(window.dataset==ds)&(window.layer==layer)&(window.window==win)]
                    e=base[(base.case==case)&(base.history=='exact')].iloc[0]
                    d=base[(base.case==case)&(base.history=='different_category')].iloc[0]
                    n=base[(base.case=='clip')&(base.history=='none')].iloc[0]
                    # Suppression uses per-target ratios, as in the FC reference study.
                    ee=allsets[ds,case,'exact'][layer+'_mean'][:,sl].mean(1)
                    dd=allsets[ds,case,'different_category'][layer+'_mean'][:,sl].mean(1)
                    nn=allsets[ds,'clip','none'][layer+'_mean'][:,sl].mean(1)
                    contrasts.append(dict(dataset=ds,case=case,window=win,layer=layer,
                        suppression_ED=float(np.mean(100*(1-ee/np.maximum(dd,1e-12)))),
                        suppression_EN=float(np.mean(100*(1-ee/np.maximum(nn,1e-12)))),
                        fisher_ED_pct=100*(e.fisher/d.fisher-1),fisher_EN_pct=100*(e.fisher/n.fisher-1),
                        within_ED_pct=100*(e.within/d.within-1),accuracy_ED_pp=100*(e.accuracy-d.accuracy),
                        margin_ED=e.margin-d.margin))
    con=pd.DataFrame(contrasts);comp=pd.DataFrame(comparison);fid=pd.DataFrame(readouts)
    for name,value in [('step_metrics',frame),('window_metrics',window),('vs_same_history_clip',comp),
        ('history_contrasts',con),('paired_concept_bootstrap',pd.DataFrame(intervals)),('fixed_svm',fid),
        ('weight_audit',pd.DataFrame(audit)),('weight_statistics',pd.DataFrame(graphs)),('update_traces',pd.DataFrame(traces))]:
        value.to_csv(DOC/f'{name}.csv',index=False,encoding='utf-8-sig')
    for name in ('protocol.json','imagenet_calibration.json','things_calibration.json'):
        assert (DOC/name).exists()
    freezechecks={}
    for ds in ('imagenet','things'):
        f=read(OUT/ds/'freeze.json')
        assert f['proxy']==read(OUT/ds/'proxy/spatial_sae.json')['proxy_hash']
        assert f['calibration']==sha(OUT/ds/'calibration.json') and f['heads']==sha(OUT/ds/'fixed_heads.pckl')
        freezechecks[ds]=dict(proxy_parameters=2112,training=len(read(OUT/ds/'manifest.json')['calibration_training']))
    assert len(audit)==1260
    summary=con[(con.window=='early')&con.case.str.endswith('_full')&(con.layer.isin(['conv4','fc2']))]
    quality=comp[(comp.window=='early')&(comp.history=='exact')&(comp.layer=='fc2')&comp.case.str.endswith('_full')]
    text=['# FC 后期机制移植到卷积层：直接读取与轻量 SAE',
        '完成 clean 小批量：ImageNet 16、THINGS 20 个目标；1260 条独立递归序列。原模型和 FC Hebbian 未改变。'
        '四种 branch/source × full、pool、shuffle、scale，以及 clip 的三种历史。非 clip 的无 preview 结果在严格恒等检查后共享同一基线。',
        '直接读取使用512通道；SAE为逐空间柱512→固定32→64 latent，2112个学习参数。'
        'SAE只产生增益，不替换原512通道卷积表征及4096维FC。两种图来源共享同一数据集编码器；'
        '每trial两次preview写入，target冻结，保留柱内充分候选和柱间mutual top32。',
        'ε=.03、λ=.03、侧向5步直接沿用FC后期设置，没有按开发结果搜索。SAE只用训练折24/30个目标及其preview拟合；'
        '各分支/来源的q RMS与κ₀也只用这些训练活动估计，开发提取前冻结。此采样是小规模训练侧定标，并非全量SAE训练或内层交叉验证。',
        '## 早期 full 的历史对比',
        'suppression_ED=逐目标平均100×(1−Exact/Different)，正数为抑制；EN为Exact相对灰屏无preview。'
        'Fisher来自完整图单位范数几何，表内为Exact相对相应历史的百分比。',
        table(summary,['dataset','case','layer','suppression_ED','suppression_EN','fisher_ED_pct','fisher_EN_pct','within_ED_pct','accuracy_ED_pp','margin_ED'],5),
        '## 早期 full 相对相同preview历史的 clip：FC2',
        table(quality,['dataset','case','mean_change_pct','fisher_change_pct','within_change_pct','accuracy_change_pp','margin_change'],5),
        '## 对照与解释边界',
        'pool关闭局部重分配但保留同一匹配阈值规则，后续活动和阈值可分开；不是逐步固定阈值的因果干预。'
        'shuffle仅在两段preview完成后打乱通道/latent对应，Exact/Different共享同一排列，并已核对打乱前权重与full相同。'
        'scale是独立重跑整个递归网络，逐网络步匹配full的全局rep4均值；不强行匹配下游FC均值，'
        '也不把此后验均值对照当作独立产生的历史机制。',
        '原通道和latent分支在不同维数上建图，因此同样top32不表示相同稀疏比例、有效边数或存储量。'
        'weight_statistics.csv报告实际图；update_traces.csv报告调制强度、阈值、κ和截零前平衡。',
        '主要SVM为仅用folds1–3原FC clean t9训练的固定StandardScaler+LinearSVC(C=1)，分别读取FC1/FC2实际响应。'
        'fixed_svm.csv同时报告时间平均准确率、窗口平均决策分数的BA/宏AUC，以及margin。没有使用latent或SAE重建读出。',
        '该批量沿用已经反复查看的开发概念，不能作为全新独立终测。所有时间窗口和对照在CSV中。'
        'paired_concept_bootstrap.csv是概念成块的描述性95%区间，未作多重比较校正；时间步不是独立样本。',
        f'完整结果位于 `{OUT}`；详细方法见实验方案.md。']
    (DOC/'实验结果.md').write_text('\n\n'.join(text),encoding='utf8')
    plot(con)
    complete=dict(complete=True,sequences=len(audit),groups=70,trial_reset=True,target_frozen=True,fc_hebb=False,
        preview_only_write=True,proxy_and_calibration_frozen=True,parameters=freezechecks)
    writejson(DOC/'complete.json',complete);writejson(DOC/'status.json',complete)
    print(summary.round(5).to_string(index=False),flush=True);print(quality.round(5).to_string(index=False),flush=True)


def plot(con):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    q=con[(con.window=='early')&(con.layer=='conv4')&con.case.str.endswith(('_full','_pool','_shuffle','_scale'))]
    fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for ax,ds in zip(axes,('imagenet','things')):
        labels=[f'{b}/{s}' for b in BRANCHES for s in SOURCES];xx=np.arange(4)
        for j,m in enumerate(METHODS):
            vv=[float(q[(q.dataset==ds)&(q.case==f'{b}_{s}_{m}')].suppression_ED.iloc[0]) for b in BRANCHES for s in SOURCES]
            ax.bar(xx+(j-1.5)*.2,vv,width=.19,label=m)
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(xx,labels);ax.set_title(ds)
        ax.set_ylabel('Conv4 Exact vs Different suppression (%)');ax.legend(fontsize=8)
    fig.savefig(DOC/'early_suppression_controls.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
