"""Absolute four-condition bar plots from existing audited results; no model runs."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/fc_degraded_k128_20261028'
DEST=DOC/'figures'/'four_conditions'
MAIN='both_hard_hard_5_5_frozen'
COLORS=('#A7AFB7','#E5A14F','#4D83B5','#459C80')
CONDITIONS=(
    ('干净','none','baseline'),
    ('退化',None,'baseline'),
    ('干净经Hebbian','none',MAIN),
    ('退化经Hebbian',None,MAIN),
)
MODES=(('predify_noise','高斯噪声'),('fractal_occlusion','分形遮挡'))
STAGES=(('fc1_output','FC1本层'),('fc2_output','最终FC2'))
# field, label, scale, number format, bounded maximum
GROUPS={
    'sharpening':(
        ('activation','平均激活 A（实际幅度）',1.,'.3f',None),
        ('within','类内离散度 W',1.,'.3f',None),
        ('between','类间离散度 B',1.,'.3f',None),
        ('fisher','Fisher：B / W',1.,'.3f',None),
    ),
    'activity':(
        ('nonzero','非零单元比例（%）',100.,'.2f',100.),
        ('weak_activity_share','弱响应活动量占比（%）',100.,'.3g',100.),
        ('effective_dimension','有效维度',1.,'.1f',None),
        ('cosine_to_clean_vanilla','与干净vanilla的配对余弦',1.,'.3f',1.),
    ),
    'svm':(
        ('svm_clean_accuracy','干净训练→测试：SVM BA（%）',100.,'.2f',100.),
        ('svm_clean_auc','干净训练→测试：SVM AUC',1.,'.4f',1.),
        ('svm_refit_accuracy','条件内重拟合：SVM BA（%）',100.,'.2f',100.),
        ('svm_refit_auc','条件内重拟合：SVM AUC',1.,'.4f',1.),
    ),
}
TITLES={'sharpening':'平均激活与表征区分度','activity':'活动分布与干净参考','svm':'SVM功能检查'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    files=[DOC/'summary.json',DOC/'fold_metrics.csv',DOC/'audit.json']
    source_hashes={p.name:sha(p) for p in files}
    summary=json.loads(files[0].read_text(encoding='utf-8'))
    assert json.loads(files[2].read_text(encoding='utf-8'))['passed']
    with files[1].open(encoding='utf-8-sig',newline='') as f:folds=list(csv.DictReader(f))
    summary_index={(r['dataset'],r['mode'],r['name'],r['stage']):r for r in summary}
    plt.rcParams.update({
        'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titleweight':'normal','font.size':10,'pdf.fonttype':42,
    })
    plotted=[]
    def values(dataset,stage,corruption,condition,metric,scale):
        label,mode,name=CONDITIONS[condition]
        mode=corruption if mode is None else mode
        key=(dataset,mode,name,stage)
        selected=sorted([r for r in folds if (r['dataset'],r['mode'],r['name'],r['stage'])==key],key=lambda r:int(r['fold']))
        assert len(selected)==5 and [int(r['fold']) for r in selected]==list(range(5))
        assert all(int(float(r['repeats']))==(1 if mode=='none' else 3) for r in selected)
        actual_field=metric
        if metric.startswith('svm_refit_') and mode=='none':
            actual_field=metric.replace('svm_refit_','svm_clean_')
        a=np.array([float(r[actual_field]) for r in selected])*scale
        assert np.isfinite(a).all()
        assert abs(a.mean()-summary_index[key][actual_field]*scale)<1e-10
        plotted.append(dict(dataset=dataset,stage=stage,corruption=corruption,condition=label,
            input_mode=mode,method=name,metric=metric,actual_source_field=actual_field,scale=scale,
            mean=float(a.mean()),minimum=float(a.min()),maximum=float(a.max()),sd=float(a.std(ddof=1)),
            **{f'fold{i}':float(v) for i,v in enumerate(a)}))
        return a

    for dataset in ('imagenet','things'):
        for group,metrics in GROUPS.items():
            fig,axes=plt.subplots(2,4,figsize=(17,7.5))
            for i,(stage,stage_label) in enumerate(STAGES):
                for j,(field,title,scale,fmt,bound) in enumerate(metrics):
                    ax=axes[i,j]
                    arrays={}
                    for mode_index,(corruption,mode_label) in enumerate(MODES):
                        for c in range(4):
                            arrays[(mode_index,c)]=values(dataset,stage,corruption,c,field,scale)
                    extent=max(float(a.max()) for a in arrays.values())
                    label_pad=max(extent*.025,1e-7)
                    for (mode_index,c),a in arrays.items():
                        x=mode_index+(c-1.5)*.18
                        mean=float(a.mean())
                        ax.bar(x,mean,width=.16,color=COLORS[c],edgecolor='white',linewidth=.7,zorder=2)
                        ax.errorbar(x,mean,yerr=np.array([[mean-a.min()],[a.max()-mean]]),
                            fmt='none',ecolor='#38434A',elinewidth=.85,capsize=2.3,capthick=.85,zorder=3)
                        label_y=float(a.max())+label_pad
                        rotation=0
                        number=format(mean,fmt)
                        if field.endswith('_auc'):
                            label_y+=extent*.055*(c%2)
                        if field=='weak_activity_share':
                            label_y+=extent*.045*(c%2)
                            if c>=2:
                                number=f'{mean:.1e}'
                                rotation=90
                        ax.text(x,label_y,number,ha='center',va='bottom',fontsize=7.8,rotation=rotation,zorder=4)
                    ax.set_xlim(-.48,1.48)
                    ax.set_xticks((0,1),[m[1] for m in MODES],fontsize=9)
                    ax.set_title(title,fontsize=11,pad=10)
                    ax.set_ylim(0,max(extent*1.17,1e-5))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax.tick_params(axis='y',labelsize=8)
                    ax.grid(axis='y',color='#DDE1E5',linewidth=.6,zorder=0)
                    ax.set_axisbelow(True)
                    if j==0:ax.set_ylabel(stage_label,fontsize=12,labelpad=12)
            handles=[Patch(facecolor=color,label=c[0]) for color,c in zip(COLORS,CONDITIONS)]
            fig.legend(handles=handles,loc='upper center',ncol=4,bbox_to_anchor=(.5,.945),frameon=False,fontsize=10)
            fig.suptitle(f'{dataset.upper()}｜四条件比较：{TITLES[group]}',fontsize=16,y=.995)
            fig.text(.5,.018,
                'FC1 corr/k128 → FC2 corr/k32；两层增益0.75、hard、各5步；连接与尺度固定取自干净训练折\n'
                '柱高＝五折均值；误差线＝五折最小—最大范围（非置信区间）；退化3次重复先在各折内平均。高斯σ=0.5，分形强度=0.3。',
                ha='center',va='bottom',fontsize=9,color='#42505A')
            fig.subplots_adjust(left=.065,right=.992,top=.84,bottom=.13,wspace=.28,hspace=.42)
            for extension in ('png','svg','pdf'):
                fig.savefig(DEST/f'{dataset}_{group}_four_conditions.{extension}',dpi=220,facecolor='white')
            plt.close(fig)
    assert len(plotted)==384
    with (DEST/'plotted_values.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(plotted[0]));writer.writeheader();writer.writerows(plotted)
    for p in files:assert sha(p)==source_hashes[p.name]
    images=sorted(DEST.glob('*.png'))
    assert len(images)==6
    notes=['# 干净—退化—干净经Hebbian—退化经Hebbian：柱状图','',
        '使用现有审计通过的绝对值，不重新运行模型、拟合分类器或更改参数。顺序固定为干净、退化、干净经Hebbian、退化经Hebbian。每张图上行为FC1本层，下行为串联最终FC2；横轴分别为高斯噪声与分形遮挡，两组干净柱引用同一组干净参考，并非额外独立样本。','',
        '柱高为五折均值。退化图像的3次随机重复先在各折内平均；误差线为五折最小—最大取值范围，仅描述折间变化，不是置信区间，不据误差线重叠作显著性判断。所有纵轴从0开始；柱上数值是均值。','',
        '## 指标口径','',
        '- 平均激活A保留实际幅度并按大类等权；W/B/Fisher使用逐图L2归一化表征。四条件都画绝对值，不混用不同基线的相对变化。较低W需结合B和Fisher判断。有效维度不预设越高或越低越好。','',
        '- 非零比例按响应>1e-7计算；弱响应活动量占比指低于单图最大响应1%的活动单元贡献的总活动份额，不是弱单元数量比例。','',
        '- 配对余弦始终以同图、同层的原始干净vanilla为参考。它是恢复性诊断，不能单独证明语义恢复。','',
        '- “干净训练→测试”使用各处理在干净训练折拟合的同设置SVM；干净两柱测试干净图片，退化两柱直接迁移测试退化图片。','',
        '- “条件内重拟合”中，干净两柱仍为干净训练/干净测试SVM；退化两柱使用退化训练折重新拟合、退化测试折评估的SVM。两种训练口径分列，均为StandardScaler+LinearSVC(C=1)，均隔离测试概念。','']
    for dataset in ('imagenet','things'):
        notes += [f'## {dataset.upper()}','']
        for group in GROUPS:
            stem=f'{dataset}_{group}_four_conditions'
            notes += [f'### {TITLES[group]}','',
                f'![{dataset} {group}](<{(DEST/(stem+".png")).as_posix()}>)','',
                f'[PNG](<{(DEST/(stem+".png")).as_posix()}>) · [SVG](<{(DEST/(stem+".svg")).as_posix()}>) · [PDF](<{(DEST/(stem+".pdf")).as_posix()}>)','']
    notes += ['## 数据核对','',
        'plotted_values.csv保存384个柱值及各自5折原值、均值、范围、SD、源字段和条件映射。所有柱高逐项与summary.json核对。audit.json记录输入与输出哈希。原实验数据和模型没有修改。','']
    (DEST/'图表说明.md').write_text('\n'.join(notes),encoding='utf-8')
    audit=dict(passed=True,source_hashes=source_hashes,script_sha256=sha(Path(__file__)),
        figures=6,formats=['png','svg','pdf'],bar_values=384,
        order=[c[0] for c in CONDITIONS],statistics='five-fold mean; min-max descriptive whiskers; degraded repeats preaveraged within folds',
        all_bar_means_match_summary=True,axes_start_at_zero=True,readout_policies_separated=True,
        no_model_runs=True,no_classifier_fits=True,source_data_unchanged=True,
        output_hashes={p.name:sha(p) for p in sorted(DEST.iterdir()) if p.is_file() and p.name!='audit.json'})
    (DEST/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: 6 figures x PNG/SVG/PDF; 384 bar values verified; source data unchanged')


if __name__=='__main__':main()
