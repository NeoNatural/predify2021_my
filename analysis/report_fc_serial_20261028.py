"""Audit and summarize actual serial FC processing and its propagation controls."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from predify_experiments.fc_serial_20261028 import DATA,OUT,DOC,DEVICE,conditions,bridge,sha
from predify_experiments import fc_finite_steps_20261028 as finite
from predify_experiments.fc_serial_svm_completion_20261028 import OUT as SVM_OUT

DATASETS=('imagenet','things')
EFFECTS=('activation_pct','within_pct','between_pct','fisher_pct','accuracy_pp','auc_delta')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write_csv(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def aggregate():
    protocol=read(OUT/'protocol.json')
    assert sha(OUT/'source_snapshot.py')==protocol['source_sha256']
    assert sha(ROOT/'predify_experiments/fc_serial_20261028.py')==protocol['source_sha256']
    for p,h in {**protocol['dependencies'],**protocol['original_model_sources']}.items():assert sha(ROOT/p)==h
    assert sha(protocol['checkpoint'])==protocol['checkpoint_sha256']
    for d in DATASETS:
        for p,h in protocol['inputs'][d].items():assert sha(DATA/d/p)==h
    weights=torch.load(protocol['checkpoint'],map_location='cpu',weights_only=True)
    w,b=(weights[f'classifier.3.{k}'].to(DEVICE) for k in ('weight','bias'))
    del weights
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(4)
    rows=[];class_rows=[]
    audit=dict(passed=True,folds=0,final_representations=0,intermediate_representations=0,
        bridge_replay_max_error=0.,fc2_only_replay_max_error=0.,original_model_unchanged=True,
        pretrained_checkpoint_unchanged=True,vanilla_cache_unchanged=True,old_code_unchanged=True,
        concept_overlap=0,test_data_used_for_matched_statistics=False,primary_readout='StandardScaler + LinearSVC C1 both layers',
        readout_schema='refit_svm_* are primary; legacy unprefixed accuracy/auc fields in summary.csv/json remain auxiliary ridge for compatibility')
    for d in DATASETS:
        with np.load(DATA/d/'activations.npz') as z:
            original1,original2=z['fc1'],z['fc2'];y,concept,fold=z['category'],z['concept'],z['fold']
        check=read(OUT/d/'bridge_check.json')
        assert check['bridge_max_error']<1e-4 and check['head_max_error']<1e-4
        for f in range(5):
            dest=OUT/d/f'fold{f}'
            complete=read(dest/'complete.json')
            assert complete['protocol_sha256']==sha(OUT/'protocol.json')
            for p,h in complete['files'].items():assert sha(dest/p)==h
            data=read(dest/'results.json');tr,te=np.array(data['train_rows']),np.array(data['test_rows'])
            svm_dest=SVM_OUT/d/f'fold{f}'
            svm_complete=read(svm_dest/'complete.json')
            assert sha(svm_dest/'results.json')==svm_complete['files']['results.json']
            svm_data=read(svm_dest/'results.json')
            assert svm_data['train_rows']==data['train_rows'] and svm_data['test_rows']==data['test_rows']
            svm_base=svm_data['results']['baseline']['fc2_output']
            assert np.array_equal(te,np.flatnonzero(fold==f)) and not set(concept[tr])&set(concept[te])
            assert len(data['results'])==57
            for l in ('fc1','fc2'):
                p=f'{d}/{l}/{f}'
                assert sha(finite.sparse.OUT/d/l/f'fold{f}'/'selection.json')==protocol['old_selections'][p]
            base=data['results']['baseline']['metrics']
            with np.load(dest/'final_fc2.npz') as final,np.load(dest/'intermediate_stages.npz') as stages,np.load(dest/'readout_scores.npz') as scores:
                assert np.array_equal(final['row'],te) and np.array_equal(final['y'],y[te])
                assert np.array_equal(final['baseline'],original2[te])
                assert np.array_equal(stages['fc1_off_0'],original1[te])
                for key in stages.files:
                    if not key.startswith('fc1_'):continue
                    tag=key[4:];x=stages[key];want=stages['input2_'+tag]
                    with torch.inference_mode():got=bridge(torch.as_tensor(x,device=DEVICE),w,b).cpu().numpy()
                    error=float(np.abs(got-want).max())
                    audit['bridge_replay_max_error']=max(audit['bridge_replay_max_error'],error)
                    assert error<1e-4
                    if tag!='off_0':assert not np.allclose(want,original2[te],atol=1e-5)
                    audit['intermediate_representations']+=2
                for name,entry in data['results'].items():
                    c,m=entry['condition'],entry['metrics'];r=final[name]
                    svm=svm_data['results'][name]['fc2_output']
                    if 'refit_svm' in m:
                        assert abs(svm['accuracy']-m['refit_svm']['accuracy'])<1e-12
                        assert abs(svm['auc']-m['refit_svm']['auc'])<1e-12
                    assert r.dtype==np.float32 and r.shape==(len(te),4096) and np.isfinite(r).all() and (r>=0).all()
                    assert scores[name].shape==(len(te),5) and np.isfinite(scores[name]).all()
                    input2=stages[f"input2_{c['m1']}_{c['t1']}"]
                    if c['m2']=='off':assert np.array_equal(r,input2)
                    if c['m2'] in ('soft','hard','inhib_soft','inhib_hard'):
                        assert np.all((r>finite.EPS).sum(1)<=(input2>finite.EPS).sum(1))
                    audit['final_representations']+=1
                    row=dict(dataset=d,fold=f,**c,**m['effects'],activation=m['activation'],within=m['within'],between=m['between'],fisher=m['fisher'],
                        combined_fc_mean_pct=m['combined_fc_mean_pct'],accuracy=m['accuracy'],auc=m['auc'],nonzero=m['nonzero'],
                        support_ratio=m['support_ratio'],max_support_ratio=m['max_support_ratio'],effective_dimension=m['effective_dimension'],
                        zero_fraction=m['zero_fraction'],collapsed=m['collapsed'],joint=m['joint'],sparse_joint=m['sparse_joint'],
                        frozen_svm_accuracy_pp=100*(m['frozen_svm']['accuracy']-base['frozen_svm']['accuracy']),
                        frozen_svm_auc_delta=m['frozen_svm']['auc']-base['frozen_svm']['auc'],
                        refit_svm_accuracy=svm['accuracy'],refit_svm_auc=svm['auc'],
                        refit_svm_accuracy_pp=100*(svm['accuracy']-svm_base['accuracy']),
                        refit_svm_auc_delta=svm['auc']-svm_base['auc'],
                        angular_ridge_accuracy_pp=100*(m['angular_ridge']['accuracy']-base['angular_ridge']['accuracy']),
                        angular_ridge_auc_delta=m['angular_ridge']['auc']-base['angular_ridge']['auc'],
                        original_imagenet_top1_pct=m['original_imagenet_top1_pct'],
                        original_imagenet_top1_pp=m['original_imagenet_top1_pct']-base['original_imagenet_top1_pct'] if d=='imagenet' else None,
                        original_imagenet_top5_pct=m['original_imagenet_top5_pct'])
                    for stage_key,prefix in (('fc1','fc1'),('propagated_fc2','propagation'),('local_fc2','local')):
                        for field in EFFECTS:row[prefix+'_'+field]=m[stage_key]['effects'][field]
                    row['local_support_ratio']=m['local_fc2']['support_ratio']
                    row['local_max_support_ratio']=m['local_fc2']['max_support_ratio']
                    rows.append(row)
                    for j,cat in enumerate(('animals','clothing','food','plants','vehicles')):
                        class_rows.append(dict(dataset=d,fold=f,name=name,category=cat,
                            activation_pct=100*(m['per_class_activation'][j]/base['per_class_activation'][j]-1),
                            within_pct=100*(m['per_class_within'][j]/base['per_class_within'][j]-1),
                            ridge_recall_pp=100*(m['recalls'][j]-base['recalls'][j]),
                            svm_recall_pp=100*(svm['recalls'][j]-svm_base['recalls'][j])))
                with np.load(finite.OUT/d/'fc2'/f'fold{f}'/'pair_representations.npz') as old:
                    for mode,role in (('soft','prior_soft'),('hard','matched_hard')):
                        for t in (1,5,20):
                            error=float(np.abs(final[f'fc2_{mode}_{t}']-old[f'{role}_t{t}']).max())
                            audit['fc2_only_replay_max_error']=max(audit['fc2_only_replay_max_error'],error)
                            assert error<1e-4
                for cal in ('frozen','matched'):
                    target=final[f'both_hard_hard_5_5_{cal}'].mean(1)
                    for ctrl in (f'gain2_hard_5_{cal}',f'gain_both_hard_5_{cal}'):
                        assert np.allclose(final[ctrl].mean(1),target,atol=2e-6)
            audit['folds']+=1
    assert audit['folds']==10 and audit['final_representations']==570
    summary=[]
    for d in DATASETS:
        for c in conditions():
            rs=[r for r in rows if r['dataset']==d and r['name']==c['name']]
            assert len(rs)==5
            s=dict(dataset=d,**c)
            for key in rs[0]:
                if key in s or key in ('fold','joint','sparse_joint','collapsed'):continue
                s[key]=float(np.mean([r[key] for r in rs])) if all(r[key] is not None for r in rs) else None
            s['joint_folds']=sum(r['joint'] for r in rs);s['sparse_joint_folds']=sum(r['sparse_joint'] for r in rs)
            s['collapsed_folds']=sum(r['collapsed'] for r in rs)
            s['mean_joint']=bool(s['activation_pct']<-.001 and s['within_pct']<-.001 and s['fisher_pct']>.001)
            summary.append(s)
    write_csv(DOC/'fold_metrics.csv',rows);write_csv(DOC/'summary.csv',summary);write_csv(DOC/'category_metrics.csv',class_rows)
    (DOC/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (DOC/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary,audit


def plots(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=DOC/'figures';figs.mkdir(exist_ok=True)
    order=['fc1_hard_5','fc2_hard_5']
    order += [f'both_{a}_{b}_5_5_{cal}' for cal in ('frozen','matched') for a in ('soft','hard') for b in ('soft','hard')]
    order += ['inhibition_only_hard_5','gain1_hard_5_matched','gain2_hard_5_matched','gain_both_hard_5_matched']
    for d in DATASETS:
        lookup={r['name']:r for r in summary if r['dataset']==d}
        fields=('activation_pct','within_pct','between_pct','fisher_pct','refit_svm_auc_delta')
        vals=np.array([[lookup[n][k] for k in fields] for n in order])
        colors=vals*np.array([-1,-1,1,1,1]);colors/=np.maximum(np.abs(colors).max(0),1e-9)
        fig,ax=plt.subplots(figsize=(13,9))
        ax.imshow(colors,cmap='RdBu',vmin=-1,vmax=1,aspect='auto')
        ax.set_yticks(range(len(order)),order,fontsize=9)
        ax.set_xticks(range(5),['Final FC2 activation %','Normalized within %','Normalized between %','Fisher %','Raw SVM AUC delta'],fontsize=9);ax.xaxis.tick_top()
        for i in range(len(order)):
            for j in range(5):ax.text(j,i,f'{vals[i,j]:+.4f}' if j==4 else f'{vals[i,j]:+.2f}',ha='center',va='center',fontsize=9,color='white' if abs(colors[i,j])>.6 else 'black')
        ax.set_title(f'{d.upper()} | true serial FC1 -> original VGG bridge -> FC2\nEffects vs vanilla final FC2; blue = favorable direction; five-fold descriptive means',pad=42)
        fig.tight_layout();fig.savefig(figs/f'{d}_serial_comparison.png',dpi=170);fig.savefig(figs/f'{d}_serial_comparison.svg');plt.close(fig)
        fig,axs=plt.subplots(2,3,figsize=(12,8))
        for i,cal in enumerate(('frozen','matched')):
            for j,field in enumerate(('activation_pct','within_pct','fisher_pct')):
                v=np.array([[lookup[f'both_hard_hard_{t1}_{t2}_{cal}'][field] for t2 in (1,5,20)] for t1 in (1,5,20)])
                benefit=v*(-1 if field!='fisher_pct' else 1)
                lim=max(float(np.abs(benefit).max()),1.)
                ax=axs[i,j];ax.imshow(benefit,cmap='RdBu',vmin=-lim,vmax=lim)
                ax.set_xticks(range(3),(1,5,20));ax.set_yticks(range(3),(1,5,20));ax.set_xlabel('FC2 steps');ax.set_ylabel('FC1 steps')
                ax.set_title(f'{cal}: {field}')
                for a in range(3):
                    for b in range(3):ax.text(b,a,f'{v[a,b]:+.2f}',ha='center',va='center',color='white' if abs(benefit[a,b])>lim*.6 else 'black')
        fig.suptitle(f'{d.upper()} | hard-hard serial horizon grid; no test-set horizon selection')
        fig.tight_layout();fig.savefig(figs/f'{d}_horizon_grid.png',dpi=170);plt.close(fig)


def report(summary,audit):
    def get(d,n):return next(r for r in summary if r['dataset']==d and r['name']==n)
    lines=['# FC1 与 FC2 真正串联的实验结果','','2026-10-28。FC1侧向输出经原始VGG线性权重＋bias＋ReLU进入FC2，再作FC2侧向更新。顺序串联，不引入反向反馈。原模型、预训练权重及缓存保持不变。','',
        '两套数据五折概念隔离、57个预定条件。参数沿用前轮，步数固定为每层1/5/20；hard-hard另完整比较非对称步数。没有按测试结果重新选参。','',
        'FC1与FC2已分别汇总，见[分层统计](分层统计.md)。以下主要比较表中的几何和readout指标属于最终FC2，不能代表FC1本层；分层报告同时列出FC1、FC2前馈输入、FC2输出及局部效果。两层现统一以相同设置的SVM作为主要readout，所有条件均已补齐；ridge仅作辅助，几何与激活指标保持原口径。','',
        '## 结果解释','','真正串联在两套数据上进一步降低平均激活，并提高归一化表征的类间/类内区分比。每层5步hard-hard、按串联训练输入重估FC2统计量时，最终FC2均值下降约49%，Fisher分别提高12.27%与17.77%；重新拟合的大类SVM平衡准确率仅变化−0.04、−0.10个百分点。','',
        '但这尚不能证明两层Hebbian具有不可替代的协同：只将FC1替换为逐图均值匹配的缩放，保留FC2侧向处理，Fisher仍提高11.68%与16.85%。当前区分度收益主要来自FC2处理，FC1的均值降低与经原始bias/ReLU传递的作用贡献较大。两层均替换为幅度匹配时，Fisher仅提高0.54%与0.61%；仅抑制约提高0.57%与0.64%。','',
        '重估FC2训练统计量与尺度带来更低均值，但Fisher并非两套数据都更高：ImageNet沿用原统计量时为+12.83%，重估后为+12.27%；THINGS从+17.44%变为+17.77%。两种口径应同时保留。','',
        '## 主要比较：5步','','所有指标均针对最终FC2相对原始vanilla FC2。仅FC1一行已经包括经原始前馈权重传到FC2的效果，不是FC1本层的独立指标。W为逐图L2归一化后的类内离散度，B为类中心间离散度，Fisher=B/W。平均激活保留实际幅度，readout亦保留原始幅度；这里的归一化几何用于另外检验表征方向变化。BA为平衡准确率，pp表示百分点，其他变化均为相对百分比。','']
    names=['fc1_hard_5','fc2_hard_5','both_hard_hard_5_5_frozen','both_hard_hard_5_5_matched']
    for d in DATASETS:
        lines += [f'### {d}','','| 条件 | 最终FC2激活% | W% | B% | Fisher% | 两层总平均激活% | SVM BA pp | SVM AUC Δ | 联合达标折数 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for n in names:
            r=get(d,n)
            lines.append(f"| {n} | {r['activation_pct']:+.2f} | {r['within_pct']:+.2f} | {r['between_pct']:+.2f} | {r['fisher_pct']:+.2f} | {r['combined_fc_mean_pct']:+.2f} | {r['refit_svm_accuracy_pp']:+.2f} | {r['refit_svm_auc_delta']:+.5f} | {r['joint_folds']}/5 |")
        lines += ['',f"![Comparison](<{(DOC/'figures'/f'{d}_serial_comparison.png').as_posix()}>)",'']
    lines += ['## FC2局部作用与串联总效果','','frozen沿用独立FC2统计量与尺度；matched由真实串联后的训练FC2输入重估连接与尺度。两种口径差异不能全部归因于连接矩阵。','',
        '| 数据集/FC2口径 | FC1传递造成的FC2激活% | FC1传递Fisher% | FC2局部激活% | FC2局部W% | FC2局部Fisher% | 最终非零% | 相对实际FC2输入的活动数比 | 相对原始FC2活动数比 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        for cal in ('frozen','matched'):
            r=get(d,f'both_hard_hard_5_5_{cal}')
            lines.append(f"| {d}/{cal} | {r['propagation_activation_pct']:+.2f} | {r['propagation_fisher_pct']:+.2f} | {r['local_activation_pct']:+.2f} | {r['local_within_pct']:+.2f} | {r['local_fisher_pct']:+.2f} | {100*r['nonzero']:.2f} | {r['local_support_ratio']:.4f} | {r['support_ratio']:.4f} |")
    lines += ['', '活动预算按当前实际FC2输入设置，局部输出不增加其活动数；相对vanilla的密度还受到FC1前馈传递影响。总平均活动按两层各4096单元等权计算；不以总量替代逐层与最终表征检查。','',
        '## soft/hard组合与幅度对照','','主要readout统一使用保留原始幅度的StandardScaler + LinearSVC(C=1)，训练划分、类别/概念平衡权重和参数与此前FC2一致；标准化和分类器只拟合训练折。ridge及去范数ridge保留作辅助诊断，不作为成功硬门槛。下面均为matched口径、每层5步。','',
        '| 数据集/条件 | 最终激活% | W% | Fisher% | SVM BA pp | SVM AUC Δ |', '|---|---:|---:|---:|---:|---:|']
    controls=['both_soft_soft_5_5_matched','both_soft_hard_5_5_matched','both_hard_soft_5_5_matched','both_hard_hard_5_5_matched',
        'inhibition_only_hard_5','gain1_hard_5_matched','gain2_hard_5_matched','gain_both_hard_5_matched']
    for d in DATASETS:
        for n in controls:
            r=get(d,n);lines.append(f"| {d}/{n} | {r['activation_pct']:+.2f} | {r['within_pct']:+.2f} | {r['fisher_pct']:+.2f} | {r['refit_svm_accuracy_pp']:+.2f} | {r['refit_svm_auc_delta']:+.5f} |")
    lines += ['', 'gain1逐图匹配真实FC1处理的均值，再经过原始桥接；gain2匹配真正hard-hard的最终均值但保持传入FC2的方向；gain_both两层均用幅度匹配。由于原始桥接有bias和ReLU，FC1纯幅度改变也可能改变FC2方向，不能把两层变化简单相加。','',
        '## 步数与兼容性','','matched的hard-hard在每层1、5、20步时，ImageNet均值变化分别为−33.43%、−49.43%、−49.64%，Fisher分别为+5.39%、+12.27%、+12.86%；THINGS均值为−33.47%、−49.39%、−49.59%，Fisher为+7.39%、+17.77%、+18.93%。5步后仍有收益但增量较小；此处不据测试结果重新选定最佳步数，也不要求达到不动点。','']
    for d in DATASETS:lines += [f"![Horizons](<{(DOC/'figures'/f'{d}_horizon_grid.png').as_posix()}>)",'']
    lines += ['固定的原始SVM检验既有下游兼容性，重新拟合的大类readout检验可读性，两者分别保留。ImageNet另检验完全固定的原始1000类head，THINGS没有该真值不计算。','',
        '| ImageNet条件 | 固定大类SVM BA变化pp | 重拟合大类SVM BA变化pp | 原始1000类Top1% | 原始Top1变化pp |','|---|---:|---:|---:|---:|']
    for n in ['baseline']+names:
        r=get('imagenet',n);lines.append(f"| {n} | {r['frozen_svm_accuracy_pp']:+.2f} | {r['refit_svm_accuracy_pp']:+.2f} | {r['original_imagenet_top1_pct']:.2f} | {r['original_imagenet_top1_pp']:+.2f} |")
    lines += ['', '不能由重拟合大类readout大体保持，就宣称原始1000类分类器也保持或改善；也不能将原始head变化等同于大类语义完全丢失。','',
        '## 审计与范围','',f"审计通过：{audit['folds']}个外层折、{audit['final_representations']}个完整最终FC2表征；桥接回放最大误差{audit['bridge_replay_max_error']:.3g}，FC2-only重现独立处理最大误差{audit['fc2_only_replay_max_error']:.3g}。6项串联/训练隔离/对照测试通过。",'',
        '同一探索样本和旧参数多轮复用，没有新增独立确认；ImageNet植物7细类、THINGS原生标注分歧继续适用。五折描述性报告，不当作独立受试者，不宣称显著性。','',
        'summary.csv/json保存所有114个数据集×条件摘要；fold_metrics.csv保存570行逐折结果，category_metrics.csv保留分大类数据。refit_svm_*列为主要readout；为兼容旧分析，未加前缀的accuracy/auc及其变化字段仍是辅助ridge，请勿混用。分层CSV中的ridge_*和refit_svm_*始终明确区分。原始输出目录保存完整FC2输出、FC1/桥接中间状态、分数、SVM、源码和哈希；新增SVM补评估另存于svm_layer_completion_20261028子目录。','',
        '```powershell','python -m predify_experiments.fc_serial_20261028 run','python analysis/report_fc_serial_20261028.py','```','']
    (DOC/'实验结果.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    summary,audit=aggregate();plots(summary);report(summary,audit)
    from analysis.report_fc_serial_layers_20261028 import aggregate as layer_aggregate, report as layer_report
    layer_report(layer_aggregate())
    for r in summary:
        if r['name'] in ('fc1_hard_5','fc2_hard_5','both_hard_hard_5_5_frozen','both_hard_hard_5_5_matched','gain_both_hard_5_matched'):
            print(r['dataset'],r['name'],{k:r[k] for k in EFFECTS},'SVM',r['refit_svm_accuracy_pp'],r['refit_svm_auc_delta'],
                'density',r['nonzero'],'fineTop1delta',r['original_imagenet_top1_pp'],'joint',r['joint_folds'])
    print('AUDIT',audit)


if __name__=='__main__':main()
