"""Audit SAE feasibility without selecting configurations or editing any learned model."""
import csv
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis import fc_sae_acceptance_20261105 as run
from analysis import fc_preview_pilot_20261105 as exp
from predify_experiments.fc_sae_proxy_20261105 import fingerprint,training_statistics

def csvsave(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def report():
    old,p=run.protocol();exp.setup();reconstruction=[];readouts=[];individual=[];diagnostics=[]
    audit=dict(passed=True,protocol_sha256=exp.sha(run.OUT/'protocol.json'),report_source_sha256=exp.sha(__file__),
        only_training_folds_fitted=True,no_Hebbian=True,no_search=True,original_model_unchanged=True,datasets={})
    for d in ('imagenet','things'):
        run.verify(run.OUT/d);a,metadata,provenance=run.data(d,old)
        assert provenance==exp.read(run.OUT/d/'data_provenance.json')
        bridge=exp.read(run.OUT/d/'bridge_acceptance.json');assert bridge['passed'];audit['datasets'][d]=dict(bridge=bridge['results'],layers={})
        with (exp.OUT/d/'reference_svm.pckl').open('rb') as f:fixed=pickle.load(f)
        for l in run.LAYERS:
            dest=run.OUT/d/l;run.verify(dest);note=exp.read(dest/'training.json');ev=exp.read(dest/'evaluation_audit.json')
            assert ev['passed'] and ev['state_before']==ev['state_after'] and ev['zero_edit_max_abs_error']==0
            center,scale=training_statistics(a[l]['fit'],a[l]['fit_weights'])
            assert note['center_sha256']==exp.array_sha(center) and np.isclose(scale,note['scale'])
            assert note['fit_array_sha256']==exp.array_sha(a[l]['fit']) and note['fit_weights_sha256']==exp.array_sha(a[l]['fit_weights'])
            model=run.fit_proxy(d,l,a[l]);assert fingerprint(model)==ev['state_before']
            reconstruction.extend(exp.read(dest/'reconstruction_metrics.json'));rr=exp.read(dest/'readout_metrics.json');readouts.extend(rr)
            with np.load(dest/'representations.npz') as z:reps={k:z[k] for k in z.files}
            with (dest/'readouts.pckl').open('rb') as f:heads=pickle.load(f)
            w=exp.weights(metadata['train']['y'],metadata['train']['concept']);scaler_error=0.
            for variant,h in heads.items():
                assert h[-1].C==1 and h[-1].n_iter_<30000
                mu=np.average(reps['train__'+variant].astype(np.float64),weights=w,axis=0)
                error=float(np.abs(mu-h[0].mean_).max());assert error<1e-9;scaler_error=max(scaler_error,error)
            score_error=0.;metric_error=0.
            with np.load(dest/'scores.npz') as z:
                for key in z.files:
                    split,variant,kind=key.split('__');x=reps[split+'__'+variant];y=metadata[split]['y'];head=fixed[l] if kind=='fixed' else heads[variant]
                    s=head.decision_function(x);err=float(np.abs(s-z[key]).max());assert err<1e-10;score_error=max(score_error,err)
                    if variant=='zero_edit':assert np.array_equal(reps[split+'__raw'],x) and np.array_equal(z[key],z[split+'__raw__fixed'])
                    saved=next(r for r in rr if(r['split'],r['variant'],r['readout'])==(split,variant,kind));now=run.previous.metrics(s,y)
                    err=max(abs(now[k]-saved[k]) for k in ('ba','auc'));assert err<1e-12;metric_error=max(metric_error,err)
                    for j,target in enumerate(metadata[split]['row']):
                        individual.append(dict(dataset=d,layer=l,split=split,target=int(target),concept=str(metadata[split]['concept'][j]),
                            category=int(y[j]),variant=variant,readout=kind,prediction=int(s[j].argmax()),correct=int(s[j].argmax()==y[j])))
            dead=reps['train__raw'].astype(np.float64).var(0)<=1e-16;h=heads['rectified'];standardized=h[0].transform(reps['formal0__rectified'])
            diagnostics.append(dict(dataset=d,layer=l,raw_train_constant_dimensions=int(dead.sum()),
                newly_variable_reconstruction_dimensions=int(np.sum(dead&(h[0].var_>1e-16))),
                adapted_coefficient_squared_share_on_these_dimensions=float(np.square(h[-1].coef_[:,dead]).sum()/np.square(h[-1].coef_).sum()),
                formal0_standardized_rms_on_these_dimensions=float(np.sqrt(np.mean(np.square(standardized[:,dead]))))))
            audit['datasets'][d]['layers'][l]=dict(state_unchanged=True,zero_edit_error=0.,zero_edit_fixed_scores_unchanged=True,
                zero_edit_rows=sum(len(reps[s+'__raw']) for s in ('train','train_preview','formal0','development','development_preview')),
                training_only_statistics_replayed=True,score_replay_error=score_error,metric_replay_error=metric_error,
                adapted_scaler_mean_error=scaler_error,final_step=note['history'][-1]['step'],
                no_heldout_evaluation_during_training=note['no_heldout_evaluation_during_optimization'])
    run.DOC.mkdir(parents=True,exist_ok=True)
    for name,rows in (('reconstruction_metrics',reconstruction),('readout_metrics',readouts),('calibration_diagnostics',diagnostics)):
        csvsave(run.DOC/(name+'.csv'),rows);exp.write_json(run.DOC/(name+'.json'),rows)
    csvsave(run.DOC/'per_target_readout.csv',individual)
    def rec(d,l,s,v='rectified'):return next(r for r in reconstruction if(r['dataset'],r['layer'],r['split'],r['variant'])==(d,l,s,v))
    def svm(d,l,s,v,k):return next(r for r in readouts if(r['dataset'],r['layer'],r['split'],r['variant'],r['readout'])==(d,l,s,v,k))
    lines=['# SAE代理可行性验收：固定512维ReLU＋L1','',
        '**工程零编辑与独立串联桥接验收通过；当前单组字典的跨概念重建与稀疏性泛化不足，SVM保真并不稳定。尚不能把它当作已通过语义保真验收的Hebbian代理。**','',
        '本轮没有Hebbian、任何新侧向连接、SAE接入原模型、原模型参数更新或参数搜索。辅助编码器/解码器离线拟合后冻结。以下结果均来自唯一预先冻结配置，未按测试效果选择容量、稀疏代价、训练轮数或检查点。','',
        '## 拟合与隔离','',
        '每个数据集、每层单独一个4096→512 ReLU→4096仿射解码器。decoder带符号，列L2范数约束为1；使用训练混合池的固定中心与一个全局RMS，无逐图排序或固定活动数。目标函数为各样本加权的0.5×归一化重建平方误差之和＋0.1×latent L1之和。全批Adam固定1000步，学习率余弦从0.001降至0.0001，只使用最后一步。SAE不使用类别标签或SVM保真正则。','',
        '|数据|拟合target（fold1–4）|去重训练preview|主验收target（fold0）|开发target复核|',
        '|---|---:|---:|---:|---:|',
        '|ImageNet|256|648|64|16|','|THINGS|320|828|80|18|','',
        '训练target、preview各占总损失权重的一半；preview按真实画面SHA去重。fold0及开发队列的概念和视觉角色不参与拟合、归一化统计或检查点选择。preview包含多个物体，仅做重建与编码统计，不赋予target类别来训练读出。两层输入均为vanilla缓存，FC2未使用任何Hebbian处理后的表示。','',
        '## 零编辑验收','',
        '实际计算z=E(x)，分别解码z及其相同副本，再先求Δ=Dec(z副本)−Dec(z)，最后输出ReLU(x＋Δ)。没有直接跳过计算、返回x。各层各数据split的Δ严格为0，输出逐元素等于原x；原SVM分数与预测完全相同。SAE状态哈希在评估前后不变。','',
        '独立FC1→原始线性桥接＋ReLU→FC2的双层零编辑也逐元素恒等。该严格比较以同一原始桥接计算为参照；与旧缓存FC2的微小批次浮点差异单独记录在audit.json。此验收证明旁路实现正确，不证明字典重建或未来非零编辑保真。','',
        '## 重建与稀疏性','',
        'FVU=Σw‖x−重建‖²/Σw‖x−训练角色中心‖²。中心只来自训练target或训练preview；不是测试集中心。FVU越低越好；它不同于相对L2误差。下面列ReLU后的完整重建，原始带符号重建另见CSV。','',
        '|数据/层|训练target FVU%|fold0 FVU%|开发target FVU%|fold0相对L2误差%|活动latent：训练→fold0|',
        '|---|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for l in run.LAYERS:
            a,b,c=[rec(d,l,s) for s in ('train','formal0','development')]
            lines.append(f"|{d} {l.upper()}|{100*a['fvu']:.3f}|{100*b['fvu']:.2f}|{100*c['fvu']:.2f}|{100*b['relative_l2']:.2f}|{a['latent_active_mean']:.1f}→{b['latent_active_mean']:.1f}|")
    lines+=['',
        '训练target几乎被重建，但隔离概念上误差大幅上升，且激活更多latent。结果符合训练样本记忆化/跨概念失配的表现；本轮不能据此唯一断定是容量、稀疏代价、训练轮数还是数据分布导致。不能因训练重建误差小或没有死latent，就认定找到了稳定语义坐标。','',
        '## 主验收：fold0的SVM功能保真','',
        '固定头为此前vanilla训练的加权StandardScaler＋LinearSVC(C=1)。适配头仅在训练target的对应表示上拟合，仍为相同C=1 SVM；latent和残差也单独训练诊断头。类别平衡BA与对原固定头预测一致率是不同指标；BA相同或更高不代表预测未改变。','',
        '|数据/层|vanilla BA%|重建固定头BA%|重建适配头BA%|latent SVM BA%|残差SVM BA%|重建与原预测一致%|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for l in run.LAYERS:
            cases=[svm(d,l,'formal0',v,k) for v,k in (('raw','fixed'),('rectified','fixed'),('rectified','adapted'),('latent','adapted'),('residual','adapted'))]
            lines.append(f'|{d} {l.upper()}|'+'|'.join(f"{100*r['ba']:.2f}" for r in cases)+f"|{100*cases[1]['agreement_with_fixed_raw']:.2f}|")
    lines+=['','## 原开发小批量复核：不与fold0混用基线','',
        '|数据/层|vanilla BA%|重建固定BA%|重建适配BA%|latent BA%|残差BA%|',
        '|---|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for l in run.LAYERS:
            cases=[svm(d,l,'development',v,k) for v,k in (('raw','fixed'),('rectified','fixed'),('rectified','adapted'),('latent','adapted'),('residual','adapted'))]
            lines.append(f'|{d} {l.upper()}|'+'|'.join(f"{100*r['ba']:.2f}" for r in cases)+'|')
    lines+=['',
        'latent在正式fold0上保留了相当一部分类别可读性，但ImageNet开发集FC1只有56.25%，说明这种保留并不跨队列稳定。完整重建有时提高BA、有时降低，也改变了不少原预测；不能据最高的单项分数宣布通过。','',
        '残差为x−带符号重建，保持符号且不经ReLU。它仍含可读类别信息，说明代理未解释所有可读成分；但StandardScaler会放大低幅度残差，因此残差SVM高于机会率不证明“主要类别信息都在残差”，也不证明latent没有学习到类别成分。','',
        '部分重建适配头反而比固定头差。训练重建几乎准确，不能保证其误差分布能跨概念泛化。进一步诊断发现，原训练target中完全不变的34–52个FC维度，在重建后均出现小方差；适配头约3.8–6.0%的系数平方量位于这些维度，fold0对应标准化RMS约6.2–17.1。这提示校准可能放大小误差/分布偏移，但不是已证实的唯一原因。本轮没有通过筛维度或重新调C修复它。','',
        '## 本轮结论及范围','',
        'FC2的跨概念重建误差小于FC1，但两层均未形成足够稳定的重建与稀疏性泛化。零编辑旁路可保持原输出，适合作为后续独立测试的工程基础；当前字典尚不足以支持“其latent干预会保留真正类别成分”的结论。本轮不接入Hebbian，也不将完整重建替换原FC。','',
        '如继续，应优先讨论训练覆盖与约束，而不是根据这些验收样本立即选更大的字典或更低的误差参数。任何后续改动需保留本配置作为基线，再使用预先固定的新验收数据或隔离的训练内验证；本轮没有执行这些调整。','',
        '完整的raw/rectified重建误差、各类recall、AUC、预测一致率、固定头分数RMSE、逐target预测和校准诊断保存在相邻CSV/JSON。模型、训练曲线、输入SHA与归一化统计来源保存在Log/fc_sae_acceptance_20261105。','']
    (run.DOC/'SAE代理可行性验收.md').write_text('\n'.join(lines),encoding='utf-8')
    plots(reconstruction,readouts);exp.validate(old);run.protocol();exp.write_json(run.DOC/'audit.json',audit)
    exp.write_json(run.DOC/'complete.json',dict(protocol_sha256=exp.sha(run.OUT/'protocol.json'),
        files={str(p.relative_to(run.DOC)):exp.sha(p) for p in run.DOC.rglob('*') if p.is_file() and p.name!='complete.json'}))
    print('SAE ACCEPTANCE AUDIT',audit,flush=True)

def plots(reconstruction,readouts):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    dest=run.DOC/'figures';dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,l in zip(axes,run.LAYERS):
        for d,color in (('imagenet','#277da1'),('things','#f8961e')):
            values=[100*next(r['fvu'] for r in reconstruction if(r['dataset'],r['layer'],r['split'],r['variant'])==(d,l,s,'rectified')) for s in ('train','formal0','development')]
            ax.plot(range(3),values,'o-',c=color,label=d)
        ax.set_xticks(range(3),['Train targets','Held-out fold 0','Development']);ax.set_yscale('log');ax.set_ylabel('Unexplained variance (%) / log scale')
        ax.set_title(l.upper());ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    for ext in ('png','svg','pdf'):fig.savefig(dest/f'reconstruction_generalization.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    cases=[('raw','fixed'),('zero_edit','fixed'),('rectified','fixed'),('rectified','adapted'),('latent','adapted'),('residual','adapted')]
    names=['Vanilla','Zero edit','Recon fixed','Recon refit','Latent','Residual']
    for row,d in enumerate(('imagenet','things')):
        for col,l in enumerate(run.LAYERS):
            ax=axes[row,col];values=[100*next(r['ba'] for r in readouts if(r['dataset'],r['layer'],r['split'],r['variant'],r['readout'])==(d,l,'formal0',v,k)) for v,k in cases]
            ax.bar(range(len(cases)),values,color=['#277da1','#43aa8b','#f9c74f','#f8961e','#9b5de5','#999999'])
            ax.set_xticks(range(len(cases)),names,rotation=25,ha='right');ax.set_ylim(0,100);ax.set_ylabel('Balanced accuracy (%)');ax.set_title(f'{d} / {l.upper()} / held-out fold 0')
    for ext in ('png','svg','pdf'):fig.savefig(dest/f'readout_fidelity.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    with threadpool_limits(limits=2):report()
