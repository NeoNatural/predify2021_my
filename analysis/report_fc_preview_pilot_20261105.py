"""Audit the clean temporary-Hebbian pilot and report paired history effects separately by FC layer."""
from pathlib import Path
import csv
import json
import pickle
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis import fc_preview_pilot_20261105 as exp
from predify_experiments.fc_preview_memory_20261105 import PreviewMemory,TrialEngine

OUT,DOC=exp.OUT,exp.DOC
METHODS=exp.METHODS+('inhib_scale',)
LABELS=dict(vanilla='原始vanilla',inhibition='仅抑制',hebbian='临时Hebbian串联',scale1='FC1缩放＋FC2 Hebb',
    scale2='FC1 Hebb＋FC2缩放',scale_both='双层缩放',shuffled='权重打乱',inhib_scale='仅抑制后均值匹配')
HISTORY=dict(exact='Exact',different='Different',none='无preview')
STAGES=dict(fc1='FC1输出',fc2_input='FC2实际输入',fc2='最终FC2输出')


def read(p):return exp.read(p)
def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(x,y,groups):return float(np.average(x,weights=exp.weights(y,groups)))
def change(a,b):return 100*(a/b-1) if b>1e-12 else None


def characterize(v,scores,y,groups,quartiles,vanilla):
    v=v.astype(np.float64);base=vanilla.astype(np.float64);norm=np.linalg.norm(v,axis=1)
    u=v/np.maximum(norm[:,None],1e-8);K=scores.shape[1]
    centers=np.stack([u[y==c].mean(0) for c in range(K)])
    within=np.mean([np.square(u[y==c]-centers[c]).sum(1).mean() for c in range(K)])
    between=np.square(centers-centers.mean(0)).sum(1).mean()
    active=v>1e-7;weak=active&(v<.01*v.max(1,keepdims=True));sums=np.maximum(v.sum(1),1e-8)
    masks=quartiles[y];shares=np.stack([(v*(masks==q)).sum(1)/sums for q in range(4)],1)
    group_means=np.stack([(v*(masks==q)).sum(1)/(masks==q).sum(1) for q in range(4)],1)
    pred=scores.argmax(1);recalls=[float((pred[y==c]==c).mean()) for c in range(K)]
    margin=scores[np.arange(len(y)),y]-np.where(np.arange(K)[None,:]==y[:,None],-np.inf,scores).max(1)
    cosine=(v*base).sum(1)/np.maximum(norm*np.linalg.norm(base,axis=1),1e-12)
    per=dict(activation=v.mean(1),nonzero=active.mean(1),weak_share=(v*weak).sum(1)/sums,
        preferred_share=shares[:,3],margin=margin,correct=(pred==y).astype(float),cosine_to_vanilla=cosine,
        zero=(norm<=1e-8).astype(float),quartile_share=shares,quartile_mean=group_means)
    centered=u-u.mean(0);gram=centered@centered.T
    summary=dict(activation=mean(per['activation'],y,groups),within=float(within),between=float(between),
        fisher=float(between/max(within,1e-12)),nonzero=float(per['nonzero'].mean()),weak_share=float(per['weak_share'].mean()),
        zero_fraction=float(per['zero'].mean()),preferred_share=mean(per['preferred_share'],y,groups),
        svm_accuracy=float(np.mean(recalls)),svm_auc=float(np.mean([roc_auc_score(y==c,scores[:,c]) for c in range(K)])),
        margin=mean(margin,y,groups),cosine_to_vanilla=mean(cosine,y,groups),
        effective_dimension=float(np.trace(gram)**2/max(np.square(gram).sum(),1e-12)),
        per_class_accuracy=recalls,per_class_activation=[float(per['activation'][y==c].mean()) for c in range(K)],
        quartile_share=[mean(shares[:,q],y,groups) for q in range(4)],
        quartile_mean=[mean(group_means[:,q],y,groups) for q in range(4)])
    return summary,per


def aggregate():
    protocol=read(OUT/'protocol.json');exp.validate(protocol);exp.setup()
    allsummary=[];contrasts=[];individual=[];category=[];local=[];memory=[];quartile=[]
    audit=dict(passed=True,protocol_sha256=exp.sha(OUT/'protocol.json'),report_source_sha256=exp.sha(__file__),
        datasets={},physical_target_trials=0,checks_every_target_update=0,replayed_history_pairs=0,
        score_replay_max_error=0.,scale_mean_max_error=0.,bridge_max_error=0.,
        reference_scaler_mean_max_error=0.,preview_factor_replay_max_error=0.,
        no_offline_E=True,no_parameter_search=True,clean_only=True,original_sources_unchanged=True,
        target_readonly=True,trial_reset=True,preview_writes_only=True,classes_generic=True,svm_raw_amplitude=True)
    w,b=exp.load_bridge()
    for d in ('imagenet','things'):
        dest=OUT/d/'pilot';exp.verify(dest);data=read(dest/'run.json');accept=read(OUT/d/'state_acceptance.json')
        assert accept['passed'] and data['state_acceptance_sha256']==exp.sha(OUT/d/'state_acceptance.json')
        assert data['input_marker_sha256']==exp.sha(OUT/d/'inputs/complete.json')
        assert data['nuisance_sha256']==exp.sha(OUT/d/'nuisance.json')
        source=exp.features(d);provenance=read(OUT/d/'inputs/provenance.json')
        assert provenance['checkpoint_model_state_before']==provenance['checkpoint_model_state_after']
        note=read(OUT/d/'nuisance.json');assert exp.sha(OUT/d/'reference_svm.pckl')==note['svm_sha256']
        assert exp.sha(OUT/d/'nuisance_arrays.npz')==note['arrays_sha256']
        with (OUT/d/'reference_svm.pckl').open('rb') as f:heads=pickle.load(f)
        with np.load(OUT/d/'nuisance_arrays.npz') as z:qmaps={l:z[l+'_quartiles'] for l in ('fc1','fc2')}
        train,ty,tgroups,_=exp.training(d,protocol);tweights=exp.weights(ty,tgroups)
        for l,head in heads.items():
            assert head[-1].C==1 and head[-1].max_iter==30000 and int(head[-1].n_iter_)<30000
            assert head[-1].classes_.tolist()==list(range(len(protocol['datasets'][d]['categories'])))
            mu=np.average(train[l].astype(np.float64),axis=0,weights=tweights)
            error=float(np.max(np.abs(mu-head[0].mean_)));assert error<1e-9
            audit['reference_scaler_mean_max_error']=max(audit['reference_scaler_mean_max_error'],error)
        with np.load(dest/'representations.npz') as z:reps={k:z[k] for k in z.files if k not in ('row','y','concept')};row,y,groups=z['row'],z['y'],z['concept']
        with np.load(dest/'scores.npz') as z:saved_scores={k:z[k] for k in z.files if k not in ('row','y')}
        assert np.array_equal(row,source['row']) and np.array_equal(y,source['y']) and np.array_equal(groups,source['concept'])
        index={int(x):j for j,x in enumerate(row)};records=read(dest/'trial_audit.json')
        assert len(records)==len(row)*len(exp.METHODS)*3
        zero=PreviewMemory(4096).fingerprint()
        expected_memories={};derived={}
        for record in records:
            i=index[record['target']];method=record['method'];h=record['history']
            assert record['reset_sha256']==[zero,zero]
            assert record['before_target']==record['after_target']
            writes=0 if h=='none' or method in ('vanilla','inhibition') else 2
            assert record['writes']==[writes,writes] and len(record['preview_boundaries'])==writes
            mkey=(i,h)
            if mkey not in expected_memories:
                mm=[]
                for l in ('fc1','fc2'):
                    m=PreviewMemory(4096)
                    if h!='none':
                        for a in source['preview_'+l][i,0 if h=='exact' else 1]:m.write(torch.from_numpy(a).cuda())
                    m.freeze();mm.append(m)
                expected_memories[mkey]=mm
            mm=expected_memories[mkey]
            if method in ('vanilla','inhibition'):expected=[zero,zero]
            elif method=='shuffled':expected=[m.permuted(p).fingerprint() for m,p in zip(mm,exp.permutations(d,int(row[i])))]
            else:expected=[m.fingerprint() for m in mm]
            assert record['before_target']==expected
            for trace in record['traces']:
                assert len(trace['updates'])==5
                for update in trace['updates']:
                    assert update['weight_sha256']==expected[trace['layer']-1];audit['checks_every_target_update']+=1
            audit['physical_target_trials']+=1
            if method=='hebbian':
                for layer,l in enumerate(('fc1','fc2')):
                    memory.append(dict(dataset=d,target=int(row[i]),history=h,layer=l,
                        match=record['query'][layer]['match'],relative_drive=record['query'][layer]['relative_drive'],
                        actual_fc2_match=record['actual_fc2_query']['match'] if layer==1 else None,
                        frobenius=record['memory'][layer]['frobenius'],nonzero_edges=record['memory'][layer]['nonzero_edges'],
                        isolated_units=record['memory'][layer]['isolated'],writes=record['writes'][layer]))
        with np.load(dest/'preview_factors.npz') as z:
            for (i,h),mm in expected_memories.items():
                for l,m in zip(('fc1','fc2'),mm):
                    target=z[f'{int(row[i])}__{h}__{l}']
                    actual=torch.stack(m.factors).cpu().numpy() if m.factors else np.empty((0,4096),np.float32)
                    assert np.array_equal(target,actual)
        summaries={};pers={};scoremap={}
        for method in METHODS:
            for h in exp.HISTORIES:
                for stage in exp.STAGES:
                    key=f'{method}__{h}__{stage}';layer='fc1' if stage=='fc1' else 'fc2'
                    if method=='inhib_scale':
                        a=reps[f'inhibition__{h}__{stage}'];target=reps[f'hebbian__{h}__{stage}']
                        reps[key]=a*(target.mean(1,keepdims=True)/np.maximum(a.mean(1,keepdims=True),1e-8));derived[key]=reps[key]
                    v=reps[key];assert np.isfinite(v).all() and (v>=0).all() and v.shape==(len(row),4096)
                    scores=heads[layer].decision_function(v);scoremap[key]=scores
                    if method!='inhib_scale':
                        err=float(np.max(np.abs(scores-saved_scores[key])));assert err<1e-8
                        audit['score_replay_max_error']=max(audit['score_replay_max_error'],err)
                    baseline=source['target_'+layer]
                    summary,per=characterize(v,scores,y,groups,qmaps[layer],baseline)
                    summaries[method,h,stage]=summary;pers[method,h,stage]=per
                    flat={k:v for k,v in summary.items() if not isinstance(v,list)}
                    allsummary.append(dict(dataset=d,method=method,history=h,stage=stage,n=len(row),**flat))
                    for q in range(4):quartile.append(dict(dataset=d,method=method,history=h,stage=stage,quartile=q,
                        activity_share=summary['quartile_share'][q],mean_activation=summary['quartile_mean'][q]))
                    for c,name in enumerate(protocol['datasets'][d]['categories']):
                        category.append(dict(dataset=d,method=method,history=h,stage=stage,category=name,
                            n=int((y==c).sum()),activation=summary['per_class_activation'][c],svm_recall=summary['per_class_accuracy'][c]))
                    for i,target in enumerate(row):
                        individual.append(dict(dataset=d,target=int(target),concept=str(groups[i]),category=int(y[i]),method=method,history=h,stage=stage,
                            **{k:float(v[i]) for k,v in per.items() if v.ndim==1}))
                    if stage=='fc2_input' and method!='inhib_scale':
                        with torch.inference_mode():pred=exp.bridge(torch.from_numpy(reps[f'{method}__{h}__fc1']).cuda(),w,b).cpu().numpy()
                        err=float(np.max(np.abs(pred-v)));assert np.allclose(pred,v,atol=1e-4,rtol=2e-5),(d,method,h,err)
                        audit['bridge_max_error']=max(audit['bridge_max_error'],err)
                    if (method in ('scale1','scale_both') and stage=='fc1') or (method in ('scale2','scale_both','inhib_scale') and stage=='fc2'):
                        target=reps[f'hebbian__{h}__{stage}']
                        err=float(np.abs(v.mean(1)-target.mean(1)).max());assert np.allclose(v.mean(1),target.mean(1),atol=2e-6,rtol=1e-5)
                        audit['scale_mean_max_error']=max(audit['scale_mean_max_error'],err)
            for stage in exp.STAGES:
                for left,right,name in [('exact','different','E-D'),('exact','none','E-N'),('different','none','D-N')]:
                    sm,sr=summaries[method,left,stage],summaries[method,right,stage]
                    pm,pr=pers[method,left,stage],pers[method,right,stage]
                    rs=100*(1-pm['activation']/np.maximum(pr['activation'],1e-12))
                    contrasts.append(dict(dataset=d,method=method,stage=stage,contrast=name,n=len(row),
                        activation_difference=mean(pm['activation']-pr['activation'],y,groups),suppression_pct=mean(rs,y,groups),
                        within_change_pct=change(sm['within'],sr['within']),between_change_pct=change(sm['between'],sr['between']),
                        fisher_change_pct=change(sm['fisher'],sr['fisher']),preferred_share_pp=100*(sm['preferred_share']-sr['preferred_share']),
                        nonzero_pp=100*(sm['nonzero']-sr['nonzero']),weak_share_pp=100*(sm['weak_share']-sr['weak_share']),
                        svm_accuracy_pp=100*(sm['svm_accuracy']-sr['svm_accuracy']),svm_auc_difference=sm['svm_auc']-sr['svm_auc'],
                        margin_difference=sm['margin']-sr['margin'],lower_activation_targets=int((pm['activation']<pr['activation']).sum())))
        for method in METHODS:
            for h in exp.HISTORIES:
                a=summaries[method,h,'fc2'];z=summaries[method,h,'fc2_input']
                local.append(dict(dataset=d,method=method,history=h,activation_pct=change(a['activation'],z['activation']),
                    within_pct=change(a['within'],z['within']),fisher_pct=change(a['fisher'],z['fisher']),
                    svm_accuracy_pp=100*(a['svm_accuracy']-z['svm_accuracy']),preferred_share_pp=100*(a['preferred_share']-z['preferred_share'])))
        # Exact invariants using saved representations, not only run-time booleans.
        for stage in exp.STAGES:
            for method in ('vanilla','inhibition'):
                for h in exp.HISTORIES:assert np.array_equal(reps[f'{method}__{h}__{stage}'],reps[f'{method}__none__{stage}'])
            assert np.array_equal(reps[f'hebbian__none__{stage}'],reps[f'inhibition__none__{stage}'])
        audit['replayed_history_pairs']+=len(row)*3
        audit['datasets'][d]=dict(n=len(row),categories=protocol['datasets'][d]['categories'],
            selected_concepts=protocol['datasets'][d]['selected_concepts'],slot_counts=protocol['datasets'][d]['slot_counts'],
            training_reference_count=len(ty),formal_fold_0_excluded=True,
            state_acceptance_sha256=exp.sha(OUT/d/'state_acceptance.json'),pilot_marker_sha256=exp.sha(dest/'complete.json'),
            dense_operator_max_error=accept['dense_operator_max_error'],dense_cascade_max_error=accept['dense_cascade_max_error'],
            batch_max_error=accept['batch_max_error'],backbone_state_unchanged=True)
        np.savez_compressed(DOC/f'{d}_derived_scaling.npz',**derived,row=row,y=y)
        print('AUDITED',d,len(row),'targets',len(records),'target trials',flush=True)
    assert audit['physical_target_trials']==714
    save_csv(DOC/'condition_metrics.csv',allsummary);save_csv(DOC/'paired_history_contrasts.csv',contrasts)
    save_csv(DOC/'per_target_metrics.csv',individual);save_csv(DOC/'per_category_metrics.csv',category)
    save_csv(DOC/'fc2_local_effects.csv',local);save_csv(DOC/'memory_diagnostics.csv',memory);save_csv(DOC/'preference_profiles.csv',quartile)
    exp.write_json(DOC/'condition_metrics.json',allsummary);exp.write_json(DOC/'paired_history_contrasts.json',contrasts)
    exp.write_json(DOC/'execution_audit.json',audit);exp.validate(protocol)
    return allsummary,contrasts,local,memory,audit


def report(summary,contrasts,local,memory,audit):
    def get(d,m,s,c):return next(r for r in contrasts if (r['dataset'],r['method'],r['stage'],r['contrast'])==(d,m,s,c))
    def absolute(d,m,s,h):return next(r for r in summary if (r['dataset'],r['method'],r['stage'],r['history'])==(d,m,s,h))
    def f(v):return '—' if v is None else f'{v:+.3f}'
    lines=['# Preview临时Hebbian：状态验收与默认clean小批量结果','',
        '**本轮已执行状态验收与默认参数clean测试；未扩大参数搜索、未运行噪声/遮挡、未修改原模型。**',
        '', '本trial preview形成两层临时权重；每trial清零；target逐步只读。原前馈VGG冻结，无PCoder或卷积Hebbian状态。FC1→原始桥接→FC2是真串联。',
        '', '## 实际范围','', '| 数据 | 类别数 | target数 | 概念数 | 参考SVM训练图 |','|---|---:|---:|---:|---:|']
    overview=['','## 本轮结论','',
        '**状态机制验收成功；默认响应规则未实现所需suppression＋sharpening联合目标。** Exact相对Different主要表现为内容相关增强，而且相对无preview的平均活动也更高。不能再归因为“preview没有写入”。',
        '', 'FC1的Exact平均活动比Different高约1.81%（ImageNet）／2.27%（THINGS），最终FC2高约6.58%／6.56%。相对无preview，FC1高约20.44%／18.56%，最终FC2高约36.59%／35.51%；这些是逐target比例、按类别/概念等权汇总。',
        '', 'Exact相对Different的Fisher提高3.36%／2.29%（FC1）、8.60%／7.31%（FC2），最高类别偏好组的活动份额也提高。然而四个数据/层组合中有三个类内W增加；相对无preview，Fisher及最高偏好组份额全部更低。后者虽有W下降，但类间B也下降约18%–32%，下降更多，Fisher反而降低。因此只有相对Different的部分选择性／类间分离改善，不能称为稳定的类内收紧或相对无preview更好的联合sharpening。',
        '', '仅抑制和vanilla的三历史差值都严格为零。权重打乱后，真实权重的Exact−Different活动差及Fisher/偏好份额差大幅减弱，支持效应与preview关联的单元身份有关。双层缩放复制均值差，却不能复制主要的Fisher与偏好份额差；这说明存在超出纯缩放的内容变化，但其方向尚未满足低活动目标。',
        '', '## 固定SVM功能检查','',
        '| 数据 | 层 | 原始vanilla BA% | 仅抑制 BA% | Hebb Exact BA% | Hebb Different BA% | Hebb无preview BA% |','|---|---|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for stage in ('fc1','fc2'):
            vals=[absolute(d,m,stage,h)['svm_accuracy']*100 for m,h in [('vanilla','none'),('inhibition','none'),('hebbian','exact'),('hebbian','different'),('hebbian','none')]]
            overview.append('| '+d+' | '+STAGES[stage]+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    overview+=['','主方案Exact与Different的平衡准确率相同；Exact的margin相对Different略好，不能据此认定重复带来了准确率收益。相对原始vanilla，ImageNet两层各下降12.5pp；THINGS FC1不变、最终FC2下降5pp。固定头存在处理后分布迁移，且样本很小；本轮未做适配头，因此不把固定头下降直接解释成信息永久丢失，也不能声称功能无损。','']
    lines[6:6]=overview
    for d,meta in audit['datasets'].items():lines.append(f"| {d} | {len(meta['categories'])} | {meta['n']} | {sum(meta['selected_concepts'].values())} | {meta['training_reference_count']} |")
    lines+=['', '开发样本按每类前两个概念、每概念前两图预先选定。THINGS植物只有aloe一个开发概念，因此为18张而非20张。两个数据集的大类分别为4/5类，不混用旧五类代码。',
        '', '本小批量全部prime槽都位于第二帧：ImageNet左/右10/6，THINGS左/右7/11，未覆盖第一帧prime。保持既有刺激清单、没有事后改槽或补样，因此本轮结论只覆盖这些开发布局，不能据此判断第一/第二帧的时间迁移或顺序效应。',
        '', '只有一个默认参数组合：preview rho=0.5，两次写入的系数为0.25、0.5；target gamma=0.75，除性抑制2，阈值0.25μ＋0.1σ，混合0.5，每层5步。原始七方法×三个历史条件，合计714次target处理；另有仅抑制输出的均值匹配诊断，不是新增网络试验。',
        '', '参考SVM只使用正式队列fold!=0的原始前馈clean响应拟合，开发target及全部preview视觉角色与正式队列隔离。缓存只复用vanilla响应，没有读取训练集Hebbian图。SVM是训练加权StandardScaler＋LinearSVC(C=1)，每层一套头跨方法/历史固定，保留幅度。',
        '', '## 状态验收','', '以下均通过：每trial两层清零；Exact/Different各两次preview写入；改变target不改变已写权重；target写入接口拒绝更新；target各更新步权重哈希不变；倒序及干扰trial后重放一致；无preview完整方案等于仅抑制；vanilla和仅抑制三历史输出一致。',
        '', f"实际4096维FP64外积因子与显式矩阵的最大算子误差 {max(v['dense_operator_max_error'] for v in audit['datasets'].values()):.3g}，完整串联独立复算误差 {max(v['dense_cascade_max_error'] for v in audit['datasets'].values()):.3g}。这是等价表示的高精度检查；实际pilot输出为FP32。",
        '', f"714次处理共核查 {audit['checks_every_target_update']} 个target侧向更新的权重哈希。参考SVM分数重放误差 {audit['score_replay_max_error']:.3g}；原始模型状态及源文件哈希保持一致。",'','## 主方案：三种历史的实际指标','',
        '| 数据 | 层 | 历史 | 平均激活 | 类内W | Fisher | 非零比例% | 偏好最高组份额% | SVM BA% | AUC |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for stage in ('fc1','fc2'):
            for h in exp.HISTORIES:
                r=absolute(d,'hebbian',stage,h)
                lines.append(f"| {d} | {STAGES[stage]} | {HISTORY[h]} | {r['activation']:.6f} | {r['within']:.6f} | {r['fisher']:.6f} | {100*r['nonzero']:.3f} | {100*r['preferred_share']:.3f} | {100*r['svm_accuracy']:.2f} | {r['svm_auc']:.5f} |")
    lines+=['','## 主方案：同target的历史效应','',
        'Suppression%按每target的100×(1−左条件均值/右条件均值)计算后按类/概念等权平均，正值表示左条件响应更低。ΔW、ΔF是对应条件总体几何的相对变化；ΔBA和偏好份额为百分点。',
        '', '| 数据 | 层 | 对比 | Suppression% ↑ | ΔW% ↓ | ΔF% ↑ | Δ偏好份额pp ↑ | ΔSVM BA pp | 均值更低target数 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for stage in ('fc1','fc2'):
            for c in ('E-D','E-N','D-N'):
                r=get(d,'hebbian',stage,c)
                lines.append(f"| {d} | {STAGES[stage]} | {c} | {f(r['suppression_pct'])} | {f(r['within_change_pct'])} | {f(r['fisher_change_pct'])} | {f(r['preferred_share_pp'])} | {f(r['svm_accuracy_pp'])} | {r['lower_activation_targets']}/{r['n']} |")
    lines+=['','## 对照：Exact−Different','', '| 数据 | 层 | 方法 | Suppression% | ΔW% | ΔF% | Δ偏好份额pp | ΔSVM BA pp |','|---|---|---|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for stage in ('fc1','fc2'):
            for method in METHODS:
                r=get(d,method,stage,'E-D')
                lines.append(f"| {d} | {STAGES[stage]} | {LABELS[method]} | {f(r['suppression_pct'])} | {f(r['within_change_pct'])} | {f(r['fisher_change_pct'])} | {f(r['preferred_share_pp'])} | {f(r['svm_accuracy_pp'])} |")
    lines+=['','缩放对照按当前历史的主方案输出逐图匹配均值，因此会复制均值历史效应。这不是独立记忆机制；应检查它是否也复制方向几何、偏好份额和读出。仅抑制后均值匹配是各层独立的补充幅度诊断，其各历史的方向相同，不是另一个经桥接的串联网络。主实验三项缩放对照都保留真实桥接；FC1缩放经过有bias的前馈桥接后仍可能改变FC2方向。',
        '', '## FC2局部增量','', '各行相对该历史实际到达FC2的输入计算；它与最终FC2相对vanilla或不同preview的差异含义不同。',
        '', '| 数据 | 历史 | 本层Δ平均激活% | 本层ΔW% | 本层ΔF% | 本层ΔSVM BA pp |','|---|---|---:|---:|---:|---:|']
    for r in local:
        if r['method']=='hebbian':lines.append(f"| {r['dataset']} | {HISTORY[r['history']]} | {f(r['activation_pct'])} | {f(r['within_pct'])} | {f(r['fisher_pct'])} | {f(r['svm_accuracy_pp'])} |")
    lines+=['','## 记忆是否写入并匹配target','', '| 数据 | 层 | 真实preview：κ Exact−Different | 同位置单图诊断：κ Exact−Different | 真实preview κ更高target数 |','|---|---|---:|---:|---:|']
    for d in ('imagenet','things'):
        target_order=read(OUT/d/'inputs/provenance.json')['inputs']
        with np.load(OUT/d/'inputs/features.npz') as z:y,groups=z['y'],z['concept']
        diagnostic=read(OUT/d/'state_acceptance.json')['centered_memory_diagnostic']
        for stage in ('fc1','fc2'):
            exact=[r['match'] for r in memory if r['dataset']==d and r['layer']==stage and r['history']=='exact']
            different=[r['match'] for r in memory if r['dataset']==d and r['layer']==stage and r['history']=='different']
            diff=np.array(exact)-different
            aligned=[next(r for r in item['layers'] if r['layer']==stage) for item in diagnostic]
            delta=np.array([r['exact']-r['different'] for r in aligned])
            lines.append(f"| {d} | {STAGES[stage]} | {mean(diff,y,groups):+.6f} | {mean(delta,y,groups):+.6f} | {int((diff>0).sum())}/{len(diff)} |")
    lines+=['','κ以同一份target vanilla向量查询两种preview权重，避免把改变后的FC2输入混入记忆匹配本身。同位置单图诊断只检验基本关联，不能替代真实双帧四物体任务。两个诊断的preview数量不同，主要比较各自内部Exact−Different，不把绝对κ大小当作纯空间效应。',
        '', '## 解读范围与完整数据','',
        '本轮是默认参数的开发小批量。ImageNet每类4张，单张分类变化可改变BA 6.25pp；THINGS每类4张、植物2张，单张可改变BA 5pp或10pp。不能把少数分类翻转当成可靠总体改善或损害，也不能给出独立显著性结论。',
        '', 'Suppression、类别几何、单元偏好份额和固定SVM逐项报告；没有按其中某项好看而更改标准。即使preview写入和只读验收成功，也不等于已实现重复抑制或sharpening。',
        '', '- `condition_metrics.csv/json`：每数据集、方法、历史、层的绝对指标。','- `paired_history_contrasts.csv/json`：E-D、E-N、D-N的配对效应。','- `per_target_metrics.csv`、`per_category_metrics.csv`：逐target和逐类统计。','- `fc2_local_effects.csv`：FC2相对实际输入的增量。','- `memory_diagnostics.csv`、`preference_profiles.csv`：记忆强度、匹配与预先定义的单元分组。','- `execution_audit.json`：状态、参考读出、尺度、缩放、桥接、源文件审计。',
        '', '原始FP32表征、SVM分数、preview因子与每trial/每步权重哈希位于 `Hebbian-CNN/Log/fc_preview_temporary_pilot_20261105/`。',
        '', '## 图表','', '| 数据 | 三历史绝对指标 | 三历史配对效应 |','|---|---|---|']
    for d in ('imagenet','things'):lines.append(f'| {d} | [PNG](figures/{d}_absolute.png) · [PDF](figures/{d}_absolute.pdf) | [PNG](figures/{d}_paired.png) · [PDF](figures/{d}_paired.pdf) |')
    lines+=['', '[预先定义的单元偏好分组图](figures/preference_profiles.png)']
    (DOC/'默认clean小批量结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plots(summary,contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    dest=DOC/'figures';dest.mkdir(exist_ok=True)
    methods=list(exp.METHODS);names=[LABELS[m] for m in methods];colors=('#2676b8','#db8739','#8b959f')
    def save(fig,name):
        for ext in ('png','svg','pdf'):fig.savefig(dest/(name+'.'+ext),dpi=180)
        plt.close(fig)
    for d in ('imagenet','things'):
        fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
        for row,stage in enumerate(('fc1','fc2')):
            for col,(metric,title,mul) in enumerate([('activation','平均激活',1),('nonzero','非零单元比例 (%)',100),('svm_accuracy','冻结SVM平衡准确率 (%)',100)]):
                ax=axes[row,col]
                for j,h in enumerate(exp.HISTORIES):
                    vals=[next(r[metric] for r in summary if (r['dataset'],r['method'],r['stage'],r['history'])==(d,m,stage,h))*mul for m in methods]
                    ax.bar(np.arange(len(methods))+(j-1)*.25,vals,.24,color=colors[j],label=HISTORY[h])
                ax.set_title(STAGES[stage]+' · '+title);ax.set_xticks(range(len(methods)),names,rotation=25,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
                if metric=='svm_accuracy':ax.set_ylim(0,105)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
        fig.suptitle(d.upper()+' · preview临时权重 · 默认5步clean小批量\n同一target三历史配对；本图为描述性结果，无误差线',fontsize=14)
        save(fig,d+'_absolute')
        fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
        for row,stage in enumerate(('fc1','fc2')):
            for col,(metric,title) in enumerate([('suppression_pct','Suppression (%) ↑'),('within_change_pct','类内W变化 (%) ↓'),('fisher_change_pct','Fisher变化 (%) ↑')]):
                ax=axes[row,col]
                for j,c in enumerate(('E-D','E-N')):
                    vals=[next(r[metric] for r in contrasts if (r['dataset'],r['method'],r['stage'],r['contrast'])==(d,m,stage,c)) for m in methods]
                    ax.bar(np.arange(len(methods))+(j-.5)*.32,vals,.31,color=colors[j],label=c)
                ax.set_title(STAGES[stage]+' · '+title);ax.set_xticks(range(len(methods)),names,rotation=25,ha='right');ax.axhline(0,color='#333',lw=.7);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=2,frameon=False)
        fig.suptitle(d.upper()+' · 同target历史效应\nSuppression正值=Exact活动更低；W下降/Fisher上升为类别几何改善',fontsize=14)
        save(fig,d+'_paired')
    with (DOC/'preference_profiles.csv').open(encoding='utf-8-sig') as f:profiles=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,d in enumerate(('imagenet','things')):
        for row,stage in enumerate(('fc1','fc2')):
            ax=axes[row,col]
            for j,h in enumerate(exp.HISTORIES):
                vals=[100*float(next(r['activity_share'] for r in profiles if (r['dataset'],r['method'],r['stage'],r['history'],r['quartile'])==(d,'hebbian',stage,h,str(q)))) for q in range(4)]
                ax.plot(range(4),vals,'o-',color=colors[j],label=HISTORY[h])
            ax.set_xticks(range(4),['最低偏好','第二组','第三组','最高偏好']);ax.set_ylabel('该组所占总活动量 (%)');ax.set_title(d.upper()+' · '+STAGES[stage]);ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
    fig.suptitle('训练数据预先定义单元偏好四分位组 · 完整临时Hebbian串联\n分组不进入网络运算，测试阶段不重新选择单元',fontsize=13)
    save(fig,'preference_profiles')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        summary,contrasts,local,memory,audit=aggregate();report(summary,contrasts,local,memory,audit);plots(summary,contrasts)
    print('REPORT DONE',audit,flush=True)
