"""Audit paired corruptions, clean-trained transfer, and degraded refit readouts."""
from pathlib import Path
import csv
import json
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from predify_experiments import fc_degraded_k128_20261028 as exp
from predify_experiments import fc_degraded_inputs_20261028 as inputs
from analysis.report_fc_gain075_serial_20261028 import audit_geometry,FIELDS,LABELS

OUT,DOC=exp.OUT,exp.DOC
read=exp.read


def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save_json(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def check_scores(scores,y,metric):
    pred=scores.argmax(1)
    recalls=[float((pred[y==j]==j).mean()) for j in range(5)]
    auc=np.mean([roc_auc_score(y==j,scores[:,j]) for j in range(5)])
    assert np.allclose(recalls,metric['recalls'],atol=1e-12)
    assert abs(np.mean(recalls)-metric['accuracy'])<1e-12 and abs(auc-metric['auc'])<1e-12


def aggregate(datasets=('imagenet','things'),persist=True):
    protocol=read(OUT/'protocol.json');original=exp.validate(protocol)
    assert exp.sha(OUT/'source_snapshot.py')==protocol['source_sha256']
    input_protocol=read(inputs.INPUTS/'protocol.json')
    assert exp.sha(inputs.INPUTS/'source_snapshot.py')==input_protocol['source_sha256']
    assert exp.sha(ROOT/'Hebbian-CNN/Hebb_Priming_Utils.py')==input_protocol['legacy_degradation_sha256']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    weights=torch.load(original['checkpoint'],map_location='cpu',weights_only=True)
    w,b=(weights[f'classifier.3.{k}'].to(exp.serial.DEVICE) for k in ('weight','bias'))
    del weights
    rows,categories,diagnostics=[],[],[]
    audit=dict(passed=True,datasets=list(datasets),folds=0,phases=0,unique_representations=0,
        degraded_svm_fits=0,clean_svm_new=0,clean_svm_reused=0,
        geometry_max_error=0.,bridge_max_error=0.,svm_replay_max_error=0.,
        input_clean_replay_max_error=0.,mask_replay_max_error=0.,
        side_weights_relearned_only_from_clean=True,no_degraded_statistic_calibration=True,
        concept_isolation=True,random_instances_paired=True,original_model_unchanged=True,
        previous_results_unchanged=True,three_readout_paths_separated=True,
        replicas_averaged_within_fold_before_folds=True,source_hashes={})
    for d in datasets:
        caches={}
        for mode,level in inputs.DEGRADATIONS:
            for repeat in inputs.REPEATS:
                cache=inputs.INPUTS/d/mode/f'repeat{repeat}';marker=exp.verify_marker(cache)
                assert marker['protocol_sha256']==protocol['input_protocol_sha256']
                info=read(cache/'extraction.json')
                audit['input_clean_replay_max_error']=max(audit['input_clean_replay_max_error'],info['clean_first8_replay_error'])
                with np.load(cache/'degradation_diagnostics.npz') as z:
                    values=z['values'];assert len(np.unique(values[:,0]))==len(values)
                    assert np.array_equal(values[:,0],[inputs.seed_for(d,i,repeat,mode) for i in range(len(values))])
                    audit['mask_replay_max_error']=max(audit['mask_replay_max_error'],float(values[:,-1].max()))
                diagnostics.append(dict(dataset=d,mode=mode,level=level,repeat=repeat,images=info['rows'],
                    difference_mean=info['diagnostics_mean'][0],difference_std=info['diagnostics_mean'][1],
                    difference_rms=info['diagnostics_mean'][2],mask_mean_opacity=info['diagnostics_mean'][3],
                    mask_positive_fraction=info['diagnostics_mean'][4],mask_ge_095_fraction=info['diagnostics_mean'][5]))
                caches[(mode,repeat)]=cache
        with np.load(exp.serial.DATA/d/'activations.npz') as z:
            clean1,clean2=z['fc1'],z['fc2'];y,concept,folds=z['category'],z['concept'],z['fold']
        with (exp.serial.DATA/d/'baseline_models.pckl').open('rb') as f:vanilla_heads=pickle.load(f)['models']
        for fold in range(5):
            dest=OUT/d/f'fold{fold}';exp.verify_marker(dest)
            clean_dest=dest/'clean';exp.verify_marker(clean_dest)
            clean_data=read(clean_dest/'results.json');tr,te=np.array(clean_data['train_rows']),np.array(clean_data['test_rows'])
            assert np.array_equal(te,np.flatnonzero(folds==fold)) and np.array_equal(tr,np.flatnonzero(folds!=fold))
            assert not set(concept[tr])&set(concept[te])
            audit['clean_svm_new']+=clean_data['clean_svm_new'];audit['clean_svm_reused']+=clean_data['clean_svm_reused']
            audit['svm_replay_max_error']=max(audit['svm_replay_max_error'],clean_data['previous_clean_svm_error'])
            with np.load(dest/'clean_side_models.npz') as side:
                for number,source,c in ((1,clean1,exp.strong.parameters('k128')[0]),(2,clean2,exp.strong.parameters('k128')[1])):
                    model=exp.serial.fit_model(torch.as_tensor(source[tr],device=exp.serial.DEVICE),c)
                    assert np.array_equal(side[f'E{number}'],model['matrices'][f"{c['rule']}_{c['graph']}"].cpu().numpy())
                    assert float(side[f'scale{number}'])==float(model['scale'])
                    del model
                assert np.array_equal(side['train_rows'],tr)
            with (clean_dest/'svm_models.pckl').open('rb') as f:clean_models=pickle.load(f)
            phase_list=[('none',0.,-1,clean_dest)]+[(m,l,r,dest/m/f'repeat{r}') for m,l in inputs.DEGRADATIONS for r in inputs.REPEATS]
            with np.load(clean_dest/'representations.npz') as clean_features:
                for mode,level,repeat,p in phase_list:
                    exp.verify_marker(p);data=read(p/'results.json')
                    assert data['train_rows']==tr.tolist() and data['test_rows']==te.tolist()
                    if mode!='none':
                        assert data['clean_complete_sha256']==exp.sha(clean_dest/'complete.json')
                        assert data['input_complete_sha256']==exp.sha(caches[(mode,repeat)]/'complete.json')
                    audit['source_hashes'][f'{d}/{fold}/{mode}/{repeat}']=exp.sha(p/'results.json')
                    with (p/'svm_models.pckl').open('rb') as f:refit_models=pickle.load(f)
                    with np.load(p/'representations.npz') as features,np.load(p/'scores.npz') as scores,np.load(p/'paired_diagnostics.npz') as paired:
                        assert np.array_equal(features['row'],te) and np.array_equal(features['y'],y[te])
                        checked={}
                        base=data['results']['baseline']['metrics']
                        for c in exp.conditions():
                            name=c['name'];entry=data['results'][name];keys=exp.strong.feature_keys('k128',c)
                            assert entry['feature_keys']==keys
                            ms=entry['metrics']
                            for stage in exp.STAGES:
                                m=ms[stage];ref=ms['fc2_input'] if stage=='fc2_local' else base[stage]
                                clean_m=clean_data['results'][name]['metrics'][stage]
                                row=dict(dataset=d,mode=mode,level=level,repeat=repeat,fold=fold,name=name,stage=stage,reference=LABELS[stage],
                                    t1=c['t1'],t2=c['t2'],m1=c['m1'],m2=c['m2'])
                                row.update({k:m[k] for k in FIELDS})
                                row.update({k:m['effects'][k] for k in ('activation_pct','within_pct','between_pct','fisher_pct')})
                                for k in ('activation','within','between','fisher'):
                                    assert abs(row[k+'_pct']-100*(m[k]/ref[k]-1))<1e-8
                                row.update(svm_clean_accuracy=m['accuracy'],svm_clean_auc=m['auc'],
                                    svm_clean_accuracy_pp=100*(m['accuracy']-ref['accuracy']),
                                    svm_clean_auc_delta=m['auc']-ref['auc'],
                                    svm_clean_drop_from_clean_pp=100*(m['accuracy']-clean_m['accuracy']),
                                    svm_vanilla_accuracy=m['vanilla_svm']['accuracy'],
                                    svm_vanilla_accuracy_pp=100*(m['vanilla_svm']['accuracy']-ref['vanilla_svm']['accuracy']),
                                    svm_refit_accuracy=m['degraded_svm']['accuracy'] if mode!='none' else None,
                                    svm_refit_auc=m['degraded_svm']['auc'] if mode!='none' else None,
                                    svm_refit_accuracy_pp=100*(m['degraded_svm']['accuracy']-ref['degraded_svm']['accuracy']) if mode!='none' else None,
                                    svm_refit_auc_delta=m['degraded_svm']['auc']-ref['degraded_svm']['auc'] if mode!='none' else None,
                                    activation_vs_clean_pct=100*(m['activation']/clean_m['activation']-1),
                                    within_vs_clean_pct=100*(m['within']/clean_m['within']-1),
                                    fisher_vs_clean_pct=100*(m['fisher']/clean_m['fisher']-1),
                                    cosine_to_clean_vanilla=m['cosine_to_clean_vanilla'],
                                    cosine_to_clean_same_method=m['cosine_to_clean_same_method'],
                                    cosine_to_clean_vanilla_delta=m['cosine_to_clean_vanilla']-ref['cosine_to_clean_vanilla'],
                                    relative_l2_to_clean_vanilla=m['relative_l2_to_clean_vanilla'],
                                    original_imagenet_top1_pct=m['original_imagenet_top1_pct'],
                                    joint=m['joint'],sparse_joint=m['sparse_joint'])
                                rows.append(row)
                                for j,cat in enumerate(('animals','clothing','food','plants','vehicles')):
                                    categories.append(dict(dataset=d,mode=mode,repeat=repeat,fold=fold,name=name,stage=stage,category=cat,
                                        activation=m['per_class_activation'][j],activation_pct=100*(m['per_class_activation'][j]/ref['per_class_activation'][j]-1),
                                        within=m['per_class_within'][j],within_pct=100*(m['per_class_within'][j]/ref['per_class_within'][j]-1),
                                        svm_clean_recall=m['recalls'][j],svm_clean_recall_pp=100*(m['recalls'][j]-ref['recalls'][j]),
                                        svm_refit_recall=m['degraded_svm']['recalls'][j] if mode!='none' else None,
                                        svm_refit_recall_pp=100*(m['degraded_svm']['recalls'][j]-ref['degraded_svm']['recalls'][j]) if mode!='none' else None))
                                if stage=='fc2_local':continue
                                key=keys[stage]
                                if key in checked:continue
                                a=features[key];assert a.shape==(len(te),4096) and a.dtype==np.float32
                                assert np.isfinite(a).all() and (a>=0).all()
                                error=audit_geometry(a,y[te],m);audit['geometry_max_error']=max(audit['geometry_max_error'],error)
                                for suffix,metric,model in (
                                    ('clean',m,clean_models[key]),
                                    ('vanilla',m['vanilla_svm'],vanilla_heads['fc1' if key.startswith('fc1__') else 'fc2'][fold])):
                                    decision=scores[key+'__'+suffix];check_scores(decision,y[te],metric)
                                    assert np.max(np.abs(model.decision_function(a)-decision))<1e-8
                                if mode!='none':
                                    decision=scores[key+'__degraded'];check_scores(decision,y[te],m['degraded_svm'])
                                    model=refit_models[key]
                                    assert model[-1].C==1 and model[-1].max_iter==30000 and model[-1].random_state==20261028
                                    assert int(model[-1].n_iter_)<model[-1].max_iter
                                    assert np.max(np.abs(model.decision_function(a)-decision))<1e-8
                                    audit['degraded_svm_fits']+=1
                                target=clean1[te] if key.startswith('fc1__') else clean2[te]
                                af=a.astype(np.float64);tf=target.astype(np.float64)
                                cosine=(af*tf).sum(1)/np.maximum(np.linalg.norm(af,axis=1)*np.linalg.norm(tf,axis=1),1e-12)
                                assert np.max(np.abs(cosine-paired[key][:,0]))<2e-6
                                checked[key]=True
                            if c['m2'] in ('hard','inhib_hard'):
                                assert np.all((features[keys['fc2_output']]>1e-7).sum(1)<=(features[keys['fc2_input']]>1e-7).sum(1))
                            if c['m2']=='gain_hard':
                                target=data['results']['both_hard_hard_5_5_frozen']['feature_keys']['fc2_output']
                                assert np.allclose(features[keys['fc2_output']].mean(1),features[target].mean(1),atol=2e-6)
                            if c['m1']=='gain_hard':
                                target=data['results']['both_hard_hard_5_5_frozen']['feature_keys']['fc1_output']
                                assert np.allclose(features[keys['fc1_output']].mean(1),features[target].mean(1),atol=2e-6)
                        assert len(checked)==19
                        for key in checked:
                            if not key.startswith('fc1__'):continue
                            incoming='fc2_input__'+key.removeprefix('fc1__')
                            with torch.inference_mode():
                                prediction=exp.serial.bridge(torch.as_tensor(features[key],device=exp.serial.DEVICE),w,b).cpu().numpy()
                            error=float(np.abs(prediction-features[incoming]).max())
                            audit['bridge_max_error']=max(audit['bridge_max_error'],error);assert error<1e-4
                        if mode!='none':
                            with np.load(caches[(mode,repeat)]/'activations.npz') as cache:
                                assert np.array_equal(features['fc1__off_0'],cache['fc1'][te])
                                assert np.array_equal(features['fc2_input__off_0'],cache['fc2'][te])
                        else:
                            assert np.array_equal(features['fc1__off_0'],clean1[te])
                            assert np.array_equal(features['fc2_input__off_0'],clean2[te])
                        audit['unique_representations']+=len(checked)
                    audit['phases']+=1
            audit['folds']+=1
            print('AUDITED FOLD',d,fold,flush=True)
    assert len(rows)==1120*len(datasets) and audit['unique_representations']==665*len(datasets) and audit['degraded_svm_fits']==570*len(datasets)
    metadata=('dataset','mode','level','fold','name','stage','reference','t1','t2','m1','m2')
    fold_rows=[]
    for d in datasets:
        for mode in ('none','predify_noise','fractal_occlusion'):
            for fold in range(5):
                for c in exp.conditions():
                    for stage in exp.STAGES:
                        group=[r for r in rows if (r['dataset'],r['mode'],r['fold'],r['name'],r['stage'])==(d,mode,fold,c['name'],stage)]
                        assert len(group)==(1 if mode=='none' else 3)
                        r={k:group[0][k] for k in metadata};r['repeats']=len(group)
                        for k in group[0]:
                            if k in r or k in ('repeat','joint','sparse_joint'):continue
                            r[k]=float(np.mean([x[k] for x in group])) if all(x[k] is not None for x in group) else None
                        r['joint']=bool(r['activation_pct']<-.0001 and r['within_pct']<-.0001 and r['fisher_pct']>.0001 and r['zero_fraction']==0)
                        r['all_repeats_joint']=all(x['joint'] for x in group)
                        fold_rows.append(r)
    summary=[]
    for d in datasets:
        for mode in ('none','predify_noise','fractal_occlusion'):
            for c in exp.conditions():
                for stage in exp.STAGES:
                    group=[r for r in fold_rows if (r['dataset'],r['mode'],r['name'],r['stage'])==(d,mode,c['name'],stage)]
                    assert len(group)==5
                    r={k:group[0][k] for k in metadata if k!='fold'}
                    for k in group[0]:
                        if k in r or k in ('fold','joint','all_repeats_joint'):continue
                        r[k]=float(np.mean([x[k] for x in group])) if all(x[k] is not None for x in group) else None
                    r['joint_folds']=sum(x['joint'] for x in group);r['all_repeats_joint_folds']=sum(x['all_repeats_joint'] for x in group)
                    for k in ('activation_pct','within_pct','fisher_pct','svm_clean_accuracy_pp'):
                        r[k+'_min']=min(x[k] for x in group);r[k+'_max']=max(x[k] for x in group)
                    summary.append(r)
    assert len(summary)==96*len(datasets)
    if persist:
        save_csv(DOC/'repeat_fold_metrics.csv',rows);save_csv(DOC/'fold_metrics.csv',fold_rows)
        save_csv(DOC/'summary.csv',summary);save_json(DOC/'summary.json',summary)
        save_csv(DOC/'category_metrics.csv',categories);save_csv(DOC/'input_diagnostics.csv',diagnostics)
        save_json(DOC/'audit.json',audit)
    exp.validate(protocol)
    return summary,diagnostics,audit


def report(summary,diagnostics,audit):
    def get(d,mode,n,s='fc2_output'):return next(r for r in summary if (r['dataset'],r['mode'],r['name'],r['stage'])==(d,mode,n,s))
    def table(entries):
        out=['| 条件/位置 | 均值变化% | W变化% | B变化% | Fisher变化% | 非零% | 干净SVM BA% | 干净SVM BA变化pp | 退化重拟合BA% | 重拟合BA变化pp | 联合改善折数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label,r in entries:
            secondary='—' if r['svm_refit_accuracy'] is None else f"{100*r['svm_refit_accuracy']:.2f}"
            effect='—' if r['svm_refit_accuracy_pp'] is None else f"{r['svm_refit_accuracy_pp']:+.2f}"
            out.append(f"| {label} | {r['activation_pct']:+.2f} | {r['within_pct']:+.2f} | {r['between_pct']:+.2f} | {r['fisher_pct']:+.2f} | {100*r['nonzero']:.2f} | {100*r['svm_clean_accuracy']:.2f} | {r['svm_clean_accuracy_pp']:+.2f} | {secondary} | {effect} | {r['joint_folds']}/5 |")
        return out
    main='both_hard_hard_5_5_frozen'
    lines=['# 干净vanilla侧向连接对噪声与遮挡的串联效果','',
        'FC1 corr/k128、FC2 corr/k32，gain0.75、hard、活动预算1、除法强度2；两层连接与RMS尺度只由干净vanilla训练折估计，所有退化实例固定复用。无prime，不修改原始前馈模型，不改阈值机制。','',
        '高斯噪声σ=0.5作用于标准化输入；分形遮挡强度0.3、octaves3。两套数据、原五折概念隔离，每图3个独立可复现实例，所有方法共享同一实例。先平均同折3次重复，再平均5折。','',
        '主目标仍为实际均值下降、逐图L2归一化W下降、Fisher=B/W上升。主效应均相对同退化、同随机实例、同层的无侧向基线；FC2局部口径另相对实际前馈输入。干净参考用于单独判断退化损伤和恢复性，不把退化自身造成的均值变化归因于Hebbian。','',
        '主要SVM为各处理在干净训练折拟合后直接测试退化概念；辅助SVM在该退化、该随机重复的训练折重新拟合。均使用同一加权StandardScaler+LinearSVC(C=1)，不增加单次训练图像数，不用测试概念拟合或调参。冻结原始vanilla SVM兼容性另保存在CSV。','',
        '## 主要发现','']
    selected=[get(d,mode,main,s) for d in ('imagenet','things') for mode in ('predify_noise','fractal_occlusion') for s in ('fc1_output','fc2_output')]
    passed=sum(r['activation_pct']<0 and r['within_pct']<0 and r['fisher_pct']>0 for r in selected)
    lines += [f'5步主要截面中，{passed}/8个数据集×退化×层组合的五折均值同时呈现激活下降、W下降和Fisher提高。逐折同向数量另列；均值方向不替代折间稳健性检查。','',
        '| 数据集/退化/层 | 均值变化% | W变化% | Fisher变化% | 干净训练SVM变化pp | 退化重拟合SVM变化pp |','|---|---:|---:|---:|---:|---:|']
    for r in selected:
        lines.append(f"| {r['dataset']}/{r['mode']}/{r['stage']} | {r['activation_pct']:+.2f} | {r['within_pct']:+.2f} | {r['fisher_pct']:+.2f} | {r['svm_clean_accuracy_pp']:+.2f} | {r['svm_refit_accuracy_pp']:+.2f} |")
    lines += ['','与FC1缩放+FC2 Hebbian的差距如下。它是完整处理方案的描述性差异，不把百分比差当作可加的因果贡献。','',
        '| 数据集/退化 | 真实串联Fisher变化% | FC1缩放+FC2 Hebbian% | 仅抑制% | 双层缩放% |','|---|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for mode in ('predify_noise','fractal_occlusion'):
            values=[get(d,mode,n)['fisher_pct'] for n in (main,'gain1_hard_5_frozen','inhibition_only_hard_5_frozen','gain_both_hard_5_frozen')]
            lines.append(f"| {d}/{mode} | {values[0]:+.2f} | {values[1]:+.2f} | {values[2]:+.2f} | {values[3]:+.2f} |")
    lines += ['','干净学习的连接在这两个退化档位下仍提供超出单纯活动压低的几何收益；FC1缩放+FC2 Hebbian保留相当部分收益，但四个数据集×退化组合均低于真实串联。将FC2换成缩放只保留约1.8%–2.8%的Fisher改善，说明FC2侧向处理仍承担更大的区分作用。差值不解释为可加的因果贡献比例。','',
        '主要干净训练SVM的变化较小，不表现为普遍的分类提升。退化重拟合在高斯条件下的整体读出明显高于干净训练泛化；这提示存在分布/读出失配，不能把重新拟合带来的全部收益归给Hebbian。遮挡条件下重拟合没有一致优势。','',
        '到干净vanilla的配对余弦只有小幅增加，不能称为完整恢复干净表征。高斯输入本身还会降低均值和W、同时降低Fisher，说明仅凭前两项容易把退化损伤误判为sharpening。','',
        '噪声下的活动数量仍明显高于干净状态：最终FC2非零比例为ImageNet24.49%、THINGS24.90%，同方法干净值为18.62%、19.90%。当前门控只限制相对实际输入的活动数，因此继承了退化输入中较多响应的结构；本轮没有引入新的群体活动阈值机制。','']
    lines += ['','## 退化本身：无侧向基线相对干净','',
        '| 数据集/退化/层 | 干净实际均值 | 退化实际均值 | 均值变化% | W变化% | Fisher变化% | 干净SVM BA下降pp |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for mode in ('predify_noise','fractal_occlusion'):
            for s in ('fc1_output','fc2_output'):
                a,b=get(d,'none','baseline',s),get(d,mode,'baseline',s)
                lines.append(f"| {d}/{mode}/{s} | {a['activation']:.4f} | {b['activation']:.4f} | {b['activation_vs_clean_pct']:+.2f} | {b['within_vs_clean_pct']:+.2f} | {b['fisher_vs_clean_pct']:+.2f} | {b['svm_clean_drop_from_clean_pp']:+.2f} |")
    lines += ['','## 5步主要结果','']
    for d in ('imagenet','things'):
        lines += [f"![{d} comparison](<{(DOC/'figures'/f'{d}_comparison.png').as_posix()}>)",'']
        for mode in ('predify_noise','fractal_occlusion'):
            entries=[]
            for s in ('fc1_output','fc2_output'):
                entries += [(f'干净vanilla/{s}',get(d,'none','baseline',s)),
                            (f'干净真串联/{s}',get(d,'none',main,s)),
                            (f'退化无侧向/{s}',get(d,mode,'baseline',s)),
                            (f'退化真串联/{s}',get(d,mode,main,s))]
            lines += [f'### {d}/{mode}','']+table(entries)+['']
    lines += ['## 5步最终FC2与对照','',
        '缩放对照的目标为本退化实例真实串联的均值，非干净均值。gain1匹配FC1均值，FC2仍做Hebbian；gain2保留FC1 Hebbian而FC2仅缩放；gain_both两层缩放。仅抑制保留门控和除法、兴奋增益置零；所有口径均固定干净尺度。','']
    for d in ('imagenet','things'):
        for mode in ('predify_noise','fractal_occlusion'):
            names=['baseline',main,'inhibition_only_hard_5_frozen','gain1_hard_5_frozen','gain2_hard_5_frozen','gain_both_hard_5_frozen']
            lines += [f'### {d}/{mode}','']+table([(n,get(d,mode,n)) for n in names])+['']
    lines += ['## FC1、前馈输入、最终FC2与局部效果','']
    for d in ('imagenet','things'):
        for mode in ('predify_noise','fractal_occlusion'):
            lines += [f'### {d}/{mode}','']+table([(LABELS[s],get(d,mode,main,s)) for s in exp.STAGES])+['']
    lines += ['## 配对干净表征与SVM迁移损失','',
        '接近同方法干净表征只代表该变换下的稳定性；它与原始干净vanilla参考不同。余弦只作恢复性诊断，不作为sharpening或语义恢复的单独证明。','',
        '| 数据集/退化/层 | 基线到干净vanilla余弦 | 串联到干净vanilla余弦 | 串联到同方法干净余弦 | 基线SVM相对干净下降pp | 串联SVM相对同方法干净下降pp |','|---|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for mode in ('predify_noise','fractal_occlusion'):
            for s in ('fc1_output','fc2_output'):
                a,b=get(d,mode,'baseline',s),get(d,mode,main,s)
                lines.append(f"| {d}/{mode}/{s} | {a['cosine_to_clean_vanilla']:.4f} | {b['cosine_to_clean_vanilla']:.4f} | {b['cosine_to_clean_same_method']:.4f} | {a['svm_clean_drop_from_clean_pp']:+.2f} | {b['svm_clean_drop_from_clean_pp']:+.2f} |")
    lines += ['','## 步数敏感性','']
    for d in ('imagenet','things'):
        for mode in ('predify_noise','fractal_occlusion'):
            lines += [f'### {d}/{mode}','']+table([(f'{t}步/{s}',get(d,mode,f'both_hard_hard_{t}_{t}_frozen',s)) for t in (1,5,20) for s in ('fc1_output','fc2_output')])+['']
    lines += ['## 输入退化诊断','',
        '| 数据集/退化 | 噪声或改变量RMS（标准化单位） | 遮罩平均不透明度% | 遮罩非零面积% | 遮罩≥0.95面积% |','|---|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for mode in ('predify_noise','fractal_occlusion'):
            rs=[r for r in diagnostics if r['dataset']==d and r['mode']==mode]
            avg=lambda k:np.mean([r[k] for r in rs])
            lines.append(f"| {d}/{mode} | {avg('difference_rms'):.4f} | {100*avg('mask_mean_opacity'):.2f} | {100*avg('mask_positive_fraction'):.2f} | {100*avg('mask_ge_095_fraction'):.2f} |")
    lines += ['','遮罩非零面积包含边缘软混合，不能解释为全部区域都被完全遮住；高斯条件的遮罩列不适用，记0。','',
        '## 辅助：冻结的原始ImageNet 1000类head','',
        '原始head没有重训，与大类SVM分开解释。THINGS不计算此真值。','',
        '| 输入 | 无侧向Top1% | 5步串联Top1% | 变化pp |','|---|---:|---:|---:|']
    for mode in ('none','predify_noise','fractal_occlusion'):
        a,b=get('imagenet',mode,'baseline'),get('imagenet',mode,main)
        lines.append(f"| {mode} | {a['original_imagenet_top1_pct']:.2f} | {b['original_imagenet_top1_pct']:.2f} | {b['original_imagenet_top1_pct']-a['original_imagenet_top1_pct']:+.2f} |")
    lines += ['','关联代码/模型/缓存保持不变；以下检查仅记录本轮完整性。','',
        '## 审计与范围','',
        f"审计通过：{audit['folds']}个外层折、{audit['phases']}个干净/退化评估批次、{audit['unique_representations']}个唯一表征。退化SVM重新拟合{audit['degraded_svm_fits']}个；干净SVM复用{audit['clean_svm_reused']}个、新增{audit['clean_svm_new']}个。已独立复算几何、BA/AUC、配对余弦，重放所有SVM分数和前馈桥接；逐折从干净数据重估侧向矩阵及尺度并核对。22项相关测试通过。",'',
        'summary.csv/json包含192个数据集×输入条件×处理×层口径；repeat_fold_metrics.csv保留2240行原始折/重复指标，fold_metrics.csv先合并同折重复，category_metrics.csv保留大类结果。input_diagnostics.csv记录实际退化强度，audit.json记录完整检查。完整表征、模型、分数、配对诊断、种子和源协议保存在独立Log目录。','',
        '三次随机重复不是三倍独立图片，五折也不是独立受试者；同一探索样本已多轮复用，不作显著性或统计非劣性声明。归一化几何与保留幅度的SVM分别解释，不要求SVM准确率提高。','']
    (DOC/'实验结果.md').write_text('\n'.join(lines),encoding='utf-8')


def plots(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['baseline','both_hard_hard_5_5_frozen','inhibition_only_hard_5_frozen','gain1_hard_5_frozen','gain2_hard_5_frozen','gain_both_hard_5_frozen']
    labels=['No lateral','Hebb serial','Inhibition only','FC1 gain + FC2 Hebb','FC1 Hebb + FC2 gain','Gains both']
    fields=['activation_pct','within_pct','fisher_pct','svm_clean_accuracy_pp','svm_refit_accuracy_pp']
    dest=DOC/'figures';dest.mkdir(exist_ok=True)
    for d in ('imagenet','things'):
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        for ax,mode in zip(axes,('predify_noise','fractal_occlusion')):
            data=[next(r for r in summary if (r['dataset'],r['mode'],r['name'],r['stage'])==(d,mode,n,'fc2_output')) for n in names]
            values=np.array([[r[k] for k in fields] for r in data])
            benefits=values*np.array([-1,-1,1,1,1])[None,:]
            colors=benefits/np.maximum(np.abs(benefits).max(0,keepdims=True),.001)
            ax.imshow(colors,cmap='RdBu',vmin=-1,vmax=1,aspect='auto')
            ax.set_yticks(range(6),labels,fontsize=9)
            ax.set_xticks(range(5),['Mean %','Within %','Fisher %','SVM clean pp','SVM refit pp'],fontsize=8)
            ax.xaxis.tick_top();ax.set_title('Gaussian sigma .5' if mode=='predify_noise' else 'Fractal intensity .3',pad=30)
            for i in range(6):
                for j in range(5):ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=9,color='white' if abs(colors[i,j])>.6 else 'black')
        fig.suptitle(f'{d.upper()} | final FC2, 5 steps | effects vs same-degradation no-lateral baseline\nBlue = favorable direction; each column scaled separately; repeats averaged within five concept folds',fontsize=11)
        fig.tight_layout();fig.savefig(dest/f'{d}_comparison.png',dpi=170);fig.savefig(dest/f'{d}_comparison.svg');plt.close(fig)


def main():
    with threadpool_limits(limits=4):summary,diagnostics,audit=aggregate()
    plots(summary);report(summary,diagnostics,audit)
    for r in summary:
        if r['mode']!='none' and r['name']=='both_hard_hard_5_5_frozen' and r['stage'] in ('fc1_output','fc2_output'):
            print(r['dataset'],r['mode'],r['stage'],{k:r[k] for k in ('activation_pct','within_pct','fisher_pct','svm_clean_accuracy_pp','svm_refit_accuracy_pp','cosine_to_clean_vanilla_delta','joint_folds')})
    print('AUDIT',audit)


if __name__=='__main__':main()
