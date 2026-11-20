"""Audit and report fixed FC1 gain .75 graph comparisons and serial controls."""
from pathlib import Path
import copy
import csv
import json
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predify_experiments.fc_gain075_serial_20261028 import OUT, DOC, GRAPHS, TIMES, STAGES, conditions, parameters, feature_keys, validate, read, sha
from predify_experiments import fc_serial_20261028 as serial

LABELS = {'fc1_output':'FC1本层 / vanilla FC1', 'fc2_input':'FC2前馈输入 / vanilla FC2',
          'fc2_output':'最终FC2 / vanilla FC2', 'fc2_local':'FC2侧向输出 / 实际FC2输入'}
FIELDS = ('activation', 'within', 'between', 'fisher', 'raw_mean', 'raw_within', 'raw_between', 'raw_fisher',
          'nonzero', 'zero_fraction', 'effective_dimension', 'population_activity_fraction', 'largest_unit_activity_share',
          'support_ratio', 'max_support_ratio', 'density_increase_images', 'revived_fraction', 'revived_activity_share', 'weak_activity_share')


def csv_save(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def json_save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def audit_geometry(a, y, m):
    a = a.astype(np.float64)
    u = a / np.maximum(np.linalg.norm(a, axis=1)[:, None], 1e-8)
    centers = np.stack([u[y == j].mean(0) for j in range(5)])
    w = np.mean([np.square(u[y == j] - centers[j]).sum(1).mean() for j in range(5)])
    b = np.square(centers - centers.mean(0)).sum(1).mean()
    values = dict(activation=np.mean([a[y == j].mean() for j in range(5)]), within=w, between=b, fisher=b/w)
    errors = [abs(float(values[k]) - m[k]) for k in values]
    assert max(errors) < 3e-6
    return max(errors)


def aggregate():
    protocol = read(OUT / 'protocol.json')
    original = validate(protocol)
    assert sha(OUT / 'source_snapshot.py') == protocol['source_sha256']
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    weights = torch.load(original['checkpoint'], map_location='cpu', weights_only=True)
    w, b = (weights[f'classifier.3.{k}'].to(serial.DEVICE) for k in ('weight', 'bias'))
    del weights
    rows, categories = [], []
    audit = dict(passed=True, folds=0, graph_conditions=0, unique_representations=0,
        new_svm_fits=0, reused_svm_fits=0, independent_geometry_max_error=0.,
        bridge_max_error=0., previous_svm_max_error=0., previous_fc1_grid_max_error=0.,
        original_model_unchanged=True, previous_experiments_unchanged=True,
        inputs_unchanged=True, concepts_disjoint=True, svm_primary=True, svm_selection=False,
        no_fixed_point_filter=True, mean_controls_matched=True, source_hashes={})
    for d in ('imagenet', 'things'):
        with np.load(serial.DATA / d / 'activations.npz') as vanilla:
            concepts, folds = vanilla['concept'], vanilla['fold']
        for fold in range(5):
            dest = OUT / d / f'fold{fold}'
            complete = read(dest / 'complete.json')
            assert complete['protocol_sha256'] == sha(OUT / 'protocol.json')
            for p, h in complete['files'].items():
                assert sha(dest / p) == h
            data = read(dest / 'results.json')
            audit['source_hashes'][f'{d}/{fold}'] = sha(dest / 'results.json')
            tr, te = np.array(data['train_rows']), np.array(data['test_rows'])
            assert np.array_equal(tr, np.flatnonzero(folds != fold)) and np.array_equal(te, np.flatnonzero(folds == fold))
            assert not set(concepts[tr]) & set(concepts[te])
            for field, value in (('new_svm_fits', data['trained']), ('reused_svm_fits', data['reused'])):
                audit[field] += value
            audit['previous_svm_max_error'] = max(audit['previous_svm_max_error'], data['old_svm_score_max_error'])
            audit['previous_fc1_grid_max_error'] = max(audit['previous_fc1_grid_max_error'], data['fc1_grid_replay_max_error'])
            with (dest / 'svm_models.pckl').open('rb') as f:
                models = pickle.load(f)
            with np.load(dest / 'representations.npz') as features, np.load(dest / 'scores.npz') as scores:
                assert np.array_equal(features['row'], te) and np.array_equal(scores['row'], te)
                checked = {}
                for graph in GRAPHS:
                    branch = data['graphs'][graph]
                    assert (branch['params1'], branch['params2']) == parameters(graph)
                    assert set(branch['results']) == {c['name'] for c in conditions()}
                    for name, entry in branch['results'].items():
                        c, metrics = entry['condition'], entry['metrics']
                        keys = feature_keys(graph, c)
                        assert entry['feature_keys'] == keys
                        base = branch['results']['baseline']['metrics']
                        for stage in STAGES:
                            m = metrics[stage]
                            ref = metrics['fc2_input'] if stage == 'fc2_local' else base[stage]
                            row = dict(dataset=d, graph=graph, fold=fold, name=name, **{k:v for k,v in c.items() if k != 'name'},
                                       stage=stage, reference=LABELS[stage])
                            row.update({k:m[k] for k in FIELDS})
                            row.update({k:m['effects'][k] for k in ('activation_pct', 'within_pct', 'between_pct', 'fisher_pct')})
                            for k in ('activation', 'within', 'between', 'fisher'):
                                assert abs(row[k+'_pct'] - 100*(m[k]/ref[k]-1)) < 1e-8
                            row.update(svm_accuracy=m['accuracy'], svm_auc=m['auc'], svm_macro_f1=m['macro_f1'],
                                svm_accuracy_pp=100*(m['accuracy']-ref['accuracy']), svm_auc_delta=m['auc']-ref['auc'],
                                frozen_svm_accuracy=m['frozen_svm']['accuracy'], frozen_svm_auc=m['frozen_svm']['auc'],
                                frozen_svm_accuracy_pp=100*(m['frozen_svm']['accuracy']-ref['frozen_svm']['accuracy']),
                                frozen_svm_auc_delta=m['frozen_svm']['auc']-ref['frozen_svm']['auc'],
                                original_imagenet_top1_pct=m['original_imagenet_top1_pct'],
                                original_imagenet_top1_pp=m['original_imagenet_top1_pct']-ref['original_imagenet_top1_pct'] if m['original_imagenet_top1_pct'] is not None else None,
                                joint=m['joint'], sparse_joint=m['sparse_joint'])
                            assert abs(row['svm_accuracy_pp']-m['effects']['accuracy_pp']) < 1e-9
                            rows.append(row)
                            for j, category in enumerate(('animals', 'clothing', 'food', 'plants', 'vehicles')):
                                categories.append(dict(dataset=d, graph=graph, fold=fold, name=name, stage=stage, category=category,
                                    activation=m['per_class_activation'][j], activation_pct=100*(m['per_class_activation'][j]/ref['per_class_activation'][j]-1),
                                    within=m['per_class_within'][j], within_pct=100*(m['per_class_within'][j]/ref['per_class_within'][j]-1),
                                    svm_recall=m['recalls'][j], svm_recall_pp=100*(m['recalls'][j]-ref['recalls'][j])))
                            if stage == 'fc2_local':
                                continue
                            key = keys[stage]
                            a, decision = features[key], scores[key]
                            assert a.dtype == np.float32 and a.shape == (len(te), 4096)
                            assert np.isfinite(a).all() and (a >= 0).all() and np.isfinite(decision).all()
                            if key not in checked:
                                error = audit_geometry(a, features['y'], m)
                                audit['independent_geometry_max_error'] = max(audit['independent_geometry_max_error'], error)
                                predicted = decision.argmax(1)
                                recalls = [float((predicted[features['y'] == j] == j).mean()) for j in range(5)]
                                auc = np.mean([roc_auc_score(features['y'] == j, decision[:,j]) for j in range(5)])
                                assert np.allclose(recalls, m['recalls'], atol=1e-12)
                                assert abs(np.mean(recalls)-m['accuracy']) < 1e-12 and abs(auc-m['auc']) < 1e-12
                                model = models[key]
                                assert model[-1].C == 1 and model[-1].max_iter == 30000 and model[-1].random_state == 20261028
                                assert int(model[-1].n_iter_) < model[-1].max_iter
                                checked[key] = m
                            else:
                                for k in ('activation', 'within', 'between', 'fisher', 'accuracy', 'auc'):
                                    assert m[k] == checked[key][k]
                        if c['m2'] in ('hard', 'inhib_hard'):
                            assert np.all((features[keys['fc2_output']] > 1e-7).sum(1) <= (features[keys['fc2_input']] > 1e-7).sum(1))
                        if c['m1'] == 'gain_hard':
                            target = branch['results']['fc1_hard_5']['feature_keys']['fc1_output']
                            assert np.allclose(features[keys['fc1_output']].mean(1), features[target].mean(1), atol=2e-6)
                        if c['m2'] == 'gain_hard':
                            target = branch['results'][f"both_hard_hard_5_5_{c['calibration']}"]['feature_keys']['fc2_output']
                            assert np.allclose(features[keys['fc2_output']].mean(1), features[target].mean(1), atol=2e-6)
                        audit['graph_conditions'] += 1
                assert len(checked) == 49
                for key in checked:
                    if not key.startswith('fc1__'):
                        continue
                    incoming = 'fc2_input__' + key.removeprefix('fc1__')
                    with torch.inference_mode():
                        predicted = serial.bridge(torch.as_tensor(features[key], device=serial.DEVICE), w, b).cpu().numpy()
                    error = float(np.abs(predicted-features[incoming]).max())
                    audit['bridge_max_error'] = max(audit['bridge_max_error'], error)
                    assert error < 1e-4
                with np.load(serial.OUT / d / f'fold{fold}' / 'final_fc2.npz') as previous:
                    for t in TIMES:
                        assert np.array_equal(features[f'fc2_output__fc2_hard_{t}'], previous[f'fc2_hard_{t}'])
                audit['unique_representations'] += len(checked)
            audit['folds'] += 1
    assert len(rows) == 1840 and audit['graph_conditions'] == 460 and audit['unique_representations'] == 490
    summary = []
    for d in ('imagenet', 'things'):
        for graph in GRAPHS:
            for c in conditions():
                for stage in STAGES:
                    rs = [r for r in rows if (r['dataset'],r['graph'],r['name'],r['stage']) == (d,graph,c['name'],stage)]
                    assert len(rs) == 5
                    s = {k:rs[0][k] for k in ('dataset','graph','name','m1','t1','m2','t2','calibration','stage','reference')}
                    for k in rs[0]:
                        if k in s or k in ('fold','joint','sparse_joint'):
                            continue
                        s[k] = float(np.mean([r[k] for r in rs])) if all(r[k] is not None for r in rs) else None
                    s['joint_folds'] = sum(r['joint'] for r in rs)
                    s['sparse_joint_folds'] = sum(r['sparse_joint'] for r in rs)
                    summary.append(s)
    assert len(summary) == 368
    paired = []
    for d in ('imagenet','things'):
        for c in conditions():
            for stage in STAGES:
                a,b = [next(r for r in summary if (r['dataset'],r['graph'],r['name'],r['stage']) == (d,g,c['name'],stage)) for g in GRAPHS]
                paired.append(dict(dataset=d,name=c['name'],stage=stage,
                    k32_minus_k128_activation_change_pp=a['activation_pct']-b['activation_pct'],
                    k32_minus_k128_within_change_pp=a['within_pct']-b['within_pct'],
                    k32_minus_k128_fisher_change_pp=a['fisher_pct']-b['fisher_pct'],
                    k32_minus_k128_svm_ba_pp=100*(a['svm_accuracy']-b['svm_accuracy'])))
    csv_save(DOC/'fold_metrics.csv',rows); csv_save(DOC/'summary.csv',summary)
    csv_save(DOC/'category_metrics.csv',categories); csv_save(DOC/'graph_comparison.csv',paired)
    json_save(DOC/'summary.json',summary); json_save(DOC/'audit.json',audit)
    validate(protocol)
    return summary,audit


def report(summary,audit):
    previous = read(ROOT/'docs/fc_serial_20261028/layer_summary.json')
    def get(d,g,n,stage='fc2_output'):
        return next(r for r in summary if (r['dataset'],r['graph'],r['name'],r['stage']) == (d,g,n,stage))
    def old(d,n,stage='fc2_output'):
        r = copy.deepcopy(next(r for r in previous if (r['dataset'],r['name'],r['stage']) == (d,n,stage)))
        r['svm_accuracy_pp'],r['svm_auc_delta'] = r['refit_svm_accuracy_pp'],r['refit_svm_auc_delta']
        return r
    def table(entries):
        lines = ['| 条件 | 均值变化% | W变化% | B变化% | Fisher变化% | 非零% | SVM BA变化pp | SVM AUC变化 |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
        for label,r in entries:
            lines.append(f"| {label} | {r['activation_pct']:+.2f} | {r['within_pct']:+.2f} | {r['between_pct']:+.2f} | {r['fisher_pct']:+.2f} | {100*r['nonzero']:.2f} | {r['svm_accuracy_pp']:+.2f} | {r['svm_auc_delta']:+.5f} |")
        return lines
    lines = ['# FC1增益0.75：k32/k128独立与串联结果', '',
        'FC1固定corr、gain0.75、hard、活动预算1；比较互为k32与k128。FC2固定corr/k32、gain0.75、hard、活动预算1。原始前馈权重与bias、预训练模型、两套vanilla缓存和旧结果不改动。每层同步1/5/20步；5步为预定主截面，无新选参，不按不动点筛选。', '',
        '主要目标为实际平均激活下降、逐图L2归一化类内W下降、Fisher=B/W提高，B单列。两层各自与vanilla基线比较；FC2局部效果另外相对其实际输入统计。SVM是功能检查，不参与选优。两层统一训练折内加权StandardScaler+LinearSVC(C=1)，与前轮相同；BA为平衡准确率，pp为百分点。', '',
        'frozen沿用vanilla训练FC2的侧向连接和尺度；matched只用当前实际进入FC2的训练表征重估二者。所有缩放对照匹配本轮同一FC1图的gain0.75/hard5结果。', '',
        '## 结果解读', '',
        '增强FC1的区分收益能够传递到最终FC2。5步matched串联中，k32的最终Fisher改善为ImageNet +13.48%、THINGS +19.83%；k128为+13.60%、+19.50%，均高于历史gain0.25/k128的+12.27%、+17.77%。FC1本层Fisher由约+1.5%提高到约+5%。但FC1均值比低增益时少下降约2个百分点，不能称为所有目标上均优。', '',
        'k32通常多降低一些均值，最终活动比例也更低；k128的FC1类内W收缩更强。ImageNet的最终Fisher略偏向k128，THINGS略偏向k32。没有跨数据集、跨指标的一致图赢家。SVM作为功能检查，不能以某个准确率高点替代sharpening判断。', '',
        '最终FC2的新增Fisher收益主要伴随类间B增大，W收缩并没有同步加强：matched下旧W变化为−2.82%/−4.00%，新k32为−2.64%/−3.91%，新k128为−2.77%/−3.92%（ImageNet/THINGS）。应称为更好的类间/类内比例，而不是所有几何指标都更好。', '',
        'FC1缩放+FC2 Hebbian仍有效，但不再近乎复现真实FC1 Hebbian：matched的缩放对照Fisher为k32 +11.47%/+16.75%、k128 +11.19%/+16.68%，真实串联分别为+13.48%/+19.83%、+13.60%/+19.50%。frozen下真实串联也有额外收益。两层纯缩放或仅抑制仅约+0.5%–0.6%；保留真实FC1但FC2仅缩放约+2.4%–2.8%。这支持增强FC1的结构变化有额外作用，同时FC2侧向处理仍承担较大区分作用，不将差值当作可加的因果贡献比例。', '',
        '原FC2非零比例为19.09%/20.38%，新串联k32为18.54%/19.83%、k128为18.62%/19.90%。相对vanilla的平均活动数下降约2.4%–2.9%；FC2局部hard门控逐图保持不超过实际输入的活动数。没有出现早期dense方法的大量弱响应铺开。', '',
        '从5步到20步，matched均值额外下降不到0.3个百分点，Fisher继续上升约0.6（ImageNet）或1.2（THINGS）个相对变化百分点。完整数值列在步数表中，不据此宣称最佳步数。', '',
        '![Sharpening tradeoff](<' + (DOC/'figures'/'sharpening_comparison.png').as_posix() + '>)', '',
        '## FC1本层：5步', '']
    for d in ('imagenet','things'):
        lines += [f'### {d}', ''] + table([('历史gain0.25/k128',old(d,'fc1_hard_5','fc1_output'))] +
            [(f'gain0.75/{g}',get(d,g,'fc1_hard_5','fc1_output')) for g in GRAPHS]) + ['']
    lines += ['## 最终FC2：5步真正串联', '', '所有数值相对vanilla FC2；仅FC1条件也经过原始前馈桥接。', '']
    for d in ('imagenet','things'):
        entries = [('仅FC2 Hebbian',get(d,'k32','fc2_hard_5')),('历史gain0.25/k128串联matched',old(d,'both_hard_hard_5_5_matched'))]
        for g in GRAPHS:
            entries += [(f'{g}/仅FC1传递',get(d,g,'fc1_hard_5'))]
            entries += [(f'{g}/串联/{cal}',get(d,g,f'both_hard_hard_5_5_{cal}')) for cal in ('frozen','matched')]
        lines += [f'### {d}', ''] + table(entries) + ['']
    lines += ['## FC1、前馈输入、FC2输出与局部效果分开', '']
    for d in ('imagenet','things'):
        entries = [(f'{g}/{LABELS[s]}',get(d,g,'both_hard_hard_5_5_matched',s)) for g in GRAPHS for s in STAGES]
        lines += [f'### {d}', ''] + table(entries) + ['']
    lines += ['## 5步对照：均值压低与Hebbian区分收益', '',
        'gain1仅将FC1替换为逐图均值匹配缩放，FC2仍做Hebbian；gain2保留FC1 Hebbian但FC2只作均值匹配缩放；gain_both两层均缩放。后两者最终均值逐图匹配真实串联。仅抑制去掉兴奋连接并保留同一hard门控与除法。matched对照按各自训练输入校准，frozen对照则共用vanilla FC2侧向模型。', '']
    for d in ('imagenet','things'):
        for cal in ('frozen','matched'):
            entries = []
            for g in GRAPHS:
                for prefix in ('both_hard_hard_5_5','gain1_hard_5','gain2_hard_5','gain_both_hard_5'):
                    n=f'{prefix}_{cal}'; entries.append((f'{g}/{prefix}',get(d,g,n)))
            entries.append(('两层仅抑制（matched）',get(d,'k32','inhibition_only_hard_5')))
            lines += [f'### {d}/{cal}', ''] + table(entries) + ['']
    lines += ['## 步数敏感性', '', '完整报告1/5/20步，不依据外层测试或SVM表现另选最佳步数。', '']
    for d in ('imagenet','things'):
        entries = [(f'{g}/{t}步/{s}',get(d,g,f'both_hard_hard_{t}_{t}_matched',s)) for g in GRAPHS for t in TIMES for s in ('fc1_output','fc2_output')]
        lines += [f'### {d}', ''] + table(entries) + ['']
    lines += ['## 辅助：原始冻结ImageNet 1000类head', '',
        '这个head未重新训练，与主要的大类SVM检查分别解释。THINGS没有对应1000类真值，不计算。', '',
        '| 条件 | 原始head Top1% | 相对vanilla变化pp |', '|---|---:|---:|']
    for label,g,n in [('vanilla','k32','baseline')] + [(f'{g}/{cal}/5步',g,f'both_hard_hard_5_5_{cal}') for g in GRAPHS for cal in ('frozen','matched')]:
        r=get('imagenet',g,n)
        lines.append(f"| {label} | {r['original_imagenet_top1_pct']:.2f} | {r['original_imagenet_top1_pp']:+.2f} |")
    lines += ['']
    lines += ['## 审计与文件', '',
        f"审计通过：{audit['folds']}折、{audit['graph_conditions']}个图×条件、{audit['unique_representations']}个唯一测试表征与SVM readout；新增拟合{audit['new_svm_fits']}个，复用{audit['reused_svm_fits']}个。FC1与旧独立网格重放最大误差{audit['previous_fc1_grid_max_error']:.3g}；旧SVM分数差异{audit['previous_svm_max_error']:.3g}；独立重算几何误差{audit['independent_geometry_max_error']:.3g}；原始前馈桥接回放误差{audit['bridge_max_error']:.3g}。10项串联、增益目标、图隔离与训练隔离测试通过。", '',
        'summary.csv/json含两数据集×两图×23条件×4统计口径，共368行。fold_metrics.csv有1840行逐折结果，category_metrics.csv保留分大类激活/类内离散度/召回，graph_comparison.csv列出k32−k128的描述性差异。完整测试激活、训练表征哈希、SVM模型和分数保存在对应Log目录。', '',
        'SVM类内预处理和所有侧向统计只使用训练折；matched同时改变连接统计和尺度，不把差异全部归因于矩阵重学。五折与旧探索样本重复使用，不作为独立受试者，不作显著性或统计非劣性声明。旧gain0.25参照是历史k128配置；比较旧k128与新k32同时改变了图和增益，应分别解读。', '']
    (DOC/'实验结果.md').write_text('\n'.join(lines),encoding='utf-8')


def plots(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    previous = read(ROOT/'docs/fc_serial_20261028/layer_summary.json')
    def get(d,g,n,s):
        return next(r for r in summary if (r['dataset'],r['graph'],r['name'],r['stage'])==(d,g,n,s))
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for i,d in enumerate(('imagenet','things')):
        for j,stage in enumerate(('fc1_output','fc2_output')):
            ax=axes[i,j]
            name='fc1_hard_5' if j==0 else 'both_hard_hard_5_5_matched'
            old=next(r for r in previous if (r['dataset'],r['name'],r['stage'])==(d,name,stage))
            points=[('old gain .25 / k128',old,'black','s')]
            for g,color in (('k32','tab:blue'),('k128','tab:orange')):
                points.append((f'gain .75 / {g}',get(d,g,name,stage),color,'o'))
                if j==1:
                    points.append((f'{g}: FC1 gain + FC2 Hebb',get(d,g,'gain1_hard_5_matched',stage),color,'^'))
                    points.append((f'{g}: gains both',get(d,g,'gain_both_hard_5_matched',stage),color,'x'))
                else:
                    points.append((f'{g}: FC1 gain only',get(d,g,'fc1_gain_5',stage),color,'x'))
            if j==1:
                points.append(('inhibition only',get(d,'k32','inhibition_only_hard_5',stage),'tab:green','D'))
            for label,r,color,marker in points:
                ax.scatter(-r['activation_pct'],r['fisher_pct'],label=label,color=color,marker=marker,s=65)
            ax.set_title(f'{d.upper()} | ' + ('FC1' if j==0 else 'final FC2, matched'))
            ax.set_xlabel('Mean activation reduction (%)');ax.set_ylabel('Fisher increase (%)')
            ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
    fig.suptitle('Fixed 5-step comparison | two sharpening objectives; SVM not used for ranking')
    fig.tight_layout()
    dest=DOC/'figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'sharpening_comparison.png',dpi=170)
    fig.savefig(dest/'sharpening_comparison.svg')
    plt.close(fig)


def main():
    summary,audit=aggregate(); plots(summary); report(summary,audit)
    for r in summary:
        if r['name']=='both_hard_hard_5_5_matched' and r['stage'] in ('fc1_output','fc2_output'):
            print(r['dataset'],r['graph'],r['stage'],{k:r[k] for k in ('activation_pct','within_pct','fisher_pct','svm_accuracy_pp','svm_auc_delta')})
    print('AUDIT',audit)


if __name__=='__main__':
    main()
