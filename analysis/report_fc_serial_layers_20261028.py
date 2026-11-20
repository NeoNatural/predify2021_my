"""Report each serial FC stage against its own explicit reference, without rerunning."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predify_experiments.fc_serial_svm_completion_20261028 import OUT as SVM_OUT, feature_keys, validate_original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Hebbian-CNN/Log/fc_serial_20261028'
DOC = ROOT / 'docs/fc_serial_20261028'
STAGES = ('fc1_output', 'fc2_input', 'fc2_output', 'fc2_local')
LABELS = {
    'fc1_output': 'FC1侧向输出 / vanilla FC1',
    'fc2_input': 'FC2前馈输入 / vanilla FC2',
    'fc2_output': 'FC2侧向输出 / vanilla FC2',
    'fc2_local': 'FC2侧向输出 / 本条件FC2前馈输入',
}
CATEGORIES = ('animals', 'clothing', 'food', 'plants', 'vehicles')
ABSOLUTE = (
    'activation', 'raw_mean', 'within', 'between', 'fisher', 'raw_within',
    'raw_between', 'raw_fisher', 'nonzero', 'zero_fraction', 'effective_dimension',
    'support_ratio', 'max_support_ratio', 'density_increase_images',
    'revived_fraction', 'revived_activity_share', 'weak_activity_share',
    'population_activity_fraction', 'largest_unit_activity_share',
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative(value, baseline):
    return 100 * (value / baseline - 1) if abs(baseline) > 1e-12 else None


def aggregate():
    rows, categories = [], []
    unique = {}
    source_hashes = {}
    svm_hashes = {}
    unique_readouts = trained = reused = 0
    max_replay_error = max_score_error = 0.
    protocol = read(SVM_OUT / 'protocol.json')
    assert hashlib.sha256((SVM_OUT / 'source_snapshot.py').read_bytes()).hexdigest() == protocol['source_sha256']
    assert hashlib.sha256((ROOT / 'predify_experiments/fc_serial_svm_completion_20261028.py').read_bytes()).hexdigest() == protocol['source_sha256']
    validate_original(protocol)
    previous = read(DOC / 'layer_summary.json') if (DOC / 'layer_summary.json').exists() else None
    for dataset in ('imagenet', 'things'):
        for fold in range(5):
            path = OUT / dataset / f'fold{fold}' / 'results.json'
            complete = read(path.parent / 'complete.json')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == complete['files']['results.json']
            source_hashes[str(path.relative_to(ROOT))] = digest
            data = read(path)
            assert not set(data['train_rows']) & set(data['test_rows'])
            svm_dir = SVM_OUT / dataset / f'fold{fold}'
            svm_complete = read(svm_dir / 'complete.json')
            assert svm_complete['protocol_sha256'] == hashlib.sha256((SVM_OUT / 'protocol.json').read_bytes()).hexdigest()
            for p, h in svm_complete['files'].items():
                assert hashlib.sha256((svm_dir / p).read_bytes()).hexdigest() == h
            svm_hashes[f'{dataset}/{fold}'] = hashlib.sha256((svm_dir / 'results.json').read_bytes()).hexdigest()
            supplement = read(svm_dir / 'results.json')
            assert supplement['train_rows'] == data['train_rows'] and supplement['test_rows'] == data['test_rows']
            assert set(supplement['results']) == set(data['results'])
            unique_readouts += supplement['unique_readouts']
            trained += supplement['trained']; reused += supplement['reused']
            max_replay_error = max(max_replay_error, supplement['replay_max_error'])
            max_score_error = max(max_score_error, supplement['existing_svm_score_max_error'])
            # Independently recover BA and per-class recall from saved scores.
            with np.load(svm_dir / 'scores.npz') as saved:
                assert saved['row'].tolist() == data['test_rows']
                checked_auc = {}
                for name, entry in data['results'].items():
                    assert supplement['feature_keys'][name] == feature_keys(entry['condition'])
                    for stage, key in supplement['feature_keys'][name].items():
                        decision = saved[key]
                        assert decision.shape == (len(saved['y']), 5) and np.isfinite(decision).all()
                        predictions = decision.argmax(1)
                        recall = [float((predictions[saved['y'] == j] == j).mean()) for j in range(5)]
                        metric = supplement['results'][name][stage]
                        assert np.allclose(recall, metric['recalls'], atol=1e-12)
                        assert np.isclose(np.mean(recall), metric['accuracy'], atol=1e-12)
                        if key not in checked_auc:
                            checked_auc[key] = np.mean([roc_auc_score(saved['y'] == j, decision[:, j]) for j in range(5)])
                        assert np.isclose(checked_auc[key], metric['auc'], atol=1e-12)
            base1, base2 = data['baseline_fc1'], data['baseline_fc2']
            for name, entry in data['results'].items():
                c, final = entry['condition'], entry['metrics']
                stage_pairs = (
                    ('fc1_output', final['fc1'], base1),
                    ('fc2_input', final['propagated_fc2'], base2),
                    ('fc2_output', final, base2),
                    ('fc2_local', final, final['propagated_fc2']),
                )
                for stage, m, ref in stage_pairs:
                    row = dict(dataset=dataset, fold=fold, **c, stage=stage,
                               reference=LABELS[stage])
                    row.update({k: m[k] for k in ABSOLUTE})
                    for field in ('activation', 'within', 'between', 'fisher'):
                        row[field + '_pct'] = relative(m[field], ref[field])
                    row['ridge_accuracy'] = m['accuracy']
                    row['ridge_accuracy_pp'] = 100 * (m['accuracy'] - ref['accuracy'])
                    row['ridge_auc'] = m['auc']
                    row['ridge_auc_delta'] = m['auc'] - ref['auc']
                    for key in ('accuracy', 'auc'):
                        row['frozen_svm_' + key] = m['frozen_svm'][key]
                    row['frozen_svm_accuracy_pp'] = 100 * (m['frozen_svm']['accuracy'] - ref['frozen_svm']['accuracy'])
                    row['frozen_svm_auc_delta'] = m['frozen_svm']['auc'] - ref['frozen_svm']['auc']
                    svm_stage = 'fc2_output' if stage == 'fc2_local' else stage
                    svm = supplement['results'][name][svm_stage]
                    svm_ref = supplement['results'][name]['fc2_input'] if stage == 'fc2_local' else supplement['results']['baseline'][svm_stage]
                    for key in ('accuracy', 'auc', 'macro_f1'):
                        row['refit_svm_' + key] = svm[key]
                    row['refit_svm_accuracy_pp'] = 100 * (svm['accuracy'] - svm_ref['accuracy'])
                    row['refit_svm_auc_delta'] = svm['auc'] - svm_ref['auc']
                    if stage == 'fc2_output' and 'refit_svm' in m:
                        assert np.isclose(svm['accuracy'], m['refit_svm']['accuracy'], atol=1e-12)
                        assert np.isclose(svm['auc'], m['refit_svm']['auc'], atol=1e-12)
                    support = final['local_fc2'] if stage == 'fc2_local' else m
                    for key in ('support_ratio', 'max_support_ratio', 'density_increase_images', 'revived_fraction', 'revived_activity_share', 'weak_activity_share'):
                        row[key] = support[key]
                    row['joint'] = bool(row['activation_pct'] < -.001 and row['within_pct'] < -.001 and row['fisher_pct'] > .001)
                    # Check explicitly that no final-FC2 effect was mislabeled as FC1.
                    stored = final['local_fc2'] if stage == 'fc2_local' else m
                    for field in ('activation_pct', 'within_pct', 'between_pct', 'fisher_pct'):
                        assert np.isclose(row[field], stored['effects'][field], atol=1e-9)
                    if stage in ('fc1_output', 'fc2_input'):
                        key = (dataset, fold, stage, c['m1'], c['t1'])
                        fingerprint = tuple(row[k] for k in ABSOLUTE) + (row['ridge_accuracy'], row['ridge_auc'], row['refit_svm_accuracy'], row['refit_svm_auc'])
                        if key in unique:
                            assert unique[key] == fingerprint
                        unique[key] = fingerprint
                    if c['m1'] == 'off' and stage == 'fc1_output':
                        assert max(abs(row[k]) for k in ('activation_pct', 'within_pct', 'fisher_pct', 'ridge_accuracy_pp', 'refit_svm_accuracy_pp')) < 1e-8
                    rows.append(row)
                    for j, category in enumerate(CATEGORIES):
                        categories.append(dict(dataset=dataset, fold=fold, name=name,
                            stage=stage, reference=LABELS[stage], category=category,
                            activation=m['per_class_activation'][j],
                            activation_pct=relative(m['per_class_activation'][j], ref['per_class_activation'][j]),
                            within=m['per_class_within'][j],
                            within_pct=relative(m['per_class_within'][j], ref['per_class_within'][j]),
                            ridge_recall=m['recalls'][j],
                            ridge_recall_pp=100 * (m['recalls'][j] - ref['recalls'][j]),
                            svm_recall=svm['recalls'][j],
                            svm_recall_pp=100 * (svm['recalls'][j] - svm_ref['recalls'][j])))
    summaries = []
    for dataset in ('imagenet', 'things'):
        names = list(dict.fromkeys(r['name'] for r in rows if r['dataset'] == dataset))
        for name in names:
            for stage in STAGES:
                members = [r for r in rows if (r['dataset'], r['name'], r['stage']) == (dataset, name, stage)]
                assert len(members) == 5
                first = members[0]
                summary = {k: first[k] for k in ('dataset', 'name', 'stage', 'reference', 'm1', 't1', 'm2', 't2', 'calibration', 'group', 'id')}
                for key in first:
                    if key in summary or key in ('fold', 'joint'):
                        continue
                    values = [r[key] for r in members]
                    summary[key] = float(np.mean(values)) if all(v is not None for v in values) else None
                summary['joint_folds'] = sum(r['joint'] for r in members)
                summaries.append(summary)
    assert len(rows) == 2280 and len(summaries) == 456 and len(unique) == 200
    assert all(r['refit_svm_accuracy'] is not None for r in summaries)
    if previous is not None:
        old = {(r['dataset'], r['name'], r['stage']): r for r in previous}
        for r in summaries:
            prior = old[(r['dataset'], r['name'], r['stage'])]
            for field in (*ABSOLUTE, 'activation_pct', 'within_pct', 'between_pct', 'fisher_pct', 'ridge_accuracy', 'ridge_auc'):
                assert r[field] == prior[field], ('Sharpening/ridge changed', field)
    save_csv(DOC / 'layer_fold_metrics.csv', rows)
    save_csv(DOC / 'layer_summary.csv', summaries)
    save_csv(DOC / 'layer_category_metrics.csv', categories)
    (DOC / 'layer_summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    audit = dict(passed=True, source_results_hashes=source_hashes,
                 fold_rows=len(rows), summary_rows=len(summaries),
                 unique_fc1_and_bridge_stages=len(unique),
                 reference_effects_checked=True, fc1_unchanged_across_fc2_settings=True,
                 readout_common_method='train-only StandardScaler + LinearSVC C1, preserved original amplitudes',
                 secondary_readout='ridge retained unchanged',
                 svm_result_hashes=svm_hashes, unique_svm_readouts=unique_readouts,
                 svm_new_fits=trained, svm_reused_fits=reused,
                 svm_representation_replay_max_error=max_replay_error,
                 existing_svm_score_max_error=max_score_error,
                 svm_score_BA_recalls_independently_checked=True,
                 svm_score_auc_independently_checked=True,
                 sharpening_and_ridge_unchanged=True,
                 duplicate_condition_rows_are_not_independent_samples=True,
                 original_model_modified=False,
                 backbone_reextraction_from_images=False,
                 additional_readout_fits=True)
    (DOC / 'layer_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return summaries


def report(summary):
    audit = read(DOC / 'layer_audit.json')
    def get(dataset, name, stage):
        return next(r for r in summary if (r['dataset'], r['name'], r['stage']) == (dataset, name, stage))
    def table(dataset, names, stages):
        result = ['| 条件 | 统计位置 / 比较基线 | 均值变化% | W变化% | B变化% | Fisher变化% | 非零单元% | SVM BA% | SVM BA变化pp | SVM AUC变化 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name in names:
            for stage in stages:
                r = get(dataset, name, stage)
                result.append(f"| {name} | {LABELS[stage]} | {r['activation_pct']:+.2f} | {r['within_pct']:+.2f} | {r['between_pct']:+.2f} | {r['fisher_pct']:+.2f} | {100*r['nonzero']:.2f} | {100*r['refit_svm_accuracy']:.2f} | {r['refit_svm_accuracy_pp']:+.2f} | {r['refit_svm_auc_delta']:+.5f} |")
        return result
    lines = ['# 串联实验：FC1与FC2分层统计', '',
        '原来的主要结果表报告的是最终FC2，不能作为FC1本层结果。本报告从已保存的逐折指标独立汇总FC1、FC2前馈输入、FC2侧向输出，并另外给出FC2侧向处理的局部效果。SVM补评估按原参数重建训练表征，并逐项核对测试输出与旧缓存一致；原模型与原始实验输出保持不变。', '',
        'FC1侧向输出相对vanilla FC1；FC2前馈输入和最终输出分别相对vanilla FC2；FC2局部效果相对本条件中已经过FC1处理的前馈输入。局部效果与总效果的百分比不能直接相加。', '',
        'W为逐图L2归一化表征的类内离散度，B为类中心间离散度，Fisher=B/W；均值保留原始激活幅度。两层以完全相同设置的SVM作为主要readout：StandardScaler + LinearSVC(C=1, dual=auto, max_iter=30000, random_state=20261028)，标准化和分类器均仅拟合训练折，使用与此前FC2一致的类别/概念平衡样本权重，保留原始幅度。BA为平衡准确率，pp为百分点。所有57条件的FC1、FC2前馈输入、最终FC2均已覆盖。ridge与冻结vanilla SVM保留在CSV中作辅助诊断。', '',
        '在每层5步hard-hard、matched设置下，FC1本层平均激活分别下降32.15%/32.12%，Fisher提高1.43%/1.51%；最终FC2平均激活下降49.43%/49.39%，Fisher提高12.27%/17.77%（ImageNet/THINGS）。相对FC2实际输入，FC2侧向处理本身另外降低26.71%/26.67%的平均激活，提高10.97%/16.26%的Fisher。这些结果区分了FC1本层sharpening、前馈传递和FC2本层sharpening，不能将最终FC2效果归给FC1。', '',
        '## 两层各自的vanilla SVM基线', '',
        '| 数据集 | 层 | SVM平衡准确率% | SVM宏平均AUC |',
        '|---|---|---:|---:|']
    for d in ('imagenet', 'things'):
        for stage, label in (('fc1_output', 'FC1'), ('fc2_output', 'FC2')):
            r = get(d, 'baseline', stage)
            lines.append(f"| {d} | {label} | {100*r['refit_svm_accuracy']:.2f} | {r['refit_svm_auc']:.5f} |")
    lines += ['', '## 每层5步hard-hard串联', '',
        'frozen沿用vanilla FC2训练统计量与尺度；matched在FC1处理后的真实训练输入上重估FC2连接与尺度。FC1在这两种设置中完全相同。', '']
    for d in ('imagenet', 'things'):
        lines += [f'### {d}', '']
        lines += table(d, ['both_hard_hard_5_5_matched'], STAGES)
        lines += ['', 'FC2沿用原统计量时：', '']
        lines += table(d, ['both_hard_hard_5_5_frozen'], ['fc2_output', 'fc2_local'])
        lines += ['']
    lines += ['## 单层与对照也分别统计', '',
        '以下均为5步。仅处理FC2时，FC1本层应为零变化；仅处理FC1时，最终FC2的变化是前馈传递效果，不是FC2本层Hebbian效果。', '']
    controls = ['fc1_hard_5', 'fc2_hard_5', 'inhibition_only_hard_5', 'gain1_hard_5_matched', 'gain2_hard_5_matched', 'gain_both_hard_5_matched']
    for d in ('imagenet', 'things'):
        lines += [f'### {d}', ''] + table(d, controls, ['fc1_output', 'fc2_output']) + ['']
    lines += ['## 不同步数下的两层结果', '',
        'matched、hard-hard，每层同为1/5/20步。完整CSV同时包含非对称步数和soft/hard组合。', '']
    for d in ('imagenet', 'things'):
        lines += [f'### {d}', ''] + table(d, [f'both_hard_hard_{t}_{t}_matched' for t in (1, 5, 20)], ['fc1_output', 'fc2_output']) + ['']
    lines += ['## 文件与检查', '',
        f"共覆盖{audit['unique_svm_readouts']}个唯一的逐折SVM readout，其中新增拟合{audit['svm_new_fits']}个、复用相同特征上的既有模型{audit['svm_reused_fits']}个。测试表征重放最大误差为{audit['svm_representation_replay_max_error']:.3g}，既有SVM分数最大差异为{audit['existing_svm_score_max_error']:.3g}。已从保存分数独立复算BA、AUC和各类召回率。", '',
        'layer_summary.csv/json：两数据集×57条件×4个统计口径，共456行；layer_fold_metrics.csv：2280行逐折结果，保留绝对值、相对变化、稀疏性和readout；layer_category_metrics.csv：分别保存每层每个大类的激活、类内离散度和召回率。', '',
        'FC1和FC2前馈输入会在多个FC2条件下重复出现。这些是同一中间状态的条件映射，不是额外样本；汇总仅取每个条件的五个折，检查了200个唯一的FC1/前馈状态。补评估重建了对应训练表征，并逐项核对测试表征与旧缓存；旧SVM模型及分数保持一致，所有sharpening与ridge统计保持不变。原有主干、侧向参数和测试划分均未改变；补评估不根据SVM表现重选条件。', '',
        'SVM模型、逐图决策分数、特征复用映射、训练/测试索引及源码协议保存在Hebbian-CNN/Log/fc_serial_20261028/svm_layer_completion_20261028；layer_audit.json记录覆盖范围、哈希、分数重算和表征重放检查。沿用原五折概念隔离与探索性范围，不作显著性声明。', '']
    (DOC / '分层统计.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    result = aggregate()
    report(result)
    for r in result:
        if r['name'] == 'both_hard_hard_5_5_matched':
            print(r['dataset'], r['stage'], {k: r[k] for k in ('activation', 'activation_pct', 'within_pct', 'between_pct', 'fisher_pct', 'refit_svm_accuracy', 'refit_svm_accuracy_pp', 'refit_svm_auc_delta')})
    print('Layer audit passed; 2280 fold rows, 456 summaries, 200 unique FC1/bridge states')
