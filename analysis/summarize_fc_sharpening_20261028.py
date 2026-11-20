"""Synthesize existing FC studies; no fitting, dynamics, or model mutation."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/fc_serial_20261028'
SOURCES = [ROOT / 'docs/fc_finite_steps_20261028' / name for name in ('grid_summary.csv', 'pair_summary.csv')]
SOURCES += [ROOT / 'docs/fc_lateral_study_20261028/summary.json', DOC / 'layer_summary.json', DOC / 'layer_audit.json']


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    grid, pairs = (load_csv(p) for p in SOURCES[:2])
    dense = json.loads(SOURCES[2].read_text(encoding='utf-8'))
    layers = json.loads(SOURCES[3].read_text(encoding='utf-8'))
    assert json.loads(SOURCES[4].read_text(encoding='utf-8'))['passed']
    # Grid CSV retains ridge-only fields; import SVM from identical pair rows.
    keys = ('dataset', 'layer', 'rule', 'graph', 'mode', 'gain', 'budget', 'step')
    def identity(r):
        return tuple(float(r[k]) if k in ('gain', 'budget', 'step') else r[k] for k in keys)
    paired = {identity(r): r for r in pairs}
    for r in grid:
        if identity(r) in paired:
            p = paired[identity(r)]
            for field in ('activation_pct', 'within_pct', 'between_pct', 'fisher_pct'):
                assert abs(float(r[field]) - float(p[field])) < 1e-9
            for field in ('refit_svm_accuracy_pp', 'refit_svm_auc_delta'):
                r[field] = p[field]
    for r in dense:
        if 'svm_accuracy_pp' in r:
            r['refit_svm_accuracy_pp'] = r['svm_accuracy_pp']
    selected = []

    def find_grid(d, l, rule='corr', graph='k128', mode='hard', gain=.75, budget=1., step=5):
        found = [r for r in grid if r['dataset'] == d and r['layer'] == l and r['rule'] == rule and r['graph'] == graph
                 and r['mode'] == mode and float(r['gain']) == gain and float(r['budget']) == budget and int(r['step']) == step]
        assert len(found) == 1
        return found[0]

    def find_layer(d, name, stage='fc2_output'):
        found = [r for r in layers if (r['dataset'], r['name'], r['stage']) == (d, name, stage)]
        assert len(found) == 1
        return found[0]

    def table(rows, source):
        result = ['| 数据集/对象 | 条件 | 均值变化% | W变化% | B变化% | Fisher变化% | 非零% | 重拟合SVM BA变化pp |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for obj, name, r in rows:
            values = {k: float(r[k]) for k in ('activation_pct', 'within_pct', 'between_pct', 'fisher_pct', 'nonzero')}
            svm = r.get('refit_svm_accuracy_pp')
            display = '未评估' if svm in (None, '') else f'{float(svm):+.2f}'
            result.append(f"| {obj} | {name} | {values['activation_pct']:+.2f} | {values['within_pct']:+.2f} | {values['between_pct']:+.2f} | {values['fisher_pct']:+.2f} | {100*values['nonzero']:.2f} | {display} |")
            selected.append(dict(source=source, object=obj, condition=name, **values,
                                 refit_svm_accuracy_pp=None if svm in (None, '') else float(svm)))
        return result

    lines = ['# 现有FC Hebbian实验的sharpening综合比较', '',
        '本次只综合已有数据，不重跑、不拟合、不调参，不修改原模型。主目标为平均激活下降、逐图L2归一化后的类内W下降与类间/类内Fisher比上升，B单列。SVM仅作功能检查；表中的未评估不代表通过或失败。', '',
        '优先的连接/抑制家族是中心化、带方差正则的正相关连接，配合互为top-k稀疏图、除法归一化及自适应硬阈值。没有一个参数在均值与区分度上处处占优：soft更压低均值，hard通常更有利于保留强响应的区分结构。FC1与FC2需各自判断，串联的最终FC2结果不能代表FC1本层。', '',
        '连接由无标签训练激活估计：正协方差除以带方差下限的两端标准差，负值置零，去掉自连接，保留互为top-k的连接，再做全局最大行和归一化。它是正则化相关规则，并非不加约束的原始共激活。硬阈值将弱驱动置零、保留超过阈值的驱动；除法另调节整体兴奋水平。', '',
        '## 连接规则：同参数正相关与正协方差', '',
        '以下均为5步、gain=0.75、活动预算1.0；FC1用k128，FC2用k32。正相关的W与Fisher改善更强，但正协方差常压低更多均值，故不是所有指标上的严格支配。原始共激活只在较早的dense实验比较过，不能当作已经完成相同稀疏硬阈值全网格验证。', '']
    rows = []
    for d in ('imagenet', 'things'):
        for l, graph in (('fc1', 'k128'), ('fc2', 'k32')):
            for rule in ('cov', 'corr'):
                rows.append((f'{d}/{l}', f'{rule}/{graph}/hard/gain0.75', find_grid(d, l, rule=rule, graph=graph)))
    lines += table(rows, 'finite_grid_plus_identical_pair_svm') + ['', '## FC1：历史配对与更强的现有探索候选', '',
        '历史串联沿用FC1 k128/gain0.25。它的硬阈值版本在两数据集五折均联合改善，但Fisher只提高约1.5%。完整有限步网格中的gain0.75使FC1区分度改善约5%，代价是均值少下降约2个百分点；k32/k128没有统一赢家。它是事后探索线索，尚无此增益的重新拟合SVM与真正串联结果，不能将其独立改善加到既有串联效果。', '']
    rows = []
    for d in ('imagenet', 'things'):
        for graph, mode, gain in [('k128', 'soft', .25), ('k128', 'hard', .25), ('k128', 'hard', .75), ('k32', 'hard', .75)]:
            r = find_grid(d, 'fc1', graph=graph, mode=mode, gain=gain)
            if gain == .25:
                r = next(p for p in pairs if p['dataset'] == d and p['layer'] == 'fc1' and p['step'] == '5' and p['role'] == ('prior_soft' if mode == 'soft' else 'matched_hard'))
            rows.append((f'{d}/fc1', f'{graph}/{mode}/gain{gain}', r))
    lines += table(rows, 'finite_grid_and_pairs') + ['',
        '更强FC1 k128/gain0.75的冻结原始SVM BA变化为ImageNet +0.84、THINGS +0.40个百分点；这与重新拟合SVM是不同功能检查，不混为一谈。将其活动预算强制收缩到75%时，两数据集W反而上升约1.23%/1.38%，Fisher下降约0.48%/0.53%。更少活动单元并不自动意味着更好的sharpening。', '',
        '## FC2：硬阈值的区分度收益更明确', '',
        '5步corr/k32/gain0.75/budget1：hard比soft少压低约7个百分点的均值，但得到更强的W下降和Fisher上升。k32在该参数的hard比较中，均值和Fisher在两套数据均优于k128和dense图。图剪枝后会重新归一化权重，因此这是完整连接方案的比较，不能把改善全归因于连接数量。', '']
    rows = []
    for d in ('imagenet', 'things'):
        for mode in ('soft', 'hard'):
            r = next(p for p in pairs if p['dataset'] == d and p['layer'] == 'fc2' and p['step'] == '5' and p['role'] == ('prior_soft' if mode == 'soft' else 'matched_hard'))
            rows.append((f'{d}/fc2', f'corr/k32/{mode}/gain0.75', r))
    lines += table(rows, 'finite_pairs') + ['', '## 仅抑制、缩放和dense对照', '',
        '单层去Hebbian但保留hard门控与除法，FC1均值下降约33%、FC2约31%，W/B/Fisher基本不变。原图活动预算为100%时，这个对照近似逐图幅度调节；它不是所有可能抑制模型的代表。纯正标量缩放改变均值而不改变归一化方向，固定乘0.5即均值−50%、Fisher不变。因此压低均值这一项并不需要Hebbian，Hebbian的额外价值应体现在方向与区分结构。', '']
    rows = []
    for d in ('imagenet', 'things'):
        for l in ('fc1', 'fc2'):
            p = next(p for p in pairs if p['dataset'] == d and p['layer'] == l and p['step'] == '5' and p['role'] == 'paired_nohebb_hard')
            rows.append((f'{d}/{l}', '无Hebbian/hard/除法/5步', p))
            r = next(r for r in dense if r['dataset'] == d and r['layer'] == l and r['family'] == 'selected')
            rows.append((f'{d}/{l}', '早期dense corr+仅除法（原步数）', r))
    lines += table(rows, 'finite_pairs_and_early_dense') + ['',
        '早期dense相关连接＋除法会显著降低W，但输出几乎100%非零；FC1的B还收缩约4.4%。稀疏连接但没有活动门控仍可能近乎全激活。故在避免大量弱响应铺开的目标下，不把仅除法的dense方案作为首选。固定阈值或减法抑制在早期所测范围内也没有提供更好的联合改善。', '',
        '## 真正串联：旧FC1 gain0.25 + FC2 gain0.75', '',
        '下表均为每层5步，数值针对最终FC2、相对vanilla FC2。matched为按各条件实际训练输入重估FC2连接和尺度，frozen为沿用vanilla FC2连接与尺度。hard-hard在区分度上最有吸引力；soft-soft进一步降低均值，但FC1本层W变差、Fisher略降，因此不能由最终FC2改善推断两层均改善。', '']
    names = ['both_soft_soft_5_5_matched', 'both_soft_hard_5_5_matched', 'both_hard_soft_5_5_matched',
             'both_hard_hard_5_5_matched', 'both_hard_hard_5_5_frozen', 'inhibition_only_hard_5',
             'gain1_hard_5_matched', 'gain2_hard_5_matched', 'gain_both_hard_5_matched', 'gain1_hard_5_frozen']
    rows = [(d + '/最终fc2', name, find_layer(d, name)) for d in ('imagenet', 'things') for name in names]
    lines += table(rows, 'serial_layer_summary') + ['',
        '仅双层抑制均值下降约52%–53%，但Fisher只提高约0.57%/0.64%；两层均换成均值匹配缩放，最终均值与真实hard-hard一致，Fisher只提高0.54%/0.61%。这些小幅几何变化可由原始前馈bias/ReLU引入，不能误称为单层纯缩放本身改变了方向。', '',
        '将FC1替换成均值匹配缩放、FC2仍做Hebbian，matched的Fisher仍提高11.68%/16.85%，接近真实串联的12.27%/17.77%；frozen下也为12.01%/16.50%，接近真实frozen的12.83%/17.44%。将FC2换成均值匹配缩放则只剩1.17%/1.30%。这支持当前串联配置中FC2侧向处理承担更明显的区分作用，但不能把两个局部百分比相加或据此分配因果贡献比例。', '',
        'matched的FC2尺度重估使均值从约−44.5%降至约−49.4%，Fisher在ImageNet略低、THINGS略高，因此不称其在全部指标上优于frozen。它同时改变统计量与尺度，不能只归因于连接重学。', '',
        '串联FC1本层保持−32.15%/−32.12%均值与+1.43%/+1.51% Fisher；FC2相对其实际输入的局部效果为均值−26.71%/−26.67%、Fisher +10.97%/+16.26%。FC2侧向门控不增加相对实际输入的活动数，但原前馈桥接使最终活动数比vanilla略高约0.5%/0.8%。', '',
        '## SVM功能检查与评价边界', '',
        '旧参数hard-hard/matched/5步：FC1重新拟合SVM BA变化为+0.02/0.00个百分点，最终FC2为−0.04/−0.10个百分点，AUC均小幅上升（ImageNet/THINGS）。独立FC2 hard5的BA变化为−0.09/−0.20个百分点。这些数值支持大类可读性大体保持，不作为sharpening的排名依据，也不是统计非劣性证明。', '',
        '每层从5步加到20步，旧参数串联matched Fisher由12.27%/17.77%增至12.86%/18.93%，均值仍约−50%，增量较小。5步是统一比较截面，不宣称测试集上确认的最佳步数，不按是否达到不动点排除配置。', '',
        '当前判断：FC2的corr/k32＋hard＋除法最有现成支持；FC1应区分已完成完整功能/串联检查的gain0.25版本，及独立网格中区分度更强的gain0.75版本。现有串联优先hard-hard；FC1缩放＋FC2 Hebbian是已有对照中接近其效果的简化通路。更强FC1尚未串联，不能承诺会进一步改善。', '',
        '同一批样本已多轮复用，全网格较优结果是探索性发现，并非新增独立确认。对均值和区分度没有指定统一权重，因此报告折中而不声称存在唯一全局最优。', '']
    (DOC / '综合比较.md').write_text('\n'.join(lines), encoding='utf-8')
    with (DOC / 'synthesis_comparisons.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)
    audit = dict(analysis_only=True, new_experiments=False, model_modified=False,
                 primary_goal='activation down; normalized within down; Fisher up',
                 svm_role='functional check only; missing refits explicitly marked',
                 comparison_step=5, new_selection_or_validation_claim=False,
                 source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
                 extracted_rows=len(selected))
    (DOC / 'synthesis_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Synthesis saved; extracted rows:', len(selected), '; no model runs or fits')


if __name__ == '__main__':
    main()
