"""Build the final Chinese report strictly from saved study tables."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.tripartite_20261027 import DOC,OUT


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def link(path,label):return f'[{label}](<{path.resolve().as_posix()}>)'
def image(name,label):return f'![{label}](<{(DOC/name).resolve().as_posix()}>)'
def number(value):return f'{float(value):+.2f}'


def main():
    primary=read(DOC/'primary_common_readout_holm.csv');secondary=read(DOC/'secondary_interaction_holm.csv')
    lines=['# 横向三等分布局：Hebbian 与 PCoder 补充实验结果',
        '', '实验标识：2026-10-27。已完成两数据集的28条件、三个噪声实现、t0–t14提取，以及共同冻结读出和逐时刻MVPA。',
        '', '## 先看直接回答', '',
        '本次联合记忆未呈现优势：在两个数据集上，主强度exact条件的全程平均MVPA准确率均低于PCoder-only和Hebbian-only（描述性比较）。共同冻结读出的预定校正检验确认，联合记忆较Hebbian-only均低7.01个百分点；Hebbian-only相对无记忆的差异未通过校正，交互检验也未确认协同效应。', '']
    names={'joint-pcoder':'联合 − PCoder-only','joint-hebb':'联合 − Hebbian-only','pcoder-hebb':'PCoder-only − Hebbian-only','hebb-none':'Hebbian-only − 无记忆'}
    for dataset in ('imagenet','things'):
        lines.append(f'**{dataset.upper()}：共同冻结读出的全时间窗主比较。**')
        lines.append('')
        for r in primary:
            if r['dataset']!=dataset:continue
            passed=float(r['holm_p'])<.05;direction='正向' if float(r['delta_pp'])>0 else '负向'
            conclusion=f'{direction}差异通过预定Holm校正' if passed else '未通过预定Holm校正，不能据此确认差异'
            lines.append(f"- {names[r['contrast']]}：{number(r['delta_pp'])} pp，95% CI [{number(r['low_pp'])}, {number(r['high_pp'])}]，Holm p={float(r['holm_p']):.4f}；{conclusion}。")
        lines.append('')
    lines+=['这些是预定全程平均准确率的条件比较，不是峰值显著性。主要MVPA曲线、三组记忆两两比较及内容控制完整呈现在下方；不能仅凭联合条件最高称为协同。',
        '', '## 设计与独立性',
        '', '两个数据集统一四类：动物、服饰、食物、车辆。使用相同的目标概念配额：各训练80图（40概念）、开发32图（16概念）、确认64图（32概念），每概念2图，四类平衡。新确认图排除全部已知历史target与preview；训练/开发允许已查看图，未把它们重新称为独立确认。当前读出训练、开发和确认的目标概念及相应preview概念池互斥；这不代表VGG预训练从未见过对应语义类别。',
        '', 'THINGS严格WordNet映射，排除明确ImageNet来源及文件字节重复；来源不明仍单独记录。ImageNet确认图与历史已用文件的字节重叠为0，两个数据集本轮target之间也没有字节重叠。结论限于这一严格映射、可用库存约束下的四类样本，不是全ImageNet/全THINGS benchmark。',
        '', '完整224×224画布沿宽度分左中右，物体64×64、中心约37/112/187。两段清晰preview各2步，目标中央、noise .5、15步；先加噪声再缩放，保存实际张量和缩放后噪声统计。四类下基础preview从其余三类各取一张，再平衡重复一个非目标大类，使用不同概念/图片；未重新引入植物。prime slot在第二段preview，类内左右各半。',
        '', '共同灰屏准备时长22步，由五层表征变化<1e-4连续3步的标签无关规则决定。缓存并恢复rep/prd/grd/prediction_error；所有数据集/阶段灰屏状态摘要一致。无carry时从同一灰屏状态开始各段；Hebbian-only保留W。W只在两段preview边界更新，target冻结；PCoder-only与联合延续预览状态。',
        '', '主强度coeff .05、预设弱强度.0125，decay .5、inh 4、稀疏阈值.9，注入PCoder4/5。FC动态更新关闭，ReLU后零W恒等运算旁路计算；主干与PCoder参数未重新训练。16个主条件+2初始化参照+10弱强度条件，共28条件。正式主实验 **29,568个序列**，另有112个旧布局开发参照序列；验收另计。',
        '', '## 时间曲线与两套读出',
        '', '主要MVPA为条件内、逐时间点、目标概念分组5折CV：所有条件与时刻共用折，三噪声随概念分组，StandardScaler只拟合训练折，LinearSVC C=1。每数据集28×15×5=2100次拟合，两数据集4200次。CV的背景preview概念可跨折出现，目标概念不跨折；因此不声称全部背景视觉输入在各折也未见。',
        '', '共同读出使用独立训练集。预设轮转采样涵盖所有时间、载体、内容与两强度，Hebbian两强度各半权重；每图270条状态、有效权重180，四载体总权重相同。每数据集21600行、有效权重14400。StandardScaler和SVM同时使用权重。开发仅作地板/极端抑制检查，不选C、强度、灰屏时长或最优时间。确认前冻结读出与方案。',
        '', 'MVPA描述条件内可提取性，共同边界衡量固定分类器可利用性。两者训练队列和拟合方式也不同，绝对准确率差不单独归因于某一个读出因素。MVPA不同拟合的margin尺度不相同；只有共同边界下的margin差可直接用于同一决策方向解释。主结果未使用t0–t3原始分数平均，也未以计算效率作为继续门槛。',
        '',image('main_timecourses.png','主时间曲线'),
        '', '| 数据集 | 读出 | 载体（exact） | 全程均值% | 全程峰高% | 首次峰时 | t14% |', '|---|---|---|---:|---:|---:|---:|']
    for dataset in ('imagenet','things'):
        for method in ('mvpa','common'):
            rows=read(DOC/dataset/f'{method}_peaks.csv');timeline=read(DOC/dataset/f'{method}_timeline.csv')
            for condition,label in [('none','无记忆/灰屏'),('hebb_exact','Hebbian-only'),('pcoder_exact','PCoder-only'),('joint_exact','联合')]:
                r=next(r for r in rows if r['condition']==condition and r['window']=='full')
                late=next(r for r in timeline if r['condition']==condition and r['timestep']=='14')
                lines.append(f"| {dataset} | {method} | {label} | {float(r['average_accuracy_pct']):.2f} | {float(r['peak_accuracy_pct']):.2f} | {r['first_peak_timestep']} | {float(late['accuracy_pct']):.2f} |")
    lines+=['','### 活动与解码并列查看','',
        '活动比值为全程FC2绝对活动均值相对灰屏无记忆的百分比，是描述性表征指标，不替代解码或主检验。',
        '', '| 数据集 | 条件 | FC2相对活动% | MVPA全程准确率% | 共同边界全程准确率% |', '|---|---|---:|---:|---:|']
    for dataset in ('imagenet','things'):
        common=read(DOC/dataset/'common_timeline.csv');mvpa=read(DOC/dataset/'mvpa_timeline.csv')
        def mean(rows,c,field):
            values=[float(r[field]) for r in rows if r['condition']==c]
            return sum(values)/len(values)
        base=mean(common,'none','mean_abs_fc2_activation')
        for condition,label in [('none','无记忆'),('hebb_exact','Hebbian主强度'),('pcoder_exact','PCoder-only'),('joint_exact','联合主强度'),('hebb_weak_exact','Hebbian弱强度'),('joint_weak_exact','联合弱强度')]:
            activity=100*mean(common,condition,'mean_abs_fc2_activation')/base
            lines.append(f"| {dataset} | {label} | {activity:.2f} | {mean(mvpa,condition,'accuracy_pct'):.2f} | {mean(common,condition,'accuracy_pct'):.2f} |")
    lines+=['', '## 三种记忆全部两两比较', '',image('mvpa_three_pairwise.png','MVPA三组配对差值'),
        '',image('common_three_pairwise.png','共同读出三组配对差值'),
        '', '下表的峰差是各条件分别选峰后的描述值；“A峰时同刻差”是在A峰时刻比较A与B。二者与全程平均差是不同问题，不能混写。未做包含重新拟合和重新选峰的嵌套重采样，故不为峰高/峰时标显著性。',
        '', '| 数据集 | 方法 | 比较 | 全程平均差pp | 各自峰高差pp | 峰时A−B | A峰时同刻差pp |', '|---|---|---|---:|---:|---:|---:|']
    for dataset in ('imagenet','things'):
        for method in ('mvpa','common'):
            rows=read(DOC/dataset/f'{method}_window_contrasts.csv')
            for comparison in ('joint-pcoder','joint-hebb','pcoder-hebb'):
                r=next(r for r in rows if r['strength']=='strong' and r['identity']=='exact' and r['window']=='full' and r['contrast']==comparison)
                lines.append(f"| {dataset} | {method} | {names[comparison]} | {number(r['delta_pp'])} | {number(r['peak_height_delta_pp'])} | {r['peak_time_delta_steps']} | {number(r['same_time_at_a_peak_delta_pp'])} |")
    lines+=['', '## 内容、强度和初始化',
        '', '以下为预定全程平均的exact内容差值，均作描述；载体有增益并不自动代表精确图片记忆有特异优势。完整早/中/晚窗口与时间曲线见原表。',
        '', '| 数据集 | 方法 | 载体 | Exact−同概念 pp | Exact−同大类 pp | Exact−中性 pp |', '|---|---|---|---:|---:|---:|']
    for dataset in ('imagenet','things'):
        for method in ('mvpa','common'):
            rows=read(DOC/dataset/f'{method}_window_contrasts.csv')
            for carrier in ('hebb','pcoder','joint'):
                values=[]
                for control in ('same_concept','same_category','neutral'):
                    r=next(r for r in rows if r['strength']=='strong' and r['identity']=='exact' and r['window']=='full' and r['contrast']==f'{carrier}_exact-minus-{control}')
                    values.append(number(r['delta_pp']))
                lines.append(f"| {dataset} | {method} | {carrier} | {' | '.join(values)} |")
    lines+=['',image('mvpa_content_controls.png','MVPA内容控制'),
        '',image('common_content_controls.png','共同读出内容控制'),'',image('weak_strength_mvpa.png','预设弱强度'),
        '',image('initialization_references.png','初始化参照'),
        '', 'exact、同概念不同图、同大类不同概念、异大类、中性slot全部报告。中性仅移除指定prime，其余三个干扰物仍包含结构化类别信息，不能称为没有类别信息。弱强度为预设敏感性检查，未根据确认成绩替换主强度。随机与前馈初始化是次级参照，主要四载体均以共同灰屏为起点。',
        '', '## 交互、统计和不确定性', '', '| 数据集 | 强度 | 全程交互pp | 概念簇95% CI | 次级Holm p |', '|---|---|---:|---|---:|']
    for r in secondary:
        lines.append(f"| {r['dataset']} | {r['strength']} | {number(r['delta_pp'])} | [{number(r['low_pp'])}, {number(r['high_pp'])}] | {float(r['holm_p']):.4f} |")
    lines+=['', '交互定义为联合−PCoder−Hebbian+无记忆。其预定次级家族为两数据集×两强度4检验，单独Holm控制；需要结合联合对两个单独载体的优势解释，准确率上的交互不自动等于普遍生物机制协同。',
        '', '共同边界主推断以32个目标概念为簇，每簇保留两图片、三个噪声和整个固定窗口。类别内概念bootstrap 10000次，概念簇双侧符号置换20000次；主强度exact全程平均的四个对比×两数据集共8检验Holm控制。边际95% CI未作同时校正，正式判定用Holm p，不用单个CI替代。分噪声差值、全部窗口及弱强度比较保存到CSV；CV折、噪声和15时间点不当作独立重复。',
        '', 'MVPA峰值/时间点为描述性结果，未运行完整分类器重新拟合的标签置换；OOF预测不能据简单独立样本检验标记信息优势显著。主结论限于冻结方案和该确认样本，不把统计未通过写成无效，也不要求先证明计算收益。',
        '', '## 旧布局与验收',
        '', '每数据集使用预定8张开发图、一个噪声、四载体exact/neutral共7条件，分别比较旧34像素/偏移.175和新64像素/偏移1/3。使用同一个新布局读出，故这是布局/读出匹配敏感性诊断；样本小，不能当作纯Hebbian因果归因或旧布局的最佳分类表现。完整曲线和三组两两差值见旧布局CSV。',
        '', '几何、split/字节隔离、灰屏摘要、W提交时机、zero-Hebb退化、no-carry内容不敏感和数值有限性通过验收。实际输入和缩放后噪声已保存；图中RGB为显示而裁剪到[0,1]，模型使用保存的标准化张量。',
        '', '续跑时修复了通用冻结JSON校验的元组/列表规范化问题：规范化到JSON实际存储形式再比较，仍拒绝真实参数变化。未改写已有CV规格文件；两数据集的模型、读出、分析程序和方案摘要均与确认前冻结值一致。前后源码与专项测试见serialization_fix_audit.json。该修复不改变任何模型或统计规则。',
        '',image('imagenet_actual_stimuli.png','ImageNet实际刺激'),'',image('things_actual_stimuli.png','THINGS实际刺激'),
        '', '## 数据、代码与重跑',
        '',link(DOC/'预先方案.md','预先方案')+'；'+link(DOC/'inventory.json','库存与划分')+'；'+link(DOC/'primary_common_readout_holm.csv','主检验8项')+'；'+link(DOC/'secondary_interaction_holm.csv','交互次级检验')+'。']
    for dataset in ('imagenet','things'):
        lines.append('')
        lines.append(dataset.upper()+'：'+'；'.join(link(DOC/dataset/name,label) for name,label in [
            ('mvpa_timeline.csv','MVPA全时间线'),('common_timeline.csv','共同读出全时间线'),('mvpa_window_contrasts.csv','MVPA三组比较与峰值'),
            ('common_window_contrasts.csv','共同边界配对差值'),('mvpa_confusion.csv','类别混淆'),('mvpa_noise.csv','分噪声'),('old_layout_pairwise.csv','旧布局配对参照'),('analysis_audit.json','分析审计')]))
    lines+=['', '原始记录、实际输入、共同灰屏、源码/权重/软件摘要和冻结读出在 '+link(OUT,'实验数据目录')+'。',
        '', '```powershell', '& C:/Users/liang/anaconda3/envs/predify2021/python.exe analysis/run_tripartite_20261027.py', '```',
        '', '该命令保留已经完成的原始提取与已冻结的读出，续跑剩余阶段；不重新挑选方案。若要更换参数、划分或统计方案，应创建新的研究输出目录，保留本次记录。']
    (DOC/'补充实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('REPORT',DOC/'补充实验结果.md',flush=True)


if __name__=='__main__':main()
