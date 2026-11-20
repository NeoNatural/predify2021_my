"""Write a Chinese report and static scientific plots from saved t9 results."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'docs/tripartite_t9_20261027'
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
datasets=['imagenet','things'];names=['ImageNet','THINGS']
carriers=[('none','No memory'),('hebb_exact','Hebbian'),('pcoder_exact','PCoder'),('joint_exact','Joint')]
data={d:read(DEST/d/'t9_timeline.csv') for d in datasets}
def curve(d,c):return sorted([r for r in data[d] if r['condition']==c],key=lambda r:int(r['timestep']))
colors=['#111111','#66CBEC','#278EFF','#D77835']
fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
for d,name,ax in zip(datasets,names,axes):
    for (c,label),color in zip(carriers,colors):ax.plot(range(15),[float(r['accuracy_pct']) for r in curve(d,c)],label=label,color=color,marker='o',ms=3)
    ax.set(title=name,xlabel='Target timestep',ylim=(0,100),xticks=range(0,15,2));ax.grid(alpha=.18);ax.legend(frameon=False,ncol=2)
axes[0].set_ylabel('Accuracy (%)');fig.suptitle('This round: t9-trained frozen readout (post-hoc)');fig.tight_layout();fig.savefig(DEST/'t9_timecourses.png',dpi=220);fig.savefig(DEST/'t9_timecourses.svg');plt.close(fig)
lines=['# 本轮三等分布局：t9 固定读出补充结果','','2026-10-27。本分析使用本轮保存的 FC2 特征；未重跑网络、刺激序列或 MVPA。确认集此前已经查看，因此为事后补充分析，不是新的独立确认。','','## 核心结果','','本轮 t9 读出没有确认稳定的联合记忆优势。16 项载体差异检验（两数据集×四对比×t9/全程两个端点）的 Holm p 均为 1.0000。未通过检验不能解释为条件等价。','','联合−Hebbian-only 的全程差值：ImageNet −2.85 pp，THINGS +0.07 pp；原跨时间共同读出均为 −7.01 pp 且通过其原预定校正。故原结论应限定于原共同读出，不能直接推广到任意固定分类器。原预定结果仍保留，新增分析不替换它。','','## 训练与平衡规则','','使用原 train 集80图、40概念，仅读取 t9；四类动物、服饰、食物、车辆。训练一个 StandardScaler+LinearSVC(C=1)，冻结后评估同一确认集64图、32概念、3噪声、28条件和15时刻。训练与确认目标概念互斥。未用开发或确认成绩选择参数。','','每图每噪声纳入全部5种内容。无记忆权重15；每种PCoder内容权重3；每种Hebbian/联合的内容×强度权重1.5。6,240个不同训练行、权重总量14,400；每图180，四载体各45。每图×条件权重与原跨时间轮转采样逐项相同，StandardScaler及SVM均使用权重。这样保持原损失总量，避免因训练行数下降而改变有效正则化规模；时间状态的多样性仍不同。','','两分类器均在50,000次迭代上限内收敛：ImageNet 13次，THINGS 11次。','','## t9 单点与全程均值','','| 数据集 | 条件 | t9准确率% | t0–t14均值% |','|---|---|---:|---:|']
for d,name in zip(datasets,names):
    for c,label in carriers:
        r=curve(d,c);lines.append(f"| {name} | {label} | {float(r[9]['accuracy_pct']):.2f} | {sum(float(v['accuracy_pct']) for v in r)/15:.2f} |")
lines+=['','![t9时间曲线](t9_timecourses.png)','','## 事后补充检验','','以目标概念为簇，保留两图、三噪声及固定窗口内时间点；类别内bootstrap 10,000次、双侧概念符号置换20,000次。下列16项共同进行Holm校正。区间为边际95%区间，非同时区间。','','| 数据集 | 端点 | 对比 | 差值pp | 95% CI | Holm p |','|---|---|---|---:|---|---:|']
for r in read(DEST/'exploratory_tests_holm.csv'):
    lines.append(f"| {r['dataset']} | {r['endpoint']} | {r['contrast']} | {float(r['delta_pp']):+.2f} | [{float(r['low_pp']):.2f}, {float(r['high_pp']):.2f}] | {float(r['holm_p']):.4f} |")
inter=read(DEST/'interaction_holm.csv')
lines+=['','交互（联合−PCoder−Hebbian+无记忆）按两数据集×两强度×两端点构成另一个8项探索性检验家族；THINGS弱强度全程正交互+0.94 pp，95% CI [0.45,1.46]，Holm p=0.0264；强强度全程+2.29 pp，p=0.0616，其余未通过校正。正交互不等于联合优于两个单独载体，不能直接宣称稳定协同，详见 interaction_holm.csv。','','## 数值与记录完整性','','保存的特征为float16。首次补充推断直接送入标准化器导致溢出，尚未保存补充成绩；随后按原共同读出的数值规则先转float32再推断。未截断特征、改动标准化参数或重新选择模型。原分析脚本和已训练头保留；转换包装器及原因记录在 execution_precision.json。','','原 head.pckl、freeze.json 和 manifest.json 的哈希在补充前后相同。新增头和预测独立保存在 Hebbian-CNN/Log/tripartite_t9_20261027；原冻结研究不修改。','','## 结论与后续优先级','','1. t9固定读出与跨时间共同读出共同作为后续主要分析；报告同一时刻对比及完整曲线，避免挑峰。','2. 原共同读出下联合低于Hebbian-only的发现具有读出依赖性；当前t9补充不确认该差异，也不确认联合优势。','3. MVPA保留为次级诊断，暂不优先投入扩展MVPA或复杂置换。曲线波动本身不证明方法失效；当前尚未量化完整重新拟合的不确定性。','4. 优先在开发数据拆分Hebbian更新强度、抑制以及PCoder状态延续的贡献。新机制需冻结方案并使用新确认数据。','','## 运行与文件','','运行：`python analysis/run_t9_tripartite_float32.py`。重复运行复用已冻结补充头并核对方案，不修改原实验。','每数据集保存 t9_timeline / categories / confusion / noise / peaks CSV。全局 readout_comparison.csv 同时列出两固定读出在t9和全程的准确率；exploratory_tests_holm.csv及interaction_holm.csv保存检验。']
(DEST/'补充结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(DEST/'后续分析优先级.md').write_text('# 当前后续分析优先级\n\n2026-10-27，依据用户最新指示，覆盖此前MVPA优先的建议。\n\n优先本轮t9固定读出与跨时间共同冻结读出；MVPA仅保留为补充诊断，暂不优先扩展。完整保留原预定研究与事后补充分析的区别。研究继续按目标时间点和记忆机制推进，不以计算效率作为门槛。\n',encoding='utf-8')
print('Report and figures written; interaction minimum adjusted p:',min(float(r['holm_p']) for r in inter))

