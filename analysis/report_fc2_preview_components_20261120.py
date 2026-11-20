"""Plot raw FC2 components and paired early contrasts from cached trials."""
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from fc2_preview_components_20261120 import OUT
from report_familiarity_timecourse_20261119 import NAMES, plt
from report_pcoder_carry_20261119 import table

ARMS = {"fresh_carry": "新批次 · FC2 增益前", "carry": "配对批次 · 保留 carry", "reset": "配对批次 · 重置 carry"}
NAMES_M = {"shared": "类别共有反应", "photo_var": "同概念照片的残差散度",
           "residual_rms": "类别空间外的残差幅度", "shared_alignment": "类别方向余弦",
           "photo_var_unit": "等幅后的照片残差散度"}
H = {"exact": ("同图", "#1969a4", "o", "-"), "same_concept": ("同概念异图", "#247555", "^", ":"),
     "different": ("异类预览", "#bd511d", "s", "--")}
PAIRS = {"exact-different": "同图 − 异类", "same_concept-different": "同概念异图 − 异类",
         "same_concept-exact": "同概念异图 − 同图"}


def curves(data, measures, title, filename):
    rows = [(ds, m) for ds in NAMES for m in measures]
    fig, axes = plt.subplots(len(rows), 3, sharex=True, sharey="row", figsize=(12, 1.75*len(rows)+1.2))
    for i, (ds, m) in enumerate(rows):
        for j, (arm, label) in enumerate(ARMS.items()):
            ax = axes[i, j]
            ax.axvspan(-.2, 4.5, color="#edf2f6", zorder=-5)
            for h, (name, color, marker, ls) in H.items():
                s = data.query("dataset==@ds and metric==@m and arm==@arm and history==@h").sort_values("step")
                np.testing.assert_array_equal(s.step, range(15))
                ax.plot(s.step, s.estimate, marker=marker, mfc="white", ms=3.2, lw=1.6, color=color, ls=ls, label=name)
            ax.set(xlim=(-.2, 14.2), xticks=[0,2,4,6,8,10,12,14])
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis="y", alpha=.22)
            if i == 0:
                ax.set_title(label, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{NAMES[ds]}\n{NAMES_M[m]}", fontsize=9)
            if i == len(rows)-1:
                ax.set_xlabel("目标出现后的递归步")
    fig.suptitle(title, x=.12, ha="left", y=.985, fontsize=15, fontweight="bold")
    fig.legend(*axes[0,0].get_legend_handles_labels(), ncol=3, loc="upper right", bbox_to_anchor=(.986,.996), frameon=False)
    fig.subplots_adjust(left=.126, right=.98, top=.892, bottom=.082, wspace=.12, hspace=.22)
    fig.savefig(OUT/filename, dpi=155, facecolor="white")
    plt.close(fig)


def early_comparison(intervals):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.6), sharex="row")
    for i, m in enumerate(("shared", "photo_var")):
        for j, ds in enumerate(NAMES):
            ax = axes[i,j]
            ax.axvline(0, color="#727f87", lw=.8)
            for row, arm in enumerate(ARMS):
                for h, offset in (("exact",.12), ("same_concept",-.12)):
                    label, color, marker, _ = H[h]
                    r = intervals.loc[(ds,arm,m,f"{h}-different","early")]
                    y = 2-row+offset
                    ax.hlines(y, r.low, r.high, color=color, lw=1.6)
                    ax.plot(r.estimate, y, marker=marker, color=color, ms=5, label=label if row==0 else None)
            ax.set(yticks=range(3), ylim=(-.45,2.45))
            ax.set_yticklabels(list(ARMS.values())[::-1] if j==0 else [])
            ax.set_xlabel(NAMES_M[m]+"的原始差值")
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis="x", alpha=.18)
            if i==0:
                ax.set_title(NAMES[ds], loc="left", fontsize=12, fontweight="bold")
    fig.suptitle("第 0–4 步 · 各预览条件减去异类预览", x=.18, ha="left", fontsize=15, fontweight="bold")
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False)
    fig.subplots_adjust(left=.18, right=.98, top=.84, bottom=.12, wspace=.09, hspace=.47)
    fig.savefig(OUT/"early_paired.png", dpi=150, facecolor="white")
    plt.close(fig)


def build():
    absolute = pd.read_csv(OUT/"absolute.csv")
    intervals = pd.read_csv(OUT/"windows.csv").set_index(["dataset","arm","metric","comparison","window"])
    def panel(arms, window, metrics):
        rows = []
        for ds in NAMES:
            for arm in arms:
                for m in metrics:
                    cells = []
                    for k in PAIRS:
                        r = intervals.loc[(ds,arm,m,k,window)]
                        cells.append(f"{r.estimate:+.5f} ［{r.low:+.5f}，{r.high:+.5f}］")
                    rows.append([NAMES[ds], ARMS.get(arm,"保留减重置"), NAMES_M[m], *cells])
        return table(["数据集","设置","指标",*PAIRS.values()],rows)

    main = ("shared", "photo_var")
    early_comparison(intervals)
    curves(absolute, main, "类别共有反应与照片残差的原始时间过程", "components_raw.png")
    curves(absolute, ("residual_rms","shared_alignment","photo_var_unit"),
           "残差总幅度与等幅后的时间过程", "components_amplitude.png")
    html = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>预览如何改变类别共有反应与照片残差</title><style>*{box-sizing:border-box}body{background:#eef2f4;color:#1e3341;margin:0;font:16px/1.7 "Microsoft YaHei",sans-serif}
main{max-width:1220px;margin:auto;background:white;padding:26px clamp(16px,4vw,56px) 55px}img{width:100%}a,summary{color:#145779}h2{margin-top:28px}
.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:6px 9px;border-bottom:1px solid #dce3e7;text-align:left}
th{background:#edf4f6}details{margin:12px 0}summary{cursor:pointer}</style><main>
<h1>同概念异图使早期的照片残差更集中</h1>
<p>以异类预览为共同参照，在两套数据中，同图与同概念异图都提高了目标出现后第 0–4 步的类别共有反应。
两者的区别更多地体现在同一概念的不同目标照片之间：保留预览后的 PCoder 状态时，同概念异图使这些照片的残差更接近，同图使它们更分散。
重置 PCoder 后，两种预览在原始残差散度上的差距接近零。</p>
<a href="components_raw.png"><img src="components_raw.png" alt="三种预览下类别共有反应与照片残差散度的原始数值"></a>
<p>左列是最近一批留出照片的目标反应，保留 carry，取施加末端 FC2 增益之前的活动。中、右列使用另一批完全配对的照片，两列均使用旧的显式 FC2 增益。
沿同一行比较中、右列可读出重置的影响。浅蓝底标出早期第 0–4 步。每条线是绝对值，各大类等权，大类内各概念等权。照片残差散度的单位为每单元活动的平方。</p>
<h2>怎样把两种成分分开</h2>
<p>每套数据仅使用原训练折缓存的 FC2 反应（同图预览、第 9 步、末端增益前），依次在照片、概念和大类上取均值。
THINGS 用 640 张训练照片、80 个概念估计五个大类中心；ImageNet 用 512 张、64 个概念估计四个大类中心。
训练折和两批测试照片的概念、图像编号和文件散列均无重合。新批次为 THINGS 97 张和 ImageNet 96 张；重置对照分别为 108 张和 96 张。
新批次 THINGS 的植物大类仅一个概念，其泛化到其他植物的变异无法在这批照片中估计。</p>
<p>以训练折各大类中心的平均值 μ 为原点，将每张目标的 4096 维反应写为 z = x − μ。
类别共有反应取 z 在正确大类方向上的有符号投影，再除以 √4096，使其具有单元活动的量纲。正确大类方向由对应训练中心减 μ 后归一化得到。
使用测试目标的大类标签只为读取对应的冻结方向。训练中心张成的类别空间分别为 4 维和 3 维。
去掉 z 在整个类别空间中的投影后得到正交残差 r。它也可能包含细概念、预览背景及递归状态的信号，不能把全部残差归因于照片身份。</p>
<p>为专门估计同一概念内不同照片的变异，在每个条件、检查点和时间步，对同一概念的每对不同目标照片 i、j 计算
‖rᵢ − rⱼ‖² / (2 × 4096)，再取平均。这个量等于该概念内每单元残差的无偏样本方差。
测试图片只用于评估差异，不参与类别方向估计。三种预览和两种 carry 设置使用相同的配对和权重。
作为参照，残差总幅度则直接计算每张图的 ‖r‖ / √4096。</p>
<h2>早期配对结果</h2><p>前两列对比均以异类预览为参照，最后一列直接比较两种重复方式。先在目标第 0–4 步和冻结检查点内取平均，再用同一组概念块作配对重采样。
区间为类内抽样 4,000 次的百分位 95% 区间，以训练方向和已有刺激配对为条件。</p>''',
            '<a href="early_paired.png"><img src="early_paired.png" alt="各预览相对异类预览的早期原始差值及配对区间"></a>',
            panel(ARMS,"early",main),
            '<h3>改变 carry 后，预览效应改变了多少</h3>',
            panel(["carry-reset"],"early",main),
            '''<p>有 carry 时，同概念异图的残差散度低于异类，也低于同图。重置后，“同概念异图减同图”的原始散度差在两套数据上均约为 −0.00003，其配对区间均跨零。
两套数据的 carry 交互区间均低于零。重置只干预预览与目标之间的 PCoder 状态，保留 Hebbian 记忆和熟悉度回路。
门值仍由该条件实际的目标反应计算，因此交互包含重置通过目标反应和门值产生的间接影响。</p>
<details><summary>单看目标出现的第 0 步</summary>''',panel(ARMS,"t0",main),'</details>',
            '''<h2>整体活动强弱能解释多少</h2>
<p>将每张图中心化后的向量按自身均方根缩放到相同幅度，仍沿原训练方向读出类别投影并重算照片残差散度。
有 carry 时，两套数据的同概念异图残差仍比异类更集中。查看残差总幅度可以看到，照片间更集中并不要求每张照片的整个残差向量都变短。</p>
<a href="components_amplitude.png"><img src="components_amplitude.png" alt="残差幅度、类别方向余弦与等幅后的残差散度"></a>
<details><summary>等幅复核的早期数值</summary>''', panel(ARMS,"early",("shared_alignment","photo_var_unit")),'''</details>
<h2>能怎样解释</h2><p>相对异类，同图的类别共有反应在两批数据中都增强。同概念异图使同概念的不同照片在类别空间以外更相似，
符合推断过程中保留共有信息、减弱照片间差异的解释。配对批次到了第 10–14 步，保留和重置的曲线已非常接近。
该分解不能确定被削弱的是纹理、姿态还是背景，也不能区分递归的非线性选择与预览状态和目标反应的简单混合。异类预览是共同参照，本实验没有无预览基线。</p>
<p><a href="absolute.csv">逐步原始值与区间</a> · <a href="contrasts.csv">逐步配对差值</a> ·
<a href="windows.csv">三个时间窗及第 0 步</a> · <a href="audit.json">数据划分与缓存核对</a> ·
<a href="../carry/index.html">原 carry 对照</a> · <a href="../fc2_ablation/lateral/fresh_replication/index.html">新批次的原始解码和 Fisher 曲线</a></p></main></html>''']
    (OUT/"index.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    build()
