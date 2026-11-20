"""Render paired FC2 trajectories from the PCoder carry intervention."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from report_familiarity_timecourse_20261119 import (
    paired_readout_overview,
    DOC,
    NAMES,
    HISTORY,
    COLORS,
    METRICS,
    RELATIVE,
    POINTS,
    GROUPS,
    differences,
)

OUT = DOC / "carry"
ARMS = {"carry": "有 carry · 延续预览状态", "reset": "无 carry · 恢复灰屏状态"}
WINDOWS = {0: "早期 0–4", 5: "中期 5–9", 10: "晚期 10–14"}


def chart(frame, ds, kind, paired, intervals):
    metrics = GROUPS[kind]
    fig, axes = plt.subplots(
        len(metrics), 2, figsize=(10.2, 2.30 * len(metrics) + 0.85), sharex=True, sharey="row"
    )
    histories = list(HISTORY)[1:] if paired else list(HISTORY)
    for col, (arm, title) in enumerate(ARMS.items()):
        sel = frame.loc[(frame.dataset == ds) & (frame.arm == arm)]
        for row, m in enumerate(metrics):
            ax = axes[row, col]
            ax.axvspan(4.5, 9.5, color="#edf2f6", zorder=-4)
            ax.grid(axis="y", lw=0.6, color="#d4dde3")
            if paired:
                ax.axhline(0, color="#657888", lw=0.85, zorder=0)
            for h in histories:
                i = list(HISTORY).index(h)
                d = sel.loc[sel.history == h].sort_values("step")
                if paired and m in RELATIVE:
                    ci = intervals.loc[
                        (intervals.arm == arm) & (intervals.reference == h) & (intervals.metric == m)
                    ].sort_values("step")
                    np.testing.assert_allclose(d[m], ci.estimate, atol=2e-5)
                    ax.fill_between(
                        d.step.to_numpy(),
                        ci.low.to_numpy(),
                        ci.high.to_numpy(),
                        color=COLORS[i],
                        alpha=0.11,
                        linewidth=0,
                    )
                ax.plot(
                    d.step,
                    d[m],
                    color=COLORS[i],
                    label=("同图相对" if paired else "") + HISTORY[h],
                    marker=["o", "s", "^"][i],
                    ls=["-", "--", "-."][i],
                    ms=3.6,
                    mfc="white",
                    lw=1.65,
                )
            ax.set_title(
                (title + "\n" if row == 0 else "") + METRICS[m],
                loc="left",
                fontweight="bold",
                fontsize=11,
            )
            if col == 0:
                if paired:
                    unit = (
                        "相对差（%）" if m in RELATIVE else "分数差" if m == "margin" else "差（百分点）"
                    )
                else:
                    unit = (
                        "%"
                        if m in POINTS
                        else {
                            "mean": "模型活动单位",
                            "within": "平方距离",
                            "between": "平方距离",
                            "margin": "类别分数",
                            "fisher": "比值",
                        }[m]
                    )
                ax.set_ylabel(unit)
            ax.set_xlim(-0.3, 14.3)
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            fmt = ScalarFormatter(useOffset=False)
            fmt.set_scientific(False)
            ax.yaxis.set_major_formatter(fmt)
            ax.tick_params(labelsize=9)
        axes[-1, col].set_xlabel("目标出现后的递归步")
    fig.suptitle(
        f"{NAMES[ds]}　|　" + ("同图相对两种对照的变化" if paired else "三种预览条件下的 FC2 反应"),
        x=0.085,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(histories),
        loc="upper left",
        bbox_to_anchor=(0.074, 0.956),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.920), h_pad=1.25, w_pad=0.9)
    name = f'{ds}_{kind}_{"effects" if paired else "absolute"}.png'
    fig.savefig(OUT / name, dpi=125, facecolor="white")
    plt.close(fig)
    return (
        f'<figure><a href="{name}"><img loading="lazy" src="{name}" alt="{NAMES[ds]} 两种设置下的'
        + ("条件差值" if paired else "三条件绝对值")
        + '"></a></figure>'
    )


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{x}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def retention_table(ds, intervals):
    sub = intervals.loc[intervals.dataset == ds].set_index(["reference", "metric", "start", "stat"])
    rows = []
    for h in ("same_concept", "different"):
        for start, m in [(0, "mean"), (5, "fisher"), (10, "fisher")]:
            on, off, change = (sub.loc[(h, m, start, k)] for k in ["carry", "reset", "change"])
            ratio_text = "分母区间跨零"
            if (h, m, start, "ratio_pct") in sub.index:
                ratio = sub.loc[(h, m, start, "ratio_pct")]
                ratio_text = f"{ratio.estimate:.1f}%［{ratio.low:.1f}, {ratio.high:.1f}］"
            sign = -1 if m == "mean" else 1
            lo, hi = sorted([sign * change.low, sign * change.high])
            rows.append(
                [
                    HISTORY[h],
                    WINDOWS[start],
                    "活动抑制" if m == "mean" else "Fisher 增益",
                    f"{sign*on.estimate:.2f}%",
                    f"{sign*off.estimate:.2f}%",
                    ratio_text,
                    f"{sign*change.estimate:+.2f}［{lo:+.2f}, {hi:+.2f}］",
                ]
            )
    return table(
        [
            "同图相比",
            "窗口",
            "效应",
            "有 carry",
            "无 carry",
            "保留比例［95% 区间］",
            "无减有，百分点［95% 区间］",
        ],
        rows,
    )


def decoding_table(ds, windows):
    rows = []
    for h in ("same_concept", "different"):
        for start in WINDOWS:
            a = windows.loc[(ds, h, start, "carry")]
            b = windows.loc[(ds, h, start, "reset")]
            cols = ["within", "between", "margin", "auc", "ba", "mvpa_auc", "mvpa_ba"]
            rows.append([HISTORY[h], WINDOWS[start]] + [f"{a[m]:+.2f} / {b[m]:+.2f}" for m in cols])
    return table(
        [
            "同图相比",
            "窗口",
            "类内 %",
            "类间 %",
            "决策间隔",
            "固定 AUC pp",
            "固定 BA pp",
            "MVPA AUC pp",
            "MVPA BA pp",
        ],
        rows,
    )


def overview(paired):
    return paired_readout_overview(paired, ARMS, OUT / "carry_overview.png",
                                  "有无 PCoder carry　|　同图相对同概念异图的重复效应")


def main():
    full, intervals, bands = [], [], []
    for ds in NAMES:
        for arm, folder in [("carry", DOC), ("reset", OUT)]:
            f = pd.read_csv(folder / f"{ds}_by_seed.csv")
            assert len(f) == 135 and len(f[["seed", "history", "step"]].drop_duplicates()) == 135
            full.append(f.assign(arm=arm))
        intervals.append(pd.read_csv(OUT / f"{ds}_window_intervals.csv"))
        bands.append(pd.read_csv(OUT / f"{ds}_pointwise_intervals.csv"))
    full = pd.concat(full, ignore_index=True)
    effects = pd.concat([differences(g).assign(arm=k) for k, g in full.groupby("arm")], ignore_index=True)
    grouping = ["dataset", "arm", "history", "step"]
    absolute = full.groupby(grouping, as_index=False)[list(METRICS)].mean()
    paired = effects.groupby(grouping, as_index=False)[list(METRICS)].mean()
    windows = (
        paired.assign(start=(paired.step // 5) * 5)
        .groupby(["dataset", "history", "start", "arm"])[list(METRICS)]
        .mean()
    )
    full.to_csv(OUT / "absolute_by_seed.csv", index=False)
    paired.to_csv(OUT / "condition_effects.csv", index=False)
    windows.to_csv(OUT / "window_effects.csv")
    intervals = pd.concat(intervals, ignore_index=True)
    bands = pd.concat(bands, ignore_index=True)
    html = [
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>预览状态是否需要带入目标阶段？</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f1f5f7;color:#203746;font:16px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1160px;margin:auto;background:white;padding:32px clamp(14px,4vw,50px) 60px}a{color:#175e89}nav a{margin-right:18px}h1,h2,h3{line-height:1.4}h1{font-size:clamp(26px,4vw,40px)}h2{border-top:2px solid #1b658d;padding-top:18px;margin-top:36px}p{max-width:95ch}img{width:100%;height:auto}figure{margin:15px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap}th,td{border-bottom:1px solid #d6e0e6;padding:7px;text-align:right}th{background:#eff4f7}summary{font-weight:700;color:#175e89;cursor:pointer}details{margin-top:18px}</style>
<main><nav><a href="../index.html">原时间曲线</a><a href="../fc2_ablation/index.html">有 carry 时关闭 FC2 类别调节</a><a href="#things">THINGS</a><a href="#imagenet">ImageNet</a><a href="#method">方法与数据</a></nav>
<h1>把预览结束时的 PCoder 状态清除后，重复效应还剩多少？</h1>
<p>每个试次先呈现两帧预览，再呈现同一个无噪声目标。左列让五个预测编码模块（PCoder）从预览结束时的状态继续计算；右列只在目标出现前将五层状态恢复为共同的灰屏初态。两边在目标阶段都继续递归更新 15 步。两帧预览写入的 Hebbian 记忆、熟悉度回路、门值定标和 FC2 调节规则都保留。熟悉度门在每步读取该条件下真实的目标反应，因此重置可以间接改变门值。</p>
<p>优先看同图与同概念异图的差别。它们的预览都出现了目标所属概念，只有前者呈现过同一张照片。异类对照还包含概念和大类匹配程度的差别。先看两列绝对曲线，再看差值曲线；同一行的两列共用纵轴。浅灰蓝底为第 5–9 步。</p>"""
    ]
    primary = intervals.loc[intervals.reference == "same_concept"].set_index(["dataset", "metric", "start", "stat"])
    def retention(dataset, metric, start):
        return primary.loc[(dataset, metric, start, "ratio_pct"), "estimate"]

    html.append(
        f'<p><b>重置的影响集中在目标出现后的前几步。</b>相对同概念异图，重置后早期重复抑制在 THINGS 上保留 '
        f'{retention("things","mean",0):.0f}%，在 ImageNet 上保留 {retention("imagenet","mean",0):.0f}%。'
        f'两组的晚期 Fisher 优势几乎重合，分别保留 {retention("things","fisher",10):.1f}% 和 '
        f'{retention("imagenet","fisher",10):.1f}%。下图将两种设置叠在一起，便于看出差异何时消退；'
        '下方的并排图保留三种预览条件各自的绝对水平。</p>'
    )
    html.append(overview(paired))
    for ds in NAMES:
        ci = bands.loc[bands.dataset == ds]
        html.append(
            f'<h2 id="{ds}">{NAMES[ds]} · '
            + ("108 张目标图，30 个概念" if ds == "things" else "96 张目标图，24 个概念")
            + "</h2>"
        )
        html.append(retention_table(ds, intervals))
        ref = windows.loc[(ds, "same_concept")]
        early_on = ref.loc[(0, "carry"), "fisher"]
        early_off = ref.loc[(0, "reset"), "fisher"]
        late = ref.loc[(10, "reset")]
        html.append(
            f'<p>同图相对同概念异图的早期 Fisher 差异从 {early_on:+.2f}% 变为 {early_off:+.2f}%。重置后，晚期类间散度相差 {late["between"]:+.2f}%，类内散度相差 {late["within"]:+.2f}%；固定分类器宏 AUC 相差 {late["auc"]:+.2f} 个百分点，逐步 MVPA 宏 AUC 相差 {late["mvpa_auc"]:+.2f} 个百分点。</p>'
        )
        html.append(
            "<p>活动抑制按 100 ×（1 − 同图活动／对照活动）计算，正值表示同图活动降低。Fisher 增益按 100 ×（同图 Fisher／对照 Fisher − 1）计算。保留比例为无 carry 的窗口效应除以有 carry 的窗口效应，100% 表示等大，负值表示方向反转。右栏直接给出两组效应的差。</p>"
        )
        html.append("<h3>激活、类别分离与分类读出</h3>")
        html.append(chart(absolute, ds, "main", False, ci))
        html.append(
            "<h3>同一批目标逐条件配对后，差异有多大？</h3><p>此图的激活差为同图相对对照的有符号变化，负值对应上表中的正抑制。色带是激活和 Fisher 各步的配对概念块 95% 区间。固定分类器和逐步 MVPA 显示三种子平均点估计；两者的宏 AUC 均衡量目标大类的辨别。</p>"
        )
        html.append(chart(paired, ds, "main", True, ci))
        html.append(
            "<h3>类内、类间与解码结果</h3><p>每格按“有 carry / 无 carry”排列，给出同图相对所列对照的变化。pp 表示百分点。正决策间隔差表示同图条件下正确类别相对最强竞争类别更占优势。</p>"
            + decoding_table(ds, windows)
        )
        html.append(
            "<details><summary>展开类内散度、类间散度、固定分类间隔与两种平衡准确率的完整曲线</summary>"
        )
        html.append(
            chart(absolute, ds, "detail", False, ci)
            + chart(paired, ds, "detail", True, ci)
            + "</details>"
        )
    html.append(
        """<h2 id="method">方法与核对</h2>
<p>目标、预览顺序、训练完成的三个检查点和原验证集定标全部配对复用。有 carry 取自原全程缓存。无 carry 在第二帧结束、短时记忆完成两次写入并冻结之后，仅恢复五层 PCoder 的表征、预测、误差梯度与预测误差。逐批核对了全部恢复张量、四层记忆的写入次数和冻结状态、重置前后记忆张量的标识和版本，以及送入熟悉度回路的两帧预览特征与原缓存逐值一致。这项对照估计恢复初态对整套回路的影响，包含它通过目标反应改变门值的间接影响。</p>
<p>早期为 0–4 步，中期 5–9 步，晚期 10–14 步。先在每个种子每步求同图对指定对照的效应，再平均种子和时间步。Fisher 是按长度归一化后的 FC2 群体活动所算的类间散度 / 类内散度。固定分类器沿用独立训练的线性 SVM；逐步 MVPA 采用相同的六折、各大类每折留出一整个概念，只在折内训练分类器和标准化，不参与回路选择。宏 AUC 衡量类别排序，平衡准确率（BA）是各大类正确率的平均。</p>
<p>区间来自 4,000 次各大类内配对概念块重采样，两组、三种预览、三个种子和所有时间步共用抽样。窗口统计量在每次重采样内重新计算；曲线色带为逐点区间。表中的分项不能解释为 carry 与 Hebbian 记忆各自独立相加的因果份额。</p>
<p><a href="absolute_by_seed.csv">两组逐种子绝对值</a> · <a href="condition_effects.csv">逐时间步条件差</a> · <a href="window_effects.csv">窗口条件差</a> · <a href="things_window_intervals.csv">THINGS 配对区间</a> · <a href="imagenet_window_intervals.csv">ImageNet 配对区间</a></p></main></html>"""
    )
    (OUT / "index.html").write_text("\n".join(html), encoding="utf8")


if __name__ == "__main__":
    main()
