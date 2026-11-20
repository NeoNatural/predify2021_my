"""Render the registered carry-on FC2 ablation with three preview histories."""

from html import escape
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from report_familiarity_timecourse_20261119 import (
    paired_readout_overview,
    DOC,
    NAMES,
    HISTORY,
    METRICS,
    GROUPS,
    RELATIVE,
    POINTS,
    COLORS,
    differences,
)
from report_pcoder_carry_20261119 import table

OUT = DOC / "fc2_ablation"
ARMS = {"on": "完整类别调节", "scale": "只保留相同平均抑制", "off": "关闭整个末端调节"}
WINDOWS = {0: "早期 0–4 步", 5: "中期 5–9 步", 10: "晚期 10–14 步"}


def save_chart(data, ds, group, relative):
    measures = GROUPS[group]
    fig, axes = plt.subplots(
        len(measures), 3, sharex=True, sharey="row", figsize=(12, 2.0 * len(measures) + 1.05)
    )
    for col, (arm, title) in enumerate(ARMS.items()):
        sel = data[(data.dataset == ds) & (data.arm == arm)]
        for row, m in enumerate(measures):
            ax = axes[row, col]
            ax.axvspan(4.5, 9.5, color="#edf3f8", zorder=-3)
            if relative:
                ax.axhline(0, lw=0.8, color="#596d7e")
            ax.grid(axis="y", lw=0.65, color="#d6dfe5")
            for i, h in enumerate(HISTORY):
                if relative and h == "exact":
                    continue
                d = sel[sel.history == h].sort_values("step")
                label = ("同图相对" if relative else "") + HISTORY[h]
                ax.plot(
                    d.step,
                    d[m],
                    label=label,
                    color=COLORS[i],
                    ls=["-", "--", "-."][i],
                    marker=["o", "s", "^"][i],
                    ms=3,
                    mfc="white",
                    lw=1.65,
                )
            ax.set_title(
                (title + "\n" if row == 0 else "") + METRICS[m],
                loc="left",
                fontsize=10.5,
                fontweight="bold",
            )
            if col == 0:
                unit = (
                    "模型单位"
                    if m == "mean"
                    else (
                        "比值"
                        if m == "fisher"
                        else (
                            "平方距离" if m in ["within", "between"] else "分数" if m == "margin" else "%"
                        )
                    )
                )
                if relative:
                    unit = "相对差（%）" if m in RELATIVE else "百分点" if m in POINTS else "分数差"
                ax.set_ylabel(unit)
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.tick_params(labelsize=8)
        axes[-1, col].set_xlabel("目标出现后的递归步")
    fig.suptitle(
        f"{NAMES[ds]}　|　" + ("同图相对两种对照的变化" if relative else "三种预览条件的绝对反应"),
        ha="left",
        x=0.064,
        fontweight="bold",
    )
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.06, 0.954),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.917), w_pad=1, h_pad=1.05)
    path = f'{ds}_{group}_{"effects" if relative else "absolute"}.png'
    fig.savefig(OUT / path, dpi=130, facecolor="white")
    plt.close(fig)
    return (
        f'<a href="{path}"><img src="{path}" loading="lazy" alt="{NAMES[ds]} 三种调节设置的时间曲线"></a>'
    )


def overview(paired):
    return paired_readout_overview(paired, {arm: ARMS[arm] for arm in ("on", "off")}, OUT / "overview.png",
                                  "保留 PCoder carry，有无 FC2 末端调节　|　同图相对同概念异图")


def build():
    full = pd.concat([pd.read_csv(OUT / f"{ds}_absolute.csv") for ds in NAMES])
    ci = pd.concat([pd.read_csv(OUT / f"{ds}_intervals.csv") for ds in NAMES])
    effects = pd.concat([differences(d).assign(arm=arm) for arm, d in full.groupby("arm")])
    group = ["dataset", "arm", "history", "step"]
    absolute = full.groupby(group, as_index=False)[list(METRICS)].mean()
    paired = effects.groupby(group, as_index=False)[list(METRICS)].mean()
    windows = (
        paired.assign(start=(paired.step // 5) * 5)
        .groupby(["dataset", "history", "start", "arm"])[list(METRICS)]
        .mean()
    )
    for r in ci[ci.comparison.isin(ARMS)].itertuples():
        np.testing.assert_allclose(
            r.estimate, windows.loc[(r.dataset, r.reference, r.start, r.comparison), r.metric], atol=2e-5
        )
    # Independent t9 verification on and off, including the original sign convention for suppression.
    archive = pd.read_csv(
        DOC.parents[1] / "priming_sharpen_20261112/learning_curve_20261113/fc2_test.csv"
    )
    lookup = paired[paired.step == 9].set_index(["dataset", "history", "arm"])
    for r in archive[
        archive.reference.isna() & archive.condition.isin(["base", "selected"])
    ].itertuples():
        metric = "mean" if r.metric == "suppression" else r.metric
        value = lookup.loc[(r.dataset, r.history, "off" if r.condition == "base" else "on"), metric]
        np.testing.assert_allclose(
            value, -r.estimate if r.metric == "suppression" else r.estimate, atol=2e-5
        )
    absolute.to_csv(OUT / "absolute.csv", index=False)
    paired.to_csv(OUT / "paired.csv", index=False)
    windows.to_csv(OUT / "windows.csv")
    ci.to_csv(OUT / "intervals.csv", index=False)
    html = [
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>有 PCoder carry 时关闭 FC2 类别调节</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#203543;font:16px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1240px;margin:auto;background:white;padding:32px clamp(14px,4vw,54px) 60px}nav a{margin-right:18px}a{color:#175e89}h1,h2,h3{line-height:1.4}h2{margin-top:40px;padding-top:16px;border-top:2px solid #285a77}p{max-width:96ch}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap}th,td{border-bottom:1px solid #d5dfe5;padding:6px 9px;text-align:right}th{background:#eef4f8}img{width:100%;margin:14px 0}summary{cursor:pointer;color:#175e89;font-weight:bold}</style>
<main><nav><a href="../index.html">返回时间曲线</a><a href="#things">THINGS</a><a href="#imagenet">ImageNet</a><a href="#methods">方法与数据</a></nav><h1>保留 PCoder 的预览状态，关闭 FC2 类别调节</h1>
<p>三种设置共享两帧预览结束时的网络状态、四层 Hebbian 抑制和相同的无噪声目标。所有差别都发生在递归网络末端。左列为原方案，熟悉度控制类别重加权和少量平均抑制。中列只去掉类别重加权，每张目标在每个时间步、每种预览和每个种子下都保留左列的平均活动。右列关闭整个末端操作，直接读取四层抑制后的 FC2 活动。</p>
<p>主要比较同图重复与同概念异图；异类预览也单独给出。观察各组内部的同图效应，再看这个效应在组间相差多少。三个种子共用同一批测试图，整个末端调节关闭后输出与门的种子无关，因此原始组只计算一次。</p>"""
    ]
    html.append(overview(paired))
    for ds, name in NAMES.items():
        html.append(
            f'<h2 id="{ds}">{name} · '
            + ("108 张目标，30 个概念" if ds == "things" else "96 张目标，24 个概念")
            + "</h2>"
        )
        view = windows.loc[(ds, "same_concept", 10)]
        gain = ci[
            (ci.dataset == ds)
            & (ci.reference == "same_concept")
            & (ci.start == 10)
            & (ci.metric == "fisher")
            & (ci.comparison == "category_increment")
        ].iloc[0]
        html.append(
            f'<p>同图相对同概念异图，晚期 Fisher 增益在完整组为 {view.loc["on","fisher"]:+.3f}%，关闭末端调节后为 {view.loc["off","fisher"]:+.3f}%。在平均活动配平的条件下，类别重加权使这项重复效应改变 {gain.estimate:+.3f} 个百分点［{gain.low:+.3f}, {gain.high:+.3f}］。</p>'
        )
        html.append('<details><summary>查看晚期各条件的绝对水平</summary><p>表中是第 10–14 步的均值。可在同一预览条件内比较关闭调节前后的变化。</p>')
        late = absolute.loc[(absolute.dataset == ds) & (absolute.step >= 10)].groupby(["arm","history"])[list(METRICS)].mean()
        abs_rows = []
        for arm, label in ARMS.items():
            for history, hname in HISTORY.items():
                values = late.loc[(arm, history)]
                abs_rows.append([label, hname] + [f"{values[m]:.4f}" for m in ("mean","within","between","fisher","margin","auc","ba","mvpa_auc","mvpa_ba")])
        html.append(table(["FC2 处理","预览条件","活动","类内","类间","Fisher","间隔","固定 AUC %","固定 BA %","MVPA AUC %","MVPA BA %"], abs_rows) + "</details>")
        html.append(
            "<h3>三种设置的条件差</h3><p>第一列指标为抑制，正值表示同图活动更低。其余指标为同图减对照，或同图相对对照的百分比变化。固定解码和逐步 MVPA 的 AUC、BA 均以百分点计。</p>"
        )
        cols = ["mean", "within", "between", "fisher", "margin", "auc", "ba", "mvpa_auc", "mvpa_ba"]
        for h in ("same_concept", "different"):
            html.append("<h4>同图相对" + HISTORY[h] + "</h4>")
            rows = []
            for start, wname in WINDOWS.items():
                for arm, label in ARMS.items():
                    d = windows.loc[(ds, h, start, arm)]
                    rows.append([wname, label] + [f'{(-d[m] if m=="mean" else d[m]):+.3f}' for m in cols])
            html.append(
                table(
                    [
                        "窗口",
                        "FC2 处理",
                        "活动抑制 %",
                        "类内 %",
                        "类间 %",
                        "Fisher %",
                        "间隔",
                        "固定 AUC",
                        "固定 BA",
                        "MVPA AUC",
                        "MVPA BA",
                    ],
                    rows,
                )
            )
        html.append(
            "<h3>时间过程</h3><p>同一行三列共用纵轴。浅色底是中期第 5–9 步。差值图中的平均活动低于零表示同图受到更强抑制。</p>"
        )
        for kind in GROUPS:
            if kind == "detail":
                html.append("<details><summary>展开类内与类间散度、决策间隔和平衡准确率</summary>")
            for rel, d in [(False, absolute), (True, paired)]:
                html.append(save_chart(d, ds, kind, rel))
            if kind == "detail":
                html.append("</details>")
    html.append(
        """<h2 id="methods">方法与数据</h2><p>在有 carry 的原轨迹上重放全部目标，提取终末调节前的 4096 维 FC2 活动 x。只保留平均抑制时，用原完整组的群体均值除以 x 的均值，对 x 的每个分量乘同一系数。这等价于沿用冻结熟悉度门的 1 − 0.003g，并将类别混合系数设为零。逐图检查该系数范围及缩放后均值。整个末端关闭时直接使用 x。三组目标阶段都保留四层原有的短时记忆抑制。第 9 步核对了调节前的逐图反应，以及原报告中完整组和关闭组的七项条件差。</p>
<p>Fisher 是单位长度归一化后类间散度与类内散度之比，因此对一张图的全部单元乘相同正数时，其 Fisher 计算用到的方向保持一致。固定解码沿用训练图上的线性 SVM。逐步 MVPA 按原六折每大类留出一整个概念，分别在各设置、时间步、预览条件的训练折标准化并拟合线性 SVM。窗口先按种子和时间步求配对条件差，再做平均。活动、几何和固定读出另以 4,000 次类内配对概念块重采样计算窗口 95% 区间。MVPA 报告六折平均点估计。</p>
<p><a href="absolute.csv">三条件逐步绝对值</a> · <a href="paired.csv">逐步条件差</a> · <a href="windows.csv">窗口汇总</a> · <a href="intervals.csv">窗口与组间差的配对区间</a></p></main></html>"""
    )
    (OUT / "index.html").write_text("\n".join(html), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lateral", action="store_true", help="Render the four-arm lateral comparison")
    if parser.parse_args().lateral:
        from report_fc2_lateral_20261120 import build as build_lateral

        build_lateral()
    else:
        build()
