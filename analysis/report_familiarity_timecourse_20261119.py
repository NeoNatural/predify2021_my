"""Build the offline FC2 time-course supplement and its three-condition plots."""

from pathlib import Path
from html import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

DOC = Path(__file__).resolve().parents[1] / "docs/hebbian_week_20261112_14/timecourse_20261119"
NAMES = {"things": "THINGS", "imagenet": "ImageNet"}
HISTORY = {"exact": "同一张图", "same_concept": "同概念的另一张图", "different": "不同大类的图"}
COLORS = ["#176294", "#b64d1a", "#25816f"]
METRICS = {
    "mean": "FC2 平均激活",
    "fisher": "Fisher 比值",
    "auc": "固定分类器 · 宏 AUC",
    "mvpa_auc": "逐步 MVPA · 宏 AUC",
    "within": "类内散度",
    "between": "类间散度",
    "margin": "固定分类器 · 决策间隔",
    "ba": "固定分类器 · 平衡准确率",
    "mvpa_ba": "逐步 MVPA · 平衡准确率",
}
GROUPS = {
    "main": ["mean", "fisher", "auc", "mvpa_auc"],
    "detail": ["within", "between", "margin", "ba", "mvpa_ba"],
}
RELATIVE = {"mean", "within", "between", "fisher"}
POINTS = {"ba", "auc", "mvpa_ba", "mvpa_auc"}
plt.rcParams.update(
    {
        "font.family": "Microsoft YaHei",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8e9aa5",
    }
)


def paired_readout_overview(paired, arms, path, title):
    """Plot the original same-image contrast alongside both category decoders."""
    panels = [
        ("mean", "活动抑制（%）"),
        ("fisher", "Fisher 增益（%）"),
        ("auc", "固定解码 · 宏 AUC\n（百分点）"),
        ("mvpa_auc", "逐步 MVPA · 宏 AUC\n（百分点）"),
        ("ba", "固定解码 · 平衡准确率\n（百分点）"),
        ("mvpa_ba", "逐步 MVPA · 平衡准确率\n（百分点）"),
        ("margin", "固定解码 · 决策间隔\n（分数差）"),
    ]
    fig, axes = plt.subplots(len(panels), 2, figsize=(10.8, 13.1), sharex=True, sharey="row")
    focus = paired.loc[paired.history == "same_concept"]
    for col, (dataset, name) in enumerate(NAMES.items()):
        axes[0, col].set_title(name, loc="left", fontweight="bold", fontsize=13)
        for row, (metric, label) in enumerate(panels):
            ax = axes[row, col]
            ax.axvspan(4.5, 9.5, color="#edf3f8", zorder=-2)
            ax.axhline(0, color="#687c8a", lw=.8)
            ax.grid(axis="y", color="#d4dde3", lw=.65)
            for (arm, legend), (color, marker, style) in zip(
                arms.items(), [("#215d89", "o", "-"), ("#b25322", "s", "--")], strict=True
            ):
                values = focus.loc[(focus.dataset == dataset) & (focus.arm == arm)].sort_values("step")
                np.testing.assert_array_equal(values.step, np.arange(15))
                y = values[metric].to_numpy()
                assert np.isfinite(y).all()
                ax.plot(values.step, -y if metric == "mean" else y, color=color, marker=marker,
                        ls=style, ms=3.5, lw=1.6, mfc="white", label=legend)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            if row == len(panels) - 1:
                ax.set_xlabel("目标出现后的递归步")
    fig.suptitle(title, x=.12, y=.986, ha="left", fontweight="bold", fontsize=15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(.112, .967), ncol=2, frameon=False)
    fig.subplots_adjust(left=.13, right=.985, top=.922, bottom=.045, hspace=.24, wspace=.12)
    fig.savefig(path, dpi=145, facecolor="white")
    plt.close(fig)
    caption = ("同图相对同概念异图。活动抑制为正表示同图活动更低，其余指标为正表示同图更高。"
               "固定解码在各步沿用同一个分类器，逐步 MVPA 在各条件、各步分别训练并按概念交叉验证。"
               "浅蓝底为第 5–9 步，曲线为逐步点估计。")
    return (f'<figure><a href="{path.name}"><img src="{path.name}" alt="{title}的活动、Fisher、固定解码和逐步 MVPA">'
            f'</a><figcaption>{caption}</figcaption></figure>')


def differences(full):
    keys = ["dataset", "seed", "step"]
    exact = full[full.history == "exact"].set_index(keys)
    pieces = []
    for history in ("same_concept", "different"):
        ref = full[full.history == history].set_index(keys)
        vals = {
            m: (100 * (exact[m] / ref[m] - 1) if m in RELATIVE else exact[m] - ref[m]) for m in METRICS
        }
        pieces.append(pd.DataFrame(vals).assign(history=history).reset_index())
    return pd.concat(pieces, ignore_index=True)


def chart(df, ds, metrics, kind, mode, intervals):
    paired = mode == "paired"
    fig, axes = plt.subplots(
        (len(metrics) + 1) // 2, 2, figsize=(11.4, 5.9 if len(metrics) == 4 else 8.1), squeeze=False
    )
    histories = list(HISTORY)[1:] if paired else list(HISTORY)
    for m, ax in zip(metrics, axes.flat):
        ax.axvspan(4.5, 9.5, color="#eaf1f7", zorder=-2)
        ax.axvline(9, color="#778798", lw=0.9, ls=":", zorder=-1)
        if paired:
            ax.axhline(0, color="#6a7885", lw=0.9)
        ax.yaxis.grid(True, color="#ced9e1", alpha=0.58, lw=0.7)
        for h in histories:
            i = list(HISTORY).index(h)
            y = df[(df.dataset == ds) & (df.history == h)].sort_values("step")
            ax.plot(
                y.step,
                y[m],
                label=("同图 − " if paired else "") + HISTORY[h],
                color=COLORS[i],
                ls=["-", "--", "-."][i],
                marker=["o", "s", "^"][i],
                ms=4,
                lw=1.8,
                markerfacecolor="white",
                markeredgewidth=0.9,
            )
            if paired and m in ("mean", "within", "between", "fisher", "margin", "auc", "ba"):
                key = "suppression" if m == "mean" else m
                old = intervals.loc[(ds, h, key)]
                low, high = (-old.ci_high, -old.ci_low) if m == "mean" else (old.ci_low, old.ci_high)
                estimate = float(y.loc[y.step == 9, m].iloc[0])
                ax.vlines(9, low, high, color=COLORS[i], linewidth=1.2)
                ax.plot([9, 9], [low, high], linestyle="none", marker="_", color=COLORS[i], ms=6)
                np.testing.assert_allclose(
                    estimate, -old.estimate if m == "mean" else old.estimate, atol=3e-4
                )
        ax.set_title(METRICS[m], loc="left", fontweight="bold", fontsize=12)
        ax.set_ylabel(
            ("相对差（%）" if m in RELATIVE else "差（百分点）" if m in POINTS else "分数差")
            if paired
            else (
                "%"
                if m in POINTS
                else (
                    "模型活动单位"
                    if m == "mean"
                    else "类别分数" if m == "margin" else "比值" if m == "fisher" else "平方距离"
                )
            )
        )
        ax.set_xlim(-0.3, 14.3)
        ax.set_xticks([0, 2, 4, 6, 8, 9, 10, 12, 14])
        ax.tick_params(labelsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel("目标出现后的递归步")
    for ax in list(axes.flat)[len(metrics) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        NAMES[ds] + "   |   " + ("同图相对两种对照的变化" if paired else "三种预览条件下的目标反应"),
        x=0.078,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=len(histories),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.5, w_pad=2.2)
    path = f"{ds}_{kind}_{mode}.png"
    fig.savefig(DOC / path, dpi=145, facecolor="white")
    plt.close(fig)
    return path


def img(path, label):
    return f'<figure><a href="{path}"><img src="{path}" alt="{escape(label)}" loading="lazy"></a><figcaption>{label}。点击图片可放大。</figcaption></figure>'


def window_table(windows, ds):
    rows = []
    for (window, ref), values in windows.loc[ds].iterrows():
        cells = [window, HISTORY[ref]] + [
            f"{values[m]:+.3f}"
            for m in ["mean", "within", "between", "fisher", "auc", "ba", "margin", "mvpa_auc", "mvpa_ba"]
        ]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    head = [
        "时间步",
        "同图相比",
        "激活 %",
        "类内 %",
        "类间 %",
        "Fisher %",
        "固定 AUC pp",
        "固定 BA pp",
        "间隔",
        "MVPA AUC pp",
        "MVPA BA pp",
    ]
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{c}</th>" for c in head)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def build():
    archive = pd.read_csv(
        DOC.parents[1] / "priming_sharpen_20261112/learning_curve_20261113/fc2_encoder_control.csv"
    )
    intervals = archive.loc[
        (archive.comparison == "history_contrast")
        & (archive.condition == "trainable")
        & (archive.seed == "mean")
    ].set_index(["dataset", "reference_history", "metric"])
    all_data = pd.concat([pd.read_csv(DOC / f"{ds}_by_seed.csv") for ds in NAMES])
    pair = differences(all_data)
    group = ["dataset", "history", "step"]
    absolute = all_data.groupby(group, as_index=False)[list(METRICS)].mean()
    paired = pair.groupby(group, as_index=False)[list(METRICS)].mean()
    absolute.to_csv(DOC / "absolute.csv", index=False)
    paired.to_csv(DOC / "paired.csv", index=False)
    windows = (
        paired.assign(
            window=pd.cut(paired.step, [-1, 4, 9, 14], labels=["早期 0–4", "中期 5–9", "晚期 10–14"])
        )
        .groupby(["dataset", "window", "history"], observed=True)[list(METRICS)]
        .mean()
    )
    windows.to_csv(DOC / "windows.csv")
    html = [
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>重复刺激如何改变 FC2 的类别表征</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#182c3a;font:16px/1.77 "Microsoft YaHei",sans-serif}main{max-width:1160px;margin:0 auto;background:white;padding:36px clamp(16px,4vw,58px) 60px}h1,h2,h3{line-height:1.35}h1{font-size:clamp(28px,4vw,41px)}h2{margin-top:45px;padding-top:18px;border-top:2px solid #1d608b}a{color:#175b88}nav a{margin-right:20px}img{width:100%;height:auto}figure{margin:22px 0}figcaption,small{color:#456273}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:7px;border-bottom:1px solid #ced9e1;text-align:right}th{background:#eaf1f5}summary{cursor:pointer;color:#175b88;font-weight:bold}details{padding:15px 0}p{max-width:94ch}</style><main>
<nav><a href="../report.html">返回实验周报</a><a href="carry/index.html">有无 PCoder carry 对照</a><a href="fc2_ablation/index.html">有 carry 时关闭 FC2 类别调节</a><a href="#things">THINGS</a><a href="#imagenet">ImageNet</a><a href="#methods">指标与数据</a></nav>
<h1>看过同一张图后，目标的类别信息怎样随时间变化？</h1>
<p><b>推荐设置</b>　四层重复抑制，加上用 8 倍训练数据学到的全图熟悉度回路，再由当前目标决定 FC2 活动向哪个类别方向调整。两帧预览在 Conv5 留下短时共激活痕迹。目标出现后，熟悉度单元受到的抑制越强，熟悉度分数越高，FC2 的调节也越强。这样可以在网络内部计算熟悉度，无须预先指出物体位置或另行裁图做相似性比较。</p>
<p>选择这套设置的依据是留出数据上的权衡。全图熟悉度回路在 THINGS、ImageNet 上区分同图与同概念异图的 AUC 分别为 0.625、0.656；换成冻结的随机编码器并延长训练后，分别为 0.626、0.609。后续增加概念、增加同概念照片或改变预览位置与对比度，都没有达到预先约定的下游 FC2 复核条件。类别方向仍借助预训练分类知识和有标签的类别原型，因此曲线支持的是整套回路的群体表征效应，不能据此推断单个神经元的调谐变窄，或类别知识由局部 Hebbian 学习自行产生。</p>
<p>读图时先比较<b>同一张图</b>与<b>同概念的另一张图</b>。例如目标一直是同一张狗的照片，只改变刚看过的是这张照片、另一张狗的照片，还是其他大类的照片。同概念对照能进一步区分具体图片重复与概念重复。</p>
<p>第一组图显示三种条件的绝对水平。紧随其后的差值图把同一批目标配对比较。平均激活的差值低于零表示重复抑制，Fisher 高于零表示类别更容易从群体活动中分开。底色为中期第 5–9 步，虚线标出原先报告的第 9 步。横轴是模型更新步数，没有换算成刺激出现后的毫秒数。</p>"""
    ]
    for ds, name in NAMES.items():
        html.append(
            f'<h2 id="{ds}">{name} · '
            + ("108 张目标，5 个大类" if ds == "things" else "96 张目标，4 个大类")
            + "</h2>"
        )
        middle = windows.loc[(ds, "中期 5–9", "same_concept")]
        late = windows.loc[(ds, "晚期 10–14", "same_concept")]
        html.append(
            f'<p><b>先看同图与同概念异图。</b>中期平均激活相差 {middle["mean"]:+.2f}%，Fisher 相差 {middle["fisher"]:+.2f}%。到了晚期，类间散度相差 {late["between"]:+.2f}%，类内散度相差 {late["within"]:+.2f}%。固定分类器的宏 AUC 相差 {late["auc"]:+.2f} 个百分点，逐步重新训练分类器后相差 {late["mvpa_auc"]:+.2f} 个百分点。</p>'
        )
        html.append(
            "<p>差值图中第 9 步的短竖线来自原实验的 4,000 次配对概念块重采样，表示该步的 95% 区间。其余时间步和新增的逐步 MVPA 显示点估计。</p>"
        )
        for kind, metrics_list in GROUPS.items():
            if kind == "detail":
                html.append(
                    "<h3>把 Fisher 和分类表现拆开看</h3><p>类内散度反映同一大类的图片反应彼此相差多远，类间散度反映不同大类的平均反应相距多远。决策间隔衡量正确类别比最强竞争类别高几分。平衡准确率（BA）只在最后选中的类别改变时才会变化。</p>"
                )
            for mode, df in [("absolute", absolute), ("paired", paired)]:
                path = chart(df, ds, metrics_list, kind, mode, intervals)
                html.append(
                    img(
                        path,
                        name
                        + "，"
                        + (
                            "三种预览条件的绝对值"
                            if mode == "absolute"
                            else "同图分别对同概念异图、异类预览的配对差值"
                        ),
                    )
                )
        html.append(
            "<h3>各阶段平均差多少？</h3><p>每个时间步先在每个种子内求条件差，再对三个种子和窗口内五步取均值。pp 表示百分点。</p>"
            + window_table(windows, ds)
        )
    html.append(
        """<h2 id="methods">指标与数据</h2><p><b>固定分类器</b>在独立训练图的 FC2 反应上学会区分大类。这里每一时间步、每种条件都用同一把尺子。<b>逐步 MVPA</b>允许每个时间步、每个条件重新学习大类的线性分界。每折每个大类留出一个完整概念，该概念的全部照片只用于评估。六折划分在所有种子、条件和时间步保持一致。标准化与线性 SVM（C = 1）只在各折训练部分拟合。先在每折算宏平均一对其余 AUC 与 BA，再平均六折。两套分类器都预测目标属于哪个大类。</p>
<p><b>宏 AUC</b>衡量正确类别的图片能否排在其他图片之前，再给每个大类相同权重。<b>BA</b>是各大类正确率的等权平均。<b>Fisher</b>是类间散度除以类内散度，计算前将每张图的 FC2 活动向量归一化到单位长度，各大类权重相同。平均激活保留原始幅度，按概念和大类配平。</p>
<p><b>设置与核对</b>　四层的抑制强度依次为 0.015、0.0075、0.015、0.015。Conv5 每个位置的 512 个通道投影到 64 个单元，保留最活跃的至多 12 个；两帧记忆叠加时第一帧保留一半。门值 g 用原验证集定标，各步沿用。FC2 的类别方向由目标自身的预训练分类证据和训练集类别原型估计，活动形状最多混合 0.45g，均值额外乘以 1 − 0.003g。这是末端读出调节，各步的调节结果不回写递归网络。长期编码和类别知识依赖训练，短时 Hebbian 记忆在每次试次重新形成。</p>
<p>两组均沿用原测试照片、三个冻结检查点和同一组预览，逐条件使用各自的目标反应。第 9 步逐图核对了调节前的 FC2 活动、三个检查点的熟悉度分数，并逐种子核对七项调节后指标。曲线为三种子均值，中晚期表显示描述性差异；各时间步相关，不能当作十五次独立复现。本次没有使用新的独立测试样本。</p>
<p><a href="absolute.csv">三条件逐步绝对值</a> · <a href="paired.csv">逐步配对差值</a> · <a href="windows.csv">三个时间窗口</a> · <a href="things_by_seed.csv">THINGS 各种子</a> · <a href="imagenet_by_seed.csv">ImageNet 各种子</a> · <a href="../../priming_sharpen_20261112/expansion_20261118/实验结果.md">后续扩样对照</a></p></main></html>"""
    )
    (DOC / "index.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    build()
