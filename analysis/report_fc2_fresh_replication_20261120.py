"""Plot target steps 5–14 on the fresh FC2 lateral replication cohort."""
import numpy as np
import pandas as pd

from report_fc2_lateral_20261120 import ARM_NAMES, ARM_COLORS, COLORS, HISTORY, MARKERS, SHORT, STYLES, unit
from report_familiarity_timecourse_20261119 import NAMES, plt
from report_pcoder_carry_20261119 import table
from score_fc2_fresh_replication_20261120 import ARMS, CONTRASTS, DATASETS, EARLY_DOC, METRICS, OUT, read

GEOMETRY = ("within", "between", "fisher")
DECODE = ("auc", "ba", "margin", "mvpa_auc", "mvpa_ba")
WINDOWS = {5: "中期 5–9", 10: "晚期 10–14"}
STEPS = tuple(range(5, 15))
COMPARISONS = {**{a: ARM_NAMES[a] for a in ARMS},
               **{a: f"{ARM_NAMES[b]}减{ARM_NAMES[c]}" for a, (b, c) in CONTRASTS.items()}}
ARM_COLORS = dict(zip(("on", "lateral", "scale", "shuffled"), ARM_COLORS, strict=True))


def image_link(path, title):
    return f'<figure><a href="{path}"><img src="{path}" alt="{title}" loading="lazy"></a></figure>'


def axes_style(ax, zero):
    ax.axvspan(4.8, 9.5, color="#eaf3f7", zorder=-4)
    ax.axvspan(9.5, 14.2, color="#f6f7f9", zorder=-4)
    ax.set_xlim(4.8, 14.2)
    ax.set_xticks(STEPS)
    if zero:
        ax.axhline(0, color="#738591", linewidth=.75)
    ax.grid(axis="y", color="#d7e0e5", linewidth=.6)
    ax.tick_params(labelsize=8)


def render_grid(filename, title, metrics, columns, *, paired):
    fig, axes = plt.subplots(len(metrics), len(columns), figsize=(11.3, 1.76 * len(metrics) + 1.10),
                             sharex=True, sharey="row")
    for col, (heading, series) in enumerate(columns):
        for row, m in enumerate(metrics):
            ax = axes[row, col]
            axes_style(ax, zero=paired)
            if row == 0:
                ax.set_title(heading, loc="left", fontweight="bold", fontsize=11)
            if row == len(metrics) - 1:
                ax.set_xlabel("目标出现后的递归步")
            if col == 0:
                ax.set_ylabel(f"{SHORT[m]}\n{unit(m, paired)}", fontsize=9)
            for i, (label, color, data) in enumerate(series):
                df = data.loc[(data.metric == m) & (data.step.isin(STEPS))].sort_values("step")
                np.testing.assert_array_equal(df.step, STEPS)
                ax.plot(df.step, df.estimate, color=color, linestyle=STYLES[i], marker=MARKERS[i],
                        markersize=3, mfc="white", linewidth=1.55, label=label)
    fig.suptitle(title, x=.121, y=.987, ha="left", fontsize=15, fontweight="bold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(.112, .976 if len(metrics) >= 4 else .965), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.131, right=.982, top=.88 if len(metrics) < 4 else .899,
                        bottom=.071, wspace=.15, hspace=.29)
    fig.savefig(OUT / filename, dpi=145, facecolor="white")
    plt.close(fig)
    return image_link(filename, title)


def effect_plot(points, metrics, key):
    columns = []
    for ds in DATASETS:
        subset = points[(points.dataset == ds) & (points.reference == "same_concept")]
        columns.append((NAMES[ds], [(ARM_NAMES[arm], ARM_COLORS[arm], subset[subset.comparison == arm])
                                    for arm in ARMS]))
    suffix = "表征几何" if key == "geometry" else "固定解码与逐步 MVPA"
    return render_grid(f"paired_{key}.png", f"同图相对同概念异图 · {suffix}", metrics, columns, paired=True)


def absolute_plot(absolute, ds, metrics, key):
    subset = absolute[absolute.dataset == ds]
    columns = []
    for arm in ARMS:
        series = [(name, COLORS[i], subset[(subset.arm == arm) & (subset.history == h)])
                  for i, (h, name) in enumerate(HISTORY.items())]
        columns.append((ARM_NAMES[arm], series))
    suffix = "表征几何及平均活动" if key == "geometry" else "固定解码与逐步 MVPA"
    return render_grid(f"{ds}_absolute_{key}.png", f"{NAMES[ds]} · 三种预览的绝对{suffix}",
                       metrics, columns, paired=False)


def decoding_plots(points):
    refs = (("different", "异类预览"), ("same_concept", "同概念异图"))
    steps = np.arange(15)
    fig, axes = plt.subplots(len(DECODE), 2, figsize=(11, 10.6), sharex=True, sharey="row")
    columns = []
    for ref, label in refs:
        for col, ds in enumerate(DATASETS):
            block = points[(points.reference == ref) & (points.dataset == ds)]
            columns.append((f"{NAMES[ds]} · 对{label}", [
                (ARM_NAMES[arm], ARM_COLORS[arm], block[block.comparison == arm]) for arm in ARMS]))
            for row, metric in enumerate(DECODE):
                ax = axes[row, col]
                data = block[(block.comparison == "lateral") & (block.metric == metric)].sort_values("step")
                np.testing.assert_array_equal(data.step, steps)
                primary = ref == "different"
                color = "#195e88" if primary else "#a25535"
                if primary:
                    ax.axvspan(4.5, 9.5, color="#edf3f6", zorder=-3)
                    ax.axhline(0, color="#768791", lw=.8)
                    ax.grid(axis="y", lw=.6, color="#d8e0e6")
                ax.plot(steps, data.estimate, color=color, lw=2 if primary else 1.45,
                        ls="-" if primary else "--", marker="o" if primary else "s", ms=3.7,
                        mfc="white", label=f"同图 − {label}", zorder=4 if primary else 3)
                ax.fill_between(steps, data.low.to_numpy(), data.high.to_numpy(), color=color, alpha=.09, lw=0)
                ax.set_xticks([0,2,4,6,8,10,12,14])
                if row == 0:
                    ax.set_title(NAMES[ds], loc="left", fontweight="bold")
                if col == 0:
                    ax.set_ylabel(f"{SHORT[metric]}\n{unit(metric, True)}")
                if row == len(DECODE)-1:
                    ax.set_xlabel("目标出现后的递归步")
    title = "真实侧向连接下，同图相对异类和同概念异图的解码差异"
    fig.suptitle(title, x=.113, y=.984, ha="left", fontweight="bold")
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(.105,.96), ncol=2, frameon=False)
    fig.subplots_adjust(left=.12, right=.98, top=.898, bottom=.057, wspace=.13, hspace=.26)
    fig.savefig(OUT / "paired_decoding.png", dpi=145, facecolor="white")
    plt.close(fig)
    return (image_link("paired_decoding.png", title) +
            '<p>实线为同图减异类预览，虚线为同图减同概念异图。正值表示同图的读出指标更高。色带为各步配对 95% 区间，浅蓝底标出第 5–9 步。全程沿用原熟悉度门。BA 为各大类正确率的平均。</p>' +
            '<h3>中晚期的真实连接、同均值缩放与打乱连接</h3>' +
            render_grid("paired_decoding_controls.png", "中晚期解码 · 左两列为异类参照，右两列为同概念异图参照",
                        DECODE, columns, paired=True))


def fisher_controls(points):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10.8, 3.5))
    for (ds, name), ax in zip(NAMES.items(), axes, strict=True):
        axes_style(ax, True)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_xlabel("目标出现后的递归步")
        for i, (key, pair) in enumerate(CONTRASTS.items()):
            p = points.loc[(points.dataset == ds) & (points.reference == "same_concept") &
                           (points.metric == "fisher") & (points.comparison == key) &
                           points.step.isin(STEPS)].sort_values("step")
            np.testing.assert_array_equal(p.step, STEPS)
            color = ("#b14423", "#4c7a5e")[i]
            x, y, lo, hi = (p[c].to_numpy() for c in ("step", "estimate", "low", "high"))
            ax.plot(x, y, color=color, linestyle=STYLES[i], label=f"真实连接减{ARM_NAMES[pair[1]]}")
            ax.fill_between(x, lo, hi, color=color, alpha=.12)
    axes[0].set_ylabel("Fisher 效应的组间差（百分点）")
    fig.suptitle("真实连接的额外同图 Fisher 效应", x=.117, y=.985, ha="left", fontsize=14, fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(.984, .986), ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.117, right=.983, top=.81, bottom=.165, wspace=.13)
    fig.savefig(OUT / "fisher_controls.png", dpi=145, facecolor="white")
    plt.close(fig)
    return image_link("fisher_controls.png", "真实连接相对两种对照的逐步 Fisher 效应变化与配对区间")


def ci(row):
    return f"{row.estimate:+.3f}［{row.low:+.3f}，{row.high:+.3f}］"


GAIN_NAMES = {"explicit": "旧 · 显式类别增益", "lateral": "新 · 稀疏侧向增益"}


def gain_figure(columns, metrics, filename, title, paired):
    fig, axes = plt.subplots(len(metrics), len(columns), sharex=True, sharey="row",
                             figsize=(10.1, 1.62 * len(metrics) + 1.25))
    for j, (label, frame) in enumerate(columns):
        axes[0, j].set_title(label, loc="left", fontsize=11, fontweight="bold")
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            ax.axvspan(4.5, 9.5, color="#edf3f6", zorder=-3)
            ax.axvspan(9.5, 14.2, color="#f7f4ed", zorder=-3)
            if paired:
                ax.axhline(0, color="#778690", lw=.7)
            for arm, name in GAIN_NAMES.items():
                data = frame[(frame.metric == metric) & (frame.comparison == arm)].sort_values("step")
                np.testing.assert_array_equal(data.step, range(15))
                old = arm == "explicit"
                ax.plot(data.step, data.estimate, label=name, color=ARM_COLORS["on" if old else "lateral"],
                        marker="s" if old else "o", markersize=3.3, mfc="white",
                        linestyle="--" if old else "-", lw=1.55)
            ax.set(xlim=(-.2, 14.2), xticks=range(0, 15, 2))
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=.23)
            if not j:
                ax.set_ylabel(f"{SHORT[metric]}\n{unit(metric, paired)}", fontsize=9)
        axes[-1, j].set_xlabel("目标出现后的递归步")
    fig.suptitle(title, x=.12, ha="left", fontsize=15, fontweight="bold", y=.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", ncol=2,
               bbox_to_anchor=(.11, .953), frameon=False)
    fig.subplots_adjust(left=.13, right=.98, bottom=.074, top=.86, hspace=.27, wspace=.15)
    fig.savefig(OUT / filename, dpi=145, facecolor="white")
    plt.close(fig)
    return image_link(filename, title)


def raw_preview_figure(absolute, metrics, filename, title):
    columns = [(ds, arm) for ds in DATASETS for arm in GAIN_NAMES]
    fig, axes = plt.subplots(len(metrics), len(columns), sharex=True,
                             figsize=(13.2, 1.65 * len(metrics) + 1.1))
    raw_units = {"mean": "模型单位", "fisher": "比值", "within": "平方距离", "between": "平方距离",
                 "auc": "0–1", "mvpa_auc": "0–1", "ba": "%", "mvpa_ba": "%", "margin": "分类器分数"}
    styles = {"exact": ("同图", "#1463a0", "-", "o"), "different": ("异类预览", "#be501c", "--", "s"),
              "same_concept": ("同概念异图", "#427460", ":", "^")}
    for col, (ds, arm) in enumerate(columns):
        subset = absolute[(absolute.dataset == ds) & (absolute.comparison == arm)]
        axes[0, col].set_title(f"{NAMES[ds]}\n{GAIN_NAMES[arm]}", fontsize=11, loc="left")
        for i, metric in enumerate(metrics):
            ax = axes[i, col]
            if col % 2:
                ax.sharey(axes[i, col - 1])
            ax.axvspan(9.5, 14.2, facecolor="#f5f1e8", zorder=-3)
            for history, (label, color, style, marker) in styles.items():
                part = subset[(subset.metric == metric) & (subset.history == history)].sort_values("step")
                np.testing.assert_array_equal(part.step, range(15))
                y = part.estimate / 100 if metric in ("auc", "mvpa_auc") else part.estimate
                ax.plot(part.step, y, color=color, ls=style, marker=marker, lw=1.45, ms=3,
                        markerfacecolor="white", label=label)
            ax.set(xlim=(-.25, 14.25), xticks=(0, 3, 6, 9, 12, 14))
            ax.tick_params(labelsize=8, labelleft=not col % 2)
            ax.grid(axis="y", color="#dce2e4", linewidth=.6)
            if col == 0:
                ax.set_ylabel(f"{SHORT[metric]}\n（{raw_units[metric]}）", fontsize=9)
        axes[-1, col].set_xlabel("目标出现后的递归步", fontsize=9)
    fig.suptitle(title, x=.08, y=.988, ha="left", fontsize=15, fontweight="bold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=3, frameon=False,
               bbox_to_anchor=(.986, .989), loc="upper right", fontsize=10)
    fig.subplots_adjust(left=.08, right=.988, top=.864, bottom=.071, wspace=.21, hspace=.22)
    fig.savefig(OUT / filename, dpi=150, facecolor="white")
    plt.close(fig)
    return image_link(filename, title)


def gain_section():
    points = pd.read_csv(OUT / "gain_points.csv")
    idx = pd.read_csv(OUT / "gain_windows.csv").set_index(["dataset", "reference", "start", "comparison", "metric"])
    seeds = pd.read_csv(OUT / "gain_by_seed.csv")
    absolute = seeds.groupby(["dataset", "history", "arm", "step"], as_index=False)[list(METRICS)].mean()
    absolute = absolute.rename(columns={"arm": "comparison"}).melt(
        id_vars=["dataset", "history", "comparison", "step"], var_name="metric", value_name="estimate")
    parts = ['<h2 id="gain">新旧 FC2 增益怎样改变重复效应？</h2>',
             '<p>旧方法根据目标反应估计各类别的概率，用相应的类别原型调节 FC2 神经元。'
             '新方法用训练折中神经元的共活动建立稀疏侧向连接，测试时由活跃神经元影响与其相连的神经元。'
             '两种方法使用相同的无噪声目标、预览历史、PCoder carry 和熟悉度门。每张目标、每个时间步的平均激活也匹配。'
             '固定解码共用同一个分类器。逐步 MVPA 在相同概念划分上为各条件分别拟合分类器。</p>',
             '<h3 id="raw">三种预览的原始时间曲线</h3><p>每格画出三种预览各自的测量值，每个点是三个冻结检查点的均值。'
             '同一数据集的两列共用纵轴刻度，可直接比较新旧增益。蓝线是同图，橙线是异类预览，绿线是同概念异图。'
             '米色底标出晚期第 10–14 步。AUC 使用 0–1 量纲，平衡准确率（BA）使用百分数。</p>',
             raw_preview_figure(absolute, ("mean", "fisher", "auc", "mvpa_auc"), "gain_raw_previews.png",
                                "三种预览的时间过程 · FC2 激活、Fisher 与类别解码"),
             '<details><summary>展开类内与类间散度、平衡准确率及决策间隔的原始曲线</summary>',
             raw_preview_figure(absolute, ("within", "between", "ba", "mvpa_ba", "margin"),
                                "gain_raw_previews_detail.png", "三种预览的时间过程 · 类别散度与分类表现"),
             '</details><h3>同图与其他预览的差值</h3>',
             '<p><strong>同图相对异类预览，第 10–14 步的 Fisher 增幅在 THINGS 从 8.57% 降至 1.87%，'
             '在 ImageNet 从 13.83% 降至 3.26%。</strong>换用侧向连接后，同类目标的反应更集中，但类别中心拉开的幅度小得多。'
             '固定解码的宏 AUC 优势也缩小。逐步 MVPA 的结果因数据集而异，THINGS 的新旧差值区间跨零，'
             'ImageNet 的同图效应从 +0.58 变为 −1.15 个百分点，新减旧为 −1.72［−3.74，−0.28］个百分点。</p>',
             '<p>先看同图减异类预览。正的 AUC 差表示看到同一张图后，分类器对类别的排序更准。'
             '两条线之间的距离表示换用侧向增益后，这个预览效应改变了多少。浅蓝底为第 5–9 步，米色底为第 10–14 步。</p>']
    for ref, ref_name in (("different", "异类预览"), ("same_concept", "同概念异图")):
        if ref == "same_concept":
            parts.append('<details><summary>更严格的参照 · 同图与同概念异图</summary>')
        cols = [(NAMES[d], points[(points.dataset == d) & (points.reference == ref)]) for d in DATASETS]
        for key, fields, label in (("decoding", DECODE, "固定解码与逐步 MVPA"),
                                   ("geometry", ("mean", *GEOMETRY), "平均激活与表征几何")):
            parts.append(gain_figure(cols, fields, f"gain_{ref}_{key}.png",
                                     f"同图相对{ref_name} · {label}", True))
        parts.append('<p>下表列出预览效应从旧方法到新方法的变化。括号内为新减旧的配对 95% 区间。'
                     'Fisher 效应用相对变化百分比，AUC 和 BA 用百分点，间隔用分类器分数。</p>')
        rows = []
        for ds in DATASETS:
            for start in (5, 10):
                for m in ("within", "between", "fisher", *DECODE):
                    a, b, delta = [idx.loc[(ds, ref, start, k, m)] for k in (*GAIN_NAMES, "lateral_minus_explicit")]
                    rows.append([NAMES[ds], WINDOWS[start], SHORT[m], f"{a.estimate:+.3f}",
                                 f"{b.estimate:+.3f}", ci(delta)])
        parts.append('<details><summary>中期和晚期的配对数值</summary>' +
                     table(["数据集", "时间窗", "指标", "旧增益", "新增益", "新减旧［95% 区间］"], rows) + '</details>')
        if ref == "same_concept":
            parts.append('</details>')
    parts.append('<details><summary>三种预览条件下，两种增益各自的绝对值</summary>')
    for ds in DATASETS:
        cols = [(name, absolute[(absolute.dataset == ds) & (absolute.history == h)]) for h, name in HISTORY.items()]
        for key, fields, label in (("decoding", DECODE, "固定解码与逐步 MVPA"),
                                   ("geometry", ("mean", *GEOMETRY), "平均激活与表征几何")):
            parts.append(gain_figure(cols, fields, f"gain_{ds}_absolute_{key}.png", f"{NAMES[ds]} · {label}", False))
    parts.append('</details><p>新旧对照数据 · <a href="gain_by_seed.csv">各预览逐种子绝对值</a> · '
                 '<a href="gain_points.csv">逐步配对效应</a> · <a href="gain_windows.csv">三个时间窗</a> · '
                 '<a href="gain_absolute_contrasts.csv">各预览的新减旧</a></p>')
    return "\n".join(parts)


def windows_table(index, ds, reference):
    fields = (*GEOMETRY, *DECODE)
    rows = []
    for start, name in WINDOWS.items():
        for arm, label in COMPARISONS.items():
            rows.append([name, label] + [ci(index.loc[(ds, reference, start, arm, m)]) for m in fields])
    return table(["时间窗", "处理或对比"] + [SHORT[m] for m in fields], rows)


def read_results():
    parts = {}
    for kind in ("by_seed", "pointwise_intervals", "windows", "absolute_contrasts"):
        frame = pd.concat([pd.read_csv(OUT / f"{ds}_{kind}.csv") for ds in DATASETS], ignore_index=True)
        assert np.isfinite(frame.select_dtypes("number").to_numpy()).all()
        parts[kind] = frame
        frame.to_csv(OUT / f"{kind}.csv", index=False)
    a = parts["by_seed"].melt(id_vars=["dataset", "arm", "seed", "history", "step"],
                               value_vars=list(METRICS), var_name="metric", value_name="estimate")
    parts["absolute"] = a.groupby(["dataset", "arm", "history", "step", "metric"], as_index=False).estimate.mean()
    parts["absolute"].to_csv(OUT / "absolute.csv", index=False)
    return parts


def build():
    data = read_results()
    idx = data["windows"].set_index(["dataset", "reference", "start", "comparison", "metric"])
    abs_idx = data["absolute_contrasts"].set_index(["dataset", "history", "start", "comparison", "metric"])
    early = read(EARLY_DOC / "fresh_inventory.json")
    old = pd.read_csv(OUT.parent / "intervals.csv").set_index(["dataset", "reference", "start", "comparison", "metric"])
    html = ["""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>冻结 FC2 侧向连接的中晚期复现</title>
<style>*{box-sizing:border-box}body{margin:0;background:#edf3f6;color:#203543;font:16px/1.75 "Microsoft YaHei",sans-serif}
main{max-width:1150px;margin:auto;background:white;padding:36px clamp(16px,4vw,46px) 60px}h1,h2,h3{line-height:1.4}
h2{border-top:2px solid #285975;padding-top:16px;margin-top:36px}a,summary{color:#175a80}p{max-width:94ch}
figure{margin:18px 0}img{display:block;width:100%;height:auto}nav a{margin-right:18px}summary{cursor:pointer;font-weight:bold}
details{padding:8px 0}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap}
th,td{padding:7px 8px;border-bottom:1px solid #d7e3e9;text-align:right}th{background:#eef4f7}th:first-child,td:first-child{text-align:left}</style></head>
<body><main><nav><a href="../index.html">上一批侧向连接结果</a><a href="../early_gate/index.html">同批照片的早期门值实验</a>
<a href="#gain">新旧 FC2 增益</a><a href="#things">THINGS</a><a href="#imagenet">ImageNet</a><a href="#data">数据</a></nav>
<h1>冻结 FC2 侧向连接的中晚期复现</h1>
<p><strong>两套数据都复现了晚期同图特异的 Fisher 增益，真实连接均高于同均值缩放和打乱连接。</strong>
THINGS 的类内收紧更明确，真实连接组内的类间散度差区间跨零。ImageNet 则同时看到类内收紧和类间拉开。
两套数据晚期额外的固定宏 AUC 与逐步 MVPA 宏 AUC 区间均跨零。ImageNet 的固定决策间隔提高，
逐步 MVPA 平衡准确率相对同均值缩放下降。</p>
<p>保留 PCoder carry、原 Conv5 熟悉度门以及冻结 FC2 k32 连接。真实连接、同均值缩放和打乱连接共用逐图逐步活动预算。
三组使用同批无噪声目标、同样的预览历史和三个冻结检查点。新批次与最初用于侧向连接诊断的图像批次分开，
也是上一轮第 0–4 步使用过的照片。此次把目标轨迹补到第 14 步并核对早期前缀。</p>
<p>主要看第 10–14 步同图相对同概念异图的 Fisher 效应，以及这一效应是否同时超过同均值缩放和打乱连接。
主图浅蓝底是第 5–9 步，浅灰底是第 10–14 步。曲线逐步，汇总表先逐种子、逐步计算同图效应，再平均。</p>""",
            gain_section(),
            '<h2 id="decoding">同图相对异类预览的固定解码与逐步 MVPA</h2>',
            decoding_plots(data["pointwise_intervals"]),
            effect_plot(data["pointwise_intervals"], GEOMETRY, "geometry"),
            fisher_controls(data["pointwise_intervals"]),
            ]
    for ds in DATASETS:
        audit = read(OUT / f"{ds}_audit.json")
        p = idx.loc[(ds, "same_concept", 10, "lateral", "fisher")]
        c = [idx.loc[(ds, "same_concept", 10, a, "fisher")] for a in CONTRASTS]
        c_old = [old.loc[(ds, "same_concept", 10, a, "fisher")]
                 for a in CONTRASTS]
        html.append(f'<h2 id="{ds}">{NAMES[ds]}</h2><p>{audit["targets"]} 张目标，{audit["concepts"]} 个概念。'
                    f'逐步 MVPA 使用 {len(audit["mvpa_categories"])} 个大类、{audit["mvpa_targets"]} 张目标。'
                    f'晚期真实连接的同图 Fisher 相对变化为 {ci(p)}%。相对同均值缩放和打乱连接的额外效应分别为'
                    f' {ci(c[0])} 和 {ci(c[1])} 个百分点。上一批的对应点估计为 {c_old[0].estimate:+.3f} 和'
                    f' {c_old[1].estimate:+.3f} 个百分点。</p>')
        html.append("<h3>同图相对异类预览</h3>")
        html.append(windows_table(idx, ds, "different"))
        html.append("<h3>同图相对同概念异图</h3>")
        html.append(windows_table(idx, ds, "same_concept"))
        html.append("<details><summary>晚期各预览各自的变化</summary>")
        rows = []
        for h, name in HISTORY.items():
            for key, pair in CONTRASTS.items():
                fields = (*GEOMETRY, "auc", "mvpa_auc")
                rows.append([name, COMPARISONS[key]] + [ci(abs_idx.loc[(ds, h, 10, key, m)]) for m in fields])
        html.append(table(["预览", "对比", "类内散度", "类间散度", "Fisher", "固定宏 AUC", "MVPA 宏 AUC"], rows) + "</details>")
        html.append("<details><summary>三种预览的绝对时间曲线</summary>")
        html.append(absolute_plot(data["absolute"], ds, ("mean", *GEOMETRY), "geometry"))
        html.append(absolute_plot(data["absolute"], ds, DECODE, "decoding"))
        html.append("</details>")
        counts = "、".join(f"{name} {n}" for name, n in early[ds]["concepts"].items())
        html.append(f"<p>概念组成包括 {counts}。</p>")
    html.append("""<h2 id="data">统计与数据</h2>
<p>窗口区间来自 4,000 次类内配对概念块重采样。MVPA 在每处理、预览、时间步重新按概念做六折，区间以得到的外折预测为条件。
THINGS 植物仅一个概念，计入几何和固定分类器结果，逐步 MVPA 使用其余四类。ImageNet 使用全部四类。
Fisher 为单位长度表征的类间散度除以类内散度。主曲线的几何指标按同图相对参照的百分比变化计算，
固定及逐步分类器的宏 AUC 和平衡准确率用百分点，决策间隔用分数差。各预览各自的几何变化保留原始指标单位。
阴影是逐步 95% 配对区间。ImageNet 少数类间散度点估计略高于百分位区间上界。</p>
<p><a href="by_seed.csv">逐种子逐步表</a> · <a href="absolute.csv">逐步绝对值</a> ·
<a href="pointwise_intervals.csv">逐步配对效应与区间</a> · <a href="windows.csv">窗口汇总</a> ·
<a href="absolute_contrasts.csv">按预览的绝对对照</a> · <a href="things_audit.json">THINGS 核对</a> ·
<a href="imagenet_audit.json">ImageNet 核对</a></p>
<pre>python -B analysis/collect_fc2_fresh_replication_20261120.py things
python -B analysis/score_fc2_fresh_replication_20261120.py things
python -B analysis/collect_fc2_fresh_replication_20261120.py imagenet
python -B analysis/score_fc2_fresh_replication_20261120.py imagenet
python -B analysis/compare_fc2_fresh_gains_20261120.py
python -B analysis/report_fc2_fresh_replication_20261120.py</pre></main></body></html>""")
    (OUT / "index.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    build()
