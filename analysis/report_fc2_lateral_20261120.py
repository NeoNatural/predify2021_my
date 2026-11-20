"""Render the paired four-arm FC2 lateral replay with original time-course conventions."""
from html import escape

import numpy as np
import pandas as pd

from report_familiarity_timecourse_20261119 import COLORS, HISTORY, METRICS as LABELS, NAMES, POINTS, plt
from report_pcoder_carry_20261119 import table
from score_fc2_lateral_20261120 import ARMS, COMPARISONS, DESTINATION as OUT, METRICS, RELATIVE, read

TITLE = "FC2 稀疏侧向连接能否保留同图特异的类别区分？"
ARM_NAMES = {
    "on": "显式类别调节", "lateral": "真实侧向连接",
    "scale": "同均值缩放", "shuffled": "打乱连接",
}
DIFF_NAMES = {f"{a}_minus_{b}": f"{ARM_NAMES[a]}减{ARM_NAMES[b]}" for a, b in COMPARISONS}
LABELS = dict(LABELS, active_pct="非零单元比例", weak_pct="弱非零单元比例")
SHORT = dict(mean="平均激活", within="类内散度", between="类间散度", fisher="Fisher",
             margin="固定间隔", auc="固定宏 AUC", ba="固定 BA", mvpa_auc="MVPA 宏 AUC",
             mvpa_ba="MVPA BA", active_pct="非零比例", weak_pct="弱非零比例")
MAIN = ("fisher", "auc", "ba", "mvpa_auc", "mvpa_ba")
POP = ("mean", "within", "between", "margin", "active_pct", "weak_pct")
WINDOWS = {5: "中期 5–9", 10: "晚期 10–14"}
ARM_COLORS = ("#164e7c", "#b14423", "#667888", "#4f8062")
STYLES = ("-", "--", ":", "-.")
MARKERS = ("o", "s", "^", "D")


def unit(metric, paired):
    if metric in POINTS or metric in ("active_pct", "weak_pct"):
        return "百分点" if paired else "%"
    if metric == "margin":
        return "分数差" if paired else "分数"
    if paired:
        return "相对差（%）"
    return {"mean": "模型单位", "within": "平方距离", "between": "平方距离", "fisher": "比值"}[metric]


def axes_style(ax, paired):
    ax.axvspan(4.5, 9.5, color="#eaf2f7", zorder=-3)
    ax.axvspan(9.5, 14.35, color="#f2f5f7", zorder=-4)
    if paired:
        ax.axhline(0, color="#7c8790", linewidth=.7)
    ax.grid(axis="y", color="#d8e1e7", linewidth=.6)
    ax.set_xticks(range(0, 15, 2))
    ax.set_xlim(-.25, 14.3)
    ax.tick_params(labelsize=8)
    ax.locator_params(axis="y", nbins=4)


def line(ax, rows, metric, label, color, style, marker):
    rows = rows.sort_values("step")
    np.testing.assert_array_equal(rows.step, np.arange(15))
    ax.plot(rows.step, rows[metric], color=color, linestyle=style, marker=marker,
            markersize=3, mfc="white", linewidth=1.5, label=label)


def curves(data, ds, metrics, group, paired):
    fig, axes = plt.subplots(len(metrics), len(ARMS), figsize=(14, 1.74 * len(metrics) + 1.18),
                             sharey="row", sharex=True)
    for col, arm in enumerate(ARMS):
        subset = data[(data.dataset == ds) & (data.arm == arm)]
        for i, metric in enumerate(metrics):
            ax = axes[i, col]
            axes_style(ax, paired)
            for j, (history, name) in enumerate(HISTORY.items()):
                if paired and history == "exact":
                    continue
                label = ("同图相对" if paired else "") + name
                line(ax, subset[subset.history == history], metric, label, COLORS[j], STYLES[j], MARKERS[j])
            if i == 0:
                ax.set_title(ARM_NAMES[arm], fontsize=11, loc="left", fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{SHORT[metric]}\n{unit(metric, paired)}", fontsize=9)
            if i == len(metrics) - 1:
                ax.set_xlabel("目标出现后的递归步")
    header = f"{NAMES[ds]} · " + ("同图相对两种对照" if paired else "三种预览的绝对时间曲线")
    fig.suptitle(header, x=.075, y=.985, ha="left", fontsize=15, fontweight="bold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(.068, .973), ncol=3, frameon=False)
    fig.subplots_adjust(left=.082, right=.985, top=.908, bottom=.053, wspace=.11, hspace=.27)
    path = f'{ds}_{group}_{"paired" if paired else "absolute"}.png'
    fig.savefig(OUT / path, dpi=140, facecolor="white")
    plt.close(fig)
    return picture(path, header)


def picture(path, alt):
    name = escape(alt)
    return f'<figure><a href="{path}"><img src="{path}" loading="lazy" alt="{name}"></a></figure>'


def overview(paired):
    fig, axes = plt.subplots(len(MAIN), len(NAMES), figsize=(10.5, 10.4), sharex=True, sharey="row")
    for col, (ds, name) in enumerate(NAMES.items()):
        subset = paired[(paired.dataset == ds) & (paired.history == "same_concept")]
        for i, metric in enumerate(MAIN):
            ax = axes[i, col]
            axes_style(ax, True)
            for j, arm in enumerate(ARMS):
                line(ax, subset[subset.arm == arm], metric, ARM_NAMES[arm], ARM_COLORS[j], STYLES[j], MARKERS[j])
            if i == 0:
                ax.set_title(name, fontsize=12, loc="left", fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{SHORT[metric]}\n{unit(metric, True)}", fontsize=9)
            if i == len(MAIN) - 1:
                ax.set_xlabel("目标出现后的递归步")
    fig.suptitle("同图相对同概念异图的类别区分时间曲线", x=.123, y=.985, ha="left", fontsize=16, fontweight="bold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(.113, .961), ncol=4, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.134, right=.979, top=.904, bottom=.055, wspace=.13, hspace=.3)
    fig.savefig(OUT / "overview.png", dpi=150, facecolor="white")
    plt.close(fig)
    return picture("overview.png", "同图相对同概念异图的 Fisher、固定分类器与逐步 MVPA")


def ci_value(row):
    return f"{row.estimate:+.3f}［{row.low:+.3f}，{row.high:+.3f}］"


def table_windows(index, ds, reference):
    cols = ("within", "between", *MAIN, "margin", "active_pct")
    lines = []
    for start, name in WINDOWS.items():
        for arm, label in (ARM_NAMES | DIFF_NAMES).items():
            lines.append([name, label] + [ci_value(index.loc[(ds, reference, start, arm, m)]) for m in cols])
    headers = ["时间窗", "条件或组间差"] + [SHORT[m] for m in cols]
    return table(headers, lines)


def absolute_gain(index, ds, absolute):
    lines = []
    for start, name in WINDOWS.items():
        for h, hist in HISTORY.items():
            val = absolute.loc[(ds, "lateral", h, start), "active_pct"]
            base = absolute.loc[(ds, "scale", h, start), "active_pct"]
            fields = ("within", "between", "fisher", "auc", "mvpa_auc", "active_pct", "weak_pct")
            lines.append([name, hist, f"{base:.2f} → {val:.2f}"] +
                         [ci_value(index.loc[(ds, h, start, "lateral_minus_scale", m)]) for m in fields])
    return table(["时间窗", "预览", "非零比例（%）", "类内散度", "类间散度", "Fisher",
                  "固定宏 AUC", "MVPA 宏 AUC", "非零（百分点）", "弱非零（百分点）"], lines)


def prepare_tables():
    by_seed = pd.concat([pd.read_csv(OUT / f"{ds}_by_seed.csv") for ds in NAMES], ignore_index=True)
    ci = pd.concat([pd.read_csv(OUT / f"{ds}_intervals.csv") for ds in NAMES], ignore_index=True)
    abs_ci = pd.concat([pd.read_csv(OUT / f"{ds}_absolute_contrasts.csv") for ds in NAMES], ignore_index=True)
    points = pd.concat([pd.read_csv(OUT / f"{ds}_pointwise_intervals.csv") for ds in NAMES], ignore_index=True)
    keys = ["dataset", "arm", "seed", "step"]
    exact = by_seed[by_seed.history == "exact"].set_index(keys)
    effects = []
    for h in ("same_concept", "different"):
        ref = by_seed[by_seed.history == h].set_index(keys)
        vals = {m: (100 * (exact[m] / ref[m] - 1) if m in RELATIVE else exact[m] - ref[m]) for m in METRICS}
        effects.append(pd.DataFrame(vals).assign(history=h).reset_index())
    effects = pd.concat(effects, ignore_index=True)
    key = ["dataset", "arm", "history", "step"]
    absolute = by_seed.groupby(key, as_index=False)[list(METRICS)].mean()
    paired = effects.groupby(key, as_index=False)[list(METRICS)].mean()
    key[-1] = "start"
    windows = paired.assign(start=paired.step // 5 * 5).groupby(key)[list(METRICS)].mean()
    absolute_windows = absolute.assign(start=absolute.step // 5 * 5).groupby(key)[list(METRICS)].mean()
    for r in ci[ci.comparison.isin(ARMS)].itertuples():
        val = windows.loc[(r.dataset, r.comparison, r.reference, r.start), r.metric]
        np.testing.assert_allclose(r.estimate, val, atol=1e-10)
    paired_index = paired.set_index(["dataset", "arm", "history", "step"])
    for r in points[points.comparison.isin(ARMS)].itertuples():
        val = paired_index.loc[
            (r.dataset, r.comparison, r.reference, r.step), r.metric]
        np.testing.assert_allclose(r.estimate, val, atol=1e-10)
    for path, data in dict(absolute=absolute, paired=paired, by_seed=by_seed, paired_by_seed=effects,
                           windows=windows.reset_index(), intervals=ci, pointwise_intervals=points,
                           absolute_contrasts=abs_ci, absolute_windows=absolute_windows.reset_index()).items():
        data.to_csv(OUT / f"{path}.csv", index=False)
    return (absolute, paired, ci.set_index(["dataset", "reference", "start", "comparison", "metric"]),
            abs_ci.set_index(["dataset", "history", "start", "comparison", "metric"]), absolute_windows)


def summarize(ds, intervals, absolute, audit):
    fisher = [intervals.loc[(ds, "same_concept", t, "lateral", "fisher")] for t in WINDOWS]
    effect = intervals.loc[(ds, "same_concept", 10, "lateral_minus_scale", "fisher")]
    shuffle = intervals.loc[(ds, "same_concept", 10, "lateral_minus_shuffled", "fisher")]
    old = intervals.loc[(ds, "same_concept", 10, "lateral_minus_on", "fisher")]
    within = intervals.loc[(ds, "same_concept", 10, "lateral", "within")]
    between = intervals.loc[(ds, "same_concept", 10, "lateral", "between")]
    fixed = intervals.loc[(ds, "same_concept", 10, "lateral_minus_scale", "auc")]
    mvpa = intervals.loc[(ds, "same_concept", 10, "lateral_minus_scale", "mvpa_auc")]
    a = absolute.loc[(ds, "scale", "exact", 10), "active_pct"]
    b = absolute.loc[(ds, "lateral", "exact", 10), "active_pct"]
    return (
        f'<p>{audit["test_count"]} 张目标、{audit["test_concepts"]} 个测试概念。真实连接的同图 Fisher 效应，'
        f'中期为 {ci_value(fisher[0])}%，晚期为 {ci_value(fisher[1])}%。晚期同图效应的组间差为'
        f'真实连接减同均值缩放 {ci_value(effect)}、减打乱连接 {ci_value(shuffle)}、'
        f'减显式调节 {ci_value(old)} 个百分点。真实连接的晚期类内散度变化为 {ci_value(within)}%，'
        f'类间散度变化为 {ci_value(between)}%，两者共同抬高 Fisher。晚期同图的非零比例由同均值缩放的 '
        f'{a:.2f}% 变为真实连接的 {b:.2f}%。</p>'
        f'<p>相对同均值缩放，真实连接在三种预览下的绝对 Fisher 和固定宏 AUC 都提高。晚期同图特异的'
        f'额外固定宏 AUC 为 {ci_value(fixed)} 个百分点，额外 MVPA 宏 AUC 为 {ci_value(mvpa)} 个百分点，区间均跨零。</p>'
    )


def methods(audits):
    graph_rows, check_rows = [], []
    for ds, audit in audits.items():
        graph = audit["graph"]
        st = graph["statistics"]
        overlap = graph["previous_graph_concept_overlap"]
        graph_rows.append([NAMES[ds], graph["n_train"], graph["concepts"], st["edges"], f'{100*st["density"]:.4f}%',
                           f'{st["degree_mean"]:.2f}', st["degree_max"], st["isolated"], f'{st["rms"]:.6f}',
                           f'{st["variance_floor"]:.6f}', len(overlap["test"])])
        gates = audit["calibration"]
        gate_err = max(s["t9_max_abs_gate_error"] for s in gates["per_seed"].values())
        check_rows.append([NAMES[ds], f'{audit["nonmodulated_t9_max_abs_error"]:.2e}', f'{gate_err:.2e}',
                           f'{max(gates["explicit_replay_max_abs_error"]):.2e}',
                           f'{max(audit["mean_budget"].values()):.2e}'])
    return [
        '<h2 id="methods">侧向连接、门值和统计</h2>',
        '<p>从本轮训练折的干净中央目标画布提取冻结 VGG16 的 4096 维 FC2 ReLU。以全阵列 RMS 尺度 s '
        '归一化，得到逐单元中心化协方差 C。令 λ = max(0.1 × 正方差中位数, 10⁻⁶)，并取 '
        'max(0, Cᵢⱼ / √[(Cᵢᵢ+λ)(Cⱼⱼ+λ)])。清空自身连接后保留互为前 32 的边，按最大行和缩放。'
        '估计只输入训练图的活动矩阵。旧图的训练概念和本轮测试有交集，因此复用规则及常数并重新估计边权。</p>',
        table(["数据", "训练图", "训练概念", "无向边", "密度", "平均度", "最大度", "孤立单元", "RMS",
               "方差下限", "旧图重叠测试概念"], graph_rows),
        '<p>对每张目标和每一步，令 u=x/s、r₀=u，侧向驱动 d=u+0.75Wr。保留与输入中大于 10⁻⁷ '
        '的活动单元数相同的高值配额，做硬阈值。保留的驱动除以 1+2mean(r)，用 0.5 的松弛系数更新 r，'
        '被当前阈值淘汰的单元置零。每个目标步都从 u 重新开始，固定迭代五次。把所得侧向活动变为均值为 1 的方向 q。'
        '打乱组用种子 20261120 固定置换 W 的行列，保留边权和图结构。三种历史和三个模型共用排列。</p>',
        '<p>末端输出 y = mean(x)(1−0.003g)[(1−0.45g) · x/mean(x) + 0.45g · q]。'
        '同均值缩放组取 q=x/mean(x)。显式组直接读取旧缓存。每个目标、历史、时间步和种子都核对活动预算。'
        '图中弱非零指高于零且低于该目标峰值 1% 的单元。即使侧向候选遵守活动配额，混合后也可能合并两份支持集。</p>',
        '<p>完整调节缓存只存输出活动。第 9 步用冻结检查点的原生熟悉度分数和原验证定标直接取门值，'
        '并核对末端调节前活动。其余步用两份缓存的 float64 群体均值从 1−0.003g 反推 g；'
        '恒等试次设为零。再用反推门值重放显式调节并逐单元核对。结果保留了缓存舍入引入的小误差。</p>',
        table(["数据", "t9 调节前最大绝对误差", "t9 反推门值最大绝对差", "显式重放最大绝对误差",
               "活动预算最大相对误差"], check_rows),
        '<p>Fisher 使用单位长度表征的类间散度除以类内散度。固定分类器沿用原线性 SVM。逐步 MVPA '
        '在每组、预览与步分别拟合，六折中每折每大类留出一个概念。95% 区间使用 4,000 次类内配对概念块重采样。'
        '组间差先在种子与步内配对，再取平均。MVPA 区间以六折的外折预测为条件，BA 按概念重加权；'
        '宏 AUC 在同折内比较正负样本，按两个概念的抽中次数乘积重加权后汇总，保持原等折点估计。'
        '重采样不重拟合 SVM，也不重新抽取连接图或打乱排列。少数 ImageNet 点估计偏离百分位区间，反映'
        '重采样分布与全样本估计的偏移。测试为已查看的同一批照片，结论适用于这一配置的机制对照。</p>',
    ]


def build():
    absolute, paired, intervals, abs_ci, abs_windows = prepare_tables()
    audits = {ds: read(OUT / f"{ds}_audit.json") for ds in NAMES}
    html = [f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width, initial-scale=1"><title>{TITLE}</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#eef3f6;color:#1b2e3c;font:16px/1.75 "Microsoft YaHei",sans-serif}}
main{{max-width:1280px;margin:auto;background:#fff;padding:36px clamp(16px,4vw,54px) 56px}}h1,h2,h3{{line-height:1.4}}
h1{{font-size:clamp(27px,4vw,40px)}}h2{{margin-top:42px;padding-top:16px;border-top:2px solid #265e81}}
a{{color:#195d86}}p{{max-width:94ch}}nav a{{margin-right:18px}}figure{{margin:14px 0 26px}}img{{width:100%;height:auto;display:block}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{padding:6px 8px;border-bottom:1px solid #d8e2e9;text-align:right}}
th{{background:#edf4f8}}th:first-child,td:first-child{{text-align:left}}summary{{cursor:pointer;color:#195d86;font-weight:bold;padding:9px 0}}</style></head>
<body><main><nav><a href="../index.html">原末端调节消融</a><a href="#things">THINGS</a><a href="#imagenet">ImageNet</a>
<a href="#methods">方法</a><a href="#data">数据</a></nav><h1>{TITLE}</h1>
<p><strong>两套数据都保留了中晚期的同图特异 Fisher 增益，幅度约为显式类别调节的四分之一。</strong>
真实连接比同均值缩放和打乱连接更强。固定解码的同图优势在中期有小幅改善，晚期差异的区间跨零。逐步 MVPA 的宏 AUC 和平衡准确率未出现同方向的稳定增益。</p>
<p>保留预览结束的 PCoder 状态、试次内 Hebbian 记忆、FC1、四层抑制和冻结的 Conv5 熟悉度门。四组仅替换 FC2 末端的活动重加权方向，逐目标配平平均活动。</p>
<p>主比较为同图减同概念异图的效应在四组之间相差多少。区间见各窗口表。几何和平均激活以相对百分比计，AUC、BA 和非零比例以百分点计，间隔以分数差计。浅蓝底标出第 5–9 步，浅灰底标出第 10–14 步。</p>""",
            overview(paired)]
    for ds, name in NAMES.items():
        html += [f'<h2 id="{ds}">{name}</h2>', summarize(ds, intervals, abs_windows, audits[ds]),
                 '<h3>中晚期同图相对同概念异图</h3>', table_windows(intervals, ds, "same_concept"),
                 '<details><summary>中晚期同图相对异类</summary>',
                 table_windows(intervals, ds, "different"), '</details>',
                 '<details open><summary>按预览查看真实连接相对缩放的绝对变化</summary>',
                 '<p>此表比较同一个预览条件内的两组活动。Fisher 和散度用原始单位，其余按表头。</p>',
                 absolute_gain(abs_ci, ds, abs_windows), '</details>',
                 '<h3>绝对与配对时间曲线</h3>', curves(absolute, ds, MAIN, "decoding", False),
                 curves(paired, ds, MAIN, "decoding", True),
                 '<details><summary>平均激活、散度、固定间隔与活动密度</summary>',
                 curves(absolute, ds, POP, "population", False), curves(paired, ds, POP, "population", True),
                 '</details>']
    html.extend(methods(audits))
    html.append("""<h2 id="data">数据与复现</h2>
<p><a href="absolute.csv">逐步绝对值</a> · <a href="paired.csv">逐步配对差</a> · <a href="by_seed.csv">逐种子逐步数据</a> ·
<a href="intervals.csv">窗口区间</a> · <a href="pointwise_intervals.csv">逐步区间</a> ·
<a href="absolute_contrasts.csv">按历史的绝对差</a> · <a href="things_audit.json">THINGS 审计</a> ·
<a href="imagenet_audit.json">ImageNet 审计</a></p>
<pre>python -B analysis/score_fc2_ablation_20261120.py things --lateral
python -B analysis/score_fc2_ablation_20261120.py imagenet --lateral
python -B analysis/report_fc2_ablation_20261120.py --lateral</pre></main></body></html>""")
    (OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
