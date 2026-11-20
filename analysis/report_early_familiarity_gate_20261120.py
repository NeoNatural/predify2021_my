"""Build the offline report focused on exact vs different in target steps 0–4."""
from html import escape

import numpy as np
import pandas as pd

from early_familiarity_gate_data_20261120 import DATASETS, OUT, STEPS, read
from report_familiarity_timecourse_20261119 import NAMES, plt
from report_pcoder_carry_20261119 import table

BOTH = {"different": "同图相对异类", "same_concept": "同图相对同概念异图"}
LABELS = {
    "mixing": "混合强度差\n百分点", "fisher": "Fisher 相对变化\n%",
    "auc": "固定宏 AUC\n百分点", "mvpa_auc": "MVPA 宏 AUC\n百分点",
    "ba": "固定平衡准确率\n百分点", "mvpa_ba": "逐步 MVPA 平衡准确率\n百分点",
    "within": "类内散度（%）", "between": "类间散度（%）", "margin": "决策间隔",
}


def val(row):
    return f"{row.estimate:+.3f}［{row.low:+.3f}，{row.high:+.3f}］"


def curve_plot(effects, *, changes=False):
    panels = ("mixing", "fisher", "auc", "ba", "margin", "mvpa_auc", "mvpa_ba")
    fig, axes = plt.subplots(len(panels), 2, figsize=(11.5, 13.1), sharex=True, sharey="row")
    for col, ds in enumerate(DATASETS):
        for i, metric in enumerate(panels):
            ax = axes[i, col]
            ax.grid(axis="y", color="#d6e2e8", linewidth=.6)
            ax.axhline(0, color="#738797", linewidth=.75)
            ax.set_xticks(STEPS)
            series = [("original", "原门", "#557184", "--", "o"),
                      ("calibrated", "新门", "#aa4f30", "-", "s")]
            if changes:
                series = [("change", "新门减原门", "#aa4f30", "-", "s")]
            for arm, label, color, line, marker in series:
                sel = effects.loc[(effects.dataset == ds) & (effects.reference == "different") &
                                  (effects.metric == metric) & (effects.comparison == arm) &
                                  (effects.step != "0–4")].sort_values("step")
                step = sel.step.astype(int).to_numpy()
                np.testing.assert_array_equal(step, np.arange(5))
                ax.plot(step, sel.estimate, color=color, linestyle=line, marker=marker, mfc="white",
                        linewidth=1.7, label=label)
                if changes:
                    ax.vlines(step, sel.low, sel.high, color=color, linewidth=1.2)
            if i == 0:
                ax.set_title(NAMES[ds], loc="left", fontweight="bold", fontsize=12)
            if col == 0:
                label = LABELS[metric]
                ax.set_ylabel("Fisher 效应变化\n百分点" if changes and metric == "fisher" else label)
            if i == len(panels) - 1:
                ax.set_xlabel("目标出现后的递归步")
    heading = "同图相对异类预览 · 重新定标带来的变化" if changes else "同图相对异类预览 · 原门与新门"
    fig.suptitle(heading, x=.124, y=.985, ha="left", fontsize=17, fontweight="bold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(.977, .985), ncol=2, frameon=False)
    fig.subplots_adjust(left=.129, right=.978, top=.925, bottom=.072, wspace=.10, hspace=.26)
    path = OUT / ("early_changes.png" if changes else "early_curves.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return f'<figure><a href="{path.name}"><img src="{path.name}" alt="同图相对异类的第 0–4 步混合强度、Fisher 和解码曲线"></a></figure>'


def format_validation(va, ds):
    selec = va.loc[va.dataset == ds]
    averages = selec.groupby(["stage", "step", "history", "arm"]).mean(numeric_only=True)["mean_mixing"].unstack()
    change = (averages["calibrated"] - averages["original"]).unstack()
    rows = []
    for (stage, t), d in change.iterrows():
        rows.append(["留出概念" if stage == "cross_validation" else "全验证折", t,
                     f'{d["exact"]:+.3f}', f'{d["different"]:+.3f}', f'{d["same_concept"]:+.3f}'])
    return table(["验证口径", "递归步", "同图", "异类", "同概念异图"], rows)


def build():
    effects = pd.concat([pd.read_csv(OUT / f"{ds}_effects.csv", dtype={"step": str}) for ds in DATASETS])
    absolute = pd.concat([pd.read_csv(OUT / f"{ds}_absolute_change.csv", dtype={"step": str}) for ds in DATASETS])
    va = pd.concat([pd.read_csv(OUT / f"{ds}_validation.csv") for ds in DATASETS])
    by_seed = pd.concat([pd.read_csv(OUT / f"{ds}_by_seed.csv") for ds in DATASETS])
    idx = effects.set_index(["dataset", "reference", "step", "metric", "comparison"])
    abs_idx = absolute.set_index(["dataset", "history", "step", "metric"])
    inventory = read(OUT / "fresh_inventory.json")
    for suffix, content in (("effects", effects), ("absolute_change", absolute),
                            ("validation", va), ("by_seed", by_seed)):
        content.to_csv(OUT / f"{suffix}.csv", index=False)
    absolute_points = by_seed.groupby(["dataset", "arm", "history", "step", "metric"], as_index=False).estimate.mean()
    absolute_points.to_csv(OUT / "absolute.csv", index=False)
    title = "用异类预览定标 FC2 熟悉度门"
    html = [f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width, initial-scale=1"><title>{title}</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#edf3f6;color:#203543;font:16px/1.75 "Microsoft YaHei",sans-serif}}
main{{max-width:1140px;margin:auto;background:white;padding:36px clamp(16px,4vw,48px) 60px}}h1,h2,h3{{line-height:1.4}}
h2{{margin-top:36px;border-top:2px solid #295b75;padding-top:17px}}a,summary{{color:#175a80}}p{{max-width:94ch}}
figure{{margin:16px 0}}img{{display:block;width:100%;height:auto}}nav a{{margin-right:18px}}summary{{cursor:pointer;font-weight:bold}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{padding:7px 8px;border-bottom:1px solid #d7e3e9;text-align:right}}
th{{background:#eef4f7}}th:first-child,td:first-child{{text-align:left}}</style></head><body><main>
<nav><a href="../index.html">侧向连接复核</a><a href="#things">THINGS</a><a href="#imagenet">ImageNet</a><a href="#data">数据</a></nav>
<h1>{title}</h1>
<p>主比较为目标出现后第 0–4 步的同图与异类预览差异。同概念异图作为次要对照。
两组共用 PCoder carry、四层短时抑制、冻结 Conv5 熟悉度编码器和 FC2 k32 连接。混合上限均为 0.45，平均活动继续按原门值乘以 1−0.003g。</p>
<p>新门仍是单调的截断线性映射。每个种子和时间步在验证图上重新定阈值，使同图平均混合强度与原门相等。
各数据集共用一个斜率倍率，从 0.25、0.5、1、2、4、8 倍中用验证集的六折概念留出选择，目标为扩大同图减异类的平均混合强度差。
在全验证集重定阈值、冻结参数后才运行新留出照片。</p>
<p><strong>新门在 THINGS 第 1 步带来的 Fisher 增益最大，ImageNet 第 3 步有小幅增益。</strong>
第 0–4 步合并后，THINGS 的同图优势扩大，ImageNet 的区间略跨零。两套数据的窗口级解码收益均不明确。
图中竖线为配对概念块重采样 95% 区间。</p>""", curve_plot(effects, changes=True),
            '<details><summary>查看原门与新门的同图优势曲线</summary>', curve_plot(effects), '</details>']
    steps = [str(t) for t in STEPS] + ["0–4"]
    for ds in DATASETS:
        frozen = read(OUT / f"{ds}_calibration.json")
        audit = read(OUT / f"{ds}_audit.json")
        col = len(audit["mvpa_categories"])
        gate_gap = idx.loc[(ds, "different", "0–4", "mixing", "change")]
        fisher = idx.loc[(ds, "different", "0–4", "fisher", "change")]
        auc = idx.loc[(ds, "different", "0–4", "auc", "change")]
        mvpa_auc = idx.loc[(ds, "different", "0–4", "mvpa_auc", "change")]
        html.append(f'<h2 id="{ds}">{NAMES[ds]}</h2><p>斜率保留原来的 {frozen["selected"]:g} 倍。'
                    f'新留出共 {audit["images"]} 张，覆盖 {audit["concepts"]} 个概念。'
                    f'逐步 MVPA 纳入满足六折留一概念的 {col} 个大类、{audit["mvpa_images"]} 张图。</p>')
        html.append(f'<p>第 0–4 步同图相对异类的平均混合强度差变化 {val(gate_gap)} 个百分点。'
                    f'Fisher 差效应变化 {val(fisher)} 个百分点。固定宏 AUC 与逐步 MVPA 宏 AUC 的差效应'
                    f'分别变化 {val(auc)} 和 {val(mvpa_auc)} 个百分点。</p>')
        for reference, name in BOTH.items():
            rows = []
            for step in steps:
                record = [step]
                for metric in ("mixing", "fisher"):
                    for arm in ("original", "calibrated"):
                        r = idx.loc[(ds, reference, step, metric, arm)]
                        record.append(f"{r.estimate:+.3f}")
                    record.append(val(idx.loc[(ds, reference, step, metric, "change")]))
                for metric in ("auc", "mvpa_auc", "ba", "mvpa_ba"):
                    record.append(val(idx.loc[(ds, reference, step, metric, "change")]))
                rows.append(record)
            if reference != "different":
                html.append(f"<details><summary>{name}</summary>")
            else:
                html.append(f"<h3>{name}</h3>")
            html.append(table(["步", "原混合差", "新混合差", "混合差变化", "原 Fisher 效应", "新 Fisher 效应",
                              "Fisher 效应变化", "固定宏 AUC 差变化", "MVPA 宏 AUC 差变化",
                              "固定 BA 差变化", "MVPA BA 差变化"], rows))
            if reference != "different":
                html.append("</details>")
        html.append("<details><summary>两种预览各自的变化与几何分解</summary>")
        rows = []
        for step in steps:
            for h, name in (("exact", "同图"), ("different", "异类")):
                metrics = ("mixing", "within", "between", "fisher", "auc", "mvpa_auc")
                rows.append([step, name] + [val(abs_idx.loc[(ds, h, step, m)]) for m in metrics])
        html.append(table(["步", "预览", "混合强度（百分点）", "类内散度", "类间散度", "Fisher",
                           "固定宏 AUC", "MVPA 宏 AUC"], rows))
        html.append("</details><details><summary>验证折配平与留出概念复算</summary>")
        html.append("<p>表中为新门减原门的平均混合强度，单位为百分点。全验证折逐步拟合阈值；留出概念用其余折拟合。</p>")
        html.append(format_validation(va, ds) + "</details>")
        counts = "、".join(f"{escape(k)} {v}" for k, v in inventory[ds]["concepts"].items())
        html.append(f"<p>新留出概念分布为 {counts}。所有留出照片都排除了已登记的训练、验证、测试和历史图像哈希。"
                    "部分概念曾出现在历史批次，照片是本次新选。</p>")
    html.append("""<h2 id="data">统计与可复现数据</h2>
<p>表中区间为 4,000 次类内配对概念块重采样；先做种子与时间步内配对，再取平均。MVPA 区间以固定外折预测为条件。
THINGS 植物只有一个可选的新照片概念，固定解码与几何含植物；留一概念的逐步 MVPA 在其余四类进行。
绝对变化表中的 Fisher、类内及类间指标保留原单位，时间曲线与效应表中的 Fisher 按同图除以参照再减一表示。</p>
<p><a href="effects.csv">逐步配对效应与区间</a> · <a href="absolute.csv">逐步绝对值</a> ·
<a href="absolute_change.csv">各预览的绝对变化</a> · <a href="by_seed.csv">逐种子</a> ·
<a href="validation.csv">验证折门值</a> · <a href="fresh_inventory.json">刺激库存</a> ·
<a href="things_calibration.json">THINGS 冻结校准</a> · <a href="imagenet_calibration.json">ImageNet 冻结校准</a> ·
<a href="things_audit.json">THINGS 复核</a> · <a href="imagenet_audit.json">ImageNet 复核</a></p>
<pre>python -B analysis/early_familiarity_gate_data_20261120.py things validation
python -B analysis/early_familiarity_gate_fit_20261120.py things fit
python -B analysis/early_familiarity_gate_data_20261120.py things fresh
python -B analysis/early_familiarity_gate_fit_20261120.py things score
# 将 things 换成 imagenet 后复现 ImageNet
python -B analysis/report_early_familiarity_gate_20261120.py</pre></main></body></html>""")
    (OUT / "index.html").write_text("\n".join(html), encoding="utf8")


if __name__ == "__main__":
    build()
