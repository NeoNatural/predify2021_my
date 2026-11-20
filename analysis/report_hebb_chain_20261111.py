"""Summarize the frozen search, layer-wise effects, and clean confirmation."""

from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hebb_chain_20261111 import DATASETS, DOC, HISTORIES, LAYERS, OUT, load_arrays, old
from report_hebb_post_pool_20261111 import PairedConcepts
from run_hebb_chain_20261111 import WINDOWS, estimate as estimate_interval, write_csv


NAMES = dict(conv4="PCoder4", conv5="PCoder5", fc1="FC1", fc2="FC2")


def csv_rows(name):
    with (DOC / f"all_{name}.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def lookup(rows, **keys):
    matches = [r for r in rows if all(r[k] == str(v) for k, v in keys.items())]
    assert len(matches) == 1, (keys, len(matches))
    return matches[0]


def ci(row, decimals=3, scale=1, signed=True):
    spec = ("+" if signed else "") + f".{decimals}f"
    val, lo, hi = [format(float(row[k]) * scale, spec) for k in ("estimate", "ci_low", "ci_high")]
    return f"{val} [{lo}, {hi}]"


def estimate(row, decimals=3, scale=1):
    return f"{float(row['estimate']) * scale:+.{decimals}f}"


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
            *["| " + " | ".join(map(str, row)) + " |" for row in rows], ""]


def plot():
    fig, axes = plt.subplots(2, 4, figsize=(12.7, 6.2), layout="constrained", sharex=True)
    styles = {"suppress": ("Four layers", "#0b6e75", "-"),
              "shared_off": ("Target bypass", "#7965a9", "--"),
              "no_hebb": ("Hebbian off", "#737d86", ":"),
              "shuffle": ("Shuffled memory", "#b15f32", "-.")}
    for i, dataset in enumerate(DATASETS):
        root = OUT / dataset
        rows = old.read(root / "manifest.json")["development"]
        s = PairedConcepts(np.array([r["category_index"] for r in rows]), np.array([r["concept"] for r in rows]), 0)
        for case, (label, color, ls) in styles.items():
            pair = [load_arrays(root / f"dev__{case}__{h}.npz") for h in HISTORIES]
            for j, layer in enumerate(LAYERS):
                e, d = [a[layer + "_mean"] for a in pair]
                rs = s.concept_weights[0] @ (100 * (1 - e / d))
                axes[i, j].plot(np.arange(15), rs, color=color, linestyle=ls, linewidth=1.8, label=label)
        for j, layer in enumerate(LAYERS):
            ax = axes[i, j]
            ax.set_title(f"{'ImageNet' if dataset == 'imagenet' else 'THINGS'} · {NAMES[layer]}", loc="left", fontsize=11)
            ax.axhline(0, color="0.78", linewidth=.8)
            ax.axvline(4.5, color="0.88", linewidth=.8)
            ax.axvline(9.5, color="0.88", linewidth=.8)
            ax.set_xticks([0, 4, 9, 14])
            ax.spines[["right", "top"]].set_visible(False)
            if i == 1:
                ax.set_xlabel("Predify target step")
            if j == 0:
                ax.set_ylabel("Repetition suppression (%)")
    axes[0, 3].legend(loc="best", frameon=False, fontsize=8)
    fig.savefig(DOC / "repetition_timecourse.png", dpi=180)
    plt.close(fig)


def readout_deltas():
    output = []
    for ds in DATASETS:
        root = OUT / ds
        for scope in ("dev", "heldout"):
            rows = old.read(root / "manifest.json")["development"] if scope == "dev" else old.read(root / "heldout_manifest.json")
            s = PairedConcepts(np.array([r["category_index"] for r in rows]), np.array([r["concept"] for r in rows]))
            for h in HISTORIES:
                names = ("suppress", "no_hebb", "shared_off", "gain")
                data = {name: load_arrays(root / f"{scope}__{name}__{h}.npz")["fc2_scores"]
                        for name in names if (root / f"{scope}__{name}__{h}.npz").exists()}
                for w, sl in enumerate(WINDOWS):
                    stats = {name: s.readout(scores[:, sl], fc=True) for name, scores in data.items()}
                    comparisons = [("suppress", ref) for ref in ("no_hebb", "shared_off")]
                    if "gain" in stats:
                        comparisons.append(("gain", "suppress"))
                    for case, ref in comparisons:
                        for metric in ("accuracy", "ba", "auc", "margin"):
                            delta = (stats[case][metric] - stats[ref][metric]) * (1 if metric == "margin" else 100)
                            output.append(dict(dataset=ds, scope=scope, history=h, window=w, condition=case, reference=ref,
                                               metric=metric, **estimate_interval(delta)))
    write_csv(DOC / "fc2_paired_readout.csv", output)


def main():
    complete = old.read(DOC / "all_complete.json")
    assert complete["scopes"] == ["dev", "heldout"]
    a = old.read(DOC / "training_a.json")
    b = old.read(DOC / "training_b.json")
    frozen = old.read(DOC / "selection.json")
    lam = ", ".join(f"{v:g}" for v in frozen["suppression"]["lam"])
    gain = "保留仅抑制作为主配置" if frozen["gain"] is None else f"通过增益配置 `{frozen['gain']}`"
    status = "通过训练读出约束" if frozen["feasible"] else "使用参考点作诊断"
    rep = csv_rows("repetition")
    geo = csv_rows("contrasts")
    read = csv_rows("readout")
    interaction = csv_rows("interactions")
    held = [r for r in rep if r["scope"] == "heldout" and r["condition"] == "suppress"
            and r["reference"] == "absolute" and r["window"] == "2"]
    rs = [float(r["estimate"]) for r in held]
    positive = sum(float(r["ci_low"]) > 0 for r in held)
    baseline_fisher = [float(r["estimate"]) for r in geo if r["scope"] == "heldout" and r["condition"] == "suppress"
                       and r["window"] == "2" and r["history"] == "exact" and r["layer"] in LAYERS
                       and r["reference"] == "no_hebb" and r["metric"] == "fisher"]
    assert len(rs) == len(baseline_fisher) == 8
    txt = ["# PCoder4、PCoder5、FC1、FC2 临时 Hebbian 串联结果", "",
           f"训练侧冻结 `{frozen['suppression']['id']}`，四层 λ 依次为 `{lam}`，{status}。{gain}。",
           "ImageNet／THINGS 的训练搜索为 24／30 个目标，clean 开发与保留 fold 0 分别各有 16／20 个目标。"
           "两套复核均使用冻结参数和分类头。下表方括号是类别内概念配对重采样的描述性 95% 区间。", "",
           f"在独立 fold 0，四层晚期重复抑制为 {min(rs):.3f}%–{max(rs):.3f}%，{positive}/8 个区间高于零。"
           f"Exact Fisher 相对全程关闭 Hebbian 的变化为 {min(baseline_fisher):+.3f}% 至 {max(baseline_fisher):+.3f}%。",
           "", "## 接线验收和训练选择", ""]
    for ds in DATASETS:
        audit = old.read(DOC / f"{ds}_audit.json")
        assert audit["passed"] and audit["zero_identity"] and audit["replay_exact"] and audit["fc_to_pc_identity"]
        assert max(audit["legacy_max_abs"].values()) == max(audit["fc_formula_max_abs"].values()) == 0
    txt += ["每套数据用一个训练 trial 验收旧 PCoder4-only 轨迹和独立 FC 公式，均逐元素复现。四份记忆各写两次，target 权重哈希不变。"
            "零记忆旁路和重放验收通过。切换 FC 开关时 PCoder 轨迹一致。",
            "", "训练筛选统计按两数据集和主要层取最差点。读出约束同时检查早期及中晚期。", ""]
    txt += table(["参数点", "最差重复抑制增量，百分点", "最差 Fisher 变化 %", "最差 FC2 margin 变化，基线 SD", "通过读出约束"],
                 [(k, f"{v['worst_rs']:+.3f}", f"{v['worst_fisher']:+.3f}", f"{v['worst_margin_sd']:+.3f}",
                   "是" if v["feasible"] else "否") for k, v in a["summary"].items()])
    txt += ["抑制后的 `ε=.005` 增益通过每个位置的局部活动守恒与非零支持检查。以下是训练 Exact、t5–14 的 Fisher 变化。", ""]
    txt += table(["数据集", "增益位置", "固定输入，%", "真实串联，%", "该数据集通过增益筛选"],
                 [(ds, NAMES[layer], f"{x:+.4f}", f"{row['fisher_percent'][layer][0]:+.4f}", "是" if row["eligible"] else "否")
                  for ds, rows in b.items() for row in rows.values() for layer, x in row["local_fisher_exact_percent"].items()])
    txt += ["晋级需要同一位置的局部和串联 Exact Fisher 都改善、四层重复抑制减幅均在 0.02 个百分点以内，且通过读出约束。", ""]
    txt += table(["数据集", "增益配置", "最差重复抑制变化，pp", "通过读出约束"],
                 [(ds, name, f"{min(row['repetition_pp'].values()):+.4f}", "是" if row["feasible"] else "否")
                  for ds, rows in b.items() for name, row in rows.items()])
    txt += ["## 重复抑制和类别几何", "",
            "正重复抑制表示同一目标在 Exact 后活动低于 Different。目标读取增量与同一 preview 末态的 target 旁路对照比较。"
            "PCoder5 活动取正分量，带符号表征的几何另列在原始表。", ""]
    for scope, title in (("dev", "Clean 开发"), ("heldout", "Fold 0 冻结复核")):
        txt += [f"### {title}", ""]
        vals = []
        for ds in DATASETS:
            for layer in LAYERS:
                keys = dict(dataset=ds, scope=scope, condition="suppress", layer=layer)
                absolute = [ci(lookup(rep, **keys, window=w, reference="absolute")) for w in range(3)]
                inc = ci(lookup(rep, **keys, window=2, reference="shared_off"))
                geom = {metric: ci(lookup(geo, **keys, window=2, history="exact", reference="no_hebb",
                                          reference_layer=layer, metric=metric), 3) for metric in ("within", "between", "fisher")}
                vals.append((ds, NAMES[layer], *absolute, inc, *geom.values()))
        txt += table(["数据集", "层", "RS t0–4 %", "RS t5–9 %", "RS t10–14 %", "晚期读取增量 pp",
                      "晚期 Δ类内 %", "晚期 Δ类间 %", "晚期 ΔFisher %"], vals)
    txt += ["表征几何相对同历史、全程关闭 Hebbian 的参照。局部输入/输出与两种历史的结果均在完整对照表。", "",
            ]
    if frozen["gain"]:
        txt += ["### 冻结小增益相对四层仅抑制", "", "以下均为晚期窗口。几何用 Exact，RS 用 Exact/Different 配对。", ""]
        values = []
        for scope in ("dev", "heldout"):
            for ds in DATASETS:
                for layer in LAYERS:
                    keys = dict(dataset=ds, scope=scope, layer=layer, window=2, condition="gain", reference="suppress")
                    rs = ci(lookup(rep, **keys))
                    g = [ci(lookup(geo, **keys, history="exact", reference_layer=layer, metric=metric), 4)
                         for metric in ("within", "between", "fisher")]
                    values.append((scope, ds, NAMES[layer], rs, *g))
        txt += table(["范围", "数据集", "层", "ΔRS pp", "Δ类内 %", "Δ类间 %", "ΔFisher %"], values)
    txt += ["## FC2 固定分类头", "",
            "准确率和平衡准确率按外层步计算后在窗口平均。AUC 使用窗口平均分数。margin 是真实类别分数减最强竞争类别。", ""]
    values = []
    for scope in ("dev", "heldout"):
        for ds in DATASETS:
            for w, label in enumerate(("t0–4", "t5–9", "t10–14")):
                keys = dict(dataset=ds, scope=scope, history="exact", window=w)
                x = [lookup(read, **keys, condition="suppress", metric=m) for m in ("accuracy", "ba", "auc", "margin")]
                y = [lookup(read, **keys, condition="no_hebb", metric=m) for m in ("accuracy", "ba", "auc", "margin")]
                both = [f"{float(b['estimate']) * scale:.{d}f} → {float(a['estimate']) * scale:.{d}f}"
                        for a, b, scale, d in zip(x, y, (100, 100, 100, 1), (2, 2, 2, 3))]
                values.append((scope, ds, label, *both))
    txt += table(["范围", "数据集", "窗口", "准确率 %", "平衡准确率 %", "宏 AUC %", "margin"], values)
    txt += ["## 机制对照", "", "### PCoder 和 FC 的交互", "",
            "使用同一 preview 末态和四份临时权重。交互量为四层同开 − 两 PCoder 单开 − 两 FC 单开 + target 全旁路。"
            "下表是晚期重复效应，使用逐目标 `100×log(Different/Exact)`。", ""]
    txt += table(["数据集", "PCoder4", "PCoder5", "FC1", "FC2"],
                 [(ds, *[ci(lookup(interaction, dataset=ds, layer=layer, window=2, metric="log_repeat_x100")) for layer in LAYERS])
                  for ds in DATASETS])
    txt += ["### 固定输入时的记忆匹配", "",
            "复用 Exact 轨迹的实际输入，分别查询 Exact 和 Different 权重。下表是 t5、t9、t14 的第一侧向步，"
            "数值为 Exact 减 Different。第一步输入相同，群体均值与标准差相同。", ""]
    values = []
    for ds in DATASETS:
        root = OUT / ds
        z = load_arrays(root / "dev_fixed_input_match.npz")["trace"]
        assert np.array_equal(z[:, 0, :, :, 0, 1], z[:, 1, :, :, 0, 1])
        delta = (z[:, 0, :, 1:, 0] - z[:, 1, :, 1:, 0]).mean(2)
        for i, layer in enumerate(LAYERS):
            values.append((ds, NAMES[layer], f"{delta[:, i, 0].mean():+.6f}", f"{delta[:, i, 2].mean():+.6f}"))
    txt += table(["数据集", "层", "Δκ", "Δ阈值，归一化活动"], values)
    txt += ["### 打乱、PCoder5 与 FC preview 对照", ""]
    values = []
    for ds in DATASETS:
        for layer in LAYERS:
            keys = dict(dataset=ds, scope="dev", layer=layer, window=2, reference="suppress")
            values.append((ds, NAMES[layer], *[ci(lookup(rep, **keys, condition=k))
                                               for k in ("shuffle", "pc4_only", "preview_fc")]))
    txt += table(["数据集", "层", "打乱记忆 ΔRS pp", "仅关 PCoder5 ΔRS pp", "FC 参与第二段 preview ΔRS pp"], values)
    max_rel = 0.
    for ds in DATASETS:
        root = OUT / ds
        a, b = [load_arrays(root / f"dev__{k}__exact.npz") for k in ("suppress", "pc4_only")]
        def first_step(field):
            differing = np.flatnonzero(np.abs(a[field] - b[field]).max(0) > 1e-9)
            return f"t{differing[0]}" if differing.size else "各步一致"
        for group in (a, b):
            group["prd5_rms"] = group["pc5_diagnostics"][:, :, 1]
        txt += [f"{ds} Exact 下只切换 PCoder5 时，`prd5` 的 RMS 首次出现差异为 {first_step('prd5_rms')}，"
                f"`rep4` 的活动均值首次出现差异为 {first_step('conv4_mean')}。", ""]
        for h in HISTORIES:
            data = {k: load_arrays(root / f"dev__{k}__{h}.npz") for k in ("suppress", "pc_only", "fc_only", "shared_off", "local")}
            for name in ("conv4", "conv5", "conv5_signed"):
                for field in ("mean", "gram"):
                    key = f"{name}_{field}"
                    assert np.array_equal(data["suppress"][key], data["pc_only"][key])
                    assert np.array_equal(data["fc_only"][key], data["shared_off"][key])
            z = data["local"]
            for layer in LAYERS:
                base = z[layer + "_mean"]
                max_rel = max(max_rel, float(np.max(np.abs(base - z[layer + "_local_mean"]) / base)))
    txt += ["两个历史的全部开发样本上，切换 FC 时 PCoder4/5 的活动均值和完整表征 Gram 逐元素一致。"
            f"固定输入的小增益使逐图总活动的最大相对变化为 `{max_rel:.2e}`。", ""]
    bridge = DOC / "fc_bridge_diagnostics.csv"
    if bridge.exists():
        with bridge.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        txt += ["### FC 桥接和等活动读出", "",
                "保留同一 PCoder 轨迹和冻结记忆，比较真实抑制与逐层等均值缩放。"
                "以下为开发批量 Exact、t10–14。FC2 桥接列只改变 FC1，FC2 本层列在已更新 FC1 的输入上开启本层抑制。", ""]
        values = []
        for ds in DATASETS:
            keys = dict(dataset=ds, history="exact", window=2, layer="fc2")
            for case, ref, name in (("pool_raw", "raw_raw", "FC1 桥接"), ("pool_pool", "pool_raw", "FC2 本层"),
                                    ("pool_pool", "scale_scale", "两层抑制相对等均值")):
                f = ci(lookup(rows, **keys, case=case, reference=ref, metric="fisher_percent"))
                m = ci(lookup(rows, **keys, case=case, reference=ref, metric="margin"))
                values.append((ds, name, f, m))
        txt += table(["数据集", "对照", "FC2 ΔFisher %", "FC2 Δmargin"], values)
    txt += ["## 结果文件", "",
            "[逐步数据](all_steps.csv) · [活动与类别几何](all_geometry.csv) · [配对几何对照](all_contrasts.csv) · "
            "[重复抑制](all_repetition.csv) · [FC2 读出](all_readout.csv) · [FC2 配对读出变化](fc2_paired_readout.csv) · [交互](all_interactions.csv) · "
            "[冻结配置](selection.json) · [重复抑制时间曲线](repetition_timecourse.png)", ""]
    if bridge.exists():
        txt += ["[FC 桥接与等活动对照完整数据](fc_bridge_diagnostics.csv)", ""]
    txt += ["使用 `predify2021` 环境从仓库根目录复现。", "", "```powershell",
            "python -B analysis/run_hebb_chain_20261111.py --stage all",
            "python -B analysis/diagnose_hebb_chain_20261111.py",
            "python -B analysis/report_hebb_chain_20261111.py", "```", ""]
    plot()
    readout_deltas()
    (DOC / "结果报告.md").write_text("\n".join(txt), encoding="utf-8")


if __name__ == "__main__":
    main()
