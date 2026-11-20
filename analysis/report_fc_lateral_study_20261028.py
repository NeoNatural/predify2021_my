"""Audit, aggregate and visualize the completed independent lateral study."""
from pathlib import Path
import copy
import csv
import json
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predify_experiments.fc_lateral_study_20261028 import DATA, OUT, DOC, CATEGORIES, sha, write_json

DATASETS = ("imagenet", "things")
LAYERS = ("fc1", "fc2")
EFFECTS = ("activation_pct", "within_pct", "between_pct", "fisher_pct", "accuracy_pp", "auc_delta")
LABELS = {
    "baseline":"原始输入", "scale_half":"纯缩放×0.5", "only_subtract":"仅减法抑制",
    "only_divide":"仅除法归一化", "only_threshold":"仅固定阈值", "selected":"内层选出的Hebbian",
    "selected_shuffle":"选出连接重排", "selected_matched_scale":"匹配均值的逐图缩放",
    "selected_matched_threshold":"支持未能匹配的阈值对照",
}
for rule, name in (("raw", "共激活"), ("cov", "正协方差"), ("corr", "正相关")):
    for inhibition, label in (("none", "无抑制"), ("subtract", "减法"), ("divide", "除法"), ("local", "局部补偿＋减法")):
        LABELS[rule+"_"+inhibition] = name+"＋"+label


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate():
    protocol = read(OUT/"protocol.json")
    assert sha(OUT/"source_snapshot.py") == protocol["source_sha256"]
    assert sha(ROOT/"predify_experiments/fc_lateral_study_20261028.py") == protocol["source_sha256"]
    for p, digest in protocol["original_model_sources"].items(): assert sha(ROOT/p) == digest
    for dataset in DATASETS:
        for p, digest in protocol["data"][dataset].items(): assert sha(DATA/dataset/p) == digest
    fold_rows, class_rows, choices, groups = [], [], [], {}
    audit = dict(passed=True, completed_folds=0, candidate_evaluations=0, representation_matrices=0,
        original_model_sources_unchanged=True, vanilla_cache_unchanged=True, checks=[])
    for dataset in DATASETS:
        with np.load(DATA/dataset/"activations.npz") as z:
            category, concept, fold = z["category"], z["concept"], z["fold"]
            originals = {l:z[l] for l in LAYERS}
        for layer in LAYERS:
            groups[dataset, layer] = {}
            for f in range(5):
                dest = OUT/dataset/layer/f"fold{f}"
                complete = read(dest/"complete.json")
                assert complete["protocol_sha256"] == sha(OUT/"protocol.json")
                for name, digest in complete["files"].items(): assert sha(dest/name) == digest
                data = read(dest/"results.json")
                selection = read(dest/"selection.json")
                tr, te = np.array(selection["train_rows"]), np.array(selection["test_rows"])
                assert np.array_equal(te, np.flatnonzero(fold == f))
                assert not set(concept[tr]) & set(concept[te])
                inner = np.array(selection["inner_fold"])
                for inner_id in range(2):
                    assert not set(concept[tr[inner == inner_id]]) & set(concept[tr[inner != inner_id]])
                assert len(selection["candidates"]) == 142
                audit["candidate_evaluations"] += 142
                audit["completed_folds"] += 1
                choice = data["choices"]["selected"]
                choices.append(dict(dataset=dataset, layer=layer, fold=f, **choice["config"],
                    inner_joint=choice["inner_joint"], inner_score=choice["score"]))
                with np.load(dest/"test_representations.npz") as z, np.load(dest/"readout_scores.npz") as scores:
                    assert np.array_equal(z["row"], te)
                    assert np.array_equal(z["y"], category[te])
                    assert np.array_equal(z["baseline"], originals[layer][te])
                    assert np.array_equal(z["scale_half"], originals[layer][te]*.5)
                    chosen_family = choice["config"]["family"]
                    assert np.array_equal(z["selected"], z[chosen_family])
                    assert np.allclose(z["selected"].mean(1), z["selected_matched_scale"].mean(1), atol=2e-6)
                    for family, met in data["results"].items():
                        assert z[family].shape == (len(te), 4096)
                        assert z[family].dtype == np.float32
                        assert np.isfinite(z[family]).all() and (z[family] >= 0).all()
                        assert scores[family].shape == (len(te), 5) and np.isfinite(scores[family]).all()
                        audit["representation_matrices"] += 1
                        entry = dict(dataset=dataset, layer=layer, fold=f, family=family,
                            activation=met["activation"], within=met["within"], between=met["between"], fisher=met["fisher"],
                            accuracy=met["accuracy"], auc=met["auc"], frozen_svm_accuracy=met["frozen_svm"]["accuracy"],
                            frozen_svm_auc=met["frozen_svm"]["auc"], converged=met["converged"], steps=met["steps"],
                            residual=met["residual"], zero_fraction=met["zero_fraction"], nonzero=met["nonzero"],
                            effective_dimension=met["effective_dimension"],
                            descriptive_joint=met["descriptive_joint"], **met["effects"])
                        if "refit_svm" in met:
                            base = data["results"]["baseline"]["refit_svm"]
                            entry.update(svm_accuracy=met["refit_svm"]["accuracy"], svm_auc=met["refit_svm"]["auc"],
                                svm_accuracy_pp=100*(met["refit_svm"]["accuracy"]-base["accuracy"]),
                                svm_auc_delta=met["refit_svm"]["auc"]-base["auc"])
                        else: entry.update(svm_accuracy=None, svm_auc=None, svm_accuracy_pp=None, svm_auc_delta=None)
                        fold_rows.append(entry)
                        groups[dataset, layer].setdefault(family, []).append(entry)
                        for c, name in enumerate(CATEGORIES):
                            base = data["results"]["baseline"]
                            class_rows.append(dict(dataset=dataset, layer=layer, fold=f, family=family, category=name,
                                activation=met["per_class_activation"][c],
                                activation_pct=100*(met["per_class_activation"][c]/base["per_class_activation"][c]-1),
                                within=met["per_class_within"][c],
                                within_pct=100*(met["per_class_within"][c]/base["per_class_within"][c]-1),
                                ridge_recall=met["recalls"][c],
                                ridge_recall_pp=100*(met["recalls"][c]-base["recalls"][c])))
            assert len(groups[dataset, layer]) == 21
    summary = []
    for (dataset, layer), families in groups.items():
        base = families["baseline"]
        for family, rows in families.items():
            assert len(rows) == 5
            s = dict(dataset=dataset, layer=layer, family=family, label=LABELS[family])
            for k in EFFECTS+("activation", "within", "between", "fisher", "accuracy", "auc", "nonzero", "effective_dimension"):
                s[k] = float(np.mean([r[k] for r in rows]))
            for k in EFFECTS:
                s[k+"_min"] = min(r[k] for r in rows)
                s[k+"_max"] = max(r[k] for r in rows)
            s.update(joint_folds=sum(r["descriptive_joint"] for r in rows),
                converged_folds=sum(r["converged"] for r in rows), zero_fraction=max(r["zero_fraction"] for r in rows),
                frozen_svm_accuracy_pp=float(np.mean([100*(r["frozen_svm_accuracy"]-b["frozen_svm_accuracy"]) for r,b in zip(rows, base)])),
                frozen_svm_auc_delta=float(np.mean([r["frozen_svm_auc"]-b["frozen_svm_auc"] for r,b in zip(rows, base)])))
            for k in ("svm_accuracy", "svm_auc", "svm_accuracy_pp", "svm_auc_delta"):
                s[k] = float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None
            s["mean_joint"] = bool(s["activation_pct"] < -1e-3 and s["within_pct"] < -1e-3 and s["fisher_pct"] > 1e-3 and not s["zero_fraction"])
            summary.append(s)
    write_csv(DOC/"fold_metrics.csv", fold_rows)
    write_csv(DOC/"category_metrics.csv", class_rows)
    write_csv(DOC/"selected_parameters.csv", choices)
    write_csv(DOC/"summary.csv", summary)
    write_json(DOC/"summary.json", summary)
    audit["checks"] = ["all 20 folds complete and all file hashes verified", "142 inner evaluations per outer fold",
        "train/test and inner concepts disjoint", "test row and category labels match original cache",
        "baseline and fixed scale exactly reconstruct original cache", "selected outputs equal their chosen family",
        "matched scaling preserves per-image means", "all saved outputs finite and nonnegative",
        "pure-scale geometry/readout invariance asserted during execution"]
    direction_protocol = OUT/"direction_readout_protocol.json"
    if direction_protocol.exists() and (DOC/"direction_readout_summary.json").exists():
        dp = read(direction_protocol)
        assert sha(OUT/"direction_readout_source_snapshot.py") == dp["source_sha256"]
        assert dp["primary_protocol_sha256"] == sha(OUT/"protocol.json")
        supplemental_rows = 0
        for dataset in DATASETS:
            for layer in LAYERS:
                for f in range(5):
                    dest = OUT/dataset/layer/f"fold{f}"
                    s = read(dest/"direction_readout.json")
                    assert s["protocol_sha256"] == sha(direction_protocol)
                    assert s["score_sha256"] == sha(dest/"direction_readout_scores.npz")
                    assert s["selected_replay_verified"] and s["scale_direction_invariance_verified"]
                    supplemental_rows += len(s["rows"])
        assert supplemental_rows == 140
        audit["posthoc_direction_readout_rows"] = supplemental_rows
        audit["posthoc_no_reselection"] = True
    write_json(DOC/"audit.json", audit)
    return summary, groups, choices, audit


def plots(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    figdir = DOC/"figures"
    figdir.mkdir(exist_ok=True)
    order = ["scale_half", "only_subtract", "only_divide", "only_threshold"]
    order += [r+"_"+i for r in ("raw", "cov", "corr") for i in ("none", "subtract", "divide", "local")]
    order += ["selected", "selected_shuffle", "selected_matched_scale", "selected_matched_threshold"]
    for dataset in DATASETS:
        for layer in LAYERS:
            lookup = {r["family"]:r for r in summary if r["dataset"] == dataset and r["layer"] == layer}
            values = np.array([[lookup[f][k] for k in ("activation_pct", "within_pct", "fisher_pct", "accuracy_pp", "auc_delta")] for f in order])
            benefit = values*np.array([-1, -1, 1, 1, 1])
            normalized = benefit/np.maximum(np.abs(benefit).max(0), 1e-9)
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(normalized, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
            ax.set_yticks(range(len(order)), [f+"*" if f=="selected_matched_threshold" else f for f in order], fontsize=9)
            ax.set_xticks(range(5), ["Mean activation %", "Within scatter %", "Fisher ratio %", "Readout BA pp", "Readout AUC delta"])
            ax.xaxis.tick_top(); ax.tick_params(axis="x", labelsize=9)
            for i in range(len(order)):
                for j in range(5):
                    txt = f"{values[i,j]:+.4f}" if j == 4 else f"{values[i,j]:+.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if abs(normalized[i,j]) > .6 else "black")
            ax.set_title(f"{dataset.upper()} {layer.upper()} | outer concept-held-out effects\nBlue = favorable; colors normalized per column; numbers = actual signed effects\nFive-fold descriptive means, not significance tests", pad=50, fontsize=12)
            fig.text(.5, .008, "* Support matching is infeasible for denser selected outputs; see report.", ha="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(figdir/f"{dataset}_{layer}_all_families.png", dpi=180)
            fig.savefig(figdir/f"{dataset}_{layer}_all_families.svg")
            plt.close(fig)


def report(summary, choices, audit):
    def get(d,l,f): return next(r for r in summary if r["dataset"]==d and r["layer"]==l and r["family"]==f)
    lines = ["# 独立 FC 侧向 Hebbian＋抑制实验结果", "", "2026-10-28。探索性五折外层概念泛化、每折两折内层选择；输入是普通 VGG16 单次前馈 FC 激活。原模型及 vanilla 缓存哈希保持不变。", "",
        "主目标是平均激活下降且表征更有区分度。本文分别保留类内收缩、类间距离、Fisher及readout；不把单纯压低活动称为成功。所有数值为相对同折基线的五折描述性均值，没有显著性或统计非劣性声明。", "",
        "## 由内层选择的 Hebbian 方法：外层结果", "",
        "| 数据集/层 | 平均激活% | 类内W% | 类间B% | Fisher% | Ridge BA pp | Ridge AUC Δ | 重拟合SVM BA pp | SVM AUC Δ | 三指标同向折数 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines[6:6] = ["## 主要发现", "",
        "在本轮71个配置范围内，20个外层训练集合的内层选择均为中心化、带方差下限的正相关连接＋除法归一化，兴奋增益0.75、抑制强度2。四个数据集×层组合在每一个外层折都同时呈现平均激活下降、归一化类内散度下降、Fisher上升。这是可复验的描述性几何sharpening候选，尚不等于已经证明稳定的分类收益或找到了全局最优参数。", "",
        "仅减法抑制和固定阈值在所测范围内降低活动，却使归一化类内离散度上升、Fisher下降；纯固定缩放或仅全局除法不能改变方向几何。连接重排仍能明显降低类内离散度，但也压缩类间距离，平均Fisher变化仅约−0.34%至+0.48%，远小于真实连接的约+5.33%至+11.69%。不能把类内变紧单独当作类别区分增强。", "",
        "同时，选出方法把输出从约19%–25%非零推至99.94%–100%非零；它降低总均值，却填充了大量弱响应，不是稀疏化。因此本轮结果支持特定几何指标的改善，不支持‘更少神经元被激活’的解释。", ""]
    for d in DATASETS:
        for l in LAYERS:
            r = get(d,l,"selected")
            lines.append(f"| {d}/{l} | {r['activation_pct']:+.2f} | {r['within_pct']:+.2f} | {r['between_pct']:+.2f} | {r['fisher_pct']:+.2f} | {r['accuracy_pp']:+.2f} | {r['auc_delta']:+.4f} | {r['svm_accuracy_pp']:+.2f} | {r['svm_auc_delta']:+.4f} | {r['joint_folds']}/5 |")
    lines += ["", "几何三指标共同改善不等于类间绝对距离一定增大：若B也下降，Fisher上升表示类间距离相对于类内离散度更好。需结合readout，而不能把这种结果描述为所有类别彼此远离。", "",
        "## 活动密度与无效的支持匹配", "", "| 数据集/层 | 原始非零比例 | 选出方法非零比例 | 原始有效维度 | 选出方法有效维度 |", "|---|---:|---:|---:|---:|"]
    for d in DATASETS:
        for l in LAYERS:
            b, r = get(d,l,"baseline"), get(d,l,"selected")
            lines.append(f"| {d}/{l} | {100*b['nonzero']:.2f}% | {100*r['nonzero']:.2f}% | {b['effective_dimension']:.2f} | {r['effective_dimension']:.2f} |")
    lines += ["", "阈值只能删除原输入中的响应，无法产生比原输入更多的非零单元，故支持匹配对照在本轮不成立：平均每图仍相差约3079至3312个活动单元。该行仍保留为实现诊断，但不能用于声称已经排除一切同等稀疏度的替代解释。仅抑制、固定阈值、纯缩放和连接重排等其他对照仍各自有效。", "",
        "## 内层实际选中的规则", "", "| 数据集/层/外层折 | 规则 | 兴奋增益 | 抑制强度 | 内层联合条件 |", "|---|---|---:|---:|---|"]
    for c in choices:
        lines.append(f"| {c['dataset']}/{c['layer']}/{c['fold']} | {LABELS[c['family']]} | {c['gain']} | {c['strength']} | {c['inner_joint']} |")
    lines += ["", "## 与对照的比较", "", "表中 Hebbian 是内层选出的整个程序，不按外层表现再挑一条参数。连接重排保留相同参数和矩阵权重/谱分布，破坏其与神经元坐标对应。匹配均值缩放逐图匹配选出方法的输出平均值，不能改变方向几何；支持匹配阈值尽量匹配活动单元数，误差逐折保存在原始结果。", ""]
    controls = ["selected", "only_subtract", "only_divide", "only_threshold", "selected_shuffle", "selected_matched_scale", "selected_matched_threshold"]
    for d in DATASETS:
        for l in LAYERS:
            lines += [f"### {d} / {l}", "", "| 条件 | 激活% | W% | B% | Fisher% | Ridge BA pp | AUC Δ | 同向折数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for f in controls:
                r = get(d,l,f)
                lines.append(f"| {LABELS[f]} | {r['activation_pct']:+.2f} | {r['within_pct']:+.2f} | {r['between_pct']:+.2f} | {r['fisher_pct']:+.2f} | {r['accuracy_pp']:+.2f} | {r['auc_delta']:+.4f} | {r['joint_folds']}/5 |")
            lines += ["", f"![All families](<{(DOC/'figures'/f'{d}_{l}_all_families.png').as_posix()}>)", ""]
    supplement = DOC/"direction_readout_summary.json"
    if supplement.exists():
        direction = read(supplement)
        lines += ["## 幅度与方向：补充机制诊断", "", "这是主网格过程中发现幅度对照可复现部分分类收益后增加的post-hoc检查，参数和样本均未重选。详见同目录 `幅度与方向读出诊断.md`。", "",
            "| 数据集/层 | 原始幅度SVM AUC Δ：真实连接 | 原始幅度SVM AUC Δ：匹配均值缩放 | 去幅度后真实连接SVM BA pp | 去幅度后真实连接SVM AUC Δ |", "|---|---:|---:|---:|---:|"]
        for d in DATASETS:
            for l in LAYERS:
                s = get(d,l,"selected")
                scale = next(r for r in direction if r["dataset"]==d and r["layer"]==l and r["family"]=="selected_matched_scale" and r["head"]=="svm")
                angular = next(r for r in direction if r["dataset"]==d and r["layer"]==l and r["family"]=="selected" and r["head"]=="svm")
                lines.append(f"| {d}/{l} | {s['svm_auc_delta']:+.5f} | {scale['auc_delta']:+.5f} | {angular['accuracy_pp']:+.3f} | {angular['auc_delta']:+.5f} |")
        lines += ["", "四种组合中，原始幅度匹配缩放的SVM AUC改善都不小于真实连接；去掉逐图幅度之后，真实连接的SVM AUC变化为约−0.00102、−0.00055、+0.00004、−0.00021，方向不一致。故不能将原始readout的小幅增益归因为Hebbian带来的稳定语义判别增强。当前较清楚的是几何上的相对收缩与Fisher改善；功能收益仍弱，且依赖读出与尺度。", ""]
    lines += ["## 读出解释", "", "主网格使用固定正则化线性ridge，全部候选与本轮同一ridge基线比较。旧SVM原始绝对分数不混入ridge比较。对内层选出的Hebbian、仅减法抑制、支持匹配阈值另做固定C=1的SVM重拟合确认；全条件还报告冻结原始SVM及scaler的结果。AUC先在各折内计算，再平均。", "",
        "冻结SVM检验原有边界兼容性；它受到活动尺度和偏置变化影响，不能仅据冻结准确率下降断言语义信息消失，也不能忽略它来声称可直接接回原模型。", "",
        "## 验证与范围", "", f"审计通过：{audit['completed_folds']}个完整外层折、{audit['candidate_evaluations']}次内层候选评价、{audit['representation_matrices']}个外层完整4096维表征矩阵。6项机制/划分/数值负对照测试通过；协议、源码、输入、原模型文件及全部输出有哈希记录。", "",
        "仅在训练输入估计无标签连接，标签用于候选筛选及readout，因此应称为无监督连接估计＋有标签模型选择。没有新训练VGG、反馈、prime、在线目标写入或视觉变换不变性实验。", "",
        "ImageNet植物只有7个细类；THINGS原生manual标签与human/WordNet存在既有分歧，未进行额外逐图人工审核。本次无完整流程重拟合bootstrap，不把五折当独立受试者，不把同图或同概念重复当独立概念。数据和规则范围之外不作推广。", "",
        "## 文件与复现", "", "- `summary.csv` / `summary.json`：全部84个数据集×层×条件摘要。", "- `fold_metrics.csv`：420行外层逐折指标及readout。", "- `category_metrics.csv`：分大类激活、类内离散及召回。", "- `selected_parameters.csv`：各外层折只由内层选择的参数。", "- `audit.json`：完整性及概念隔离审计。", "- `执行方案.md`：结果前固定的规则、网格、选择准则。", "- 原始输出目录 `Hebbian-CNN/Log/fc_lateral_study_20261028/`：每折选择明细、表征、分数、SVM及哈希。", "",
        "```powershell", "python -m predify_experiments.fc_lateral_study_20261028 run", "python analysis/report_fc_lateral_study_20261028.py", "```", ""]
    (DOC/"实验结果.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    summary, groups, choices, audit = aggregate()
    plots(summary)
    report(summary, choices, audit)
    for r in summary:
        if r["family"] in ("selected", "only_subtract", "selected_shuffle", "selected_matched_scale", "selected_matched_threshold"):
            print(r["dataset"], r["layer"], r["family"], {k:r[k] for k in EFFECTS}, "SVM", r["svm_accuracy_pp"], r["svm_auc_delta"], "joint", r["joint_folds"])
    print("AUDIT", audit)


if __name__ == "__main__": main()
