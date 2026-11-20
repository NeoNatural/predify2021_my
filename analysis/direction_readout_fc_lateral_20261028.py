"""Post-grid fixed-parameter diagnosis: separate angular readout from amplitude."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predify_experiments.fc_lateral_study_20261028 import (
    DATA, OUT, DOC, DEVICE, sha, write_json, connection_model, dynamics,
    matched_controls, ridge_scores, svm_fit, classification,
)


def norm(x): return x/x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def freeze():
    protocol = dict(posthoc=True, no_reselection=True,
        reason="ImageNet FC1 matched-mean scaling reproduced much of raw readout AUC improvement; angular geometry alone cannot identify classification source",
        conditions=["baseline", "selected", "selected_shuffle"],
        angular_readouts="row L2 normalization followed by identical fixed ridge and refitted fixed C=1 SVM, train-only weighted standardization",
        additional="refitted raw SVM for selected_matched_scale; mathematical angular-invariance check",
        primary_protocol_sha256=sha(OUT/"protocol.json"), source_sha256=sha(__file__))
    path = OUT/"direction_readout_protocol.json"
    if path.exists(): assert json.loads(path.read_text(encoding="utf-8")) == protocol
    else:
        write_json(path, protocol)
        (OUT/"direction_readout_source_snapshot.py").write_bytes(Path(__file__).read_bytes())


def run():
    freeze()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    all_rows = []
    for dataset in ("imagenet", "things"):
        with np.load(DATA/dataset/"activations.npz") as z:
            y, groups, folds = z["category"], z["concept"], z["fold"]
            xs = {layer:torch.as_tensor(z[layer], device=DEVICE) for layer in ("fc1", "fc2")}
        for layer, x in xs.items():
            for f in range(5):
                dest = OUT/dataset/layer/f"fold{f}"
                path = dest/"direction_readout.json"
                if path.exists():
                    current = json.loads(path.read_text(encoding="utf-8"))
                    assert current["protocol_sha256"] == sha(OUT/"direction_readout_protocol.json")
                    all_rows.extend(current["rows"]); continue
                selection = json.loads((dest/"selection.json").read_text(encoding="utf-8"))
                tr, te = np.array(selection["train_rows"]), np.array(selection["test_rows"])
                ordered = np.concatenate([tr, te])
                model = connection_model(x[tr])
                selected, _ = dynamics(x[ordered], model, selection["choices"]["selected"]["config"])
                rng = np.random.default_rng(20261028+f)
                perm = torch.as_tensor(rng.permutation(x.shape[1]), device=DEVICE)
                e = model["matrices"][selection["choices"]["selected"]["config"]["rule"]]
                shuffled, _ = dynamics(x[ordered], model, selection["choices"]["selected"]["config"], e[perm][:,perm])
                scaled, _, _ = matched_controls(x[ordered], selected)
                with np.load(dest/"test_representations.npz") as z:
                    assert np.allclose(selected[len(tr):].cpu().numpy(), z["selected"], atol=2e-6)
                    assert np.allclose(shuffled[len(tr):].cpu().numpy(), z["selected_shuffle"], atol=2e-6)
                assert torch.allclose(norm(scaled), norm(x[ordered]), atol=1e-6)
                transformed = dict(baseline=x[ordered], selected=selected, selected_shuffle=shuffled)
                rows, saved = [], {}
                baselines = {}
                for family, r in transformed.items():
                    u = norm(r)
                    ridge = ridge_scores(u[:len(tr)], u[len(tr):], y[tr], groups[tr])
                    svm, _ = svm_fit(u[:len(tr)], u[len(tr):], y[tr], groups[tr])
                    saved[family+"_ridge"] = ridge
                    saved[family+"_svm"] = svm
                    for head, scores in (("ridge", ridge), ("svm", svm)):
                        metrics = classification(scores, y[te])
                        if family == "baseline": baselines[head] = metrics
                        row = dict(dataset=dataset, layer=layer, fold=f, family=family, head=head,
                            normalization="row_l2", accuracy=metrics["accuracy"], auc=metrics["auc"],
                            accuracy_pp=100*(metrics["accuracy"]-baselines[head]["accuracy"]),
                            auc_delta=metrics["auc"]-baselines[head]["auc"])
                        rows.append(row)
                raw_scores, _ = svm_fit(scaled[:len(tr)], scaled[len(tr):], y[tr], groups[tr])
                raw = classification(raw_scores, y[te])
                original_results = json.loads((dest/"results.json").read_text(encoding="utf-8"))
                base = original_results["results"]["baseline"]["refit_svm"]
                rows.append(dict(dataset=dataset, layer=layer, fold=f, family="selected_matched_scale", head="svm",
                    normalization="raw", accuracy=raw["accuracy"], auc=raw["auc"],
                    accuracy_pp=100*(raw["accuracy"]-base["accuracy"]), auc_delta=raw["auc"]-base["auc"]))
                saved["matched_scale_raw_svm"] = raw_scores
                np.savez_compressed(dest/"direction_readout_scores.npz", row=te, y=y[te], **saved)
                write_json(path, dict(protocol_sha256=sha(OUT/"direction_readout_protocol.json"), rows=rows,
                    score_sha256=sha(dest/"direction_readout_scores.npz"), selected_replay_verified=True,
                    scale_direction_invariance_verified=True))
                all_rows.extend(rows)
                print("DIRECTION COMPLETE", dataset, layer, f, flush=True)
    assert len(all_rows) == 140
    with (DOC/"direction_readout_fold_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    summary = []
    for dataset in ("imagenet", "things"):
        for layer in ("fc1", "fc2"):
            for family in ("baseline", "selected", "selected_shuffle", "selected_matched_scale"):
                for head in ("ridge", "svm"):
                    rs = [r for r in all_rows if r["dataset"]==dataset and r["layer"]==layer and r["family"]==family and r["head"]==head]
                    if not rs: continue
                    summary.append(dict(dataset=dataset, layer=layer, family=family, head=head, normalization=rs[0]["normalization"],
                        **{k:float(np.mean([r[k] for r in rs])) for k in ("accuracy", "auc", "accuracy_pp", "auc_delta")}))
    write_json(DOC/"direction_readout_summary.json", summary)
    lines = ["# 幅度与方向 readout 的补充诊断", "", "本分析在主网格执行过程中，观察到匹配均值缩放可带来接近的 raw readout 改善后增加；属于 post-hoc 机制诊断。未重新选择连接、抑制参数、折或标签。", "",
        "每图先L2归一化，之后在训练集拟合相同预处理和固定ridge/SVM，检验方向上的类别可读性。额外拟合raw幅度匹配缩放的SVM，比较原始尺度下的分类改善来源。", "",
        "| 数据集/层 | 条件 | 读出 | 输入 | BA变化pp | AUC变化 |", "|---|---|---|---|---:|---:|"]
    for r in summary:
        if r["family"] == "baseline": continue
        lines.append(f"| {r['dataset']}/{r['layer']} | {r['family']} | {r['head']} | {r['normalization']} | {r['accuracy_pp']:+.3f} | {r['auc_delta']:+.5f} |")
    lines += ["", "比较基线均为同一种读出及输入归一化流程，不能把angular与raw的绝对分数混用。没有统计显著性声明。纯逐图缩放的方向与基线相同，逐折进行了数值检查；无须据其raw readout变化推断方向结构改变。", ""]
    (DOC/"幅度与方向读出诊断.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if "--freeze" in sys.argv: freeze()
    else:
        with torch.inference_mode(): run()
