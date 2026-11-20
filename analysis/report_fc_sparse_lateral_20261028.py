"""Audit and report round2 sparse/adaptive FC experiments."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from predify_experiments.fc_sparse_lateral_20261028 import DATA,OUT,DOC,sha,EPS

DATASETS=("imagenet","things")
LAYERS=("fc1","fc2")
EFFECTS=("activation_pct","within_pct","between_pct","fisher_pct","accuracy_pp","auc_delta")


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path,rows):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def raw_geometry(x,y):
    centers=np.stack([x[y==c].mean(0,dtype=np.float64) for c in range(5)])
    within=float(np.mean([np.square(x[y==c]-centers[c]).sum(1).mean() for c in range(5)]))
    between=float(np.square(centers-centers.mean(0)).sum(1).mean())
    return within,between,between/max(within,1e-20)


def audit_and_aggregate():
    protocol=read(OUT/"protocol.json")
    assert sha(OUT/"source_snapshot.py")==protocol["source_sha256"]
    assert sha(ROOT/"predify_experiments/fc_sparse_lateral_20261028.py")==protocol["source_sha256"]
    for p,h in {**protocol["dependencies"],**protocol["original_model_sources"]}.items():assert sha(ROOT/p)==h
    for d in DATASETS:
        for p,h in protocol["inputs"][d].items():assert sha(DATA/d/p)==h
    records,choices,category_rows=[],[],[]
    audit=dict(passed=True,folds=0,candidate_evaluations=0,representation_matrices=0,
        original_model_unchanged=True,round1_code_unchanged=True,vanilla_cache_unchanged=True,
        max_selected_support_ratio=0.,old_dense_replay_max_error=0.)
    for d in DATASETS:
        with np.load(DATA/d/"activations.npz") as z:
            y,groups,folds=z["category"],z["concept"],z["fold"]
            original={l:z[l] for l in LAYERS}
        for l in LAYERS:
            for f in range(5):
                dest=OUT/d/l/f"fold{f}"
                complete=read(dest/"complete.json")
                assert complete["protocol_sha256"]==sha(OUT/"protocol.json")
                for p,h in complete["files"].items():assert sha(dest/p)==h
                data=read(dest/"results.json");selection=read(dest/"selection.json")
                tr,te=np.array(selection["train_rows"]),np.array(selection["test_rows"])
                assert np.array_equal(te,np.flatnonzero(folds==f))
                assert not set(groups[tr])&set(groups[te])
                inner=np.array(selection["inner_fold"])
                for j in (0,1):assert not set(groups[tr[inner==j]])&set(groups[tr[inner!=j]])
                assert len(selection["candidates"])==132
                audit["candidate_evaluations"]+=132;audit["folds"]+=1
                choice=data["choices"]["selected"]
                choices.append(dict(dataset=d,layer=l,fold=f,**choice["config"],inner_joint=choice["inner_joint"],inner_score=choice["score"],
                    graph_density=data["graph_statistics"][choice["config"]["rule"]+"_"+choice["config"]["graph"]]["density"]))
                for graph,stats in data["graph_statistics"].items():
                    if graph.endswith("k32"):assert stats["degree_max"]<=32
                    if graph.endswith("k128"):assert stats["degree_max"]<=128
                audit["old_dense_replay_max_error"]=max(audit["old_dense_replay_max_error"],data["previous_dense_replay_max_error"])
                with np.load(dest/"test_representations.npz") as z,np.load(dest/"readout_scores.npz") as scores:
                    assert np.array_equal(z["row"],te) and np.array_equal(z["y"],y[te])
                    assert np.array_equal(z["baseline"],original[l][te])
                    assert np.array_equal(z["selected"],z[choice["config"]["family"]])
                    before=(z["baseline"]>EPS).sum(1)
                    raw_base=raw_geometry(z["baseline"],y[te])
                    after=(z["selected"]>EPS).sum(1)
                    assert np.all(after<=before)
                    audit["max_selected_support_ratio"]=max(audit["max_selected_support_ratio"],float((after/np.maximum(before,1)).max()))
                    assert np.allclose(z["selected"].mean(1),z["selected_matched_scale"].mean(1),atol=2e-6)
                    for family,m in data["results"].items():
                        r=z[family]
                        assert r.shape==(len(te),4096) and r.dtype==np.float32 and np.isfinite(r).all() and (r>=0).all()
                        assert scores[family].shape==(len(te),5) and np.isfinite(scores[family]).all()
                        if family.endswith("_soft") or family.endswith("_hard"):
                            c=data["choices"][family]["config"]
                            assert np.all((r>EPS).sum(1)<=np.ceil(before*c["budget"]))
                        audit["representation_matrices"]+=1
                        base=data["results"]["baseline"]
                        raw_w,raw_b,raw_f=raw_geometry(r,y[te])
                        row=dict(dataset=d,layer=l,fold=f,family=family,**m["effects"],
                            activation=m["activation"],within=m["within"],between=m["between"],fisher=m["fisher"],
                            accuracy=m["accuracy"],auc=m["auc"],nonzero=m["nonzero"],support_ratio=m["support_ratio"],
                            max_support_ratio=m["max_support_ratio"],revived_fraction=m["revived_fraction"],
                            revived_activity_share=m["revived_activity_share"],weak_activity_share=m["weak_activity_share"],
                            converged=m["converged"],steps=m["steps"],residual=m["residual"],zero_fraction=m["zero_fraction"],
                            descriptive_joint=m["descriptive_joint"],sparse_joint=m["sparse_joint"],effective_dimension=m["effective_dimension"],
                            frozen_svm_accuracy_pp=100*(m["frozen_svm"]["accuracy"]-base["frozen_svm"]["accuracy"]),
                            frozen_svm_auc_delta=m["frozen_svm"]["auc"]-base["frozen_svm"]["auc"],
                            raw_within=raw_w,raw_between=raw_b,raw_fisher=raw_f,
                            raw_within_pct=100*(raw_w/raw_base[0]-1),raw_between_pct=100*(raw_b/raw_base[1]-1),
                            raw_fisher_pct=100*(raw_f/raw_base[2]-1))
                        for field in ("refit_svm","angular_svm","angular_ridge"):
                            row[field+"_accuracy_pp"]=100*(m[field]["accuracy"]-base[field]["accuracy"]) if field in m else None
                            row[field+"_auc_delta"]=m[field]["auc"]-base[field]["auc"] if field in m else None
                        records.append(row)
                        for c,name in enumerate(("animals","clothing","food","plants","vehicles")):
                            category_rows.append(dict(dataset=d,layer=l,fold=f,family=family,category=name,
                                activation_pct=100*(m["per_class_activation"][c]/base["per_class_activation"][c]-1),
                                within_pct=100*(m["per_class_within"][c]/base["per_class_within"][c]-1),
                                ridge_recall_pp=100*(m["recalls"][c]-base["recalls"][c])))
    assert audit["folds"]==20 and audit["old_dense_replay_max_error"]<2e-6
    summary=[]
    for d in DATASETS:
        for l in LAYERS:
            families=sorted({r["family"] for r in records if r["dataset"]==d and r["layer"]==l})
            assert len(families)==28
            for family in families:
                rs=[r for r in records if r["dataset"]==d and r["layer"]==l and r["family"]==family]
                assert len(rs)==5
                result=dict(dataset=d,layer=l,family=family)
                for k in records[0]:
                    if k in ("dataset","layer","fold","family","converged","descriptive_joint","sparse_joint"):continue
                    result[k]=float(np.mean([r[k] for r in rs])) if all(r[k] is not None for r in rs) else None
                result.update(converged_folds=sum(r["converged"] for r in rs),
                    joint_folds=sum(r["descriptive_joint"] for r in rs),sparse_joint_folds=sum(r["sparse_joint"] for r in rs),
                    mean_joint=bool(result["activation_pct"]<-.001 and result["within_pct"]<-.001 and result["fisher_pct"]>.001),
                    support_ratio_max=max(r["max_support_ratio"] for r in rs))
                summary.append(result)
    write_csv(DOC/"fold_metrics.csv",records);write_csv(DOC/"summary.csv",summary)
    write_csv(DOC/"category_metrics.csv",category_rows);write_csv(DOC/"selected_parameters.csv",choices)
    (DOC/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (DOC/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary,choices,audit


def plots(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs=DOC/"figures";figs.mkdir(exist_ok=True)
    order=["previous_dense","only_soft","only_hard"]
    order += [f"{rule}_{graph}_{mode}" for rule in ("cov","corr") for graph in ("dense","k32","k128") for mode in ("divide","soft","hard")]
    order += ["selected","selected_no_hebb","selected_shuffle","selected_matched_scale","selected_matched_threshold"]
    for d in DATASETS:
        for l in LAYERS:
            lookup={r["family"]:r for r in summary if r["dataset"]==d and r["layer"]==l}
            columns=("activation_pct","within_pct","fisher_pct","nonzero","auc_delta")
            vals=np.array([[lookup[f][c] for c in columns] for f in order]);vals[:,3]*=100
            colors=vals*np.array([-1,-1,1,-1,1])
            colors/=np.maximum(np.abs(colors).max(0),1e-9)
            fig,ax=plt.subplots(figsize=(12,12))
            ax.imshow(colors,cmap="RdBu",vmin=-1,vmax=1,aspect="auto")
            ax.set_yticks(range(len(order)),[f+" !" if lookup[f]["converged_folds"]<5 else f for f in order],fontsize=9)
            ax.set_xticks(range(5),["Mean activation %","Within scatter %","Fisher ratio %","Active units %","Raw readout AUC delta"],fontsize=9)
            ax.xaxis.tick_top()
            for i in range(len(order)):
                for j in range(5):
                    ax.text(j,i,f"{vals[i,j]:+.4f}" if j==4 else f"{vals[i,j]:+.2f}",ha="center",va="center",fontsize=9,
                        color="white" if abs(colors[i,j])>.6 else "black")
            ax.set_title(f"{d.upper()} {l.upper()} | sparse Hebbian + adaptive threshold\nBlue: lower A/W/density or higher Fisher/AUC; ! = not all folds converged\nDensity is a constraint, not the sharpening objective; no significance claims",pad=50)
            fig.tight_layout();fig.savefig(figs/f"{d}_{l}_comparison.png",dpi=170)
            fig.savefig(figs/f"{d}_{l}_comparison.svg");plt.close(fig)
    trace=DOC/"training_stability_trace.csv"
    if trace.exists():
        with trace.open(encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for mode,budget in (("soft",1.),("hard",1.),("hard",.75)):
            rs=[r for r in rows if r['mode']==mode and float(r['budget'])==budget]
            label=f"{mode}, budget={budget}"
            axs[0].semilogy([int(r['step']) for r in rs],[float(r['max_relative_residual']) for r in rs],label=label)
            axs[1].plot([int(r['step']) for r in rs],[float(r['mean_switching_units']) for r in rs],label=label)
        axs[0].axhline(1e-5,color='black',linestyle=':',label='convergence tolerance')
        axs[0].set_ylabel('Max relative residual');axs[1].set_ylabel('Mean units switching support per image')
        for ax in axs:ax.set_xlabel('Iteration');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle('Training-only post-hoc diagnostic: ImageNet FC1 fold0, corr top128, gain0.25')
        fig.tight_layout();fig.savefig(figs/'threshold_stability.png',dpi=170);plt.close(fig)


def report(summary,choices,audit):
    def get(d,l,f):return next(r for r in summary if r["dataset"]==d and r["layer"]==l and r["family"]==f)
    lines=["# 第二轮：稀疏 Hebbian＋自适应阈值结果","", "2026-10-28。复用第一轮探索样本，五外层折、每折两内层折；原模型、vanilla缓存和第一轮代码均未修改。", "",
        "原始幅度下的readout是功能评价，去幅度后的readout只用于来源诊断，不是成功硬门槛。主目标仍是平均活动下降＋表征区分度提高；逐图活动数不增加是这轮防过密约束。", "",
        "## 内层选出的稀疏自适应候选","", "| 数据集/层 | 平均激活% | W% | B% | Fisher% | 非零比例 | Ridge BA pp | Ridge AUC Δ | SVM BA pp | SVM AUC Δ | 稀疏联合同向折数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    qualified=[f"{d}/{l}" for d in DATASETS for l in LAYERS if get(d,l,'selected')['mean_joint']]
    unqualified=[f"{d}/{l}" for d in DATASETS for l in LAYERS if not get(d,l,'selected')['mean_joint']]
    graph_only=[get(d,l,f'corr_{graph}_divide')['nonzero']*100 for d in DATASETS for l in LAYERS for graph in ('k32','k128')]
    lines[6:6]=["## 主要发现","",
        "三项几何指标的五折均值同向改善组合："+('、'.join(qualified) if qualified else '无')+"；未同时达标组合："+('、'.join(unqualified) if unqualified else '无')+"。是否每折都同向另见表格，不能以平均结果掩盖失败折。所有所选候选均逐图保持活动数不增加。","",
        f"仅连接稀疏并不足以解决过密：正相关top32/top128在仅除法条件下，非零比例仍为约{min(graph_only):.2f}%–{max(graph_only):.2f}%。自适应活动阈值才实际限制了弱响应铺开。","",
        "本轮主候选限定在稀疏＋adaptive家族，参数只用内层选择。不能将其称为全部模型的外层最优；dense、自适应但不剪枝、只抑制及配对去Hebbian对照均完整保留。稀疏图在剪枝后重新归一化增益，因而结果不是单纯删除连接的因果消融。","",
        "幅度造成的readout改善不被删去。本轮把raw readout作为功能检验，把去范数readout作为机制诊断；若raw收益来自有用增益调节，仍可报告为功能收益，但不自动证明方向几何或语义连接增强。","" ]
    for d in DATASETS:
        for l in LAYERS:
            r=get(d,l,"selected")
            lines.append(f"| {d}/{l} | {r['activation_pct']:+.2f} | {r['within_pct']:+.2f} | {r['between_pct']:+.2f} | {r['fisher_pct']:+.2f} | {100*r['nonzero']:.2f}% | {r['accuracy_pp']:+.2f} | {r['auc_delta']:+.5f} | {r['refit_svm_accuracy_pp']:+.2f} | {r['refit_svm_auc_delta']:+.5f} | {r['sparse_joint_folds']}/5 |")
    lines += ["", "联合同向要求平均激活下降、归一化类内W下降、Fisher上升、收敛且无零向量，并满足逐图活动数不超过原始输入。若候选没有同时达标，仍报告其实际失败，不能只因输出稀疏就称为成功。Fisher是相对区分度，B单独保留；两个类别都靠近共同方向也可能降低W，需结合B和readout。", "",
        "## 实际选择","", "| 数据集/层/折 | 连接 | 图 | 阈值 | 兴奋增益 | 活动预算 | 内层联合 | 图非零连接比例 |", "|---|---|---|---|---:|---:|---|---:|"]
    for c in choices:
        lines.append(f"| {c['dataset']}/{c['layer']}/{c['fold']} | {c['rule']} | {c['graph']} | {c['mode']} | {c['gain']} | {c['budget']} | {c['inner_joint']} | {100*c['graph_density']:.3f}% |")
    lines += ["", "soft：减去每图自适应分位阈值后ReLU；hard：阈值以下静默、阈值以上保留驱动。两者都使用除法归一化。活动预算为原图非零数的75%或100%，由无标签输入决定。互为top-k的图每单元至多k条连接；剪枝后重新归一化总增益，单边权重可能增大。", "",
        "## 对照与招募","", "均值匹配缩放逐图匹配候选活动量；去Hebbian使用候选相同阈值预算与除法强度，不重新选参。旧dense参考固定为第一轮corr/divide(.75,2)，并逐折验证其表征回放。", ""]
    controls=["baseline","previous_dense","selected","selected_no_hebb","selected_shuffle","selected_matched_scale","selected_matched_threshold"]
    for d in DATASETS:
        for l in LAYERS:
            lines += [f"### {d}/{l}","", "| 条件 | 激活% | W% | Fisher% | 非零% | 新招募单元占全体% | 新招募活动量占比% | Ridge AUC Δ |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for f in controls:
                r=get(d,l,f)
                lines.append(f"| {f} | {r['activation_pct']:+.2f} | {r['within_pct']:+.2f} | {r['fisher_pct']:+.2f} | {100*r['nonzero']:.2f} | {100*r['revived_fraction']:.2f} | {100*r['revived_activity_share']:.2f} | {r['auc_delta']:+.5f} |")
            lines += ["",f"![Comparison](<{(DOC/'figures'/f'{d}_{l}_comparison.png').as_posix()}>)",""]
    lines += ["## 原始幅度与去幅度后的readout","", "| 数据集/层 | 原始SVM BA pp | 原始SVM AUC Δ | 匹配均值缩放SVM AUC Δ | 去幅度SVM BA pp | 去幅度SVM AUC Δ |", "|---|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        for l in LAYERS:
            r=get(d,l,"selected");scale=get(d,l,"selected_matched_scale")
            lines.append(f"| {d}/{l} | {r['refit_svm_accuracy_pp']:+.3f} | {r['refit_svm_auc_delta']:+.5f} | {scale['refit_svm_auc_delta']:+.5f} | {r['angular_svm_accuracy_pp']:+.3f} | {r['angular_svm_auc_delta']:+.5f} |")
    lines += ["", "幅度可以携带真实类别和可靠性信息，不能把幅度带来的准确率改善一律删除。两种readout分别回答“系统是否有功能收益”和“去掉样本范数后方向可读性是否改善”。纯统一缩放在标准化重拟合读出中抵消；逐图不同缩放则不一定。这里不要求raw收益在去幅度后保留，亦不将raw收益自动归因为方向几何。", "",
        "## 保留幅度的几何指标","", "| 数据集/层 | 原始W变化% | 原始B变化% | 原始Fisher变化% | 归一化Fisher变化% |", "|---|---:|---:|---:|---:|"]
    for d in DATASETS:
        for l in LAYERS:
            r=get(d,l,"selected")
            lines.append(f"| {d}/{l} | {r['raw_within_pct']:+.2f} | {r['raw_between_pct']:+.2f} | {r['raw_fisher_pct']:+.2f} | {r['fisher_pct']:+.2f} |")
    lines += ["", "原始W/B保留径向幅度信息，但两者都可因缩放而下降，因此不单独用原始W下降判定几何sharpening。原始/归一化Fisher及两种readout并列保留；没有将幅度信息从所有评价中删除。", "",
        "## 固定点收敛","", "| 数据集/层 | soft条件收敛折数/总折数 | hard条件收敛折数/总折数 | 主候选收敛折数 |", "|---|---:|---:|---:|"]
    for d in DATASETS:
        for l in LAYERS:
            soft=[r for r in summary if r['dataset']==d and r['layer']==l and r['family'].endswith('_soft')]
            hard=[r for r in summary if r['dataset']==d and r['layer']==l and r['family'].endswith('_hard')]
            lines.append(f"| {d}/{l} | {sum(r['converged_folds'] for r in soft)}/{5*len(soft)} | {sum(r['converged_folds'] for r in hard)}/{5*len(hard)} | {get(d,l,'selected')['converged_folds']}/5 |")
    lines += ["", "未满足1e-5固定点残差的条件不能当作稳定机制；输出有限不等于已收敛。硬门控的活动集切换可能妨碍固定点，全部失败信息保留。", "",
        "训练集上的补充诊断（不参与选参）将ImageNet FC1 fold0的corr/top128、增益0.25延长到300步：soft预算1.0残差约1.47e-8；hard预算1.0约1.08e-3、预算0.75约6.61e-3，仍高于阈值。是少量单元切换而非活动爆炸；不能把这一个训练诊断推广成所有hard实现都不稳定。", "",
        f"![Stability](<{(DOC/'figures/threshold_stability.png').as_posix()}>)", "",
        "## 验证与限制","", f"审计通过：{audit['folds']}个外层折，{audit['candidate_evaluations']}次内层候选评价，{audit['representation_matrices']}套完整4096维外层表征。所选候选逐图支持比最大{audit['max_selected_support_ratio']:.6f}；旧dense回放最大误差{audit['old_dense_replay_max_error']:.3g}。5项新机制测试通过。", "",
        "每折连接仅从该训练激活估计，标签用于模型选择与readout。外层数据没有参与本轮参数选择，但本轮机制设计参考了同一探索样本的第一轮结果，不能称为全新独立确认。ImageNet植物7细类、THINGS原生标注分歧等限制继续保留。五折结果描述性报告，不把折当独立受试者，不宣称显著。", "",
        "全部家族、个别失败/未收敛配置、分大类数据与角度读出均保留在summary.csv、fold_metrics.csv、category_metrics.csv和逐折结果。连接稀疏并不逻辑保证活动稀疏；活动变稀疏也不逻辑保证表征更有区分度。", "",
        "## 复现","", "```powershell","python -m predify_experiments.fc_sparse_lateral_20261028 run","python analysis/report_fc_sparse_lateral_20261028.py","```", ""]
    (DOC/"实验结果.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    summary,choices,audit=audit_and_aggregate()
    plots(summary);report(summary,choices,audit)
    for r in summary:
        if r["family"] in ("selected","selected_no_hebb","previous_dense","selected_shuffle"):
            print(r["dataset"],r["layer"],r["family"],{k:r[k] for k in EFFECTS},"density",r["nonzero"],
                "SVM",r["refit_svm_accuracy_pp"],r["refit_svm_auc_delta"],"joint",r["sparse_joint_folds"])
    print("AUDIT",audit)


if __name__=="__main__":main()
