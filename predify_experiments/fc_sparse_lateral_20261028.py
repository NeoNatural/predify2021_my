"""Round 2: sparse Hebbian graphs and per-image adaptive activity thresholds."""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from . import fc_lateral_study_20261028 as core
from .fc_lateral_vanilla_20261028 import ROOT, sha, write_json

DATA = core.DATA
OUT = ROOT/"Hebbian-CNN/Log/fc_sparse_lateral_20261028"
DOC = ROOT/"docs/fc_sparse_lateral_20261028"
DEVICE = core.DEVICE
SEED = core.SEED
EPS = 1e-7
MODES = ("divide", "soft", "hard")


def configurations():
    configs = [dict(family="baseline", rule="zero", graph="zero", mode="identity", gain=0., budget=1.),
        dict(family="scale_half", rule="zero", graph="zero", mode="scale", gain=0., budget=1.)]
    for mode in ("soft", "hard"):
        for budget in (.75, 1.):
            configs.append(dict(family="only_"+mode, rule="zero", graph="zero", mode=mode, gain=0., budget=budget))
    for rule in ("cov", "corr"):
        for graph in ("dense", "k32", "k128"):
            for mode in MODES:
                for gain in (.25, .75):
                    for budget in ((1.,) if mode == "divide" else (.75, 1.)):
                        configs.append(dict(family=f"{rule}_{graph}_{mode}", rule=rule, graph=graph,
                            mode=mode, gain=gain, budget=budget))
    for i,c in enumerate(configs): c["id"] = i
    return configs


def mutual_topk(matrix, k):
    """Symmetric mutual-neighbor graph: every unit has degree at most k."""
    index = matrix.topk(min(k, matrix.shape[1]-1), dim=1).indices
    mask = torch.zeros_like(matrix, dtype=torch.bool)
    mask.scatter_(1, index, True)
    mask = mask & mask.T & (matrix > 0)
    sparse = matrix*mask
    sparse.fill_diagonal_(0)
    retained_gain = sparse.sum(1).max()
    sparse = sparse/retained_gain.clamp_min(1e-8)
    degree = mask.sum(1)
    return sparse, dict(density=float(mask.float().mean()), degree_max=int(degree.max()),
        degree_mean=float(degree.float().mean()), isolated=int((degree==0).sum()),
        retained_gain_before_renormalization=float(retained_gain))


def connection_model(train):
    model = core.connection_model(train)
    matrices, statistics = {}, {}
    for rule in ("cov", "corr"):
        e = model["matrices"][rule]
        matrices[rule+"_dense"] = e
        statistics[rule+"_dense"] = dict(density=float((e>0).float().mean()),
            degree_max=int((e>0).sum(1).max()), degree_mean=float((e>0).sum(1).float().mean()))
        for k in (32, 128):
            matrices[f"{rule}_k{k}"], statistics[f"{rule}_k{k}"] = mutual_topk(e,k)
    model["matrices"] = matrices
    model["graph_statistics"] = statistics
    return model


def adaptive_target(drive, count, mode):
    """Quantile threshold; hard rejects weak values, soft also subtracts threshold."""
    values = drive.sort(dim=1, descending=True).values
    d = drive.shape[1]
    hi = values.gather(1,(count-1).clamp(0,d-1)[:,None])
    lo = values.gather(1,count.clamp(0,d-1)[:,None])
    theta = (hi+lo)*.5
    theta = torch.where((count>=d)[:,None], torch.zeros_like(theta), theta)
    theta = torch.where((count==0)[:,None], values[:,:1]+1., theta)
    target = torch.relu(drive-theta) if mode=="soft" else torch.where(drive>theta,drive,0.)
    return target,theta


def dynamics(original, model, config, matrix=None):
    mode = config["mode"]
    if mode == "identity": return original.clone(), dict(steps=0,residual=0.,converged=True)
    if mode == "scale": return original*.5, dict(steps=0,residual=0.,converged=True)
    e = matrix if matrix is not None else model["matrices"].get(config["rule"]+"_"+config["graph"])
    if mode == "divide":
        c = dict(inhibition="divide", gain=config["gain"], strength=2.,rule="unused")
        return core.dynamics(original,model,c,matrix=e)
    x = original/model["scale"]
    count = torch.ceil((original>EPS).sum(1)*config["budget"]).long()
    r = x.clone()
    residual = math.inf
    for step in range(1,core.MAX_STEPS+1):
        drive = x+config["gain"]*(r@e.T) if config["gain"] else x
        target,theta = adaptive_target(drive,count,mode)
        target = target/(1+2*r.mean(1,keepdim=True))
        updated = (1-core.ALPHA)*r+core.ALPHA*target
        # Do not leave leaky residual tails on units vetoed by the current threshold.
        updated = torch.where(target>0,updated,0.)
        if step%5 == 0:
            residual = float(((updated-r).norm(dim=1)/(x.norm(dim=1)+1e-8)).max())
            if not math.isfinite(residual) or float(updated.abs().max()) > 1e5: break
            if residual < core.TOL:
                r = updated
                break
        r = updated
    return r*model["scale"],dict(steps=step,residual=residual,
        converged=bool(math.isfinite(residual) and residual<core.TOL),threshold_mean=float(theta.mean()))


def measurements(r, original, y, detailed=False):
    m = core.geometry(r,y,detailed)
    active, before = r>EPS, original>EPS
    ratios = active.sum(1)/before.sum(1).clamp_min(1)
    revived = active & ~before
    weak = active & (r<r.max(1,keepdim=True).values*.01)
    m.update(support_ratio=float(ratios.mean()),max_support_ratio=float(ratios.max()),
        density_increase_images=int((active.sum(1)>before.sum(1)).sum()),
        revived_fraction=float(revived.float().mean()),
        revived_activity_share=float(((r*revived).sum(1)/r.sum(1).clamp_min(1e-8)).mean()),
        weak_activity_share=float(((r*weak).sum(1)/r.sum(1).clamp_min(1e-8)).mean()))
    return m


def choose(rows, configs):
    choices = {}
    for family in sorted({c["family"] for c in configs}):
        options = []
        for c in (c for c in configs if c["family"]==family):
            rs = [r for r in rows if r["config"]["id"]==c["id"]]
            assert len(rs)==2
            stable = all(r["metrics"]["converged"] and r["metrics"]["zero_fraction"]==0 for r in rs)
            density_guard = all(r["metrics"]["max_support_ratio"]<=1.00001 for r in rs)
            delta = [core.effects(r["metrics"],r["baseline"]) for r in rs]
            guard = stable and np.mean([v["accuracy_pp"] for v in delta])>=-2 and np.mean([v["auc_delta"] for v in delta])>=-.005
            score = float(np.mean([core.joint_score(r["metrics"],r["baseline"]) for r in rs])) if stable else -1e6
            options.append(dict(config=c,score=score,stable=bool(stable),density_guard=bool(density_guard),
                readout_guard=bool(guard),inner_joint=bool(stable and density_guard and guard and score>1e-6),
                inner_effects={k:float(np.mean([v[k] for v in delta])) for k in delta[0]}))
        choices[family] = max(options,key=lambda o:(o["inner_joint"],o["density_guard"],o["readout_guard"],o["stable"],o["score"],-o["config"]["gain"]))
    eligible = [o for o in choices.values() if o["config"]["graph"] in ("k32","k128") and o["config"]["mode"] in ("soft","hard")]
    choices["selected"] = copy.deepcopy(max(eligible,key=lambda o:(o["inner_joint"],o["readout_guard"],o["stable"],o["score"])))
    return choices


def freeze():
    p = dict(version=1,exploratory_reuse_of_round1_cohort=True,configs=configurations(),outer_folds=5,inner_folds=2,
        connection="positive covariance or regularized positive correlation; dense or mutual top32/top128, symmetric zero self; renormalize max row sum to1",
        activity="division strength2; adaptive per-image count=ceil(original active count times .75 or1); soft subtracts threshold, hard only vetoes below threshold; vetoed units immediately zero to avoid leaky tails",
        dynamics=dict(alpha=core.ALPHA,tolerance=core.TOL,max_steps=core.MAX_STEPS),
        selection="same weakest log sharpening score as round1 plus no-density-increase guard; selected restricted to sparse adaptive families; raw readout guard only; angular readout diagnostic never vetoes raw benefits",
        readout="same fixed ridge for grid; raw C1 SVM for selected, paired no-Hebb, matched-scale, previous dense; angular ridge/SVM baseline vs selected as diagnosis",
        limitations="same exploratory samples reused after viewing round1; current nested selection is not a new independent confirmation",
        source_sha256=sha(__file__),dependencies={p:sha(ROOT/p) for p in ["predify_experiments/fc_lateral_study_20261028.py","predify_experiments/fc_lateral_vanilla_20261028.py"]},
        original_model_sources=core.freeze()["original_model_sources"],
        inputs={d:{f:sha(DATA/d/f) for f in ["manifest.json","activations.npz","baseline_models.pckl"]} for d in ("imagenet","things")},
        round1_protocol_sha256=sha(core.OUT/"protocol.json"))
    path = OUT/"protocol.json"
    if path.exists(): assert json.loads(path.read_text(encoding="utf-8"))==p,"Frozen source/protocol changed"
    else:
        write_json(path,p)
        (OUT/"source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    return p


def run(dataset,layer):
    protocol=freeze()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with np.load(DATA/dataset/"activations.npz") as z:
        x=torch.as_tensor(z[layer],device=DEVICE)
        y,groups,folds=z["category"],z["concept"],z["fold"]
    yt=torch.as_tensor(y,device=DEVICE)
    with (DATA/dataset/"baseline_models.pckl").open("rb") as f:heads=pickle.load(f)["models"][layer]
    configs=configurations()
    started=time.monotonic()
    for f in range(5):
        dest=OUT/dataset/layer/f"fold{f}"
        if (dest/"complete.json").exists(): print("EXISTS",dataset,layer,f,flush=True);continue
        tr,te=np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
        assert not set(groups[tr])&set(groups[te])
        inner=core.inner_folds(y[tr],groups[tr],SEED+f)
        rows=[]
        for inner_id in range(2):
            local_train,local_val=np.flatnonzero(inner!=inner_id),np.flatnonzero(inner==inner_id)
            a,b=tr[local_train],tr[local_val]
            model=connection_model(x[a])
            baseline=None
            for c in configs:
                r,diag=dynamics(x[tr],model,c)
                m=measurements(r[local_val],x[b],yt[b])
                m.update(core.classification(core.ridge_scores(r[local_train],r[local_val],y[a],groups[a]),y[b]),**diag)
                if c["family"]=="baseline":baseline=m
                rows.append(dict(inner=inner_id,config=c,metrics=m,baseline=baseline))
            print("INNER",dataset,layer,f,inner_id,"configs",len(configs),"seconds",round(time.monotonic()-started,1),flush=True)
        choices=choose(rows,configs)
        write_json(dest/"selection.json",dict(train_rows=tr.tolist(),test_rows=te.tolist(),inner_fold=inner.tolist(),choices=choices,candidates=rows))
        model=connection_model(x[tr])
        order=np.concatenate([tr,te])
        transformed,diagnostics={},{}
        for family,choice in choices.items():
            if family=="selected":continue
            transformed[family],diagnostics[family]=dynamics(x[order],model,choice["config"])
        chosen=choices["selected"]["config"]
        transformed["selected"]=transformed[chosen["family"]]
        diagnostics["selected"]=diagnostics[chosen["family"]]
        perm=torch.as_tensor(np.random.default_rng(SEED+f).permutation(x.shape[1]),device=DEVICE)
        e=model["matrices"][chosen["rule"]+"_"+chosen["graph"]]
        transformed["selected_shuffle"],diagnostics["selected_shuffle"]=dynamics(x[order],model,chosen,e[perm][:,perm])
        nohebb=dict(chosen,gain=0.)
        transformed["selected_no_hebb"],diagnostics["selected_no_hebb"]=dynamics(x[order],model,nohebb)
        scaled,threshold,mismatch=core.matched_controls(x[order],transformed["selected"])
        transformed["selected_matched_scale"],transformed["selected_matched_threshold"]=scaled,threshold
        diagnostics["selected_matched_scale"]=dict(steps=0,residual=0.,converged=True)
        diagnostics["selected_matched_threshold"]=dict(steps=0,residual=0.,converged=True,support_mismatch_mean=mismatch)
        previous=dict(rule="corr",graph="dense",mode="divide",gain=.75,budget=1.)
        transformed["previous_dense"],diagnostics["previous_dense"]=dynamics(x[order],model,previous)
        with np.load(core.OUT/dataset/layer/f"fold{f}"/"test_representations.npz") as z:
            reference_error=float(np.abs(transformed["previous_dense"][len(tr):].cpu().numpy()-z["selected"]).max())
        assert reference_error<2e-6
        results,arrays,scores_saved,svm_heads={},{},{},{}
        angular_base=None
        for family,r in transformed.items():
            a,b=r[:len(tr)],r[len(tr):]
            m=measurements(b,x[te],yt[te],True)
            scores=core.ridge_scores(a,b,y[tr],groups[tr])
            m.update(core.classification(scores,y[te]),**diagnostics[family])
            fixed=heads[f].decision_function(b.cpu().numpy())
            m["frozen_svm"]=core.classification(fixed,y[te])
            scores_saved[family]=scores
            scores_saved[family+"_frozen"]=fixed
            if family in ("selected","selected_no_hebb","selected_matched_scale","previous_dense"):
                s,h=core.svm_fit(a,b,y[tr],groups[tr]);m["refit_svm"]=core.classification(s,y[te])
                scores_saved[family+"_svm"],svm_heads[family]=s,h
            if family=="baseline":m["refit_svm"]=m["frozen_svm"]
            if family in ("baseline","selected"):
                an=a/a.norm(dim=1,keepdim=True).clamp_min(1e-8)
                bn=b/b.norm(dim=1,keepdim=True).clamp_min(1e-8)
                s,h=core.svm_fit(an,bn,y[tr],groups[tr])
                m["angular_svm"]=core.classification(s,y[te]);scores_saved[family+"_angular_svm"]=s
                m["angular_ridge"]=core.classification(core.ridge_scores(an,bn,y[tr],groups[tr]),y[te])
                svm_heads[family+"_angular"]=h
            results[family]=m;arrays[family]=b.cpu().numpy()
        base=results["baseline"]
        for family,m in results.items():
            m["effects"]=core.effects(m,base)
            m["descriptive_joint"]=bool(core.joint_score(m,base)>1e-6)
            m["sparse_joint"]=bool(m["descriptive_joint"] and m["max_support_ratio"]<=1.00001)
        for family in ("scale_half","selected_matched_scale"):
            assert abs(results[family]["within"]/base["within"]-1)<2e-5
            assert abs(results[family]["fisher"]/base["fisher"]-1)<2e-5
        assert results["selected"]["max_support_ratio"]<=1.00001
        write_json(dest/"results.json",dict(dataset=dataset,layer=layer,fold=f,choices=choices,results=results,
            graph_statistics=model["graph_statistics"],previous_dense_replay_max_error=reference_error))
        np.savez_compressed(dest/"test_representations.npz",row=te,y=y[te],concept=groups[te],**arrays)
        np.savez_compressed(dest/"readout_scores.npz",row=te,y=y[te],**scores_saved)
        with (dest/"svm_models.pckl").open("wb") as handle:pickle.dump(svm_heads,handle,protocol=pickle.HIGHEST_PROTOCOL)
        write_json(dest/"complete.json",dict(protocol_sha256=sha(OUT/"protocol.json"),
            files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},seconds=time.monotonic()-started))
        print("OUTER",dataset,layer,f,"selected",chosen,"effects",results["selected"]["effects"],
            "density",results["selected"]["nonzero"],"joint",results["selected"]["sparse_joint"],
            "seconds",round(time.monotonic()-started,1),flush=True)
        del model,transformed,e
    for p,h in protocol["original_model_sources"].items():assert sha(ROOT/p)==h


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","run"))
    p.add_argument("--dataset",choices=("imagenet","things"))
    p.add_argument("--layer",choices=("fc1","fc2"))
    a=p.parse_args()
    if a.action=="freeze":print("FROZEN",len(freeze()["configs"]),"configurations",flush=True)
    else:
        with torch.inference_mode():
            for dataset in ([a.dataset] if a.dataset else ["imagenet","things"]):
                for layer in ([a.layer] if a.layer else ["fc1","fc2"]):run(dataset,layer)


if __name__=="__main__":main()
