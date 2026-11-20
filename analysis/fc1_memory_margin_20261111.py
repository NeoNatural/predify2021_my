"""Frozen FC1 pilot diagnostics. No fitting or parameter selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.fc_sae_gated_search_20261106 import OUT as SOURCE, load_models
from analysis import fc_preview_pilot_20261105 as exp
from predify_experiments.fc_sae_gated_20261106 import memory_from_preview, query, side
from predify_experiments.fc_sae_gain_20261105 import model_hash

OUT = ROOT / 'docs/fc1_memory_margin_20261111'
HISTORIES = ('exact', 'different', 'none')
COLORS = ('#c84b31', '#2468a0', '#777777')

def margin(head, values, labels):
    scores = head.decision_function(values)
    classes = head.classes_
    correct = classes[None, :] == labels[:, None]
    return scores[correct] - np.where(correct, -np.inf, scores).max(axis=1)

def bootstrap_weights(labels, concepts):
    rng = np.random.default_rng(20261111)
    categories = np.unique(labels)
    weights = np.zeros((5000, len(labels)))
    for category in categories:
        groups = np.unique(concepts[labels == category])
        samples = rng.integers(len(groups), size=(len(weights), len(groups)))
        for j, group in enumerate(groups):
            mask = (labels == category) & (concepts == group)
            frequency = (samples == j).sum(axis=1) / len(groups)
            weights[:, mask] = frequency[:, None] / (len(categories) * mask.sum())
    return weights

def estimate(values, weights, boot):
    values = np.asarray(values)
    mean = weights @ values
    low, high = np.quantile(boot @ values, [.025, .975], axis=0)
    return dict(mean=np.asarray(mean).tolist(), low=np.asarray(low).tolist(), high=np.asarray(high).tolist())

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exp.setup()
    cfg = exp.read(SOURCE / 'selection.json')['config']
    calibration = exp.read(SOURCE / 'calibration.json')
    assert cfg['epsilon'] == .03 and cfg['lambda'] == .03 and cfg['steps'] == 5
    summary = {'config': cfg, 'datasets': {}, 'bootstrap': '5000 concept resamples within each category; pointwise intervals', 'source_hashes': {}}
    scatter, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    curves, curve_axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    all_trials, all_steps = [], []
    for dataset_index, dataset in enumerate(('imagenet', 'things')):
        data = exp.features(dataset)
        labels, concepts = data['y'], data['concept']
        assert len(labels) == (16 if dataset == 'imagenet' else 18)
        proxy = load_models(dataset)[0]
        proxy_before = model_hash(proxy)
        head_path = exp.OUT / dataset / 'reference_svm.pckl'
        with head_path.open('rb') as stream:
            head = pickle.load(stream)['fc1']
        paths = [head_path, SOURCE/'selection.json', SOURCE/'calibration.json', exp.OUT/dataset/'inputs/features.npz']
        for path in paths:
            summary['source_hashes'][str(path)] = exp.sha(path)
        x = torch.from_numpy(data['target_fc1']).cuda()
        cal = calibration[dataset][0]
        natural, first_outputs, matches = {}, {}, {}
        endpoint_error = 0.
        with np.load(SOURCE/dataset/'representations.npz') as saved:
            assert np.array_equal(saved['row'], data['row'])
            for history_index, history in enumerate(HISTORIES):
                preview = None if history == 'none' else torch.from_numpy(data['preview_fc1'][:, history_index]).cuda()
                memory = memory_from_preview(proxy, preview, len(x), x.device, x.dtype)
                frozen_w = memory.W.clone()
                z, h, q, kappa = query(proxy, x, memory)
                matches[history] = (.5*(z*h).sum(-1)).cpu().numpy()
                if history != 'none':
                    pool, _ = side(x, proxy, memory, cal, 0., cfg['lambda'], 1)
                    first_outputs[history+'_pool'] = pool.cpu().numpy()
                states = [x.cpu().numpy()]
                # Existing side() is replayed from x for each prefix, preserving production semantics.
                for step in range(1, 6):
                    response, _ = side(x, proxy, memory, cal, cfg['epsilon'], cfg['lambda'], step)
                    states.append(response.cpu().numpy())
                if history != 'none':
                    first_outputs[history+'_full'] = states[1]
                natural[history] = np.stack([margin(head, v, labels) for v in states], axis=1)
                endpoint_error = max(endpoint_error, float(np.max(np.abs(states[-1]-saved['full__'+history+'__fc1']))))
                assert torch.equal(frozen_w, memory.W)
                if history == 'none':
                    assert all(np.array_equal(v, states[0]) for v in states)
        assert endpoint_error < 3e-5
        assert proxy_before == model_hash(proxy)
        assert all(np.array_equal(natural[h][:, 0], natural['none'][:, 0]) for h in HISTORIES)
        sums = {key: value.sum(axis=1, dtype=np.float64) for key, value in first_outputs.items()}
        common = np.min(np.stack(list(sums.values())), axis=0)
        assert (common > 0).all(), 'Zero-total target requires separate reporting'
        matched = {}
        total_error = 0.
        scales = {}
        for key, values in first_outputs.items():
            scales[key] = common/sums[key]
            scaled = values.astype(np.float64)*scales[key][:, None]
            total_error = max(total_error, float(np.max(np.abs(scaled.sum(axis=1)-common))))
            matched[key] = margin(head, scaled, labels)
        delta_s = matches['exact']-matches['different']
        advantage = matched['exact_full']-matched['different_full']
        gain = advantage-(matched['exact_pool']-matched['different_pool'])
        weights = exp.weights(labels, concepts)
        weights = weights/weights.sum()
        boot = bootstrap_weights(labels, concepts)
        differences = {h: natural['exact']-natural[h] for h in ('different', 'none')}
        metrics = {'delta_S': delta_s, 'A': advantage, 'G': gain,
                   'early_Exact_Different': differences['different'][:, 1:3].mean(axis=1),
                   'early_Exact_None': differences['none'][:, 1:3].mean(axis=1),
                   'final_Exact_Different': differences['different'][:, 5],
                   'final_Exact_None': differences['none'][:, 5]}
        result = {key: estimate(value, weights, boot) for key, value in metrics.items()}
        result.update(n=len(labels), concepts=len(np.unique(concepts)), endpoint_max_error=endpoint_error,
                      matched_total_max_error=total_error, zero_total_count=0, proxy_hash=proxy_before,
                      stronger_match_count=int((delta_s>0).sum()), A_positive_count=int((advantage>0).sum()),
                      G_positive_count=int((gain>0).sum()),
                      spearman_S_A=float(spearmanr(delta_s, advantage).statistic),
                      spearman_S_G=float(spearmanr(delta_s, gain).statistic),
                      scaling_range=[min(float(v.min()) for v in scales.values()), max(float(v.max()) for v in scales.values())])
        result['curves'] = {h: estimate(natural[h], weights, boot) for h in HISTORIES}
        result['differences'] = {h: estimate(v, weights, boot) for h, v in differences.items()}
        summary['datasets'][dataset] = result
        for i, row in enumerate(data['row']):
            record = dict(dataset=dataset, target=int(row), concept=str(concepts[i]), category=int(labels[i]),
                          delta_S=float(delta_s[i]), A=float(advantage[i]), G=float(gain[i]), common_total=float(common[i]))
            record.update({key+'_margin':float(value[i]) for key,value in matched.items()})
            record.update({key+'_scale':float(value[i]) for key,value in scales.items()})
            all_trials.append(record)
            for history in HISTORIES:
                for step in range(6):
                    all_steps.append(dict(dataset=dataset,target=int(row),concept=str(concepts[i]),category=int(labels[i]),history=history,step=step,margin=float(natural[history][i,step])))
        for column, (name, values) in enumerate((('A', advantage), ('G', gain))):
            ax = axes[dataset_index,column]
            ax.scatter(delta_s, values, c=labels, cmap='tab10', s=44, edgecolors='white')
            ax.axhline(0,color='#999999',lw=.8); ax.axvline(0,color='#999999',lw=.8)
            ax.set(xlabel='Exact - Different memory match (delta S)', ylabel=name+' (fixed SVM margin)', title=dataset.upper()+' | '+name)
            stat=result[name]
            ax.text(.03,.97,f"Weighted mean {stat['mean']:+.5f}\n95% CI [{stat['low']:+.5f}, {stat['high']:+.5f}]",transform=ax.transAxes,va='top',fontsize=9)
        for history,color in zip(HISTORIES,COLORS):
            stat=result['curves'][history]
            curve_axes[dataset_index,0].plot(range(6),stat['mean'],'o-',label=history.title(),color=color)
        for history,color in zip(('different','none'),COLORS[:2]):
            stat=result['differences'][history]
            ax=curve_axes[dataset_index,1]
            ax.plot(range(6),stat['mean'],'o-',label='Exact - '+history.title(),color=color)
            ax.fill_between(range(6),stat['low'],stat['high'],color=color,alpha=.15)
        for column in range(2):
            ax=curve_axes[dataset_index,column]
            ax.set(xlabel='FC1 lateral update step',ylabel='Fixed SVM margin' if column==0 else 'Paired margin difference',title=dataset.upper())
            ax.set_xticks(range(6));ax.legend(fontsize=9);ax.grid(alpha=.15)
        curve_axes[dataset_index,1].axhline(0,color='#777777',lw=.8)
        print(dataset, json.dumps(result, ensure_ascii=False), flush=True)
    for path,digest in summary['source_hashes'].items():
        assert exp.sha(path)==digest
    pd.DataFrame(all_trials).to_csv(OUT/'matched_activity_per_target.csv',index=False)
    pd.DataFrame(all_steps).to_csv(OUT/'margin_steps_per_target.csv',index=False)
    (OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    scatter.suptitle('FC1 step 1 | equal total activity | 34 development targets')
    curves.suptitle('FC1 natural trajectories | frozen vanilla classifier\nShading: pointwise 95% concept-bootstrap intervals for paired differences')
    for figure,name in ((scatter,'memory_match_margin'),(curves,'margin_trajectories')):
        figure.savefig(OUT/(name+'.png'),dpi=180)
        figure.savefig(OUT/(name+'.pdf'))

if __name__ == '__main__':
    main()
