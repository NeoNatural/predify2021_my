"""Fit on train, calibrate on development, then audit the frozen independent test."""
from pathlib import Path
import csv
import json
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predify_experiments.readout_init_20261012 import OUT, DOC, PAPER, SEED, CONDITIONS, TRAIN_CONDITIONS, budget_index, immutable_json, sha
from predify_experiments.supplementary import load, save


def write(name, rows):
    DOC.mkdir(parents=True, exist_ok=True)
    with (DOC / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def data(split):
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf8'))
    expected = {(r['dataset_index'], s) for r in manifest[split] for s in manifest['noise_seeds'][split]}
    files = sorted((OUT / split).glob('*.pckl'))
    expected_conditions = set(TRAIN_CONDITIONS if split == 'train' else CONDITIONS)
    values = {}; calls = {}; target_calls = {}; durations = {}; rows = []
    seen = set()
    protected = {r['dataset_index'] for s in ('train', 'dev', 'test') for r in manifest[s]}
    for path in files:
        result = load(path); trial = result['trial']; key = trial['dataset_index'], result['noise']
        assert key in expected and key not in seen
        seen.add(key)
        assert {r['condition'] for r in result['records']} == expected_conditions
        assert len({r['target_hash'] for r in result['records']}) == 1
        rows.append((key, trial['category_index'], trial['imagenet_label']))
        for rec in result['records']:
            condition = rec['condition']
            assert rec['fc2'].shape == (15,4096) and np.isfinite(rec['fc2']).all()
            assert not ((set(rec['preview_indices']) - {trial['dataset_index'], None}) & protected)
            for budget in (9,15):
                budget_index(rec['total_backbone_calls'], budget)
            values.setdefault(condition, {})[key] = rec['fc2']
            calls.setdefault(condition, {})[key] = rec['total_backbone_calls']
            target_calls.setdefault(condition, {})[key] = rec['target_backbone_calls']
            durations.setdefault(condition, {})[key] = rec['elapsed_seconds']
    assert seen == expected, f'{split} missing {len(expected-seen)} files'
    rows.sort()
    keys = [r[0] for r in rows]
    return dict(keys=keys, labels=np.array([r[1] for r in rows]), synsets=np.array([r[2] for r in rows]),
                features={c:np.stack([v[k] for k in keys]) for c,v in values.items()},
                calls={c:np.stack([v[k] for k in keys]) for c,v in calls.items()},
                target_calls={c:np.stack([v[k] for k in keys]) for c,v in target_calls.items()},
                seconds={c:np.stack([v[k] for k in keys]) for c,v in durations.items()})


def fit():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.exceptions import ConvergenceWarning
    if (OUT / 'frozen_readout.json').exists():
        raise ValueError('Readout already frozen; do not refit')
    train = data('train')
    x = np.concatenate([train['features'][c].reshape(-1,4096) for c in TRAIN_CONDITIONS]).astype(np.float32)
    y = np.tile(np.repeat(train['labels'],15), len(TRAIN_CONDITIONS))
    legacy = load(PAPER / 'shared_decoder/analysis/shared_decoders.pckl')['classifiers'][('predify_noise','fc2',9)]
    print('FIT', x.shape, 'independent images',len({k[0] for k in train['keys']}), flush=True)
    attempts = []; heads = {'legacy':legacy}
    for name in ('pooled','matched_t9'):
        fit_x = x if name == 'pooled' else np.concatenate([train['features'][c][:,9] for c in TRAIN_CONDITIONS]).astype(np.float32)
        fit_y = y if name == 'pooled' else np.tile(train['labels'],len(TRAIN_CONDITIONS))
        for max_iter in (50000,200000):
            classifier = make_pipeline(StandardScaler(), LinearSVC(C=1., dual='auto', max_iter=max_iter, random_state=SEED))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                classifier.fit(fit_x,fit_y)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            attempts.append(dict(readout=name,max_iter=max_iter,n_iter=int(classifier[-1].n_iter_),converged=converged))
            if converged:
                break
        if not converged:
            raise RuntimeError(f'{name} SVM did not converge; test must not run')
        heads[name] = classifier
    result = dict(heads=heads, training_keys=train['keys'], fit_attempts=attempts,
                  fit_source_sha256=sha(__file__), manifest_sha256=sha(OUT/'manifest.json'),
                  legacy_source_sha256=sha(PAPER/'shared_decoder/analysis/shared_decoders.pckl'))
    save(OUT / 'fitted_readouts.pckl', result)
    write('fit_diagnostics.csv', attempts)
    print('FIT_COMPLETE', attempts, flush=True)


def score(head, features):
    return head.decision_function(features.astype(np.float32).reshape(-1,4096)).reshape(features.shape[0],15,5)


def freeze():
    if (OUT / 'frozen_readout.json').exists():
        raise ValueError('Already frozen; no recalibration')
    fitted = load(OUT/'fitted_readouts.pckl')
    dev = data('dev')
    assert not ({k[0] for k in fitted['training_keys']} & {k[0] for k in dev['keys']})
    scales = {}; summary = []
    for name, head in fitted['heads'].items():
        all_scores = {c:score(head,f) for c,f in dev['features'].items()}
        scales[name] = np.maximum(np.median(np.concatenate([s.std(-1) for s in all_scores.values()]),axis=0),1e-6)
        for c,s in all_scores.items():
            for t in range(15):
                summary.append(dict(readout=name,condition=c,timestep=t,accuracy_pct=100*float(np.mean(s[:,t].argmax(-1)==dev['labels'])),
                                    mean_score_sd=float(s[:,t].std(-1).mean()),frozen_temperature=float(scales[name][t])))
    fitted['temperatures']=scales
    fitted['calibration_keys']=dev['keys']
    fitted['calibration_source_sha256']=sha(__file__)
    (OUT / ('analysis_frozen_'+sha(__file__)[:12]+'.py')).write_bytes(Path(__file__).read_bytes())
    save(OUT/'readouts.pckl',fitted)
    frozen=dict(readouts_sha256=sha(OUT/'readouts.pckl'),manifest_sha256=sha(OUT/'manifest.json'),
                protocol_sha256=sha(DOC/'预先方案.md'), primary_readout='pooled', primary_endpoint='total_budget_9',
                primary_controls=['none_feedforward','carry_same_synset','carry_different_category'],
                temperatures={k:v.tolist() for k,v in scales.items()},train_images=200,development_images=100,test_images=100,
                candidate_selection='none; pooled head and total budget 9 specified before development extraction',
                early_calibration='per-timestep median SD across development images/noises/conditions; identical scale for all conditions')
    immutable_json(OUT/'frozen_readout.json',frozen)
    write('development_diagnostics.csv',summary)
    print('FROZEN', json.dumps(frozen), flush=True)


def paired_stats(a,b,keys,synsets):
    ids=sorted({k[0] for k in keys});id_indices=[np.array([j for j,k in enumerate(keys) if k[0]==i]) for i in ids]
    delta=100*(a.astype(float)-b.astype(float));clusters=np.array([delta[ix].mean() for ix in id_indices])
    rng=np.random.default_rng(SEED);draw=rng.integers(len(ids),size=(20000,len(ids)))
    lo,hi=np.quantile(clusters[draw].mean(1),[.025,.975])
    rng=np.random.default_rng(SEED);permutations=(rng.integers(2,size=(20000,len(ids)))*2-1)@clusters
    p=(np.sum(np.abs(permutations)>=abs(clusters.sum())-1e-9)+1)/20001
    group=np.array([synsets[ix[0]] for ix in id_indices]);unique=np.unique(group)
    sums=np.array([clusters[group==g].sum() for g in unique]);sizes=np.array([sum(group==g) for g in unique])
    draw=np.random.default_rng(SEED).integers(len(unique),size=(20000,len(unique)))
    slo,shi=np.quantile(sums[draw].sum(1)/sizes[draw].sum(1),[.025,.975])
    seeds=sorted({k[1] for k in keys});by_seed=[float(delta[[k[1]==s for k in keys]].mean()) for s in seeds]
    return dict(images=len(ids),accuracy_a_pct=100*float(a.mean()),accuracy_b_pct=100*float(b.mean()),
                delta_pp=float(clusters.mean()),ci_low_pp=float(lo),ci_high_pp=float(hi),signflip_p=float(p),
                synset_ci_low_pp=float(slo),synset_ci_high_pp=float(shi),per_noise_delta_pp=json.dumps(by_seed),
                wins_trials=int(np.sum(a&~b)),losses_trials=int(np.sum(~a&b)))


def analyze():
    frozen=json.loads((OUT/'frozen_readout.json').read_text())
    assert sha(OUT/'readouts.pckl')==frozen['readouts_sha256']
    assert sha(OUT/'manifest.json')==frozen['manifest_sha256']
    assert sha(DOC/'预先方案.md')==frozen['protocol_sha256']
    paths=list((OUT/'test').glob('*.pckl'))
    assert (OUT/'frozen_readout.json').stat().st_mtime < min(p.stat().st_mtime for p in paths)
    fitted=load(OUT/'readouts.pckl');test=data('test');keys=test['keys'];labels=test['labels'];synsets=test['synsets']
    assert not ({k[0] for k in keys} & {k[0] for k in fitted['training_keys']+fitted['calibration_keys']})
    scores={};endpoints={};timeline=[];accuracies=[];effects=[];categories=[]
    for name,head in fitted['heads'].items():
        for condition,features in test['features'].items():
            s=score(head,features);scores[(name,condition)]=s
            for t in range(15):
                timeline.append(dict(readout=name,condition=condition,timestep=t,total_calls=int(test['calls'][condition][0,t]),
                                     accuracy_pct=100*float(np.mean(s[:,t].argmax(-1)==labels)),mean_score_sd=float(s[:,t].std(-1).mean())))
            specs={f't{t}':s[:,t] for t in (0,1,3,9)}
            specs['mean_t0_t3_raw']=s[:,:4].mean(1)
            specs['mean_t0_t3_calibrated']=(s[:,:4]/fitted['temperatures'][name][None,:4,None]).mean(1)
            for budget in (9,15):
                assert np.all(test['calls'][condition]==test['calls'][condition][0])
                t=budget_index(test['calls'][condition][0],budget)
                specs[f'total_budget_{budget}']=s[:,t]
            for budget in (4,10):
                assert np.all(test['target_calls'][condition]==test['target_calls'][condition][0])
                t=budget_index(test['target_calls'][condition][0],budget)
                specs[f'target_budget_{budget}']=s[:,t]
            for specification,v in specs.items():
                pred=v.argmax(-1);correct=pred==labels
                endpoints[(name,condition,specification)]=correct
                accuracies.append(dict(readout=name,condition=condition,endpoint=specification,accuracy_pct=100*float(correct.mean())))
                if specification in ('total_budget_9','mean_t0_t3_raw','mean_t0_t3_calibrated'):
                    for c in range(5):
                        mask=labels==c
                        categories.append(dict(readout=name,condition=condition,endpoint=specification,category=c,
                                               accuracy_pct=100*float(correct[mask].mean()),prediction_counts=json.dumps(np.bincount(pred[mask],minlength=5).tolist())))
        specs=sorted({k[2] for k in endpoints if k[0]==name})
        for specification in specs:
            for control in CONDITIONS:
                if control=='carry_exact':continue
                effects.append(dict(readout=name,endpoint=specification,contrast='carry_exact-minus-'+control,
                                    **paired_stats(endpoints[(name,'carry_exact',specification)],endpoints[(name,control,specification)],keys,synsets)))
    primary=[r for r in effects if r['readout']=='pooled' and r['endpoint']=='total_budget_9' and r['contrast'].removeprefix('carry_exact-minus-') in frozen['primary_controls']]
    order=np.argsort([r['signflip_p'] for r in primary]);running=0
    for rank,j in enumerate(order):
        running=max(running,min(1.,primary[j]['signflip_p']*(len(primary)-rank)))
        primary[j]['holm_p']=running
    for row in effects:row.setdefault('holm_p','')
    write('test_accuracy.csv',accuracies);write('test_contrasts.csv',effects);write('test_timeline.csv',timeline)
    write('test_categories.csv',categories);write('primary_contrasts.csv',primary)
    save(OUT/'test_predictions.pckl',dict(keys=keys,labels=labels,synsets=synsets,scores=scores,endpoints=endpoints))
    immutable_json(DOC/'test_audit.json',dict(files=len(paths),images=len({k[0] for k in keys}),sequences=len(keys)*7,
                   split_isolation=True,paired_targets=True,finite=True,preview_target_isolation=True,
                   readout_frozen_before_test=True,backbone_budgets_verified=True,
                   readout_sha256=frozen['readouts_sha256']))
    print('PRIMARY',json.dumps(primary,indent=2),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    names={'carry_exact':'Exact preview','carry_same_synset':'Same synset','carry_different_category':'Different category','none_random':'No memory, random init','none_feedforward':'No memory, feedforward init'}
    for ax,head in zip(axes,('legacy','pooled')):
        for condition,label in names.items():
            rows=[r for r in timeline if r['readout']==head and r['condition']==condition]
            ax.plot([r['total_calls'] for r in rows],[r['accuracy_pct'] for r in rows],marker='.',label=label)
        ax.axvline(9,color='gray',linestyle='--',linewidth=.8)
        ax.set_title(head+' frozen readout');ax.set_xlabel('Total backbone calls, including previews and initialization')
        ax.grid(alpha=.2);ax.set_xlim(1,20)
    axes[0].set_ylabel('Five-category accuracy (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Independent test: readout, initialization and matched compute')
    fig.tight_layout();fig.savefig(DOC/'equal_budget_readout.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    {'fit':fit,'freeze':freeze,'analyze':analyze}[sys.argv[1]]()
