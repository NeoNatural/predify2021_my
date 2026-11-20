"""Decision-direction audit and image-clustered supplementary statistics."""
from pathlib import Path
import csv
import json
import pickle
import sys
import numpy as np
from scipy.stats import binomtest, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predify_experiments.supplementary import OUT as DATA, PAPER, load, CARRIERS, IDENTITIES
OUT = ROOT / 'docs/supplementary_20261009'
OUT.mkdir(parents=True, exist_ok=True)


def write(name, rows):
    if not rows:
        return
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def margin(scores, labels):
    labels = np.asarray(labels)
    correct = scores[np.arange(len(scores)), labels]
    other = scores.copy()
    other[np.arange(len(scores)), labels] = -np.inf
    return correct - other.max(axis=-1)


def interval(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261009)
    draws=rng.integers(len(values),size=(10000,len(values)))
    low,high=np.quantile(values[draws].mean(axis=1),[.025,.975])
    return dict(delta=float(values.mean()),low=float(low),high=float(high))


def stage_a():
    decoders=load(PAPER/'shared_decoder/analysis/shared_decoders.pckl')['classifiers']
    results={}
    for p in (PAPER/'causal_pilot_calibrated').glob('*.pckl'):
        r=load(p);m=r['run_metadata'];results[(m['degradation_mode'],m['memory_carrier'],m['prime_identity'])]=r
    summaries, individual, confusion, temporal=[],[],[],[]
    for mode in ('predify_noise','fractal_occlusion'):
        baseline=next(r for (m,c,i),r in results.items() if m==mode and c=='none')
        y=baseline['category_labels']
        for carrier in CARRIERS:
            same=results[(mode,carrier,'same')];different=results[(mode,carrier,'different')]
            for t in range(10):
                clf=decoders[(mode,'fc2',t)]
                records=[same,different,baseline]
                scores=[clf.decision_function(r['target_representations']['fc2'][:,t].astype(np.float32)) for r in records]
                margins=[margin(s,y) for s in scores]
                predictions=[s.argmax(-1) for s in scores]
                correct=[p==y for p in predictions]
                for name,a,b in [('same-none',0,2),('different-none',1,2),('same-different',0,1)]:
                    delta=margins[a]-margins[b]
                    recovery=records[a]['representation_recovery'][:,t,4,0]-records[b]['representation_recovery'][:,t,4,0]
                    activation=records[a]['target_representations']['fc2'][:,t].astype(float).mean(1)-records[b]['target_representations']['fc2'][:,t].astype(float).mean(1)
                    rho,p=spearmanr(recovery,delta)
                    summaries.append(dict(mode=mode,carrier=carrier,timestep=t,contrast=name,
                        accuracy_delta_pp=100*float(np.mean(correct[a].astype(float)-correct[b].astype(float))),
                        wins=int(np.sum(correct[a]&~correct[b])),losses=int(np.sum(~correct[a]&correct[b])),
                        **interval(delta),recovery_margin_spearman=float(rho),correlation_p=float(p),
                        recovery_up_margin_down=int(np.sum((recovery>0)&(delta<0))),
                        activation_delta_mean=float(activation.mean())))
                    for j,index in enumerate(baseline['dataset_indices']):
                        individual.append(dict(mode=mode,carrier=carrier,timestep=t,contrast=name,index=int(index),
                            label=int(y[j]),margin_delta=float(delta[j]),recovery_delta=float(recovery[j]),
                            activation_delta=float(activation[j]),correct_a=bool(correct[a][j]),correct_b=bool(correct[b][j])))
                    for true in range(5):
                        for pred in range(5):
                            confusion.append(dict(mode=mode,carrier=carrier,timestep=t,contrast=name,true=true,predicted=pred,
                                count_delta=int(np.sum((y==true)&(predictions[a]==pred))-np.sum((y==true)&(predictions[b]==pred)))))
                if t>0:
                    previous=decoders[(mode,'fc2',t-1)]
                    for identity,r,current in zip(('same','different','none'),records,correct):
                        prev=previous.predict(r['target_representations']['fc2'][:,t-1].astype(np.float32))==y
                        temporal.append(dict(mode=mode,carrier=carrier,identity=identity,timestep=t,
                            new_errors=int(np.sum(prev&~current)),recovered_errors=int(np.sum(~prev&current))))
    write('stage_a_margin_summary.csv',summaries)
    write('stage_a_per_image.csv',individual)
    write('stage_a_confusion.csv',confusion)
    write('stage_a_temporal_errors.csv',temporal)
    print('STAGE A t9 noise',json.dumps([r for r in summaries if r['mode']=='predify_noise' and r['timestep']==9],indent=2))


def audit_design():
    manifest=load(DATA/'manifest.pckl')
    paths=sorted((DATA/'stage_b').glob('*.pckl'))
    expected={(c,i) for c in CARRIERS for i in IDENTITIES}|{('none','exact'),('target_only','exact')}
    seen=set();rows=[]
    for path in paths:
        d=load(path);trial=d['trial'];records=d['records'];index=trial['dataset_index']
        key=(index,d['noise_seed']);assert key not in seen;seen.add(key)
        assert {(r['carrier'],r['identity']) for r in records}==expected
        assert len(records)==17 and len({r['target_hash'] for r in records})==1
        assert key[1] in manifest['noise_seeds']
        assert index not in manifest['excluded_previous_images'] and index not in manifest['confirmation_indices']
        none=next(r for r in records if r['carrier']=='none')
        target=next(r for r in records if r['carrier']=='target_only')
        assert np.array_equal(none['logits'],target['logits'])
        for layer in none['features']:
            assert np.array_equal(none['features'][layer],target['features'][layer])
        for r in records:
            assert np.isfinite(r['logits']).all() and np.isfinite(r['recovery']).all()
            assert r['recovery'].shape==(10,2,5,3)
            assert not (set(r['preview_indices'])&set(manifest['confirmation_indices']))
        rows.append(dict(file=path.name,index=index,noise_seed=key[1],records=17,
            finite=True,target_paired=True,none_target_only_exact=True,confirmation_isolated=True))
    required={(r['dataset_index'],s) for r in manifest['exploration'] for s in manifest['noise_seeds']}
    assert seen==required,'Stage B not complete'
    write('stage_b_file_audit.csv',rows)
    summary=dict(passed=True,files=len(paths),images=200,conditions=17,noise_realizations=3,
                 trial_sequences=len(paths)*17,assertions=list(rows[0])[4:],
                 confirmation_images_reserved=len(manifest['confirmation_indices']))
    (OUT/'stage_b_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


def analyze_b():
    paths=sorted((DATA/'stage_b').glob('*.pckl'))
    if not paths:
        return
    manifest=load(DATA/'manifest.pckl')
    assert len(paths)==len(manifest['exploration'])*3, 'Incomplete stage B'
    ids=sorted(r['dataset_index'] for r in manifest['exploration'])
    synsets={r['dataset_index']:r['imagenet_label'] for r in manifest['exploration']}
    scores, recovery, activation, seconds={},{},{},{}
    for path in paths:
        d=load(path)
        i=d['trial']['dataset_index'];s=d['noise_seed'];y=d['trial']['category_index']
        for r in d['records']:
            key=r['carrier'],r['identity']
            scores.setdefault(key,{})[(i,s)]=(r['scores']['fc2'],y)
            recovery.setdefault(key,{})[(i,s)]=r['recovery'][:,:,4,0]
            activation.setdefault(key,{})[(i,s)]=np.abs(r['features']['fc2'].astype(float)).mean(-1)
            seconds.setdefault(key,[]).append(r['seconds'])
    keys=sorted(next(iter(scores.values())))
    labels=np.array([scores[('none','exact')][k][1] for k in keys])
    values={k:np.stack([v[q][0] for q in keys]) for k,v in scores.items()}
    correct={k:v.argmax(-1)==labels[:,None] for k,v in values.items()}
    reference_keys=sorted(scores[('none','exact')])
    assert keys==reference_keys
    for key in scores:
        assert sorted(scores[key])==keys
    accuracy=[]
    for key,c in correct.items():
        for t in range(10):
            accuracy.append(dict(carrier=key[0],identity=key[1],timestep=t,accuracy_pct=100*float(c[:,t].mean()),
                images=len(ids),trials=len(keys),mean_target_seconds=float(np.mean(seconds[key]))))
    write('stage_b_accuracy.csv',accuracy)
    early_rows=[]
    for specification in ('t1','t3','mean_t0_t3','t9'):
        baseline_scores=endpoint(values[('none','exact')],specification)
        baseline_correct=baseline_scores.argmax(-1)==labels
        for identity in IDENTITIES:
            current_scores=endpoint(values[('pcoder_carry',identity)],specification)
            current_correct=current_scores.argmax(-1)==labels
            for contrast,control_key in [('none',('none','exact')),('different',('pcoder_carry','different_category'))]:
                control_scores=endpoint(values[control_key],specification)
                control_correct=control_scores.argmax(-1)==labels
                trial_d=(current_correct.astype(float)-control_correct.astype(float))*100
                cluster_d=np.array([trial_d[[q[0]==i for q in keys]].mean() for i in ids])
                early_rows.append(dict(endpoint=specification,identity=identity,contrast=contrast,
                    accuracy_pct=100*float(current_correct.mean()),control_accuracy_pct=100*float(control_correct.mean()),
                    **interval(cluster_d),margin_delta=float(np.mean(margin(current_scores,labels)-margin(control_scores,labels)))))
    write('stage_b_early_endpoints.csv',early_rows)
    budget_rows=[]
    for t in range(1,10):
        av=correct[('pcoder_carry','exact')][:,t]
        bv=correct[('none','exact')][:,t-1]
        trial_d=100*(av.astype(float)-bv.astype(float))
        cluster_d=np.array([trial_d[[q[0]==i for q in keys]].mean() for i in ids])
        budget_rows.append(dict(pcoder_timestep=t,none_timestep=t-1,target_backbone_calls=t+1,
            pcoder_total_backbone_calls=5+t+1,none_total_backbone_calls=6+t+1,
            **interval(cluster_d),scope='same target forward count; preview forward counts differ'))
    write('stage_b_target_compute_aligned.csv',budget_rows)
    effects=[]
    for carrier in CARRIERS:
        for identity in IDENTITIES:
            a=(carrier,identity)
            contrasts=[('none',('none','exact'))]
            if identity=='exact':
                contrasts += [(x,(carrier,x)) for x in IDENTITIES[1:]]
            for contrast,b in contrasts:
                for t in range(10):
                    av,bv=correct[a][:,t],correct[b][:,t]
                    trial_d=(av.astype(float)-bv.astype(float))*100
                    cluster_d=np.array([trial_d[[q[0]==i for q in keys]].mean() for i in ids])
                    win=int(np.sum(av&~bv));loss=int(np.sum(~av&bv))
                    cluster_sum=np.array([trial_d[[q[0]==i for q in keys]].sum() for i in ids])
                    rng=np.random.default_rng(20261009)
                    permutations=(rng.integers(2,size=(20000,len(ids)))*2-1)@cluster_sum
                    cluster_p=(np.sum(np.abs(permutations)>=abs(cluster_sum.sum())-1e-9)+1)/20001
                    margins_a=margin(values[a][:,t],labels);margins_b=margin(values[b][:,t],labels)
                    cluster_margin=np.array([(margins_a-margins_b)[[q[0]==i for q in keys]].mean() for i in ids])
                    synset_labels=np.array([synsets[i] for i in ids])
                    unique_synsets=np.unique(synset_labels)
                    sums=np.array([cluster_d[synset_labels==s].sum() for s in unique_synsets])
                    sizes=np.array([np.sum(synset_labels==s) for s in unique_synsets])
                    draws=np.random.default_rng(20261009).integers(len(sums),size=(10000,len(sums)))
                    synset_low,synset_high=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975])
                    recover_d=np.array([recovery[a][q][t,1]-recovery[b][q][t,1] for q in keys])
                    cluster_recover=np.array([recover_d[[q[0]==i for q in keys]].mean() for i in ids])
                    recover_ff=np.array([recovery[a][q][t,0]-recovery[b][q][t,0] for q in keys])
                    act_d=np.array([(activation[a][q][t]-activation[b][q][t])/max(abs(activation[b][q][t]),1e-12)*100 for q in keys])
                    rho,rho_p=spearmanr(cluster_margin,cluster_recover)
                    # McNemar is reported per realization, never pooling dependent noises.
                    exact_ps=[]
                    for noise in manifest['noise_seeds']:
                        mask=np.array([q[1]==noise for q in keys])
                        w=int(np.sum(av[mask]&~bv[mask]));l=int(np.sum(~av[mask]&bv[mask]))
                        exact_ps.append(float(binomtest(w,w+l).pvalue) if w+l else 1.)
                    effects.append(dict(carrier=carrier,identity=identity,contrast=contrast,timestep=t,
                        **interval(cluster_d),wins_trials=win,losses_trials=loss,cluster_signflip_p=float(cluster_p),
                        mcnemar_per_noise=json.dumps(exact_ps),margin_delta=float(np.mean(margins_a-margins_b)),
                        synset_low_pp=float(synset_low),synset_high_pp=float(synset_high),
                        recovery_t9_reference_delta=float(recover_d.mean()),recovery_ff_reference_delta=float(recover_ff.mean()),
                        recovery_margin_spearman=float(rho),correlation_p=float(rho_p),activation_delta_pct=float(act_d.mean())))
    write('stage_b_contrasts.csv',effects)
    categories=[]
    for key,c in correct.items():
        for category in range(5):
            mask=labels==category
            categories.append(dict(carrier=key[0],identity=key[1],category=category,
                accuracy_t9_pct=100*float(c[mask,9].mean()),trials=int(mask.sum())))
    write('stage_b_categories.csv',categories)
    save_predictions=dict(keys=keys,labels=labels,scores=values,correctness=correct)
    with (OUT/'stage_b_predictions.pckl').open('wb') as f:
        pickle.dump(save_predictions,f,protocol=pickle.HIGHEST_PROTOCOL)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    baseline=100*float(correct[('none','exact')][:,9].mean())
    for carrier,color in zip(CARRIERS,('#1976d2','#00897b','#d97706')):
        y=[100*float(correct[(carrier,x)][:,9].mean()) for x in IDENTITIES]
        axes[0].plot(range(5),y,'o-',label=carrier,color=color)
        select=[r for r in accuracy if r['carrier']==carrier and r['identity']=='exact']
        axes[1].plot([r['timestep'] for r in select],[r['accuracy_pct'] for r in select],label=carrier,color=color)
    axes[0].axhline(baseline,color='black',ls='--',label='no memory')
    select=[r for r in accuracy if r['carrier']=='none']
    axes[1].plot([r['timestep'] for r in select],[r['accuracy_pct'] for r in select],'k--',label='no memory')
    axes[0].set_xticks(range(5),['Exact','Same synset','Same category','Different category','Neutral'],rotation=20)
    axes[0].set_title('Frozen FC2 t9 head: endpoint by prime content')
    axes[1].set_title('Exact prime: one common frozen head over time')
    for ax in axes:
        ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[1].set_xlabel('Target timestep')
    fig.tight_layout();fig.savefig(OUT/'stage_b_results.png',dpi=170);plt.close(fig)
    print('STAGE B t9',json.dumps([r for r in effects if r['timestep']==9],indent=2))


def analyze_c():
    paths=sorted((DATA/'stage_c').glob('*.pckl'))
    assert len(paths)==150, 'Incomplete stage C'
    records={}
    ids=set()
    labels={}
    for path in paths:
        d=load(path);i=d['trial']['dataset_index'];s=d['noise_seed'];ids.add(i)
        labels[(i,s)]=d['trial']['category_index']
        for r in d['records']:
            key=r['carrier'],r['identity'],r['scale'],r['inhibition']
            records.setdefault(key,{})[(i,s)]=r['scores']['fc2']
    ids=sorted(ids);keys=sorted(labels);y=np.array([labels[k] for k in keys])
    values={k:np.stack([v[q] for q in keys]) for k,v in records.items()}
    correct={k:v.argmax(-1)==y[:,None] for k,v in values.items()}
    effects=[]
    baseline=('none','exact',1.,4)
    for carrier in ('hebb_only','hebb_and_pcoder'):
        for scale,inhibition in ((1.,4),(.25,4),(1.,0),(0.,4)):
            a=(carrier,'exact',scale,inhibition)
            for contrast,b in [('none',baseline),('different',(carrier,'different_category',scale,inhibition)),('current',(carrier,'exact',1.,4))]:
                av,bv=correct[a][:,9],correct[b][:,9]
                trial_d=(av.astype(float)-bv.astype(float))*100
                cluster_d=np.array([trial_d[[q[0]==i for q in keys]].mean() for i in ids])
                effects.append(dict(carrier=carrier,scale=scale,inhibition=inhibition,contrast=contrast,images=len(ids),
                    exact_accuracy_pct=100*float(av.mean()),control_accuracy_pct=100*float(bv.mean()),
                    **interval(cluster_d),wins_trials=int(np.sum(av&~bv)),losses_trials=int(np.sum(~av&bv)),
                    margin_delta=float(np.mean(margin(values[a][:,9],y)-margin(values[b][:,9],y)))))
    write('stage_c_contrasts.csv',effects)
    candidates=[]
    for r in effects:
        if r['contrast']=='none' and r['scale']>0:
            diff=next(x for x in effects if x['carrier']==r['carrier'] and x['scale']==r['scale'] and x['inhibition']==r['inhibition'] and x['contrast']=='different')
            if r['delta']>0 and diff['delta']>0 and r['margin_delta']>0:
                candidates.append(dict(carrier=r['carrier'],scale=r['scale'],inhibition=r['inhibition'],
                                       timestep=9,net_delta_pp=r['delta'],same_different_delta_pp=diff['delta']))
    candidate=max(candidates,key=lambda r:min(r['net_delta_pp'],r['same_different_delta_pp'])) if candidates else None
    (OUT/'stage_c_candidate.json').write_text(json.dumps(dict(candidate=candidate,rule='positive net and identity accuracy, positive net margin; maximize weaker accuracy contrast; exploratory selection'),indent=2),encoding='utf-8')
    print('STAGE C',json.dumps(effects,indent=2));print('CANDIDATE',candidate)


def endpoint(scores, specification):
    if specification=='mean_t0_t3':
        return scores[:,:4].mean(axis=1)
    return scores[:,int(specification.removeprefix('t'))]


def freeze_candidate():
    # Candidates use only exploration data; confirmation is still unopened.
    data=load(OUT/'stage_b_predictions.pckl')
    scores=data['scores'];labels=data['labels'];candidates=[]
    for specification in ('t1','t3','mean_t0_t3','t9'):
        a=endpoint(scores[('pcoder_carry','exact')],specification)
        b=endpoint(scores[('none','exact')],specification)
        d=endpoint(scores[('pcoder_carry','different_category')],specification)
        net=100*float(np.mean((a.argmax(-1)==labels).astype(float)-(b.argmax(-1)==labels).astype(float)))
        identity=100*float(np.mean((a.argmax(-1)==labels).astype(float)-(d.argmax(-1)==labels).astype(float)))
        dm=float(np.mean(margin(a,labels)-margin(b,labels)))
        if net>0 and identity>0 and dm>0:
            candidates.append(dict(carrier='pcoder_carry',scale=1.,inhibition=4,endpoint=specification,
                                    net_delta_pp=net,same_different_delta_pp=identity,selection_source='stage_b'))
    c_path=OUT/'stage_c_candidate.json'
    if c_path.exists():
        c=json.loads(c_path.read_text(encoding='utf-8'))['candidate']
        if c:
            c['endpoint']='t9';c['selection_source']='stage_c';candidates.append(c)
    candidate=max(candidates,key=lambda c:min(c['net_delta_pp'],c['same_different_delta_pp'])) if candidates else None
    frozen=dict(candidate=candidate,readout='pre-existing frozen noise FC2 t9',candidates=candidates,
                rule='positive net and identity accuracy and positive net margin; maximize smaller accuracy gain',
                minimum_useful_effect_pp=2.,confirmation_images=200,confirmation_noise_seeds=[202610191,202610192,202610193])
    (OUT/'selection_decision.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    if candidate:
        frozen_path=OUT/'frozen_candidate.json'
        if frozen_path.exists():
            if json.loads(frozen_path.read_text(encoding='utf-8'))!=candidate:
                raise ValueError('Refusing to replace an existing frozen candidate; use a new study output directory')
        else:
            frozen_path.write_text(json.dumps(candidate,indent=2),encoding='utf-8')
    print(json.dumps(frozen,indent=2))


def analyze_d(occlusion=False):
    prefix='stage_e_occlusion' if occlusion else 'stage_d'
    paths=sorted((DATA/prefix).glob('*.pckl'))
    assert len(paths)==600, 'Incomplete confirmation'
    frozen=json.loads((OUT/'frozen_candidate.json').read_text(encoding='utf-8'))
    manifest=load(DATA/'manifest.pckl')
    assert (OUT/'frozen_candidate.json').stat().st_mtime < min(p.stat().st_mtime for p in paths), 'Candidate was modified after confirmation started'
    scores={};labels={};synsets={};ids=set();recoveries={};activations={};durations={}
    for path in paths:
        d=load(path);i=d['trial']['dataset_index'];s=d['noise_seed'];ids.add(i)
        assert i in manifest['confirmation_indices'] and i not in manifest['excluded_previous_images']
        assert s in (202610191,202610192,202610193)
        assert {(r['carrier'],r['identity']) for r in d['records']}=={(frozen['carrier'],'exact'),(frozen['carrier'],'different_category'),('none','exact')}
        assert len(d['records'])==3 and len({r['target_hash'] for r in d['records']})==1
        assert (i,s) not in labels
        for r in d['records']:
            assert np.isfinite(r['logits']).all() and np.isfinite(r['recovery']).all()
            assert not ((set(r['preview_indices']) & set(manifest['confirmation_indices'])) - {i})
        labels[(i,s)]=d['trial']['category_index'];synsets[i]=d['trial']['imagenet_label']
        for r in d['records']:
            key=r['carrier'],r['identity']
            scores.setdefault(key,{})[(i,s)]=r['scores']['fc2']
            recoveries.setdefault(key,{})[(i,s)]=r['recovery'][:,1,4,0]
            activations.setdefault(key,{})[(i,s)]=np.abs(r['features']['fc2'].astype(float)).mean(-1)
            durations.setdefault(key,[]).append(r['seconds'])
    keys=sorted(labels);ids=sorted(ids);y=np.array([labels[q] for q in keys])
    assert set(ids)==set(manifest['confirmation_indices'])
    assert len(keys)==600
    (OUT/f'{prefix}_audit.json').write_text(json.dumps(dict(passed=True,files=600,images=200,new_sequences=1800,
        candidate_frozen_before_results=True,paired_targets=True,finite=True,target_images_disjoint=True),indent=2),encoding='utf-8')
    values={k:endpoint(np.stack([v[q] for q in keys]),frozen['endpoint']) for k,v in scores.items()}
    timeline_rows=[]
    for key,records in scores.items():
        full=np.stack([records[q] for q in keys])
        for t in range(10):
            timeline_rows.append(dict(carrier=key[0],identity=key[1],timestep=t,
                                     accuracy_pct=100*float(np.mean(full[:,t].argmax(-1)==y))))
    write(f'{prefix}_timeline.csv',timeline_rows)
    correct={k:v.argmax(-1)==y for k,v in values.items()}
    a=(frozen['carrier'],'exact');effects=[]
    for contrast,b in [('none',('none','exact')),('different',(frozen['carrier'],'different_category'))]:
        av,bv=correct[a],correct[b]
        trial_d=(av.astype(float)-bv.astype(float))*100
        cluster_d=np.array([trial_d[[q[0]==i for q in keys]].mean() for i in ids])
        cluster_sum=np.array([trial_d[[q[0]==i for q in keys]].sum() for i in ids])
        synset_labels=np.array([synsets[i] for i in ids]);unique_synsets=np.unique(synset_labels)
        sums=np.array([cluster_d[synset_labels==s].sum() for s in unique_synsets])
        sizes=np.array([np.sum(synset_labels==s) for s in unique_synsets])
        draws=np.random.default_rng(20261009).integers(len(sums),size=(10000,len(sums)))
        synset_low,synset_high=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975])
        rng=np.random.default_rng(20261009)
        perms=(rng.integers(2,size=(20000,len(ids)))*2-1)@cluster_sum
        p=(np.sum(np.abs(perms)>=abs(cluster_sum.sum())-1e-9)+1)/20001
        exact=[]
        for noise in (202610191,202610192,202610193):
            mask=np.array([q[1]==noise for q in keys]);w=int(np.sum(av[mask]&~bv[mask]));l=int(np.sum(~av[mask]&bv[mask]))
            exact.append(float(binomtest(w,w+l).pvalue) if w+l else 1.)
        times=list(range(4)) if frozen['endpoint']=='mean_t0_t3' else [int(frozen['endpoint'][1:])]
        recover_delta=np.array([(recoveries[a][q][times]-recoveries[b][q][times]).mean() for q in keys])
        act_delta=np.array([((activations[a][q][times]-activations[b][q][times])/np.maximum(np.abs(activations[b][q][times]),1e-12)*100).mean() for q in keys])
        effects.append(dict(contrast=contrast,endpoint=frozen['endpoint'],images=len(ids),trials=len(keys),
            accuracy_same_pct=100*float(av.mean()),accuracy_control_pct=100*float(bv.mean()),**interval(cluster_d),
            wins_trials=int(np.sum(av&~bv)),losses_trials=int(np.sum(~av&bv)),
            cluster_signflip_p=float(p),mcnemar_per_noise=json.dumps(exact),
            margin_delta=float(np.mean(margin(values[a],y)-margin(values[b],y))),
            synset_low_pp=float(synset_low),synset_high_pp=float(synset_high),
            recovery_delta=float(recover_delta.mean()),activation_delta_pct=float(act_delta.mean()),
            mean_full_target_seconds=float(np.mean(durations[a]))))
    write(f'{prefix}_contrasts.csv',effects)
    success=all(r['low']>0 and r['cluster_signflip_p']<.05 for r in effects)
    ruled_out=next(r for r in effects if r['contrast']=='none')['high']<2
    outcome=dict(frozen=frozen,contrasts=effects,passed=success,minimum_useful_effect_pp=2.,
        useful_effect_ruled_out=ruled_out,conclusion='positive net benefit confirmed' if success else 'no confirmed benefit; interval excludes 2 pp' if ruled_out else 'insufficient evidence; 2 pp remains compatible')
    outcome['degradation']='fractal_occlusion_0.3' if occlusion else 'predify_noise_0.5'
    (OUT/('occlusion_outcome.json' if occlusion else 'confirmation_outcome.json')).write_text(json.dumps(outcome,indent=2),encoding='utf-8')
    print(json.dumps(outcome,indent=2))


if __name__=='__main__':
    if '--occlusion' in sys.argv:
        analyze_d(occlusion=True)
    elif '--audit' in sys.argv:
        audit_design()
    elif '--freeze' in sys.argv:
        freeze_candidate()
    elif '--stage-d' in sys.argv:
        analyze_d()
    elif '--stage-c' in sys.argv:
        analyze_c()
    elif '--stage-b' in sys.argv:
        analyze_b()
    else:
        stage_a()
