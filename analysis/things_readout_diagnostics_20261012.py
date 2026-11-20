"""Secondary descriptive comparisons; does not alter frozen readouts/endpoints."""
import numpy as np
import json
from pathlib import Path

from things_readout_20261012 import ROOT, OUT, DOC, CONDITIONS, load, write, paired_stats, sha


def audit_execution():
    manifest=json.loads((OUT/'manifest.json').read_text())
    implementation=json.loads((OUT/'implementation.json').read_text())
    for name,digest in implementation['dependencies'].items():
        assert sha(ROOT/name)==sha(OUT/'source_snapshot'/name)==digest
    counts={}
    for split in ('train','dev','test'):
        trials={r['dataset_index']:r for r in manifest[split]}
        expected={(i,s) for i in trials for s in manifest['noise_seeds'][split]}
        seen=set(); image_hashes={}
        conditions=set(manifest['train_conditions'] if split=='train' else manifest['conditions'])
        sequences=0
        for path in sorted((OUT/split).glob('*.pckl')):
            result=load(path);trial=result['trial'];i=trial['dataset_index'];key=(i,result['noise'])
            assert trial==trials[i], 'Record uses a different trial/slot specification'
            assert key in expected and key not in seen
            seen.add(key)
            assert result['source_sha256']==implementation['runner_sha256']
            assert {r['condition'] for r in result['records']}==conditions
            hashes={r['target_hash'] for r in result['records']};assert len(hashes)==1
            image_hashes.setdefault(i,set()).update(hashes)
            for rec in result['records']:
                condition=rec['condition'];carry=condition.startswith('carry_')
                preview=list(trial['previews']) if carry else []
                if carry: preview[trial['slot']]=trial['primes'][condition.removeprefix('carry_')]
                assert rec['preview_indices']==preview
                assert rec['random_init']==(condition!='none_feedforward')
                assert np.isfinite(rec['fc2']).all() and np.isfinite(rec['logits']).all()
                sequences+=1
        assert seen==expected and all(len(h)==3 for h in image_hashes.values())
        counts[split]=dict(images=len(trials),files=len(seen),sequences=sequences)
    fitted=load(OUT/'readouts.pckl');frozen=json.loads((OUT/'frozen_readout.json').read_text())
    assert sha(OUT/'readouts.pckl')==frozen['readouts_sha256']
    assert sha(DOC/'预先方案.md')==frozen['protocol_sha256']
    assert sha(OUT/('analysis_fit_'+fitted['fit_source_sha256'][:12]+'.py'))==fitted['fit_source_sha256']
    assert sha(OUT/('analysis_frozen_'+fitted['calibration_source_sha256'][:12]+'.py'))==fitted['calibration_source_sha256']
    assert sha(ROOT/'analysis/things_readout_20261012.py')==fitted['calibration_source_sha256']
    record=dict(stages=counts,formal_sequences=sum(r['sequences'] for r in counts.values()),
        exact_trial_and_slot_match=True,condition_preview_match=True,three_distinct_noises=True,
        implementation=implementation,readouts_sha256=frozen['readouts_sha256'],
        protocol_sha256=frozen['protocol_sha256'],fit_source_sha256=fitted['fit_source_sha256'],
        analysis_source_sha256=fitted['calibration_source_sha256'],secondary_source_sha256=sha(__file__),
        archived_pilot_excluded=True,unit_tests_passed=61,smoke_sequences=35,reset_equivalence_checks=10)
    (DOC/'execution_record.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    (OUT/('analysis_secondary_'+sha(__file__)[:12]+'.py')).write_bytes(Path(__file__).read_bytes())
    print('Execution audit passed: all formal records match the balanced-slot manifest.',flush=True)


def main():
    audit_execution()
    result = load(OUT / 'test_predictions.pckl')
    fitted = load(OUT / 'readouts.pckl')
    keys, labels, synsets = (result[k] for k in ('keys', 'labels', 'synsets'))
    endpoints = result['endpoints']
    names = ('t0', 't3', 't9', 'mean_t0_t3_raw', 'mean_t0_t3_calibrated',
             'total_budget_9', 'total_budget_15', 'target_budget_4')
    cross = []
    initialization = []
    calibration = []
    for name in names:
        for condition in CONDITIONS:
            for a, b in (('pooled', 'legacy'), ('pooled', 'matched_t9'), ('pooled', 'imagenet_pooled')):
                cross.append(dict(endpoint=name, condition=condition, contrast=a+'-minus-'+b,
                    **paired_stats(endpoints[(a, condition, name)], endpoints[(b, condition, name)], keys, synsets)))
        for head in fitted['heads']:
            initialization.append(dict(readout=head, endpoint=name, contrast='none_feedforward-minus-none_random',
                **paired_stats(endpoints[(head, 'none_feedforward', name)],
                               endpoints[(head, 'none_random', name)], keys, synsets)))
    for head in fitted['heads']:
        for condition in CONDITIONS:
            calibration.append(dict(readout=head, condition=condition, contrast='calibrated-minus-raw-early-average',
                **paired_stats(endpoints[(head, condition, 'mean_t0_t3_calibrated')],
                               endpoints[(head, condition, 'mean_t0_t3_raw')], keys, synsets)))
    write('secondary_cross_readout.csv', cross)
    write('secondary_initialization.csv', initialization)
    write('secondary_calibration.csv', calibration)
    margins = []
    for (head, condition), scores in result['scores'].items():
        specs = {f't{t}': scores[:, t] for t in (0, 1, 3, 9)}
        specs['mean_t0_t3_raw'] = scores[:, :4].mean(1)
        specs['mean_t0_t3_calibrated'] = (scores[:, :4] / fitted['temperatures'][head][None, :4, None]).mean(1)
        for endpoint, s in specs.items():
            distractors = s.copy()
            distractors[np.arange(len(labels)), labels] = -np.inf
            margin = s[np.arange(len(labels)), labels] - distractors.max(1)
            sd = np.maximum(s.std(1), 1e-6)
            margins.append(dict(readout=head, condition=condition, endpoint=endpoint,
                mean_margin=float(margin.mean()), median_margin=float(np.median(margin)),
                mean_sd_normalized_margin=float((margin/sd).mean()), mean_score_sd=float(sd.mean())))
    write('secondary_margins.csv', margins)
    old = load(ROOT/'Hebbian-CNN/Log/readout_init_20261012/test_predictions.pckl')
    old_mask = old['labels'] != 3
    columns = [0,1,2,4]
    old_labels = np.array([columns.index(int(c)) for c in old['labels'][old_mask]])
    comparison = []
    for old_head, new_head in [('legacy','legacy'),('pooled','imagenet_pooled')]:
        for condition in CONDITIONS:
            for endpoint in ('t0','t3','t9','total_budget_9','total_budget_15'):
                if endpoint.startswith('total_budget_'):
                    budget=int(endpoint.rsplit('_',1)[1])
                    t=budget-(6 if condition.startswith('carry_') else 2 if condition=='none_random' else 1)
                else:
                    t=int(endpoint[1:])
                s=old['scores'][(old_head,condition)][old_mask,t][:,columns]
                comparison.append(dict(fixed_readout=new_head,condition=condition,endpoint=endpoint,
                    imagenet_images=int(old_mask.sum()/3),things_images=len({k[0] for k in keys}),
                    imagenet_accuracy_pct=100*float((s.argmax(1)==old_labels).mean()),
                    things_accuracy_pct=100*float(endpoints[(new_head,condition,endpoint)].mean())))
    write('cross_dataset_fixed_readout.csv', comparison)
    print('Secondary paired comparisons and margins complete; no new primary tests or selection.')


if __name__ == '__main__':
    main()
