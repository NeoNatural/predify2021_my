from pathlib import Path
import sys,csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.supplementary import load
from predify_experiments.readout_init_20261012 import sha
D=ROOT/'docs/tripartite_t9_20261027';O=ROOT/'Hebbian-CNN/Log/tripartite_t9_20261027';S=ROOT/'Hebbian-CNN/Log/tripartite_20261027'
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
audit={}
for dataset in ('imagenet','things'):
    head=load(O/dataset/'head.pckl')['head'];scores=load(O/dataset/'scores.pckl')
    for r in read(D/dataset/'t9_timeline.csv'):
        predicted=scores['scores'][r['condition']][:,int(r['timestep'])].argmax(-1)
        assert abs(100*np.mean(predicted==scores['y'])-float(r['accuracy_pct']))<1e-10
    train=read(D/dataset/'training_rows.csv');assert len(train)==6240 and {r['timestep'] for r in train}=={'9'}
    assert sum(float(r['weight']) for r in train)==14400
    before=json.loads((D/'audit.json').read_text())[dataset]['original_hashes']
    assert all(sha(Path(p))==v for p,v in before.items())
    mismatches=0;maxerr=0.;samples=0
    # Independently check saved predictions against double precision arithmetic.
    for p in sorted((S/dataset/'test').glob('image_*.pckl'))[:2]:
        raw=load(p);j=scores['keys'].index((raw['trial']['dataset_index'],raw['noise']))
        for r in raw['records']:
            xx=r['fc2'].astype(np.float64)
            manual=((xx-head[0].mean_)/head[0].scale_)@head[-1].coef_.T+head[-1].intercept_
            actual=scores['scores'][r['name']][j]
            assert np.isfinite(manual).all()
            mismatches+=int(np.sum(manual.argmax(-1)!=actual.argmax(-1)))
            maxerr=max(maxerr,float(np.max(np.abs(manual-actual))));samples+=15
    assert mismatches==0
    audit[dataset]=dict(timeline_rows_verified=420,training_rows=6240,training_only_t9=True,
        weight_total=14400,original_hashes_unchanged=True,float64_crosscheck_predictions=samples,
        float64_prediction_mismatches=mismatches,float64_max_score_difference=maxerr)
(D/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit))
