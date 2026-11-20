"""Post-hoc t9-only frozen readout; never modifies the original frozen study."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,gc,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import analysis.tripartite_20261027 as a
from predify_experiments.supplementary import load,save
from predify_experiments.readout_init_20261012 import sha,immutable_json

DEST=ROOT/'docs/tripartite_t9_20261027'
OUT=ROOT/'Hebbian-CNN/Log/tripartite_t9_20261027'
PAIRS=[('joint-pcoder','joint_exact','pcoder_exact'),('joint-hebb','joint_exact','hebb_exact'),('pcoder-hebb','pcoder_exact','hebb_exact'),('hebb-none','hebb_exact','none')]

def sampler(d):
    # Collapsing the original 15-time balanced identity schedule onto t9 gives
    # the same per-image/content/carrier/strength loss mass, without duplicates.
    specs=[('none',15.)]+[(f'pcoder_{i}',3.) for i in a.IDENTITIES]
    specs += [(f'{c}_{prefix}{i}',1.5) for c in ('hebb','joint') for prefix in ('','weak_') for i in a.IDENTITIES]
    audit=[];x=[];y=[];w=[]
    for j,(image,noise) in enumerate(d['keys']):
        for c,weight in specs:
            x.append(d['features'][c][j,9]);y.append(d['y'][j]);w.append(weight)
            audit.append(dict(image=image,noise=noise,timestep=9,condition=c,weight=weight))
    x=np.asarray(x,np.float32);y=np.asarray(y);w=np.asarray(w)
    assert x.shape==(6240,4096) and w.sum()==14400
    for image in {k[0] for k in d['keys']}:
        r=[v for v in audit if v['image']==image]
        assert sum(v['weight'] for v in r)==180
        for carrier in ('none','pcoder','hebb','joint'):
            assert sum(v['weight'] for v in r if v['condition'].split('_')[0]==carrier)==45
    return x,y,w,audit

def holm(rows):
    order=np.argsort([r['signflip_p'] for r in rows]);running=0
    for rank,index in enumerate(order):
        running=max(running,(len(rows)-rank)*rows[index]['signflip_p'])
        rows[index]['holm_p']=min(running,1.)
    return rows

def main():
    DEST.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(status='post-hoc supplement; confirmation previously inspected',training_time=9,
        estimator='StandardScaler + LinearSVC(C=1, dual=auto); same seed and convergence policy',
        sampler='all five identities at t9, none weight15, PCoder identity weight3, Hebbian/joint identity-strength weight1.5',
        unique_rows=6240,total_loss_weight=14400,per_image_weight=180,
        rationale='exact per-image/carrier/content/strength weight totals match original common readout; fewer distinct temporal states',
        endpoints=['t9','full t0-t14 mean'],holm_family='2 datasets x 4 strong exact contrasts x 2 endpoints = 16',
        interaction_family='2 datasets x 2 strengths x 2 endpoints = 8; separate exploratory Holm',
        intervals='class-stratified concept bootstrap10000; signflip20000; marginal95% CI',
        no_tuning_on_dev_or_test=True,no_new_backbone_runs=True,source_sha=sha(__file__))
    immutable_json(DEST/'analysis_plan.json',protocol)
    alltests=[];interactions=[];comparison=[];audits={}
    original_doc=a.DOC;a.DOC=DEST
    for dataset in ('imagenet','things'):
        dest=DEST/dataset;dest.mkdir(exist_ok=True);base=OUT/dataset;base.mkdir(exist_ok=True)
        originals=[a.OUT/dataset/f for f in ('head.pckl','freeze.json','manifest.json')]
        before={str(p):sha(p) for p in originals}
        print(dataset,'LOAD TRAIN',flush=True);d=a.data(dataset,'train')
        train_groups=set(d['groups']);x,y,w,records=sampler(d)
        # Independently verify exact loss-mass equivalence to the original sampler.
        from collections import defaultdict
        old_x,old_y,old_w,old_records=a.training_rows(d,json.loads((a.OUT/dataset/'manifest.json').read_text()))
        def totals(rr):
            result=defaultdict(float)
            for r in rr:result[(r['image'],r['condition'])]+=r['weight']
            return dict(result)
        assert totals(records)==totals(old_records)
        del old_x,old_y,old_w,old_records,d;gc.collect()
        headpath=base/'head.pckl'
        if headpath.exists():
            saved=load(headpath);assert saved['plan_sha']==sha(DEST/'analysis_plan.json');head=saved['head'];attempts=saved['attempts']
        else:
            print(dataset,'FIT',x.shape,'weight',w.sum(),flush=True)
            head,attempts=a.fit_svm(x,y,w)
            save(headpath,dict(head=head,attempts=attempts,plan_sha=sha(DEST/'analysis_plan.json'),training_time=9))
        a.write(dest/'training_rows.csv',records);a.write(dest/'fit.csv',attempts)
        immutable_json(base/'readout_frozen.json',dict(head_sha=sha(headpath),plan_sha=sha(DEST/'analysis_plan.json'),source_sha=sha(__file__),status='frozen before supplemental evaluation; existing confirmation already seen'))
        del x,y,w,records;gc.collect()
        print(dataset,'EVALUATE',attempts,flush=True);d=a.data(dataset,'test');assert not train_groups.intersection(d['groups'])
        scores={}
        for c,x in d['features'].items():
            scores[c]=head.decision_function(x.reshape(-1,4096)).reshape(len(d['y']),15,4)
            assert np.isfinite(scores[c]).all()
        save(base/'scores.pckl',dict(keys=d['keys'],y=d['y'],groups=d['groups'],scores=scores))
        a.summarize_scores(dataset,d,scores,'t9')
        correct={c:(v.argmax(-1)==d['y'][:,None]).astype(float) for c,v in scores.items()}
        for endpoint,ix in [('t9',[9]),('full',list(range(15)))]:
            for name,ca,cb in PAIRS:
                delta=100*(correct[ca]-correct[cb])[:,ix].mean(1)
                alltests.append(dict(dataset=dataset,endpoint=endpoint,contrast=name,**a.cluster_stats(delta,d),status='post-hoc exploratory'))
            for strength,prefix in [('strong',''),('weak','weak_')]:
                delta=100*(correct[f'joint_{prefix}exact']-correct['pcoder_exact']-correct[f'hebb_{prefix}exact']+correct['none'])[:,ix].mean(1)
                interactions.append(dict(dataset=dataset,endpoint=endpoint,strength=strength,**a.cluster_stats(delta,d),status='post-hoc exploratory'))
        old=load(a.OUT/dataset/'common_scores.pckl');assert old['keys']==d['keys'] and np.array_equal(old['y'],d['y'])
        for c in correct:
            oc=(old['scores'][c].argmax(-1)==d['y'][:,None])
            for endpoint,ix in [('t9',[9]),('full',list(range(15)))]:
                comparison.append(dict(dataset=dataset,condition=c,endpoint=endpoint,t9_readout_accuracy=100*correct[c][:,ix].mean(),common_readout_accuracy=100*oc[:,ix].mean(),delta_pp=100*(correct[c]-oc)[:,ix].mean(),status='descriptive'))
        assert before=={str(p):sha(p) for p in originals}
        audits[dataset]=dict(originals_unchanged=True,original_hashes=before,concept_split_disjoint=True,training_only_t9=True,
            all_conditions=28,test_rows=192,test_concepts=32,finite_scores=True,converged=attempts[-1]['converged'],
            training_loss_mass_matches_common=True,head_sha=sha(headpath))
        del d,scores,correct,old;gc.collect();print(dataset,'DONE',flush=True)
    a.write(DEST/'exploratory_tests_holm.csv',holm(alltests));a.write(DEST/'interaction_holm.csv',holm(interactions));a.write(DEST/'readout_comparison.csv',comparison)
    immutable_json(DEST/'audit.json',audits)
    print('COMPLETE',json.dumps(alltests),flush=True)

if __name__=='__main__':main()
