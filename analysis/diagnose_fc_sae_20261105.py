"""Read-only SAE error decomposition and existing-SVM counterfactual diagnostics; no fitting."""
import csv
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis import fc_sae_acceptance_20261105 as exp
from analysis import fc_preview_pilot_20261105 as pilot

OUT=ROOT/'docs/fc_sae_diagnosis_20261105'

def csvsave(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run():
    old,protocol=exp.protocol();OUT.mkdir(parents=True,exist_ok=True)
    projections=[];classifiers=[];coverage=[];audits=[]
    for d in ('imagenet','things'):
        exp.verify(exp.OUT/d);manifest=pilot.read(pilot.native.OUT/d/'manifest.json')
        roles=pilot.read(exp.previous.OUT/d/'inputs/audit.json')['roles']
        with np.load(exp.OUT/d/'target_metadata.npz') as z:meta={s:{k:z[f'{s}__{k}'] for k in ('y','concept','row')} for s in ('train','formal0','development')}
        fit_concepts={r['concept'] for r in roles};assert fit_concepts.isdisjoint(set(meta['formal0']['concept'])|set(meta['development']['concept']))
        provenance=pilot.read(exp.OUT/d/'data_provenance.json')
        coverage.append(dict(dataset=d,categories=manifest['categories'],train_target_count=len(meta['train']['row']),
            train_target_concepts=len(set(meta['train']['concept'])),all_fit_role_concepts=len(fit_concepts),
            formal0_targets=len(meta['formal0']['row']),formal0_concepts=len(set(meta['formal0']['concept'])),
            development_targets=len(meta['development']['row']),development_concepts=len(set(meta['development']['concept'])),
            concept_overlap_formal0=len(fit_concepts&set(meta['formal0']['concept'])),
            concept_overlap_development=len(fit_concepts&set(meta['development']['concept'])),
            unique_training_preview_frames=provenance['train_role_counts']['preview']))
        with (pilot.OUT/d/'reference_svm.pckl').open('rb') as f:fixed=pickle.load(f)
        for l in exp.LAYERS:
            dest=exp.OUT/d/l;exp.verify(dest)
            before={name:pilot.sha(dest/name) for name in ('model.pth','readouts.pckl','representations.npz','scores.npz')}
            state=torch.load(dest/'model.pth',map_location='cpu',weights_only=True)
            D=state['decoder'].numpy().astype(np.float64);center=state['center'].numpy().astype(np.float64);scale=float(state['scale'])
            Q,R=np.linalg.qr(D,mode='reduced');singular=np.linalg.svd(R,compute_uv=False)
            assert singular[-1]>1e-8 and np.max(np.abs(Q.T@Q-np.eye(Q.shape[1])))<1e-10
            with np.load(dest/'representations.npz') as z:a={k:z[k] for k in z.files}
            with (dest/'readouts.pckl').open('rb') as f:heads=pickle.load(f)
            original=fixed[l];adapt=heads['rectified'];train=a['train__raw'].astype(np.float64)
            constant=np.ptp(train,axis=0)==0;traincenter=train.mean(0)
            assert np.max(np.abs(original[-1].coef_[:,constant]))<1e-12
            for split in ('train','formal0','development'):
                x=a[split+'__raw'].astype(np.float64);hat=a[split+'__reconstruction'].astype(np.float64)
                v=x/scale-center;projected=(v@Q)@Q.T
                outside=(projected-v)*scale;inside=(hat-x)-outside
                w=pilot.weights(meta[split]['y'],meta[split]['concept']);den=np.average(np.square(x-traincenter).sum(1),weights=w)
                total=np.average(np.square(hat-x).sum(1),weights=w);out=np.average(np.square(outside).sum(1),weights=w)
                inc=np.average(np.square(inside).sum(1),weights=w);cross=np.average(2*(outside*inside).sum(1),weights=w)
                assert abs((out+inc+cross)-total)<1e-8 and abs(cross)/max(total,1e-12)<1e-5
                projections.append(dict(dataset=d,layer=l,split=split,raw_fvu=total/den,decoder_subspace_fvu=out/den,
                    coding_constraint_fvu=inc/den,cross_term_fvu=cross/den,subspace_share_of_error=out/total,
                    full_rank=int(len(singular)),smallest_singular_value=float(singular[-1]),largest_singular_value=float(singular[0])))
                raw=a[split+'__raw'];rect=a[split+'__rectified'];standardized=adapt[0].transform(rect)
                full=adapt[-1].decision_function(standardized)
                contribution=standardized[:,constant]@adapt[-1].coef_[:,constant].T
                counterfactual=full-contribution
                cases=dict(fixed_on_raw=original.decision_function(raw),fixed_on_reconstruction=original.decision_function(rect),
                    adapted_on_raw=adapt.decision_function(raw),adapted_on_reconstruction=full,
                    adapted_remove_raw_training_constant_dimensions=counterfactual)
                for case,scores in cases.items():
                    classifiers.append(dict(dataset=d,layer=l,split=split,case=case,
                        **exp.previous.metrics(scores,meta[split]['y']),
                        disagreements_with_adapted_full=int(np.sum(scores.argmax(1)!=full.argmax(1))),
                        raw_training_constant_dimensions=int(constant.sum()),
                        standardized_rms_on_constant_dimensions=float(np.sqrt(np.mean(np.square(standardized[:,constant]))))))
            after={name:pilot.sha(dest/name) for name in before};assert before==after
            audits.append(dict(dataset=d,layer=l,source_hashes=before,unchanged=True,
                trainable_parameters=2*4096*512+512,QR_orthogonality_error=float(np.max(np.abs(Q.T@Q-np.eye(Q.shape[1]))))))
    for name,rows in (('concept_coverage',coverage),('error_decomposition',projections),('readout_diagnostics',classifiers)):
        pilot.write_json(OUT/(name+'.json'),rows);csvsave(OUT/(name+'.csv'),rows)
    exp.protocol();pilot.write_json(OUT/'audit.json',dict(passed=True,no_fitting=True,no_parameter_search=True,
        no_Hebbian=True,source_sha256=pilot.sha(__file__),sae_protocol_sha256=pilot.sha(exp.OUT/'protocol.json'),models=audits))
    print('COVERAGE',coverage)
    print('PROJECTION',[r for r in projections if r['split']=='formal0'])
    print('READOUT',[r for r in classifiers if r['split']=='formal0'])

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
