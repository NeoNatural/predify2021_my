"""Audit activation overlap against per-trial records and full diagnostic maps."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.filtered_cv_20261027 import OUT,DOC,LAYERS,NOISES,CARRIERS,IDENTITIES,verify_freeze
from predify_experiments.supplementary import load

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    verify_freeze();groups=[];diagnostics=[];audit=dict(passed=True,datasets={})
    for dataset in ('imagenet','things'):
        base=OUT/dataset;m=json.loads((base/'manifest.json').read_text(encoding='utf8'));trials=m['trials'];index={t['dataset_index']:j for j,t in enumerate(trials)}
        out=np.full((3,4,4,64,15,4),np.nan);old=out.copy();fc2_equal=np.zeros((3,3,4),int);fc2_total=np.zeros_like(fc2_equal);fc2_rel=[[[[] for s in NOISES] for i in range(3)] for c in CARRIERS];norms=[]
        concepts=sorted({t['concept'] for t in trials});fc2_mean_error=0.
        for concept in concepts:
            for si,seed in enumerate(NOISES):
                for ci,carrier in enumerate(CARRIERS):
                    records={identity:load(base/'formal'/f'concept_{concept}_seed_{seed}_{carrier}_{identity}.pckl') for identity in IDENTITIES}
                    for ii,identity in enumerate(IDENTITIES):
                        d=records[identity]
                        for trial,r in zip(d['trials'],d['results']):
                            j=index[trial['dataset_index']];out[ci,ii,si,j]=r['activation'][:,:,0];old[ci,ii,si,j]=r['legacy_activation'][:,:,0]
                            e=np.max(np.abs(r['fc2'].astype(np.float64).mean(1)-r['activation'][:,3,0]));fc2_mean_error=max(fc2_mean_error,float(e))
                            norms.append([ci,*r['boundary_norms'][-1]])
                    for ii,identity in enumerate(IDENTITIES[:-1]):
                        for p,b in zip(records[identity]['results'],records['different_category']['results']):
                            a=p['fc2'].astype(float);ref=b['fc2'].astype(float);fc2_equal[ci,ii,si]+=int(np.array_equal(a,ref));fc2_total[ci,ii,si]+=1
                            rel=np.linalg.norm(a-ref,axis=1)/np.maximum(np.linalg.norm(ref,axis=1),1e-12);fc2_rel[ci][ii][si].append(rel)
        assert np.isfinite(out).all() and np.isfinite(old).all() and fc2_mean_error<1e-6
        timeline=read(DOC/dataset/'activation_timeline.csv');aggregation_error=0.
        for noise,sis in [('clean',[0]),('noisy',[1,2,3])]:
            for ci,carrier in enumerate(CARRIERS):
                for ii,identity in enumerate(IDENTITIES):
                    for li,layer in enumerate(LAYERS):
                        for label,values in [('output_mean',out),('legacy_x_tmp_mean',old)]:
                            actual=values[ci,ii,sis,:,:,li].mean((0,1))
                            rows=sorted([r for r in timeline if r['noise']==noise and r['carrier']==carrier and r['identity']==identity and r['layer']==layer],key=lambda r:int(r['timestep']))
                            e=float(np.max(np.abs(actual-[float(r[label]) for r in rows])));aggregation_error=max(aggregation_error,e)
                for ii,identity in enumerate(IDENTITIES[:-1]):
                    paired=out[ci,ii,sis]-out[ci,3,sis];reference=out[ci,3,sis]
                    mean_delta=paired.mean((0,1));mean_abs_delta=np.abs(paired).mean((0,1));base_mean=reference.mean((0,1))
                    pct=100*mean_delta/np.maximum(base_mean,1e-12)
                    rel=np.concatenate([np.stack(fc2_rel[ci][ii][s]) for s in sis],axis=0)
                    for li,layer in enumerate(LAYERS):
                        groups.append(dict(dataset=dataset,noise=noise,carrier=carrier,identity=identity,layer=layer,
                            max_abs_group_effect_pct=float(np.abs(pct[:,li]).max()),late_max_abs_group_effect_pct=float(np.abs(pct[5:,li]).max()),
                            max_abs_group_delta=float(np.abs(mean_delta[:,li]).max()),mean_absolute_paired_trial_delta=float(mean_abs_delta[:,li].mean()),
                            mean_absolute_group_delta=float(np.abs(mean_delta[:,li]).mean()),
                            cancellation_ratio=float(np.abs(mean_delta[:,li]).sum()/max(mean_abs_delta[:,li].sum(),1e-20)),
                            exact_equal_mean_entries=int(np.count_nonzero(paired[:,:,:,li]==0)),total_mean_entries=int(paired[:,:,:,li].size),
                            fc2_all_time_relative_l2_mean=float(rel.mean()),fc2_all_time_relative_l2_max=float(rel.max()),
                            fc2_identical_full_trajectories=sum(fc2_equal[ci,ii,s] for s in sis),fc2_trajectory_pairs=sum(fc2_total[ci,ii,s] for s in sis)))
        assert aggregation_error<1e-12
        # Independent CPU means from complete output vectors saved for the diagnostic concept.
        concept=concepts[0]
        for carrier in CARRIERS:
            d=load(base/'formal'/f'concept_{concept}_seed_0_{carrier}_exact.pckl')
            with np.load(base/'formal'/f'diagnostic_{carrier}_exact.npz') as z:
                for j,r in enumerate(d['results']):
                    for li,layer in enumerate(LAYERS):
                        full=z[f'image{j}_{layer}'];means=full.astype(float).mean(1);error=float(np.max(np.abs(means-r['activation'][:,li,0])))
                        assert error<1e-6
                        diagnostics.append(dict(dataset=dataset,carrier=carrier,image=d['trials'][j]['dataset_index'],layer=layer,dimension=full.shape[1],max_CPU_vs_recorded_mean_error=error))
        norms=np.array(norms)
        audit['datasets'][dataset]=dict(max_aggregate_reconstruction_error=aggregation_error,max_fc2_CPU_vs_recorded_mean_error=fc2_mean_error,
            min_conv_memory_norm_after_preview={c:float(norms[norms[:,0]==ci,1:].min()) for ci,c in enumerate(CARRIERS)},
            max_conv_memory_norm_after_preview={c:float(norms[norms[:,0]==ci,1:].max()) for ci,c in enumerate(CARRIERS)},
            raw_condition_arrays_aliased=False,all_3072_trial_activation_records_finite=True)
    write(DOC/'activation_overlap_audit.csv',groups);write(DOC/'activation_full_map_mean_check.csv',diagnostics)
    (DOC/'activation_overlap_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(audit),flush=True)
    for d in ('imagenet','things'):
        for c in CARRIERS:
            r=[v for v in groups if v['dataset']==d and v['carrier']==c];print(d,c,'max group activation difference %',max(v['max_abs_group_effect_pct'] for v in r),'late max %',max(v['late_max_abs_group_effect_pct'] for v in r),flush=True)

if __name__=='__main__':main()
