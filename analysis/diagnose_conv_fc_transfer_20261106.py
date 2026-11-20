"""Offline diagnosis of completed CSVs only. No model, fitting, or parameter search."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/conv_fc_transfer_20261106'
DEST=ROOT/'docs/conv_fc_transfer_diagnosis_20261106'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    traces=pd.read_csv(SOURCE/'update_traces.csv')
    windows=pd.read_csv(SOURCE/'window_metrics.csv')
    effects=pd.read_csv(SOURCE/'vs_same_history_clip.csv')
    graphs=pd.read_csv(SOURCE/'weight_statistics.csv')
    early=traces[traces.t<5]
    fields=['kappa','threshold','modulation_rms','gain_min','gain_max','density','active_code']
    stage=early[early.step.isin([1,5])].groupby(['dataset','case','history','step'])[fields].mean().reset_index()
    stage['modulation_rms_pct']=100*stage.modulation_rms
    stage.to_csv(DEST/'early_control_strength.csv',index=False,encoding='utf-8-sig')
    index=['dataset','case','target','t','step']
    a=traces[traces.history=='exact'].set_index(index)
    b=traces[traces.history=='different_category'].set_index(index)
    paired=100*(a[['kappa','threshold']]/b[['kappa','threshold']]-1)
    paired=paired.rename(columns={'kappa':'kappa_ED_pct','threshold':'threshold_ED_pct'}).reset_index()
    paired.to_csv(DEST/'paired_matching_threshold.csv',index=False,encoding='utf-8-sig')
    rows=[];looprows=[];fullpool=[]
    for ds in ('imagenet','things'):
        cal=json.loads((SOURCE/f'{ds}_calibration.json').read_text())
        for stem in ('direct_rep4','direct_prd5','sae_rep4','sae_prd5'):
            case=stem+'_full'
            e=stage[(stage.dataset==ds)&(stage.case==case)&(stage.history=='exact')].set_index('step')
            q=paired[(paired.dataset==ds)&(paired.case==case)&(paired.t<5)&(paired.step==5)]
            rows.append(dict(dataset=ds,branch_source=stem,gain_RMS_pct=e.loc[5,'modulation_rms_pct'],
                gain_min=e.loc[5,'gain_min'],gain_max=e.loc[5,'gain_max'],
                kappa=e.loc[5,'kappa'],kappa_over_training_k0=e.loc[5,'kappa']/cal[stem]['kappa0'],
                threshold=e.loc[5,'threshold'],kappa_ED_pct=q.kappa_ED_pct.mean(),
                threshold_ED_pct=q.threshold_ED_pct.mean(),
                first_active_code=e.loc[1,'active_code'],last_active_code=e.loc[5,'active_code']))
            for method in ('full','pool'):
                v=stage[(stage.dataset==ds)&(stage.case==stem+'_'+method)&(stage.history=='exact')].set_index('step')
                looprows.append(dict(dataset=ds,branch_source=stem,method=method,
                    kappa_step1_to5_pct=100*(v.loc[5,'kappa']/v.loc[1,'kappa']-1),
                    threshold_step1_to5_pct=100*(v.loc[5,'threshold']/v.loc[1,'threshold']-1),
                    density_step1_to5_pp=100*(v.loc[5,'density']-v.loc[1,'density'])))
            for side_step in (1,5):
                f=early[(early.dataset==ds)&(early.case==case)&(early.history=='exact')&(early.step==side_step)].set_index(['target','t'])
                p=early[(early.dataset==ds)&(early.case==stem+'_pool')&(early.history=='exact')&(early.step==side_step)].set_index(['target','t'])
                fullpool.append(dict(dataset=ds,branch_source=stem,step=side_step,
                    threshold_full_vs_pool_pct=float((100*(f.threshold/p.threshold-1)).mean()),
                    kappa_full_vs_pool_pct=float((100*(f.kappa/p.kappa-1)).mean())))
    summary=pd.DataFrame(rows)
    summary.to_csv(DEST/'mechanism_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(looprows).to_csv(DEST/'within_step_dynamics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(fullpool).to_csv(DEST/'full_pool_thresholds.csv',index=False,encoding='utf-8-sig')
    # Geometry difference within the same history/window, with both numerator and denominator.
    geo=[]
    for ds in ('imagenet','things'):
        for stem in ('direct_rep4','direct_prd5','sae_rep4','sae_prd5'):
            q=windows[(windows.dataset==ds)&(windows.history=='exact')&(windows.window=='early')&(windows.layer=='fc2')].set_index('case')
            f=q.loc[stem+'_full']
            for control in ('pool','scale'):
                p=q.loc[stem+'_'+control]
                geo.append(dict(dataset=ds,branch_source=stem,control=control,
                    within_full_vs_control_pct=100*(f.within/p.within-1),
                    between_full_vs_control_pct=100*(f['between']/p['between']-1),
                    fisher_full_vs_control_pct=100*(f.fisher/p.fisher-1)))
    pd.DataFrame(geo).to_csv(DEST/'geometry_full_vs_controls.csv',index=False,encoding='utf-8-sig')
    e=effects[(effects.window=='early')&(effects.history=='exact')&effects.case.str.endswith('_full')]
    propagation=e.pivot(index=['dataset','case'],columns='layer',values='mean_change_pct').reset_index()
    propagation.to_csv(DEST/'propagation_mean_changes.csv',index=False,encoding='utf-8-sig')
    graph=graphs[graphs.case.str.endswith('_full')&(graphs.history=='exact')].groupby(['dataset','case'])[
        ['intra_edges','inter_edges','intra_max_rowsum','inter_max_rowsum']].mean().reset_index()
    graph['channels']=np.where(graph.case.str.startswith('sae'),64,512)
    graph['inter_candidate_density_pct']=100*graph.inter_edges/(8*graph.channels**2)
    graph.to_csv(DEST/'connectivity_comparison.csv',index=False,encoding='utf-8-sig')
    plot(traces,paired)
    names=['update_traces.csv','window_metrics.csv','vs_same_history_clip.csv','weight_statistics.csv',
           'imagenet_calibration.json','things_calibration.json']
    audit=dict(analysis_only=True,new_conditions_run=0,models_fitted=0,parameter_search_run=False,
        source_sha256={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in names},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Trace RMS includes all original units, including zeros; not activity-weighted RMS.',
                'Kappa and threshold traces are spatial means; no exact per-position saturation or covariance decomposition.',
                'Full and pool used different closed-loop preview2/target paths; associations are not same-W/x/tau causal effects.'])
    (DEST/'analysis_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(summary.round(6).to_string(index=False))
    print(pd.DataFrame(looprows).round(6).to_string(index=False))
    print(pd.DataFrame(fullpool).round(6).to_string(index=False))
    print(pd.DataFrame(geo).round(6).to_string(index=False))
    print(propagation.round(6).to_string(index=False))


def plot(traces,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    colors={'direct_rep4_full':'#187b9d','direct_prd5_full':'#61a8bc','sae_rep4_full':'#d27a16','sae_prd5_full':'#ac4b2e'}
    for row,ds in enumerate(('imagenet','things')):
        for case,color in colors.items():
            q=traces[(traces.dataset==ds)&(traces.case==case)&(traces.history=='exact')&(traces.step==5)].groupby('t').modulation_rms.mean()*100
            gap=paired[(paired.dataset==ds)&(paired.case==case)&(paired.step==5)].groupby('t')[['kappa_ED_pct','threshold_ED_pct']].mean()
            style='--' if '_prd5_' in case else '-'
            axes[row,0].plot(q.index,q,color=color,ls=style,label=case.replace('_full',''))
            axes[row,1].plot(gap.index,gap.kappa_ED_pct,color=color,ls=style)
            axes[row,2].plot(gap.index,gap.threshold_ED_pct,color=color,ls=style)
        for col,title in enumerate(('Exact modulation RMS, all units (%)','Exact / Different kappa - 1 (%)','Exact / Different threshold - 1 (%)')):
            ax=axes[row,col];ax.set_title(f'{ds}: {title}');ax.set_xlabel('Target network step')
            ax.spines[['top','right']].set_visible(False)
            if col:ax.axhline(0,color='black',lw=.6)
        axes[row,0].legend(fontsize=8)
    fig.savefig(DEST/'gain_matching_threshold.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
