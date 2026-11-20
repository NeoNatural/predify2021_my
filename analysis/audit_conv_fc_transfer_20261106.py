"""Post-run integrity checks and local-input/closed-loop decomposition; no new model runs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from analysis.conv_fc_transfer_20261106 import OUT,DOC,CASES,read,sha,writejson


def main():
    protocol=read(OUT/'protocol.json')
    for name,expected in protocol['source_sha256'].items():assert sha(ROOT/name)==expected,name
    rows=[];state=[];maximum_balance=0.;maximum_local_increase=-np.inf;maximum_scale_error=0.
    for ds in ('imagenet','things'):
        spec=read(OUT/ds/'manifest.json')
        for case in CASES:
            for hist in (('none','exact','different_category') if case=='clip' else ('exact','different_category')):
                file=OUT/ds/'development'/case/f'{hist}.npz';note=read(file.with_suffix('.json'))
                assert sha(file)==note['sha256']
                with np.load(file) as z:
                    incoming=z['conv4_input_mean'];outgoing=z['conv4_mean']
                    if case.endswith(('_full','_pool','_shuffle')):
                        increase=float((outgoing-incoming).max());maximum_local_increase=max(maximum_local_increase,increase)
                        assert increase<1e-6
                    if case.endswith('_full'):
                        for win,sl in [('early',slice(0,5)),('middle',slice(5,10)),('late',slice(10,15))]:
                            rows.append(dict(dataset=ds,case=case,history=hist,window=win,
                                local_suppression_pct=float(np.mean(100*(1-outgoing[:,sl].mean(1)/incoming[:,sl].mean(1)))),
                                input_mean=float(incoming[:,sl].mean()),output_mean=float(outgoing[:,sl].mean())))
                paired=None
                if case.endswith(('_shuffle','_scale')):
                    source=case.rsplit('_',1)[0]+'_full'
                    paired=read(OUT/ds/'development'/source/f'{hist}.json')['trials']
                for j,tr in enumerate(note['trials']):
                    assert tr['reset_zero'] and tr['before_target']==tr['after_target']
                    assert tr['graph']['inter_degree_max']<=32
                    for key in ('intra_max_rowsum','inter_max_rowsum'):assert tr['graph'][key]<.75001
                    if case.endswith('_shuffle'):assert tr['written_weight']==paired[j]['written_weight']
                    if case.endswith('_scale'):
                        error=max(abs(a-b) for a,b in zip(tr['means'],paired[j]['means']))
                        maximum_scale_error=max(maximum_scale_error,error);assert error<2e-6
                    for steps in tr['trace']:
                        for step in steps:maximum_balance=max(maximum_balance,abs(step['preclip_balance']))
                    state.append(dict(dataset=ds,case=case,history=hist,target=tr['target']))
    assert len(state)==1260
    local=pd.DataFrame(rows);local.to_csv(DOC/'local_input_output.csv',index=False,encoding='utf-8-sig')
    c=pd.read_csv(DOC/'vs_same_history_clip.csv')
    f=c[(c.window=='early')&(c.history=='exact')&(c.layer=='fc2')&c.case.str.endswith('_full')].copy()
    comparisons=[]
    for row in f.itertuples(index=False):
        base=row.case.rsplit('_',1)[0]
        p=c[(c.dataset==row.dataset)&(c.window=='early')&(c.history=='exact')&(c.layer=='fc2')&(c.case==base+'_pool')].iloc[0]
        s=c[(c.dataset==row.dataset)&(c.window=='early')&(c.history=='exact')&(c.layer=='fc2')&(c.case==base+'_scale')].iloc[0]
        comparisons.append(dict(dataset=row.dataset,case=row.case,
            fisher_extra_vs_pool_pp=row.fisher_change_pct-p.fisher_change_pct,
            fisher_extra_vs_scale_pp=row.fisher_change_pct-s.fisher_change_pct,
            accuracy_extra_vs_pool_pp=row.accuracy_change_pp-p.accuracy_change_pp,
            margin_extra_vs_pool=row.margin_change-p.margin_change))
    pd.DataFrame(comparisons).to_csv(DOC/'full_vs_controls.csv',index=False,encoding='utf-8-sig')
    audit=dict(passed=True,sequences=1260,source_hashes_verified=True,
        local_output_check_methods=['full','pool','shuffle'],
        local_output_never_exceeds_input=True,max_local_mean_increase=maximum_local_increase,
        max_relative_preclip_balance_error=maximum_balance,max_scale_profile_error=maximum_scale_error,
        shuffle_preview_equal_to_full=True,all_target_weights_frozen=True,
        all_graph_degrees_at_most32=True,audit_source_sha256=sha(__file__))
    writejson(DOC/'final_audit.json',audit)
    print(audit)
    print(local[(local.window=='early')&(local.history=='exact')].round(7).to_string(index=False))
    print(pd.DataFrame(comparisons).round(7).to_string(index=False))


if __name__=='__main__':main()
