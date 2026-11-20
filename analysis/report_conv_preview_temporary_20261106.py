"""Readout fidelity, full-map geometry, paired priming and immutable-memory audit."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import numpy as np
import pandas as pd
from analysis.conv_preview_temporary_20261106 import OUT,DOC,CASES,LAYERS,sha,writejson
from predify_experiments.supplementary import load

WINDOWS={'early':slice(0,5),'middle':slice(5,10),'late':slice(10,15)}


def geometry(g,labels):
    n=len(labels);classsum=sum(g[np.ix_(labels==c,labels==c)].sum()/sum(labels==c) for c in np.unique(labels))
    w=max(float(np.trace(g)-classsum),0.)/n
    b=max(float(classsum-g.sum()/n),0.)/n
    return dict(within=w,between=b,fisher=b/max(w,1e-20))


def bootstrap(value,concepts,draws=4000):
    # Trial pairs stay together; noise seeds and timesteps are never independent replicates.
    units=np.array([value[concepts==c].mean() for c in np.unique(concepts)])
    rng=np.random.default_rng(20261106)
    sampled=units[rng.integers(len(units),size=(draws,len(units)))].mean(1)
    return [float(v) for v in np.quantile(sampled,[.025,.975])]


def table(frame,cols,digits=5):
    frame=frame[cols].copy()
    for c in frame.select_dtypes(include='number'):frame[c]=frame[c].round(digits)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join(
        '| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    metrics=[];contrasts=[];comparisons=[];infer=[];fidelity=[];auditrows=[];all_frozen={};weightstats=[]
    datasets=('imagenet','things')
    for ds in datasets:
        spec=json.loads((OUT/ds/'manifest.json').read_text(encoding='utf8'))
        dev=spec['development'];labels=np.array([t['category_index'] for t in dev]);concepts=np.array([t['concept'] for t in dev])
        head=load(OUT/ds/'reference_head.pckl')['head'];savedhead=json.loads((OUT/ds/'reference_head.json').read_text())
        assert sha(OUT/ds/'reference_head.pckl')==savedhead['sha256']
        assert not {t['concept'] for t in spec['training']}&set(concepts)
        base={}
        for seed in (0,2921):
            for ident in ('exact','different_category'):
                with np.load(OUT/ds/'clip'/f'{seed}_{ident}.npz') as z:base[seed,ident]=z['fc2'].copy()
        casegroups={}
        for case in CASES:
            groups={}
            for seed in (0,2921):
                for ident in ('exact','different_category'):
                    file=OUT/ds/case/f'{seed}_{ident}.npz'
                    info=json.loads(file.with_suffix('.json').read_text(encoding='utf8'))
                    assert sha(file)==info['file_sha256']
                    with np.load(file) as z:data={k:z[k] for k in z.files}
                    assert np.array_equal(data['indices'],[t['dataset_index'] for t in dev])
                    assert np.array_equal(data['labels'],labels)
                    fc=data['fc2'];n,t,d=fc.shape
                    score=head.decision_function(fc.reshape(n*t,d)).reshape(n,t,-1)
                    prediction=head.classes_[score.argmax(2)];accuracy=(prediction==labels[:,None]).astype(float)
                    correct=score[np.arange(n)[:,None],np.arange(t)[None,:],labels[:,None]]
                    other=score.copy();other[np.arange(n)[:,None],np.arange(t)[None,:],labels[:,None]]=-np.inf
                    margin=correct-other.max(2)
                    ref=base[seed,ident];refpred=head.predict(ref.reshape(n*t,d)).reshape(n,t)
                    cosine=(fc.astype(np.float64)*ref).sum(2)/np.maximum(np.linalg.norm(fc,axis=2)*np.linalg.norm(ref,axis=2),1e-12)
                    data['accuracy']=accuracy;data['margin']=margin
                    groups[seed,ident]=data
                    for tr in info['trials']:
                        assert tr['before_target']==tr['after_target']
                        assert tr['reset']['writes']==0 and tr['reset']['intra_edges']==0 and tr['reset']['inter_edges']==0
                        expected=0 if CASES[case][1]=='none' else 2
                        assert tr['writes']==expected
                        assert [p['writes'] for p in tr['preview_boundaries']]==([0,0] if expected==0 else [1,2])
                        for p in tr['preview_boundaries']:
                            assert p['inter_degree_max']<=32
                            weightstats.append(dict(dataset=ds,case=case,seed=seed,identity=ident,
                                target=tr['dataset_index'],stage=p['stage'],intra_edges=p['intra_edges'],
                                inter_edges=p['inter_edges'],intra_max_rowsum=p['intra_max_rowsum'],
                                inter_max_rowsum=p['inter_max_rowsum']))
                        key=(ds,case,ident,tr['dataset_index'])
                        if seed==0:all_frozen[key]=tr['before_target']
                        else:assert all_frozen[key]==tr['before_target'],'Target noise leaked into preview memory'
                        auditrows.append(dict(dataset=ds,case=case,seed=seed,identity=ident,target=tr['dataset_index'],
                            writes=tr['writes'],target_frozen=True,reset_zero=True,weight_sha256=tr['before_target']))
                    for win,sl in WINDOWS.items():
                        fidelity.append(dict(dataset=ds,case=case,seed=seed,identity=ident,window=win,
                            accuracy=float(accuracy[:,sl].mean()),margin=float(margin[:,sl].mean()),
                            agreement=float((prediction[:,sl]==refpred[:,sl]).mean()),cosine=float(cosine[:,sl].mean())))
                    for layer in LAYERS:
                        for ti in range(15):
                            row=dict(dataset=ds,case=case,seed=seed,identity=ident,layer=layer,t=ti,
                                mean=float(data[layer+'_mean'][:,ti].mean()),
                                mean_abs=float(data[layer+'_mean_abs'][:,ti].mean()),
                                density=float(data[layer+'_density'][:,ti].mean()),
                                negative=float(data[layer+'_negative'][:,ti].mean()),
                                zero_vectors=int((data[layer+'_norm'][:,ti]==0).sum()),
                                **geometry(data[layer+'_gram'][ti],labels))
                            row.update(accuracy=float(accuracy[:,ti].mean()) if layer=='fc2' else np.nan,
                                       margin=float(margin[:,ti].mean()) if layer=='fc2' else np.nan)
                            metrics.append(row)
            casegroups[case]=groups
        # All target-level paired effects, with concept-block intervals.
        for case,groups in casegroups.items():
            family=CASES[case][0];control=family+'_none' if family!='clip' else 'clip'
            for seed in (0,2921):
                for win,sl in WINDOWS.items():
                    for layer in LAYERS:
                        for metric,suffix in [('mean','_mean'),('density','_density')]+(
                            [('accuracy',None),('margin',None)] if layer=='fc2' else []):
                            key=layer+suffix if suffix else metric
                            value=(groups[seed,'exact'][key][:,sl]-groups[seed,'different_category'][key][:,sl]).mean(1)
                            ctrl=casegroups[control]
                            ctrlvalue=(ctrl[seed,'exact'][key][:,sl]-ctrl[seed,'different_category'][key][:,sl]).mean(1)
                            lo,hi=bootstrap(value,concepts);dlo,dhi=bootstrap(value-ctrlvalue,concepts)
                            infer.append(dict(dataset=ds,case=case,control=control,seed=seed,window=win,layer=layer,
                                metric=metric,exact_minus_different=float(value.mean()),ci_low=lo,ci_high=hi,
                                additional_priming=float((value-ctrlvalue).mean()),did_ci_low=dlo,did_ci_high=dhi,
                                concepts=len(np.unique(concepts))))
    frame=pd.DataFrame(metrics);fid=pd.DataFrame(fidelity)
    window=frame.assign(window=frame.t.map(lambda t:'early' if t<5 else 'middle' if t<10 else 'late')).groupby(
        ['dataset','case','seed','identity','layer','window'],as_index=False).mean(numeric_only=True)
    keys=['dataset','case','seed','layer','window']
    e=window[window.identity=='exact'].set_index(keys);d=window[window.identity=='different_category'].set_index(keys)
    con=e[['mean','density','within','between','fisher','accuracy','margin']]-d[['mean','density','within','between','fisher','accuracy','margin']]
    con.columns=[c+'_exact_minus_different' for c in con.columns]
    con['mean_priming_pct']=100*(e['mean']/d['mean']-1)
    con=con.reset_index()
    for row in window.itertuples(index=False):
        q=window[(window.dataset==row.dataset)&(window.case=='clip')&(window.seed==row.seed)&
                 (window.identity==row.identity)&(window.layer==row.layer)&(window.window==row.window)].iloc[0]
        comparisons.append(dict(dataset=row.dataset,case=row.case,seed=row.seed,identity=row.identity,
            layer=row.layer,window=row.window,mean_change_pct=100*(row.mean/q['mean']-1),
            density_change_pct=100*(row.density/q['density']-1),
            fisher_change_pct=100*(row.fisher/q['fisher']-1),accuracy_delta_pp=100*(row.accuracy-q['accuracy'])))
    comp=pd.DataFrame(comparisons)
    for name,value in [('step_metrics',frame),('window_metrics',window),('priming_contrasts',con),
                       ('vs_clip',comp),('paired_concept_bootstrap',pd.DataFrame(infer)),
                       ('readout_fidelity',fid),('weight_audit',pd.DataFrame(auditrows)),
                       ('weight_statistics',pd.DataFrame(weightstats))]:
        value.to_csv(DOC/f'{name}.csv',index=False,encoding='utf-8-sig')
    # Four equally weighted dataset x noise cells, both identities for fidelity/activity.
    s=comp[(comp.window=='early')&(comp.layer=='conv4')].groupby('case')[['mean_change_pct','density_change_pct']].mean()
    af=comp[(comp.window=='early')&(comp.layer=='fc2')].groupby('case').accuracy_delta_pp.mean()
    ff=fid[fid.window=='early'].groupby('case')[['agreement','cosine']].mean()
    summary=s.join(af).join(ff).reset_index();summary['agreement_pct']=100*summary.pop('agreement')
    ce=con[(con.window=='early')&(con.layer=='conv4')].rename(columns={'mean_priming_pct':'conv4_priming_pct'})
    fe=con[(con.window=='early')&(con.layer=='fc2')].rename(columns={'mean_priming_pct':'fc2_priming_pct'})
    priming=ce[['dataset','case','seed','conv4_priming_pct']].merge(fe[['dataset','case','seed',
        'fc2_priming_pct','within_exact_minus_different','fisher_exact_minus_different','accuracy_exact_minus_different']],
        on=['dataset','case','seed'])
    priming['accuracy_priming_pp']=100*priming.pop('accuracy_exact_minus_different')
    did=[]
    for row in priming.itertuples(index=False):
        if row.case=='clip' or row.case.endswith('_none'):continue
        family=CASES[row.case][0]
        control=priming[(priming.dataset==row.dataset)&(priming.seed==row.seed)&(priming.case==family+'_none')].iloc[0]
        did.append(dict(dataset=row.dataset,case=row.case,seed=row.seed,
            conv4_RS_extra_pct=row.conv4_priming_pct-control.conv4_priming_pct,
            fc2_RS_extra_pct=row.fc2_priming_pct-control.fc2_priming_pct,
            within_extra=row.within_exact_minus_different-control.within_exact_minus_different,
            fisher_extra=row.fisher_exact_minus_different-control.fisher_exact_minus_different,
            accuracy_extra_pp=row.accuracy_priming_pp-control.accuracy_priming_pp))
    did=pd.DataFrame(did);did.to_csv(DOC/'additional_priming.csv',index=False,encoding='utf-8-sig')
    checks=dict(complete=True,sequences=len(auditrows),expected=36*4*7,
        hebbian_sequences=sum(r['writes']==2 for r in auditrows),writes=sum(r['writes'] for r in auditrows),
        all_target_weights_frozen=True,all_trials_reset_zero=True,noise_does_not_change_preview_weights=True,
        no_training_graphs_loaded=True,fc_hebb=False,fold0_unused=True)
    assert checks['sequences']==checks['expected']
    writejson(DOC/'complete.json',checks)
    text=['# Preview 临时卷积 Hebbian 修正实验',
          '已完成两数据集 16/20 个开发目标，7 组（六组机制＋clip 校准基线），clean/noisy × exact/different-category，共 '
          f"{len(auditrows)} 条真实递归序列。每条为 2+2 preview 和 15 步 target。旧固定图结果没有复用。",
          '每 trial 权重清零；两段 endpoint 各写一次，A₂=0.25H₁+0.5H₂；第二段 preview 使用第一段权重。'
          'target 既不学习也不衰减。所有序列的重置、写入数、冻结哈希、clean/noisy 的 preview 权重一致检查均通过。'
          '旧卷积与 FC Hebbian 均关闭，PCoder 状态在 trial 内延续。',
          '## 早期 t0–4 的保真与活动',
          '相对相同输入、身份和噪声下的 clip。两数据集×两噪声条件等权；准确率为百分点，密度为相对百分比。',
          table(summary,['case','accuracy_delta_pp','agreement_pct','cosine','mean_change_pct','density_change_pct'],4),
          '## 早期 Exact 相对 Different',
          '响应百分比小于零才是 suppression。within 为单位范数类内散度，负差表示更内聚；Fisher 正差表示比值增加。'
          '两者及固定读出应结合判断，不把整体稀疏化等同于 sharpening。',
          table(priming,list(priming.columns),6),
          '## 临时 Hebbian 相对同版本无 Hebbian 的额外 priming',
          'RS_extra 是两组各自 Exact/Different 响应百分比之差；负值表示重复抑制方向更强。'
          'within_extra 小于零、fisher_extra 大于零、accuracy_extra 大于零分别是对应的改善方向。',
          table(did,list(did.columns),6),
          '## 范围与核查',
          '本轮是相同开发集上的机制修正小实验，没有独立测试结论。每数据集只有 8/10 个开发概念。'
          'paired_concept_bootstrap.csv 提供按概念成块的配对 95% bootstrap 区间；时间步和噪声不作为独立重复，'
          '未进行多重比较校正，区间用于描述不确定性。',
          '完整 early/middle/late、逐步几何、读出一致率与余弦、权重审计分别见同目录 CSV。'
          '卷积几何由完整特征图的双精度单位范数 Gram 计算，未用空间池化替代；不保存原始卷积图，'
          '因此不能事后进行逐单元调谐分析。FC1/FC2 完整特征仍在 E 盘。',
          f'原始紧凑结果：`{OUT}`。参数与权重生命周期详见实验方案.md 和 protocol.json。']
    (DOC/'实验结果.md').write_text('\n\n'.join(text),encoding='utf8')
    plot(frame,con)
    print(json.dumps(checks),flush=True)
    print(summary.round(5).to_string(index=False),flush=True)


def plot(frame,con):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors={'clip':'#666666','gate_none':'#a3b3bc','gate_rep4':'#187b9d','gate_prd5':'#da7b16',
            'multiply_none':'#a3b3bc','multiply_rep4':'#187b9d','multiply_prd5':'#da7b16'}
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for di,ds in enumerate(('imagenet','things')):
        for fi,family in enumerate(('gate','multiply')):
            for li,layer in enumerate(('conv4','fc2')):
                ax=axes[di,fi*2+li]
                for case in ('clip',family+'_none',family+'_rep4',family+'_prd5'):
                    e=frame[(frame.dataset==ds)&(frame.case==case)&(frame.layer==layer)&(frame.identity=='exact')].groupby('t')['mean'].mean()
                    d=frame[(frame.dataset==ds)&(frame.case==case)&(frame.layer==layer)&(frame.identity=='different_category')].groupby('t')['mean'].mean()
                    ax.plot(e.index,100*(e/d-1),label=case,color=colors[case],ls='--' if case=='clip' else '-')
                ax.axhline(0,color='black',lw=.7);ax.set_title(f'{ds}: {family}, {layer}')
                ax.set_xlabel('Target step');ax.set_ylabel('Exact / Different - 1 (%)')
                if di==0 and li==0:ax.legend(fontsize=7)
    fig.savefig(DOC/'priming_timecourses.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
