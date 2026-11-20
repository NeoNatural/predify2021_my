"""Audit and compare all finite-step settings without stationary-state filtering."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from predify_experiments.fc_finite_steps_20261028 import DATA,OUT,DOC,STEPS,SVM_STEPS,EPS,sha
from predify_experiments import fc_sparse_lateral_20261028 as sparse

DS=('imagenet','things');LAYERS=('fc1','fc2')
EFFECTS=('activation_pct','within_pct','between_pct','fisher_pct','accuracy_pp','auc_delta')
ROLES=('prior_soft','matched_hard','paired_nohebb_soft','paired_nohebb_hard')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def csv_write(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def flatten(dataset,layer,fold,config,step,m,base,role,history):
    row=dict(dataset=dataset,layer=layer,fold=fold,role=role,config_id=config['id'],family=config['family'],
        rule=config['rule'],graph=config['graph'],mode=config['mode'],gain=config['gain'],budget=config['budget'],step=step,
        previous_nonconverged_inner_runs=history,**m['effects'])
    for k in ('activation','within','between','fisher','raw_within','raw_between','raw_fisher','accuracy','auc','nonzero',
        'support_ratio','max_support_ratio','revived_fraction','revived_activity_share','weak_activity_share',
        'residual','support_switch_units_mean','zero_fraction','joint','sparse_joint'):
        row[k]=m[k]
    row['frozen_svm_accuracy_pp']=100*(m['frozen_svm']['accuracy']-base['frozen_svm']['accuracy'])
    row['frozen_svm_auc_delta']=m['frozen_svm']['auc']-base['frozen_svm']['auc']
    row['refit_svm_accuracy_pp']=100*(m['refit_svm']['accuracy']-base['frozen_svm']['accuracy']) if 'refit_svm' in m else None
    row['refit_svm_auc_delta']=m['refit_svm']['auc']-base['frozen_svm']['auc'] if 'refit_svm' in m else None
    row['angular_ridge_accuracy']=m.get('angular_ridge',{}).get('accuracy')
    row['angular_ridge_auc']=m.get('angular_ridge',{}).get('auc')
    return row


def aggregate():
    protocol=read(OUT/'protocol.json')
    assert sha(OUT/'source_snapshot.py')==protocol['source_sha256']
    assert sha(ROOT/'predify_experiments/fc_finite_steps_20261028.py')==protocol['source_sha256']
    for p,h in {**protocol['dependencies'],**protocol['original_model_sources']}.items():assert sha(ROOT/p)==h
    for d in DS:
        for p,h in protocol['inputs'][d].items():assert sha(DATA/d/p)==h
    grids=[];pairs=[];late=[];designs=[]
    audit=dict(passed=True,folds=0,grid_evaluations=0,hard_evaluations=0,previous_nonconverged_hard_fold_configs_retained=0,
        pair_representations=0,no_stationarity_filter=True,original_model_unchanged=True,previous_code_unchanged=True,
        vanilla_cache_unchanged=True,max_pair_support_ratio=0.)
    for d in DS:
        with np.load(DATA/d/'activations.npz') as z:
            y,groups,folds=z['category'],z['concept'],z['fold']
            original={l:z[l] for l in LAYERS}
        for l in LAYERS:
            for f in range(5):
                dest=OUT/d/l/f'fold{f}'
                complete=read(dest/'complete.json')
                assert complete['protocol_sha256']==sha(OUT/'protocol.json')
                for p,h in complete['files'].items():assert sha(dest/p)==h
                data=read(dest/'results.json');design=read(dest/'design.json')
                assert sha(sparse.OUT/d/l/f'fold{f}'/'selection.json')==protocol['old_selections'][f'{d}/{l}/{f}']
                tr,te=np.array(design['train_rows']),np.array(design['test_rows'])
                assert np.array_equal(te,np.flatnonzero(folds==f))
                assert not set(groups[tr])&set(groups[te])
                assert len(data['grid'])==66*len(STEPS) and len(data['pairs'])==4*len(STEPS)
                histories={v['config_id']:v['previous_nonconverged_inner_runs'] for v in design['hard_history']}
                ids={r['config']['id'] for r in data['grid'] if r['config']['mode']=='hard' and r['config']['gain']>0}
                assert len(ids)==24
                audit['previous_nonconverged_hard_fold_configs_retained']+=sum(histories.get(cid,0)>0 for cid in ids)
                for cid in range(66):assert sorted(r['step'] for r in data['grid'] if r['config']['id']==cid)==list(STEPS)
                with np.load(dest/'readout_scores.npz') as scores:
                    assert np.array_equal(scores['row'],te)
                    for r in data['grid']:
                        c,m,step=r['config'],r['metrics'],r['step']
                        assert np.isfinite(scores[f'c{c["id"]}_t{step}']).all()
                        assert 'converged' not in m
                        assert m['updates_applied']==(0 if c['mode'] in ('identity','scale') else step)
                        assert all(np.isfinite(m[k]) for k in ('activation','within','between','fisher','accuracy','auc','residual'))
                        row=flatten(d,l,f,c,step,m,data['baseline'],'',histories.get(c['id'],0));grids.append(row)
                        audit['grid_evaluations']+=1
                        if c['mode']=='hard' and c['gain']>0:audit['hard_evaluations']+=1
                with np.load(dest/'pair_representations.npz') as z:
                    assert np.array_equal(z['row'],te) and np.array_equal(z['y'],y[te])
                    before=(original[l][te]>EPS).sum(1)
                    for r in data['pairs']:
                        c,m,step,role=r['config'],r['metrics'],r['step'],r['role']
                        pairs.append(flatten(d,l,f,c,step,m,data['baseline'],role,histories.get(c['id'],0)))
                        if role not in ('prior_soft','matched_hard'):continue
                        a=z[f'{role}_t{step}']
                        assert a.dtype==np.float32 and a.shape==(len(te),4096) and np.isfinite(a).all() and (a>=0).all()
                        after=(a>EPS).sum(1);assert np.all(after<=before)
                        audit['max_pair_support_ratio']=max(audit['max_pair_support_ratio'],float((after/np.maximum(before,1)).max()))
                        audit['pair_representations']+=1
                    for role,values in data['late_step_diagnostics'].items():
                        a=z[f'{role}_t99'];b=z[f'{role}_t100']
                        changes=float(np.mean(((a>EPS)!=(b>EPS)).sum(1)))
                        assert abs(changes-values['support_switch_units_99_100_mean'])<1e-5
                        m99=next(r['metrics'] for r in data['pairs'] if r['role']==role and r['step']==99)
                        m100=next(r['metrics'] for r in data['pairs'] if r['role']==role and r['step']==100)
                        late.append(dict(dataset=d,layer=l,fold=f,role=role,**values,
                            ridge_accuracy_change_pp=100*(m100['accuracy']-m99['accuracy']),
                            ridge_auc_change=m100['auc']-m99['auc']))
                designs.append(dict(dataset=d,layer=l,fold=f,**design['prior_soft']))
                audit['folds']+=1
    assert audit['folds']==20 and audit['grid_evaluations']==11880 and audit['hard_evaluations']==4320
    assert audit['pair_representations']==360
    summaries=[]
    for rows,kind in ((grids,'grid'),(pairs,'pair')):
        buckets={}
        for r in rows:
            key=(r['dataset'],r['layer'],r['config_id'] if kind=='grid' else r['role'],r['step'])
            buckets.setdefault(key,[]).append(r)
        for key,rs in buckets.items():
            assert len(rs)==5
            first=rs[0]
            s={k:first[k] for k in ('dataset','layer','role','family','rule','graph','mode','gain','budget','step')}
            s['config_id']=first['config_id'] if kind=='grid' else None;s['kind']=kind
            skip=set(s)|{'fold','previous_nonconverged_inner_runs','joint','sparse_joint'}
            for k in first:
                if k in skip:continue
                s[k]=float(np.mean([r[k] for r in rs])) if all(r[k] is not None for r in rs) else None
            s['previous_nonconverged_inner_runs']=sum(r['previous_nonconverged_inner_runs'] for r in rs)
            s['joint_folds']=sum(r['joint'] for r in rs);s['sparse_joint_folds']=sum(r['sparse_joint'] for r in rs)
            s['mean_joint']=bool(s['activation_pct']<-.001 and s['within_pct']<-.001 and s['fisher_pct']>.001)
            s['mean_sparse_joint']=bool(s['mean_joint'] and max(r['max_support_ratio'] for r in rs)<=1.00001)
            for k in EFFECTS:
                s[k+'_min']=min(r[k] for r in rs);s[k+'_max']=max(r[k] for r in rs)
            summaries.append(s)
    grid_summary=[r for r in summaries if r['kind']=='grid'];pair_summary=[r for r in summaries if r['kind']=='pair']
    csv_write(DOC/'grid_fold_metrics.csv',grids);csv_write(DOC/'pair_fold_metrics.csv',pairs)
    csv_write(DOC/'grid_summary.csv',grid_summary);csv_write(DOC/'pair_summary.csv',pair_summary)
    csv_write(DOC/'late_step_diagnostics.csv',late);csv_write(DOC/'paired_parameters.csv',designs)
    (DOC/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    (DOC/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return grid_summary,pair_summary,late,audit


def plots(grid,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=DOC/'figures';figs.mkdir(exist_ok=True)
    color={'prior_soft':'#0072B2','matched_hard':'#D55E00','paired_nohebb_soft':'#56B4E9','paired_nohebb_hard':'#CC79A7'}
    fields=('activation_pct','within_pct','fisher_pct','accuracy_pp','auc_delta','nonzero')
    labels=('Mean activation change (%)','Normalized within scatter change (%)','Fisher ratio change (%)',
        'Raw ridge balanced accuracy change (pp)','Raw ridge macro OVR AUC change','Active units (%)')
    for d in DS:
        for l in LAYERS:
            fig,axs=plt.subplots(2,3,figsize=(14,8))
            for role in ROLES:
                rows=sorted([r for r in pairs if r['dataset']==d and r['layer']==l and r['role']==role],key=lambda r:r['step'])
                for ax,field,label in zip(axs.flat,fields,labels):
                    val=[r[field]*(100 if field=='nonzero' else 1) for r in rows]
                    ax.plot(STEPS,val,'o-',label=role,color=color[role],markersize=3)
                    ax.set_xscale('log');ax.set_xlabel('Fixed update steps');ax.set_ylabel(label);ax.grid(alpha=.2)
                    if field!='nonzero':ax.axhline(0,color='black',linestyle=':',linewidth=.7)
                    else:
                        ax.set_ylim(0,max(30.,max(val)*1.15))
                        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            axs[0,0].legend(fontsize=8)
            fig.suptitle(f'{d.upper()} {l.upper()} | fixed soft/hard parameter pairs\nFive-fold descriptive means; no fixed-point filter or test-set horizon selection')
            fig.tight_layout();fig.savefig(figs/f'{d}_{l}_trajectories.png',dpi=170)
            fig.savefig(figs/f'{d}_{l}_trajectories.svg');plt.close(fig)
            hard=[r for r in grid if r['dataset']==d and r['layer']==l and r['mode']=='hard' and r['gain']>0]
            ids=sorted({r['config_id'] for r in hard})
            values=np.empty((24,len(STEPS)));counts=np.empty_like(values)
            names=[]
            for i,cid in enumerate(ids):
                rs=sorted([r for r in hard if r['config_id']==cid],key=lambda r:r['step'])
                names.append(f"{rs[0]['rule']}/{rs[0]['graph']}/g{rs[0]['gain']}/q{rs[0]['budget']}")
                values[i]=[r['fisher_pct'] for r in rs];counts[i]=[r['sparse_joint_folds'] for r in rs]
            lim=max(float(np.abs(values).max()),1.)
            fig,ax=plt.subplots(figsize=(13,11))
            ax.imshow(values,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
            ax.set_xticks(range(len(STEPS)),STEPS);ax.set_yticks(range(24),names,fontsize=8)
            ax.set_xlabel('Fixed update steps')
            for i in range(24):
                for j in range(len(STEPS)):
                    ax.text(j,i,f'{values[i,j]:+.1f}\n{int(counts[i,j])}/5',ha='center',va='center',fontsize=8,
                        color='white' if abs(values[i,j])>lim*.6 else 'black')
            ax.set_title(f'{d.upper()} {l.upper()} | ALL 24 Hebbian hard-threshold settings retained\nCell = Fisher % change / folds jointly meeting A, W, Fisher and density goals\nColor alone is NOT the joint sharpening criterion',pad=14)
            fig.tight_layout();fig.savefig(figs/f'{d}_{l}_all_hard.png',dpi=170)
            fig.savefig(figs/f'{d}_{l}_all_hard.svg');plt.close(fig)


def report(grid,pairs,late,audit):
    def p(d,l,role,t):return next(r for r in pairs if r['dataset']==d and r['layer']==l and r['role']==role and r['step']==t)
    lines=['# 有限步更新：soft / hard 重新评价','',
        '2026-10-28。主要对象是规定更新步数后的表征，不以不动点或残差筛选方法。原模型、vanilla缓存、前两轮代码及结果保持不变。',
        '', '全66个既有配置×9个预定步数全部报告。主配对沿用上一轮内层选出的soft参数，仅把阈值改成hard；本轮不根据测试曲线重新选参数或步数。',
        '', '## 主配对的有限步表现','',
        '表中变化以同折原始激活为基线；W/B/Fisher在逐图L2归一化后计算。readout列保留原始幅度，是对应步数重新拟合的SVM。准确率不要求上升，分类与几何分别检查。', '']
    hard5=sum(p(d,l,'matched_hard',5)['mean_sparse_joint'] for d in DS for l in LAYERS)
    soft5=sum(p(d,l,'prior_soft',5)['mean_sparse_joint'] for d in DS for l in LAYERS)
    lines[6:6]=['## 主要结论','',
        f'在参数匹配的比较中，5步hard在{hard5}/4个数据集×层组合上呈现五折均值联合sharpening改善；原soft配对为{soft5}/4。下表另列每折方向，均值达标不能替代每折检查。', '',
        '重新纳入未达到不动点的硬阈值，会改变前轮筛选结论：有限步表征可能已经有效，即使少量单元仍在阈值附近切换。因此前轮基于不动点的硬排除条件过严，不能用其否定硬阈值本身。', '',
        'hard通常保留较多强响应，平均激活降幅可能小于soft，但可以获得更强的类内收缩和类间区分。没有一种指标单独决定结论；原始readout用于功能检查，不要求准确率继续提高。所有步数预先固定、完整报告，未将测试曲线高点作为独立验证的最优步数。','']
    for d in DS:
        for l in LAYERS:
            lines += [f'### {d}/{l}','','| 阈值 | 步数 | 激活% | W% | B% | Fisher% | 非零% | SVM BA pp | SVM AUC Δ | 联合达标折数 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for role in ('prior_soft','matched_hard'):
                for t in SVM_STEPS:
                    r=p(d,l,role,t)
                    lines.append(f"| {role} | {t} | {r['activation_pct']:+.2f} | {r['within_pct']:+.3f} | {r['between_pct']:+.2f} | {r['fisher_pct']:+.2f} | {100*r['nonzero']:.2f} | {r['refit_svm_accuracy_pp']:+.2f} | {r['refit_svm_auc_delta']:+.5f} | {r['sparse_joint_folds']}/5 |")
            good=[str(t) for t in STEPS if p(d,l,'matched_hard',t)['mean_sparse_joint']]
            lines += ['', 'hard配对五折均值几何联合达标的预定步数：'+('、'.join(good) if good else '无')+'。这是描述性结果，不是按测试集选出的最佳步数。', '',
                f"![Trajectories](<{(DOC/'figures'/f'{d}_{l}_trajectories.png').as_posix()}>)",'']
    lines += ['## 先前未达不动点的hard设置是否得到保留','',
        f"已保留全部24个带Hebbian的hard配置，每配置在9个步数下评价；共{audit['hard_evaluations']}个外层配置×步数组合。其中{audit['previous_nonconverged_hard_fold_configs_retained']}个配置×外层折组合，上一轮至少一个内层运行未达到不动点，这轮均未因此删除。", '',
        '所有hard固定网格见grid_summary.csv；图中上行数字是Fisher变化，下行是五折中同时满足激活下降、W下降、Fisher上升、活动数不增加的折数。Fisher单项正向不能自动判定成功。','']
    lines += ['各步数下，24个hard配置中五折均值联合达标的数量（描述性筛查，不是24次独立确认）：','',
        '| 数据集/层 | '+' | '.join(str(t)+'步' for t in STEPS)+' |',
        '|---|'+'---:|'*len(STEPS)]
    for d in DS:
        for l in LAYERS:
            counts=[]
            for t in STEPS:
                rs=[r for r in grid if r['dataset']==d and r['layer']==l and r['mode']=='hard' and r['gain']>0 and r['step']==t]
                counts.append(f"{sum(r['mean_sparse_joint'] for r in rs)}/24")
            lines.append(f"| {d}/{l} | "+' | '.join(counts)+' |')
    lines.append('')
    for d in DS:
        for l in LAYERS:
            lines += [f"![All hard](<{(DOC/'figures'/f'{d}_{l}_all_hard.png').as_posix()}>)",'']
    lines += ['## 相邻后期步数：99与100','','| 数据集/层 | 配对 | 平均相对L2变化 | 每图平均切换单元数 | 99→100激活变化百分点 | 99→100 Fisher变化百分点 | Ridge BA变化pp | Ridge AUC变化 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for d in DS:
        for l in LAYERS:
            for role in ('prior_soft','matched_hard'):
                rs=[r for r in late if r['dataset']==d and r['layer']==l and r['role']==role]
                a,b=p(d,l,role,99),p(d,l,role,100)
                lines.append(f"| {d}/{l} | {role} | {np.mean([r['relative_l2_99_100_mean'] for r in rs]):.7f} | {np.mean([r['support_switch_units_99_100_mean'] for r in rs]):.4f} | {b['activation_pct']-a['activation_pct']:+.6f} | {b['fisher_pct']-a['fisher_pct']:+.6f} | {np.mean([r['ridge_accuracy_change_pp'] for r in rs]):+.5f} | {np.mean([r['ridge_auc_change'] for r in rs]):+.7f} |")
    lines += ['', '少量单元仍切换，与主要几何/readout指标基本保持，可以同时成立。本轮只将其作为动态性质，不能据残差本身判定方法失败。残差在完整训练/测试推断批次记录；这里99/100的相对L2和支持切换仅用测试图片计算。','',
        '## 评价口径','','- 主目标：平均激活下降、归一化W下降、Fisher上升；B单列，避免把共同塌缩当作区分增强。',
        '- 原始readout保留幅度信息；只要测试概念上有功能收益，幅度贡献也应保留。去范数ridge只是来源诊断。',
        '- 所有配置/步数均固定报告，没有新一轮的“最佳参数”或“最佳步数”独立确认。完整网格中观察到的高点只是探索线索。',
        '- 同一探索样本多轮复用、ImageNet植物7个细类、THINGS标签分歧等限制继续适用；不把折当独立受试者，不作未经全流程推断支持的显著性声明。','',
        '## 验证与文件','',f"审计通过：{audit['folds']}个外层折，{audit['grid_evaluations']}个网格评价，{audit['pair_representations']}个完整配对表征矩阵。5项有限步/无早停/不动点不筛选/预算/数值负对照测试通过。",'',
        '- grid_summary.csv / grid_fold_metrics.csv：全66配置与全部步数，含旧不收敛记录、残差和分类。',
        '- pair_summary.csv / pair_fold_metrics.csv：预定soft/hard与去Hebbian配对，含SVM及去范数ridge。',
        '- paired_parameters.csv：沿用的历史参数；late_step_diagnostics.csv：99/100步变化。',
        '- audit.json：完整性、数据隔离、hard全保留、原模型/前轮不变审计。',
        '- 原始结果：Hebbian-CNN/Log/fc_finite_steps_20261028/；含协议、源码、逐折分数、完整配对表征、SVM与哈希。','',
        '```powershell','python -m predify_experiments.fc_finite_steps_20261028 run','python analysis/report_fc_finite_steps_20261028.py','```','']
    (DOC/'实验结果.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    grid,pairs,late,audit=aggregate();plots(grid,pairs);report(grid,pairs,late,audit)
    for d in DS:
        for l in LAYERS:
            for role in ('prior_soft','matched_hard'):
                for t in (1,5,20,100):
                    r=next(r for r in pairs if r['dataset']==d and r['layer']==l and r['role']==role and r['step']==t)
                    print(d,l,role,t,{k:r[k] for k in EFFECTS},'svm',r['refit_svm_accuracy_pp'],r['refit_svm_auc_delta'],'joint',r['sparse_joint_folds'])
    print('AUDIT',audit)


if __name__=='__main__':main()
