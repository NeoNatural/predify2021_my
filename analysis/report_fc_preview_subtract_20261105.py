"""Layer-resolved sharpening, readout and state audit for the single-beta subtractive pilot."""
from __future__ import annotations
import argparse
import csv
import pickle
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis import fc_preview_pilot_20261105 as exp
from analysis import fc_preview_refit_20261105 as previous
from analysis import fc_preview_subtract_20261105 as run
from analysis.report_fc_preview_pilot_20261105 import characterize,mean,change

LABEL=dict(vanilla='vanilla',inhibition='旧普通仅抑制参考',pool_only='记忆驱动仅抑制',hebbian='减法Hebbian串联',
    scale1='FC1缩放＋FC2减法Hebb',scale2='FC1减法Hebb＋FC2缩放',scale_both='双层缩放',shuffled='临时权重打乱',inhib_scale='旧仅抑制后均值匹配')
LAYER=dict(fc1='FC1输出',fc2_input='FC2实际输入',fc2='最终FC2输出')

def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def responses():
    old,p=run.protocol();summary=[];contrasts=[];perrows=[];local=[];pool=[]
    audit=dict(passed=True,report_source_sha256=exp.sha(__file__),protocol_sha256=exp.sha(run.OUT/'protocol.json'),datasets={},
        no_original_model_edits=True,no_SAE=True,no_search=True,clean_only=True)
    for d in ('imagenet','things'):
        run.verify(run.OUT/d/'pilot');a=exp.read(run.OUT/d/'pilot/audit.json');assert a['passed']
        with np.load(run.OUT/d/'pilot/representations.npz') as z:
            reps={k:z[k] for k in z.files if '__' in k};y,groups,row=z['y'],z['concept'],z['row']
        with (exp.OUT/d/'reference_svm.pckl').open('rb') as f:fixed=pickle.load(f)
        with np.load(exp.OUT/d/'nuisance_arrays.npz') as z:qmaps={l:z[l+'_quartiles'] for l in ('fc1','fc2')}
        ns=exp.nuisance(d,old);w,b=exp.load_bridge();stats={};pers={};scale_error=0.;bridge_error=0.
        import torch
        for m in run.ALL_METHODS:
            for h in exp.HISTORIES:
                for s in exp.STAGES:
                    key=f'{m}__{h}__{s}';v=reps[key];l='fc1' if s=='fc1' else 'fc2';base=reps[f'vanilla__exact__{s}']
                    assert np.isfinite(v).all() and (v>=0).all()
                    scores=fixed[l].decision_function(v);ss,per=characterize(v,scores,y,groups,qmaps[l],base)
                    incoming=base if s!='fc2' else reps[f'{m}__{h}__fc2_input']
                    active=v>1e-7;revived=active&(incoming<=1e-7)
                    ss.update(revived_units=mean(revived.mean(1),y,groups),
                        revived_mass=mean((v*revived).sum(1)/np.maximum(v.sum(1),1e-8),y,groups))
                    stats[m,h,s]=ss;pers[m,h,s]=per
                    summary.append(dict(dataset=d,method=m,history=h,stage=s,n=len(y),**ss))
                    for j,target in enumerate(row):
                        perrows.append(dict(dataset=d,target=int(target),category=int(y[j]),concept=str(groups[j]),method=m,history=h,stage=s,
                            activation=float(per['activation'][j]),nonzero=float(per['nonzero'][j]),preferred_share=float(per['preferred_share'][j]),
                            fixed_margin=float(per['margin'][j]),fixed_correct=int(per['correct'][j])))
                    if m!='inhib_scale' and s=='fc2_input':
                        with torch.inference_mode():expected=torch.relu(torch.nn.functional.linear(torch.from_numpy(reps[f'{m}__{h}__fc1']).cuda(),w,b)).cpu().numpy()
                        err=float(np.abs(expected-v).max());bridge_error=max(bridge_error,err);assert np.allclose(expected,v,atol=2e-5,rtol=2e-5)
                    if (m in ('scale1','scale_both') and s=='fc1') or (m in ('scale2','scale_both','inhib_scale') and s=='fc2') or (m=='inhib_scale' and s=='fc1'):
                        ref=reps[f'hebbian__{h}__{s}'];err=float(np.abs(v.mean(1)-ref.mean(1)).max());scale_error=max(scale_error,err);assert err<2e-6
        for m in run.ALL_METHODS:
            for s in exp.STAGES:
                for left,right in (('exact','different'),('exact','none'),('different','none')):
                    a,b=stats[m,left,s],stats[m,right,s];pa,pb=pers[m,left,s],pers[m,right,s]
                    rs=mean(100*(1-pa['activation']/np.maximum(pb['activation'],1e-12)),y,groups)
                    contrasts.append(dict(dataset=d,method=m,stage=s,contrast=f'{left}-{right}',suppression=rs,
                        within_change=change(a['within'],b['within']),between_change=change(a['between'],b['between']),
                        fisher_change=change(a['fisher'],b['fisher']),preferred_pp=100*(a['preferred_share']-b['preferred_share']),
                        fixed_ba_pp=100*(a['svm_accuracy']-b['svm_accuracy']),lower_targets=int(np.sum(pa['activation']<pb['activation'])),n=len(y)))
            for h in exp.HISTORIES:
                a,b=stats[m,h,'fc2'],stats[m,h,'fc2_input']
                local.append(dict(dataset=d,method=m,history=h,activation_change=change(a['activation'],b['activation']),
                    within_change=change(a['within'],b['within']),between_change=change(a['between'],b['between']),fisher_change=change(a['fisher'],b['fisher']),
                    preferred_pp=100*(a['preferred_share']-b['preferred_share']),fixed_ba_pp=100*(a['svm_accuracy']-b['svm_accuracy'])))
        zero=exp.PreviewMemory(4096).fingerprint();steps=0
        old_records=exp.read(exp.OUT/d/'pilot/trial_audit.json');expected={(r['target'],r['history']):r['before_target'] for r in old_records if r['method']=='hebbian'}
        a_records=exp.read(run.OUT/d/'pilot/audit.json')['records']
        # Explicit loop kept separate from response statistics for state acceptance.
        for r in a_records:
            assert r['reset_sha256']==[zero,zero] and r['before_target']==r['after_target']
            assert r['writes']==([0,0] if r['history']=='none' or r['method'] in ('vanilla','inhibition') else [2,2])
            if r['method']=='hebbian':assert r['before_target']==expected[r['target'],r['history']]
            for tr in r['traces']:
                for t in tr['updates']:assert t['weight_sha256']==r['before_target'][tr['layer']-1];steps+=1
                if r['method'] in ('hebbian','pool_only'):
                    for t in tr['updates']:
                        pool.append(dict(dataset=d,target=r['target'],method=r['method'],history=r['history'],layer='fc'+str(tr['layer']),**t))
        audit['datasets'][d]=dict(test=len(y),condition_trials=len(a_records),checked_updates=steps,
            dense_fp64_error=exp.read(run.OUT/d/'pilot/audit.json')['dense_fp64_error'],bridge_max_error=bridge_error,scale_mean_max_error=scale_error)
    run.DOC.mkdir(exist_ok=True,parents=True)
    for name,data in (('condition_metrics',summary),('history_contrasts',contrasts),('fc2_local_effects',local),('pool_diagnostics',pool),('per_target_metrics',perrows)):
        save_csv(run.DOC/(name+'.csv'),data)
        if name!='per_target_metrics':exp.write_json(run.DOC/(name+'.json'),data)
    exp.write_json(run.DOC/'response_audit.json',audit);exp.validate(old)
    print('RESPONSE AUDIT',audit,flush=True)

def finish():
    old,p=run.protocol();summary=exp.read(run.DOC/'condition_metrics.json');contrasts=exp.read(run.DOC/'history_contrasts.json');readouts=[]
    audit=exp.read(run.DOC/'response_audit.json');assert audit['passed'];audit['readouts']={};audit['final_report_source_sha256']=exp.sha(__file__)
    individual=[]
    for d in ('imagenet','things'):
        run.verify(run.OUT/d/'training');run.verify(run.OUT/d/'readouts')
        readouts.extend(exp.read(run.OUT/d/'readouts/metrics.json'))
        with (exp.OUT/d/'reference_svm.pckl').open('rb') as f:fixed=pickle.load(f)
        with np.load(run.OUT/d/'training/representations.npz') as z:train={k:z[k] for k in z.files if '__' in k};y,g=z['y'],z['concept']
        with (run.OUT/d/'readouts/models.pckl').open('rb') as f:bank=pickle.load(f)
        checked=set();max_mu_error=0.
        for key,sig in bank['mapping'].items():
            if sig in checked:continue
            checked.add(sig);parts=key.split('__');kind,m,s=parts[0],parts[1],parts[-1]
            if kind=='shared':x,yy,w=previous.pool_histories([train[f'{m}__{h}__{s}'] for h in exp.HISTORIES],y,g)
            else:x,yy,w=train['__'.join(parts[1:])],y,exp.weights(y,g)
            assert sig==previous.signature(x,yy,w) and abs(w.sum()-len(y))<1e-7
            head=bank['heads'][sig];assert head[-1].C==1 and head[-1].n_iter_<30000
            err=float(np.abs(np.average(x.astype(np.float64),axis=0,weights=w)-head[0].mean_).max());assert err<1e-9;max_mu_error=max(max_mu_error,err)
        note=exp.read(run.OUT/d/'readouts/audit.json');audit['readouts'][d]=dict(train=note['train_count'],test=note['test_count'],models=len(checked),
            no_test_fit=note['test_not_used_for_fit'],scaler_mean_error=max_mu_error,max_iter=int(max(h[-1].n_iter_ for h in bank['heads'].values())))
        score_error=0.
        with np.load(run.OUT/d/'readouts/scores.npz') as z,np.load(run.OUT/d/'pilot/representations.npz') as rep:
            ty,tg,tr=z['y'],z['concept'],z['row']
            for key in z.files:
                if '__' not in key:continue
                kind,m,h,s=key.split('__');v=z[key];pred=v.argmax(1)
                original=fixed['fc1' if s=='fc1' else 'fc2'];x=rep[f'{m}__{h}__{s}']
                if kind=='fixed':replay=original.decision_function(x)
                elif kind=='separate':replay=bank['heads'][bank['mapping'][f'separate__{m}__{h}__{s}']].decision_function(x)
                else:
                    shared=bank['heads'][bank['mapping'][f'shared__{m}__{s}']]
                    replay=shared.decision_function(x) if kind=='shared' else original[-1].decision_function(shared[0].transform(x))
                err=float(np.abs(replay-v).max());assert err<1e-10;score_error=max(score_error,err)
                margins=v[np.arange(len(ty)),ty]-np.where(np.arange(v.shape[1])[None,:]==ty[:,None],-np.inf,v).max(1)
                for j,target in enumerate(tr):individual.append(dict(dataset=d,target=int(target),concept=str(tg[j]),category=int(ty[j]),
                    method=m,history=h,stage=s,readout=kind,prediction=int(pred[j]),correct=int(pred[j]==ty[j]),margin=float(margins[j])))
        audit['readouts'][d]['score_replay_max_error']=score_error
    save_csv(run.DOC/'readout_metrics.csv',readouts);exp.write_json(run.DOC/'readout_metrics.json',readouts)
    save_csv(run.DOC/'per_target_readout.csv',individual)
    def absolute(d,m,s,h):return next(r for r in summary if (r['dataset'],r['method'],r['stage'],r['history'])==(d,m,s,h))
    def contrast(d,m,s,left,right):return next(r for r in contrasts if (r['dataset'],r['method'],r['stage'],r['contrast'])==(d,m,s,f'{left}-{right}'))
    def svm(d,m,s,h,k):return next(r for r in readouts if (r['dataset'],r['method'],r['stage'],r['history'],r['readout'])==(d,m,s,h,k))
    lines=['# 原FC空间：单一β=4减法抑制小批量结果','',
        '只改变target读取机制，保留每trial清零、两帧preview临时写入、target只读以及真实FC1→FC2串联。没有SAE、参数搜索、噪声/遮挡或原模型修改。测试仍为ImageNet16张、THINGS18张。机制解读见同目录《结果解读与机制诊断.md》。','',
        '固定参数：ρ=.5、γ=.75、β=4、α=.5、每层5步。β=4直接沿用旧卷积默认值，没有按本测试集或训练表现挑选。新读取：h=γWr，v=ReLU(x+h−βmean(h))，r下一步=.5r+.5v；不额外叠加旧硬门控/除法。与旧方案并非等平均抑制强度比较，不能将单组结果称为最优机制。','',
        '**基线定义变化需注意：新规则W=0时等于vanilla；上一轮无preview则等于旧普通仅抑制。** 因此跨轮同时报告Exact−Different及相同vanilla参照。','',
        '“记忆驱动仅抑制”保留本trial的W和抑制池，移除局部+h；“旧普通仅抑制参考”则完全复用上一轮的无记忆硬门控＋除法。缩放为同历史主方案逐图均值匹配；三项缩放均有真实桥接。“旧仅抑制后均值匹配”是逐层事后诊断，不是实际串联。','',
        '0.5混合发生在ReLU之后，因此净输入被清零的单元仍可能保留先前状态的衰减尾部；固定5步不要求不动点收敛。没有把小尾部另行截断。','',
        '## 主方案：成对suppression与几何','',
        'suppression为逐target的100(1−左/右)后按类别/概念等权，正值表示响应更低。W/B为L2归一化后的类内/类间统计，Fisher=B/W。偏好组由原训练折定义；非零比例沿用逐图等权，活动及偏好份额按类别/概念等权。','',
        '|数据|层|比较|suppression%↑|Δ类内W%↓|Δ类间B%|ΔFisher%↑|Δ最高偏好份额pp↑|活动更低target数|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for s in ('fc1','fc2'):
            for a,b in (('exact','different'),('exact','none')):
                r=contrast(d,'hebbian',s,a,b)
                lines.append(f'|{d}|{LAYER[s]}|{a}−{b}|'+ '|'.join(f"{r[k]:+.3f}" for k in ('suppression','within_change','between_change','fisher_change','preferred_pp'))+f"|{r['lower_targets']}/{r['n']}|")
    lines+=['','## 主方案：绝对指标与SVM','',
        '固定头保留原vanilla训练的SVM；共享适配头使用256/320个原训练折target在同一新机制下的训练表示，三历史各权重1/3，总权重不变。各历史单独拟合为辅助诊断。均为加权StandardScaler＋LinearSVC(C=1)，不调参；测试及preview概念/原图均与训练隔离。FC2输入只报告固定头诊断，不额外拟合适配头。','',
        '|数据|层|历史|平均活动|Fisher|非零%|偏好份额%|固定BA%|共享BA%|分条件BA%|共享AUC|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for s in ('fc1','fc2'):
            for h in exp.HISTORIES:
                a=absolute(d,'hebbian',s,h);sv=[svm(d,'hebbian',s,h,k) for k in ('fixed','shared','separate')]
                lines.append(f"|{d}|{LAYER[s]}|{h}|{a['activation']:.6f}|{a['fisher']:.6f}|{100*a['nonzero']:.3f}|{100*a['preferred_share']:.3f}|"+'|'.join(f"{100*x['ba']:.2f}" for x in sv)+f"|{sv[1]['auc']:.5f}|")
    lines+=['','## 全部对照：Exact−Different与Exact读出','',
        '|数据|层|方法|suppression%|ΔFisher%|Δ偏好pp|Exact固定BA%|Exact共享BA%|Exact共享AUC|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for s in ('fc1','fc2'):
            for m in run.ALL_METHODS:
                r=contrast(d,m,s,'exact','different');f=svm(d,m,s,'exact','fixed');a=svm(d,m,s,'exact','shared')
                lines.append(f"|{d}|{LAYER[s]}|{LABEL[m]}|{r['suppression']:+.3f}|{r['fisher_change']:+.3f}|{r['preferred_pp']:+.3f}|{100*f['ba']:.2f}|{100*a['ba']:.2f}|{a['auc']:.5f}|")
    old_stats=exp.read(exp.DOC/'condition_metrics.json');old_cs=exp.read(exp.DOC/'paired_history_contrasts.json')
    lines+=['','## 与上一轮除法＋门控主方案比较','',
        '两轮使用相同target、preview临时权重、增益、步数和原始桥接；读取形式改变，并取消旧普通抑制底座。旧方案无preview≠新方案无preview，不能混用。下面均以同一vanilla作为活动和几何参照，活动下降为汇总均值之比。','',
        '|数据|层|读取|Exact相对vanilla均值下降%|Exact相对vanilla ΔFisher%|Exact−Different suppression%|',
        '|---|---|---|---:|---:|---:|']
    for d in ('imagenet','things'):
        for s in ('fc1','fc2'):
            raw=absolute(d,'vanilla',s,'exact')
            for family in ('previous','subtract'):
                if family=='previous':
                    a=next(r for r in old_stats if (r['dataset'],r['method'],r['stage'],r['history'])==(d,'hebbian',s,'exact'))
                    c=next(r for r in old_cs if (r['dataset'],r['method'],r['stage'],r['contrast'])==(d,'hebbian',s,'E-D'))
                    rs=c['suppression'] if 'suppression' in c else c['suppression_pct']
                else:a=absolute(d,'hebbian',s,'exact');rs=contrast(d,'hebbian',s,'exact','different')['suppression']
                name='旧除法＋门控' if family=='previous' else '新减法'
                lines.append(f"|{d}|{LAYER[s]}|{name}|{100*(1-a['activation']/raw['activation']):.3f}|{change(a['fisher'],raw['fisher']):+.3f}|{rs:+.3f}|")
    local=exp.read(run.DOC/'fc2_local_effects.json')
    lines+=['','## FC2局部变化','',
        '|数据|历史|FC2相对实际输入：均值变化%|Δ类内W%|Δ类间B%|ΔFisher%|Δ偏好pp|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in local:
        if r['method']!='hebbian':continue
        lines.append(f"|{r['dataset']}|{r['history']}|"+'|'.join(f"{r[k]:+.3f}" for k in ('activation_change','within_change','between_change','fisher_change','preferred_pp'))+'|')
    lines+=['','## 验收与解释边界','',
        '新pilot逐trial preview权重哈希与旧pilot完全一致；每trial清零、写入次数、target各步只读、倒序重放、真实桥接与缩放均值匹配均已核对。实际4096维FP64显式矩阵独立复算误差见audit.json。原普通仅抑制及vanilla输出逐元素重放一致。','',
        '补拟合仅使用原训练折，固定头和适配头均保留；适配恢复不等于全部信息保留，不恢复也不证明信息永久丢失。共享头内历史对比使用同一边界；分条件头之间边界不同。即使两个方法输出相同，因适配头训练历史池不同也可能给出不同分数。','',
        'β=4是单组预先指定默认值；两数据集总计34个测试target、仅一个权重置换/target/层。ImageNet为8个开发概念，THINGS为9个（植物仅aloe），所有prime仍位于第二帧，未扩充布局。结果不代表最优参数或可靠的置换显著性。完整逐target、池活动、各类别recall及分层读出在相邻CSV/JSON。所有测试仍为clean。','']
    (run.DOC/'减法抑制小批量结果.md').write_text('\n'.join(lines),encoding='utf-8')
    exp.validate(old);run.protocol();exp.write_json(run.DOC/'audit.json',audit)
    plot(summary,contrasts)
    exp.write_json(run.DOC/'complete.json',dict(protocol_sha256=exp.sha(run.OUT/'protocol.json'),
        files={str(p.relative_to(run.DOC)):exp.sha(p) for p in run.DOC.rglob('*') if p.is_file() and p.name!='complete.json'}))
    print('FINAL AUDIT',audit,flush=True)

def plot(summary,contrasts):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    ms=['hebbian','pool_only','scale_both','shuffled'];names=['Hebb subtract','Memory inhibition','Both scales','Shuffled']
    for row,d in enumerate(('imagenet','things')):
        for col,s in enumerate(('fc1','fc2')):
            ax=axes[row,col];x=np.arange(len(ms));vals=[]
            for m in ms:
                vals.append(next(r['suppression'] for r in contrasts if (r['dataset'],r['method'],r['stage'],r['contrast'])==(d,m,s,'exact-different')))
            ax.bar(x,vals,color=['#277da1','#90be6d','#f9c74f','#999999']);ax.axhline(0,c='black',lw=.7)
            ax.set_xticks(x,names,rotation=15,ha='right');ax.set_title(f'{d} / {s.upper()}');ax.set_ylabel('Exact vs Different suppression (%)')
    dest=run.DOC/'figures';dest.mkdir(exist_ok=True)
    for ext in ('png','svg','pdf'):fig.savefig(dest/f'exact_different_suppression.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    cases=[('hebbian','exact'),('hebbian','different'),('pool_only','exact'),('scale_both','exact'),('shuffled','exact')]
    names=['Hebb E','Hebb D','Pool-only E','Scale E','Shuffle E']
    for row,d in enumerate(('imagenet','things')):
        for col,s in enumerate(('fc1','fc2')):
            ax=axes[row,col];base=next(r for r in summary if(r['dataset'],r['method'],r['stage'],r['history'])==(d,'vanilla',s,'exact'))
            values=[next(r for r in summary if(r['dataset'],r['method'],r['stage'],r['history'])==(d,m,s,h)) for m,h in cases]
            x=np.arange(len(cases));ax.bar(x-.18,[r['activation']/base['activation'] for r in values],.36,label='Mean activity',color='#277da1')
            ax.bar(x+.18,[r['fisher']/base['fisher'] for r in values],.36,label='Fisher',color='#f8961e')
            ax.axhline(1,c='black',lw=.8,ls='--');ax.set_xticks(x,names,rotation=15,ha='right');ax.set_ylabel('Ratio to vanilla')
            ax.set_title(f'{d} / {s.upper()}');ax.set_ylim(0,1.25)
            if row==0 and col==0:ax.legend(frameon=False)
    for ext in ('png','svg','pdf'):fig.savefig(dest/f'activity_and_fisher.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['responses','finish']);a=p.parse_args()
    exp.setup()
    with threadpool_limits(limits=2):{'responses':responses,'finish':finish}[a.action]()
