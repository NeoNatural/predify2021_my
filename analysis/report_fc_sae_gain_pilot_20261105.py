"""Audit the tiny SAE gain-only pilot using physical FC metrics and readouts exclusively."""
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
from analysis import fc_sae_gain_pilot_20261105 as run
from predify_experiments.fc_sae_gain_20261105 import Memory
from analysis.report_fc_preview_pilot_20261105 import characterize,mean,change

LABEL=dict(vanilla='vanilla',inhibition='旧普通仅抑制参考',pool_only='记忆驱动仅抑制',hebbian='轻量SAE增益串联',
    scale1='FC1缩放＋FC2代理增益',scale2='FC1代理增益＋FC2缩放',scale_both='双层缩放',shuffled='latent权重打乱',inhib_scale='旧仅抑制后均值匹配')
LAYER=dict(fc1='FC1输出',fc2_input='FC2实际输入',fc2='最终FC2输出')

def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def responses():
    old,p=run.protocol();summary=[];contrasts=[];perrows=[];local=[];pool=[]
    audit=dict(passed=True,report_source_sha256=exp.sha(__file__),protocol_sha256=exp.sha(run.OUT/'protocol.json'),datasets={},
        no_original_model_edits=True,SAE_gain_only=True,no_proxy_output_readouts=True,no_search=True,clean_only=True)
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
                    assert v.shape[1]==4096
                    if s in ('fc1','fc2') and m in ('hebbian','pool_only','shuffled'):
                        assert np.count_nonzero(v[incoming==0])==0,'Multiplicative side path revived an original zero'
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
        zero=Memory(64).fingerprint();reference_zero=exp.PreviewMemory(4096).fingerprint();steps=0
        proxies=[exp.read(run.OUT/d/'proxy'/f'{l}_acceptance.json') for l in ('fc1','fc2')]
        for p in proxies:
            assert p['passed'] and p['budget']['trainable']==2112 and p['budget']['tensor_payload_bytes']==29316
            assert p['projection_row_orthogonality_max_error']<1e-6
        expected_proxy=[p['state_sha256'] for p in proxies]
        a_records=exp.read(run.OUT/d/'pilot/audit.json')['records']
        # Explicit loop kept separate from response statistics for state acceptance.
        for r in a_records:
            assert r['reset_sha256']==[zero,zero] and r['before_target']==r['after_target']
            assert r['writes']==([0,0] if r['history']=='none' or r['method'] in ('vanilla','inhibition') else [2,2])
            assert r['proxy_before']==r['proxy_after']==expected_proxy
            for tr in r['traces']:
                expected=reference_zero if r['method']=='inhibition' else r['before_target'][tr['layer']-1]
                for t in tr['updates']:assert t['weight_sha256']==expected;steps+=1
                if r['method'] in ('hebbian','pool_only'):
                    for t in tr['updates']:
                        pool.append(dict(dataset=d,target=r['target'],method=r['method'],history=r['history'],layer='fc'+str(tr['layer']),**t))
        audit['datasets'][d]=dict(test=len(y),condition_trials=len(a_records),checked_updates=steps,
            dense_fp64_error=exp.read(run.OUT/d/'pilot/audit.json')['dense_fp64_error'],bridge_max_error=bridge_error,scale_mean_max_error=scale_error,
            proxy_acceptance=proxies,original_zero_revival_count=0)
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
            head=bank['heads'][sig];assert head[-1].C==1 and head[-1].n_iter_<30000 and head[-1].n_features_in_==4096
            err=float(np.abs(np.average(x.astype(np.float64),axis=0,weights=w)-head[0].mean_).max());assert err<1e-9;max_mu_error=max(max_mu_error,err)
        note=exp.read(run.OUT/d/'readouts/audit.json');audit['readouts'][d]=dict(train=note['train_count'],test=note['test_count'],models=len(checked),
            no_test_fit=note['test_not_used_for_fit'],scaler_mean_error=max_mu_error,max_iter=int(max(h[-1].n_iter_ for h in bank['heads'].values())))
        score_error=0.
        fixed_agreement={}
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
                if kind=='fixed' and m=='hebbian' and s in ('fc1','fc2'):
                    base=original.decision_function(rep[f'vanilla__exact__{s}'])
                    fixed_agreement[f'{s}__{h}']=dict(prediction_agreement=float(np.mean(v.argmax(1)==base.argmax(1))),
                        maximum_score_change=float(np.abs(v-base).max()))
                margins=v[np.arange(len(ty)),ty]-np.where(np.arange(v.shape[1])[None,:]==ty[:,None],-np.inf,v).max(1)
                for j,target in enumerate(tr):individual.append(dict(dataset=d,target=int(target),concept=str(tg[j]),category=int(ty[j]),
                    method=m,history=h,stage=s,readout=kind,prediction=int(pred[j]),correct=int(pred[j]==ty[j]),margin=float(margins[j])))
        audit['readouts'][d]['score_replay_max_error']=score_error
        audit['readouts'][d]['fixed_main_vs_vanilla']=fixed_agreement
    save_csv(run.DOC/'readout_metrics.csv',readouts);exp.write_json(run.DOC/'readout_metrics.json',readouts)
    save_csv(run.DOC/'per_target_readout.csv',individual)
    runtime=[benchmark(d) for d in ('imagenet','things')]
    exp.write_json(run.DOC/'runtime_payload.json',runtime);audit['runtime_payload']=runtime
    def absolute(d,m,s,h):return next(r for r in summary if (r['dataset'],r['method'],r['stage'],r['history'])==(d,m,s,h))
    def contrast(d,m,s,left,right):return next(r for r in contrasts if (r['dataset'],r['method'],r['stage'],r['contrast'])==(d,m,s,f'{left}-{right}'))
    def svm(d,m,s,h,k):return next(r for r in readouts if (r['dataset'],r['method'],r['stage'],r['history'],r['readout'])==(d,m,s,h,k))
    lines=['# 轻量SAE只计算Hebbian增益：状态验收与clean小批量','',
        '使用唯一冻结的轻量方案：SAE仅计算乘性Hebbian增益，原4096维FC响应继续前馈并送入SVM。不生成完整FC重建、不拟合latent或残差SVM。原模型未修改，没有参数搜索或噪声/遮挡。测试仍为ImageNet16张、THINGS18张，readout仍为4/5大类。机制解读见同目录《结果解读与机制诊断.md》。','',
        '辅助结构为固定P=RHS（4096→32）＋32→64→32共享权重ReLU SAE。快速Hadamard算子与转置不保存稠密投影矩阵。只用训练fold1–4的256/320个target和648/828张去重preview拟合，角色损失各占一半，无类别监督；L1=.1，Adam固定1000步、lr .001→.0001，只用最后一步。测试概念及视觉角色不参与拟合。','',
        '固定更新：ρ=.5、η=.75、β=4、α=.5、每层5步。z=E(P r)，h=Wz z，q=PᵀAᵀh，m=.75tanh(q)，正向增加量B⁺=x×max(m,0)，抑制池I=4mean(B⁺)，候选v=ReLU(x×(1+m)−I)，r下一步=.5r+.5v。所有均值在单张图内计算。x为实际到达该层的原4096维输入，不以SAE重建替代。','',
        '**无preview时Wz=0、增益=1、池=0，完整方案严格等于vanilla。** 每trial清零；preview两帧编码只写临时64维记忆；target只读，编码器、解码器及投影统计均冻结。FC2预览记忆仍来自原vanilla preview，target则处理FC1修改后经真实桥接得到的输入。','',
        '“记忆驱动仅抑制”保留该trial权重及其产生的抑制池，将局部乘性增益设为1；“旧普通仅抑制参考”完全复用原无记忆硬门控＋除法。三项缩放对照按当前历史主方案逐图匹配均值并保留真实桥接。“旧仅抑制后均值匹配”仅是逐层事后幅度诊断。打乱在64维latent单元对应上执行，同一trial/层的Exact与Different共用同一置换。','',
        '每层可训练参数2112、固定浮点统计33、显式临时Hebbian状态4096，共6241个浮点数；固定投影signs为4096字节、rows为256字节，全部张量负载29316字节，低于原8194个FP32数值的32776字节。种子等少量Python元数据、瞬时工作空间与训练优化器状态不在张量负载中。辅助坐标重建只作训练监控。','',
        '乘性通路及非负抑制不点亮原本严格为0的局部FC单元；桥接仍可能改变FC2输入的活动集。混合在ReLU后，原弱响应可能留有有限步尾部，未追加截断或固定活动数量。','',
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
    old_doc=exp.ROOT/'docs/fc_preview_subtract_20261105'
    old_stats=exp.read(old_doc/'condition_metrics.json');old_cs=exp.read(old_doc/'history_contrasts.json')
    lines+=['','## 与原FC加性减法主方案比较','',
        '两轮使用相同target、vanilla preview、原始桥接和有限步数，但记忆坐标、驱动形式及有效干预强度不同。两者无preview均为vanilla。下面只是相同参照下的描述，不能把更接近vanilla直接称为更好的sharpening。活动下降为汇总均值之比，与逐target suppression略有差异。','',
        '|数据|层|读取|Exact相对vanilla均值下降%|Exact相对vanilla ΔFisher%|Exact−Different suppression%|',
        '|---|---|---|---:|---:|---:|']
    for d in ('imagenet','things'):
        for s in ('fc1','fc2'):
            raw=absolute(d,'vanilla',s,'exact')
            for family in ('previous','subtract'):
                if family=='previous':
                    a=next(r for r in old_stats if (r['dataset'],r['method'],r['stage'],r['history'])==(d,'hebbian',s,'exact'))
                    c=next(r for r in old_cs if (r['dataset'],r['method'],r['stage'],r['contrast'])==(d,'hebbian',s,'exact-different'))
                    rs=c['suppression'] if 'suppression' in c else c['suppression_pct']
                else:a=absolute(d,'hebbian',s,'exact');rs=contrast(d,'hebbian',s,'exact','different')['suppression']
                name='原FC加性减法' if family=='previous' else '轻量SAE乘性增益'
                lines.append(f"|{d}|{LAYER[s]}|{name}|{100*(1-a['activation']/raw['activation']):.3f}|{change(a['fisher'],raw['fisher']):+.3f}|{rs:+.3f}|")
    local=exp.read(run.DOC/'fc2_local_effects.json')
    lines+=['','## FC2局部变化','',
        '|数据|历史|FC2相对实际输入：均值变化%|Δ类内W%|Δ类间B%|ΔFisher%|Δ偏好pp|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in local:
        if r['method']!='hebbian':continue
        lines.append(f"|{r['dataset']}|{r['history']}|"+'|'.join(f"{r[k]:+.3f}" for k in ('activation_change','within_change','between_change','fisher_change','preferred_pp'))+'|')
    lines+=['','## 参数负载与参考运行开销','',
        '下表对每个数据集的第一个Exact target预热一次后重放3次，包含当前审计框架的哈希核对和桥接辅助计算；不是优化后的纯算法速度。峰值为PyTorch已分配张量相对预热后常驻基线的额外量，不含驱动等非PyTorch分配。','',
        '|数据|含审计的单trial平均秒数|额外峰值张量字节|单层模块常驻张量字节|',
        '|---|---:|---:|---:|']
    for r in runtime:lines.append(f"|{r['dataset']}|{r['seconds_per_trial']:.5f}|{r['extra_peak_tensor_bytes']}|29316|")
    lines+=['','## 验收与解释边界','',
        '每trial清零、preview写入次数、改变target不改变已写权重、target写入拒绝、target各步记忆只读、代理参数冻结、倒序重放、真实桥接、缩放均值匹配及原严格零单元不被局部点亮均已核对。FP64显式64维外积写入与原4096维乘性串联独立复算误差见audit.json；固定投影行正交性实际4096维检查通过。没有声称64维记忆与旧4096维权重数值相同。原普通仅抑制及vanilla输出逐元素重放一致。','',
        '补拟合仅使用原训练折，固定头和适配头均保留；适配恢复不等于全部信息保留，不恢复也不证明信息永久丢失。共享头内历史对比使用同一边界；分条件头之间边界不同。即使两个方法输出相同，因适配头训练历史池不同也可能给出不同分数。','',
        'β=4是单组预先指定默认值；两数据集总计34个测试target、仅一个权重置换/target/层。ImageNet为8个开发概念，THINGS为9个（植物仅aloe），所有prime仍位于第二帧，未扩充布局。结果不代表最优参数或可靠的置换显著性。完整逐target、池活动、各类别recall及分层读出在相邻CSV/JSON。所有测试仍为clean。','']
    (run.DOC/'轻量SAE增益小批量结果.md').write_text('\n'.join(lines),encoding='utf-8')
    exp.validate(old);run.protocol();exp.write_json(run.DOC/'audit.json',audit)
    plot(summary,contrasts)
    exp.write_json(run.DOC/'complete.json',dict(protocol_sha256=exp.sha(run.OUT/'protocol.json'),
        files={str(p.relative_to(run.DOC)):exp.sha(p) for p in run.DOC.rglob('*') if p.is_file() and p.name!='complete.json'}))
    print('FINAL AUDIT',audit,flush=True)

def benchmark(d):
    import time
    import torch
    from predify_experiments.fc_sae_gain_20261105 import GainProxy,Projection,TrialEngine,model_hash
    proxies=[]
    for l in ('fc1','fc2'):
        dest=run.OUT/d/'proxy';note=exp.read(dest/(l+'.json'));state=torch.load(dest/(l+'.pth'),map_location='cpu',weights_only=True)
        p=GainProxy(Projection(4096,32,note['seed']+1000),state['center'],float(state['scale']),64,note['seed']+2000)
        p.load_state_dict(state,strict=True);proxies.append(p.cuda().eval().requires_grad_(False))
    w,b=exp.load_bridge();scales=exp.read(exp.OUT/d/'nuisance.json')['scales'];engine=TrialEngine(w,b,proxies,(scales['fc1'],scales['fc2']))
    a=exp.features(d);x=torch.from_numpy(a['target_fc1'][:1]).cuda();p1=torch.from_numpy(a['preview_fc1'][0,0]).cuda();p2=torch.from_numpy(a['preview_fc2'][0,0]).cuda()
    initial=[model_hash(p) for p in proxies]
    with torch.inference_mode():
        warm=engine.run(x,p1,p2,'hebbian');del warm;torch.cuda.synchronize()
        base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for _ in range(3):
            output=engine.run(x,p1,p2,'hebbian');del output
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
    assert initial==[model_hash(p) for p in proxies]
    return dict(dataset=d,existing_target=int(a['row'][0]),repetitions=3,seconds_per_trial=float(elapsed/3),
        base_tensor_bytes=int(base),extra_peak_tensor_bytes=int(peak-base),includes_audit_overhead=True,proxy_unchanged=True)

def plot(summary,contrasts):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    ms=['hebbian','pool_only','scale_both','shuffled'];names=['SAE gain','Memory inhibition','Both scales','Shuffled']
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
            x=np.arange(len(cases));ax.bar(x-.18,[100*(r['activation']/base['activation']-1) for r in values],.36,label='Mean activity',color='#277da1')
            ax.bar(x+.18,[100*(r['fisher']/base['fisher']-1) for r in values],.36,label='Fisher',color='#f8961e')
            ax.axhline(0,c='black',lw=.8,ls='--');ax.set_xticks(x,names,rotation=15,ha='right');ax.set_ylabel('Change from vanilla (%)')
            ax.set_title(f'{d} / {s.upper()}')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.04),ncol=2,frameon=False)
    for ext in ('png','svg','pdf'):fig.savefig(dest/f'activity_and_fisher.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['responses','finish']);a=p.parse_args()
    exp.setup()
    with threadpool_limits(limits=2):{'responses':responses,'finish':finish}[a.action]()
