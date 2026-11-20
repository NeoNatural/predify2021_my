"""Layer-separated report of the frozen gated six-point clean pilot."""
from pathlib import Path
import sys,csv,pickle
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.fc_sae_gated_search_20261106 import OUT,DOC,protocol,GRID,METHODS,HIST,LAYERS,svm_metrics
from analysis import fc_preview_pilot_20261105 as exp

def csvfile(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (DOC/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def finish():
    old,p=protocol();selection=exp.read(OUT/'selection.json');cal=exp.read(OUT/'calibration.json');screen=exp.read(OUT/'training_screen.json')
    independent=exp.read(OUT/'independent_acceptance.json');assert independent['passed']
    summary=[];contrasts=[];svms=[];per=[];audits=[];traces=[];local=[];score_error=0.
    for d in ('imagenet','things'):
        dest=OUT/d;marker=exp.read(dest/'complete.json');assert marker['passed'] and marker['protocol_sha256']==exp.sha(OUT/'protocol.json')
        for f,h in marker['files'].items():assert exp.sha(dest/f)==h,f
        aa=exp.read(dest/'audit.json');assert aa['passed'] and aa['selection_sha256']==exp.sha(OUT/'selection.json');audits.append(aa)
        summary.extend(exp.read(dest/'summary.json'));contrasts.extend(exp.read(dest/'contrasts.json'));svms.extend(exp.read(dest/'readouts.json'));per.extend(exp.read(dest/'per_target.json'))
        with (exp.OUT/d/'reference_svm.pckl').open('rb') as f:fixed=pickle.load(f)
        with np.load(dest/'representations.npz') as rep,np.load(dest/'scores.npz') as scores:
            y,g=scores['y'],scores['concept']
            for key in scores.files:
                if '__' not in key:continue
                kind,method,h,l=key.split('__')
                if kind=='fixed':head=fixed[l]
                else:
                    info=next(i for i in aa['fit_info'] if i['method']==method and i['layer']==l)
                    path=dest/'heads'/(info['training_signature']+'.pckl');assert exp.sha(path)==info['sha256']
                    with path.open('rb') as f:head=pickle.load(f)
                assert head[-1].C==1 and head[-1].n_features_in_==4096
                replay=head.decision_function(rep[f'{method}__{h}__{l}']);err=float(np.abs(replay-scores[key]).max());score_error=max(score_error,err);assert err<1e-10
            for method in METHODS:
                for h in HIST:
                    v=rep[f'{method}__{h}__fc2'];incoming=rep[f'{method}__{h}__fc2_input']
                    def geom(x):
                        u=x.astype(np.float64);u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
                        centers=np.stack([u[y==c].mean(0) for c in np.unique(y)])
                        w=np.mean([np.square(u[y==c]-centers[k]).sum(1).mean() for k,c in enumerate(np.unique(y))]);b=np.square(centers-centers.mean(0)).sum(1).mean()
                        return b/w
                    local.append(dict(dataset=d,method=method,history=h,local_suppression=100*(1-np.average(v.mean(1),weights=exp.weights(y,g))/np.average(incoming.mean(1),weights=exp.weights(y,g))),
                        local_fisher_change=100*(geom(v)/geom(incoming)-1)))
        for record in aa['records']:
            assert record['reset'] and record['before']==record['after'] and record['writes']==([0,0] if record['history']=='none' else [2,2])
            for k,tt in enumerate(record['traces']):
                assert len(tt)==5
                for t in tt:traces.append(dict(dataset=d,method=record['method'],history=record['history'],layer=LAYERS[k],**t))
    for name,rows in (('training_screen',screen),('layer_metrics',summary),('history_contrasts',contrasts),('readouts',svms),('per_target',per),('update_traces',traces),('fc2_local',local)):
        exp.write_json(DOC/(name+'.json'),rows);csvfile(name,rows)
    for name in ('selection','calibration','independent_acceptance'):exp.write_json(DOC/(name+'.json'),exp.read(OUT/(name+'.json')))
    audit=dict(passed=True,source_sha256=exp.sha(__file__),protocol_sha256=exp.sha(OUT/'protocol.json'),selection_sha256=exp.sha(OUT/'selection.json'),
        no_original_model_edits=True,SAE_gain_only=True,training_only_calibration_and_selection=True,clean_only=True,
        certified_joint_goal=selection['certified'],independent_fp64=independent,score_replay_max_error=score_error,
        datasets=[{k:v for k,v in a.items() if k!='records'} for a in audits],new_unit_tests=5)
    exp.write_json(DOC/'audit.json',audit)
    cfg=selection['config']
    def getread(d,m,h,l,kind):return next(r for r in svms if (r['dataset'],r['method'],r['history'],r['layer'],r['kind'])==(d,m,h,l,kind))
    lines=['# 门控回投＋独立群体阈值：六组训练筛查与冻结clean复核','',
        '2026-11-06。本轮仅比较ε∈{0,.01,.03}与λ∈{.01,.03}共6组，两层共享同一组ε、λ，T固定5。SAE沿用训练折1–4拟合的2112参数checkpoint；只计算增益。原FC1→原前馈桥接→FC2和所有固定/共享重拟合SVM均使用实际4096维FC响应，没有使用重建或latent代替前馈或readout。原模型与既有报告均未修改。','',
        '## 实现与状态','',
        'z=E(r)，h=Wz，q_g=PᵀAᵀ[(z>0)⊙h]，κ=zᵀh/(‖z‖²+10⁻¹²)。u=tanh(q_g/c)，ū_x=Σxu/Σx，m=ε(u−ū_x)，τ=λ[μ(r)+σ_population(r)]κ/(κ+κ₀)，r_next=ReLU[x(1+m)−τ]。x为该层实际输入，r₀=x；FC2输入来自干预后FC1的真实原前馈桥接。μ、σ在每张图的4096个单元上计算，没有排序或活动数量配额。','',
        '每个逻辑trial创建全零W；两个实际preview依次写入，W=.25 offdiag(u₁u₁ᵀ)+.5 offdiag(u₂u₂ᵀ)。target阶段只读，没有学习、衰减或SAE更新。向量化只并行不同trial，不共享它们的W。无preview输出严格等于vanilla。原零单元不被局部点亮；每步局部增益范围为[1−2ε,1+2ε]。五步更新按有限预算评价，不要求不动点收敛。','',
        '5项新单元测试通过；真实4096维、独立稠密投影FP64公式复核最大误差'+f"{max(r['fp64_max_error'] for r in independent['rows']):.3g}"+'。全部pilot trial的preview写入与旧Memory独立复核、每步W不变、target写入拒绝、倒序/单trial重放、原桥接、均值匹配和SAE参数不变检查通过。','',
        '## 训练折定标与选点','',
        'ImageNet256、THINGS320个训练target及各自preview用于定标和筛查。c为训练Exact/Different在vanilla查询上、原非零单元q_g的合并RMS；κ₀为对应严格正κ的均值，之后固定。既有clean小批量的16/18个target均不参与本轮定标或选点，且概念与训练折隔离。SAE和固定SVM已拟合过这些训练折，因此这是训练侧筛查，不是内层交叉验证。','',
        '|数据／层|c|κ₀|','|---|---:|---:|']
    for d in ('imagenet','things'):
        for l,c in zip(LAYERS,cal[d]):lines.append(f"|{d} {l}|{c['q_rms']:.6f}|{c['kappa0']:.6f}|")
    lines+=['',
        '预先定义的活动/功能约束：两个数据集×两层×Exact/Different共8个单元的活动下降均为1%–15%、固定SVM BA损失≤1个百分点、平均真实类别margin至少保留vanilla的95%。满足后按最差Fisher变化、平均Fisher变化、最差margin保留率依次选点；全部Fisher不降才可声明联合目标通过。这里1%的下限是本轮工程筛选口径，不是生物阈值或sharpening定义。','',
        '|ε|λ|活动下降范围|最差ΔFisher|平均ΔFisher|最差margin保留|活动/功能约束通过|',
        '|---:|---:|---:|---:|---:|---:|:---:|']
    for c,r in zip(GRID,selection['ranking']):
        rr=[s for s in screen if s['name']==c['name']]
        lines.append(f"|{c['epsilon']:g}|{c['lambda']:g}|{min(x['suppression'] for x in rr):.3f}%–{max(x['suppression'] for x in rr):.3f}%|{r['min_fisher']:+.3f}%|{r['mean_fisher']:+.3f}%|{100*r['min_margin_retention']:.2f}%|{'是' if r['eligible'] else '否'}|")
    lines+=['',
        f"冻结ε={cfg['epsilon']}、λ={cfg['lambda']}，联合目标验收={'通过' if selection['certified'] else '未通过'}。该点是满足活动/功能约束的唯一点，而不是Fisher最佳点。尤其ε=.01、λ=.03的一项训练活动下降约0.992%，仅比1%下限少约0.008个百分点；若将下限改为0.9%，仅抑制ε=0、λ=.03会有更小的Fisher损伤。此边界敏感性应明确保留，不能把冻结点解释成增益机制优胜。没有根据后续pilot结果改选参数。",'',
        '## 冻结clean小批量：FC1与FC2分别统计','',
        'suppression按同target比例计算后按类别/概念等权，正值表示活动下降。Fisher使用逐图L2归一化后的类间/类内散度比；变化相对无preview。非零阈值沿用1e−7，偏好组由原训练折定义。', '',
        '|数据／层|历史|对无preview suppression|Exact−Different suppression|ΔFisher|Δ类内散度|Δ偏好份额|Δ非零比例|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in contrasts:
        if r['name']!='full':continue
        lines.append(f"|{r['dataset']} {r['layer']}|{r['history']}|{r['suppression']:.3f}%|{r['exact_different_suppression']:.3f}%|{r['fisher_change']:+.3f}%|{r['within_change']:+.3f}%|{r['preferred_pp']:+.3f}pp|{r['nonzero_pp']:+.3f}pp|")
    lines+=['','## 固定与重拟合SVM，包括margin','',
        '重拟合使用同一StandardScaler＋LinearSVC(C=1)，仅训练折上的实际原FC表示；Exact、Different、无preview三历史各1/3权重、总样本质量不增加，使用一个共享头比较历史。没有分别为Exact与Different挑分类器。margin是真实大类分数减最大其他类分数，按类别/概念等权。', '',
        '|数据／层|历史|固定BA|共享BA|固定margin|固定Δmargin对无preview|共享margin|共享Δmargin对无preview|固定AUC|共享AUC|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for l in LAYERS:
            for h in HIST:
                f=getread(d,'full',h,l,'fixed');s=getread(d,'full',h,l,'shared')
                lines.append(f"|{d} {l}|{h}|{100*f['ba']:.2f}%|{100*s['ba']:.2f}%|{f['margin']:.5f}|{f['margin_change_vs_none']:+.5f}|{s['margin']:.5f}|{s['margin_change_vs_none']:+.5f}|{f['auc']:.5f}|{s['auc']:.5f}|")
    lines+=['',
        '固定头的分数尺度跨历史相同；共享头内也相同。但固定与重拟合头之间没有做分数校准，不能仅凭重拟合margin绝对值更大就断言信息更多。各方法的无preview响应均等于vanilla，共享头的无preview分数仍可能因训练时混合了其他历史而与原固定头不同。', '',
        '## 同池仅抑制、同均值缩放、权重打乱','',
        'pool_only保留相同κ驱动群体阈值规则，仅设ε=0。第一步阈值相同，后续查询轨迹不同，阈值可以不同；不能当作逐步固定阈值的因果消融。scale_both用逐图正标量匹配完整方法两层最终均值，保留真实原桥接；FC1几何按定义保持，FC2可能因原桥接偏置而略变。shuffled使用每trial固定latent排列，Exact/Different成对共享排列，只打乱W的单元对应。', '',
        '|数据／层|方法|Exact suppression|Exact−Different suppression|ΔFisher|固定BA|共享BA|固定Δmargin|共享Δmargin|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in contrasts:
        if r['history']!='exact':continue
        d,m,l=r['dataset'],r['name'],r['layer'];f=getread(d,m,'exact',l,'fixed');s=getread(d,m,'exact',l,'shared')
        lines.append(f"|{d} {l}|{m}|{r['suppression']:.3f}%|{r['exact_different_suppression']:.3f}%|{r['fisher_change']:+.3f}%|{100*f['ba']:.2f}%|{100*s['ba']:.2f}%|{f['margin_change_vs_none']:+.5f}|{s['margin_change_vs_none']:+.5f}|")
    lines+=['','## 更新过程与FC2局部效应','',
        '训练定标保证不同输入不会被逐图归一化到相同增益强度。局部调制截零前总活动变化应近零；抑制池不由假想正增强计算，但κ和群体统计仍随r变化，因此闭环中增强与抑制仍存在间接耦合。', '',
        '|数据／层|方法|Exact第一步调制RMS|第1→5步κ|第1→5步阈值|', '|---|---|---:|---:|---:|']
    for d in ('imagenet','things'):
        for l in LAYERS:
            for m in ('full','pool_only'):
                ts=[r for r in traces if (r['dataset'],r['layer'],r['method'],r['history'])==(d,l,m,'exact')]
                lines.append(f"|{d} {l}|{m}|{100*ts[0]['modulation_rms']:.3f}%|{ts[0]['kappa']:.5f}→{ts[-1]['kappa']:.5f}|{ts[0]['threshold']:.6f}→{ts[-1]['threshold']:.6f}|")
    lines+=['', '|数据|历史|FC2局部suppression对实际输入|FC2局部ΔFisher|', '|---|---|---:|---:|']
    for r in local:
        if r['method']=='full':lines.append(f"|{r['dataset']}|{r['history']}|{r['local_suppression']:.3f}%|{r['local_fisher_change']:+.3f}%|")
    lines+=['',
        '## 验收结论与范围','',
        '本轮仅有6个训练侧候选及1个冻结设置的既有clean开发小批量复核，没有参数范围扩展，没有使用小批量重新选点。这34个target不是全新独立终测；单次权重置换也不是多次置换显著性检验。所有绝对指标、不同历史、原FC固定/共享SVM分数与逐target margin保存在CSV/JSON及原始npz中。', '',
        '联合sharpening收益需要同时看到低活动与更有区分度的原FC表征，并对比仅抑制和同均值缩放；BA不变、Exact匹配更强或非零单元减少均不能单独替代该验收。']
    exact=[r for r in contrasts if r['name']=='full' and r['history']=='exact']
    worse_pool=sum(r['fisher_change']<next(a['fisher_change'] for a in contrasts if (a['dataset'],a['layer'],a['name'],a['history'])==(r['dataset'],r['layer'],'pool_only','exact')) for r in exact)
    worse_scale=sum(r['fisher_change']<next(a['fisher_change'] for a in contrasts if (a['dataset'],a['layer'],a['name'],a['history'])==(r['dataset'],r['layer'],'scale_both','exact')) for r in exact)
    lines+=['','## 结果解读','',
        f"冻结完整方法的Exact活动下降范围为{min(r['suppression'] for r in exact):.3f}%–{max(r['suppression'] for r in exact):.3f}%，Fisher变化范围为{min(r['fisher_change'] for r in exact):+.3f}%–{max(r['fisher_change'] for r in exact):+.3f}%。它相对同池仅抑制在{worse_pool}/4个数据/层单元的Fisher更低，相对同均值缩放在{worse_scale}/4个单元更低；这些是本小批量的观察值，不作统计显著性结论。",'',
        '偏好份额提高、非零/弱尾减少描述的是单个响应向量更集中；Fisher描述跨图像的大类结构。应分别报告，不能因前者变好就覆盖后者的变化。局部增益沿当前记忆匹配方向，并未通过类别监督变成类别最优方向；增益的零均值约束也只在截零前成立，且后续查询κ会改变池强度。', '',
        'margin需要结合对照看：有些FC2条件的完整方法可比仅抑制或缩放少损失一些margin，即存在局部读出收益；这仍不能替代跨层稳定的几何收益。固定/共享BA保持也不代表决策分数恒等。', '',
        '本轮结论是完成了状态和功能验收、验证了可控的低活动与响应集中，但联合sharpening目标尚未通过；没有把冻结点提升为原模型的新默认机制。']
    (DOC/'六组比较与冻结复核报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4.2));colors={0.:'#277da8',.01:'#f8961e',.03:'#c44e52'}
    for c in GRID:
        rr=[r for r in screen if r['name']==c['name']]
        ax[0].scatter(np.mean([r['suppression'] for r in rr]),np.mean([r['fisher_change'] for r in rr]),s=75,
            marker='o' if c['lambda']==.01 else 's',color=colors[c['epsilon']])
    for e,color in colors.items():ax[0].scatter([],[],color=color,label=f'epsilon={e:g}')
    ax[0].axhline(0,color='grey',lw=.7);ax[0].set(xlabel='Mean training suppression (%)',ylabel='Mean Fisher change (%)',title='Six-point training screen');ax[0].legend(fontsize=8)
    xx=np.arange(4);labels=[d+' '+l for d in ('imagenet','things') for l in LAYERS]
    for k,m in enumerate(METHODS):
        rr=[next(r for r in contrasts if (r['dataset'],r['layer'],r['name'],r['history'])==(d,l,m,'exact')) for d in ('imagenet','things') for l in LAYERS]
        ax[1].bar(xx+(k-1.5)*.18,[r['fisher_change'] for r in rr],width=.18,label=m)
    ax[1].axhline(0,color='grey',lw=.7);ax[1].set_xticks(xx,labels,rotation=15);ax[1].set(ylabel='Fisher change vs no preview (%)',title='Frozen clean Exact: layer-separated controls')
    ax[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.23),ncol=4)
    fig.tight_layout(rect=(0,.10,1,1))
    for ext in ('png','svg','pdf'):fig.savefig(DOC/('six_point_comparison.'+ext),dpi=180)
    plt.close(fig);exp.validate(old)
    files=[f for f in DOC.iterdir() if f.is_file() and f.name!='complete.json']
    exp.write_json(DOC/'complete.json',dict(passed=True,protocol_sha256=exp.sha(OUT/'protocol.json'),selection_sha256=exp.sha(OUT/'selection.json'),source_sha256=exp.sha(__file__),files={f.name:exp.sha(f) for f in files}))
    print('REPORT COMPLETE',DOC,flush=True)

if __name__=='__main__':finish()
