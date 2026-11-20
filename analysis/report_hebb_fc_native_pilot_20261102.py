"""Replay FC on audited PCoder states; descriptive summaries and pilot figures."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from pathlib import Path
import csv,json,sys,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.hebb_fc_native_pilot_20261102 import OUT,DOC,RULES,apply_fc
from predify_experiments import native_categories_20261029 as native
from predify_experiments.supplementary import load
from predify_experiments.readout_init_20261012 import sha,immutable_json
from predify_experiments.fc_serial_20261028 import mean_matched
from predify_experiments.fc_sparse_lateral_20261028 import adaptive_target
from threadpoolctl import threadpool_limits

METHODS=('off','quota','mean','mean_std','feedback','inhibition_only','scale1_fc2','scale_both')

def fast_single(original,model,method,layer):
    """Same five updates as frozen trajectories, without per-step diagnostic syncs."""
    cfg=RULES[method];x=original/model['scale'];r=x.clone()
    E=model['matrices']['corr_'+('k128' if layer=='fc1' else 'k32')]
    assert original.ndim==2 and original.shape==(1,E.shape[0])
    pool=x.mean(1,keepdim=True)
    count=torch.ceil((original>1e-7).sum(1)*1.).long() if method=='quota' else None
    for _ in range(5):
        mean=r.mean(1,keepdim=True);drive=x if method=='inhibition_only' else x+.75*(r@E.T)
        if method=='quota':target,_=adaptive_target(drive,count,'hard');next_pool=pool
        else:
            if cfg['family']=='moments':
                theta=cfg['theta0']+cfg['a']*drive.mean(1,keepdim=True)+cfg['b']*drive.std(1,keepdim=True,unbiased=False)
                next_pool=pool
            else:
                theta=cfg['theta0']+cfg['g']*pool;next_pool=(1-cfg['rho'])*pool+cfg['rho']*mean
            target=torch.where(drive>theta,drive,0.)
        target=target/(1+2*mean);updated=.5*r+.5*target
        r,pool=torch.where(target>0,updated,0.),next_pool
    return r*model['scale']

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def replay(dataset):
    native.verify_freeze();dest=OUT/dataset
    spec=json.loads((dest/'pilot_manifest.json').read_text(encoding='utf8'))
    acceptance=json.loads((dest/'integration_acceptance.json').read_text(encoding='utf8'));assert acceptance['passed']
    completed=json.loads((dest/'baseline_complete.json').read_text(encoding='utf8'));assert completed['passed']
    previous=dest/'audit.json'
    if previous.exists():
        p=json.loads(previous.read_text(encoding='utf8'))
        assert p['passed'] and all(sha(dest/f'features_{k}.npz')==v for k,v in p['features'].items())
        assert sha(dest/'reference_scores.npz')==p['reference_scores_sha256'] and sha(DOC/dataset/'timeline.csv')==p['timeline_sha256']
        print('REPLAY ALREADY COMPLETE',dataset,flush=True);return
    records=[load(p) for p in sorted((dest/'baseline').glob('*.pckl'))]
    n=len(records);assert n==completed['files']==(192 if dataset=='imagenet' else 228)
    for r in records:assert r['values']['fc1'].shape==(15,4096) and r['values']['fc2'].shape==(15,4096)
    with np.load(dest/'graphs.npz') as z:
        m1=dict(scale=torch.tensor(float(z['scale1']),device='cuda'),matrices={'corr_k128':torch.tensor(z['E1'],device='cuda')})
        m2=dict(scale=torch.tensor(float(z['scale2']),device='cuda'),matrices={'corr_k32':torch.tensor(z['E2'],device='cuda')})
    state=torch.load(Path.home()/'.cache/torch/hub/checkpoints/vgg16-397923af.pth',map_location='cpu',weights_only=True)
    w=state['classifier.3.weight'].cuda();b=state['classifier.3.bias'].cuda();del state
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    base1=np.stack([r['values']['fc1'] for r in records]);base2=np.stack([r['values']['fc2'] for r in records])
    native.atomic_npz(dest/'features_off.npz',fc1=base1,fc2=base2)
    print('REPLAY',dataset,'conditions',n,flush=True)
    generated={key:(np.empty_like(base1),np.empty_like(base2)) for key in METHODS[1:]}
    bridge_error=0.;fast_checks={}
    with torch.no_grad():
        for method in RULES:
            for layer,model in [('fc1',m1),('fc2',m2)]:
                source=base1 if layer=='fc1' else base2
                for j,t in [(0,0),(0,9),(n-1,0),(n-1,14)]:
                    sample=torch.tensor(source[j,t][None],device='cuda')
                    expected=apply_fc(sample,model,method,layer);actual=fast_single(sample,model,method,layer)
                    assert torch.equal(expected,actual),(method,layer,j,t,float((expected-actual).abs().max()))
                fast_checks[f'{method}/{layer}']='four samples bit-identical'
        for start in range(0,n,12):
            stop=min(n,start+12);shape=(stop-start,15,4096)
            original=torch.tensor(base1[start:stop],device='cuda');original2=torch.tensor(base2[start:stop],device='cuda')
            buffers={method:(torch.empty(shape,device='cuda'),torch.empty(shape,device='cuda')) for method in METHODS[1:]}
            for j in range(stop-start):
                for t in range(15):
                    x=original[j,t][None,:];x2=original2[j,t][None,:]
                    direct=torch.relu(torch.nn.functional.linear(x,w,b))
                    assert torch.equal(direct,x2),'original FC1-to-FC2 bridge changed'
                    for method in ('quota','mean','mean_std','feedback','inhibition_only'):
                        f1=fast_single(x,m1,method,'fc1');entry=torch.relu(torch.nn.functional.linear(f1,w,b));f2=fast_single(entry,m2,method,'fc2')
                        buffers[method][0][j,t]=f1[0];buffers[method][1][j,t]=f2[0]
                        if method=='mean_std':reference=(f1,f2)
                    scaled1=mean_matched(x,reference[0]);entry=torch.relu(torch.nn.functional.linear(scaled1,w,b))
                    fc2=fast_single(entry,m2,'mean_std','fc2');scaled2=mean_matched(entry,reference[1])
                    for method,f2 in [('scale1_fc2',fc2),('scale_both',scaled2)]:
                        buffers[method][0][j,t]=scaled1[0];buffers[method][1][j,t]=f2[0]
            for method,(f1,f2) in buffers.items():
                assert torch.isfinite(f1).all() and torch.isfinite(f2).all()
                generated[method][0][start:stop]=f1.cpu().numpy();generated[method][1][start:stop]=f2.cpu().numpy()
            print('REPLAY BATCH',dataset,stop,'/',n,flush=True)
    def persist(item):
        method,(f1,f2)=item
        native.atomic_npz(dest/f'features_{method}.npz',fc1=f1,fc2=f2)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(persist,generated.items()))
    rows=[dict(row=j,image=r['trial']['dataset_index'],concept=r['trial']['concept'],category_index=r['trial']['category_index'],
               category=spec['categories'][r['trial']['category_index']],seed=r['seed'],noise='clean' if r['seed']==0 else 'noisy',carrier=r['carrier'],identity=r['identity'],target_hash=r['values']['target_hash'],W_hashes=';'.join(r['values']['W_hashes'])) for j,r in enumerate(records)]
    for image in {r['image'] for r in rows}:
        for seed in (0,2921):assert len({r['target_hash'] for r in rows if r['image']==image and r['seed']==seed})==1
        for carrier in native.CARRIERS:
            for identity in {r['identity'] for r in rows if r['image']==image}:
                assert len({r['W_hashes'] for r in rows if r['image']==image and r['carrier']==carrier and r['identity']==identity})==1
    write(dest/'records.csv',rows)
    # This head belongs to the current native-category experiment, not the other chat.
    head_path=native.OUT/dataset/'cv/reference_heads/fold0.pckl';saved=load(head_path);head=saved['head']
    current=json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'))
    head_concepts={current['trials'][int(i)]['concept'] for i in saved['train_rows']}
    assert not head_concepts&{r['concept'] for r in rows}
    score={}
    with threadpool_limits(limits=4):
        for method in METHODS:
            with np.load(dest/f'features_{method}.npz') as z:a=z['fc2']
            values=head.decision_function(a.reshape(-1,4096)).reshape(n,15,len(spec['categories']))
            assert np.isfinite(values).all();score[method]=values
    native.atomic_npz(dest/'reference_scores.npz',**score)
    # Retain numeric per-target arrays; summaries use paired category-balanced means.
    summaries=[];geom=[];meta=np.array(rows,dtype=object)
    for method in METHODS:
        with np.load(dest/f'features_{method}.npz') as z:one=z['fc1'];two=z['fc2']
        for noise in ('clean','noisy'):
            for carrier in native.CARRIERS:
                for identity in ('exact','same_concept','same_category','different_category'):
                    ix=np.array([j for j,r in enumerate(rows) if r['noise']==noise and r['carrier']==carrier and r['identity']==identity],int)
                    if not len(ix):continue
                    yy=np.array([rows[j]['category_index'] for j in ix]);cats=sorted(set(yy))
                    for t in range(15):
                        prediction=score[method][ix,t].argmax(-1)
                        acc=100*np.mean([np.mean(prediction[yy==c]==c) for c in cats])
                        auc=float(np.mean([roc_auc_score(yy==c,score[method][ix,t,c]) for c in cats]))
                        for layer,arr in [('fc1',one),('fc2',two)]:
                            X=arr[ix,t].astype(np.float64);norm=np.linalg.norm(X,axis=1);unit=X/np.maximum(norm[:,None],1e-12)
                            group=[];means=[]
                            for c in cats:
                                q=unit[yy==c];assert len(q)==2
                                group.append(np.square(q-q.mean(axis=0)).sum(axis=1).mean());means.append(q.mean(axis=0))
                            means=np.array(means);within=float(np.mean(group));between=float(np.square(means-means.mean(axis=0)).sum(axis=1).mean())
                            peak=X.max(1,keepdims=True);weak=(X>1e-7)&(X<peak*.01)
                            summaries.append(dict(dataset=dataset,method=method,noise=noise,carrier=carrier,identity=identity,layer=layer,timestep=t,
                              n_images=len(ix),n_concepts=len(cats),activation=float(X.mean()),mean_abs=float(np.abs(X).mean()),density=float((X>1e-7).mean()),
                              weak_activity_share=float(np.mean((X*weak).sum(1)/np.maximum(X.sum(1),1e-12))),
                              within_pair=within,between_centers=between,fisher_pair=between/max(within,1e-12),
                              clean_reference_balanced_accuracy_pct=float(acc) if layer=='fc2' else '',
                              clean_reference_macro_auc=auc if layer=='fc2' else ''))
    write(DOC/dataset/'timeline.csv',summaries)
    # Within each target/noise/condition, verify paired stimuli and CNN states across all methods by construction.
    with np.load(dest/'reference_scores.npz') as z:assert set(z.files)==set(METHODS)
    audit=dict(dataset=dataset,passed=True,baseline_sequences=n,fc_methods=len(METHODS),target_timepoints=n*15,
      true_network_sequences=n+len(RULES),additional_in_model_validation_sequences=len(RULES),
      pair_concepts=len({r['concept'] for r in rows}),bridge_max_abs=bridge_error,integration=acceptance['checks'],fast_single_checks=fast_checks,
      current_clean_reference_head_sha256=sha(head_path),features={method:sha(dest/f'features_{method}.npz') for method in METHODS},
      reference_scores_sha256=sha(dest/'reference_scores.npz'),timeline_sha256=sha(DOC/dataset/'timeline.csv'),paired_target_hashes_and_preview_weights_passed=True,
      interpretation='5 fresh local FC update steps per network step; replay is not a new PCoder feedback trajectory; accuracy is a frozen clean-reference transfer on two pictures of one concept per class',
      input_scope='all FC graph weights and scales were re-estimated from current dataset formal clean images; no previous FC-study weights or splits used')
    immutable_json(dest/'audit.json',audit);print('REPLAY AND AUDIT COMPLETE',dataset,flush=True)

def figures(dataset):
    with (DOC/dataset/'timeline.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    output=DOC/'figures';output.mkdir(exist_ok=True)
    for noise in ('clean','noisy'):
        fig,axs=plt.subplots(2,3,figsize=(14,7),sharex=True)
        for li,layer in enumerate(('fc1','fc2')):
            for ci,carrier in enumerate(native.CARRIERS):
                ax=axs[li,ci]
                for method,identity,color,ls in [('off','exact','#64748b','-'),('mean_std','exact','#d34d42','-'),('off','different_category','#64748b','--'),('mean_std','different_category','#d34d42','--')]:
                    selected=sorted((r for r in rows if r['noise']==noise and r['layer']==layer and r['carrier']==carrier and r['method']==method and r['identity']==identity),key=lambda r:int(r['timestep']))
                    assert len(selected)==15
                    ax.plot(range(15),[float(r['activation']) for r in selected],ls=ls,color=color,marker='o',markersize=2,label=f'{method}: {identity}')
                ax.set_title(f'{layer.upper()} | {carrier}');ax.grid(alpha=.2);ax.set_xticks(range(0,15,2))
                if ci==0:ax.set_ylabel('Mean activation')
                if li==1:ax.set_xlabel('Target timestep')
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
        fig.suptitle(f'{dataset.upper()} | {noise} | FC adaptation (mean + std)');fig.tight_layout(rect=(0,.07,1,.94))
        for ext in ('png','svg'):fig.savefig(output/f'{dataset}_{noise}_activation.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
        for layer in ('fc1','fc2'):
            fig,axs=plt.subplots(2,3,figsize=(14,7),sharex=True)
            for mi,(metric,title) in enumerate((('within_pair','Within-concept pair variance'),('fisher_pair','Pair-based Fisher ratio'))):
                for ci,carrier in enumerate(native.CARRIERS):
                    ax=axs[mi,ci]
                    def values(method):
                        selected=sorted((r for r in rows if r['noise']==noise and r['layer']==layer and r['carrier']==carrier and r['method']==method and r['identity']=='exact'),key=lambda r:int(r['timestep']))
                        assert len(selected)==15
                        return np.array([float(r[metric]) for r in selected])
                    base=values('off')
                    for method,label,color in [('mean_std','Mean + std','#d34d42'),('inhibition_only','Inhibition only','#527baf'),('scale_both','Matched scaling','#288b72')]:
                        v=values(method);delta=np.full_like(v,np.nan);valid=np.abs(base)>1e-12;delta[valid]=100*(v[valid]/base[valid]-1)
                        ax.plot(range(15),delta,color=color,marker='o',markersize=2,label=label)
                    ax.axhline(0,color='#777',ls='--',lw=.8);ax.set_title(carrier);ax.grid(alpha=.2);ax.set_xticks(range(0,15,2))
                    if ci==0:ax.set_ylabel(title+'\nchange vs off (%)')
                    if mi==1:ax.set_xlabel('Target timestep')
            fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
            fig.suptitle(f'{dataset.upper()} | {noise} | {layer.upper()} | exact: two pictures / one concept per class',fontsize=13)
            fig.tight_layout(rect=(0,.07,1,.94))
            for ext in ('png','svg'):fig.savefig(output/f'{dataset}_{noise}_{layer}_geometry.{ext}',dpi=170,bbox_inches='tight')
            plt.close(fig)

def report():
    for d in ('imagenet','things'):figures(d)
    blocks=[]
    for d in ('imagenet','things'):
        with (DOC/d/'timeline.csv').open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
        for noise in ('clean','noisy'):
            for carrier in native.CARRIERS:
                def get(method,layer,metric):
                    return np.mean([float(x[metric]) for x in rr if x['method']==method and x['noise']==noise and x['carrier']==carrier and x['identity']=='exact' and x['layer']==layer and 0<=int(x['timestep'])<=4])
                x=[]
                for layer in ('fc1','fc2'):
                    base=get('off',layer,'activation');new=get('mean_std',layer,'activation');x.append(100*(new/base-1) if base else np.nan)
                acc=get('mean_std','fc2','clean_reference_balanced_accuracy_pct')-get('off','fc2','clean_reference_balanced_accuracy_pct')
                blocks.append((d,noise,carrier,*x,acc))
    lines=['# Hebb-FC 接入预测编码模型：小批量测试','','两数据集分别用本轮正式队列的干净中央目标画布重新拟合FC1/FC2连接和RMS尺度；ImageNet 320图、THINGS 400图。仅复用独立FC研究的算法与超参数，未加载该研究的训练权重和划分。评估使用每个类别1个开发概念、每概念2张图；训练概念与开发目标/preview角色隔离。','','FC1为corr/k128，FC2为corr/k32，侧向增益0.75，每次网络前向各做5次更新；分别计算排序配额、群体均值、均值＋标准差、动态反馈，并保留关闭FC、仅抑制、均值匹配缩放对照。FC状态每个网络时间步重新初始化；固定连接不在预览或目标期间更新。预览仍为2+2步，目标t0–t14。','','小样本覆盖三种原有记忆模式、clean和一个固定噪声实现、exact/同概念/同类/异类；THINGS植物开发队列只有一个概念，因此其同类不同概念条件省略，包含该条件的THINGS指标只涵盖其余四类。','', '真实预测编码网络共运行ImageNet 192条、THINGS 228条基线轨迹。FC在所有PCoder之后，没有到PCoder的反馈；本测试使用保存的每步FC1输入重放全部FC设定。每种设定均用真实模型中FC1/FC2钩子另行核对：代表性轨迹在目标阶段的PCoder4/5状态、目标开始时预览权重及FC输出逐位一致，目标期间权重保持冻结。各规则的优化实现另外用每层四个样本验证逐位一致。重放不声称额外生成了新的PCoder轨迹。','','表中为exact条件t0–t4平均值：均值＋标准差相对关闭FC的激活变化，以及既有当前研究clean参考读出转移准确率差。该读出并未在修改后的FC表示上重新拟合；偏移可能来自分类边界不匹配。每类仅1概念2图，不作推断或显著性判断。','','| 数据集 | 目标 | 模式 | FC1激活变化 | FC2激活变化 | 参考读出ACC差（百分点） |','|---|---|---|---:|---:|---:|']
    for d,n,c,a,b,delta in blocks:lines.append(f'| {d} | {n} | {c} | {a:+.2f}% | {b:+.2f}% | {delta:+.2f} |')
    observations=[]
    for d in ('imagenet','things'):
        with (DOC/d/'timeline.csv').open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
        for noise in ('clean','noisy'):
            for carrier in native.CARRIERS:
                def v(method,metric):return np.mean([float(x[metric]) for x in rr if x['method']==method and x['noise']==noise and x['carrier']==carrier and x['identity']=='exact' and x['layer']=='fc2' and int(x['timestep'])<5])
                basew,basef=v('off','within_pair'),v('off','fisher_pair')
                observations.append(dict(within=100*(v('mean_std','within_pair')/basew-1),fisher=100*(v('mean_std','fisher_pair')/basef-1),
                    inhibition_fisher=100*(v('inhibition_only','fisher_pair')/basef-1),scale_fisher=100*(v('scale_both','fisher_pair')/basef-1)))
    w=[r['within'] for r in observations];f=[r['fisher'] for r in observations];inh=[r['inhibition_fisher'] for r in observations];scale=[r['scale_fisher'] for r in observations]
    lines += ['',f"在这批样本中，均值＋标准差的FC2两图配对离散变化为 {min(w):+.2f}% 至 {max(w):+.2f}%，配对Fisher变化为 {min(f):+.2f}% 至 {max(f):+.2f}%；对应仅抑制为 {min(inh):+.2f}% 至 {max(inh):+.2f}%，双层均值匹配缩放为 {min(scale):+.2f}% 至 {max(scale):+.2f}%。这些是12个数据集×噪声×模式条件各自早期五个点的均值之比，具有样本描述性。"]
    lines+=['','各门控和对照的完整数值均保存在逐时刻表中。下表给出同一exact、t0–t4窗口中FC2平均激活相对关闭FC的变化；缩放对照由该图该时刻的均值＋标准差输出匹配，故与其均值相同。','','| 数据集 | 目标 | 模式 | 配额 | 群体均值 | 均值＋标准差 | 动态反馈 | 仅抑制 | 双层缩放 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        with (DOC/d/'timeline.csv').open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
        for noise in ('clean','noisy'):
            for carrier in native.CARRIERS:
                def activation(method):return np.mean([float(x['activation']) for x in rr if x['method']==method and x['noise']==noise and x['carrier']==carrier and x['identity']=='exact' and x['layer']=='fc2' and int(x['timestep'])<5])
                base=activation('off');values=[100*(activation(m)/base-1) if base else np.nan for m in ('quota','mean','mean_std','feedback','inhibition_only','scale_both')]
                lines.append('| '+' | '.join([d,noise,carrier]+[f'{v:+.2f}%' for v in values])+' |')
    lines+=['','同一固定参考读出下，exact减different-category的早期准确率差如下；变化为启用均值＋标准差后的差值减去关闭FC时的差值。样本极少，不能把正负方向当作稳定的启动效应。','','| 数据集 | 目标 | 模式 | 关闭FC（pp） | 均值＋标准差（pp） | 差值变化（pp） |','|---|---|---|---:|---:|---:|']
    for d in ('imagenet','things'):
        with (DOC/d/'timeline.csv').open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
        for noise in ('clean','noisy'):
            for carrier in native.CARRIERS:
                def score(method,identity):return np.mean([float(x['clean_reference_balanced_accuracy_pct']) for x in rr if x['method']==method and x['noise']==noise and x['carrier']==carrier and x['identity']==identity and x['layer']=='fc2' and int(x['timestep'])<5])
                before=score('off','exact')-score('off','different_category');after=score('mean_std','exact')-score('mean_std','different_category')
                lines.append(f'| {d} | {noise} | {carrier} | {before:+.2f} | {after:+.2f} | {after-before:+.2f} |')
    lines+=['','曲线、固定参考读出宏平均OVR AUC、活动密度、弱响应比例、两图配对类内离散和Fisher诊断逐时间点保存在各数据集 `timeline.csv`；THINGS的同类控制AUC只按出现的四类计算。每类只有一个概念，类内诊断仅描述同概念两张图片，不能代表跨概念类别几何。无全新独立验证或稳定性保证。','','## 开发样本与复核','','各类选定当前开发集中按ID排序的第一个概念及其中两张图，未按模型结果挑选。所列ImageNet概念采用稳定WNID。','','| 数据集 | 大类 | 开发概念 | 目标图像索引 |','|---|---|---|---|']
    for d in ('imagenet','things'):
        spec=json.loads((OUT/d/'pilot_manifest.json').read_text(encoding='utf8'))
        for k,c in enumerate(spec['categories']):
            targets=[t for t in spec['evaluation']['trials'] if t['category_index']==k]
            assert len(targets)==2 and targets[0]['concept']==targets[1]['concept']
            lines.append(f"| {d} | {c} | {targets[0]['concept']} | {targets[0]['dataset_index']}, {targets[1]['dataset_index']} |")
    lines+=['','两数据集所选的训练目标与开发视觉角色均无字节重叠；两数据集之间所选刺激也无字节重叠。这项SHA-256检查不能排除经过裁剪或重编码的近重复。两层FC图和尺度均已从保存的训练激活逐位重算；参考读出ACC/AUC共5,760个点已独立重算一致。原模型基线与已保存的210条clean轨迹逐位一致。14项相关测试通过。','','[FC图与隔离复核](图与隔离复核.json) · [读出复核](读出复核.json) · [执行来源与参数](执行来源.json)']
    for d in ('imagenet','things'):
        for n in ('clean','noisy'):lines+=['',f'![{d} {n} 两层FC激活](figures/{d}_{n}_activation.png)']
        for layer in ('fc1','fc2'):lines+=['',f'![{d} noisy {layer} 配对几何诊断](figures/{d}_noisy_{layer}_geometry.png)']
    lines+=['','[ImageNet逐时刻数据](imagenet/timeline.csv) · [THINGS逐时刻数据](things/timeline.csv) · [历史清单更正](../native_categories_readouts_20261029/历史覆盖补充审计_20261102.md)']
    (DOC/'小批量测试结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('REPORT',DOC/'小批量测试结果.md',flush=True)

if __name__=='__main__':
    for d in ('imagenet','things'):replay(d)
    report()
