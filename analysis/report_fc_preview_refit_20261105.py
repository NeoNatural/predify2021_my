"""Readout refit audit and existing-response mechanism diagnostics; no new network rule."""
import csv
import pickle
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis import fc_preview_pilot_20261105 as exp
from analysis import fc_preview_refit_20261105 as ref
from analysis.report_fc_preview_pilot_20261105 import LABELS,STAGES,HISTORY

DOC=exp.DOC/'refit'

def csvsave(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run():
    old,p=ref.protocol();DOC.mkdir(parents=True,exist_ok=True)
    summary=[];individual=[];diagnostics=[];audit=dict(passed=True,datasets={},fixed_score_max_error=0.,
        model_scaler_mean_max_error=0.,no_test_fit=True,no_parameter_search=True,original_sources_unchanged=True,
        source_sha256=exp.sha(__file__),refit_protocol_sha256=exp.sha(ref.OUT/'protocol.json'))
    for d in ('imagenet','things'):
        for folder in ('inputs','representations','readouts'):ref.verify(ref.OUT/d/folder)
        aa=exp.read(ref.OUT/d/'inputs/audit.json');ra=exp.read(ref.OUT/d/'representations/audit.json');fa=exp.read(ref.OUT/d/'readouts/audit.json')
        assert aa['passed'] and aa['backbone_before']==aa['backbone_after'] and ra['passed'] and fa['passed']
        summary.extend(exp.read(ref.OUT/d/'readouts/metrics.json'))
        with np.load(ref.OUT/d/'representations/features.npz') as z:
            train={k:z[k] for k in z.files if '__' in k};y,groups=z['y'],z['concept']
        with (ref.OUT/d/'readouts/models.pckl').open('rb') as f:bank=pickle.load(f)
        checked=set()
        for method in ref.METHODS:
            for stage in exp.STAGES:
                for h in ('shared',*exp.HISTORIES):
                    if h=='shared':
                        key=f'shared__{method}__{stage}';x,yy,w=ref.pool_histories([train[f'{method}__{hh}__{stage}'] for hh in exp.HISTORIES],y,groups)
                    else:key=f'separate__{method}__{h}__{stage}';x,yy,w=train[f'{method}__{h}__{stage}'],y,exp.weights(y,groups)
                    sig=bank['mapping'][key];assert sig==ref.signature(x,yy,w)
                    if sig in checked:continue
                    checked.add(sig);head=bank['heads'][sig];assert head[-1].C==1 and head[-1].n_iter_<30000
                    error=float(np.abs(head[0].mean_-np.average(x.astype(np.float64),weights=w,axis=0)).max())
                    assert error<1e-9;assert abs(w.sum()-len(y))<1e-7
                    audit['model_scaler_mean_max_error']=max(audit['model_scaler_mean_max_error'],error)
        zero=exp.PreviewMemory(4096).fingerprint()
        for r in ra['records']:
            assert r['reset']==[zero,zero] and r['before_target']==r['after_target']
            assert r['writes']==([0,0] if r['history']=='none' or r['method'] in ('vanilla','inhibition') else [2,2])
        with np.load(exp.OUT/d/'pilot/representations.npz') as z:
            test={k:z[k] for k in z.files if '__' in k};ty,tgroups,trows=z['y'],z['concept'],z['row']
        ref.add_inhib_scale(test);weights=exp.weights(ty,tgroups)
        with np.load(ref.OUT/d/'readouts/test_scores.npz') as z:scores={k:z[k] for k in z.files if '__' in k}
        with np.load(exp.OUT/d/'pilot/scores.npz') as z:
            for k in z.files:
                if '__' not in k:continue
                error=float(np.abs(z[k]-scores['fixed__'+k]).max());assert error==0
                audit['fixed_score_max_error']=max(audit['fixed_score_max_error'],error)
        with threadpool_limits(limits=2):
            for method in ref.METHODS:
                for h in exp.HISTORIES:
                    for stage in exp.STAGES:
                        key=f'{method}__{h}__{stage}';a=test[key]
                        for kind in ('fixed','shared','separate','scaler_only'):
                            s=scores[kind+'__'+key];pred=s.argmax(1)
                            margin=s[np.arange(len(ty)),ty]-np.where(np.arange(s.shape[1])[None,:]==ty[:,None],-np.inf,s).max(1)
                            for j,target in enumerate(trows):
                                individual.append(dict(dataset=d,target=int(target),concept=str(tgroups[j]),category=int(ty[j]),method=method,history=h,
                                    stage=stage,readout=kind,prediction=int(pred[j]),correct=int(pred[j]==ty[j]),margin=float(margin[j])))
                        if stage=='fc2_input':continue
                        b=test['vanilla__exact__fc1'] if stage=='fc1' else test[f'{method}__{h}__fc2_input']
                        active=a>1e-7;revived=active&(b<=1e-7);removed=(~active)&(b>1e-7)
                        avg=lambda v:float(np.average(v,weights=weights))
                        u=a.astype(np.float64)/np.maximum(np.linalg.norm(a.astype(np.float64),axis=1,keepdims=True),1e-8)
                        center=np.average(u,axis=0,weights=weights)
                        diagnostics.append(dict(dataset=d,method=method,history=h,stage=stage,
                            nonzero=avg(active.mean(1)),input_nonzero=avg((b>1e-7).mean(1)),revived_unit_fraction=avg(revived.mean(1)),
                            revived_activity_share=avg((a*revived).sum(1)/np.maximum(a.sum(1),1e-8)),removed_unit_fraction=avg(removed.mean(1)),
                            common_direction_energy=float(center@center),
                            cosine_to_incoming=avg((a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-8))))
        audit['datasets'][d]=dict(train=fa['train_count'],test=fa['test_count'],models_verified=len(checked),
            training_condition_trials=ra['condition_trials'],bridge_cache_max_error=ra['bridge_raw_cache_max_error'],
            model_aliases=fa['model_aliases'],all_training_targets_and_previews_disjoint_from_test=True)
        del train,test,scores,bank
    exp.validate(old);ref.protocol()
    csvsave(DOC/'readout_metrics.csv',summary);exp.write_json(DOC/'readout_metrics.json',summary)
    csvsave(DOC/'per_target_readout.csv',individual);csvsave(DOC/'support_diagnostics.csv',diagnostics)
    exp.write_json(DOC/'audit.json',audit)
    def get(d,m,s,h,k):return next(x for x in summary if (x['dataset'],x['method'],x['stage'],x['history'],x['readout'])==(d,m,s,h,k))
    lines=['# 临时Hebbian：SVM重拟合与机制诊断','',
        '本补充只重新拟合读出并分析已经保存的默认clean响应。没有改变网络响应规则、原始模型、preview写入方式、默认参数或34个测试target；没有扩大参数搜索。','',
        'ImageNet使用原正式训练折fold≠0的256个target，THINGS使用320个；各自preview也必须属于相同训练折。已核对目标/preview概念及原图哈希均不涉及formal fold0或开发测试队列，并核对旧刺激缓存等于当前重新渲染。只复用原vanilla训练响应；所有适配训练响应均由本trial preview建立临时权重、target只读的原默认串联重新生成。','',
        '固定头保留此前vanilla训练的StandardScaler＋LinearSVC。新头仍使用相同模型与C=1，类别/概念加权，不做超参数搜索。共享适配头：同一方法、同一层的三历史训练表示一起拟合，三份历史各权重1/3，总训练权重不变；三种测试历史共用一套头。分条件头：Exact、Different、无preview各自拟合，仅检查条件内可线性读出的信息；它们的边界不同，不能用其差值证明priming。另列“仅校准”：替换为共享适配训练分布的逐单元均值/标准差，保留原SVM系数及截距，不重新拟合SVM。','',
        'BA为类别平衡准确率；数据集保持4类／5类。FC2输入指经FC1处理、真实前馈桥接之后且尚未做FC2侧向更新的激活。FC2局部增量需同时查看这一行。','',
        '## 主方案与仅抑制：BA（%）','',
        '|数据|层/阶段|方法|历史|固定vanilla头|共享适配头|分条件适配头|仅均值/方差校准|',
        '|---|---|---|---|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for stage in exp.STAGES:
            for m,hs in (('vanilla',('exact',)),('inhibition',('none',)),('hebbian',exp.HISTORIES)):
                for h in hs:
                    vals=[100*get(d,m,stage,h,k)['ba'] for k in ('fixed','shared','separate','scaler_only')]
                    lines.append(f"|{d}|{STAGES[stage]}|{LABELS[m]}|{HISTORY[h]}|"+'|'.join(f'{v:.2f}' for v in vals)+'|')
    lines+=['','## Exact下的全部对照：BA（%）与AUC','',
        '|数据|层|方法|固定BA|共享BA|分条件BA|固定AUC|共享AUC|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for stage in ('fc1','fc2'):
            for m in ref.METHODS:
                a,b,c=[get(d,m,stage,'exact',k) for k in ('fixed','shared','separate')]
                lines.append(f"|{d}|{STAGES[stage]}|{LABELS[m]}|{100*a['ba']:.2f}|{100*b['ba']:.2f}|{100*c['ba']:.2f}|{a['auc']:.5f}|{b['auc']:.5f}|")
    lines+=['','## 原本为零的单元被激活：主方案','',
        'FC1相对原始FC1输入，FC2相对该历史的实际FC2输入。以下均按类别/概念加权；此前主报告非零比例列使用逐图等权，因此THINGS数值略有差异。','',
        '|数据|层|历史|输出非零%|被激活的原零单元占全部单元%|它们占输出活动%|原非零单元被清零%|',
        '|---|---|---|---:|---:|---:|---:|']
    for r in diagnostics:
        if r['method']!='hebbian':continue
        lines.append(f"|{r['dataset']}|{STAGES[r['stage']]}|{HISTORY[r['history']]}|"+'|'.join(f"{100*r[k]:.3f}" for k in ('nonzero','revived_unit_fraction','revived_activity_share','removed_unit_fraction'))+'|')
    lines+=['','## 共同方向成分','',
        '定义C=‖Σᵢ ŵᵢ aᵢ/‖aᵢ‖₂‖²，ŵ为类别/概念等权后归一化的样本权重。C越高表示不同target的单位向量越朝同一方向集中；它是共同成分的描述，不证明该成分必定来自图像背景。纯逐图正缩放不会改变同一层的C。','',
        '|数据|层|vanilla|仅抑制/无preview|Hebb Exact|Hebb Different|打乱 Exact|',
        '|---|---|---:|---:|---:|---:|---:|']
    for d in ('imagenet','things'):
        for s in ('fc1','fc2'):
            vals=[next(r['common_direction_energy'] for r in diagnostics if (r['dataset'],r['stage'],r['method'],r['history'])==(d,s,m,h))
                for m,h in (('vanilla','exact'),('inhibition','none'),('hebbian','exact'),('hebbian','different'),('shuffled','exact'))]
            lines.append(f'|{d}|{STAGES[s]}|'+'|'.join(f'{v:.5f}' for v in vals)+'|')
    lines+=['','## 审计和解读边界','',
        f"两数据集重新核对了{sum(v['training_condition_trials'] for v in audit['datasets'].values())}个训练条件trial的清零、写入次数与target前后权重哈希。新头全部收敛；训练StandardScaler均值独立复算最大误差{audit['model_scaler_mean_max_error']:.3g}，旧固定头分数重放误差为0。原始模型及旧实验源文件哈希保持一致。",'',
        '各层共享头按该层处理后的训练表示拟合，FC2输入也单独有诊断头。不能直接把两个不同边界的margin绝对值作增益比较；同一共享头内的Exact/Different比较才处在共同决策坐标中。完整逐target预测、margin、BA、AUC及各类recall保存在相邻CSV/JSON。','',
        '小批量只有ImageNet16张/8概念、THINGS18张/9概念；一个target即可改变BA若干百分点。重拟合恢复支持读出坐标失配，不能证明所有原信息都保留；未恢复也不能证明信息永久丢失，它只说明在当前训练量与固定C的线性SVM下没有恢复。分条件头和共享头都是功能诊断，不替代suppression与表征几何指标。','',
        '机制分析与下一步公式见同目录《机制解释与下一步方案.md》。原默认响应、suppression、Fisher及偏好份额结论均不因重拟合而改变。','']
    (DOC/'SVM重拟合结果.md').write_text('\n'.join(lines),encoding='utf-8')
    exp.write_json(DOC/'complete.json',dict(refit_protocol_sha256=exp.sha(ref.OUT/'protocol.json'),
        files={n:exp.sha(DOC/n) for n in ('readout_metrics.csv','readout_metrics.json','per_target_readout.csv',
            'support_diagnostics.csv','audit.json','SVM重拟合结果.md','机制解释与下一步方案.md')}))
    print('REPORT',DOC,'AUDIT',audit,flush=True)

if __name__=='__main__':run()
