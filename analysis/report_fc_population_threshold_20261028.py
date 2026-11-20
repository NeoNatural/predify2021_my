"""Independent audit, fold-balanced tables and scientific plots for population thresholds."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import sys
import time
from collections import defaultdict
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from predify_experiments import fc_population_threshold_20261028 as exp
from analysis.report_fc_degraded_k128_20261028 import check_scores
from analysis.report_fc_gain075_serial_20261028 import FIELDS,LABELS

OUT,DOC=exp.OUT,exp.DOC
METHODS=('quota',)+exp.FAMILIES
NAMES=dict(quota='排序配额',moments='群体均值',moments_std='均值＋标准差',feedback='动态反馈')
MODES=('none','predify_noise','fractal_occlusion')
MODE_NAMES=dict(none='干净',predify_noise='高斯噪声 σ=0.5',fractal_occlusion='分形遮挡 0.3')
MAIN='both_hard_hard_5_5_frozen'
read=exp.read
WAIT=False


def await_complete(dest):
    if not WAIT:return
    start=time.monotonic()
    while True:
        try:read(dest/'complete.json');return
        except (FileNotFoundError,json.JSONDecodeError):
            assert time.monotonic()-start<1800,('Timed out awaiting completed experiment',str(dest))
            time.sleep(3)


def save_json(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def ratio(a,b):return 100*(a/b-1) if b>1e-12 else None


def independent_geometry(a,y,m):
    v=a.astype(np.float64);norm=np.linalg.norm(v,axis=1)
    u=v/np.maximum(norm[:,None],1e-8)
    centers=np.stack([u[y==j].mean(0) for j in range(5)])
    w=np.mean([np.square(u[y==j]-centers[j]).sum(1).mean() for j in range(5)])
    b=np.square(centers-centers.mean(0)).sum(1).mean()
    metrics=dict(activation=np.mean([v[y==j].mean() for j in range(5)]),within=w,between=b,fisher=b/max(w,1e-12))
    error=max(abs(float(value)-m[k]) for k,value in metrics.items())
    assert error<3e-6,(error,metrics,m)
    assert abs(float((a>1e-7).mean())-m['nonzero'])<1e-7
    return error


def check_model(model):
    assert model[-1].C==1 and model[-1].max_iter==30000 and model[-1].random_state==20261028
    assert int(model[-1].n_iter_)<model[-1].max_iter


def independent_update(original,E,scale,c,steps=5):
    x=original/scale;r=x.clone();pool=x.mean(1,keepdim=True)
    for _ in range(steps):
        mean=r.mean(1,keepdim=True);drive=x+.75*torch.nn.functional.linear(r,E)
        if c['family']=='moments':theta=c['a']*drive.mean(1,keepdim=True)+c['b']*torch.sqrt(((drive-drive.mean(1,keepdim=True))**2).mean(1,keepdim=True))
        else:theta=c['g']*pool
        candidate=drive*(drive>theta)/(1+2*mean)
        nxt=(r+candidate)/2*(candidate>0)
        if c['family']=='feedback':pool=(1-c['rho'])*pool+c['rho']*mean
        r=nxt
    return r*scale


def aggregate(datasets=('imagenet','things'),only_fold=None):
    protocol=read(OUT/'protocol.json');original=exp.validate(protocol)
    assert exp.sha(OUT/'source_snapshot.py')==protocol['source_sha256']
    prior_audit=read(exp.old.DOC/'audit.json');assert prior_audit['passed']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    weights=torch.load(original['checkpoint'],map_location='cpu',weights_only=True)
    w,b=(weights[f'classifier.3.{k}'].to(exp.core.DEVICE) for k in ('weight','bias'));del weights
    rows,categories,dynamics,selections=[],[],[],[]
    audit=dict(passed=True,datasets=list(datasets),folds=0,new_phases=0,reused_quota_phases=0,new_representations=0,
        new_clean_svm_fits=0,reused_clean_svm=0,new_degraded_svm_fits=0,reused_degraded_svm=0,
        geometry_max_error=0.,bridge_max_error=0.,svm_replay_max_error=0.,trajectory_replay_max_error=0.,original_model_unchanged=True,
        clean_side_weights_exactly_reused=True,inner_and_outer_concepts_isolated=True,svm_selection=False,
        no_rank_or_quota_in_new_rules=True,no_fixed_point_filter=True,controls_mean_matched=True,
        repeats_averaged_within_fold=True,old_audit_sha256=exp.sha(exp.old.DOC/'audit.json'),phase_hashes={})
    for d in datasets:
        with np.load(exp.serial.DATA/d/'activations.npz') as z:
            vanilla1,vanilla2=z['fc1'],z['fc2'];y,concept,folds=z['category'],z['concept'],z['fold']
        with (exp.serial.DATA/d/'baseline_models.pckl').open('rb') as f:heads=pickle.load(f)['models']
        for fold in (range(5) if only_fold is None else [only_fold]):
            folder=OUT/d/f'fold{fold}';selection=read(folder/'selection.json')
            tr,te=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            assert selection['train_rows']==tr.tolist() and not set(concept[tr])&set(concept[te])
            for piece in selection['inner_folds']:
                it,iv=piece['train_rows'],piece['validation_rows']
                assert set(it)|set(iv)==set(tr) and not set(it)&set(iv)
                assert not set(concept[it])&set(concept[iv])
            side=exp.old.OUT/d/f'fold{fold}'/'clean_side_models.npz'
            assert exp.sha(side)==read(OUT/'calibration_protocol.json')['side_files'][f'{d}/{fold}']
            with np.load(side) as z:
                E1,E2=(torch.as_tensor(z[k],device=exp.core.DEVICE) for k in ('E1','E2'))
                s1,s2=float(z['scale1']),float(z['scale2'])
            for method in METHODS:
                dest=exp.old.OUT/d/f'fold{fold}' if method=='quota' else folder/method
                await_complete(dest)
                marker=exp.old.verify_marker(dest)
                if method!='quota':
                    assert marker['protocol_sha256']==exp.sha(OUT/'protocol.json') and marker['selection_sha256']==exp.sha(folder/'selection.json')
                    assert marker['side_sha256']==exp.sha(side)
                    config=read(dest/'clean/results.json')['config']
                    for layer in ('fc1','fc2'):
                        c=config[layer]
                        selections.append(dict(dataset=d,fold=fold,method=method,layer=layer,**{k:c.get(k) for k in ('a','b','g','rho','theta0')}))
                        if method!='moments_std':
                            assert c==selection['results'][method][layer]
                            chosen=exp.rules.rank(selection['results'][method][layer+'_grid'])
                            assert c==chosen['config']
                        else:assert c['b']>0
                clean_dest=dest/'clean';clean_data=read(clean_dest/'results.json')
                with (clean_dest/'svm_models.pckl').open('rb') as f:primary=pickle.load(f)
                phase_list=[('none',0.,-1,clean_dest)]+[(m,l,r,dest/m/f'repeat{r}') for m,l in exp.inputs.DEGRADATIONS for r in exp.inputs.REPEATS]
                for mode,level,repeat,p in phase_list:
                    exp.old.verify_marker(p);data=read(p/'results.json')
                    assert data['train_rows']==tr.tolist() and data['test_rows']==te.tolist()
                    if method=='quota':
                        assert exp.sha(p/'results.json')==prior_audit['source_hashes'][f'{d}/{fold}/{mode}/{repeat}']
                        audit['reused_quota_phases']+=1
                    else:
                        audit['new_phases']+=1
                        if mode=='none':audit['new_clean_svm_fits']+=17;audit['reused_clean_svm']+=2
                        else:audit['new_degraded_svm_fits']+=8;audit['reused_degraded_svm']+=2
                        audit['phase_hashes'][f'{d}/{fold}/{method}/{mode}/{repeat}']=exp.sha(p/'results.json')
                    base=data['results']['baseline']['metrics']
                    for name,entry in data['results'].items():
                        c=entry['condition'];ms=entry['metrics']
                        for stage,m in ms.items():
                            reference=ms['fc2_input'] if stage=='fc2_local' else base[stage]
                            cm=clean_data['results'][name]['metrics'][stage]
                            row=dict(dataset=d,method=method,mode=mode,level=level,repeat=repeat,fold=fold,name=name,stage=stage,t1=c['t1'],t2=c['t2'],m1=c['m1'],m2=c['m2'])
                            row.update({k:m[k] for k in FIELDS})
                            for k in ('activation','within','between','fisher'):
                                row[k+'_pct']=ratio(m[k],reference[k])
                                expected=m['effects'][k+'_pct']
                                assert row[k+'_pct'] is None or abs(row[k+'_pct']-expected)<1e-8
                            row.update(nonzero_pp=100*(m['nonzero']-reference['nonzero']),
                                svm_clean_accuracy=m['accuracy'],svm_clean_auc=m['auc'],svm_clean_accuracy_pp=100*(m['accuracy']-reference['accuracy']),
                                svm_clean_auc_delta=m['auc']-reference['auc'],svm_clean_drop_from_clean_pp=100*(m['accuracy']-cm['accuracy']),
                                svm_vanilla_accuracy=m['vanilla_svm']['accuracy'],svm_vanilla_accuracy_pp=100*(m['vanilla_svm']['accuracy']-reference['vanilla_svm']['accuracy']),
                                svm_refit_accuracy=m['degraded_svm']['accuracy'] if m['degraded_svm'] else None,
                                svm_refit_auc=m['degraded_svm']['auc'] if m['degraded_svm'] else None,
                                svm_refit_accuracy_pp=100*(m['degraded_svm']['accuracy']-reference['degraded_svm']['accuracy']) if m['degraded_svm'] and reference['degraded_svm'] else None,
                                cosine_to_clean_vanilla=m['cosine_to_clean_vanilla'],cosine_to_clean_same_method=m['cosine_to_clean_same_method'],
                                relative_l2_to_clean_vanilla=m['relative_l2_to_clean_vanilla'],original_imagenet_top1_pct=m['original_imagenet_top1_pct'],joint=m['joint'])
                            rows.append(row)
                            for j,cat in enumerate(('animals','clothing','food','plants','vehicles')):
                                categories.append(dict(dataset=d,method=method,mode=mode,repeat=repeat,fold=fold,name=name,stage=stage,category=cat,
                                    activation=m['per_class_activation'][j],activation_pct=ratio(m['per_class_activation'][j],reference['per_class_activation'][j]),
                                    within=m['per_class_within'][j],within_pct=ratio(m['per_class_within'][j],reference['per_class_within'][j]),
                                    svm_clean_recall=m['recalls'][j],svm_clean_recall_pp=100*(m['recalls'][j]-reference['recalls'][j]),
                                    svm_refit_recall=m['degraded_svm']['recalls'][j] if m['degraded_svm'] else None))
                    if method=='quota':continue
                    for key,diag in data['dynamics'].items():dynamics.append(dict(dataset=d,method=method,mode=mode,repeat=repeat,fold=fold,key=key,**diag))
                    with (p/'svm_models.pckl').open('rb') as f:models=pickle.load(f)
                    assert len(models)==(19 if mode=='none' else 10)
                    with np.load(p/'representations.npz') as features,np.load(p/'scores.npz') as scores:
                        assert np.array_equal(features['row'],te) and np.array_equal(features['y'],y[te])
                        checked=set()
                        for entry in data['results'].values():
                            keys=entry['feature_keys'];c=entry['condition']
                            for stage in ('fc1_output','fc2_input','fc2_output'):
                                key=keys[stage]
                                if key in checked:continue
                                checked.add(key);a=features[key];m=entry['metrics'][stage]
                                assert a.shape==(len(te),4096) and a.dtype==np.float32 and np.isfinite(a).all() and (a>=0).all()
                                audit['geometry_max_error']=max(audit['geometry_max_error'],independent_geometry(a,y[te],m))
                                for suffix,metric,model in [('clean',m,primary[key]),('vanilla',m['vanilla_svm'],heads['fc1' if key.startswith('fc1__') else 'fc2'][fold])]:
                                    decision=scores[key+'__'+suffix];check_scores(decision,y[te],metric);check_model(model)
                                    err=float(np.max(np.abs(model.decision_function(a)-decision)));assert err<1e-8
                                    audit['svm_replay_max_error']=max(audit['svm_replay_max_error'],err)
                                if mode!='none' and key in exp.auxiliary_keys():
                                    check_scores(scores[key+'__degraded'],y[te],m['degraded_svm']);check_model(models[key])
                                    err=float(np.max(np.abs(models[key].decision_function(a)-scores[key+'__degraded'])));assert err<1e-8
                                    audit['svm_replay_max_error']=max(audit['svm_replay_max_error'],err)
                                else:assert m['degraded_svm'] is None
                            if c['m1']=='gain_hard':
                                target=data['results'][MAIN]['feature_keys']['fc1_output']
                                assert np.allclose(features[keys['fc1_output']].mean(1),features[target].mean(1),atol=2e-6)
                            if c['m2']=='gain_hard':
                                target=data['results'][MAIN]['feature_keys']['fc2_output']
                                assert np.allclose(features[keys['fc2_output']].mean(1),features[target].mean(1),atol=2e-6)
                        assert len(checked)==19;audit['new_representations']+=19
                        old_phase=exp.old.OUT/d/f'fold{fold}'/('clean' if mode=='none' else f'{mode}/repeat{repeat}')
                        with np.load(old_phase/'representations.npz') as previous,np.load(old_phase/'scores.npz') as previous_scores:
                            for key in exp.BASE_KEYS:
                                assert np.array_equal(features[key],previous[key])
                                assert np.allclose(scores[key+'__clean'],previous_scores[key+'__clean'],atol=1e-12,rtol=0)
                                if mode!='none':assert np.allclose(scores[key+'__degraded'],previous_scores[key+'__degraded'],atol=1e-12,rtol=0)
                        # Independent formula replay on a small held-out batch;
                        # same-image computation cannot depend on other images.
                        sample=np.arange(min(8,len(te)))
                        with torch.inference_mode():
                            r1=independent_update(torch.as_tensor(features[exp.BASE_KEYS[0]][sample],device=exp.core.DEVICE),E1,s1,config['fc1'])
                            r2=independent_update(exp.serial.bridge(r1,w,b),E2,s2,config['fc2'])
                        for stage,v in (('fc1_output',r1),('fc2_output',r2)):
                            target=data['results'][MAIN]['feature_keys'][stage]
                            err=float(np.max(np.abs(v.cpu().numpy()-features[target][sample])))
                            assert err<1e-4,(d,fold,method,mode,err)
                            audit['trajectory_replay_max_error']=max(audit['trajectory_replay_max_error'],err)
                        for key in checked:
                            if not key.startswith('fc1__'):continue
                            incoming='fc2_input__'+key.removeprefix('fc1__')
                            with torch.inference_mode():prediction=exp.serial.bridge(torch.as_tensor(features[key],device=exp.core.DEVICE),w,b).cpu().numpy()
                            err=float(np.abs(prediction-features[incoming]).max());assert err<1e-4
                            audit['bridge_max_error']=max(audit['bridge_max_error'],err)
                        if mode=='none':source1,source2=vanilla1,vanilla2
                        else:
                            cache=exp.inputs.INPUTS/d/mode/f'repeat{repeat}'
                            assert data['input_complete_sha256']==exp.sha(cache/'complete.json')
                            with np.load(cache/'activations.npz') as z:source1,source2=z['fc1'],z['fc2']
                        assert np.array_equal(features[exp.BASE_KEYS[0]],source1[te]) and np.array_equal(features[exp.BASE_KEYS[1]],source2[te])
                print('AUDITED',d,fold,method,flush=True)
            audit['folds']+=1
    metadata=('dataset','method','mode','level','name','stage','t1','t2','m1','m2')
    def mean_group(group,keep):
        row={k:group[0][k] for k in keep}
        for k in group[0]:
            if k in row or k in ('fold','repeat','joint'):continue
            row[k]=float(np.mean([r[k] for r in group])) if all(r[k] is not None for r in group) else None
        return row
    by_fold=defaultdict(list)
    for r in rows:by_fold[tuple(r[k] for k in ('dataset','method','mode','fold','name','stage'))].append(r)
    folds_out=[]
    for group in by_fold.values():
        assert len(group)==(1 if group[0]['mode']=='none' else 3)
        r=mean_group(group,metadata+('fold',));r['repeats']=len(group)
        r['joint']=all(r[k] is not None for k in ('activation_pct','within_pct','fisher_pct')) and r['activation_pct']<-.0001 and r['within_pct']<-.0001 and r['fisher_pct']>.0001 and r['zero_fraction']==0
        r['all_repeats_joint']=all(x['joint'] for x in group);folds_out.append(r)
    groups=defaultdict(list)
    for r in folds_out:groups[tuple(r[k] for k in ('dataset','method','mode','name','stage'))].append(r)
    summary=[]
    for group in groups.values():
        assert len(group)==(5 if only_fold is None else 1)
        r=mean_group(group,metadata)
        r.pop('all_repeats_joint',None)
        r['joint_folds']=sum(x['joint'] for x in group)
        r['all_repeats_joint_folds']=sum(x['all_repeats_joint'] for x in group)
        for k in ('activation_pct','within_pct','fisher_pct','nonzero_pp','svm_clean_accuracy_pp'):
            vals=[x[k] for x in group]
            r[k+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 and all(x is not None for x in vals) else None
            r[k+'_min']=min(vals) if all(x is not None for x in vals) else None
            r[k+'_max']=max(vals) if all(x is not None for x in vals) else None
        summary.append(r)
    suffix=('' if len(datasets)==2 else '_'+datasets[0])+('' if only_fold is None else f'_fold{only_fold}')
    save_csv(DOC/f'repeat_fold_metrics{suffix}.csv',rows);save_csv(DOC/f'fold_metrics{suffix}.csv',folds_out)
    save_csv(DOC/f'summary{suffix}.csv',summary);save_json(DOC/f'summary{suffix}.json',summary)
    save_csv(DOC/f'category_metrics{suffix}.csv',categories);save_csv(DOC/f'dynamics{suffix}.csv',dynamics);save_csv(DOC/f'selections{suffix}.csv',selections)
    save_json(DOC/f'audit{suffix}.json',audit);exp.validate(protocol)
    return summary,selections,audit


def plots(summary,preview=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    dest=DOC/('preview' if preview else 'figures');dest.mkdir(exist_ok=True)
    colors=('#7c8798','#2d79b7','#3a9c81','#d98532')
    def get(d,method,mode,stage,name=MAIN):return next(r for r in summary if (r['dataset'],r['method'],r['mode'],r['stage'],r['name'])==(d,method,mode,stage,name))
    for d in (('imagenet',) if preview else ('imagenet','things')):
        for stage in (('fc1_output',) if preview else ('fc1_output','fc2_output','fc2_local')):
            fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
            metrics=[('activation_pct','平均激活变化 (%) ↓'),('within_pct','归一化类内离散 W 变化 (%) ↓'),('fisher_pct','Fisher B/W 变化 (%) ↑'),('nonzero_pp','非零单元比例变化 (百分点)'),('weak_activity_share','弱响应所占活动量 (%)'),('svm_clean_accuracy_pp','干净训练 SVM 平衡准确率变化 (百分点)')]
            for ax,(metric,title) in zip(axes.flat,metrics):
                for i,method in enumerate(METHODS):
                    vals=[get(d,method,mode,stage)[metric]*(100 if metric=='weak_activity_share' else 1) for mode in MODES]
                    err=None if metric=='weak_activity_share' else [get(d,method,mode,stage)[metric+'_sd'] for mode in MODES]
                    if err is not None and any(x is None for x in err):err=None
                    ax.bar(np.arange(3)+(i-1.5)*.19,vals,.18,label=NAMES[method],color=colors[i],yerr=err,capsize=2)
                ax.set_xticks(range(3),[MODE_NAMES[m] for m in MODES]);ax.set_title(title);ax.axhline(0,color='#333333',lw=.6);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            note='单外折排版预览，无误差线' if preview else '误差线为五折 SD'
            reference='变化相对该方法实际 FC2 输入' if stage=='fc2_local' else '变化相对同输入条件 vanilla'
            fig.suptitle(f'{d.upper()} · {LABELS[stage]} · 5 步串联\n{reference}；弱响应为绝对占比；{note}',fontsize=13)
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=4,frameon=False)
            for ext in ('png','svg','pdf'):fig.savefig(dest/f'{d}_{stage}.{ext}',dpi=180)
            plt.close(fig)
        if preview:continue
        fig,axes=plt.subplots(2,3,figsize=(15,7.5),layout='constrained')
        cnames=[MAIN,'inhibition_only_hard_5_frozen','gain1_hard_5_frozen','gain2_hard_5_frozen','gain_both_hard_5_frozen']
        short=['Hebbian串联','仅抑制','FC1缩放＋FC2 Hebb','FC1 Hebb＋FC2缩放','两层缩放']
        for row,stage in enumerate(('fc1_output','fc2_output')):
            for col,metric in enumerate(('activation_pct','within_pct','fisher_pct')):
                ax=axes[row,col]
                for i,method in enumerate(METHODS):
                    vals=[np.mean([get(d,method,m,stage,n)[metric] for m in MODES]) for n in cnames]
                    ax.bar(np.arange(5)+(i-1.5)*.19,vals,.18,color=colors[i],label=NAMES[method])
                ax.set_xticks(range(5),short,rotation=22,ha='right');ax.axhline(0,color='#333333',lw=.6);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
                ax.set_title(('FC1' if row==0 else '最终FC2')+' · '+dict(activation_pct='平均激活变化 (%) ↓',within_pct='类内 W 变化 (%) ↓',fisher_pct='Fisher变化 (%) ↑')[metric])
        fig.suptitle(f'{d.upper()} · 对照比较（干净、噪声、遮挡三条件等权描述均值）',fontsize=13)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=4,frameon=False)
        for ext in ('png','svg','pdf'):fig.savefig(dest/f'{d}_controls.{ext}',dpi=180)
        plt.close(fig)


def report(summary,selections,audit):
    def get(d,method,mode,stage,name=MAIN):return next(r for r in summary if (r['dataset'],r['method'],r['mode'],r['stage'],r['name'])==(d,method,mode,stage,name))
    def fmt(x,n=2):return '—' if x is None else f'{x:+.{n}f}'
    text=['# 群体活动自适应阈值：独立 FC 串联实验','',
        '固定 FC1 corr/k128、FC2 corr/k32，Hebbian 增益均为 0.75。两层权重与 RMS 尺度逐字复用上一轮干净 vanilla 训练折估计；原模型、前馈桥接权重及退化图像实现均未修改。',
        '', '## 机制和训练口径','',
        r'每层把物理输入激活 $h$ 归一化为 $x=h/s$，从 $r_0=x$ 开始。$d_t=x+0.75Er_t$；群体活动为每张图所有 4096 个单元的均值，包含零值。',
        '',r'均值/标准差阈值：$\theta_t=a\,\mathrm{mean}(d_t)+b\,\mathrm{std}(d_t)$（总体标准差）。动态反馈：$I_0=\mathrm{mean}(x)$，$\theta_t=gI_t$，$I_{t+1}=(1-\rho)I_t+\rho\,\mathrm{mean}(r_t)$。每图独立重置，反馈使用当前更新前的活动，具有一步延迟。',
        '',r'共同更新：$q_t=d_t\mathbf{1}[d_t>\theta_t]/(1+2\,\mathrm{mean}(r_t))$；$r_{t+1}=\mathbf{1}[q_t>0](0.5r_t+0.5q_t)$；输出为 $sr_T$。硬门控不从存活响应中减去阈值，并清除未过门单元的混合残余。',
        '', '新阈值没有每张图排序、固定存活数量或目标放电率。活动比例可以升高或降低；这不是数值错误。排序配额对照逐字复用上一轮结果。',
        '', '测试图自身的均值/标准差或反馈活动只是前向计算的输入；测试集之间不共享状态，也不据此更新参数、Hebbian矩阵或RMS尺度。',
        '', '每个外折只用干净训练概念再作两折内验证。FC1 按激活、归一化类内 W 和 Fisher 的最小对数改善选择；FC2 同时要求相对实际输入及 vanilla FC2 的改善。优先两内折均改善，然后最大化平均分；SVM 不参与选择。',
        '', '预先候选网格：a∈{0.25,0.5,1}，b∈{0,0.1,0.25,0.5}；g∈{0.25,0.5,1,2,4}，ρ∈{0.25,0.5,1}，阈值常数项为0。无约束均值/标准差族全部选择 b=0，故图中称“群体均值”。另在外折评估前声明 b>0 的补充族，FC2 在该族所选 FC1 下重新做同一干净内验证。完整网格、每折参数及失败候选保存在日志中。',
        '', '实际选择在两数据集、五外折和两层中完全一致：纯均值 a=0.25,b=0；均值＋标准差 a=0.25,b=0.1；动态反馈 g=0.25,ρ=1。ρ=1 表示这组最佳候选没有额外低通平滑，仍有更新时序带来的一步延迟，不能把它解释为慢速稳态调节。60个分层选择均在两个干净内折达到三指标同向改善，已另行重放参数排序审计。',
        '', '主分析 T=5；同时保留 1、20 步。没有不动点收敛筛选。噪声 σ=0.5 与旧分形遮挡强度0.3各3个固定随机实例；强度0.3不是遮挡面积30%。三次随机实例先在外折内取均值，再对五个概念隔离外折等权平均。',
        '', '相对变化先在每折、每随机实例内对其对应基线计算，再按上述层级平均；平均 Fisher 是各次 B/W 的平均，不是把平均 B 除以平均 W。',
        '', 'SVM 为 StandardScaler＋LinearSVC（C=1），只在训练概念拟合。主检查是每种处理的干净训练→相应测试；辅助退化重拟合仅覆盖5步 FC1/最终FC2及全部对照，1/20步和前馈桥接仅做主检查。另保留原始 vanilla SVM 兼容性检查。',
        '', '文中 SVM 准确率均指五大类召回率等权平均的平衡准确率，AUC 为五类 one-vs-rest 的宏平均；每类召回率也单独保留。',
        '', '## 5步结果','', '以下变化均相对同输入条件的 vanilla；A 为类平衡平均激活，W 为单位长度归一化后的类内离散，F 为 B/W。非零比例与 SVM 差值单位为百分点。绝对值、B、弱活动、零向量、每类指标和各折明细在 CSV 中。']
    overview=['','## 结果概览','', '下面的跨条件均值仅用于概览，不能替代各数据集、退化条件和外折的分层结果。没有按测试表现重新选择参数。',
        '', '| 层 | 阈值 | 平均 ΔA% | 平均 ΔW% | 平均 ΔF% | 平均 Δ非零比例 pp | 平均弱活动占比% | SVM 主检查差值范围 pp |','|---|---|---:|---:|---:|---:|---:|---:|']
    for stage in ('fc1_output','fc2_output'):
        for method in METHODS:
            group=[get(d,method,m,stage) for d in ('imagenet','things') for m in MODES]
            vals=[np.mean([r[k] for r in group]) for k in ('activation_pct','within_pct','fisher_pct','nonzero_pp')]
            weak=100*np.mean([r['weak_activity_share'] for r in group]);svm=[r['svm_clean_accuracy_pp'] for r in group]
            overview.append('| '+('FC1' if stage=='fc1_output' else '最终FC2')+' | '+NAMES[method]+' | '+' | '.join(fmt(v) for v in vals)+f' | {weak:.4f} | {fmt(min(svm))} 至 {fmt(max(svm))} |')
    overview+=['','“弱活动”定义为正响应小于该图最大响应1%的单元，其指标是这些单元所占总活动量；非零单元比例另行报告。这两个指标不能互换。',
        '', '群体阈值能否替代排序配额，需要区分两个目标：一是 A/W/F 所衡量的能量和类别几何；二是避免输出过密及大量弱响应。允许活动集自然变化后，它们可能出现取舍。下表只统计与原配额相比的方向，不做显著性推断。',
        '', '| 层 | 阈值 | A更低条件数 | W更低条件数 | F更高条件数 | 弱活动更少条件数 | 密度不高于vanilla条件数 |','|---|---|---:|---:|---:|---:|---:|']
    for stage in ('fc1_output','fc2_output'):
        for method in exp.FAMILIES:
            pairs=[(get(d,method,m,stage),get(d,'quota',m,stage)) for d in ('imagenet','things') for m in MODES]
            counts=[sum(a[k]<b[k] for a,b in pairs) for k in ('activation','within')]
            counts+=[sum(a['fisher']>b['fisher'] for a,b in pairs),sum(a['weak_activity_share']<b['weak_activity_share'] for a,b in pairs),sum(a['nonzero_pp']<=1e-5 for a,b in pairs)]
            overview.append('| '+('FC1' if stage=='fc1_output' else '最终FC2')+' | '+NAMES[method]+' | '+' | '.join(f'{c}/6' for c in counts)+' |')
    f2pairs=[(get(d,'moments',m,'fc2_output'),get(d,'quota',m,'fc2_output')) for d in ('imagenet','things') for m in MODES]
    domination=sum(a['activation']<b['activation'] and a['within']<b['within'] and a['fisher']>b['fisher'] for a,b in f2pairs)
    f1pairs=[(get(d,'feedback',m,'fc1_output'),get(d,'quota',m,'fc1_output')) for d in ('imagenet','things') for m in MODES]
    geometry_wins=sum(a['within']<b['within'] and a['fisher']>b['fisher'] for a,b in f1pairs)
    overview+=['',f'本轮结果：FC1 动态反馈在 {geometry_wins}/6 个数据/输入组合中比配额获得更低 W 和更高 F，但平均激活的下降较少，并明显增加活动密度。最终 FC2 的群体均值阈值在 {domination}/6 个组合中同时取得比配额更低 A、更低 W 和更高 F，也会增加活动密度。因此没有一种新阈值在两个层次及所有考虑维度上全面占优。',
        '', '若要在去掉排序配额后仍避免大量弱响应，均值＋标准差是本轮更接近原配额密度与弱活动水平的替代方案；其整体 W/F 略有改善，平均激活下降与配额接近。这个判断兼顾了输出形态，并不表示它在纯 A/W/F 三目标上优于另外两种自适应阈值。',
        '', '动态反馈变密与其更新规则一致：活动下降后反馈池也下降，后续阈值随之降低，Hebbian 邻居提供的弱驱动可能重新过门。标准差项在群体响应分布较宽时抬高阈值，减少弱响应。这里是基于机制与观察的解释，尚不是额外的因果消融证明。']
    text[4:4]=overview+['']
    for stage in ('fc1_output','fc2_output','fc2_local'):
        text+=['',f'### {LABELS[stage]}','', '| 数据 | 输入 | 阈值 | ΔA% ↓ | ΔW% ↓ | ΔF% ↑ | Δ非零比例 | ΔSVM主检查 | ΔSVM退化重拟合 | 三指标同向外折 |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for d in ('imagenet','things'):
            for mode in MODES:
                for method in METHODS:
                    r=get(d,method,mode,stage)
                    text.append(f"| {d} | {MODE_NAMES[mode]} | {NAMES[method]} | {fmt(r['activation_pct'])} | {fmt(r['within_pct'])} | {fmt(r['fisher_pct'])} | {fmt(r['nonzero_pp'])} | {fmt(r['svm_clean_accuracy_pp'])} | {fmt(r['svm_refit_accuracy_pp'])} | {r['joint_folds']}/5 |")
    text+=['','## 对照与有限步敏感性','', '各条件仍完整区分仅抑制、FC1缩放＋FC2 Hebbian、FC1 Hebbian＋FC2缩放、两层缩放。缩放逐图匹配当前阈值族、当前输入的5步真串联平均激活；因此它是诊断对照，计算目标仍依赖真串联，不是独立部署算法。仅抑制关闭两层 E 项，保留同一族阈值参数、除性抑制和更新规则；它未单独寻优。']
    text+=['', '对照的主要含义：仅抑制能更强地压低 FC1 平均激活，却没有带来相应 W/F 改善；两层缩放在最终 FC2 的平均 Fisher 提升约0.43%–0.53%，远小于真串联的15.23%–17.81%。FC1缩放＋FC2 Hebbian仍保留12.58%–14.31%的平均 Fisher提升，说明最终收益中较大部分来自FC2侧向处理，但完整FC1塑形还提供额外收益。这里不能推论所有抑制算法都无效，也不能仅凭这些对照证明学习所得图谱优于任意随机侧向图谱。']
    for stage in ('fc1_output','fc2_output'):
        text+=['',f'### {LABELS[stage]}：跨两数据集与三输入条件的等权描述均值','', '| 阈值 | 配置 | ΔA% | ΔW% | ΔF% | ΔSVM主检查 pp |','|---|---|---:|---:|---:|---:|']
        for method in METHODS:
            for name,label in [(MAIN,'Hebbian串联'),('inhibition_only_hard_5_frozen','仅抑制'),('gain1_hard_5_frozen','FC1缩放＋FC2 Hebb'),('gain2_hard_5_frozen','FC1 Hebb＋FC2缩放'),('gain_both_hard_5_frozen','两层缩放')]:
                group=[get(d,method,m,stage,name) for d in ('imagenet','things') for m in MODES]
                vals=[np.mean([r[k] for r in group]) for k in ('activation_pct','within_pct','fisher_pct','svm_clean_accuracy_pp')]
                text.append('| '+NAMES[method]+' | '+label+' | '+' | '.join(fmt(v) for v in vals)+' |')
    text+=['','### 1、5、20步：跨两数据集和三输入条件等权描述均值','', '| 层 | 阈值 | 步数 | ΔA% | ΔW% | ΔF% | Δ非零比例 pp | ΔSVM主检查 pp |','|---|---|---:|---:|---:|---:|---:|---:|']
    for stage in ('fc1_output','fc2_output'):
        for method in METHODS:
            for t in (1,5,20):
                group=[get(d,method,m,stage,f'both_hard_hard_{t}_{t}_frozen') for d in ('imagenet','things') for m in MODES]
                vals=[np.mean([r[k] for r in group]) for k in ('activation_pct','within_pct','fisher_pct','nonzero_pp','svm_clean_accuracy_pp')]
                text.append('| '+stage+' | '+NAMES[method]+f' | {t} | '+' | '.join(fmt(v) for v in vals)+' |')
    text+=['','## 解释边界','',
        '平均激活降低本身不等于 sharpening。W 与 F 在逐图单位长度归一化表征上计算，纯标量缩放应保持其方向几何；同时报告未归一化几何，不排除幅度带来的 SVM 准确率变化。SVM 是功能检查，不是主要优化目标。',
        '', '较低 W 也可能来自共同方向塌缩，因此一并检查 B、F、零向量、有效维度、最大单元活动份额、每类指标和主 SVM。三指标同向的折数是描述性稳健性检查，不是独立显著性检验。',
        '', '本实验比较的是固定干净图谱与固定增益下的阈值方案。最优候选位于较弱阈值边界时，只说明当前候选范围和所用 sharpening 目标偏好较弱抑制，不能称为全局最优。动态反馈是群体活动的简化代理，不意味着训练成了生物神经回路。',
        '', '这仍是沿用此前两套样本的探索性机制比较。内外折概念隔离防止本轮参数拟合直接使用测试概念，但反复研究同一批样本不等于获得了新的独立确认数据。',
        '', '## 审计与文件','', f"审计通过：新增 {audit['new_phases']} 个输入阶段、{audit['new_representations']} 组测试表征；新拟合干净 SVM {audit['new_clean_svm_fits']} 个、退化 SVM {audit['new_degraded_svm_fits']} 个，基线模型另复用。排序配额 {audit['reused_quota_phases']} 个旧阶段逐字复用。",
        '',f"独立 NumPy 几何复算最大绝对误差 {audit['geometry_max_error']:.3g}，前馈桥接最大误差 {audit['bridge_max_error']:.3g}，SVM 分数重放误差 {audit['svm_replay_max_error']:.3g}。原模型及旧实验哈希校验通过。",
        '', '- `summary.csv/json`：每层、每输入、每阈值、每配置汇总。','- `fold_metrics.csv`、`repeat_fold_metrics.csv`：外折与随机实例明细。','- `category_metrics.csv`：五大类分别统计。','- `selections.csv`：每折每层阈值参数；完整候选在日志 selection JSON。','- `dynamics.csv`：全输入批次（含训练表征）的阈值、群体反馈、活动集切换和残差诊断，未用于收敛筛选；主指标始终只在外折测试表征计算。','- `figures/`：六幅分层主图和两幅对照图，PNG/SVG/PDF。','- `audit.json`：输入、模型、概念隔离、几何和 SVM 审计。']
    text+=['','## 柱状图索引','', '| 数据 | FC1 | 最终FC2 | FC2本层增量 | 对照 |','|---|---|---|---|---|']
    for d in ('imagenet','things'):
        links=[f'[PNG](figures/{d}_{stage}.png) · [PDF](figures/{d}_{stage}.pdf) · [SVG](figures/{d}_{stage}.svg)' for stage in ('fc1_output','fc2_output','fc2_local','controls')]
        text.append('| '+d+' | '+' | '.join(links)+' |')
    (DOC/'实验结果.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('imagenet','things'));p.add_argument('--fold',type=int)
    p.add_argument('--render-only',action='store_true');p.add_argument('--wait',action='store_true');a=p.parse_args();WAIT=a.wait
    with threadpool_limits(limits=4):
        if a.render_only:s=read(DOC/'summary.json');selections=[];audit=read(DOC/'audit.json')
        else:s,selections,audit=aggregate((a.dataset,) if a.dataset else ('imagenet','things'),a.fold)
        if not a.dataset and a.fold is None:report(s,selections,audit);plots(s)
    print('REPORT DONE',{k:v for k,v in audit.items() if k!='phase_hashes'},flush=True)
