"""Verify every OOF curve, audit model counts, then render study figures/report."""
from pathlib import Path
import sys,json,csv,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from predify_experiments.native_categories_20261029 import OUT,DOC,CARRIERS,IDENTITIES,conditions,verify_freeze
from predify_experiments.supplementary import load
from predify_experiments.readout_init_20261012 import sha
from analysis.prepare_native_20261029 import read,write
import analysis.plot_adaptive_20261028 as style
KINDS=('mvpa','t9_mode_specific','t9_clean_reference')
HEADINGS={'mvpa':'Time/condition-specific MVPA','t9_mode_specific':'Mode-specific frozen t9 readout','t9_clean_reference':'Clean no-memory t9 transfer'}

def audit():
    verify_freeze();result={};total=0
    for dataset,n in [('imagenet',320),('things',400)]:
        m=json.loads((OUT/dataset/'manifest.json').read_text(encoding='utf8'));k=len(m['categories']);assert len(m['trials'])==n
        summary=read(DOC/dataset/'cv_summary.csv');lookup={(r['noise'],r['readout'],r['condition'],int(r['timestep'])):r for r in summary};count=0;maxerr=0.
        for noise in ('clean','noisy'):
            for kind in KINDS:
                for c in conditions():
                    with np.load(OUT/dataset/'cv'/noise/kind/f"{c['name']}.npz") as z:
                        scores=z['scores'];y=z['y'];fold=z['fold'];assert scores.shape==(n*(1 if noise=='clean' else 3),15,k)
                        assert np.isfinite(scores).all();np.testing.assert_array_equal(scores.argmax(-1),z['predictions'])
                    for t in range(15):
                        acc=[];auc=[]
                        for f in range(5):
                            mask=fold==f;acc.append(np.mean(scores[mask,t].argmax(-1)==y[mask])*100)
                            auc.append(np.mean([roc_auc_score(y[mask]==j,scores[mask,t,j]) for j in range(k)]))
                        r=lookup[noise,kind,c['name'],t]
                        errors=[abs(float(r['accuracy_fold_mean'])-np.mean(acc)),abs(float(r['roc_auc_fold_mean'])-np.mean(auc)),abs(float(r['accuracy_oof_pooled'])-np.mean(scores[:,t].argmax(-1)==y)*100)]
                        maxerr=max(maxerr,max(errors));assert max(errors)<1e-10
                        count+=1
        modelcounts={kind:len(list((OUT/dataset/'cv').glob(f'*/{kind}/*.pckl'))) for kind in ('mvpa','t9_mode_specific')}
        modelcounts['t9_clean_reference']=len(list((OUT/dataset/'cv/reference_heads').glob('fold*.pckl')))
        assert modelcounts==dict(mvpa=1800,t9_mode_specific=30,t9_clean_reference=5),modelcounts
        for p in (OUT/dataset/'cv').rglob('*.pckl'):
            v=load(p);assert not set(v['train_rows'])&set(v['test_rows'])
            if 'attempts' in v:assert v['attempts'][-1]['converged']
        references=list((OUT/dataset/'formal').glob('reference_*.pckl'));assert len(references)==n
        input_hashes={}
        for t in m['trials']:
            with np.load(OUT/dataset/'formal'/f"stimuli_{t['dataset_index']}.npz") as z:
                for seed in (0,2921,2922,2923):
                    a=z[f'target_{seed}'];assert a.shape==(3,224,224) and a.dtype==np.float32 and np.isfinite(a).all()
                    input_hashes[t['dataset_index'],seed]=hashlib.sha256(a.tobytes()).hexdigest()
        records=list((OUT/dataset/'formal').glob('concept_*.pckl'));assert len(records)==n//4*48
        for path in records:
            v=load(path)
            for t,r in zip(v['trials'],v['results']):
                assert r['target_hash']==input_hashes[t['dataset_index'],v['seed']] and r['target_W_frozen']
                assert len(r['commits'])==(0 if v['condition']['carrier']=='pcoder' else 4)
        for path in references:
            v=load(path);assert v['result']['target_hash']==input_hashes[v['trial']['dataset_index'],0] and v['result']['commits']==[]
        grams=list((OUT/dataset/'grams').glob('*.npz'));assert len(grams)==48*4
        for p in grams:
            with np.load(p) as z:assert int(z['images_per_concept'])==4 and len(z['concepts'])==n//4
        complete=json.loads((OUT/dataset/'formal/complete.json').read_text(encoding='utf8'));assert complete['sequences']==48*n
        result[dataset]=dict(targets=n,concepts=n//4,classes=k,formal_sequences=48*n,reference_sequences=n,models=modelcounts,recomputed_curve_points=count,max_metric_error=maxerr,gram_files=len(grams),actual_input_hashes_verified=len(input_hashes),target_W_frozen=True);total+=49*n
    assert total==35280
    value=dict(passed=True,datasets=result,target_sequences=total,cv_models=3670,verified_at=time.time(),limitations=['Classification results are descriptive CV estimates; no significance stars.','ImageNet includes previously viewed concepts with unused images.','THINGS freshness is relative to traceable experiment history, not pretraining.','Perceptual hash and contact review cannot prove all possible transformed duplicates are absent.'])
    (DOC/'final_audit.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');return value

def figures():
    style.DOC=DOC;style.OUT=OUT;style.DEST=DOC/'figures';style.DEST.mkdir(parents=True,exist_ok=True)
    summary={d:read(DOC/d/'cv_summary.csv') for d in ('imagenet','things')}
    for dataset in ('imagenet','things'):
        n=320 if dataset=='imagenet' else 400;chance=25 if dataset=='imagenet' else 20
        for kind in KINDS:
            for field,label,ref,physical in [('accuracy_fold_mean','Accuracy (%)',chance,(0,100)),('roc_auc_fold_mean','Macro OVR ROC-AUC',.5,(0,1))]:
                for noise in ('clean','noisy'):
                    ys={}
                    for c in CARRIERS:
                        for i in IDENTITIES:
                            rows=sorted([r for r in summary[dataset] if r['readout']==kind and r['noise']==noise and r['condition']==f'{c}_{i}'],key=lambda r:int(r['timestep']))
                            assert len(rows)==15;ys[c,i]=np.array([float(r[field]) for r in rows])
                    for mode in ('adaptive','linear','late','delta'):
                        ts=np.arange(5,15) if mode=='late' else np.arange(15)
                        series={(c,i):((v-ys[c,'different_category']) if mode=='delta' else v)[ts] for (c,i),v in ys.items() if mode!='delta' or i!='different_category'}
                        spec=style.Axis(np.concatenate(list(series.values())),0 if mode=='delta' else None if mode=='late' else ref,None if mode=='delta' else physical,compress=mode=='adaptive')
                        fig,axs=plt.subplots(1,3,figsize=(14.5,4.5),sharex=True,sharey=True)
                        for ax,c in zip(axs,CARRIERS):
                            for i in IDENTITIES[:-1] if mode=='delta' else IDENTITIES:style.curve(ax,ts,series[c,i],i)
                            spec.apply(ax,record=dict(dataset=dataset,readout=kind,noise=noise,metric=field,mode=mode,carrier=c));ax.set_title(style.LABELS[c]);ax.set_xlabel('Target timestep');ax.set_xticks(ts[::2]);ax.grid(alpha=.18)
                        axs[0].set_ylabel(('Difference vs different-category (pp)' if field.startswith('accuracy') else 'Difference vs different-category (AUC)') if mode=='delta' else label)
                        axes=axs[0].get_legend_handles_labels();fig.legend(*axes,loc='lower center',ncol=4,frameon=False)
                        foot=f'Five-fold arithmetic mean; N={n} clean / {3*n} noisy. Original timepoints; no smoothing or significance inference.'
                        if spec.compressed:foot+=' Grey regions compress y; ticks retain true values.'
                        fig.suptitle(f'{dataset.upper()} | {HEADINGS[kind]} | {noise} | {mode}');fig.text(.5,.09,foot,ha='center',fontsize=8)
                        fig.tight_layout(rect=(0,.13,1,.93));style.save(fig,f'{dataset}_{kind}_{noise}_{field}_{mode}','readout')
        print(dataset,'READOUT FIGURES COMPLETE',flush=True)
    activation={d:read(DOC/d/'activation_timeline.csv') for d in ('imagenet','things')}
    geometry={d:read(DOC/d/'geometry_timeline.csv') for d in ('imagenet','things')}
    effects={d:read(DOC/d/'representation_effects.csv') for d in ('imagenet','things')}
    style.representation_figures(activation,geometry,effects)
    (DOC/'figure_manifest.json').write_text(json.dumps(dict(figures=style.SAVED,axes=style.AXES),indent=2),encoding='utf8')

def report(audit):
    lines=['# 原生类别扩样与两套t9读出：实验结果','','已完成35280条target轨迹（34560条正式条件＋720条唯一clean参考）和3670个CV分类器；开发验收另计。',
           '', 'THINGS：五类、100个全新概念、400张正式目标；ImageNet：四类、80概念、320张未用目标图，其中一部分概念曾用于历史实验（详见数据验收表），本轮为扩样探索研究。植物开发队列经用户确认改为1概念，其余每类4概念。',
           '', 'MVPA保持逐条件、逐时刻、概念分组五折训练。t9-A按模式和噪声独立拟合，四种内容合并等权；t9-B每图一份无记忆clean t9参考，迁移到全部条件。B下降可反映分布与边界不匹配，不能据此声称信息减少。',
           '', '下表为预定early（t0–t4）窗口、相对同模式different-category的准确率差值（百分点）。均为描述性CV估计，不将折、噪声或时间当独立受试者；无显著性星号。',
           '', '| 数据集 | 读出 | target | 模式 | Exact | 同概念 | 同大类 |','|---|---|---|---|---:|---:|---:|']
    for dataset in ('imagenet','things'):
        rr=read(DOC/dataset/'cv_window_contrasts.csv')
        for kind in KINDS:
            for noise in ('clean','noisy'):
                for carrier in CARRIERS:
                    vals=[next(r for r in rr if r['readout']==kind and r['noise']==noise and r['carrier']==carrier and r['identity']==i and r['window']=='early') for i in IDENTITIES[:-1]]
                    lines.append('| '+' | '.join([dataset,kind,noise,carrier]+[f"{float(v['delta_accuracy_pp']):+.2f}" for v in vals])+' |')
    lines+=['','完整时程、ACC与macro OVR ROC-AUC、单点t9、early/middle/late/full窗口和归一化时间积分分别保存在各数据集CSV。机会水平：THINGS20%、ImageNet25%，ROC-AUC均为0.5；不直接比较两数据集的绝对成绩。',
            '', '四层激活与sharpening覆盖三模式、clean/noisy、四种内容；完整flatten表征的FP32概念和与FP64 Gram及平方和支持原定义和10000次类别分层概念bootstrap。没有采用GAP或随机投影。区间是边际95%区间，不是跨层/时刻同时区间。',
            '', '[完整数据审计](final_audit.json)；[执行方案](执行方案.md)；[图表清单](figure_manifest.json)。']
    for d in ('imagenet','things'):
        for k in KINDS:
            lines+=['',f'![{d} {k} noisy ACC](figures/{d}_{k}_noisy_accuracy_fold_mean_adaptive.png)']
        lines+=['',f'![{d} 四层激活差值](figures/{d}_noisy_output_delta_three_modes.png)', '',f'![{d} FC2 sharpening](figures/{d}_noisy_fc2_sharpening_effects_three_modes.png)']
    (DOC/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':
    checked=audit();figures();report(checked)
    print('REPORT COMPLETE',DOC/'实验结果.md',flush=True)
