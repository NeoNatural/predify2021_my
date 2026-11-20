"""Audit, aggregate and render completed concept-fold Hebb-FC study."""
from pathlib import Path
import csv,json,sys,argparse
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.hebb_fc_full_20261103 import OUT,DOC,atomic_json
from analysis.hebb_fc_full_replay_20261103 import METHODS
from predify_experiments import native_categories_20261029 as native

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def report():
    native.verify_freeze();timeline=[];geometry=[];audit={}
    for dataset in ('imagenet','things'):
        m=json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'));trials=m['trials'];n=len(trials);k=len(m['categories'])
        assert json.loads((OUT/dataset/'extraction_complete.json').read_text(encoding='utf8'))['baseline_fc2_bit_identical']
        for fold in range(5):
            dest=OUT/dataset/'folds'/f'fold{fold}';rep=json.loads((dest/'replay_complete.json').read_text(encoding='utf8'))
            assert json.loads((dest/'decode_complete.json').read_text(encoding='utf8'))['complete']
            assert (dest/'geometry_complete.json').exists();geometry.extend(read(dest/'geometry.csv'))
            test=np.array([j for j,t in enumerate(trials) if t['fold']==fold]);n_test=len(test)
            assert rep['test_rows']==test.tolist()
            for noise,si in [('clean',[0]),('noisy',[1,2,3])]:
                for condition in native.conditions():
                    name=condition['name']
                    for method in ('off',*METHODS):
                        if method=='off':
                            zs={}
                            for kind in ('mvpa','t9_mode_specific','t9_clean_reference'):
                                path=native.OUT/dataset/'cv'/noise/kind/f'{name}.npz'
                                with np.load(path) as z:
                                    mask=z['fold']==fold;zs[kind]=z['scores'][mask];y=z['y'][mask]
                        else:
                            path=dest/'scores'/method/f'{noise}_{name}.npz'
                            with np.load(path) as z:
                                y=z['y'];assert np.array_equal(z['rows'],np.tile(test,len(si)))
                                zs={kind:z[kind] for kind in ('mvpa','t9_mode_specific','t9_clean_reference')}
                        assert len(y)==n_test*len(si) and set(y)==set(range(k))
                        for kind,scores in zs.items():
                            assert scores.shape==(len(y),15,k) and np.isfinite(scores).all()
                            for t in range(15):
                                v=scores[:,t];acc=100*np.mean(v.argmax(1)==y)
                                auc=np.mean([roc_auc_score(y==j,v[:,j]) for j in range(k)])
                                timeline.append(dict(dataset=dataset,fold=fold,method=method,noise=noise,condition=name,readout=kind,timestep=t,accuracy_pct=float(acc),macro_auc=float(auc)))
        audit[dataset]={'targets':n,'concepts':len(m['concept_folds']),'folds':5,'classes':k,'sequences':n*48,'references':n}
    write(DOC/'fold_readout_timeline.csv',timeline);write(DOC/'fold_geometry_timeline.csv',geometry)
    grouped=defaultdict(list)
    for r in timeline:
        key=tuple(r[k] for k in ('dataset','method','noise','condition','readout','timestep'));grouped[key].append(r)
    means=[]
    for key,rs in grouped.items():
        assert len(rs)==5 and {r['fold'] for r in rs}==set(range(5))
        means.append(dict(zip(('dataset','method','noise','condition','readout','timestep'),key),
                          accuracy_pct=float(np.mean([r['accuracy_pct'] for r in rs])),macro_auc=float(np.mean([r['macro_auc'] for r in rs]))))
    write(DOC/'readout_timeline.csv',means)
    by={(r['dataset'],r['method'],r['noise'],r['condition'],r['readout'],r['timestep']):r for r in means}
    summaries=[]
    for dataset in audit:
        for method in ('off',*METHODS):
            for noise in ('clean','noisy'):
                for carrier in native.CARRIERS:
                    for kind in ('mvpa','t9_mode_specific','t9_clean_reference'):
                        vals={ident:np.mean([by[dataset,method,noise,f'{carrier}_{ident}',kind,t]['accuracy_pct'] for t in range(5)]) for ident in native.IDENTITIES}
                        summaries.append(dict(dataset=dataset,method=method,noise=noise,carrier=carrier,readout=kind,
                            exact_accuracy_pct=vals['exact'],different_category_accuracy_pct=vals['different_category'],
                            exact_minus_different_category_pp=vals['exact']-vals['different_category']))
    write(DOC/'early_readout_summary.csv',summaries)
    lookup={(r['dataset'],int(r['fold']),r['method'],r['layer'],r['condition'],int(r['seed']),int(r['timestep'])):r for r in geometry}
    grows=[]
    for dataset in audit:
        for method in METHODS:
            for noise,seeds in [('clean',[0]),('noisy',[2921,2922,2923])]:
                for carrier in native.CARRIERS:
                    for layer in ('fc1','fc2'):
                        for fold in range(5):
                            values=[]
                            for seed in seeds:
                                for t in range(5):
                                    q=(dataset,fold,method,layer,carrier+'_exact',seed,t);b=(dataset,fold,'off',layer,carrier+'_exact',seed,t)
                                    a=lookup[q];o=lookup[b]
                                    values.append((float(a['normalized_fisher']),float(o['normalized_fisher']),float(a['activation']),float(o['activation'])))
                            z=np.array(values);grows.append(dict(dataset=dataset,method=method,noise=noise,carrier=carrier,layer=layer,fold=fold,
                                fisher_change_pct=100*(z[:,0].mean()/max(z[:,1].mean(),1e-12)-1),activation_change_pct=100*(z[:,2].mean()/max(z[:,3].mean(),1e-12)-1)))
    write(DOC/'early_geometry_effects_by_fold.csv',grows)
    lines=['# Predify 与 Hebb-FC：概念分组五折实验','','完整重建了 34,560 条预览条件轨迹和 720 条无预览参考轨迹；每条都与原实验 FC2 逐位核对。每个外层折仅用其余四折的干净目标拟合 FC 连接和尺度，读出训练不使用测试折。预览角色也按概念归属于各自的折。',
       '','FC 使用均值＋标准差门控（FC1 k128、FC2 k32），与关闭、仅抑制和逐样本均值匹配缩放对照比较。各网络时间步做五次 FC 内部迭代。FC 在全部 PCoder 后，重放复用相同 Predify 状态；批量重放不是新的 PCoder 轨迹。','',
       '噪声结果包含 3 个固定实现；每类按概念分组五折；MVPA 每条件每时刻训练，t9-A 合并同模式四种预览等权训练，t9-B 在无预览干净参考 t9 训练。以下均为描述性五折平均，early 为 t0–t4，未把折或噪声实现当作独立样本进行显著性检验。','',
       '| 数据集 | FC | 目标 | 模式 | 读出 | exact ACC | exact − 异类（百分点） |','|---|---|---|---|---|---:|---:|']
    for r in summaries:
        if r['method'] not in ('off','mean_std') or r['readout']!='t9_mode_specific':continue
        lines.append('| '+ ' | '.join([r['dataset'],r['method'],r['noise'],r['carrier'],r['readout'],f"{r['exact_accuracy_pct']:.2f}%",f"{r['exact_minus_different_category_pp']:+.2f}"])+' |')
    lines+=['','下表为 FC2 exact 的归一化 Fisher 比值变化，先在每折、每噪声和时间点计算，然后以 early 窗口的均值之比计算相对变化，再对五折平均；仅描述 held-out 折内类别几何。',
       '','| 数据集 | FC | 目标 | 模式 | Fisher 相对变化 |','|---|---|---|---|---:|']
    for dataset in audit:
        for method in METHODS:
            for noise in ('clean','noisy'):
                for carrier in native.CARRIERS:
                    v=[r['fisher_change_pct'] for r in grows if r['dataset']==dataset and r['method']==method and r['noise']==noise and r['carrier']==carrier and r['layer']=='fc2']
                    lines.append(f'| {dataset} | {method} | {noise} | {carrier} | {np.mean(v):+.2f}% |')
    lines+=['','本研究为探索性复测，原数据历史覆盖补充审计发现部分概念与图片曾出现在独立 FC 研究；不能称为全新独立验证。FC 超参数也沿用先前探索值。批量数值偏差在各折 replay_complete.json 中记录。',
      '','[读出时程](readout_timeline.csv) · [分折读出](fold_readout_timeline.csv) · [分折几何](fold_geometry_timeline.csv) · [早期读出](early_readout_summary.csv) · [早期几何](early_geometry_effects_by_fold.csv) · [审计](audit.json)']
    DOC.mkdir(parents=True,exist_ok=True);(DOC/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    atomic_json(DOC/'audit.json',{'passed':True,'datasets':audit,'readout_rows':len(timeline),'geometry_rows':len(geometry),
       'qualification':'descriptive CV; prior FC experiment concept/image overlap; batch replay is not bit-identical',
       'execution':'fold-specific graph and fixed readout training exclude complete test fold'})
    print('REPORT COMPLETE',DOC/'实验结果.md',flush=True)

if __name__=='__main__':report()
