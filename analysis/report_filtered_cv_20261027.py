"""English scientific figures and Chinese report from fixed saved tables."""
from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.filtered_cv_20261027 import OUT,DOC,CARRIERS,IDENTITIES,LAYERS,NOISES
LABELS=dict(hebb='Hebbian-only',pcoder='PCoder carry-only',joint='Joint')
COLORS=dict(exact='#bf4342',same_concept='#d99220',same_category='#278575',different_category='#586779')
TEXT=dict(exact='Exact',same_concept='Same concept',same_category='Same category',different_category='Different category')

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def link(path,label):return f'[{label}](<{path.resolve().as_posix()}>)'
def img(path,label):return f'![{label}](<{path.resolve().as_posix()}>)'
def savefig(fig,name):
    dest=DOC/'figures'/name;dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=180,bbox_inches='tight');plt.close(fig);return dest

def write_figure_index():
    lines=['# 本轮更新图表索引','',
        '各载体内部均以different-category为基线；clean/noisy分列。CV为五折均值；sharpening阴影为边际描述性95%概念bootstrap区间。',
        '', '分类图的accuracy单位为%，差值为pp；ROC-AUC及其差值使用0–1尺度。output与legacy x_tmp激活分别绘图。']
    if (DOC/'自适应坐标与三种记忆图册.md').exists():
        lines+=['','**最新版本：** '+link(DOC/'自适应坐标与三种记忆图册.md','激活重叠核查、ACC/AUC自适应坐标、三种记忆模式完整图册')]
    for dataset in ('imagenet','things'):
        lines+=['',f'## {dataset.upper()}','', '| 图组 | 图表 |', '|---|---|']
        for p in sorted((DOC/'figures').glob(dataset+'_*.png')):
            group='输入' if 'actual_inputs' in p.name else '历史激活' if 'legacy_activation' in p.name else '输出激活' if 'output_activation' in p.name else 'Sharpening' if 'sharpening' in p.name else '交叉验证读出'
            lines.append(f'| {group} | {link(p,p.stem)} |')
    lines+=['',link(DOC/'figures/filter_response.png','滤波核与频响'),'',link(DOC/'实验结果.md','完整实验报告')]
    if (DOC/'advanced_figures/all_factor_effect_matrix.svg').exists():
        lines+=['','## 高维可视化','',link(DOC/'advanced_figures/all_factor_effect_matrix.svg','四指标×四层×全部条件×时间：矢量总览'),'',link(DOC/'advanced_figures/all_factor_effect_matrix.png','高分辨率总览'),'',link(DOC/'高维可视化说明.md','联动矩阵与3D快照说明')]
    if (DOC/'2D绘图更新与图册.md').exists():
        lines+=['','## 新版2D绘图','',link(DOC/'2D绘图更新与图册.md','ACC压缩/线性/后期/差值，以及原始激活与sharpening完整图册')]
    (DOC/'图表索引.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def decoding_figures(dataset):
    rows=read(DOC/dataset/'cv_summary.csv');differences=read(DOC/dataset/'cv_content_contrasts.csv');paths=[]
    for method in ('t9','mvpa'):
        for metric,label,field in [('accuracy','Accuracy (%)','accuracy_fold_mean'),('roc_auc','Macro OVR ROC-AUC','roc_auc_fold_mean')]:
            fig,axs=plt.subplots(3,2,figsize=(12,10),sharex=True,sharey=True)
            for ri,carrier in enumerate(CARRIERS):
                for col,noise in enumerate(('clean','noisy')):
                    ax=axs[ri,col]
                    for identity in IDENTITIES:
                        values=[r for r in rows if r['readout']==method and r['noise']==noise and r['condition']==f'{carrier}_{identity}'];values.sort(key=lambda r:int(r['timestep']))
                        ax.plot(range(15),[float(r[field]) for r in values],color=COLORS[identity],label=TEXT[identity],lw=1.8)
                    ax.axhline(25 if metric=='accuracy' else .5,color='#aaa',ls=':');ax.grid(alpha=.15);ax.set_title(f'{LABELS[carrier]} | {noise}')
                    if method=='t9':ax.axvline(9,color='#bbb',ls='--',lw=.8)
                    if col==0:ax.set_ylabel(label)
                    if ri==2:ax.set_xlabel('Target timestep')
            axs[0,0].legend(fontsize=8,ncol=2);fig.suptitle(f'{dataset.upper()} | {"Shared t9 head: temporal generalization" if method=="t9" else "Time-specific MVPA"}\nArithmetic mean over 5 concept/preview-disjoint folds; descriptive')
            fig.tight_layout(rect=(0,0,1,.94));paths.append(savefig(fig,f'{dataset}_{method}_{metric}.png'))
        fig,axs=plt.subplots(3,2,figsize=(12,9),sharex=True)
        for ri,carrier in enumerate(CARRIERS):
            for col,noise in enumerate(('clean','noisy')):
                ax=axs[ri,col]
                for identity in IDENTITIES[:-1]:
                    rs=[r for r in differences if r['readout']==method and r['noise']==noise and r['carrier']==carrier and r['identity']==identity];rs.sort(key=lambda r:int(r['timestep']))
                    ax.plot(range(15),[float(r['delta_accuracy_pp']) for r in rs],label=TEXT[identity],color=COLORS[identity])
                ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.15);ax.set_title(f'{LABELS[carrier]} | {noise}');ax.set_ylabel('Accuracy difference (pp)')
                if ri==2:ax.set_xlabel('Target timestep')
        axs[0,0].legend(fontsize=8);fig.suptitle(f'{dataset.upper()} | {method.upper()} | Within-carrier priming minus different category\nPaired within each fold; no significance inference');fig.tight_layout(rect=(0,0,1,.94));paths.append(savefig(fig,f'{dataset}_{method}_content_effects.png'))
        fig,axs=plt.subplots(3,2,figsize=(12,9),sharex=True)
        for ri,carrier in enumerate(CARRIERS):
            for col,noise in enumerate(('clean','noisy')):
                ax=axs[ri,col]
                for identity in IDENTITIES[:-1]:
                    rs=[r for r in differences if r['readout']==method and r['noise']==noise and r['carrier']==carrier and r['identity']==identity];rs.sort(key=lambda r:int(r['timestep']))
                    ax.plot(range(15),[float(r['delta_roc_auc']) for r in rs],label=TEXT[identity],color=COLORS[identity])
                ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.15);ax.set_title(f'{LABELS[carrier]} | {noise}');ax.set_ylabel('ROC-AUC difference (0-1 scale)')
                if ri==2:ax.set_xlabel('Target timestep')
        axs[0,0].legend(fontsize=8);fig.suptitle(f'{dataset.upper()} | {method.upper()} | Within-carrier priming minus different category\nMacro OVR ROC-AUC; paired within each fold; descriptive');fig.tight_layout(rect=(0,0,1,.94));paths.append(savefig(fig,f'{dataset}_{method}_auc_content_effects.png'))
    return paths

def representation_figures(dataset):
    rows=read(DOC/dataset/'representation_effects.csv');paths=[]
    for carrier in CARRIERS:
        fig,axs=plt.subplots(4,2,figsize=(12,12),sharex=True)
        for ri,layer in enumerate(LAYERS):
            for col,noise in enumerate(('clean','noisy')):
                ax=axs[ri,col]
                for identity in IDENTITIES[:-1]:
                    rs=[r for r in rows if r['carrier']==carrier and r['layer']==layer and r['noise']==noise and r['identity']==identity and r['metric']=='output_mean_effect_pct'];rs.sort(key=lambda r:int(r['timestep']))
                    v=np.array([[float(r[k]) for k in ('value','low','high')] for r in rs]);ax.plot(range(15),v[:,0],color=COLORS[identity],label=TEXT[identity]);ax.fill_between(range(15),v[:,1],v[:,2],color=COLORS[identity],alpha=.12)
                ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.15);ax.set_title(f'{layer.upper()} | {noise}');ax.set_ylabel('Output mean change (%)')
                if ri==3:ax.set_xlabel('Target timestep')
        axs[0,0].legend(fontsize=8);fig.suptitle(f'{dataset.upper()} | {LABELS[carrier]} | Output activation vs different category\nMarginal descriptive 95% concept-bootstrap intervals (10,000)');fig.tight_layout(rect=(0,0,1,.94));paths.append(savefig(fig,f'{dataset}_{carrier}_output_activation.png'))
        for layer in LAYERS:
            fig,axs=plt.subplots(3,2,figsize=(12,9),sharex=True)
            metrics=[('normalized_trace_effect_pct','Normalized trace change (%)'),('within_cosine_delta_x1e3','Within-class cosine difference (x1e3)'),('normalized_fisher_effect_pct','Normalized Fisher change (%)')]
            for ri,(metric,label) in enumerate(metrics):
                for col,noise in enumerate(('clean','noisy')):
                    ax=axs[ri,col]
                    for identity in IDENTITIES[:-1]:
                        rs=[r for r in rows if r['carrier']==carrier and r['layer']==layer and r['noise']==noise and r['identity']==identity and r['metric']==metric];rs.sort(key=lambda r:int(r['timestep']))
                        v=np.array([[float(r[k]) for k in ('value','low','high')] for r in rs]);ax.plot(range(15),v[:,0],color=COLORS[identity],label=TEXT[identity]);ax.fill_between(range(15),v[:,1],v[:,2],color=COLORS[identity],alpha=.12)
                    ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.15);ax.set_title(noise);ax.set_ylabel(label)
                    if ri==2:ax.set_xlabel('Target timestep')
            axs[0,0].legend(fontsize=8);fig.suptitle(f'{dataset.upper()} | {LABELS[carrier]} | {layer.upper()} full flattened output\nWithin-carrier effects; marginal descriptive 95% concept-bootstrap intervals');fig.tight_layout(rect=(0,0,1,.94));paths.append(savefig(fig,f'{dataset}_{carrier}_{layer}_sharpening.png'))
    return paths

def stimuli_figures(dataset):
    m=json.loads((OUT/dataset/'manifest.json').read_text(encoding='utf8'));trial=m['trials'][0];index=trial['dataset_index'];mean=np.array([.485,.456,.406]);std=np.array([.229,.224,.225])
    with np.load(OUT/dataset/'formal'/f'stimuli_{index}.npz') as z:
        fig,axs=plt.subplots(2,4,figsize=(13,6))
        keys=['exact_preview0','exact_preview1','target_0','target_2721']
        for ax,key in zip(axs[0],keys):ax.imshow(np.clip(z[key].transpose(1,2,0)*std+mean,0,1));ax.set_title(key);ax.axis('off')
        prime_x0=(5,155)[trial['slot']%2]
        crops=[z[keys[0]][:,80:144,5:69],z[keys[1]][:,80:144,prime_x0:prime_x0+64]]+[z[key][:,80:144,80:144] for key in keys[2:]]
        for ax,value,key in zip(axs[1],crops,keys):ax.imshow(np.clip(value.transpose(1,2,0)*std+mean,0,1),interpolation='nearest');ax.set_title('64 px crop: '+key);ax.axis('off')
        fig.suptitle(f'{dataset.upper()} | First prespecified target {index}; RGB clipped for display only');fig.tight_layout();return savefig(fig,f'{dataset}_actual_inputs.png')

def legacy_activation_figures(dataset):
    rows=read(DOC/dataset/'representation_effects.csv');paths=[]
    for carrier in CARRIERS:
        fig,axs=plt.subplots(4,2,figsize=(12,12),sharex=True)
        for ri,layer in enumerate(LAYERS):
            for col,noise in enumerate(('clean','noisy')):
                ax=axs[ri,col]
                for identity in IDENTITIES[:-1]:
                    rs=[r for r in rows if r['carrier']==carrier and r['layer']==layer and r['noise']==noise and r['identity']==identity and r['metric']=='legacy_x_tmp_mean_effect_pct'];rs.sort(key=lambda r:int(r['timestep']))
                    v=np.array([[float(r[k]) for k in ('value','low','high')] for r in rs]);ax.plot(range(15),v[:,0],color=COLORS[identity],label=TEXT[identity]);ax.fill_between(range(15),v[:,1],v[:,2],color=COLORS[identity],alpha=.12)
                ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.15);ax.set_title(f'{layer.upper()} | {noise}');ax.set_ylabel('Legacy x_tmp mean change (%)')
                if ri==3:ax.set_xlabel('Target timestep')
        axs[0,0].legend(fontsize=8);fig.suptitle(f'{dataset.upper()} | {LABELS[carrier]} | Legacy x_tmp activation vs different category\nMarginal descriptive 95% concept-bootstrap intervals (10,000)');fig.tight_layout(rect=(0,0,1,.94));paths.append(savefig(fig,f'{dataset}_{carrier}_legacy_activation.png'))
    return paths

def filter_figure():
    with np.load(OUT/'filter.npz') as z:k=z['kernel'];freq=z['frequency'];db=z['response_db']
    fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].plot(np.arange(len(k))-len(k)//2,k);axs[0].set_xlabel('Input object pixels');axs[0].set_title('153-tap symmetric Kaiser FIR')
    axs[1].plot(freq,db);axs[1].axvline(.8/7,color='green',ls='--',label='Pass edge');axs[1].axvline(1/7,color='red',ls='--',label='New Nyquist / stop edge');axs[1].axhline(-60,color='black',ls=':');axs[1].set_ylim(-120,2);axs[1].set_xlim(0,.3);axs[1].set_xlabel('Cycles per input object pixel');axs[1].set_ylabel('Amplitude (dB)');axs[1].legend(fontsize=8)
    fig.tight_layout();return savefig(fig,'filter_response.png')

def main():
    audit=json.loads((DOC/'final_audit.json').read_text(encoding='utf8'));unit=json.loads((DOC/'unit_acceptance.json').read_text(encoding='utf8'))
    figure=filter_figure();generated={}
    for dataset in ('imagenet','things'):
        generated[dataset]=[stimuli_figures(dataset),*decoding_figures(dataset),*representation_figures(dataset),*legacy_activation_figures(dataset)]
    lines=['# 外周退化、抗混叠与 clean/noisy 交叉验证实验结果','',
        '按2026-10-27修订Markdown计划执行。三种记忆分别以同载体的 different-category 为 non-priming 基线；不进行载体之间的比较、排名或协同检验。',
        '',f'已完成 **{audit["sequences"]:,}条正式轨迹、{audit["timepoints"]:,}个target时间点和{audit["svm_fits"]:,}次SVM拟合**。原始数据审计通过；{unit["tests"]}项测试通过。',
        '', '预定t9共同读出的精确重复accuracy变化较小，ROC-AUC变化方向也不完全一致。当前结果提供各载体内部的描述性priming估计，不能把个别正差值写成已经确认的识别优势；其他内容和完整时程按预定口径保留。',
        '', '## 解释边界','',
        '本轮复用此前已查看的每数据集64张目标（32概念，每概念2图），属于修订输入协议下的探索性重复测量研究。模型预训练未改变。结果不能作为全新独立确认，也不能把旧新版差异归因于单个滤波因素。',
        '', '下列分类差值均为描述性CV估计：不把五折当五次独立实验，不用折SD/√5表示严格标准误，未进行包含分类器重新拟合的置换/概念bootstrap显著性检验。表征区间为10,000次按类别分层的概念簇bootstrap边际95%区间，未作同时校正。',
        '', '## 输入与划分','',
        '两数据集统一动物、服饰、食物、车辆四类；每折测试目标数为16/16/16/8/8。先固定目标概念折，再将所有预览概念分配到同一折；补入8个preview-only概念以保证各折每类至少两个概念。训练/测试折之间目标、prime、背景distractor的概念和图片字节重叠均为0，其他目标图不作预览。',
        '', 'THINGS沿用此前严格WordNet映射、已知ImageNet来源排除及去重后的图像库存。来源不明不等于已经证明来源独立；结论限于这组匹配概念的四类样本，不是全数据集benchmark。',
        '', '中央target：VGG官方确定性PIL抗混叠resize与center crop→224标准化物体→clean或σ=0.5加性噪声→153-tap对称Kaiser FIR→对齐像素中心的双线性224→64降采样。外周preview另加按通道均值的对比度0.5，以及降采样前σ=3.5（最终像素σ=1）的高斯平滑，再经过相同FIR与采样。背景RGB0.5，画布224，左右中心约37/187，中央112。',
        '', 'FIR通带边缘0.8/7、阻带起点1/7 cycles/input-pixel；实测通带波动0.00534 dB、阻带衰减69.30 dB。反射边界；有限核按容差验收，不称理想无限带限。正弦光栅的AC衰减和棋盘抗混叠、均值与对比度公式均通过；预览DC均值与振荡AC分开检查。',
        '',img(figure,'滤波核与频响'),
        '', '只用22步共同灰屏，五层完整rep/prd/grd/prediction_error快照持久化并深拷贝复用，复现原灰屏摘要。Hebbian coeff=.05、decay=.5、inh=4、稀疏阈值.9，PCoder4/5注入；每段preview末各更新一次，target冻结W。FC动态更新关闭。clean只提取一次，noisy三种种子，clean/noisy共享全部preview。',
        '', '### 最终目标噪声强度','', '| 数据集 | 过滤后物体内SD均值 | RMSE均值 | SNR均值(dB) |', '|---|---:|---:|---:|']
    for dataset in ('imagenet','things'):
        ns=[r for r in read(DOC/dataset/'input_noise.csv') if int(r['seed'])!=0 and r['condition']=='hebb_exact']
        lines.append(f'| {dataset} | {np.mean([float(r["sd"]) for r in ns]):.4f} | {np.mean([float(r["rmse"]) for r in ns]):.4f} | {np.mean([float(r["snr_db"]) for r in ns]):.2f} |')
    lines+=['', 'σ=0.5定义在低通前的标准化224物体上；低通后不重新放大噪声。SNR以标准化clean物体的均方值为信号功率，不等于原始RGB感知SNR。',
        '', '## t9共同读出与逐时刻MVPA','',
        't9：每数据集/噪声/折训练一个共同FC2头，仅用该折训练目标的t9，12条件等权，随后冻结并评估t0–t14；非t9是时间泛化。MVPA：各条件、各时刻分别训练。均为训练折StandardScaler+LinearSVC(C=1,dual=auto)，无性能驱动调参；仅允许预定的收敛迭代上限重试。',
        '', '两套读出共享同一组概念折。noisy每种实现权重1/3、clean权重1；t9再以每条件1/12保证每个训练目标总权重为1（标量权重同时用于Scaler和SVM）。主要报告五折算术均值及折SD，另外保存按测试目标数加权的OOF accuracy。ROC-AUC逐折按四类OVR decision score计算后macro平均，不混合不同折的原始margin计算pooled AUC。',
        '', '### 固定t9的内容效应','',
        '以下均为同载体内prime−different-category，先在各折计算再平均。Δaccuracy单位为百分点；ΔROC-AUC为0–1尺度。括号为折SD，不是置信区间。',
        '', '| 数据集 | target | 读出 | 载体 | 内容 | Δaccuracy pp（fold SD） | ΔROC-AUC（fold SD） |', '|---|---|---|---|---|---:|---:|']
    for dataset in ('imagenet','things'):
        rows=read(DOC/dataset/'cv_window_contrasts.csv')
        for r in rows:
            if r['window']!='t9':continue
            lines.append(f'| {dataset} | {r["noise"]} | {r["readout"]} | {LABELS[r["carrier"]]} | {TEXT[r["identity"]]} | {float(r["delta_accuracy_pp"]):+.2f} ({float(r["delta_accuracy_fold_sd"]):.2f}) | {float(r["delta_roc_auc"]):+.4f} ({float(r["delta_auc_fold_sd"]):.4f}) |')
    lines+=['', 'clean/noisy分别拟合边界，所以两组原始accuracy差异不是同一分类器的纯抗噪差异。完整15时刻、early/middle/late/full、固定t9、逐折/逐噪声/逐类AUC均在原表；归一化梯形时间积分另命名，端点半权重，不能与15点均值混写。']
    for dataset in ('imagenet','things'):
        lines+=['',f'### {dataset.upper()} 全时间曲线','',img(DOC/'figures'/f'{dataset}_t9_accuracy.png','t9共同读出accuracy'),'',img(DOC/'figures'/f'{dataset}_t9_roc_auc.png','t9共同读出ROC-AUC'),'',img(DOC/'figures'/f'{dataset}_mvpa_accuracy.png','逐时刻MVPA accuracy'),'',img(DOC/'figures'/f'{dataset}_mvpa_roc_auc.png','逐时刻MVPA ROC-AUC')]
        lines+=['',img(DOC/'figures'/f'{dataset}_t9_content_effects.png','t9读出accuracy内容效应'),'',img(DOC/'figures'/f'{dataset}_t9_auc_content_effects.png','t9读出ROC-AUC内容效应')]
    lines+=['', '## 四层激活与sharpening','',
        '逐时间点记录conv4、conv5、fc1、fc2的模块输出与历史x_tmp：mean、mean(abs)、非零比例、max、25/50/75分位数。对比使用条件均值之比，baseline接近零记NA。conv4/conv5分别完整展平401408/100352维，FC层4096维；没有用GAP向量替代sharpening。',
        '', '每概念保留n=2、Σx、Σu、Σ||x||²、Σ||u||²及零向量数；标量Σ||x||²是向量Σx²对特征维求和，对所需trace指标代数充分。原始FP32字节经无损shuffle+ZIP保存；两图累加精度与FP64直接计算在合成验收中一致到规定容差。Gram矩阵以FP64计算，bootstrap重新构造类别质心及所有几何指标。',
        '', 'raw trace variance降低可能仅是幅度下降；normalized trace降低、within-class cosine升高及normalized Fisher升高需结合解释。normalized trace与within cosine存在代数联系，不能作为两份独立证据。四类定义的几何量不与旧200类绝对值直接比较。noisy先逐种子计算几何再平均，未把同图三噪声当独立类别实例。',
        '', 'complete sharpening保留旧组合逻辑：legacy激活下降、normalized trace下降、normalized Fisher上升；另报告output版本。它是描述性方向组合，不是显著性检验，也不意味着分类性能改善。',
        '', '### 输出与历史激活的数值核对','', '| 数据集 | 层 | 所有轨迹最小output | output mean与legacy x_tmp mean最大绝对差 |', '|---|---|---:|---:|']
    for dataset in ('imagenet','things'):
        info=audit['datasets'][dataset]
        for layer in LAYERS:lines.append(f'| {dataset} | {layer} | {info["output_min"][layer]:.6g} | {info["max_output_vs_legacy_mean_difference"][layer]:.6g} |')
    lines+=['', '### 固定t9、FC2的表征效应','',
        '仅列出预定t9与固定读出层FC2用于紧凑展示，四层全部时刻完整保存并绘图。区间是概念簇bootstrap边际95%区间。',
        '', '| 数据集 | target | 载体 | 内容 | output激活% [CI] | legacy x_tmp激活% [CI] | normalized trace% [CI] | normalized Fisher% [CI] |', '|---|---|---|---|---|---|---|---|']
    for dataset in ('imagenet','things'):
        rows=read(DOC/dataset/'representation_effects.csv')
        for noise in ('clean','noisy'):
            for carrier in CARRIERS:
                for identity in IDENTITIES[:-1]:
                    values=[]
                    for metric in ('output_mean_effect_pct','legacy_x_tmp_mean_effect_pct','normalized_trace_effect_pct','normalized_fisher_effect_pct'):
                        r=next(r for r in rows if r['noise']==noise and r['carrier']==carrier and r['identity']==identity and r['layer']=='fc2' and r['timestep']=='9' and r['metric']==metric)
                        values.append(f'{float(r["value"]):+.2f} [{float(r["low"]):+.2f}, {float(r["high"]):+.2f}]')
                    lines.append(f'| {dataset} | {noise} | {LABELS[carrier]} | {TEXT[identity]} | '+ ' | '.join(values)+' |')
    lines+=['', '## 文件与复现','',link(DOC/'预先方案.md','冻结计划副本')+'；'+link(DOC/'inventory_acceptance.json','库存/隔离审计')+'；'+link(DOC/'input_acceptance.json','输入验收')+'；'+link(DOC/'gray_acceptance.json','灰屏验收')+'；'+link(DOC/'unit_acceptance.json','测试')+'；'+link(DOC/'final_audit.json','最终审计')+'。']
    for dataset in ('imagenet','things'):
        lines+=['',f'### {dataset.upper()} 文件','',img(DOC/'figures'/f'{dataset}_actual_inputs.png','实际输入'),'',
            '；'.join(link(DOC/dataset/name,label) for name,label in [('cv_summary.csv','CV均值/SD/OOF accuracy'),('cv_fold_metrics.csv','逐折/逐类AUC'),('cv_noise_metrics.csv','分噪声CV'),('cv_content_contrasts.csv','全时间内容效应'),('cv_window_contrasts.csv','窗口内容效应'),('temporal_integrals.csv','时间积分'),('activation_per_image.csv','逐图双路径激活'),('activation_timeline.csv','激活汇总'),('geometry_timeline.csv','完整几何'),('geometry_fold_diagnostics.csv','折内几何'),('representation_effects.csv','表征效应及CI'),('representation_window_effects.csv','窗口表征效应及CI'),('input_noise.csv','实际目标噪声'),('final_preview_contrast.csv','实际preview对比度')])+'。',
            '', '全部英文图：']
        lines+=['',*['- '+link(p,p.name) for p in generated[dataset]]]
    lines+=['', '原始实际刺激、轨迹、概念充分统计、诊断完整输出、20个t9头、3600个MVPA模型、OOF分数、折与权重、源码/权重/环境哈希见 '+link(OUT,'原始实验目录')+'。',
        '', '复现/续跑：', '', '```powershell', '& C:/Users/liang/anaconda3/envs/predify2021/python.exe analysis/run_filtered_cv_20261027.py', '```',
        '', '驱动程序跳过已完成阶段，冻结检查拒绝参数或源码变化。若需要新设计，应使用新目录，保留本轮结果。']
    lines+=['',link(DOC/'图表索引.md','更新图表索引')]
    (DOC/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8');write_figure_index();print('REPORT COMPLETE',flush=True)

if __name__=='__main__':main()
