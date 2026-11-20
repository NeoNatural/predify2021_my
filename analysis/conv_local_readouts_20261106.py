"""Re-tabulate existing Conv4/Conv5 outcomes; FC2 is an auxiliary accuracy check."""
from pathlib import Path
import hashlib,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/conv_fc_transfer_20261106'
OUT=ROOT/'docs/conv_local_readouts_20261106'
NAMES={'direct_rep4':'原通道·rep4','direct_prd5':'原通道·prd5','sae_rep4':'SAE·rep4','sae_prd5':'SAE·prd5'}


def table(frame):
    q=frame.copy()
    for c in q.select_dtypes(include='number'):q[c]=q[c].map(lambda x:f'{x:.4f}')
    return '| '+' | '.join(q.columns)+' |\n| '+' | '.join(['---']*len(q.columns))+' |\n'+'\n'.join(
        '| '+' | '.join(map(str,row))+' |' for row in q.itertuples(index=False,name=None))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    w=pd.read_csv(SOURCE/'window_metrics.csv');h=pd.read_csv(SOURCE/'history_contrasts.csv')
    a=pd.read_csv(SOURCE/'fixed_svm.csv');l=pd.read_csv(SOURCE/'local_input_output.csv')
    local=[];repetition=[];acc=[]
    for ds in ('imagenet','things'):
        for win in ('early','middle','late'):
            q=w[(w.dataset==ds)&(w.window==win)]
            for stem in NAMES:
                case=stem+'_full';z=q[q.history=='exact']
                o=z[(z.case==case)&(z.layer=='conv4')].iloc[0]
                incoming=z[(z.case==case)&(z.layer=='conv4_input')].iloc[0]
                record=dict(dataset=ds,case=case,window=win,
                    local_suppression_pct=float(l[(l.dataset==ds)&(l.case==case)&(l.history=='exact')&(l.window==win)].local_suppression_pct.iloc[0]),
                    within_vs_input_pct=100*(o.within/incoming.within-1),
                    between_vs_input_pct=100*(o['between']/incoming['between']-1),
                    fisher_vs_input_pct=100*(o.fisher/incoming.fisher-1),
                    input_within=incoming.within,input_between=incoming['between'],input_fisher=incoming.fisher,
                    output_within=o.within,output_between=o['between'],output_fisher=o.fisher)
                for tag,control in [('pool',stem+'_pool'),('scale',stem+'_scale'),('clip','clip')]:
                    c=z[(z.case==control)&(z.layer=='conv4')].iloc[0]
                    for metric in ('within','between','fisher'):
                        record[f'{metric}_vs_{tag}_pct']=100*(o[metric]/c[metric]-1)
                local.append(record)
            for case in q.case.unique():
                for layer in ('conv4','conv5'):
                    hh=h[(h.dataset==ds)&(h.case==case)&(h.window==win)&(h.layer==layer)].iloc[0]
                    e=q[(q.case==case)&(q.layer==layer)&(q.history=='exact')].iloc[0]
                    d=q[(q.case==case)&(q.layer==layer)&(q.history=='different_category')].iloc[0]
                    repetition.append(dict(dataset=ds,case=case,window=win,layer=layer,
                        suppression_ED_pct=hh.suppression_ED,within_ED_pct=hh.within_ED_pct,
                        between_ED_pct=100*(e['between']/d['between']-1),fisher_ED_pct=hh.fisher_ED_pct,
                        suppression_EN_pct=hh.suppression_EN,fisher_EN_pct=hh.fisher_EN_pct,
                        exact_mean=e['mean'],different_mean=d['mean'],exact_fisher=e.fisher,different_fisher=d.fisher))
            for case in q.case.unique():
                for hist in ('exact','different_category')+ (('none',) if case=='clip' else ()):
                    v=a[(a.dataset==ds)&(a.window==win)&(a.case==case)&(a.history==hist)&(a.layer=='fc2')].iloc[0]
                    acc.append(dict(dataset=ds,case=case,window=win,history=hist,
                        time_average_accuracy_pct=100*v.time_average_accuracy,
                        window_mean_score_BA_pct=100*v.window_mean_score_BA))
    local=pd.DataFrame(local);repetition=pd.DataFrame(repetition);acc=pd.DataFrame(acc)
    for name,frame in [('conv4_local_exact_all_windows',local),('conv4_conv5_repetition_all_windows',repetition),('fc2_accuracy_all_windows',acc)]:
        frame.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    first=local[local.window=='early'].copy()
    first['数据']=first.dataset.map({'imagenet':'ImageNet','things':'THINGS'})
    first['方案']=first.case.str.removesuffix('_full').map(NAMES)
    first=first.rename(columns={'local_suppression_pct':'局部抑制%', 'within_vs_input_pct':'Δ类内%','between_vs_input_pct':'Δ类间%',
        'fisher_vs_input_pct':'ΔFisher对输入%','fisher_vs_pool_pct':'ΔFisher对pool%'})
    first=first[['数据','方案','局部抑制%','Δ类内%','Δ类间%','ΔFisher对输入%','ΔFisher对pool%']]
    second=[]
    for ds in ('imagenet','things'):
        for stem in NAMES:
            q=repetition[(repetition.dataset==ds)&(repetition.case==stem+'_full')&(repetition.window=='early')].set_index('layer')
            second.append({'数据':'ImageNet' if ds=='imagenet' else 'THINGS','方案':NAMES[stem],
                'Conv4重复抑制%':q.loc['conv4','suppression_ED_pct'],'Conv4 ΔFisher E/D%':q.loc['conv4','fisher_ED_pct'],
                'Conv5重复抑制%':q.loc['conv5','suppression_ED_pct'],'Conv5 ΔFisher E/D%':q.loc['conv5','fisher_ED_pct']})
    second=pd.DataFrame(second)
    third=[]
    for case,name in [('clip','无Hebbian＋preview')]+[(k+'_full',v) for k,v in NAMES.items()]:
        row={'方案':name}
        for ds,label in [('imagenet','ImageNet'),('things','THINGS')]:
            for hist,hh in [('exact','Exact'),('different_category','Different')]:
                v=acc[(acc.dataset==ds)&(acc.case==case)&(acc.window=='early')&(acc.history==hist)].iloc[0]
                row[f'{label} {hh}%']=v.time_average_accuracy_pct
        third.append(row)
    third=pd.DataFrame(third)
    text=['# 以实际干预附近层为主的结果重评',
        '本轮仅重排已完成实验数据；没有新条件、模型训练或参数搜索。主要评价层改为实际受干预的Conv4（PCoder4.rep），Conv5作为相邻传播层；最终FC2仅作分类准确率辅助检查。',
        '## 1. Conv4算子本身：Exact输出相对它自己的实际输入',
        '下表为early=t0–4，均为full。局部抑制是逐target计算100×(1−输出均值/输入均值)再平均；正数表示减活动。'
        '几何在完整卷积图L2归一化后逐网络步计算，随后按窗口平均；Δ几何为两个窗口指标的相对百分比变化。'
        'Δ类内小于0、Δ类间大于0、ΔFisher大于0分别为对应改善方向。最后一列换用各自完整pool序列作参照，不能与“对输入”混为一谈。',
        table(first),
        '局部减活动已经发生。相对实际输入，八个full的Fisher点估计均略降，原通道比SAE接近保真；原通道rep4在ImageNet仅−0.0024%，不能把这种微小点差称为可靠损伤。'
        '相对pool，Conv4却有5/8个组合的Fisher略升（ImageNet四组及THINGS SAE/rep4）。因此此前“8/8低于pool”是FC2结果，不能外推成局部Conv4全劣。'
        'full与pool具有不同的闭环输入/preview2权重，这列是完整机制比较，不是同W同输入的一次局部增益因果效应。',
        '## 2. Repetition：同一层Exact相对Different',
        '重复抑制=逐target平均100×(1−Exact均值/Different均值)，正数才是suppression；ΔFisher E/D为Exact相对Different的Fisher百分比变化。'
        '这与第一表“输出相对实际输入”是不同问题。',table(second),
        '早期Conv4只有THINGS原通道rep4的重复抑制点估计微正（0.0051%），八个full的概念块区间都跨零；Conv5八个组合均为Exact响应更高。'
        'ImageNet Conv4的Exact Fisher略高于Different，但对应类内散度均增加约0.71%–0.88%；THINGS Conv4的Exact Fisher均更低。'
        '这些局部层结果尚不支持稳定的repetition suppression与类别sharpening联合收益；该判断不需要以FC2 Fisher作为主证据。',
        '## 3. 辅助检查：最终FC2固定readout准确率',
        '下表为t0–4每步准确率再平均，单位%；不是将窗口平均决策分数再分类。分类头仅用训练folds1–3的原FC响应拟合，读取实际4096维FC2。',table(third),
        '无preview（灰屏时长匹配）的早期FC2时间平均准确率为ImageNet73.75%、THINGS71.00%。'
        '相对同历史无Hebbian的Exact，full在ImageNet三组不变、一组−1.25个百分点，THINGS均+1个百分点。'
        '这仅作功能兼容性辅助判断，不用准确率不变替代局部sharpening。',
        '## 全部窗口与原始量',
        '- conv4_local_exact_all_windows.csv：full在early/middle/late对输入、pool、clip、scale的全部几何变化，同时保留原始Within/Between/Fisher。',
        '- conv4_conv5_repetition_all_windows.csv：full、pool、shuffle、scale、clip在两层、三个窗口的重复效应与原始均值/Fisher。',
        '- fc2_accuracy_all_windows.csv：所有方法、历史和窗口的最终准确率；另保留窗口平均分数的BA并清楚命名。',
        '窗口固定为early t0–4、middle t5–9、late t10–14。全部为既有16/20目标开发批量的描述性点估计；时间步不作为独立样本。'
        '后续以Conv4为主要机制验收层、Conv5为传播辅助层；FC2以分类保真为主，独立列示。']
    (OUT/'局部层主指标与最终准确率.md').write_text('\n\n'.join(text),encoding='utf8')
    audit=dict(analysis_only=True,new_experiments=0,main_layers=['conv4','conv5'],auxiliary='FC2 accuracy',
        source_sha256={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in ['window_metrics.csv','history_contrasts.csv','fixed_svm.csv','local_input_output.csv']},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(table(first));print(table(second));print(table(third))


if __name__=='__main__':main()
