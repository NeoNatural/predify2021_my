"""Four preview identities in a 2x2 grid of FC settings for all measures."""
from pathlib import Path
import json,runpy
import numpy as np

root=Path(__file__).resolve().parents[1]
ctx=runpy.run_path(str(root/'analysis/prepare_hebb_fc_priming_20261103.py'))
base=json.loads((root/'.slides_build/hebb_fc_full_20261103_v3/slides-ready.json').read_text(encoding='utf8'))
gi=ctx['ctx']['gi'];panel=ctx['panel'];series=ctx['series'];ids=ctx['identities'];methods=ctx['methods'];carriers=ctx['carriers']
slides=base['slides'][:37]
metrics=[('activation','Mean activation','Mean output'),('normalized_within','Within-class dispersion','Normalized within'),
         ('cosine','Within-class cosine similarity','Within cosine'),('normalized_between','Between-class dispersion','Normalized between'),
         ('normalized_fisher','Between-to-within separation','Between / within')]
for carrier,clabel in carriers:
 for layer in ('fc1','fc2'):
  for field,title,label in metrics:
   for noise in ('clean','noisy'):
    for dataset in ('imagenet','things'):
     seeds=(0,) if noise=='clean' else (2921,2922,2923);panels=[]
     settings=methods if field!='activation' else methods[:-1]
     for method,mlabel in settings:
      ss=[]
      for identity,ilabel in ids:
       values=[]
       for t in range(15):
        vals=[]
        for fold in range(5):
         for seed in seeds:
          r=gi[dataset,fold,method,layer,carrier+'_'+identity,seed,t]
          if field=='cosine':
           assert int(r['n_images'])//(4 if dataset=='imagenet' else 5)==16
           vals.append(1.-16./15.*float(r['normalized_within']))
          else:vals.append(float(r[field]))
        values.append(np.mean(vals))
       ss.append(series(ilabel,values,'fold_geometry_timeline.csv'))
      p=panel(dataset,ss,label,compress=False);p['title']=mlabel;panels.append(p)
     shared=panel(dataset,[v for p in panels for v in p['series']],label,compress=False)
     for p in panels:
      for key in ('axis','min','max'):p[key]=shared[key]
     foot='FC panels share one y-axis scale. Curves show absolute values; no confidence intervals.'
     if field=='activation':foot='Matched scale omitted: its mean matches Hebb-FC. FC panels share one y-axis scale.'
     slides.append(dict(title=f'{title}: {layer.upper()}',sub=f'{dataset.upper()}, {clabel}, {noise} central target',
      foot=foot,note='每个面板对应一种FC设置，四条线分别为Exact、Same concept、Same category、Different category，所有面板共享纵轴。每个点为每折测试数据几何指标在五折及适用噪声实现上的算术均值。类内与类间均基于单位归一化样本，余弦从每类16图的类内离散度推导。Fisher是类间与类内比值。此图无显著性检验与置信区间。平均激活为原始输出均值，不作归一化。',
      seconds=25,target_noise=noise,panels=panels))
assert len(slides)==157
out=root/'.slides_build/hebb_fc_full_20261103_v4';out.mkdir(parents=True,exist_ok=True)
(out/'slides-ready.json').write_text(json.dumps({'slides':slides,'sources':base['sources']},ensure_ascii=False),encoding='utf8')
print('Prepared',len(slides),'slides;',sum(len(s['panels']) for s in slides),'panels')
