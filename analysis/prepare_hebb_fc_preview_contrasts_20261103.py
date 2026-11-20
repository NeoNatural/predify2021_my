"""Within-FC preview contrasts for activation and representation panels."""
from pathlib import Path
import json,runpy
import numpy as np

root=Path(__file__).resolve().parents[1]
ctx=runpy.run_path(str(root/'analysis/prepare_hebb_fc_priming_grid_20261103.py'))
base=json.loads((root/'.slides_build/hebb_fc_full_20261103_v4/slides-ready.json').read_text(encoding='utf8'))
helper=ctx['ctx'];panel=helper['panel'];series=helper['series']
slides=base['slides'];converted=0
for slide in slides[37:]:
 dataset='imagenet' if slide['sub'].startswith('IMAGENET') else 'things'
 title=slide['title'];cosine=title.startswith('Within-class cosine')
 if title.startswith('Mean activation'):ylabel='Mean output difference'
 elif title.startswith('Within-class dispersion'):ylabel='Within difference'
 elif cosine:ylabel='Cosine difference (x1000)'
 elif title.startswith('Between-class dispersion'):ylabel='Between difference'
 elif title.startswith('Between-to-within separation'):ylabel='Fisher difference'
 else:raise AssertionError(title)
 panels=[]
 for old in slide['panels']:
  reference=old['series'][3]['values'];assert old['series'][3]['name']=='Different category'
  curves=[]
  for item in old['series'][:3]:
   vals=[(a-b)*(1000 if cosine else 1) for a,b in zip(item['values'],reference)]
   curves.append(series(item['name'],vals,'fold_geometry_timeline.csv'))
  p=panel(dataset,curves,ylabel,ref=0.,compress=True);p['title']=old['title'];panels.append(p)
 shared=panel(dataset,[v for p in panels for v in p['series'] if v['name']!='Zero'],ylabel,ref=0.,compress=True)
 # Two rows leave less vertical space: keep true-value ticks far enough apart.
 axis=shared['axis'];knots=axis['knots'];mapped=axis['mapped'];slopes=axis['slopes']
 def ymap(y):
  k=int(np.clip(np.searchsorted(knots,y,side='right')-1,0,len(slopes)-1))
  return mapped[k]+(y-knots[k])*slopes[k]
 kept=[];candidates=sorted(axis['ticks'],key=lambda x:(abs(x)>1e-12,abs(x)))
 for tick in candidates:
  if all(abs(ymap(tick)-ymap(other))>=.12*(mapped[-1]-mapped[0]) for other in kept):kept.append(tick)
 axis['ticks']=sorted(kept)
 for p in panels:
  for key in ('axis','min','max'):p[key]=shared[key]
 slide['panels']=panels
 slide['foot']='Each curve minus different-category within the same FC setting. Panels share one y-axis.'
 if title.startswith('Mean activation'):slide['foot']='Matched scale omitted. Each curve minus different-category; panels share one y-axis.'
 slide['note']='相同数据集、carrier、FC设置、层、噪声和时间点内，各内容条件减去different-category；Zero为后者减自身。差值为五折及适用噪声实现上的配对均值。余弦差乘1000；其他指标均为原始单位差，不是百分比。归一化类内散度、类间散度、Fisher和余弦基于每折测试集；没有显著性检验或置信区间。均值匹配对照在mean activation省略。'
 converted+=1
assert converted==120 and len(slides)==157
out=root/'.slides_build/hebb_fc_full_20261103_v5';out.mkdir(parents=True,exist_ok=True)
(out/'slides-ready.json').write_text(json.dumps(base,ensure_ascii=False),encoding='utf8')
print('Prepared',len(slides),'slides; preview-contrast representation pages',converted)
