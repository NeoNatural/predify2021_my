"""Show all preview identities separately for each FC condition and dataset."""
from pathlib import Path
import json,runpy

ctx=runpy.run_path(str(Path(__file__).with_name('prepare_hebb_fc_figures_20261103.py')))
ri=ctx['ri'];series=ctx['series'];panel=ctx['panel'];methods=ctx['methods'];carriers=ctx['carriers'];headings=ctx['headings']
ctx['colors']['Same category']=ctx['colors']['Same class'];ctx['colors']['Different category']=ctx['colors']['Other class']
identities=[('exact','Exact'),('same_concept','Same concept'),('same_category','Same category'),('different_category','Different category')]
ROOT=ctx['ROOT'];base=json.loads((ctx['BUILD']/'slides-ready.json').read_text(encoding='utf8'))
slides=[base['slides'][0]]
for kind in ('mvpa','t9_mode_specific','t9_clean_reference'):
 for carrier,clabel in carriers:
  for noise in ('clean','noisy'):
   for ds in ('imagenet','things'):
    panels=[]
    for method,mlabel in methods:
     ss=[]
     for ident,label in identities:
      vals=[float(ri[ds,method,noise,f'{carrier}_{ident}',kind,t]['accuracy_pct']) for t in range(15)]
      ss.append(series(label,vals,'readout_timeline.csv'))
     p=panel(ds,ss,'Accuracy (%)',25. if ds=='imagenet' else 20.,(0.,100.));p['title']=mlabel;panels.append(p)
    # Each of the four methods uses the same transform so heights are comparable.
    all_curves=[s for p in panels for s in p['series'] if s['name']!='Chance']
    shared=panel(ds,all_curves,'Accuracy (%)',25. if ds=='imagenet' else 20.,(0.,100.))
    for p in panels:
     for key in ('axis','min','max'):p[key]=shared[key]
    slides.append(dict(title=f'{headings[kind]}: {clabel} content trajectories',
       sub=f'{ds.upper()}, {noise} central target; FC settings shown side by side',
       foot='Four FC panels share one y-axis scale. Different-category is the preview control.',
       note='本页每个面板分别为Off、Hebb-FC、Inhibition和Matched scale；每个面板均包含Exact、Same concept、Same category、Different category四种预览，五折算术均值。每个FC条件单独拟合读出；MVPA还按条件和时刻分别拟合。灰色段为坐标压缩区域，刻度是真实值，显示斜率不能直接比较。无显著性检验。',
       seconds=25,target_noise=noise,panels=panels))
slides.extend(base['slides'][37:])
assert len(slides)==97 and len(slides[1]['panels'])==4 and len(slides[37]['panels'])==2
dest=ROOT/'.slides_build/hebb_fc_full_20261103_v3';dest.mkdir(parents=True,exist_ok=True)
(dest/'slides-ready.json').write_text(json.dumps(dict(slides=slides,sources=base['sources']),ensure_ascii=False),encoding='utf8')
print('Prepared',len(slides),'slides,',36,'four-panel content pages')
