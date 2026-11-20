"""Prepare slide chart data from audited five-fold Hebb-FC outputs."""
from pathlib import Path
from collections import defaultdict
import ast,csv,json,hashlib
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/hebb_fc_full_20261103';BUILD=ROOT/'.slides_build/hebb_fc_full_20261103';BUILD.mkdir(parents=True,exist_ok=True)
tree=ast.parse((ROOT/'analysis/plot_adaptive_20261028.py').read_text(encoding='utf8'))
scope={'np':np};exec(compile(ast.Module(body=[v for v in tree.body if isinstance(v,(ast.FunctionDef,ast.ClassDef)) and v.name in ('bounds','Axis')],type_ignores=[]),'axis','exec'),scope)
Axis=scope['Axis']
def read(name):
 with (DOC/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
readouts=read('readout_timeline.csv');geometry=read('fold_geometry_timeline.csv')
ri={(r['dataset'],r['method'],r['noise'],r['condition'],r['readout'],int(r['timestep'])):r for r in readouts}
gi={(r['dataset'],int(r['fold']),r['method'],r['layer'],r['condition'],int(r['seed']),int(r['timestep'])):r for r in geometry}
colors={'Off':'#111111','Hebb-FC':'#237df5','Inhibition':'#52c6ed','Matched scale':'#d66d2c','Zero':'#aaaaaa','Chance':'#aaaaaa',
        'Exact':'#111111','Same concept':'#52c6ed','Same class':'#237df5','Other class':'#d66d2c'}
methods=[('off','Off'),('mean_std','Hebb-FC'),('inhibition_only','Inhibition'),('scale_both','Matched scale')]
identities=[('exact','Exact'),('same_concept','Same concept'),('same_category','Same class'),('different_category','Other class')]
carriers=[('hebb','Hebbian-only'),('pcoder','PCoder carry-only'),('joint','Joint')]
headings={'mvpa':'MVPA','t9_mode_specific':'t9-A','t9_clean_reference':'t9-B'}
def series(name,vals,source):
 assert len(vals)==15 and np.isfinite(vals).all()
 return dict(name=name,xValues=list(range(15)),values=[float(v) for v in vals],color=colors[name],source={'path':str(DOC/source)})
def panel(dataset,ss,y,ref=None,physical=None,compress=True):
 values=np.array([v for s in ss for v in s['values']]);a=Axis(values,ref,physical,compress)
 if ref is not None:
  nm='Chance' if physical else 'Zero';ss=ss+[series(nm,[ref]*15,'readout_timeline.csv')]
 cand=[]
 if ref is not None and a.lo<=ref<=a.hi:cand.append(ref)
 cand+=list(MaxNLocator(nbins=5).tick_values(*a.focus))+[a.lo,a.hi]
 for c in a.compressed:cand+=list(MaxNLocator(nbins=3).tick_values(c['low'],c['high']))
 ticks=[];gap=.075*(a.mapped[-1]-a.mapped[0])
 for v in cand:
  if a.lo<=v<=a.hi and not any(abs(a.forward(v)-a.forward(x))<gap for x in ticks):ticks.append(float(v))
 axis=dict(knots=a.knots.tolist(),mapped=a.mapped.tolist(),slopes=a.slopes.tolist(),ticks=sorted(ticks),compressed_segments=a.compressed)
 return dict(title=dataset.upper(),series=ss,min=a.lo,max=a.hi,yTitle=y,xmax=14,axis=axis)
slides=[]
def slide(title,sub,foot,note,panels):slides.append(dict(title=title,sub=sub,foot=foot,note=note,seconds=25,target_noise=sub,panels=panels))
slide('Predify and Hebb-FC','Concept-grouped cross-validation on clean and noisy targets','ImageNet and THINGS | exploratory follow-up',
 '图源和原始数值见 docs/hebb_fc_full_20261103。每个外层折只用训练概念拟合FC连接与读出。FC在PCoder之后，各条件共用预测编码状态。',[])
for kind in ('mvpa','t9_mode_specific','t9_clean_reference'):
 for carrier,clabel in carriers:
  for noise in ('clean','noisy'):
   panels=[]
   for ds in ('imagenet','things'):
    ss=[]
    for ident,label in identities:
     vals=[float(ri[ds,'mean_std',noise,f'{carrier}_{ident}',kind,t]['accuracy_pct']) for t in range(15)]
     ss.append(series(label,vals,'readout_timeline.csv'))
    panels.append(panel(ds,ss,'Accuracy (%)',25. if ds=='imagenet' else 20.,(0.,100.)))
   slide(f'{headings[kind]}: {clabel} content trajectories',f'{noise.capitalize()} central target; Hebb-FC enabled; four preview contents',
     'Five-fold means; chance is a reference. Axes may compress sparse regions.',
     '本页使用mean_std FC设置；四种预览内容、同一carrier，准确率为各折算术均值。阴影区为压缩段，刻度仍为真实值，视觉斜率不可直接比较。',panels)
   panels=[]
   for ds in ('imagenet','things'):
    ss=[]
    for method,label in methods:
     vals=[float(ri[ds,method,noise,f'{carrier}_exact',kind,t]['accuracy_pct'])-float(ri[ds,method,noise,f'{carrier}_different_category',kind,t]['accuracy_pct']) for t in range(15)]
     ss.append(series(label,vals,'readout_timeline.csv'))
    panels.append(panel(ds,ss,'Exact - different (pp)',0.))
   slide(f'{headings[kind]}: {clabel} preview contrast',f'{noise.capitalize()} central target; exact minus different-category preview',
     'Each curve uses its own fold-trained readout. Zero means no preview contrast.',
     '同一数据集、时间、carrier内，exact准确率减去different-category准确率；四种FC方法分别拟合读出。该值为描述性估计，未作显著性检验。',panels)
metrics=[('activation','Mean activation','Mean output',False),('normalized_within','Within-class dispersion','Within change (%)',True),
 ('cosine','Within-class cosine similarity','Cosine delta (x1000)',False),('normalized_between','Between-class dispersion','Between change (%)',True),('normalized_fisher','Between-to-within separation','Fisher change (%)',True)]
for carrier,clabel in carriers:
 for layer in ('fc1','fc2'):
  for field,title,label,pct in metrics:
   for noise in ('clean','noisy'):
    panels=[]
    for ds in ('imagenet','things'):
     seeds=(0,) if noise=='clean' else (2921,2922,2923);ss=[]
     for method,name in methods:
      if field=='activation' and method=='scale_both':continue
      if field!='activation' and method=='off':continue
      vals=[]
      for t in range(15):
       terms=[]
       for fold in range(5):
        for seed in seeds:
         r=gi[ds,fold,method,layer,carrier+'_exact',seed,t];b=gi[ds,fold,'off',layer,carrier+'_exact',seed,t]
         if field=='cosine':
          # Each held-out category contains four concepts x four targets = 16.
          assert int(r['n_images'])//(4 if ds=='imagenet' else 5)==16
          terms.append(-16/15*(float(r['normalized_within'])-float(b['normalized_within']))*1000)
         elif pct:
          denom=float(b[field]);assert denom>1e-12
          terms.append(100*(float(r[field])/denom-1))
         else:terms.append(float(r[field]))
       vals.append(np.mean(terms))
      ss.append(series(name,vals,'fold_geometry_timeline.csv'))
     panels.append(panel(ds,ss,label,None if field=='activation' else 0.,compress=field!='activation'))
    slide(f'{title}: {layer.upper()}',f'{clabel}, {noise} central target; exact preview',
      ('Raw mean output; five-fold means. Matched-scale omitted because its mean matches Hebb-FC.' if field=='activation' else
       'Within each fold, compare with FC-off; five-fold means. No confidence intervals.'),
      '每折的测试样本具有多个类别概念。百分比先逐折逐噪声相对FC关闭计算后取均值；余弦由每类16样本的归一化类内散度恒等式推导，非额外独立证据。Fisher=类间/类内；其上升本身不说明类内收紧。',panels)
assert len(slides)==97
sources={str(DOC/n):hashlib.sha256((DOC/n).read_bytes()).hexdigest() for n in ('readout_timeline.csv','fold_geometry_timeline.csv')}
(BUILD/'slides-ready.json').write_text(json.dumps(dict(slides=slides,sources=sources),ensure_ascii=False),encoding='utf8')
print('PREPARED',len(slides),'slides')
