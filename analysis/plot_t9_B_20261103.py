from pathlib import Path
import csv,ast,hashlib,json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator,FuncFormatter
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'docs/native_categories_readouts_20261029';O=ROOT/'deliverables/t9_B_figures_20261103';O.mkdir(parents=True,exist_ok=True)
tree=ast.parse((ROOT/'analysis/plot_adaptive_20261028.py').read_text(encoding='utf8'));ns={'np':np,'MaxNLocator':MaxNLocator,'FuncFormatter':FuncFormatter}
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in ('bounds','Axis')],type_ignores=[]),'axis','exec'),ns);Axis=ns['Axis']
ids=[('exact','Exact','#111111'),('same_concept','Same concept','#52c6ed'),('same_category','Same category','#237df5'),('different_category','Different category','#d66d2c')];carriers={'hebb':'Hebbian-only','pcoder':'PCoder carry-only','joint':'Joint'};data={};sources={}
for ds in ('imagenet','things'):
 for file in ('cv_summary','cv_content_contrasts'):
  p=D/ds/(file+'.csv');sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  with p.open(encoding='utf-8-sig',newline='') as f:data[ds,file]=[r for r in csv.DictReader(f) if r['readout']=='t9_clean_reference']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cccccc','xtick.color':'#555555','ytick.color':'#555555','axes.labelcolor':'#555555','pdf.fonttype':42,'svg.fonttype':'none'})
allvalues=[];axis_report=[]
for mode in ('adaptive','linear'):
 with PdfPages(O/f't9_B_{mode}.pdf') as pdf:
  for metric in ('accuracy','content_difference'):
   for carrier,lab in carriers.items():
    for noise in ('noisy','clean'):
     fig,axes=plt.subplots(1,2,figsize=(13.333,7.5));title='Category accuracy' if metric=='accuracy' else 'Accuracy difference from different category'
     fig.suptitle('t9-B: '+title,x=.05,y=.94,ha='left',fontsize=24);fig.text(.05,.855,f'{lab}, {noise} central target; clean no-memory t9 decoder',fontsize=14)
     for ax,ds in zip(axes,('imagenet','things')):
      vals=[]
      for ident,name,color in ids:
       if metric=='content_difference' and ident=='different_category':continue
       f='cv_summary' if metric=='accuracy' else 'cv_content_contrasts';field='accuracy_fold_mean' if metric=='accuracy' else 'delta_accuracy_pp'
       rr=[r for r in data[ds,f] if r['noise']==noise and (r['condition']==carrier+'_'+ident if metric=='accuracy' else r['carrier']==carrier and r['identity']==ident)]
       rr.sort(key=lambda r:int(r['timestep']));assert [int(r['timestep']) for r in rr]==list(range(15)),(ds,metric,carrier,noise,ident)
       y=[float(r[field]) for r in rr];assert np.isfinite(y).all();vals.extend(y);ax.plot(range(15),y,color=color,label=name,linewidth=1.8,marker='o',markersize=3)
       if mode=='adaptive':
        allvalues.extend(dict(dataset=ds,metric=metric,carrier=carrier,noise=noise,identity=ident,timestep=t,value=v) for t,v in enumerate(y))
      ref=(25 if ds=='imagenet' else 20) if metric=='accuracy' else 0
      a=Axis(vals,reference=ref,physical=(0,100) if metric=='accuracy' else None,compress=mode=='adaptive');a.apply(ax,unit='%' if metric=='accuracy' else ' pp')
      if metric=='accuracy':ax.plot([],[],color='#777777',ls='--',lw=.8,label='Chance')
      else:ax.plot([],[],color='#777777',ls='--',lw=.8,label='Zero')
      ax.set_title(ds.upper(),loc='left',fontsize=16,pad=38);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=5 if metric=='accuracy' else 4,frameon=False,fontsize=8,handlelength=1.5,columnspacing=.8,handletextpad=.4)
      ax.set_xlim(0,14);ax.set_xticks(range(0,15,2));ax.set_xlabel('Target timestep');ax.set_ylabel('Accuracy (%)' if metric=='accuracy' else 'Accuracy difference (percentage points)');ax.grid(axis='y',color='#e1e1e1',linewidth=.7);ax.set_axisbelow(True)
      axis_report.append(dict(mode=mode,metric=metric,carrier=carrier,noise=noise,dataset=ds,axis=a.meta()))
     fig.text(.05,.12,'Five-fold mean. t9-B transfers the clean no-memory t9 readout to each condition.',fontsize=11)
     fig.text(.05,.08,'Gray bands indicate compressed ranges; labels show scale factors.' if mode=='adaptive' else 'Linear axes; each panel uses its own range.',fontsize=10,color='#555555')
     fig.subplots_adjust(left=.09,right=.97,top=.65,bottom=.24,wspace=.31)
     stem=f'{metric}_{carrier}_{noise}_{mode}';fig.savefig(O/(stem+'.png'),dpi=180,facecolor='white');fig.savefig(O/(stem+'.svg'),facecolor='white');pdf.savefig(fig,facecolor='white');plt.close(fig)
with (O/'values.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allvalues[0]));w.writeheader();w.writerows(allvalues)
(O/'README.txt').write_text('t9-B: clean no-memory t9 decoder transferred to each target condition. Five-fold mean accuracy and paired content differences in percentage points.\nEach PDF has 12 plots: accuracy and content difference, 3 carriers, 2 target conditions; ImageNet and THINGS side by side. Gray bands on adaptive plots show compressed ranges and scale factors; linear plots are provided for checking. Each panel has its own range.\nSources SHA-256:\n'+''.join(f'{p}: {h}\n' for p,h in sources.items()),encoding='utf8')
(O/'axis_manifest.json').write_text(json.dumps(axis_report,indent=2),encoding='utf8')
with zipfile.ZipFile(ROOT/'deliverables/t9_B_figures_20261103.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(O.iterdir()):z.write(p,p.name)
print('Created 12 plots in each axis mode;',len(allvalues),'data points')
