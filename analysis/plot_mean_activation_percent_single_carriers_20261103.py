from pathlib import Path
import csv, hashlib, json, zipfile, ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, FuncFormatter

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'deliverables'/'mean_activation_percent_single_carriers_20261103'; OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'docs'/'native_categories_readouts_20261029'
tree=ast.parse((ROOT/'analysis/plot_adaptive_20261028.py').read_text(encoding='utf-8'))
ns={'np':np,'MaxNLocator':MaxNLocator,'FuncFormatter':FuncFormatter}
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in ('bounds','Axis')],type_ignores=[]),'axis','exec'),ns); Axis=ns['Axis']
colors={'exact':'#111111','same_concept':'#52c6ed','same_category':'#237df5','different_category':'#d66d2c'}
labels={'exact':'Exact','same_concept':'Same concept','same_category':'Same category','different_category':'Different category'}
layers={'conv4':'Conv4 (PCoder4)','conv5':'Conv5 (PCoder5)','fc1':'FC1','fc2':'FC2'};carriers={'hebb':'Hebbian-only','pcoder':'PCoder-only'}
rows={}
for ds in ('imagenet','things'):
 with (DATA/ds/'activation_timeline.csv').open(encoding='utf-8-sig',newline='') as f:rows[ds]=list(csv.DictReader(f))
lookup={}
for ds,rs in rows.items():
 for r in rs:
  key=(ds,r['carrier'],r['noise'],r['layer'],r['identity'],int(r['timestep']))
  assert key not in lookup;lookup[key]=float(r['output_mean'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cccccc','xtick.color':'#555555','ytick.color':'#555555','axes.labelcolor':'#555555','pdf.fonttype':42,'svg.fonttype':'none'})
result=[];manifest=[]
for carrier in carriers:
 for layer in layers:
  for noise in ('noisy','clean'):
   for ds in rows:
    for identity in labels:
     for t in range(15):
      val=lookup[ds,carrier,noise,layer,identity,t]; base=lookup[ds,carrier,noise,layer,'different_category',t]
      assert np.isfinite(base) and abs(base)>1e-12
      pct=100*(val-base)/base
      result.append(dict(dataset=ds,carrier=carrier,noise=noise,layer=layer,identity=identity,timestep=t,output_mean=val,different_category_mean=base,change_pct=pct))
with (OUT/'values.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
for mode in ('adaptive','linear'):
 with PdfPages(OUT/f'mean_activation_percent_{mode}.pdf') as pdf:
  for carrier,ctitle in carriers.items():
   for layer,ltitle in layers.items():
    for noise in ('noisy','clean'):
     fig,axes=plt.subplots(1,2,figsize=(13.333,7.5));fig.suptitle(f'Mean activation change: {ltitle}',x=.05,y=.94,ha='left',fontsize=25)
     fig.text(.05,.855,f'{ctitle}, {noise} central target; relative to different category at each timestep',fontsize=14)
     for ax,ds in zip(axes,rows):
      values=[]
      for identity in labels:
       rr=[r for r in result if all(r[k]==v for k,v in dict(dataset=ds,carrier=carrier,noise=noise,layer=layer,identity=identity).items())]
       assert len(rr)==15 and [r['timestep'] for r in rr]==list(range(15))
       y=[r['change_pct'] for r in rr];values.extend(y)
       ax.plot(range(15),y,color=colors[identity],label=labels[identity],linewidth=1.8,marker='o',markersize=3.1)
      a=Axis(values,reference=0,compress=mode=='adaptive');a.apply(ax,unit='%')
      ax.set_title(ds.upper(),loc='left',fontsize=16,pad=38);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=4,frameon=False,fontsize=8,handlelength=1.5,columnspacing=1,handletextpad=.4)
      ax.set_xlim(0,14);ax.set_xticks(range(0,15,2));ax.set_xlabel('Target timestep');ax.set_ylabel('Change from different category (%)');ax.grid(axis='y',color='#e1e1e1',linewidth=.7);ax.set_axisbelow(True)
      manifest.append(dict(mode=mode,dataset=ds,carrier=carrier,noise=noise,layer=layer,axis=a.meta()))
     fig.text(.05,.12,'(Condition - different category) / different category × 100, computed at each timestep.',fontsize=11)
     fig.text(.05,.08,'Gray bands indicate compressed axis ranges; labels show the scale factor.' if mode=='adaptive' else 'Linear axes; each panel uses its own range.',fontsize=10,color='#555555')
     fig.subplots_adjust(left=.09,right=.97,top=.65,bottom=.24,wspace=.31)
     stem=f'{carrier}_{noise}_{layer}_{mode}';fig.savefig(OUT/(stem+'.png'),dpi=180,facecolor='white');fig.savefig(OUT/(stem+'.svg'),facecolor='white');pdf.savefig(fig,facecolor='white');plt.close(fig)
source_hashes={str(DATA/ds/'activation_timeline.csv'):hashlib.sha256((DATA/ds/'activation_timeline.csv').read_bytes()).hexdigest() for ds in rows}
(OUT/'README.txt').write_text('Percentage change in mean activation\n\n100 * (condition output_mean - different_category output_mean) / different_category output_mean, independently per dataset, carrier, target noise, layer and timestep. Different category is therefore exactly 0%.\nAdaptive plots use a monotonic piecewise y-axis where a sparse tail warrants compression; gray bands and scale factors identify compressed intervals. Linear-axis plots are included for comparison. Each panel has its own axis range.\n16 figures per axis mode, with ImageNet and THINGS side by side. values.csv contains all source values, denominators and percentages.\n\nSources and SHA-256:\n'+''.join(f'{p}: {h}\n' for p,h in source_hashes.items()),encoding='utf-8')
(OUT/'axis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
zip_path=ROOT/'deliverables'/'mean_activation_percent_single_carriers_20261103.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.iterdir()):z.write(p,p.name)
print('Created 16 figures in each of two axis modes; verified',len(result),'points. Compressed panels:',sum(bool(m['axis']['compressed_segments']) for m in manifest if m['mode']=='adaptive'))
