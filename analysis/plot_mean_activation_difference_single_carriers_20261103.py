from pathlib import Path
import csv, hashlib, json, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, ScalarFormatter
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables'/'mean_activation_difference_single_carriers_20261103';OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'docs'/'native_categories_readouts_20261029'
colors={'exact':'#111111','same_concept':'#52c6ed','same_category':'#237df5','different_category':'#d66d2c'}
labels={'exact':'Exact','same_concept':'Same concept','same_category':'Same category','different_category':'Different category'}
layers={'conv4':'Conv4 (PCoder4)','conv5':'Conv5 (PCoder5)','fc1':'FC1','fc2':'FC2'};carriers={'hebb':'Hebbian-only','pcoder':'PCoder-only'}
rows={};lookup={}
for ds in ('imagenet','things'):
 with (DATA/ds/'activation_timeline.csv').open(encoding='utf-8-sig',newline='') as f:rows[ds]=list(csv.DictReader(f))
 for r in rows[ds]:
  key=(ds,r['carrier'],r['noise'],r['layer'],r['identity'],int(r['timestep']))
  assert key not in lookup;lookup[key]=float(r['output_mean'])
result=[]
for carrier in carriers:
 for layer in layers:
  for noise in ('noisy','clean'):
   for ds in rows:
    for identity in labels:
     for t in range(15):
      value=lookup[ds,carrier,noise,layer,identity,t];baseline=lookup[ds,carrier,noise,layer,'different_category',t]
      assert np.isfinite(value) and np.isfinite(baseline)
      result.append(dict(dataset=ds,carrier=carrier,noise=noise,layer=layer,identity=identity,timestep=t,output_mean=value,different_category_mean=baseline,difference=value-baseline))
with (OUT/'values.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cccccc','xtick.color':'#555555','ytick.color':'#555555','axes.labelcolor':'#555555','pdf.fonttype':42,'svg.fonttype':'none'})
with PdfPages(OUT/'mean_activation_difference.pdf') as pdf:
 for carrier,ctitle in carriers.items():
  for layer,ltitle in layers.items():
   for noise in ('noisy','clean'):
    fig,axes=plt.subplots(1,2,figsize=(13.333,7.5));fig.suptitle(f'Mean activation difference: {ltitle}',x=.05,y=.94,ha='left',fontsize=25)
    fig.text(.05,.855,f'{ctitle}, {noise} central target; relative to different category at each timestep',fontsize=14)
    for ax,ds in zip(axes,rows):
     for identity in labels:
      rr=[r for r in result if all(r[k]==v for k,v in dict(dataset=ds,carrier=carrier,noise=noise,layer=layer,identity=identity).items())]
      assert len(rr)==15 and [r['timestep'] for r in rr]==list(range(15))
      ax.plot(range(15),[r['difference'] for r in rr],color=colors[identity],label=labels[identity],linewidth=1.8,marker='o',markersize=3.1)
     ax.set_title(ds.upper(),loc='left',fontsize=16,pad=38);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=4,frameon=False,fontsize=8,handlelength=1.5,columnspacing=1,handletextpad=.4)
     ax.set_xlim(0,14);ax.set_xticks(range(0,15,2));ax.set_xlabel('Target timestep');ax.set_ylabel('Mean output difference (raw units)')
     ax.yaxis.set_major_locator(MaxNLocator(nbins=5));fmt=ScalarFormatter(useOffset=False,useMathText=True);fmt.set_powerlimits((-3,4));ax.yaxis.set_major_formatter(fmt)
     ax.grid(axis='y',color='#e1e1e1',linewidth=.7);ax.set_axisbelow(True)
    fig.text(.05,.12,'Condition mean output − different-category mean output, computed at each timestep.',fontsize=11)
    fig.text(.05,.08,'Linear axes; each panel uses its own range.',fontsize=10,color='#555555')
    fig.subplots_adjust(left=.09,right=.97,top=.65,bottom=.24,wspace=.31)
    stem=f'{carrier}_{noise}_{layer}';fig.savefig(OUT/(stem+'.png'),dpi=180,facecolor='white');fig.savefig(OUT/(stem+'.svg'),facecolor='white');pdf.savefig(fig,facecolor='white');plt.close(fig)
source_hashes={str(DATA/ds/'activation_timeline.csv'):hashlib.sha256((DATA/ds/'activation_timeline.csv').read_bytes()).hexdigest() for ds in rows}
(OUT/'README.txt').write_text('Mean activation difference\n\ncondition output_mean - different_category output_mean, independently for each dataset, carrier, target noise, layer and timestep. Values retain raw activation units. Different category is exactly zero.\n16 figures, each with ImageNet and THINGS side by side. Linear axes are scaled independently for each panel. values.csv contains both inputs and the difference.\n\nSources and SHA-256:\n'+''.join(f'{p}: {h}\n' for p,h in source_hashes.items()),encoding='utf-8')
zip_path=ROOT/'deliverables'/'mean_activation_difference_single_carriers_20261103.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.iterdir()):z.write(p,p.name)
print('Created 16 figures,',len(result),'data points; baseline all zero:',all(r['difference']==0 for r in result if r['identity']=='different_category'))
