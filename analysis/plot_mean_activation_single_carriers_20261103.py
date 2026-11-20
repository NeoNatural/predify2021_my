from pathlib import Path
import csv, hashlib, json, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables'/'mean_activation_single_carriers_20261103'
OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'docs'/'native_categories_readouts_20261029'
colors={'exact':'#111111','same_concept':'#52c6ed','same_category':'#237df5','different_category':'#d66d2c'}
labels={'exact':'Exact','same_concept':'Same concept','same_category':'Same category','different_category':'Different category'}
layers={'conv4':'Conv4 (PCoder4)','conv5':'Conv5 (PCoder5)','fc1':'FC1','fc2':'FC2'}
carrier_names={'hebb':'Hebbian-only','pcoder':'PCoder-only'}
rows={}
for ds in ('imagenet','things'):
    p=DATA/ds/'activation_timeline.csv'
    with p.open(encoding='utf-8-sig',newline='') as f: rows[ds]=list(csv.DictReader(f))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cccccc','xtick.color':'#555555','ytick.color':'#555555','axes.labelcolor':'#555555','pdf.fonttype':42,'svg.fonttype':'none'})
manifest=[]
with PdfPages(OUT/'mean_activation_single_carriers.pdf') as pdf:
 for carrier,ctitle in carrier_names.items():
  for layer,ltitle in layers.items():
   for noise in ('noisy','clean'):
    fig,axes=plt.subplots(1,2,figsize=(13.333,7.5))
    fig.suptitle(f'Mean activation: {ltitle}',x=.05,y=.94,ha='left',fontsize=25)
    fig.text(.05,.855,f'{ctitle}, {noise} central target; full layer output mean',fontsize=15)
    for ax,ds in zip(axes,('imagenet','things')):
     values=[]
     for identity in labels:
      rr=[r for r in rows[ds] if all(r[k]==v for k,v in dict(carrier=carrier,noise=noise,layer=layer,identity=identity).items())]
      rr.sort(key=lambda r:int(r['timestep']))
      assert [int(r['timestep']) for r in rr]==list(range(15)),(ds,carrier,layer,noise,identity)
      y=np.array([float(r['output_mean']) for r in rr]); assert np.isfinite(y).all();values.extend(y)
      ax.plot(range(15),y,color=colors[identity],label=labels[identity],linewidth=1.8,marker='o',markersize=3.1)
     ax.set_title(ds.upper(),loc='left',fontsize=16,pad=38)
     ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=4,frameon=False,fontsize=8,handlelength=1.5,columnspacing=1.0,handletextpad=.4)
     ax.set_xlim(0,14);ax.set_xticks(range(0,15,2));ax.set_xlabel('Target timestep');ax.set_ylabel('Mean output')
     ax.grid(axis='y',color='#e1e1e1',linewidth=.7);ax.set_axisbelow(True)
     ax.yaxis.set_major_locator(MaxNLocator(nbins=4));fmt=ScalarFormatter(useOffset=False);fmt.set_scientific(False);ax.yaxis.set_major_formatter(fmt)
    fig.text(.05,.12,'Similar global means do not establish identical feature vectors.',fontsize=12)
    fig.subplots_adjust(left=.075,right=.97,top=.65,bottom=.24,wspace=.28)
    stem=f'{carrier}_{noise}_{layer}'
    fig.savefig(OUT/(stem+'.png'),dpi=180,facecolor='white')
    fig.savefig(OUT/(stem+'.svg'),facecolor='white')
    pdf.savefig(fig,facecolor='white');plt.close(fig)
    manifest.append(dict(carrier=carrier,noise=noise,layer=layer,png=stem+'.png',svg=stem+'.svg'))
source_hashes={str(DATA/ds/'activation_timeline.csv'):hashlib.sha256((DATA/ds/'activation_timeline.csv').read_bytes()).hexdigest() for ds in rows}
(OUT/'README.txt').write_text('Mean activation: Hebbian-only and PCoder-only\n\n16 figures: 2 carriers x 2 target conditions x 4 layers. Each figure shows ImageNet and THINGS, with four preview identities and target timesteps 0-14.\nMean output is output_mean from activation_timeline.csv; it is the full layer output mean, not legacy_x_tmp_mean. Each panel has its own linear y-axis, so compare numerical tick values across panels. No additional experiments or inferential tests were performed.\n\nSources and SHA-256:\n'+''.join(f'{p}: {h}\n' for p,h in source_hashes.items()),encoding='utf-8')
(OUT/'manifest.json').write_text(json.dumps({'figures':manifest,'sources_sha256':source_hashes},indent=2),encoding='utf-8')
zip_path=ROOT/'deliverables'/'mean_activation_single_carriers_20261103.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.iterdir()):z.write(p,p.name)
print('Created',len(manifest),'figures:',OUT,'and',zip_path)
