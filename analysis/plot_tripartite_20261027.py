"""Rebuild scientific figures from the saved 2026-10-27 study outputs."""
from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.tripartite_20261027 import OUT,DOC,IDENTITIES
COLORS={'none':'#59636e','hebb':'#3465a4','pcoder':'#00897b','joint':'#dc7d25'}
LABELS={'none':'No memory (grey)','hebb':'Hebbian-only','pcoder':'PCoder-only','joint':'Combined'}


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def curve(rows,condition,field='accuracy_pct'):
    selected=sorted([r for r in rows if r['condition']==condition],key=lambda r:int(r['timestep']))
    return np.array([float(r[field]) for r in selected])


def plots():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for i,dataset in enumerate(('imagenet','things')):
        for j,method in enumerate(('mvpa','common')):
            rows=read(DOC/dataset/f'{method}_timeline.csv');ax=axes[i,j]
            for carrier in ('none','hebb','pcoder','joint'):
                name='none' if carrier=='none' else carrier+'_exact'
                ax.plot(range(15),curve(rows,name),label=LABELS[carrier],color=COLORS[carrier],lw=2)
            ax.axhline(25,color='black',lw=.7,ls=':');ax.set_ylim(0,100);ax.set_xticks(range(0,15,2));ax.grid(alpha=.2)
            ax.set_title(dataset.upper()+(' — target-concept grouped MVPA' if method=='mvpa' else ' — one frozen common readout'))
            ax.set_xlabel('Target timestep');ax.set_ylabel('Four-class accuracy (%)')
    axes[0,0].legend(fontsize=8);fig.suptitle('Exact-prime time courses: common grey start, 64-pixel objects')
    fig.tight_layout();fig.savefig(DOC/'main_timecourses.png',dpi=180);plt.close(fig)
    for method in ('mvpa','common'):
        fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
        for i,dataset in enumerate(('imagenet','things')):
            rows=read(DOC/dataset/f'{method}_contrasts.csv')
            for j,contrast in enumerate(('joint-pcoder','joint-hebb','pcoder-hebb')):
                selected=sorted([r for r in rows if r['contrast']==contrast and r['strength']=='strong' and r['identity']=='exact'],key=lambda r:int(r['timestep']))
                y=[float(r['delta_pp']) for r in selected];ax=axes[i,j]
                ax.plot(range(15),y,'o-',ms=3,color='#3465a4');ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.2)
                ax.set_title(dataset.upper()+' '+contrast);ax.set_xlabel('Target timestep');ax.set_ylabel('Accuracy difference (pp)')
        fig.suptitle(method.upper()+': all three memory pairwise contrasts — descriptive timepoints')
        fig.tight_layout();fig.savefig(DOC/f'{method}_three_pairwise.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True,sharey=True)
        for i,dataset in enumerate(('imagenet','things')):
            rows=read(DOC/dataset/f'{method}_timeline.csv')
            for j,carrier in enumerate(('hebb','pcoder','joint')):
                ax=axes[i,j]
                for identity,color in zip(IDENTITIES,('#00897b','#3465a4','#dc7d25','#b84a62','#777777')):
                    ax.plot(range(15),curve(rows,carrier+'_'+identity),label=identity,color=color,lw=1.5)
                ax.set_title(dataset.upper()+' '+LABELS[carrier]);ax.set_ylim(0,100);ax.grid(alpha=.2)
                ax.set_xlabel('Target timestep');ax.set_ylabel('Accuracy (%)')
        axes[0,0].legend(fontsize=7);fig.suptitle(method.upper()+': complete prime-content controls, main strength')
        fig.tight_layout();fig.savefig(DOC/f'{method}_content_controls.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for i,dataset in enumerate(('imagenet','things')):
        rows=read(DOC/dataset/'mvpa_timeline.csv')
        for j,carrier in enumerate(('hebb','joint')):
            ax=axes[i,j]
            for name,label,color in [(carrier+'_exact','coeff .05',COLORS[carrier]),(carrier+'_weak_exact','coeff .0125','#9a67bd'),('none','No memory','#59636e'),('pcoder_exact','PCoder-only','#00897b')]:
                ax.plot(range(15),curve(rows,name),label=label,color=color,lw=1.5)
            ax.set_title(dataset.upper()+' '+LABELS[carrier]);ax.set_ylim(0,100);ax.grid(alpha=.2);ax.set_xlabel('Target timestep');ax.set_ylabel('MVPA accuracy (%)')
    axes[0,0].legend(fontsize=8);fig.suptitle('Prespecified Hebbian strength sensitivity — no winner selection')
    fig.tight_layout();fig.savefig(DOC/'weak_strength_mvpa.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for i,dataset in enumerate(('imagenet','things')):
        for j,method in enumerate(('mvpa','common')):
            rows=read(DOC/dataset/f'{method}_timeline.csv');ax=axes[i,j]
            for name,label,color in [('none','Common grey','#59636e'),('none_random','Random init','#b84a62'),('none_feedforward','Feedforward init','#3465a4'),('pcoder_exact','PCoder exact','#00897b')]:
                ax.plot(range(15),curve(rows,name),label=label,color=color,lw=1.8)
            ax.set_title(dataset.upper()+' '+method.upper());ax.set_ylim(0,100);ax.grid(alpha=.2)
            ax.set_xlabel('Target timestep');ax.set_ylabel('Accuracy (%)')
    axes[0,0].legend(fontsize=8);fig.suptitle('Secondary initialization references — target-time aligned')
    fig.tight_layout();fig.savefig(DOC/'initialization_references.png',dpi=180);plt.close(fig)


def stimuli():
    mean=np.array([.485,.456,.406])[:,None,None];std=np.array([.229,.224,.225])[:,None,None]
    for dataset in ('imagenet','things'):
        m=json.loads((OUT/dataset/'manifest.json').read_text());fig,axes=plt.subplots(4,8,figsize=(15,8))
        for row,trial in enumerate(m['dev'][:4]):
            index=trial['dataset_index'];noise=m['noise_seeds']['dev'][0]
            source=OUT/dataset/'smoke'
            with np.load(source/f'stimulus_{index}_seed_{noise}.npz') as z:panels=[z['clean'],z['target']]
            with np.load(source/f'previews_{index}.npz') as z:
                panels+=[z['exact_preview0']]+[z[i+'_preview1'] for i in IDENTITIES]
            for col,value in enumerate(panels):
                axes[row,col].imshow(np.clip(value*std+mean,0,1).transpose(1,2,0));axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                if row==0:axes[row,col].set_title(['Clean','Noisy','Preview 1','Exact','Same concept','Same category','Different','Neutral'][col],fontsize=9)
                if col==0:axes[row,col].set_ylabel(m['categories'][trial['category_index']])
        fig.suptitle(dataset.upper()+': actual normalized input canvases, thirds along width')
        fig.tight_layout();fig.savefig(DOC/f'{dataset}_actual_stimuli.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    if '--stimuli' in sys.argv:stimuli()
    else:plots()
