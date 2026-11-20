"""Static high-dimensional overview; no re-estimation or cross-carrier contrasts."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm,LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/tripartite_filtered_cv_20261027'
METRICS=[('output_mean_effect_pct','Output activation change (%)'),('normalized_trace_effect_pct','Normalized trace change (%)'),('within_cosine_delta_x1e3','Within-class cosine difference (x1e3)'),('normalized_fisher_effect_pct','Normalized Fisher change (%)')]
LAYERS=('conv4','conv5','fc1','fc2')
CONTEXTS=[(d,n,c,i) for d in ('imagenet','things') for n in ('clean','noisy') for c in ('hebb','pcoder','joint') for i in ('exact','same_concept','same_category')]

def main():
    values=np.full((4,4,36,15),np.nan)
    for dataset in ('imagenet','things'):
        with (DOC/dataset/'representation_effects.csv').open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                keys=[m[0] for m in METRICS]
                if r['metric'] in keys:values[keys.index(r['metric']),LAYERS.index(r['layer']),CONTEXTS.index((dataset,r['noise'],r['carrier'],r['identity'])),int(r['timestep'])]=float(r['value'])
    assert np.isfinite(values).all()
    cmap=LinearSegmentedColormap.from_list('blue_neutral_orange',['#2471a3','#fafafa','#dc6512'])
    fig=plt.figure(figsize=(19,24));gs=fig.add_gridspec(4,5,width_ratios=[1,1,1,1,.065],left=.16,right=.95,bottom=.055,top=.91,hspace=.33,wspace=.09)
    labels=[f'{"ImageNet" if d=="imagenet" else "THINGS"} {n} | {dict(hebb="H",pcoder="P",joint="J")[c]} | {dict(exact="Exact",same_concept="Concept",same_category="Category")[i]}' for d,n,c,i in CONTEXTS]
    scales=[]
    for mi,(key,title) in enumerate(METRICS):
        a=values[mi];absval=np.abs(a);m=absval.max();positive=absval[absval>1e-12];c=float(np.median(positive)) if positive.size else 1.
        norm=FuncNorm((lambda x,c=c:np.sign(x)*np.log1p(np.abs(x)/c),lambda x,c=c:np.sign(x)*np.expm1(np.abs(x))*c),vmin=-m,vmax=m)
        for li,layer in enumerate(LAYERS):
            ax=fig.add_subplot(gs[mi,li]);im=ax.imshow(a[li],aspect='auto',interpolation='none',cmap=cmap,norm=norm,extent=[-.5,14.5,35.5,-.5])
            ax.set_title(layer.upper(),fontsize=12,pad=9);ax.set_xticks([0,4,9,14],['t0','t4','t9','t14']);ax.tick_params(axis='x',labelsize=10,length=0)
            ax.set_yticks(range(36));ax.tick_params(axis='y',labelsize=8,length=0)
            if li==0:ax.set_yticklabels(labels)
            else:ax.set_yticklabels([])
            for y in (8.5,17.5,26.5):ax.axhline(y,color='#666666',lw=.65)
            for y in (2.5,5.5,11.5,14.5,20.5,23.5,29.5,32.5):ax.axhline(y,color='#cccccc',lw=.35)
            for spine in ax.spines.values():spine.set_linewidth(.4);spine.set_color('#cccccc')
            if li==0:fig.text(.16,ax.get_position().y1+.025,title,fontsize=14,fontweight='medium',ha='left')
        cb=fig.colorbar(im,cax=fig.add_subplot(gs[mi,4]));ticks=sorted(set([-m,-c,0,c,m]));cb.set_ticks(ticks);cb.ax.set_yticklabels([f'{v:.3g}' for v in ticks],fontsize=9);cb.outline.set_visible(False)
        scales.append(dict(metric=key,max_absolute=float(m),symlog_constant=c,transform='sign(x)*ln(1+abs(x)/c)',same_range_for_all_layers_contexts=True))
    fig.suptitle('Priming effects: dataset x noise x carrier x content x layer x time',fontsize=19,y=.975)
    fig.text(.16,.956,'All effects are relative to different category within the same carrier/noise/time. Color is signed magnitude, not significance.',fontsize=11)
    fig.text(.16,.027,'H = Hebbian-only   P = PCoder carry-only   J = Joint.  Exact / Concept / Category refer to preview content.\nSymmetric-log color scale; separate units/ranges per metric. No between-carrier contrasts or rankings.',fontsize=11)
    out=DOC/'advanced_figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'all_factor_effect_matrix.png',dpi=220);fig.savefig(out/'all_factor_effect_matrix.svg');plt.close(fig)
    (out/'color_scales.json').write_text(json.dumps(scales,indent=2),encoding='utf8');print('STATIC TENSOR MATRIX COMPLETE')

if __name__=='__main__':main()
