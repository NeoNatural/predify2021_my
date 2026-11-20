"""Content controls for each carrier: faithful linear and disclosed nonlinear axes."""
from pathlib import Path
import csv,json,hashlib,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'docs/tripartite_t9_20261027'
OUT=ROOT/'deliverables/T9_Content_Controls_EN';OUT.mkdir(parents=True,exist_ok=True)
PPT=ROOT/'deliverables/Predify_Hebbian_Tripartite_Supplement_EN_v2_t9.pptx'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ppt_before=sha(PPT)
datasets=['imagenet','things'];dataset_names=['ImageNet','THINGS']
carriers=['hebb','pcoder','joint'];carrier_names=['Hebbian-only','PCoder carry only','Joint memory']
identities=['exact','same_concept','same_category','different_category','neutral']
labels=['Exact','Same concept','Same category','Different category','Neutral slot']
colors=['#111111','#66CBEC','#278EFF','#D77835','#8E63B8']
markers=['o','s','^','D','v']
data={};source_rows=[]
for d in datasets:
    with (SRC/d/'t9_timeline.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for c in carriers:
        for identity in identities:
            rr=sorted([r for r in rows if r['condition']==f'{c}_{identity}'],key=lambda r:int(r['timestep']))
            assert [int(r['timestep']) for r in rr]==list(range(15))
            data[d,c,identity]=np.array([float(r['accuracy_pct']) for r in rr])
            source_rows.extend(rr)
plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.titlesize':15,'axes.labelsize':12,
    'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
KNOT=45.;LOW_FACTOR=.25
def forward(v):
    v=np.asarray(v);return np.where(v<KNOT,KNOT+(v-KNOT)*LOW_FACTOR,v)
def inverse(v):
    v=np.asarray(v);return np.where(v<KNOT,KNOT+(v-KNOT)/LOW_FACTOR,v)
# Continuous, monotone and exactly invertible; every datum fits both views.
grid=np.linspace(10,75,1001)
assert np.all(np.diff(forward(grid))>0) and np.allclose(inverse(forward(grid)),grid)
assert all(np.all((v>=10)&(v<=75)) for v in data.values())
figures=[]
def save(fig,stem):
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'{stem}.{ext}',dpi=240,facecolor='white')
    figures.append(stem);plt.close(fig)
def decorate(ax,mode):
    ax.set_xlim(-.25,14.25);ax.set_xticks(range(0,15,2));ax.set_xlabel('Target timestep')
    ax.set_ylabel('Accuracy (%)');ax.grid(axis='y',color='#DDDDDD',lw=.7);ax.set_axisbelow(True)
    if mode=='nonlinear':
        ax.set_yscale('function',functions=(forward,inverse));ax.set_ylim(10,75)
        ax.yaxis.set_major_locator(FixedLocator([10,25,45,50,55,60,65,70,75]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
        ax.axhspan(10,45,color='#F2F2F2',zorder=0)
        ax.axhline(45,color='#999999',lw=.8,ls=':')
        ax.text(.98,.035,'10–45% compressed ×0.25',transform=ax.transAxes,ha='right',va='bottom',fontsize=10,color='#555555')
    elif mode=='late':
        ax.set_xlim(4.75,14.25);ax.set_xticks(range(5,15));ax.set_ylim(45,70);ax.set_yticks(range(45,71,5))
    else:ax.set_ylim(10,75);ax.set_yticks([10,25,35,45,55,65,75])
    if mode!='late':ax.axhline(25,color='#AAAAAA',lw=.8,ls='--')
def accuracy_figure(selected,mode,stem,title):
    nr=len(selected);fig,axes=plt.subplots(nr,2,figsize=(14.6,6.8 if nr==1 else 4.1*nr+1.5),squeeze=False)
    fig.subplots_adjust(left=.065,right=.985,bottom=.17 if nr==1 else .065,top=.73 if nr==1 else .88,hspace=.42,wspace=.21)
    for r,c in enumerate(selected):
        for col,d in enumerate(datasets):
            ax=axes[r,col]
            for identity,label,color,marker in zip(identities,labels,colors,markers):
                v=data[d,c,identity];xx=np.arange(15)
                if mode=='late':xx=xx[5:];v=v[5:];assert np.all((v>=45)&(v<=70))
                ax.plot(xx,v,label=label,color=color,marker=marker,lw=1.8,ms=3.6)
            decorate(ax,mode);ax.set_title(f'{dataset_names[col]} · {carrier_names[carriers.index(c)]}',loc='left',pad=12)
    fig.suptitle(title,x=.065,y=.985,ha='left',fontsize=21)
    desc='t9-trained frozen readout · Main strength · 64 targets × 3 noises per dataset · Post-hoc descriptive analysis'
    fig.text(.065,.90 if nr==1 else .95,desc,fontsize=11.5)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper left',bbox_to_anchor=(.06,.865 if nr==1 else .935),ncol=5,frameon=False,fontsize=11.5)
    foot={'nonlinear':'Continuous nonlinear y-axis: 10–45% is compressed fourfold; 45–75% is linear. Tick labels show actual accuracy. Same scale in every panel.',
          'linear':'Linear y-axis and identical limits in every panel. Dashed line: 25% chance. No smoothing, curve selection, or per-time significance tests.',
          'late':'Linear y-axis; t5–t14 only, with identical limits in every panel. Full trajectories are supplied separately. This display window is not a new test endpoint.'}[mode]
    fig.text(.065,.025,foot,fontsize=10.2)
    save(fig,stem)

accuracy_figure(carriers,'nonlinear','01_all_carriers_nonlinear','t9 readout: content controls across memory carriers')
for i,c in enumerate(carriers,2):accuracy_figure([c],'nonlinear',f'{i:02}_{c}_nonlinear',f't9 readout: {carrier_names[carriers.index(c)]} content controls')
accuracy_figure(carriers,'linear','05_all_carriers_linear','t9 readout: content controls on a linear accuracy axis')
accuracy_figure(carriers,'late','06_late_window_linear','t9 readout: later content differences at a readable scale')

fig,axes=plt.subplots(3,2,figsize=(14.6,13.8));fig.subplots_adjust(left=.065,right=.985,bottom=.065,top=.88,hspace=.42,wspace=.21)
diffs=[data[d,c,'exact']-data[d,c,identity] for d in datasets for c in carriers for identity in identities[1:]]
lo=5*np.floor(min(v.min() for v in diffs)/5);hi=5*np.ceil(max(v.max() for v in diffs)/5)
for r,c in enumerate(carriers):
    for col,d in enumerate(datasets):
        ax=axes[r,col]
        for identity,label,color,marker in zip(identities[1:],labels[1:],colors[1:],markers[1:]):
            ax.plot(range(15),data[d,c,'exact']-data[d,c,identity],label='Exact − '+label,color=color,marker=marker,lw=1.8,ms=3.6)
        ax.set(title=f'{dataset_names[col]} · {carrier_names[r]}',xlabel='Target timestep',ylabel='Accuracy difference (pp)',xlim=(-.25,14.25),ylim=(lo,hi),xticks=range(0,15,2))
        ax.axhline(0,color='#555555',lw=1);ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
fig.suptitle('t9 readout: exact-preview differences from content controls',x=.065,y=.985,ha='left',fontsize=21)
fig.text(.065,.95,'Positive values favor exact repetition. All comparisons use the same timestep and the same fixed classifier.',fontsize=11.5)
handles,legend_labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='upper left',bbox_to_anchor=(.06,.935),ncol=4,frameon=False,fontsize=11)
fig.text(.065,.025,'Linear difference axis with common limits. Descriptive paired accuracy differences; no per-time corrected significance is claimed.',fontsize=10.5)
save(fig,'07_exact_minus_controls')

with (OUT/'plotted_accuracy.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(source_rows[0]));w.writeheader();w.writerows(source_rows)
metadata=dict(source_sha256={str(SRC/d/'t9_timeline.csv'):sha(SRC/d/'t9_timeline.csv') for d in datasets},
    source_rows=len(source_rows),main_strength=.05,times=list(range(15)),
    nonlinear_axis=dict(knot_percent=45,lower_slope=.25,upper_slope=1,limits=[10,75],transform='y<45: 45+0.25*(y-45); otherwise y',labels='actual accuracy percent'),
    late_display_only=[5,14],linear_difference_limits=[float(lo),float(hi)],
    no_data_smoothing=True,no_new_fits=True,no_new_inferential_tests=True,ppt_unchanged=sha(PPT)==ppt_before)
assert metadata['ppt_unchanged'] and len(source_rows)==450
(OUT/'provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
(OUT/'README.md').write_text('''# t9 readout 内容控制图

依据本轮三等分布局的 t9 固定读出，分别展示 Hebbian-only、PCoder carry only、Joint memory；主强度0.05，全部五种内容，两个数据集。图中文字为英文。未修改PPT，未重拟合模型，未新增显著性检验。

## 文件

- 01：三载体×两数据集总览，连续非线性纵轴。
- 02 / 03 / 04：Hebbian、PCoder、联合的独立双面板图，适合分别放入slide。
- 05：完全相同的数据，线性纵轴对照。
- 06：t5–t14局部展示，线性纵轴45–70%。这是展示窗口，不是新选择的统计端点。
- 07：Exact减去其余四种内容，线性差值轴，单位pp。

每幅图提供PNG、SVG和PDF。plotted_accuracy.csv包含全部450个绘图数据点；provenance.json记录来源哈希、缩放参数及PPT未改动的检查。

## 为什么这样缩放

普通log轴会压缩较高准确率区，未必有助于观察后期差别。本次使用连续分段线性尺度：45%以下每个百分点占用的高度为45%以上的1/4；45%以上保持线性。公式：y<45时，f(y)=45+0.25(y−45)；否则f(y)=y。全部面板使用同一尺度，坐标刻度仍标真实准确率，低区灰底并明确注明压缩，未省略任何区间。25%虚线为四类机会水平。

此显示会改变不同区间的视觉斜率，不能用曲线的视觉斜率比较变化速率。比较条件差距时优先核对05线性图、06后期线性局部图和07差值图。观察到的小幅分离不等于显著差异；三噪声和15时间点不应当作独立样本。Neutral仅移除prime slot，其他干扰物仍保留。
''',encoding='utf-8')
archive=ROOT/'deliverables/T9_Content_Controls_EN.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):z.write(p,p.name)
print(json.dumps(dict(figures=len(figures),files=len(list(OUT.iterdir())),archive=str(archive),validation=metadata),indent=2))
