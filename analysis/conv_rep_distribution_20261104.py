import sys,json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.hebb_fc_native_pilot_20261102 import test_inputs
from predify_experiments import native_categories_20261029 as native
from predify_experiments.supplementary import load
from predify_experiments.tripartite_20261027 import restore_state,set_init,gray_image
from predify_experiments.sequence import run_recurrent_sequence,reset_hebbian_state
from torchvision.models import VGG16_Weights
OUT=ROOT/'docs/conv_column_signed_20261104';OUT.mkdir(exist_ok=True)
torch,model,layers,rec,_=native.setup_model();rec.close()
for l in layers:l.compute_enabled=False;l.update_enabled=False
saved=load(native.OUT/'gray_snapshot.pckl');common=[{k:v.cuda() for k,v in item.items()} for item in saved['state']];grey=gray_image(torch)
filt=native.FilteredInput(torch);transform=VGG16_Weights.IMAGENET1K_V1.transforms()
rows=[]
def stats(a):
 a=a.detach().float().flatten();ab=a.abs();neg=a<0
 return dict(n=a.numel(),minimum=a.min().item(),maximum=a.max().item(),mean=a.mean().item(),meanabs=ab.mean().item(),rms=a.square().mean().sqrt().item(),negative_fraction=neg.float().mean().item(),negative_below_1e7=(a < -1e-7).float().mean().item(),negative_mass=(ab[neg].sum()/ab.sum().clamp_min(1e-20)).item(),**{f'q{q}':v for q,v in zip(('001','01','50','99'),torch.quantile(a,torch.tensor([.001,.01,.5,.99],device=a.device)).tolist())})
for dataset in ('imagenet','things'):
 spec=json.loads((ROOT/'Hebbian-CNN/Log/hebb_fc_native_pilot_20261102'/dataset/'pilot_manifest.json').read_text(encoding='utf8'))
 for trial in spec['evaluation']['trials']:
  targets,previews=test_inputs(dataset,trial,transform,filt,torch)
  for seed,target in targets.items():
   for ident in ('exact','different_category'):
    reset_hebbian_state(layers);model.reset();set_init(model,False);restore_state(model,common);model.input_mem=grey[None]
    def callback(phase):
     def after(t,logits):
      for k in range(1,6):rows.append(dict(dataset=dataset,target=trial['dataset_index'],concept=trial['concept'],seed=seed,identity=ident,phase=phase,t=t,layer=k,**stats(getattr(model,f'pcoder{k}').rep)))
     return after
    with torch.no_grad():
     for j,pv in enumerate(previews[ident]):run_recurrent_sequence(model,pv[None],2,hebbian_layers=layers,commit_update=False,preserve_model_state=True,after_step=callback('preview'+str(j)))
     run_recurrent_sequence(model,target[None],15,hebbian_layers=layers,commit_update=False,preserve_model_state=True,after_step=callback('target'))
    with (OUT/'rep_distribution.csv').open('w',newline='',encoding='utf8') as f:
     w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(dataset,trial['dataset_index'],seed,ident,'rows',len(rows),flush=True)
summary={}
for k in range(1,6):
 r=[a for a in rows if a['layer']==k and a['phase']=='target'];summary[k]=dict(minimum=min(a['minimum'] for a in r),mean_negative_fraction=np.mean([a['negative_fraction'] for a in r]),max_negative_fraction=max(a['negative_fraction'] for a in r),mean_negative_mass=np.mean([a['negative_mass'] for a in r]))
(OUT/'distribution_summary.json').write_text(json.dumps(dict(complete=True,rows=len(rows),layers=summary),indent=2),encoding='utf8');print(summary,flush=True)
