"""Verify nonzero feature vectors required for the cosine/dispersion identity."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.hebb_fc_full_20261103 import OUT,atomic_json
from analysis.hebb_fc_full_replay_20261103 import batched,matched
from predify_experiments import native_categories_20261029 as native

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
state=torch.load(Path.home()/'.cache/torch/hub/checkpoints/vgg16-397923af.pth',map_location='cpu',weights_only=True)
w=state['classifier.3.weight'].cuda();b=state['classifier.3.bias'].cuda();del state
counts={};mins={}
with torch.no_grad():
 for dataset in ('imagenet','things'):
  manifest=json.loads((native.OUT/dataset/'manifest.json').read_text(encoding='utf8'))
  row={t['dataset_index']:j for j,t in enumerate(manifest['trials'])}
  original=np.load(native.OUT/dataset/'features.npy',mmap_mode='r')
  for fold in range(5):
   with np.load(OUT/dataset/'graphs'/f'fold{fold}.npz') as z:
    m1={'scale':torch.tensor(float(z['scale1']),device='cuda'),'matrices':{'corr_k128':torch.tensor(z['E1'],device='cuda')}}
    m2={'scale':torch.tensor(float(z['scale2']),device='cuda'),'matrices':{'corr_k32':torch.tensor(z['E2'],device='cuda')}}
   for carrier in native.CARRIERS:
    for seed in native.NOISES:
     for concept in sorted(c for c,f in manifest['concept_folds'].items() if f==fold):
      with np.load(OUT/dataset/'base'/f'concept_{concept}_seed_{seed}_{carrier}_exact.npz') as z:
       x=torch.tensor(z['fc1'].reshape(-1,4096),device='cuda');ids=z['images']
      ci=next(j for j,c in enumerate(native.conditions()) if c['name']==carrier+'_exact')
      off2=torch.tensor(np.asarray(original[ci,native.NOISES.index(seed),[row[int(k)] for k in ids]]).reshape(-1,4096),device='cuda')
      y=batched(x,m1,'mean_std','fc1');i=batched(x,m1,'inhibition_only','fc1');s=matched(x,y)
      bridge=lambda a:torch.relu(torch.nn.functional.linear(a,w,b))
      v=batched(bridge(y),m2,'mean_std','fc2');iv=batched(bridge(i),m2,'inhibition_only','fc2');sv=matched(bridge(s),v)
      for name,a in [('off_fc1',x),('mean_std_fc1',y),('inhibition_only_fc1',i),('scale_both_fc1',s),('off_fc2',off2),('mean_std_fc2',v),('inhibition_only_fc2',iv),('scale_both_fc2',sv)]:
       norm=torch.linalg.vector_norm(a,dim=1);key=f'{dataset}/{name}';counts[key]=counts.get(key,0)+int((norm==0).sum());mins[key]=min(mins.get(key,float('inf')),float(norm.min()))
   print('COSINE AUDIT',dataset,fold,flush=True)
   del m1,m2;torch.cuda.empty_cache()
assert not any(counts.values()),counts
atomic_json(OUT/'cosine_nonzero_audit.json',{'passed':True,'zero_vector_counts':counts,'minimum_norms':mins,
 'scope':'held-out exact-preview inputs, both layers; all three carriers, four seeds, 15 timesteps, all five folds; all four FC methods'})
