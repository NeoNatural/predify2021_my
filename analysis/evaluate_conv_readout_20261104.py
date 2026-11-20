import sys,json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.supplementary import load
O=ROOT/'Hebbian-CNN/Log/conv_column_nonnegative_20261104';D=ROOT/'docs/conv_column_nonnegative_20261104';rows=[]
for ds in ('imagenet','things'):
 m=json.loads((O/ds/'manifest.json').read_text());head=load(ROOT/'Hebbian-CNN/Log/native_categories_readouts_20261029'/ds/'cv/reference_heads/fold0.pckl')['head']
 for tr in m['test']:
  for seed in (0,2921):
   for ident in ('exact','same_concept','same_category','different_category'):
    for mode in m['modes']:
     with np.load(O/ds/f"{tr['dataset_index']}_{seed}_{ident}_{mode}.npz") as z:
      x=z['fc2'];pred=head.predict(x);scores=head.decision_function(x)
      for t in range(15):rows.append(dict(dataset=ds,target=tr['dataset_index'],concept=tr['concept'],category=tr['category_index'],seed=seed,identity=ident,mode=mode,t=t,correct=int(pred[t]==tr['category_index']),prediction=int(pred[t]),margin=float(scores[t,tr['category_index']]-np.max(np.delete(scores[t],tr['category_index'])))))
 print(ds,'done',flush=True)
with (D/'readout_per_target.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
import pandas as pd
x=pd.DataFrame(rows);x['window']=np.where(x.t<5,'early',np.where(x.t<10,'middle','late'));g=x.groupby(['dataset','seed','mode','window']).correct.mean().unstack('mode');print(g.round(4).to_string());g.to_csv(D/'readout_windows.csv')
a=x.groupby(['dataset','seed','identity','mode','window']).correct.mean().unstack('mode');a.to_csv(D/'readout_content_windows.csv')
