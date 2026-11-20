"""Training-only post-hoc trace of adaptive threshold fixed-point behavior."""
import csv
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from predify_experiments.fc_sparse_lateral_20261028 import DATA,DOC,DEVICE,connection_model,adaptive_target


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    with np.load(DATA/'imagenet/activations.npz') as z:
        x=torch.as_tensor(z['fc1'][z['fold']!=0],device=DEVICE)
    model=connection_model(x)
    normalized=x/model['scale']
    rows=[]
    for mode,budget in [('soft',1.),('hard',1.),('hard',.75)]:
        r=normalized.clone();previous_mask=r>0
        count=torch.ceil((x>1e-7).sum(1)*budget).long()
        for step in range(1,301):
            drive=normalized+.25*(r@model['matrices']['corr_k128'].T)
            target,theta=adaptive_target(drive,count,mode)
            target=target/(1+2*r.mean(1,keepdim=True))
            updated=torch.where(target>0,.5*r+.5*target,0.)
            mask=updated>0
            if step%10==0:
                rows.append(dict(mode=mode,budget=budget,step=step,
                    max_relative_residual=float(((updated-r).norm(dim=1)/normalized.norm(dim=1)).max()),
                    mean_switching_units=float((mask!=previous_mask).sum(1).float().mean()),
                    maximum_activity=float(updated.max())))
            r=updated;previous_mask=mask
        print('TRACE',mode,budget,rows[-1],flush=True)
    path=DOC/'training_stability_trace.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':
    with torch.inference_mode():main()
