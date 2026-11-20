"""Observe inactive encoder routes in the original default; no update or fitting."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.fc_sae_gain_diagnosis_20261106 import load_models,memory,read,OUT,DOC
from analysis import fc_preview_pilot_20261105 as exp
from analysis import fc_sae_gain_pilot_20261105 as pilot
from analysis import fc_sae_acceptance_20261105 as src

def geometry(x,y):
    u=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    centers=torch.stack([u[y==c].mean(0) for c in np.unique(y)])
    within=torch.stack([(u[y==c]-centers[k]).square().sum(-1).mean() for k,c in enumerate(np.unique(y))]).mean()
    between=(centers-centers.mean(0)).square().sum(-1).mean()
    return float(between/within),1-u@u.T

@torch.inference_mode()
def main():
    exp.setup();rows=[];geometries=[];old=exp.read(exp.OUT/'protocol.json')
    for d in ('imagenet','things'):
        a=exp.features(d);models=load_models(d);data,meta,_=src.data(d,old)
        from scipy.stats import spearmanr
        for k,p in enumerate(models):
            layer='fc'+str(k+1)
            for scope,key in (('train','train'),('development','development')):
                x=torch.from_numpy(data[layer][scope]).cuda();yy=meta[key]['y'];v=p.projection(x);z=p.encode_projected(v)
                f0,dist=geometry(x,yy);tri=torch.triu_indices(len(x),len(x),1,device='cuda');reference=dist[tri[0],tri[1]].cpu().numpy()
                for variant,t in (('projection',v),('latent',z)):
                    ff,dd=geometry(t,yy);rho=spearmanr(reference,dd[tri[0],tri[1]].cpu().numpy()).statistic
                    geometries.append(dict(dataset=d,layer=layer,scope=scope,variant=variant,fisher=ff,raw_fisher=f0,
                        fisher_change=100*(ff/f0-1),distance_rank_correlation=float(rho),readout_fitted=False))
        with np.load(pilot.OUT/d/'pilot/representations.npz') as f:
            for k,p in enumerate(models):
                layer='fc'+str(k+1)
                for history,idx in (('exact',0),('different',1)):
                    x=torch.from_numpy(a['target_fc1'] if k==0 else f[f'hebbian__{history}__fc2_input']).cuda()
                    W=memory(p,torch.from_numpy(a['preview_'+layer][:,idx]).cuda());z,h,q=read(p,x,W)
                    hg=h*(z>0);qg=p.gain_signal(hg)
                    rows.append(dict(dataset=d,layer=layer,history=history,
                        h_energy_on_inactive_latents=float((h*(z==0)).square().sum()/h.square().sum()),
                        gated_to_original_q_norm=float(qg.norm()/q.norm()),
                        gated_original_q_cosine=float(((qg*q).sum(-1)/(qg.norm(dim=-1)*q.norm(dim=-1))).mean())))
    result=dict(source_sha256=exp.sha(__file__),monitor_only=True,not_used_to_select_parameters=True,rows=rows,geometry=geometries)
    exp.write_json(OUT/'gate_monitor.json',result);exp.write_json(DOC/'gate_monitor.json',result);print(result)

if __name__=='__main__':main()
