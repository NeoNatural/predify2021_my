"""Read-only mechanism diagnosis for the frozen gain-only SAE pilot."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
import sys,pickle
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis import fc_preview_pilot_20261105 as exp
from analysis import fc_preview_refit_20261105 as refit
from analysis import fc_sae_gain_pilot_20261105 as pilot
from analysis import fc_sae_acceptance_20261105 as src
from predify_experiments.fc_sae_gain_20261105 import Projection,GainProxy,model_hash

OUT=ROOT/'Hebbian-CNN/Log/fc_sae_gain_adjust_20261106'
DOC=ROOT/'docs/fc_sae_gain_adjust_20261106'

def load_models(d):
    models=[]
    for l in ('fc1','fc2'):
        path=pilot.OUT/d/'proxy'/(l+'.pth');note=exp.read(path.with_suffix('.json'))
        assert exp.sha(path)==note['checkpoint_sha256']
        state=torch.load(path,map_location='cuda',weights_only=True)
        p=GainProxy(Projection(),state['center'].cpu(),state['scale'].cpu()).cuda()
        p.load_state_dict(state);p.eval().requires_grad_(False);models.append(p)
    return models

def memory(proxy,preview):
    # Separate fresh zero matrix per trial; preview-only two-frame writes.
    z=proxy.encode(preview);u=z/z.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    W=torch.zeros(len(z),64,64,device=z.device,dtype=z.dtype)
    for j in range(2):
        outer=u[:,j,:,None]*u[:,j,None,:];outer.diagonal(dim1=-2,dim2=-1).zero_();W=.5*W+.5*outer
    return W

def read(proxy,x,W):
    z=proxy.encode(x);h=torch.bmm(W,z.unsqueeze(-1)).squeeze(-1)
    return z,h,proxy.gain_signal(h)

def quantiles(x):return [float(v) for v in torch.quantile(x.flatten().float(),torch.tensor([.01,.1,.5,.9,.99],device=x.device))]

@torch.inference_mode()
def diagnose():
    exp.setup();old=exp.read(exp.OUT/'protocol.json');exp.validate(old);OUT.mkdir(exist_ok=True,parents=True);DOC.mkdir(exist_ok=True,parents=True)
    rows=[];compression=[]
    for d in ('imagenet','things'):
        models=load_models(d);hashes=[model_hash(p) for p in models];data,_,_=src.data(d,old);a=exp.features(d)
        with np.load(pilot.OUT/d/'pilot/representations.npz') as f:reps={k:f[k] for k in f.files}
        with np.load(exp.OUT/d/'nuisance_arrays.npz') as f:maps={l:f[l+'_quartiles'] for l in ('fc1','fc2')}
        for p,l in zip(models,('fc1','fc2')):
            x=torch.from_numpy(data[l]['train']).cuda();v=p.projection(x);xc=x-x.mean(0);vc=p.projection(xc)
            for name in ('train','development'):
                xx=torch.from_numpy(data[l][name]).cuda();vv=p.projection(xx);zz=p.encode_projected(vv);hat=p.decode_projected(zz)
                residual=(hat-vv).square().sum();variance=(vv-v.mean(0)).square().sum()
                compression.append(dict(dataset=d,layer=l,split=name,projection_energy=float(vv.square().sum()/xx.square().sum()),
                    centered_variance_capture=float(p.projection(xx-x.mean(0)).square().sum()/(xx-x.mean(0)).square().sum()),
                    projected_fvu=float(residual/variance),active_latents=float((zz>0).float().sum(-1).mean()),
                    latent_norm=float(zz.norm(dim=-1).mean()),A_rank=int(torch.linalg.matrix_rank(p.A)),
                    matrix_spectral_norm=float(torch.linalg.matrix_norm(p.A,ord=2))))
            for history,idx in (('exact',0),('different',1)):
                x=torch.from_numpy(a['target_fc1'] if l=='fc1' else reps[f'hebbian__{history}__fc2_input']).cuda()
                pp=torch.from_numpy(a['preview_'+l][:,idx]).cuda();W=memory(p,pp);before=W.clone();z,h,q=read(p,x,W)
                mod=.75*torch.tanh(q);increase=x*mod.clamp_min(0);tau=4*increase.mean(-1,keepdim=True);pre=x*(1+mod)-tau
                active=x>0;newcut=active&(pre<=0);r=x.clone()
                for _ in range(5):
                    _,_,qt=read(p,r,W);mt=.75*torch.tanh(qt);t=4*(x*mt.clamp_min(0)).mean(-1,keepdim=True)
                    r=.5*r+.5*(x*(1+mt)-t).clamp_min(0)
                mask=torch.from_numpy(maps[l][a['y']]).cuda()
                shares=[]
                for k in range(4):
                    mk=mask==k;shares.append(float((x*mod*mk).sum()/(x*mk).sum().clamp_min(1e-12)))
                rows.append(dict(dataset=d,layer=l,history=history,n=len(x),q_rms=float(q.square().mean().sqrt()),
                    active_q_quantiles=quantiles(q[active]),active_gain_quantiles=quantiles((1+mod)[active]),
                    active_modulation_rms=float(mod[active].square().mean().sqrt()),
                    activation_weighted_modulation=float((x*mod).sum()/x.sum()),
                    positive_increase_over_input=float(increase.sum()/x.sum()),
                    negative_decrease_over_input=float((x*mod.clamp_max(0)).abs().sum()/x.sum()),
                    pool_over_global_mean=float((tau/x.mean(-1,keepdim=True)).mean()),
                    pool_over_active_mean=float((tau/(x.sum(-1,keepdim=True)/active.sum(-1,keepdim=True))).mean()),
                    pool_over_max=float((tau/x.max(-1,keepdim=True).values).mean()),
                    all_pre_negative_fraction=float((pre<0).float().mean()),
                    originally_active_cut_fraction=float(newcut.sum()/active.sum()),
                    final_cut_exact_zero_fraction=float((newcut&(r==0)).sum()/newcut.sum().clamp_min(1)),
                    final_weak_nonzero_fraction=float(((r>0)&(r<.01*r.max(-1,keepdim=True).values)).sum()/active.sum()),
                    saturating_tanh_fraction=float((q.abs()>1).float().mean()),
                    latent_match=float(((z*h).sum(-1)/z.square().sum(-1).clamp_min(1e-12)).mean()),
                    gain_by_preference_quartile=shares,
                    h_over_z_norm=float((h.norm(dim=-1)/z.norm(dim=-1).clamp_min(1e-12)).mean())))
                assert torch.equal(W,before)
        assert hashes==[model_hash(p) for p in models]
    exp.validate(old);result=dict(passed=True,quantiles=[.01,.1,.5,.9,.99],diagnostics=rows,compression=compression)
    exp.write_json(OUT/'diagnosis.json',result);exp.write_json(DOC/'diagnosis.json',result)
    print(result,flush=True)

if __name__=='__main__':diagnose()
