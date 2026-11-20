"""Independent dense-projection FP64 acceptance of the frozen six-point mechanism."""
import sys,copy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.fc_sae_gated_search_20261106 import OUT,protocol,data,load_models
from analysis import fc_preview_pilot_20261105 as exp
from predify_experiments.fc_sae_gated_20261106 import cascade

@torch.inference_mode()
def check():
    old,p=protocol();exp.setup();cfg=exp.read(OUT/'selection.json')['config'];cal=exp.read(OUT/'calibration.json');w,b=exp.load_bridge();w=w.double();b=b.double();rows=[]
    for d in ('imagenet','things'):
        models=[copy.deepcopy(p).double() for p in load_models(d)];a=data(d,'pilot');x=torch.from_numpy(a['target_fc1'][:1]).cuda().double()
        def independent(v,proxy,preview,cc):
            P=proxy.projection.adjoint(torch.eye(32,device='cuda',dtype=torch.float64))
            def code(t):return torch.relu((t@P.T/proxy.scale-proxy.center)@proxy.A.T+proxy.b)
            W=torch.zeros(64,64,device='cuda',dtype=torch.float64)
            if preview is not None:
                z=code(preview[0]);u=z/z.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                W=.25*u[0,:,None]*u[0,None,:]+.5*u[1,:,None]*u[1,None,:];W.fill_diagonal_(0)
            saved=W.clone();r=v.clone()
            for _ in range(5):
                z=code(r);h=z@W.T;g=((h*(z>0))@proxy.A)@P
                u=torch.tanh(g/cc['q_rms']);mu=(v*u).sum(-1,keepdim=True)/v.sum(-1,keepdim=True).clamp_min(1e-12)
                m=cfg['epsilon']*(u-mu);k=(z*h).sum(-1,keepdim=True)/(z.square().sum(-1,keepdim=True)+1e-12)
                tau=cfg['lambda']*(r.mean(-1,keepdim=True)+r.std(dim=-1,keepdim=True,correction=0))*k/(k+cc['kappa0'])
                r=(v*(1+m)-tau).clamp_min(0)
            assert torch.equal(saved,W);return r
        for h,idx in (('exact',0),('different',1),('none',None)):
            pp=[None,None] if idx is None else [torch.from_numpy(a['preview_'+l][:1,idx]).cuda().double() for l in ('fc1','fc2')]
            rr,_=cascade(x,pp,models,w,b,cal[d],cfg)
            one=independent(x,models[0],pp[0],cal[d][0]);incoming=torch.relu(one@w.T+b)
            two=independent(incoming,models[1],pp[1],cal[d][1]);err=max(float((rr['fc1']-one).abs().max()),float((rr['fc2']-two).abs().max()))
            assert err<1e-10
            rows.append(dict(dataset=d,history=h,fp64_max_error=err))
    exp.validate(old);result=dict(passed=True,source_sha256=exp.sha(__file__),rows=rows,no_original_model_edits=True)
    exp.write_json(OUT/'independent_acceptance.json',result);print(result)

if __name__=='__main__':check()
