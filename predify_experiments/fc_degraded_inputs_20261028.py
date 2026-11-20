"""Paired Gaussian/fractal input caches for the independent frozen FC study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .fc_lateral_vanilla_20261028 import ROOT, OUT as CLEAN, sha, write_json
from . import fc_gain075_serial_20261028 as strong

OUT = ROOT / 'Hebbian-CNN/Log/fc_degraded_k128_20261028'
DOC = ROOT / 'docs/fc_degraded_k128_20261028'
INPUTS = OUT / 'inputs'
REPEATS = (0, 1, 2)
DEGRADATIONS = (('predify_noise', .5), ('fractal_occlusion', .3))
sys.path.insert(0, str(ROOT / 'Hebbian-CNN'))
from Hebb_Priming_Utils import apply_input_degradation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seed_for(dataset, row_index, repeat, mode):
    key = f'FC-degraded-20261028/{dataset}/{row_index}/{repeat}/{mode}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**31 - 1)


def occlusion_mask(image, intensity, seed):
    """Reproduce the legacy mask for measuring opacity, verified against its output."""
    batched = image.unsqueeze(0)
    h, w = image.shape[-2:]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    fractal = torch.zeros((1, 1, h, w), dtype=image.dtype)
    for octave in range(3):
        layer = torch.rand((1, 1, 2**(octave+2), 2**(octave+2)), generator=generator)
        fractal += F.interpolate(layer, size=(h,w), mode='bilinear', align_corners=False) * .5**octave
    fractal /= 1.5
    kernel = torch.tensor([[-1.,-1.,-1.],[-1.,8.,-1.],[-1.,-1.,-1.]], dtype=image.dtype).view(1,1,3,3)
    edge = F.conv2d(batched.mean(1,keepdim=True), kernel, padding=1).abs()
    low, high = edge.amin((-2,-1),keepdim=True), edge.amax((-2,-1),keepdim=True)
    edge = (edge-low)/(high-low+1e-5)
    hard = (fractal*(1+2.5*edge) > 1.4-intensity**.7*1.25).to(image.dtype)
    width = 5 if intensity < .5 else 9
    return F.avg_pool2d(hard, width, stride=1, padding=width//2)[0]


def corrupt(image, dataset, index, repeat, mode, level):
    seed = seed_for(dataset,index,repeat,mode)
    altered = apply_input_degradation(image,mode=mode,level=level,seed=seed,fractal_octaves=3)
    difference = altered-image
    opacity = covered = strong_mask = replay = 0.
    if mode == 'fractal_occlusion':
        mask = occlusion_mask(image,level,seed)
        black = torch.tensor([-.485/.229,-.456/.224,-.406/.225],dtype=image.dtype)[:,None,None]
        replay = float(((1-mask)*image+mask*black-altered).abs().max())
        assert replay < 1e-6
        opacity,covered,strong_mask = float(mask.mean()),float((mask>0).float().mean()),float((mask>=.95).float().mean())
    diagnostic = np.array([seed,float(difference.mean()),float(difference.std()),float(difference.square().mean().sqrt()),
                           opacity,covered,strong_mask,replay],dtype=np.float64)
    return altered,diagnostic


def freeze():
    old = read(strong.OUT/'protocol.json')
    strong.validate(old)
    original = read(ROOT/'Hebbian-CNN/Log/fc_serial_20261028/protocol.json')
    protocol = dict(version=1,source_sha256=sha(__file__),
        clean_protocol_sha256=sha(strong.OUT/'protocol.json'),
        legacy_degradation_sha256=sha(ROOT/'Hebbian-CNN/Hebb_Priming_Utils.py'),
        checkpoint=original['checkpoint'],checkpoint_sha256=original['checkpoint_sha256'],
        clean_inputs=original['inputs'],repeats=REPEATS,degradations=DEGRADATIONS,
        location='224x224 RGB after original torchvision ImageNet preprocessing; before unmodified VGG16',
        randomization='independent reproducible seed per dataset/image/repeat/type; same cached realization used by every method',
        noise='legacy Gaussian added in normalized units; no clipping',
        occlusion='legacy edge-guided fractal black blend, octaves3; intensity .3 is not area .3',
        diagnostics=['seed','difference_mean','difference_std','difference_rms','mask_mean_opacity',
                     'mask_positive_fraction','mask_ge_095_fraction','legacy_mask_replay_error'])
    INPUTS.mkdir(parents=True,exist_ok=True); DOC.mkdir(exist_ok=True)
    p=INPUTS/'protocol.json'
    if p.exists():
        assert read(p)==json.loads(json.dumps(protocol)),'Frozen extraction changed'
    else:
        write_json(p,protocol);(INPUTS/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return protocol


def extract(dataset,batch_size=32):
    from PIL import Image
    from torch.utils.data import Dataset,DataLoader
    from torchvision.models import vgg16,VGG16_Weights
    protocol=freeze()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=strong.serial.DEVICE
    manifest=read(CLEAN/dataset/'manifest.json'); rows=manifest['rows']
    for name,h in protocol['clean_inputs'][dataset].items():assert sha(CLEAN/dataset/name)==h
    model=vgg16(weights=None)
    model.load_state_dict(torch.load(protocol['checkpoint'],map_location='cpu',weights_only=True))
    model.eval().to(device)
    transform=VGG16_Weights.IMAGENET1K_V1.transforms()
    captured={}
    handles=[model.classifier[k].register_forward_hook(
        lambda module,inputs,output,name=name:captured.__setitem__(name,output.detach().clone()))
        for name,k in (('fc1',1),('fc2',4))]
    with np.load(CLEAN/dataset/'activations.npz') as z:
        meta={k:z[k] for k in ('category','concept','fold','dataset_index')}
        clean_first={k:z[k][:8] for k in ('fc1','fc2','logits')}
    first=[]
    for row in rows[:8]:
        with Image.open(row['path']) as im:first.append(transform(im.convert('RGB')))
    with torch.inference_mode():
        logits=model(torch.stack(first).to(device))
    replay=max(float(np.abs(captured[k].cpu().numpy()-clean_first[k]).max()) for k in ('fc1','fc2'))
    replay=max(replay,float(np.abs(logits.cpu().numpy()-clean_first['logits']).max()))
    assert replay<2e-4

    class Images(Dataset):
        def __init__(self,mode,level,repeat):self.mode,self.level,self.repeat=mode,level,repeat
        def __len__(self):return len(rows)
        def __getitem__(self,index):
            row=rows[index]
            assert sha(row['path'])==row['sha256']
            with Image.open(row['path']) as im:image=transform(im.convert('RGB'))
            image,diagnostic=corrupt(image,dataset,index,self.repeat,self.mode,self.level)
            return image,index,diagnostic
    try:
        for mode,level in DEGRADATIONS:
            for repeat in REPEATS:
                dest=INPUTS/dataset/mode/f'repeat{repeat}'
                if (dest/'complete.json').exists():
                    for p,h in read(dest/'complete.json')['files'].items():assert sha(dest/p)==h
                    print('EXISTS',dataset,mode,repeat,flush=True);continue
                started=time.monotonic()
                arrays={k:np.empty((len(rows),size),np.float32) for k,size in (('fc1',4096),('fc2',4096),('logits',1000))}
                diagnostics=np.empty((len(rows),8),np.float64)
                loader=DataLoader(Images(mode,level,repeat),batch_size=batch_size,shuffle=False,num_workers=0)
                bridge_error=0.
                with torch.inference_mode():
                    for images,indices,diag in loader:
                        logits=model(images.to(device));ix=indices.numpy()
                        for k in ('fc1','fc2'):arrays[k][ix]=captured[k].cpu().numpy()
                        arrays['logits'][ix]=logits.cpu().numpy();diagnostics[ix]=diag.numpy()
                        predicted=torch.relu(captured['fc1']@model.classifier[3].weight.T+model.classifier[3].bias)
                        bridge_error=max(bridge_error,float((predicted-captured['fc2']).abs().max()))
                        if int(indices[-1])//512!=int(indices[0])//512 or int(indices[-1])==len(rows)-1:
                            print('EXTRACT',dataset,mode,repeat,int(indices[-1])+1,'/',len(rows),'seconds',round(time.monotonic()-started,1),flush=True)
                assert bridge_error<1e-4
                assert np.unique(diagnostics[:,0]).size==len(rows)
                assert all(np.isfinite(a).all() for a in arrays.values())
                dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/'activations.npz',**arrays,**meta)
                np.savez_compressed(dest/'degradation_diagnostics.npz',row=np.arange(len(rows)),values=diagnostics)
                write_json(dest/'extraction.json',dict(dataset=dataset,mode=mode,level=level,repeat=repeat,rows=len(rows),
                    clean_first8_replay_error=replay,bridge_replay_error=bridge_error,
                    diagnostics_mean=diagnostics[:,1:].mean(0).tolist(),diagnostics_min=diagnostics[:,1:].min(0).tolist(),
                    diagnostics_max=diagnostics[:,1:].max(0).tolist(),seconds=time.monotonic()-started))
                write_json(dest/'complete.json',dict(protocol_sha256=sha(INPUTS/'protocol.json'),files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
                print('CACHE DONE',dataset,mode,repeat,'seconds',round(time.monotonic()-started,1),flush=True)
    finally:
        for h in handles:h.remove()
    assert sha(protocol['checkpoint'])==protocol['checkpoint_sha256']
    print('EXTRACTION COMPLETE',dataset,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','extract'))
    parser.add_argument('--dataset',choices=('imagenet','things'))
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    if args.action=='freeze':print('FROZEN INPUTS',freeze()['degradations'])
    else:
        for d in ([args.dataset] if args.dataset else ['imagenet','things']):extract(d,args.batch_size)
