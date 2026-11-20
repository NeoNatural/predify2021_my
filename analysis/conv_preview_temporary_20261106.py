"""Corrected preview-local convolution pilot, isolated and resumable by group."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import conv_development_20261104 as old
from predify_experiments.conv_preview_memory_20261106 import DEFAULTS, PreviewMemory, PreviewOperator
from predify_experiments.sequence import reset_hebbian_state, run_recurrent_sequence

OUT = Path('E:/Predify_Experiments_20261105/conv_preview_temporary_20261106')
DOC = ROOT/'docs/conv_preview_temporary_20261106'
CASES = {'clip': ('clip', 'none'), **{f'{family}_{source}': (family, source)
         for family in ('gate', 'multiply') for source in ('none', 'rep4', 'prd5')}}
LAYERS = ('conv4', 'conv5', 'fc1', 'fc2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def writejson(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    temp.replace(path)


def immutable(path, value):
    if path.exists():
        assert json.loads(path.read_text(encoding='utf8')) == value, f'Changed protocol: {path}'
    else:
        writejson(path, value)


def status(**kw):
    writejson(DOC/'status.json', dict(updated=time.time(), **kw))


def prepare():
    assert OUT.parent.is_dir(), 'E: output parent missing; no automatic fallback'
    files = [Path(__file__), ROOT/'predify_experiments/conv_preview_memory_20261106.py',
             ROOT/'analysis/conv_development_20261104.py',
             ROOT/'analysis/conv_development_core_20261104.py',
             ROOT/'predify_experiments/sequence.py', ROOT/'Hebbian-CNN/predify_hebb_inject.py',
             ROOT/'analysis/report_conv_preview_temporary_20261106.py']
    protocol = dict(version=1, config=DEFAULTS, cases=CASES, fc_hebb=False, legacy_conv_hebb=False,
                    memory_source='current trial only; no training-set E matrices are opened',
                    timing='reset W and PCoder to common gray; preview1 2 steps; write endpoint; '
                           'preview2 2 steps using W1; write endpoint; freeze; target 15 steps',
                    write='H_j=positive spatial mean outer products, source RMS then separate global '
                          'row-sum normalization; A_j=.5*A_(j-1)+.5*H_j; A2=.25*H1+.5*H2',
                    structure='all off-diagonal same-position channel pairs; 8 neighboring columns; '
                              'mutual top32 across channel-offset pairs from accumulated inter trace; '
                              'shared weights at every spatial position',
                    target='read only; neither decay nor writes; apply to rep4 before its prediction '
                           'and downstream; prd5 source is current endpoint after pcoder5 forward',
                    source_activation='rep4 is the actual modified endpoint rep4; prd5 is unmodified '
                                      'pcoder5 prediction of that same rep4 coordinate system',
                    nohebb='same gate/multiply, inhibition, clipping and PCoder carry; W identically zero',
                    storage='FC1/FC2 full features; per-trial activation/density and unit-norm full-map '
                            'Gram matrices per timestep; no convolutional raw maps persisted',
                    readout='fixed StandardScaler + LinearSVC(C=1), clean t9 FC2 folds1-3; dev fold4 only; '
                            'fold0 untouched; no outcome-dependent parameter search',
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files})
    # JSON-roundtrip canonicalizes tuples before comparing on resume.
    protocol = json.loads(json.dumps(protocol))
    immutable(OUT/'protocol.json', protocol); immutable(DOC/'protocol.json', protocol)
    for ds in ('imagenet', 'things'):
        manifest, train, dev = old.split(ds)
        scale_source = OUT.parent/f'conv_development_20261104/{ds}/rep4/graph_info.json'
        info = json.loads(scale_source.read_text(encoding='utf8'))
        assert info['train_count'] == len(train) and info['site'] == 'rep4'
        spec = dict(dataset=ds, categories=manifest['categories'], training=train, development=dev,
                    scale=info['scale'], scale_source=str(scale_source), scale_source_sha256=sha(scale_source),
                    source_manifest_sha256=sha(old.c.native.OUT/ds/'manifest.json'),
                    noise_seeds=[0,2921], identities=['exact','different_category'],
                    protocol_sha256=sha(OUT/'protocol.json'), expected_sequences=len(dev)*4*len(CASES))
        immutable(OUT/ds/'manifest.json', spec); immutable(DOC/f'{ds}_manifest.json', spec)
        print('FROZEN', ds, len(dev), 'targets;', spec['expected_sequences'], 'sequences', flush=True)


def fit_head(ds, spec):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from predify_experiments.supplementary import load, save
    file = OUT/ds/'reference_head.pckl'
    if file.exists(): return load(file)['head']
    x, y, hashes = [], [], {}
    for tr in spec['training']:
        path = old.c.native.OUT/ds/'formal'/f"reference_{tr['dataset_index']}.pckl"
        item = load(path); x.append(item['result']['fc2'][9]); y.append(tr['category_index'])
        hashes[str(tr['dataset_index'])] = sha(path)
    head = make_pipeline(StandardScaler(), LinearSVC(C=1, dual='auto', max_iter=200000, random_state=20261104))
    head.fit(np.asarray(x), y)
    assert head[-1].n_iter_ < 200000
    save(file, dict(head=head, training_indices=[t['dataset_index'] for t in spec['training']]))
    writejson(OUT/ds/'reference_head.json', dict(training_file_sha256=hashes, sha256=sha(file),
        training_accuracy=float(head.score(np.asarray(x), y)), classes=head.classes_.tolist()))
    return head


def run_trial(target, previews, op):
    c = old.c; memory = op.memory
    memory.reset(); op.calls = 0; reset_hebbian_state(c.layers); c.reset()
    assert memory.writes == 0 and not memory.intra.count_nonzero() and not memory.trace_inter.count_nonzero()
    audit = dict(reset=memory.stats(), preview_boundaries=[], source=op.source, family=op.family)
    features = {k: [] for k in LAYERS}; capture_steps = []
    with torch.no_grad():
        for stage, pv in enumerate(previews):
            before = memory.fingerprint()
            run_recurrent_sequence(c.model, pv[None], 2, hebbian_layers=c.layers,
                                   commit_update=False, preserve_model_state=True)
            assert memory.fingerprint() == before
            if op.source != 'none':
                activity = c.model.pcoder4.rep if op.source == 'rep4' else c.model.pcoder5.prd
                info = memory.write(activity)
                info['source_sha256'] = hashlib.sha256(activity.detach().cpu().numpy().tobytes()).hexdigest()
            else:
                info = memory.stats()
            audit['preview_boundaries'].append(dict(stage=stage+1, **info))
        memory.freeze(); frozen = memory.fingerprint()
        versions = tuple(x._version for x in (memory.intra, memory.trace_inter, memory.inter))
        audit['before_target'] = frozen
        def after(t, _):
            assert versions == tuple(x._version for x in (memory.intra, memory.trace_inter, memory.inter))
            capture_steps.append(t)
            for layer, value in (('conv4', c.model.pcoder4.rep), ('conv5', c.model.pcoder5.rep),
                                 ('fc1', c.cache['fc1']), ('fc2', c.cache['fc2'])):
                a = value.detach().cpu().numpy().reshape(-1).copy()
                assert np.isfinite(a).all()
                features[layer].append(a)
        run_recurrent_sequence(c.model, target[None], 15, hebbian_layers=c.layers, commit_update=False,
                               preserve_model_state=True, after_step=after)
    audit['after_target'] = memory.fingerprint(); audit['writes'] = memory.writes
    assert audit['after_target'] == frozen and op.calls == 19 and capture_steps == list(range(15))
    assert memory.writes == (0 if op.source == 'none' else 2)
    assert all(not l.compute_enabled and not l.update_enabled for l in c.layers)
    assert all(not l.boost_weight.count_nonzero() for l in c.layers if hasattr(l, 'boost_weight'))
    return {k: np.stack(v) for k,v in features.items()}, audit


def compact_group(all_features):
    """Keep exact full-map dot-product geometry, without storing raw conv maps."""
    result = {}
    for layer in LAYERS:
        values = [item[layer] for item in all_features]
        result[layer+'_mean'] = np.array([v.mean(1, dtype=np.float64) for v in values])
        result[layer+'_mean_abs'] = np.array([np.abs(v).mean(1, dtype=np.float64) for v in values])
        result[layer+'_density'] = np.array([(v != 0).mean(1) for v in values])
        result[layer+'_negative'] = np.array([(v < 0).mean(1) for v in values])
        grams, norms = [], []
        for t in range(15):
            x = np.array([v[t] for v in values], dtype=np.float64)
            norm = np.linalg.norm(x, axis=1)
            z = x/np.maximum(norm[:,None], 1e-12)
            grams.append(z@z.T); norms.append(norm)
        result[layer+'_gram'] = np.stack(grams)
        result[layer+'_norm'] = np.stack(norms, axis=1)
        if layer.startswith('fc'): result[layer] = np.stack(values)
    return result


def run(datasets, smoke=False):
    prepare(); p4, p5 = old.initialize(); c = old.c
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    transform=c.VGG16_Weights.IMAGENET1K_V1.transforms(); filt=c.native.FilteredInput(torch)
    for ds in datasets:
        spec=json.loads((OUT/ds/'manifest.json').read_text(encoding='utf8')); fit_head(ds,spec)
        memory=PreviewMemory(512, device='cuda')
        if smoke:
            targets,pvs=c.test_inputs(ds,spec['development'][0],transform,filt,torch)
            checks={}
            for family in ('gate','multiply'):
                for source in ('none','rep4','prd5'):
                    op=PreviewOperator(memory,spec['scale'],family,source);p4.hebb=op
                    start=time.time();a,au=run_trial(targets[0],pvs['exact'],op)
                    b,bu=run_trial(targets[0],pvs['different_category'],op)
                    again,cu=run_trial(targets[0],pvs['exact'],op)
                    assert all(np.array_equal(a[k],again[k]) for k in LAYERS), 'Trial carry-over'
                    assert au['before_target']==cu['before_target']
                    if source!='none': assert au['before_target']!=bu['before_target'], 'No preview content sensitivity'
                    assert c.model.pcoder5.pmodule is p5
                    checks[f'{family}_{source}']=dict(seconds=time.time()-start, exact=au, different=bu,
                        repeat_exact_bitwise=True, pcoder5_prediction_module_unchanged=True)
                    print('SMOKE',ds,family,source,round(time.time()-start,2),flush=True)
            writejson(DOC/f'{ds}_smoke.json',checks)
            continue
        for case,(family,source) in CASES.items():
            folder=OUT/ds/case;folder.mkdir(exist_ok=True)
            for seed in spec['noise_seeds']:
                for ident in spec['identities']:
                    file=folder/f'{seed}_{ident}.npz';auditfile=file.with_suffix('.json')
                    if file.exists() and auditfile.exists():
                        saved=json.loads(auditfile.read_text(encoding='utf8'))
                        assert saved['file_sha256']==sha(file)
                        print('REUSE',ds,case,seed,ident,flush=True);continue
                    op=PreviewOperator(memory,spec['scale'],family,source);p4.hebb=op
                    all_features=[];audits=[];start=time.time()
                    for i,tr in enumerate(spec['development']):
                        status(stage='extracting',dataset=ds,case=case,seed=seed,identity=ident,
                               trial=i+1,total=len(spec['development']),complete=False)
                        targets,pvs=c.test_inputs(ds,tr,transform,filt,torch)
                        values,audit=run_trial(targets[seed],pvs[ident],op)
                        audit['dataset_index']=tr['dataset_index'];audits.append(audit);all_features.append(values)
                        assert c.model.pcoder5.pmodule is p5
                    result=compact_group(all_features)
                    result['indices']=np.array([tr['dataset_index'] for tr in spec['development']])
                    result['labels']=np.array([tr['category_index'] for tr in spec['development']])
                    c.native.atomic_npz(file,**result)
                    writejson(auditfile,dict(dataset=ds,case=case,seed=seed,identity=ident,
                        seconds=time.time()-start,trials=audits,file_sha256=sha(file)))
                    print('GROUP',ds,case,seed,ident,round(time.time()-start,2),'seconds',flush=True)
                    del all_features,result
    status(stage='smoke_complete' if smoke else 'extraction_complete',complete=True,datasets=datasets)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','smoke','run'))
    p.add_argument('--dataset',choices=('imagenet','things','both'),default='both');args=p.parse_args()
    if args.action=='prepare':prepare()
    else:run(['imagenet','things'] if args.dataset=='both' else [args.dataset],args.action=='smoke')
