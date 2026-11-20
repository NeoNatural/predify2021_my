"""Audit complete raw records, exact presented tensors and freeze chronology."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.tripartite_20261027 import OUT,DOC
from predify_experiments.supplementary import load
from predify_experiments.readout_init_20261012 import sha


def main():
    implementation=json.loads((OUT/'implementation.json').read_text());summary={};sequences=0
    shared_grey=None
    for dataset in ('imagenet','things'):
        base=OUT/dataset;m=json.loads((base/'manifest.json').read_text());summary[dataset]={}
        for split,count in [('train',240),('dev',96),('test',192),('old_layout',8)]:
            directory=base/split;paths=list(directory.glob('image_*.pckl'));assert len(paths)==count
            frozen=json.loads((base/'freeze.json').read_text())
            if split=='test':assert (base/'freeze.json').stat().st_mtime < min(p.stat().st_mtime for p in paths)
            grey=json.loads((directory/'grey_state_hash.json').read_text())
            if shared_grey is None:shared_grey=grey
            else:assert grey==shared_grey,'Grey snapshot changed'
            n=0
            for p in paths:
                d=load(p);index=d['trial']['dataset_index'];noise=d['noise']
                with np.load(directory/f'stimulus_{index}_seed_{noise}.npz') as stimulus:
                    target=stimulus['target'];clean=stimulus['clean']
                    digest=hashlib.sha256(target.tobytes()).hexdigest()
                assert all(r['target_hash']==digest for r in d['records'])
                for r in d['records']:
                    assert np.isfinite(r['fc2']).all() and np.isfinite(r['logits']).all()
                    assert np.isfinite(r['activation']).all() and np.isfinite(r['internal_error']).all()
                    assert r['target_W_frozen']
                assert d['source_sha']==implementation['source_hashes']['predify_experiments/tripartite_20261027.py']
                n+=len(d['records'])
                with np.load(directory/f'previews_{index}.npz') as previews:
                    assert len(previews.files)==10 and all(previews[key].shape==(3,224,224) for key in previews.files)
            summary[dataset][split]=dict(files=len(paths),sequences=n,actual_input_hashes_match=True,
                finite=True,gray_identical=True,preview_arrays_present=True,freeze_before_test=True)
            sequences+=n
        assert sha(base/'head.pckl')==frozen['head_sha']
        assert sha(DOC/'预先方案.md')==frozen['protocol_sha']
    assert sequences==29680
    result=dict(passed=True,formal_main_sequences=29568,old_layout_sequences=112,
        total=sequences,datasets=summary,peak_significance_claimed=False)
    (DOC/'final_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
