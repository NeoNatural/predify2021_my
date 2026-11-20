"""Match the original common-readout float32 inference conversion.

Saved features are float16. Scaler.transform preserves input dtype, so direct
float16 evaluation can overflow when a t9-trained scaler sees an early state.
Conversion restores the intended common-readout numerical policy; no clipping,
feature selection, model fitting, or scaler parameters are changed.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import analysis.t9_readout_tripartite_20261027 as study
original_data=study.a.data
def float32_data(*args,**kwargs):
    d=original_data(*args,**kwargs)
    d['features']={c:x.astype(np.float32) for c,x in d['features'].items()}
    return d
if __name__=='__main__':
    study.immutable_json(study.DEST/'execution_precision.json',dict(
        wrapper_sha=study.sha(__file__),fitting_source_sha=study.sha(study.__file__),
        change='Convert stored float16 features to float32 before scaling, matching original common readout.',
        unchanged='training features, weights, scaler, classifier, C, evaluation labels, statistical rules',
        reason='Initial supplemental evaluation rejected float16 overflow; no supplemental scores had been saved.'))
    study.a.data=float32_data
    study.main()
