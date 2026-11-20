"""Compare frozen full-field and two-bank familiarity on cached fold 3 Conv5."""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from priming_familiarity_20261112 import Familiarity, checked
from priming_familiarity_data_20261112 import OUT, TRAIN_HISTORIES, paths
from priming_familiarity_report_20261112 import paired_auc, write
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_search_20261112 import DATASETS, DOC, read, save
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


HISTORIES = ('exact', 'same_concept', 'different')
COMPARISONS = (('average', 'full'), ('competition', 'full'), ('competition', 'average'))


@torch.inference_mode()
def scores(model, target, preview):
    assert target.shape[-2:] == preview.shape[-2:] == (14, 14)
    z = model.encode(target)
    full = model.memory(preview)
    left, right = (model.memory(half) for half in preview.split(7, dim=-1))
    combined = (left + right) / 2
    torch.testing.assert_close(combined, full, atol=2e-7, rtol=2e-5)
    results, read_errors = {}, {}
    for name, query in (('native_target', z), ('fixed_exact_target', z[:, :1])):
        ref = model.reference(query)
        whole = model.read_encoded(query, ref, full)
        reconstructed = model.read_encoded(query, ref, combined)
        torch.testing.assert_close(reconstructed, whole, atol=2e-6, rtol=2e-5)
        ls = model.read_encoded(query, ref, left)
        rs = model.read_encoded(query, ref, right)
        average = (ls + rs) / 2
        competition = torch.maximum(ls, rs)
        assert torch.all(average <= whole + 2e-6)  # Rectified inhibition is concave in memory.
        assert torch.all(average <= competition)
        empty = model.read_encoded(query, ref, torch.zeros_like(full))
        assert torch.count_nonzero(empty) == 0
        results[name] = dict(full=whole.numpy(), average=average.numpy(), competition=competition.numpy())
        read_errors[name] = float((reconstructed - whole).abs().max())
    for arm, values in results['native_target'].items():
        np.testing.assert_allclose(values[:, 0], results['fixed_exact_target'][arm][:, 0], atol=2e-6, rtol=2e-5)
    return results, dict(matrix_max_abs_error=float((combined - full).abs().max()), read_max_abs_error=read_errors,
                         cleared_memory_zero=True)


def fold3(dataset, registered):
    path, note = paths(dataset, 'train')
    assert digest(path) == registered['train_sha256'] and digest(note) == registered['train_audit_sha256']
    info = read(note)
    assert tuple(info['histories']) == TRAIN_HISTORIES and info['step'] == 9
    rows = read(OUT.parent / dataset / 'chain_manifest.json')['trials']
    with np.load(path) as archive:
        keep = archive['folds'] == 3
        target, preview = (torch.from_numpy(archive[k][keep]) for k in ('conv', 'memory'))
        np.testing.assert_array_equal(archive['rows'], [r['dataset_index'] for r in rows])
        np.testing.assert_array_equal(archive['folds'], [r['fold'] for r in rows])
    selected = [r for r in rows if r['fold'] == 3]
    paired = PairedConcepts(np.array([r['category_index'] for r in selected]), np.array([r['concept'] for r in selected]))
    model = Familiarity().eval()
    model.load_state_dict(torch.load(OUT / dataset / 'circuit.pth', map_location='cpu', weights_only=True))
    return model, target, preview, paired


def evaluate(dataset, registered, rows):
    model, target, preview, paired = fold3(dataset, registered)
    measurements, audit = scores(model, target, preview)
    weights = paired.concept_weights
    for query, outcomes in measurements.items():
        def emit(arm, ref, history, metric, values):
            rows.append(dict(dataset=dataset, query=query, arm=arm, reference=ref, history=history,
                             metric=metric, **estimate(values)))

        for h in (1, 2):
            negative = HISTORIES[h]
            auc = {arm: paired_auc(v[:, 0], v[:, h], weights) for arm, v in outcomes.items()}
            if query == 'native_target':
                key = ('auc_same', 'auc_different')[h - 1]
                np.testing.assert_allclose(auc['full'][0], registered['chosen'][key], atol=.001)
            for arm, result in auc.items():
                emit(arm, '', negative, 'auc', result)
            for arm, ref in COMPARISONS:
                emit(arm, ref, negative, 'auc', auc[arm] - auc[ref])
        boost = outcomes['competition'] - outcomes['average']
        for h, history in enumerate(HISTORIES):
            emit('competition', 'average', history, 'mean_score_delta', weights @ boost[:, h])
        for h in (1, 2):
            emit('competition', 'average', HISTORIES[h], 'exact_minus_control_score_delta',
                 weights @ (boost[:, 0] - boost[:, h]))
    return dict(targets=len(target), **audit)


def main():
    reg = checked()
    rows = []
    audit = {dataset: evaluate(dataset, reg['datasets'][dataset], rows) for dataset in DATASETS}
    write('partition_fold3', rows)
    save(DOC / 'chain/familiarity_partition_audit.json', audit)


if __name__ == '__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        main()
