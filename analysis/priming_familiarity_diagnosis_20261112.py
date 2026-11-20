"""Diagnose the frozen familiarity circuit using cached features and scores only."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

from priming_familiarity_20261112 import DECAY, Familiarity, checked, gate
from priming_familiarity_data_20261112 import OUT, TRAIN_HISTORIES, paths
from priming_familiarity_report_20261112 import paired_auc, write
from priming_familiarity_verify_20261112 import SCOPES, apply_gate, archive, target_direction
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_confirm_20261112 import contrast, metrics
from priming_sharpen_object_precision_20261112 import fitted
from priming_sharpen_same_concept_data_20261112 import frozen
from priming_sharpen_search_20261112 import DATASETS, DOC, read, save
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


POSITIONS = {'left': 5, 'center': 80, 'right': 155}
SIDES = ('left', 'right')
NEGATIVES = ('same_concept', 'different')


def regions():
    # Conv5_3 is after four 2x2 pools. These are cell-center masks in the 224px input grid.
    coord = 8 + 16 * np.arange(14)
    y = (80 <= coord) & (coord < 144)
    return {k: np.flatnonzero((y[:, None] & ((left <= coord) & (coord < left + 64))).ravel())
            for k, left in POSITIONS.items()}


def aucs(scores, weights):
    return {name: paired_auc(scores[:, 0], scores[:, i], weights) for i, name in enumerate(NEGATIVES, 1)}


def tensor(x):
    return torch.from_numpy(x)


def tokens(x):
    return x.float().flatten(-2).transpose(-1, -2)


def normalized_matrix(a, norm):
    return a * (norm / a.flatten(-2).norm(dim=-1, keepdim=True).clamp_min(1e-12))[..., None]


def detection_rows(table, dataset, experiment, method, measurements, baseline=None):
    for name, values in measurements.items():
        table.append(dict(dataset=dataset, experiment=experiment, method=method, negative=name, comparison='absolute',
                          **estimate(values)))
        if baseline is not None:
            table.append(dict(dataset=dataset, experiment=experiment, method=method, negative=name,
                              comparison='minus_reference', **estimate(values - baseline[name])))


def across_preview_positions(z, rows, masks):
    """Pair each eligible query with other trials showing its image on both sides."""
    src, side, trial, vectors = [], [], [], []
    for j, row in enumerate(rows):
        seen = set()
        for h, history in enumerate(TRAIN_HISTORIES):
            images = row['previews'].copy()
            images[row['slot']] = row['primes'][history]
            for s in (2, 3):
                key = images[s], s % 2
                if key in seen:
                    continue
                seen.add(key)
                src.append(images[s])
                side.append(s % 2)
                trial.append(j)
                vectors.append(z[j, h, 1, masks[SIDES[s % 2]]].mean(0))
    u = F.normalize(torch.stack(vectors), dim=-1)
    cosine = (u @ u.T).numpy()
    src, side, trial = map(np.asarray, (src, side, trial))
    repeated = defaultdict(list)
    for q, source in enumerate(src):
        query = []
        for destination in (side[q], 1 - side[q]):
            valid = (side == destination) & (trial != trial[q])
            pos = cosine[q, valid & (src == source)]
            neg = cosine[q, valid & (src != source)]
            if len(pos) == 0 or len(neg) == 0:
                break
            query.append(float(((pos[:, None] > neg) + .5 * (pos[:, None] == neg)).mean()))
        if len(query) == 2:
            repeated[int(source)].append(query)
    values = np.stack([np.mean(value, axis=0) for value in repeated.values()])
    draws = np.random.default_rng(2026111230).integers(0, len(values), (4000, len(values)))
    means = np.vstack((values.mean(0), values[draws].mean(1)))
    return dict(images=len(values), same=estimate(means[:, 0]), opposite=estimate(means[:, 1]),
                opposite_minus_same=estimate(means[:, 1] - means[:, 0]))


@torch.inference_mode()
def feature_diagnosis(dataset, registration, table):
    path, meta = paths(dataset, 'train')
    spec = registration['datasets'][dataset]
    assert digest(path) == spec['train_sha256'] and digest(meta) == spec['train_audit_sha256']
    full_rows = read(OUT.parent / dataset / 'chain_manifest.json')['trials']
    with np.load(path) as f:
        use = f['folds'] == 3
        x, p = (tensor(f[k][use]) for k in ('conv', 'memory'))
        rows = [r for r, flag in zip(full_rows, use) if flag]
        np.testing.assert_array_equal(f['rows'][use], [r['dataset_index'] for r in rows])
    assert x.shape[-2:] == (14, 14) and {r['slot'] // 2 for r in rows} == {1}
    paired = PairedConcepts(np.array([r['category_index'] for r in rows]), np.array([r['concept'] for r in rows]))
    w = paired.concept_weights
    masks = regions()
    z_model = Familiarity().eval()
    z_model.load_state_dict(torch.load(OUT / dataset / 'circuit.pth', map_location='cpu', weights_only=True))
    z = z_model.encode(x)
    zp = z_model.encode(p)
    hp = z_model.reference(zp)
    frames = hp.transpose(-1, -2) @ zp / zp.shape[-2]
    full = DECAY * frames[:, :, 0] + frames[:, :, 1]
    response = z_model.read_encoded(z, z_model.reference(z), full).numpy()
    for i, name in enumerate(NEGATIVES):
        np.testing.assert_allclose(aucs(response, w)[name][0], spec['chosen'][('auc_same', 'auc_different')[i]], atol=.001)
    obj, opposite = [], []
    for i, row in enumerate(rows):
        side = row['slot'] % 2
        for dest, mask in ((obj, masks[SIDES[side]]), (opposite, masks[SIDES[1 - side]])):
            dest.append(hp[i, :, 1, mask].transpose(-1, -2) @ zp[i, :, 1, mask] / zp.shape[-2])
    obj = torch.stack(obj)
    opposite = torch.stack(opposite)
    other = (full - obj).clamp_min(0)
    # Match the memory scale per trial to its mean full-field norm across histories.
    scale = full.flatten(-2).norm(dim=-1).mean(1, keepdim=True)[..., None]
    normed = {name: normalized_matrix(a, scale) for name, a in
              (('full', full), ('changed_slot', obj), ('other_slot', opposite), ('outside_changed_slot', other))}
    full_read = {}
    for read_name, query in (('full_read', z[:, 0:1]), ('center_read', z[:, 0:1, masks['center']])):
        h = z_model.reference(query)
        measured = {name: aucs(z_model.read_encoded(query, h, matrix).numpy(), w) for name, matrix in normed.items()}
        base = measured['full']
        for name in normed:
            detection_rows(table, dataset, read_name, name, measured[name], base if name != 'full' else None)
        if read_name == 'full_read':
            full_read = measured
        else:
            for name in normed:
                detection_rows(table, dataset, 'center_vs_full_read', name, measured[name], full_read[name])
        detection_rows(table, dataset, read_name, 'raw_full', aucs(z_model.read_encoded(query, h, full).numpy(), w), base)
    obj_share = (obj.flatten(-2).norm(dim=-1) / full.flatten(-2).norm(dim=-1)).mean(0)
    positions = {}
    conv5 = None
    for name, target, preview in (('conv5', tokens(x), tokens(p)), ('code64', z, zp)):
        center = F.normalize(target[:, 0, masks['center']].mean(1), dim=-1)
        patch = torch.stack([preview[j, :, 1, masks[SIDES[row['slot'] % 2]]].mean(1)
                             for j, row in enumerate(rows)])
        cross = (center[:, None] * F.normalize(patch, dim=-1)).sum(-1).numpy()
        measured = aucs(cross, w)
        detection_rows(table, dataset, 'center_to_periphery', name, measured, conv5)
        if name == 'conv5':
            conv5 = measured
        positions[name] = across_preview_positions(preview, rows, masks)
    return response, dict(fold3=len(rows), object_norm_fraction=obj_share.tolist(), position=positions)


def soft_calibration(scores, threshold, slope):
    positive = np.maximum(scores - threshold, 0).astype(np.float64)
    balance = np.array([.5, .25, .25])
    target = gate(scores, threshold, slope).mean(0) @ balance

    def mean_difference(k):
        return (-np.expm1(-k * positive)).mean(0) @ balance - target

    soft_slope = brentq(mean_difference, 0., 1e6)
    return soft_slope, float(target)


def calibration_diagnosis(dataset, training_scores, registration, cfg, bank, head, table, downstream):
    cal = registration['datasets'][dataset]['calibration']
    threshold, slope = cal['threshold'], cal['slope']
    soft_slope, mean = soft_calibration(training_scores, threshold, slope)
    for scope in SCOPES:
        data = archive(dataset, scope)
        pair = PairedConcepts(data['label'], data['concept'])
        score = data['score_new'][:, :3, 9]
        positive = np.maximum(score - threshold, 0)
        values = {'raw': score, 'floor_only': positive, 'ceiling_only': np.minimum(score, threshold + 1 / slope),
                  'clipped': data['gate_new'][:, :3, 9], 'smooth': -np.expm1(-soft_slope * positive.astype(np.float64))}
        base = aucs(score, pair.concept_weights)
        for name, v in values.items():
            detection_rows(table, dataset, f'calibration_{scope}', name, aucs(v, pair.concept_weights), base)
        detection_rows(table, dataset, f'smooth_vs_clipped_{scope}', 'smooth',
                       aucs(values['smooth'], pair.concept_weights), aucs(values['clipped'], pair.concept_weights))
        clipped = {}
        for name in ('clipped', 'smooth'):
            g = values[name]
            x = data['fc2'][:, :, 9]
            r = [metrics(apply_gate(x[:, i], target_direction(bank, x[:, i], cfg), g[:, i], cfg), head, pair)
                 for i in range(3)]
            for h, ref in enumerate(NEGATIVES, 1):
                for k, v in contrast(r[0], r[h]).items():
                    if name == 'clipped':
                        clipped[ref, k] = v
                    downstream.append(dict(dataset=dataset, scope=scope, calibration=name, reference=ref, metric=k,
                                           **estimate(v)))
                    if name == 'smooth':
                        downstream.append(dict(dataset=dataset, scope=scope, calibration='smooth_minus_clipped',
                                               reference=ref, metric=k, **estimate(v - clipped[ref, k])))
    return dict(threshold=threshold, clipped_slope=slope, soft_slope=soft_slope, fold3_balanced_mean=mean)


def main():
    reg = checked()
    cfg = frozen()['configs']['target_only']
    kernel = weights()
    table, downstream, summaries = [], [], {}
    for ds in DATASETS:
        training_scores, diagnostics = feature_diagnosis(ds, reg, table)
        bank, head, _ = fitted(ds, kernel)
        diagnostics['calibration'] = calibration_diagnosis(ds, training_scores, reg, cfg, bank, head, table, downstream)
        summaries[ds] = diagnostics
    write('diagnosis_auc', table)
    write('diagnosis_fc2', downstream)
    save(DOC / 'chain/familiarity_diagnosis_summary.json', summaries)


if __name__ == '__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        main()
