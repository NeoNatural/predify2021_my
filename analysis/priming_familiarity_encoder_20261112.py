"""Compare matched encoder-only training arms on cached Conv5 training folds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

from priming_familiarity_20261112 import BATCH, DECAY, Familiarity, auc, calibrate, checked, loss
from priming_familiarity_data_20261112 import OUT, paths
from priming_familiarity_diagnosis_20261112 import NEGATIVES, SIDES, aucs, regions
from priming_familiarity_report_20261112 import write
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_object_precision_20261112 import fitted
from priming_sharpen_same_concept_data_20261112 import PLAN, frozen
from priming_sharpen_search_20261112 import DATASETS, control_gate, read, save
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


FOLDER = OUT / 'encoder_followup'
DESIGN = FOLDER / 'design.json'
RESULT = FOLDER / 'frozen.json'
ARMS = ('checkpoint', 'continued', 'paired')
EPOCHS = 16
LR = .0002
WEIGHT_DECAY = .001
PAIR_WEIGHT = .25
TEMPERATURE = .1
SEED = 2026111241


def design():
    return dict(
        arms=ARMS, epochs=EPOCHS, batch=BATCH, lr=LR, weight_decay=WEIGHT_DECAY,
        pair_weight=PAIR_WEIGHT, temperature=TEMPERATURE, shuffle_seed=SEED,
        fit_folds=[1, 2], validation_fold=3, target_step=9, primary_negative='same_concept',
        screen='By dataset: at least one continuation arm improves both primary AUC point estimates over checkpoint',
        familiarity_freeze_sha256=digest(OUT / 'frozen.json'), chain_plan_sha256=digest(PLAN),
        source_sha256=digest(Path(__file__)),
    )


def register():
    planned = design() | {'arms': list(ARMS)}
    if DESIGN.exists():
        assert read(DESIGN) == planned
    else:
        save(DESIGN, planned)


def load_model(state, train_encoder=False):
    model = Familiarity().cuda()
    model.load_state_dict(state)
    model.requires_grad_(False)
    model.encoder.requires_grad_(train_encoder)
    return model.train(train_encoder)


def probes(model, target, previews, slot_mask, central_mask):
    z, p = model.encode(target), model.encode(previews)
    reference = model.reference(p)
    frames = reference.transpose(-1, -2) @ p / p.shape[-2]
    memory = DECAY * frames[:, :, 0] + frames[:, :, 1]
    raw = model.read_encoded(z, model.reference(z), memory)
    center = F.normalize(z[:, 0, central_mask].mean(1), dim=-1)
    indices = slot_mask[:, None, :, None].expand(-1, p.shape[1], -1, p.shape[-1])
    patch = F.normalize(p[:, :, 1].gather(2, indices).mean(2), dim=-1)
    similarity = (center[:, None] * patch).sum(-1)
    return raw, similarity


def train(state, x, p, slots, central, train_ids, weight):
    model = load_model(state, train_encoder=True)
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = torch.Generator().manual_seed(SEED)
    steps = 0
    for _ in range(EPOCHS):
        shuffled = train_ids[torch.randperm(len(train_ids), generator=rng).to(train_ids.device)]
        for batch in shuffled.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            raw, sim = probes(model, x[batch], p[batch], slots[batch], central)
            ranking = F.softplus((sim[:, 1] - sim[:, 0]) / TEMPERATURE).mean()
            objective = loss(model, raw) + weight * ranking
            assert torch.isfinite(objective)
            objective.backward()
            optimizer.step()
            steps += 1
    trained = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for key in state.keys() - {'encoder.weight', 'encoder.bias'}:
        assert torch.equal(trained[key], state[key]), key
    assert any(not torch.equal(trained[key], state[key]) for key in ('encoder.weight', 'encoder.bias'))
    return trained, steps


@torch.inference_mode()
def validate(state, x, p, slots, central, dev_ids):
    model = load_model(state)
    result = [probes(model, x[j], p[j], slots[j], central) for j in dev_ids.split(BATCH)]
    return tuple(torch.cat(values).cpu().numpy() for values in zip(*result))


def fit_dataset(ds, original, kernel, table):
    path, meta = paths(ds, 'train')
    assert digest(path) == original['train_sha256'] and digest(meta) == original['train_audit_sha256']
    model_path = OUT / ds / 'circuit.pth'
    base = torch.load(model_path, weights_only=True, map_location='cpu')
    with np.load(path) as data:
        x = torch.from_numpy(data['conv']).cuda()
        p = torch.from_numpy(data['memory']).cuda()
        f, old_p, folds, ids = (data[k] for k in ('fc2', 'old_preview', 'folds', 'rows'))
    assert set(folds) == {1, 2, 3}
    rows = read(OUT.parent / ds / 'chain_manifest.json')['trials']
    np.testing.assert_array_equal(folds, [r['fold'] for r in rows])
    np.testing.assert_array_equal(ids, [r['dataset_index'] for r in rows])
    assert {r['slot'] // 2 for r in rows} == {1}
    mask = regions()
    central = torch.as_tensor(mask['center'], device='cuda')
    slots = torch.as_tensor(np.stack([mask[SIDES[r['slot'] % 2]] for r in rows]), device='cuda')
    train_ids = torch.as_tensor(np.flatnonzero(folds < 3), device='cuda')
    dev_ids = torch.as_tensor(np.flatnonzero(folds == 3), device='cuda')
    dev_rows = [r for r in rows if r['fold'] == 3]
    paired = PairedConcepts(np.array([r['category_index'] for r in dev_rows]), np.array([r['concept'] for r in dev_rows]))
    states = {'checkpoint': base}
    steps = {'checkpoint': 0}
    for arm, weight in (('continued', 0.), ('paired', PAIR_WEIGHT)):
        states[arm], steps[arm] = train(base, x, p, slots, central, train_ids, weight)
    assert steps['continued'] == steps['paired'] == EPOCHS * ((len(train_ids) + BATCH - 1) // BATCH)
    validation = {arm: validate(state, x, p, slots, central, dev_ids) for arm, state in states.items()}
    for i, name in enumerate(('auc_same', 'auc_different'), 1):
        np.testing.assert_allclose(auc(validation['checkpoint'][0], i), original['chosen'][name], atol=.001)

    results = {}
    for arm in ARMS:
        full, local = validation[arm]
        results[arm] = {
            'local': aucs(local, paired.concept_weights),
            'full_raw': aucs(full, paired.concept_weights),
        }
    primary = NEGATIVES[0]
    passed = [arm for arm in ARMS[1:] if all(results[arm][probe][primary][0] > results['checkpoint'][probe][primary][0]
                                             for probe in ('local', 'full_raw'))]
    for arm, measurements in results.items():
        for probe, negative_results in measurements.items():
            for negative, values in negative_results.items():
                table.append(dict(dataset=ds, arm=arm, reference='', probe=probe, negative=negative, **estimate(values)))
                refs = () if arm == 'checkpoint' else ('checkpoint',) if arm == 'continued' else ('checkpoint', 'continued')
                for ref in refs:
                    table.append(dict(dataset=ds, arm=arm, reference=ref, probe=probe, negative=negative,
                                      **estimate(values - results[ref][probe][negative])))
    print('SCREEN', ds, {k: {p: round(float(r[primary][0]), 4) for p, r in m.items()} for k, m in results.items()},
          'eligible', passed, flush=True)

    if passed:
        bank, _, cuts = fitted(ds, kernel)
        f, old_p = f[folds == 3], old_p[folds == 3]
        old_gate = np.column_stack([control_gate(bank.bank.similarity(f[:, i], old_p[:, i]).max(1), cuts[.5])
                                    for i in range(3)])
    frozen_models = {}
    for arm in ARMS:
        dest = FOLDER / ds / f'{arm}.pth'
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix('.tmp')
        torch.save({k: v for k, v in states[arm].items() if k.startswith('encoder.')}, tmp)
        tmp.replace(dest)
        cal = calibrate(validation[arm][0], old_gate) if passed else None
        frozen_models[arm] = dict(encoder_sha256=digest(dest), calibration=cal, optimizer_steps=steps[arm])
    return dict(eligible=passed, models=frozen_models, train_sha256=digest(path),
                base_circuit_sha256=digest(model_path), fit_n=len(train_ids), fold3_n=len(dev_rows))


def check_result():
    register()
    report = read(RESULT)
    assert report['design_sha256'] == digest(DESIGN)
    for ds, item in report['datasets'].items():
        assert digest(OUT / ds / 'circuit.pth') == item['base_circuit_sha256']
        for arm, state in item['models'].items():
            assert digest(FOLDER / ds / f'{arm}.pth') == state['encoder_sha256']
    return report


def main():
    original = checked()
    frozen()
    register()
    if RESULT.exists():
        check_result()
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    kernel = weights()
    table = []
    result = {ds: fit_dataset(ds, original['datasets'][ds], kernel, table) for ds in DATASETS}
    write('encoder_fold3', table)
    save(RESULT, dict(design_sha256=digest(DESIGN), datasets=result))
    print('ENCODERS FROZEN', digest(RESULT), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
