"""Separate amplitude, target evidence, and preview evidence in the frozen run."""

from __future__ import annotations

import csv

import numpy as np
from threadpoolctl import threadpool_limits

from priming_sharpen_confirm_20261112 import contrast, metrics
from priming_sharpen_corroboration_20261112 import freeze, update
from priming_sharpen_object_precision_20261112 import features, fitted
from priming_sharpen_search_20261112 import DATASETS, DOC
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def main():
    cfg = freeze()['config']
    kernel = weights()
    table = []
    for dataset in DATASETS:
        bank, head, cuts = fitted(dataset, kernel)
        x, p, labels, meta = features(dataset, 'validation')
        exact, different, none = x.transpose(1, 0, 2)
        q = bank.query(exact, p[:, 0])
        q_different = bank.query(exact, p[:, 1])
        q_swap = dict(q, preview_posterior=q_different['preview_posterior'])
        full = update(bank, exact, q, cuts, cfg)
        controls = {
            'scale_only': update(bank, exact, q, cuts, dict(cfg, mixing=0.)),
            'target_only': update(bank, exact, q, cuts, dict(cfg, weight=0.)),
            'swap_preview_evidence': update(bank, exact, q_swap, cuts, cfg),
        }
        for control in controls.values():
            np.testing.assert_allclose(full.mean(1), control.mean(1), rtol=3e-6, atol=1e-8)
        paired = PairedConcepts(labels, np.array([r['concept'] for r in meta]))
        measurements = {
            name: metrics(z, head, paired)
            for name, z in dict(full=full, raw_exact=exact, raw_different=different,
                                raw_none=none, **controls).items()
        }
        comparisons = [('full', ref) for ref in ('raw_exact', *controls)]
        comparisons.extend(('raw_exact', ref) for ref in ('raw_different', 'raw_none'))
        for condition, reference in comparisons:
            changes = contrast(measurements[condition], measurements[reference])
            table.extend(dict(dataset=dataset, condition=condition, reference=reference,
                              metric=metric, **estimate(delta)) for metric, delta in changes.items())
    path = DOC / 'chain/object_corroboration_controls.csv'
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
