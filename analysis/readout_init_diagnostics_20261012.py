"""Secondary descriptive comparisons; does not alter frozen readouts/endpoints."""
import numpy as np

from readout_init_20261012 import OUT, CONDITIONS, load, write, paired_stats


def main():
    result = load(OUT / 'test_predictions.pckl')
    fitted = load(OUT / 'readouts.pckl')
    keys, labels, synsets = (result[k] for k in ('keys', 'labels', 'synsets'))
    endpoints = result['endpoints']
    names = ('t0', 't3', 't9', 'mean_t0_t3_raw', 'mean_t0_t3_calibrated',
             'total_budget_9', 'total_budget_15', 'target_budget_4')
    cross = []
    initialization = []
    calibration = []
    for name in names:
        for condition in CONDITIONS:
            for a, b in (('pooled', 'legacy'), ('pooled', 'matched_t9')):
                cross.append(dict(endpoint=name, condition=condition, contrast=a+'-minus-'+b,
                    **paired_stats(endpoints[(a, condition, name)], endpoints[(b, condition, name)], keys, synsets)))
        for head in fitted['heads']:
            initialization.append(dict(readout=head, endpoint=name, contrast='none_feedforward-minus-none_random',
                **paired_stats(endpoints[(head, 'none_feedforward', name)],
                               endpoints[(head, 'none_random', name)], keys, synsets)))
    for head in fitted['heads']:
        for condition in CONDITIONS:
            calibration.append(dict(readout=head, condition=condition, contrast='calibrated-minus-raw-early-average',
                **paired_stats(endpoints[(head, condition, 'mean_t0_t3_calibrated')],
                               endpoints[(head, condition, 'mean_t0_t3_raw')], keys, synsets)))
    write('secondary_cross_readout.csv', cross)
    write('secondary_initialization.csv', initialization)
    write('secondary_calibration.csv', calibration)
    margins = []
    for (head, condition), scores in result['scores'].items():
        specs = {f't{t}': scores[:, t] for t in (0, 1, 3, 9)}
        specs['mean_t0_t3_raw'] = scores[:, :4].mean(1)
        specs['mean_t0_t3_calibrated'] = (scores[:, :4] / fitted['temperatures'][head][None, :4, None]).mean(1)
        for endpoint, s in specs.items():
            distractors = s.copy()
            distractors[np.arange(len(labels)), labels] = -np.inf
            margin = s[np.arange(len(labels)), labels] - distractors.max(1)
            sd = np.maximum(s.std(1), 1e-6)
            margins.append(dict(readout=head, condition=condition, endpoint=endpoint,
                mean_margin=float(margin.mean()), median_margin=float(np.median(margin)),
                mean_sd_normalized_margin=float((margin/sd).mean()), mean_score_sd=float(sd.mean())))
    write('secondary_margins.csv', margins)
    print('Secondary paired comparisons and margins complete; no new primary tests or selection.')


if __name__ == '__main__':
    main()
