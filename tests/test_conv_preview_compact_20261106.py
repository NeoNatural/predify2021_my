import numpy as np

from analysis.conv_preview_temporary_20261106 import compact_group, LAYERS
from analysis.report_conv_preview_temporary_20261106 import geometry, bootstrap


def test_compact_full_map_geometry_matches_direct_features():
    rng = np.random.default_rng(7)
    values = [{k: rng.normal(size=(15, 17)).astype(np.float32) for k in LAYERS} for _ in range(6)]
    values[0]['conv4'][0] = 0
    saved = compact_group(values)
    labels = np.array([0, 0, 0, 1, 1, 1])
    for layer in LAYERS:
        for t in (0, 4, 14):
            x = np.array([v[layer][t] for v in values], dtype=np.float64)
            z = x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            w = sum(np.square(z[labels==c]-z[labels==c].mean(0)).sum() for c in (0, 1))/6
            b = sum(sum(labels==c)*np.square(z[labels==c].mean(0)-z.mean(0)).sum() for c in (0, 1))/6
            got = geometry(saved[layer+'_gram'][t], labels)
            assert np.allclose([got['within'], got['between'], got['fisher']], [w, b, b/w], atol=1e-12)
    assert np.array_equal(saved['fc2'], np.stack([v['fc2'] for v in values]))


def test_concept_interval_does_not_treat_images_as_independent_replicates():
    # Both concept means are exactly two; image-level bootstrap would be wider.
    assert bootstrap(np.array([0., 4., 1., 3.]), np.array(['a', 'a', 'b', 'b'])) == [2., 2.]
