import numpy as np
import pytest
import torch

from predify_experiments.conv_preview_memory_20261106 import (
    DEFAULTS, OFFSETS, PreviewMemory, PreviewOperator, coactivation, mutual_sparse)


def test_accumulation_sparse_reciprocity_and_freeze():
    torch.manual_seed(7)
    q = dict(DEFAULTS, inter_k=3)
    m = PreviewMemory(6, config=q)
    a, b = torch.rand(1, 6, 4, 5), torch.rand(1, 6, 4, 5)
    a0, al, _ = coactivation(a); b0, bl, _ = coactivation(b)
    m.write(a)
    assert torch.allclose(m.intra, .5*a0)
    m.write(b)
    assert torch.allclose(m.intra, .25*a0+.5*b0)
    assert torch.allclose(m.trace_inter, .25*al+.5*bl)
    assert torch.equal(m.inter, mutual_sparse(.25*al+.5*bl, 3))
    assert m.intra.diagonal().count_nonzero() == 0
    assert (m.intra>0).sum() == 6*5  # all within-column candidate connections
    assert (m.inter>0).sum((1, 2, 3)).max() <= 3
    assert torch.allclose(m.intra, m.intra.T)
    for dy, dx in OFFSETS:
        assert torch.allclose(m.inter[:, :, dy+1, dx+1], m.inter[:, :, 1-dy, 1-dx].T, atol=1e-7)
    before = m.fingerprint(); m.freeze()
    with pytest.raises(RuntimeError): m.write(a)
    for family in ('gate', 'multiply'):
        op = PreviewOperator(m, 1., family, 'rep4', q)
        for _ in range(15):
            y = op(a)
            assert torch.isfinite(y).all() and (y>=0).all()
        assert m.fingerprint() == before
    m.reset()
    assert m.writes == 0 and m.intra.count_nonzero() == 0 and m.trace_inter.count_nonzero() == 0


def test_offset_outer_products_against_explicit_positions():
    a = torch.arange(1., 1+3*3*4).reshape(1, 3, 3, 4)
    z = a/a.square().mean().sqrt()
    expected = torch.zeros(3, 3, 3, 3)
    for dy, dx in OFFSETS:
        pairs = [torch.outer(z[0, :, y, x], z[0, :, y+dy, x+dx])
                 for y in range(3) for x in range(4) if 0<=y+dy<3 and 0<=x+dx<4]
        expected[:, :, dy+1, dx+1] = torch.stack(pairs).mean(0)*np.exp(-.5*(dy*dy+dx*dx))
    expected /= expected.sum((1, 2, 3)).max()
    assert torch.allclose(coactivation(a)[1], expected, atol=1e-7)


def test_zero_preview_and_trial_reset_remove_prior_influence():
    m = PreviewMemory(6)
    x = torch.rand(1, 6, 4, 5); x[..., 0, :] = 0
    for family in ('gate', 'multiply'):
        op = PreviewOperator(m, 1., family, 'rep4')
        none = PreviewOperator(m, 1., family, 'none')
        m.reset(); y0 = op(x)
        assert torch.equal(y0, none(x))
        m.write(x); m.write(x); m.freeze()
        assert not torch.equal(op(x), y0)
        if family == 'multiply': assert op(x)[x==0].count_nonzero() == 0
        m.reset(); m.write(torch.zeros_like(x)); m.write(torch.zeros_like(x)); m.freeze()
        assert torch.equal(op(x), y0)

