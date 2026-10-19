"""
Tests for the mixed-type nearest-neighbour CMI and the predictive transfer entropy.
"""

import numpy as np
import pytest

from infoflow.predictive import predictive_transfer_entropy


def _mixed_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 3, n)
    x = rng.integers(0, 2, n)
    y = np.roll(x ^ (z == 1) ^ (rng.random(n) < 0.2), 1)
    c = rng.normal(size=n)
    w = np.roll(c, 1) + 0.5 * rng.normal(size=n)
    return np.stack([x, y, z, c, w], axis=1).astype(float)


def _columns(data, target, variables):
    from infoflow.network import _as_discrete
    from infoflow.selection import _MixedColumns

    d, _, _ = _as_discrete(data, None, "auto", 3, 0)
    return _MixedColumns(d, target, variables, 1, prng=0)


def test_mixed_matches_plugin_on_categorical_data():
    data = _mixed_data()
    m = _columns(data, 1, [(0, 1), (2, 1)])
    assert m.categorical == {0, 1, 2}
    x, _ = m.joint([(0, 1)])
    z, _ = m.joint([(2, 1)])
    y_, x_, z_ = data[1:, 1], data[:-1, 0], data[:-1, 2]

    def H(*cols):
        _, n = np.unique(np.stack(cols), axis=1, return_counts=True)
        p = n / n.sum()
        return -np.sum(p * np.log2(p))

    plugin = H(y_, z_) + H(x_, z_) - H(y_, x_, z_) - H(z_)
    assert m.cmi(x, 1, z, 1) == pytest.approx(plugin, abs=0.02)


def test_mixed_is_unbiased_for_independent_continuous_and_exact_for_gaussian():
    data = _mixed_data()
    m = _columns(data, 1, [(0, 1), (2, 1), (3, 1)])
    c, _ = m.joint([(3, 1)])
    zx = np.column_stack([m.joint([(2, 1)])[0], m.joint([(0, 1)])[0]])
    assert abs(m.cmi(c, 1, zx, 2)) < 0.03
    g = _columns(data, 4, [(3, 1), (0, 1)])
    c, _ = g.joint([(3, 1)])
    x, _ = g.joint([(0, 1)])
    assert g.cmi(c, 1, x, 1) == pytest.approx(0.5 * np.log2(5), abs=0.08)
    assert abs(g.cmi(x, 1, c, 1)) < 0.03


@pytest.mark.slow
def test_mixed_selection_finds_categorical_and_continuous_parents():
    from infoflow.network import infer_multiplex
    from infoflow.selection import SkeletonSettings

    data = _mixed_data(2000)
    net = infer_multiplex(
        data,
        skeleton_settings=SkeletonSettings(estimator="mixed", n_perm_max_stat=100, n_perm_min_stat=100),
        targets=[1, 4],
        interpret=False,
        node_layer=False,
        prng=0,
    )
    assert set(net.skeleton[1].parents()) >= {0, 2}
    assert 3 not in net.skeleton[1].parents()
    assert net.skeleton[4].parents() == [3]


def test_predictive_te_gaussian_matches_analytic():
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.normal(size=n)
    y = np.zeros(n)
    y[1:] = 0.8 * x[:-1] + 0.5 * rng.normal(size=n - 1)
    res = predictive_transfer_entropy(x, y)
    assert res.value == pytest.approx(0.5 * np.log2(1 + 0.64 / 0.25), abs=0.05)
    assert res.pvalue < 0.01
    null = predictive_transfer_entropy(rng.normal(size=n), y)
    assert abs(null.value) < 0.01 and null.pvalue > 0.01


def test_predictive_te_categorical_and_conditions():
    from infoflow.datasets import common_driver

    data = common_driver(6000, seed=0)  # z drives x (lag 1) and y (lag 2)
    x, y, z = data.T
    bivariate = predictive_transfer_entropy(x, y, model="categorical")
    conditioned = predictive_transfer_entropy(x, y, conditions=[z], history_length=2, model="categorical")
    assert bivariate.value > 0.3
    assert abs(conditioned.value) < 0.02


def test_predictive_te_accepts_callable_model():
    from infoflow.predictive import _gaussian

    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y = np.roll(x, 1) + rng.normal(size=2000)
    a = predictive_transfer_entropy(x, y, model="gaussian").value
    b = predictive_transfer_entropy(x, y, model=_gaussian).value
    assert a == pytest.approx(b)
    with pytest.raises(ValueError):
        predictive_transfer_entropy(x, y, model="nope")
