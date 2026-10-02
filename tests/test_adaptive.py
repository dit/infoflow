"""
Tests for MDL histograms, adaptive conditioning, and the strata / debiased selection options.
"""

import dataclasses
from itertools import product
from math import comb, factorial

import numpy as np
import pytest

from infoflow import datasets
from infoflow.adaptive import joint_histogram, log_regret, mdl_histogram
from infoflow.benchmarks import conditioning_scaling
from infoflow.data import DiscreteData
from infoflow.network import estimate_edge
from infoflow.preprocess import preprocess
from infoflow.selection import SkeletonSettings, _ranks, select_parents

FAST = SkeletonSettings(
    n_perm_max_stat=50, n_perm_min_stat=50, n_perm_omnibus=50, n_perm_max_seq=50, n_perm_pairs=50, n_perm_tdmi=50
)


def test_regret_matches_brute_force():
    n, K = 8, 3
    total = 0.0
    for counts in product(range(n + 1), repeat=K):
        if sum(counts) == n:
            ways = factorial(n) / np.prod([factorial(c) for c in counts])
            total += ways * np.prod([(c / n) ** c for c in counts if c])
    assert log_regret(n, K) == pytest.approx(np.log(total))
    assert log_regret(n, 1) == 0.0
    assert log_regret(n, 2) == pytest.approx(
        np.log(sum(comb(n, h) * (h / n) ** h * ((n - h) / n) ** (n - h) for h in range(n + 1)))
    )


def test_one_dimensional_histograms():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-3, 0.5, 2000), rng.normal(3, 0.5, 2000)])
    cuts = mdl_histogram(x)
    assert len(cuts) >= 2 and np.any(np.abs(cuts) < 2)
    assert len(mdl_histogram(_ranks(rng.normal(size=3000)))) == 0  # uniform margins need no cuts


def test_joint_histogram_resolves_dependence_and_ignores_noise():
    rng = np.random.default_rng(1)
    n = 6000
    z = rng.normal(size=n)
    y = z**2 + 0.3 * rng.normal(size=n)
    noise = rng.normal(size=(n, 2))
    columns = [_ranks(y), _ranks(z), *[_ranks(c) for c in noise.T]]
    cuts = joint_histogram(columns, fixed={0: 6})
    assert len(cuts[0]) == 5  # the fixed column keeps its six bins
    assert len(cuts[1]) >= 3 and np.any(cuts[1] < 0.4) and np.any(cuts[1] > 0.6)  # both tails of z
    assert all(len(c) == 0 for c in cuts[2:])
    warm = joint_histogram(columns, fixed={0: 6}, initial=[None, cuts[1], None, None])
    assert [len(c) for c in warm] == [len(c) for c in cuts]


def test_adaptive_estimator_is_flat_in_dimension():
    null = conditioning_scaling(
        "gaussian", ("adaptive",), n_samples=(1500,), dims=(0, 4, 8), n_reps=2, coupled=False, prng=0
    )
    assert np.all(np.abs(null["mean"].values) < 0.01)
    ds = conditioning_scaling("gaussian", ("adaptive",), n_samples=(1500,), dims=(0, 8), n_reps=2, prng=0)
    near, far = ds["mean"].sel(estimator="adaptive", n=1500).values
    assert far == pytest.approx(near, abs=0.05) and near > 0.15


def test_strata_debiased_selection():
    data = DiscreteData.from_discrete([datasets.chain(3000, seed=0)])
    settings = dataclasses.replace(FAST, null="strata", statistic="debiased")
    assert select_parents(data, 2, None, None, settings, 0).parents() == [1]
    rng = np.random.default_rng(2)
    noise = DiscreteData.from_discrete([rng.integers(0, 3, size=(3000, 3))])
    assert all(not select_parents(noise, t, None, None, settings, 0).sources for t in range(3))
    with pytest.raises(ValueError, match="null"):
        SkeletonSettings(null="nearby").check()
    with pytest.raises(ValueError, match="statistic"):
        SkeletonSettings(statistic="zscore").check()


def test_adaptive_selection_and_layer_symbols():
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.4 * y[t - 1] + 0.8 * x[t - 1] + 0.5 * rng.normal()
    result = preprocess(np.stack([x, y], axis=1), max_lag=2, prng=0)
    settings = dataclasses.replace(
        FAST, estimator="adaptive", null="strata", statistic="debiased", synergy_search=False, tdmi_screen=False
    )
    sk = select_parents(result.data, 1, result.embeddings, None, settings, 0)
    assert sk.parents() == [0]
    stats, delay = estimate_edge(
        result.data, 1, sk.source_variables(0), list(sk.target_past), n_boot=20, n_null=20, prng=0, adaptive=(6, 4)
    )
    assert delay == 1 and stats.flow.intrinsic > 0.1
    with pytest.raises(ValueError, match="raw series"):
        select_parents(
            DiscreteData.from_discrete([np.stack([x > 0, y > 0], axis=1).astype(int)]), 1, None, None, settings, 0
        )
