"""
Tests for discretizers, DiscreteData, and alignment of realizations.
"""

from math import factorial

import numpy as np
import pytest
from dit.inference import Trials, ordinal_patterns
from hypothesis import given, settings
from hypothesis import strategies as st

from infoflow.data import DiscreteData, as_trials, realizations
from infoflow.preprocess import EqualFrequency, EqualWidth, Identity, Ordinal, Threshold, discretize, reencode


def test_as_trials_layouts():
    x = np.zeros((10, 3))
    assert [t.shape for t in as_trials(x)] == [(10, 3)]
    assert [t.shape for t in as_trials(np.zeros(10))] == [(10, 1)]
    assert [t.shape for t in as_trials(Trials([np.zeros((5, 2)), np.zeros((7, 2))]))] == [(5, 2), (7, 2)]
    assert [t.shape for t in as_trials([np.zeros(8), np.ones(8)])] == [(8, 2)]


def test_equal_frequency_is_balanced_and_rank_based():
    x = np.random.default_rng(0).exponential(size=(4000, 1))
    d = discretize(x, EqualFrequency(4))
    counts = np.bincount(d.past[0][0], minlength=4)
    assert np.all(np.abs(counts - 1000) < 20)
    transformed = discretize(np.log(x), EqualFrequency(4))
    assert np.array_equal(d.past[0], transformed.past[0])


def test_ordinal_encoding_matches_dit():
    x = np.random.default_rng(1).normal(size=(300, 1))
    d = discretize(x, Ordinal(3, 2))
    assert d.past_offset[0] == 4 and d.present_offset[0] == 0
    assert d.past_alphabet[0] == factorial(3) and d.present_alphabet[0] == 4
    assert d.lag_step[0] == 5
    assert np.array_equal(d.past[0][0, 4:], ordinal_patterns(x[:, 0], 3, 2))
    # The present is the quartile of x_t alone.
    quartile = np.searchsorted(np.quantile(x[:, 0], [0.25, 0.5, 0.75]), x[:, 0], side="right")
    assert np.array_equal(d.present[0][0], quartile)
    assert np.all(d.past[0][0, :4] == -1)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10**6), lags=st.lists(st.integers(1, 5), min_size=1, max_size=3))
def test_realizations_alignment(seed, lags):
    """
    Each realization's variable values equal past[p][t - lag] and its present equals present[target][t].
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(60, 2))
    d = discretize(x, [Ordinal(3, 1), EqualFrequency(3)])
    variables = [(i % 2, lag) for i, lag in enumerate(lags)]
    r = realizations(d, 1, variables)
    for row in range(len(r)):
        t = r.time[row]
        assert r.present[row] == d.present[0][1, t]
        for k, (p, lag) in enumerate(variables):
            assert r.values[row, k] == d.past[0][p, t - lag] >= 0


def test_realizations_never_cross_trials():
    trials = Trials([np.arange(10.0)[:, None], np.arange(10.0, 25.0)[:, None]])
    d = discretize(trials, EqualWidth(5))
    r = realizations(d, 0, [(0, 3)])
    assert len(r) == (10 - 3) + (15 - 3)
    assert set(r.trial.tolist()) == {0, 1}
    assert r.time.min() == 3


def test_threshold_and_identity():
    x = np.random.default_rng(2).normal(size=(1000, 1))
    d = discretize(x, Threshold(quantile=0.8))
    assert abs(d.past[0].mean() - 0.2) < 0.01
    discrete = np.random.default_rng(3).integers(0, 3, size=(100, 2))
    a = discretize(discrete, Identity())
    b = DiscreteData.from_discrete(discrete)
    assert np.array_equal(a.past[0], b.past[0])


def test_reencode_applies_fitted_discretizers():
    """
    Surrogates built on the raw series are encoded with the original fit.
    """
    x = np.random.default_rng(4).normal(size=(500, 2))
    d = discretize(x, [EqualFrequency(4), Ordinal(3, 1)])
    shifted = np.roll(x, 50, axis=0)
    s = reencode(d, [shifted])
    assert np.array_equal(s.past[0][0], d.discretizers[0].encode(shifted[:, 0]).past)
    assert np.array_equal(s.present[0][1], d.discretizers[1].encode(shifted[:, 1]).present)
    # The quantile edges are those of the original data, not refitted.
    assert np.array_equal(s.discretizers[0].edges, d.discretizers[0].edges)


def test_discretize_validates_count():
    with pytest.raises(ValueError):
        discretize(np.zeros((10, 2)), [EqualFrequency(2)])
