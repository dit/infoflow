"""
Tests for delay heuristics, screening, scaling regions, scoring, and the preprocessing pipeline.
"""

import numpy as np
import pytest

from infoflow import datasets
from infoflow.preprocess import (
    EqualFrequency,
    Ordinal,
    block_entropy_scaling,
    curvature_delay,
    delay_candidates,
    preprocess,
    scaling_region,
    screen,
    split_missing,
)


def test_delay_heuristics_on_sine():
    t = np.arange(4000)
    x = np.sin(2 * np.pi * t / 40)
    assert curvature_delay(x, 30)[0] in (9, 10, 11)
    # A noise-free periodic sequence aliases with the bins; a little noise avoids it.
    noisy = x + 0.1 * np.random.default_rng(0).normal(size=len(x))
    candidates, details = delay_candidates(noisy, 30)
    assert details["auto_mi"]["delay"] in (9, 10, 11)
    assert 1 in candidates and details["auto_mi"]["delay"] in candidates


def test_delay_candidates_invariant_to_monotone_transform():
    x = datasets.mute_network(2000, seed=0)[:, 0]
    a, _ = delay_candidates(x, 15)
    b, _ = delay_candidates(np.exp(x / 3), 15)
    assert a == b


def test_screening():
    rng = np.random.default_rng(0)
    stationary = rng.normal(size=4000)
    drifting = np.sin(np.arange(4000) / 3) * np.linspace(0, 3, 4000) ** 2 + rng.normal(scale=0.05, size=4000)
    quantized = np.round(rng.normal(size=4000))
    data = np.stack([stationary, drifting, quantized], axis=1)
    data[10, 0] = np.nan
    results = screen([data])
    assert not results[0].nonstationary and results[0].n_missing == 1
    assert results[2].tie_fraction > 0.2
    assert [len(t) for t in split_missing([data])] == [3989]


def test_scaling_region_and_block_entropy():
    x = np.arange(20.0)
    y = np.where(x < 5, 3 * x, 15 + 0.5 * (x - 5))
    assert scaling_region(x, y).slope == pytest.approx(0.5, abs=0.05)
    rng = np.random.default_rng(1)
    s, prev = [], 1
    for _ in range(50000):
        prev = 0 if prev == 1 else int(rng.integers(2))
        s.append(prev)
    region, lengths, _ = block_entropy_scaling(s)
    assert region.slope == pytest.approx(2 / 3, abs=0.02)
    assert region.intercept == pytest.approx(0.2516, abs=0.03)


def test_preprocess_recovers_markov_order():
    """
    A binary series depending on lag 2 gets a lag budget of 2.
    """
    rng = np.random.default_rng(2)
    n = 6000
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = x[t - 2] if rng.random() < 0.9 else rng.normal()
    x = x + 0.01 * rng.normal(size=n)
    result = preprocess(x[:, None], prng=0)
    node = result.report.nodes[0]
    assert result.embeddings[0].max_lag >= 2
    assert node.chosen["score"] > 0.2


def test_preprocess_monotone_invariance():
    x = datasets.mute_network(2000, seed=1)[:, :3]
    a = preprocess(x, prng=0)
    b = preprocess(np.sign(x) * np.abs(x) ** 3 + 5, prng=0)
    for na, nb in zip(a.report.nodes, b.report.nodes, strict=True):
        assert na.chosen == nb.chosen
        assert na.lag_budget == nb.lag_budget
    for pa, pb in zip(a.data.past, b.data.past, strict=True):
        assert np.array_equal(pa, pb)


def test_preprocess_overrides_and_report():
    x = datasets.mute_network(1500, seed=2)[:, :2]
    fixed = preprocess(x, discretizer=Ordinal(3, 1), prng=0)
    assert all(isinstance(d, Ordinal) for d in fixed.data.discretizers)
    assert all(n.rule == "user" for n in fixed.report.nodes)
    per = preprocess(x, discretizer=[EqualFrequency(3), Ordinal(2, 1)], prng=0)
    assert isinstance(per.data.discretizers[0], EqualFrequency)
    assert "lag budget" in per.report.summary()
    capped = preprocess(x, max_lag=1, prng=0)
    assert all(e.max_lag <= 1 for e in capped.embeddings)
