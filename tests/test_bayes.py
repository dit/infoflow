"""
Tests for the hierarchical Bayesian layer posterior.
"""

import importlib.util

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from infoflow.bayes import (
    ContextPrior,
    _decompose,
    fit_context_prior,
    layer_posterior,
    log_bayes_factor,
    posterior_tables,
)
from infoflow.measures import intrinsic_flow


@st.composite
def _tables_and_priors(draw):
    Ky, Kx, Kw = (draw(st.integers(1, 3)) for _ in range(3))
    seed = draw(st.integers(0, 2**31))
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 6, size=(Ky, Kx, Kw)).astype(float)
    counts[0, 0, :] += 1  # every context observed
    G = draw(st.integers(1, 3))
    m = rng.dirichlet(np.ones(Ky * Kx), size=(3, G))
    alpha = rng.lognormal(0.0, 2.0, size=(3, G))
    weights = rng.dirichlet(np.ones(G), size=3)
    return counts, ContextPrior(alpha=alpha, m=m, weights=weights), seed


@settings(max_examples=30, deadline=None)
@given(_tables_and_priors())
def test_posterior_draws_are_joints_with_exact_layer_identities(case):
    counts, prior, seed = case
    _, Q = intrinsic_flow(counts, restarts=0, prng=seed)
    for p in posterior_tables(counts, prior, n_draws=3, prng=seed):
        assert p.shape == counts.shape
        assert np.all(p >= 0)
        assert p.sum() == pytest.approx(1.0)
        te, tdmi, iif, syn, shared = _decompose(p, Q, np.random.default_rng(seed))
        assert te == pytest.approx(iif + syn, abs=1e-12)
        assert tdmi == pytest.approx(iif + shared, abs=1e-12)
        assert min(iif, syn, shared) >= 0


def test_large_concentration_pools_contexts_to_the_shared_law():
    counts = np.zeros((2, 2, 3))
    counts[0, 0, 0] = counts[1, 1, 1] = counts[0, 1, 2] = 5
    m = np.array([0.1, 0.2, 0.3, 0.4])
    prior = ContextPrior(alpha=np.array([[1e7]]), m=m[None, None, :], weights=np.ones((1, 1)))
    for p in posterior_tables(counts, prior, n_draws=5, prng=0):
        conditional = p / p.sum(axis=(0, 1), keepdims=True)
        assert np.allclose(conditional.reshape(4, 3).T, m, atol=1e-2)


def test_mixture_assigns_each_context_to_the_component_matching_its_counts():
    laws = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
    counts = np.zeros((2, 2, 4))
    for w in range(4):
        counts[:, :, w] = (200 * laws[w % 2]).reshape(2, 2)
    prior = ContextPrior(alpha=np.full((1, 2), 1e7), m=laws[None], weights=np.full((1, 2), 0.5))
    for p in posterior_tables(counts, prior, n_draws=5, prng=0):
        conditional = (p / p.sum(axis=(0, 1), keepdims=True)).reshape(4, 4).T
        assert np.allclose(conditional, laws[np.arange(4) % 2], atol=1e-2)


def test_bayes_factor_favours_dependence_only_when_present():
    rng = np.random.default_rng(0)
    n = 2000
    x, w = rng.integers(0, 2, n), rng.integers(0, 2, n)
    assert log_bayes_factor(x ^ w, x, w) > 10
    assert log_bayes_factor(rng.integers(0, 2, n), x, w) < 0
    assert log_bayes_factor(x ^ w, x) < 0


requires_pymc = pytest.mark.skipif(importlib.util.find_spec("pymc") is None, reason="needs pymc")


def _fit(counts):
    return fit_context_prior(counts, draws=200, tune=200, chains=1, prng=0)


def test_unknown_layer_method_is_rejected():
    from infoflow.layers import layer_statistics

    x = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="layer_method"):
        layer_statistics(x, x, None, n_boot=0, n_null=0, layer_method="mcmc")


def _xor(n, extra_bits, seed):
    rng = np.random.default_rng(seed)
    x, w0 = rng.integers(0, 2, n), rng.integers(0, 2, n)
    y = x ^ w0 ^ (rng.random(n) < 0.05)
    return y, x, np.column_stack([w0, rng.integers(0, 2, (n, extra_bits))])


@requires_pymc
def test_bayes_layer_statistics_use_posterior_estimates_and_keep_the_nulls():
    from infoflow.layers import layer_statistics

    stats = layer_statistics(*_xor(2000, 0, 0), n_boot=50, n_null=39, layer_method="bayes", prng=0)
    assert stats.method == "bayes"
    assert stats.n_boot == 0
    flow = stats.flow
    assert flow.te == pytest.approx(flow.intrinsic + flow.synergistic)
    assert flow.tdmi == pytest.approx(flow.intrinsic + flow.shared)
    low, high = stats.ci["synergistic"]
    assert low <= flow.synergistic <= high
    assert {layer for layer, p in stats.pvalue.items() if p <= 0.05} == {"synergistic"}


@requires_pymc
@pytest.mark.parametrize(("extra_bits", "method"), [(0, "crossfit"), (6, "bayes")])
def test_auto_layer_method_switches_on_samples_per_context(extra_bits, method):
    from infoflow.layers import layer_statistics

    stats = layer_statistics(*_xor(1000, extra_bits, 1), n_boot=0, n_null=0, layer_method="auto", prng=0)
    assert stats.method == method


@requires_pymc
def test_infer_multiplex_reports_the_layer_method_per_edge():
    from infoflow import datasets
    from infoflow.network import infer_multiplex

    data = datasets.xor_synergy(1000, noise=0.1, seed=0)
    ds = infer_multiplex(data, max_lag=2, n_boot=0, n_null=39, layer_method="bayes", prng=0).dataset
    estimated = ds["kind"].values != ""
    assert estimated.any()
    assert set(ds["layer_method"].values[estimated]) == {"bayes"}
    assert set(ds["layer_method"].values[~estimated]) == {""}
    assert ds["significant"].sel(layer="synergistic").values[0, 2]
    assert ds["significant"].sel(layer="synergistic").values[1, 2]


@requires_pymc
def test_alpha_is_large_for_identical_context_laws_and_small_for_distinct_ones():
    law = np.array([0.4, 0.1, 0.2, 0.3])
    identical = np.repeat((1000 * law).reshape(2, 2, 1), 6, axis=2)
    distinct = np.zeros((2, 2, 4))
    for w in range(4):
        distinct.reshape(4, 4)[w, w] = 250
    assert np.median(_fit(identical).alpha) > 50
    assert np.median(_fit(distinct).alpha) < 1


@requires_pymc
def test_posterior_layers_match_plugin_at_large_n():
    rng = np.random.default_rng(3)
    n = 20_000
    x, w = rng.integers(0, 2, n), rng.integers(0, 2, n)
    y = np.where(rng.random(n) < 0.8, x ^ w, rng.integers(0, 2, n))
    post = layer_posterior(y, x, w, n_draws=40, draws=200, tune=200, chains=1, prng=0)
    mean = post.mean
    assert mean["te"] == pytest.approx(post.raw["te"], abs=0.01)
    assert mean["tdmi"] == pytest.approx(post.raw["tdmi"], abs=0.01)
    s = post.samples
    np.testing.assert_allclose(s.te, s.intrinsic + s.synergistic, atol=1e-12)
    np.testing.assert_allclose(s.tdmi, s.intrinsic + s.shared, atol=1e-12)
    low, high = post.interval()["synergistic"]
    assert low <= mean["synergistic"] <= high
    assert post.as_flow().synergistic == pytest.approx(mean["synergistic"])
