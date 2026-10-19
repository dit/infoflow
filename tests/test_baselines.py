"""
Tests for the baseline methods and their use in the validation harness.
"""

import numpy as np
import pytest

from infoflow.baselines import run_baseline, var_granger
from infoflow.benchmarks import network_validation, random_network, simulate_network


def test_var_granger_recovers_var_links():
    graph = random_network(6, 2.0, max_lag=3, prng=2)
    data = simulate_network(graph, 3000, "var", prng=0)
    base = var_granger(data, max_lag=3)
    truth = set(graph.edges())
    assert truth <= base.found
    assert len(base.found - truth) <= 1
    for s, t in truth:
        assert base.lags[s, t] == graph.edges[s, t]["lag"]
    assert base.score.shape == (6, 6) and np.all(np.diag(base.score) == 0)


def test_var_granger_empty_network_is_quiet():
    data = np.random.default_rng(0).normal(size=(2000, 5))
    assert len(var_granger(data, max_lag=2).found) <= 1


def test_unknown_baseline():
    with pytest.raises(ValueError):
        run_baseline("nope", np.zeros((10, 2)))


def test_tigramite_baseline_finds_driver():
    pytest.importorskip("tigramite.pcmci")
    graph = random_network(4, 1.5, max_lag=2, prng=3)
    data = simulate_network(graph, 1500, "var", prng=1)
    for method in ("pcmciplus", "lpcmci"):
        base = run_baseline(method, data, max_lag=2, alpha=0.01)
        assert set(graph.edges()) & base.found
        assert base.score.shape == (4, 4)


@pytest.mark.slow
def test_network_validation_compares_methods():
    ds = network_validation(
        "var", n_nodes=(5,), n_samples=(1500,), n_reps=1, max_lag=2, methods=("infoflow", "var"), prng=0
    )
    assert list(ds["algorithm"].values) == ["infoflow", "var"]
    var = ds.sel(algorithm="var")
    assert float(var["auroc"].sel(level="skeleton").squeeze()) > 0.9
    assert np.isnan(float(var["precision"].sel(level="intrinsic").squeeze()))
    assert np.isfinite(float(ds.sel(algorithm="infoflow", level="skeleton")["auroc"].squeeze()))


def test_network_validation_lorenz96():
    ds = network_validation("lorenz96", n_nodes=(6,), n_samples=(1000,), n_reps=1, max_lag=1, methods=("var",), prng=0)
    assert float(ds["recall"].sel(algorithm="var", level="skeleton").squeeze()) > 0.3
