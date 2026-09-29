"""
Phase 3 milestone: raw continuous data to multiplex networks.
"""

import numpy as np
import pytest

from infoflow import datasets
from infoflow.network import infer_multiplex
from infoflow.selection import SkeletonSettings

from ._generators import exact_flows, mixture, sample_pairs

FAST = SkeletonSettings(
    n_perm_max_stat=100,
    n_perm_min_stat=100,
    n_perm_omnibus=200,
    n_perm_max_seq=200,
    n_perm_pairs=50,
    n_perm_tdmi=100,
)


def _sig(net, layer):
    return net.dataset["significant"].sel(layer=layer).values


def _coupled_ar(n, lag, seed, coupling=0.8):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3))
    for t in range(lag, n):
        x[t, 0] = 0.5 * x[t - 1, 0] + rng.normal()
        x[t, 1] = 0.3 * x[t - 1, 1] + coupling * np.tanh(x[t - lag, 0]) + rng.normal(scale=0.5)
        x[t, 2] = 0.6 * x[t - 1, 2] + rng.normal()
    return x


def test_raw_pipeline_recovers_delay():
    net = infer_multiplex(_coupled_ar(3000, 3, seed=0), skeleton_settings=FAST, n_boot=50, n_null=50, prng=0)
    sig = _sig(net, "intrinsic") | _sig(net, "synergistic")
    assert sig[0, 1]
    assert int(net.dataset["delay"].values[0, 1]) == 3
    assert not sig[1, 0] and not sig[0, 2] and not sig[2, 1]
    assert net.report is not None and len(net.report.nodes) == 3


def test_network_is_invariant_to_monotone_transforms():
    x = _coupled_ar(1500, 2, seed=1)
    a = infer_multiplex(x, skeleton_settings=FAST, n_boot=30, n_null=30, prng=0)
    b = infer_multiplex(np.exp(x / 4) + 1, skeleton_settings=FAST, n_boot=30, n_null=30, prng=0)
    assert np.array_equal(a.dataset["significant"].values, b.dataset["significant"].values)
    assert np.allclose(a.dataset["weight"].values, b.dataset["weight"].values, equal_nan=True)


def test_uncoupled_false_positive_rate():
    """
    Independent autocorrelated processes: almost no significant flow in any layer.
    """
    false = 0
    runs = 8
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        x = np.zeros((1500, 2))
        for t in range(1, 1500):
            x[t] = 0.8 * x[t - 1] + rng.normal(size=2)
        net = infer_multiplex(x, skeleton_settings=FAST, n_boot=40, n_null=40, prng=seed)
        false += int(_sig(net, "intrinsic").sum() + _sig(net, "synergistic").sum())
    assert false <= 1


def test_discrete_sofic_generator_matches_exact_and_delay():
    generator = mixture(a_copy=0.6, a_xor=0.3)
    exact = exact_flows(generator)
    data = sample_pairs(generator, 15000, seed=3)
    net = infer_multiplex(data, max_lag=2, skeleton_settings=FAST, n_boot=50, n_null=50, prng=0)
    assert int(net.dataset["delay"].values[0, 1]) == 1
    w = net.dataset["weight"]
    assert float(w.sel(layer="intrinsic").values[0, 1]) == pytest.approx(exact["intrinsic"], abs=0.03)
    assert float(w.sel(layer="synergistic").values[0, 1]) == pytest.approx(exact["synergistic"], abs=0.03)


def test_coupled_logistic_direction():
    net = infer_multiplex(datasets.coupled_logistic(3000, seed=0), skeleton_settings=FAST, prng=0)
    te = net.dataset["weight"].sel(layer="te").values
    assert np.nan_to_num(te[0, 1]) > np.nan_to_num(te[1, 0]) + 0.3


@pytest.mark.slow
def test_raw_mute_network():
    net = infer_multiplex(datasets.mute_network(3000, seed=0), skeleton_settings=FAST, n_boot=50, n_null=50, prng=0)
    sig = _sig(net, "intrinsic")
    recovered = [e for e in datasets.MUTE_EDGES if sig[e]]
    assert len(recovered) >= 4
    assert int(net.dataset["delay"].values[0, 2]) == 3
