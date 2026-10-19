"""
Tests for the O-information rate and its gradients.
"""

import numpy as np
import pytest

from infoflow.datasets import common_driver, noisy_eca, xor_synergy
from infoflow.oinfo import compare_with_layers, o_information_rate, oir_gradients


def test_xor_is_synergy_dominated():
    assert o_information_rate(xor_synergy(5000, seed=0)) < -0.5


def test_common_driver_is_redundancy_dominated():
    data = common_driver(5000, noise=0.05, seed=0)
    # y copies z at lag 2, so a one-step history sees none of the shared drive.
    assert abs(o_information_rate(data, history=1)) < 0.02
    assert o_information_rate(data, history=2) > 0.3


def test_independent_processes_have_zero_oir():
    data = np.random.default_rng(0).integers(0, 2, size=(5000, 3))
    assert abs(o_information_rate(data)) < 0.01


def test_two_processes_have_zero_oir():
    assert o_information_rate(xor_synergy(500, seed=0), processes=[0, 2]) == 0.0


def test_gradients_identities():
    ds = oir_gradients(xor_synergy(4000, seed=1))
    oir = float(ds["oir"])
    # With three processes every two-process OIR vanishes, so each gradient is the full OIR.
    assert np.allclose(ds["gradient"].values, oir)
    link = ds["link"].values
    assert np.allclose(link, link.T) and np.allclose(np.diag(link), 0)
    assert np.allclose(link[np.triu_indices(3, 1)], oir)


@pytest.mark.slow
def test_eca_rule_90_oir_disagrees_with_directed_synergy():
    from infoflow.network import infer_multiplex

    data = noisy_eca(5000, n_cells=5, rule=90, noise=0.02, seed=0)
    net = infer_multiplex(data, max_lag=2, interpret=False, node_layer=False, prng=0)
    syn = net.dataset["significant"].sel(layer="synergistic").values
    # Each cell is the XOR of its neighbours: every neighbour edge is synergistic ...
    assert all(syn[(i - 1) % 5, i] and syn[(i + 1) % 5, i] for i in range(5))
    # ... but the nearly deterministic ring has a small joint entropy rate, so the OIR
    # and every link gradient are positive (redundancy-dominated), neighbours or not.
    ds = compare_with_layers(net, data)
    assert np.all(ds["link"].values[np.triu_indices(5, 1)] > 0)


def test_compare_with_synergistic_layer_on_xor():
    from infoflow.network import infer_multiplex

    data = xor_synergy(3000, noise=0.05, seed=0)
    net = infer_multiplex(data, max_lag=2, interpret=False, node_layer=False, prng=0)
    ds = compare_with_layers(net, data)
    assert ds["link"].values[0, 1] < 0
    assert ds["synergistic"].values[0, 2] > 0.3 and ds["synergistic"].values[1, 2] > 0.3
