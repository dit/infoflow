"""
The channel layer: a discrete memoryless channel, its mutual information, and its capacity per edge.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import infoflow
from infoflow.channels import edge_channel, infer_channels
from infoflow.data import DiscreteData
from infoflow.network import infer_multiplex
from infoflow.selection import SkeletonSettings

FAST = SkeletonSettings(
    n_perm_max_stat=100,
    n_perm_min_stat=100,
    n_perm_omnibus=200,
    n_perm_max_seq=200,
    n_perm_pairs=50,
    n_perm_tdmi=100,
)
QUICK = dict(skeleton_settings=FAST, n_boot=50, n_null=50, node_layer=False, interpret=False, prng=0)


def _h2(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def _bsc(n, delay, noise, p_one, seed):
    rng = np.random.default_rng(seed)
    x = (rng.random(n) < p_one).astype(np.int64)
    y = np.roll(x, delay) ^ (rng.random(n) < noise)
    return np.stack([x, y], axis=1)


def test_binary_symmetric_channel():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 50000)
    y = x ^ (rng.random(50000) < 0.1)
    channel, optimal, mi, capacity = edge_channel(x, y)
    assert channel.values == pytest.approx(np.array([[0.9, 0.1], [0.1, 0.9]]), abs=0.01)
    assert capacity == pytest.approx(1 - _h2(0.1), abs=0.01)
    assert mi == pytest.approx(capacity, abs=0.01)
    assert optimal.values == pytest.approx([0.5, 0.5], abs=0.02)


def test_z_channel_capacity_and_input():
    """
    The Z channel with crossover 1/2 has capacity log2(5/4) at P(X = 1) = 2/5.
    """
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, 100000)
    y = np.where(x == 1, rng.integers(0, 2, 100000), 0)
    _, optimal, mi, capacity = edge_channel(x, y)
    assert capacity == pytest.approx(np.log2(5 / 4), abs=0.005)
    assert float(optimal.sel(input=1)) == pytest.approx(0.4, abs=0.02)
    assert mi < capacity


@settings(max_examples=30, deadline=None)
@given(
    st.integers(2, 4),
    st.integers(2, 4),
    st.integers(20, 400),
    st.integers(0, 2**32 - 1),
)
def test_mi_is_at_most_capacity_is_at_most_log_alphabet(kx, ky, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, kx, n)
    y = (x + rng.integers(0, ky, n)) % ky
    channel, optimal, mi, capacity = edge_channel(x, y, "plugin")
    assert 0.0 <= mi <= capacity + 1e-9
    assert capacity <= np.log2(min(channel.sizes["input"], channel.sizes["output"])) + 1e-9
    assert float(optimal.sum()) == pytest.approx(1.0)


def test_layer_separates_capacity_from_driven_information():
    """
    A biased source drives a binary symmetric channel below its capacity.
    """
    net = infer_multiplex(_bsc(20000, 2, 0.1, 0.1, 0), max_lag=3, channels=True, **QUICK)
    assert int(net.dataset["delay"].sel(source="x0", target="x1")) == 2
    edge = net.channels.dataset.sel(source="x0", target="x1")
    assert float(edge["capacity"]) == pytest.approx(1 - _h2(0.1), abs=0.02)
    expected_mi = _h2(0.1 * 0.9 + 0.9 * 0.1) - _h2(0.1)
    assert float(edge["mi"]) == pytest.approx(expected_mi, abs=0.02)
    assert int(edge["n_inputs"]) == int(edge["n_outputs"]) == 2
    assert np.isnan(float(net.channels.dataset["mi"].sel(source="x1", target="x0")))
    assert set(net.channels.channels) == {("x0", "x1")}
    assert "capacity" in net.summary()
    assert net.settings["channel_settings"]["estimator"] is None


def test_infer_channels_on_an_inferred_network(tmp_path):
    data = DiscreteData.from_discrete(_bsc(6000, 1, 0.2, 0.5, 1))
    net = infer_multiplex(data, max_lag=2, **QUICK)
    layer = infer_channels(data, net, {"estimator": "plugin"})
    assert layer.dataset.attrs["estimator"] == "plugin"
    assert float(layer.dataset["capacity"].sel(source="x0", target="x1")) == pytest.approx(1 - _h2(0.2), abs=0.03)
    net.channels = layer
    path = tmp_path / "net.pkl"
    net.save(path)
    loaded = infoflow.MultiplexNetwork.load(path)
    assert loaded.channels.channels[("x0", "x1")].equals(layer.channels[("x0", "x1")])
