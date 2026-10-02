"""
Tests for the optional lag-0 layer and the PID hyperedge layer.
"""

import numpy as np
import pytest

from infoflow import datasets
from infoflow.contemporaneous import contemporaneous_layer, to_tigramite
from infoflow.data import DiscreteData
from infoflow.embedding import Embedding
from infoflow.hyperedges import pid_hyperedges
from infoflow.selection import SkeletonSettings, infer_skeleton

FAST = SkeletonSettings(
    n_perm_max_stat=50, n_perm_min_stat=50, n_perm_omnibus=100, n_perm_max_seq=100, n_perm_pairs=50, n_perm_tdmi=50
)


def test_contemporaneous_xor_leaves_no_pairwise_links():
    """
    y_t = x_t xor z_t: every pair is independent (faithfulness fails), so no x - z link appears.
    """
    rng = np.random.default_rng(0)
    n = 4000
    x, z = rng.integers(0, 2, n), rng.integers(0, 2, n)
    y = x ^ z
    y = np.where(rng.random(n) < 0.9, y, 1 - y)
    data = DiscreteData.from_discrete(np.stack([x, y, z], axis=1))
    skeleton = infer_skeleton(data, Embedding(max_lag=1), settings=FAST, prng=0)
    graph, values, _ = contemporaneous_layer(data, skeleton, n_perm=99, prng=0)
    assert graph[0, 2] == 0


def test_contemporaneous_chain_orientation():
    rng = np.random.default_rng(1)
    n = 4000
    x, z = rng.integers(0, 2, n), rng.integers(0, 2, n)
    y = np.where(rng.random(n) < 0.5, x, z)
    data = DiscreteData.from_discrete(np.stack([x, y, z], axis=1))
    skeleton = infer_skeleton(data, Embedding(max_lag=1), settings=FAST, prng=0)
    graph, _, _ = contemporaneous_layer(data, skeleton, n_perm=99, prng=0)
    assert graph[0, 1] == 2 and graph[2, 1] == 2
    assert graph[0, 2] == 0


def test_pid_hyperedges_xor():
    data = DiscreteData.from_discrete(datasets.xor_synergy(3000, seed=0))
    skeleton = infer_skeleton(data, Embedding(max_lag=1), settings=FAST, prng=0)
    edges = pid_hyperedges(data, skeleton)
    assert len(edges) == 1
    (e,) = edges
    assert e["target"] == 2 and e["sources"] == (0, 1)
    assert e["synergy"] == pytest.approx(1.0, abs=0.02)
    assert e["redundancy"] == pytest.approx(0.0, abs=0.02)


def test_tigramite_adapter_optional():
    pytest.importorskip("tigramite")
    data = DiscreteData.from_discrete(datasets.chain(500, seed=0))
    frame = to_tigramite(data)
    assert frame.N == 3


def test_infer_multiplex_optional_layers():
    from infoflow.network import infer_multiplex

    net = infer_multiplex(
        datasets.xor_synergy(2000, seed=0),
        max_lag=1,
        skeleton_settings=FAST,
        contemporaneous=True,
        hyperedges=True,
        prng=0,
    )
    assert "contemporaneous" in net.dataset
    h = net.hyperedges
    assert h.sizes["hyperedge"] == 1 and float(h["synergy"].squeeze()) > 0.9
    assert str(h["consensus"].item()) == "synergistic" and bool(h["significant"].item())
