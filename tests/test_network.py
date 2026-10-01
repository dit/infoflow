"""
Phase 2 milestone: multiplex networks on benchmarks with known structure.
"""

import networkx as nx
import numpy as np
import pytest

from infoflow import datasets
from infoflow.network import infer_multiplex
from infoflow.preprocess import EqualFrequency, discretize
from infoflow.selection import SkeletonSettings

FAST = SkeletonSettings(
    n_perm_max_stat=100,
    n_perm_min_stat=100,
    n_perm_omnibus=200,
    n_perm_max_seq=200,
    n_perm_pairs=50,
    n_perm_tdmi=100,
)


def _sig(net, layer, s, t):
    names = net.names
    return bool(net.dataset["significant"].sel(layer=layer, source=names[s], target=names[t]))


def test_xor_is_synergistic():
    net = infer_multiplex(datasets.xor_synergy(3000, seed=1), max_lag=2, skeleton_settings=FAST, prng=0)
    for s in (0, 1):
        assert net.dataset["kind"].values[s, 2] == "parent"
        assert _sig(net, "synergistic", s, 2)
        assert net.dataset["weight"].sel(layer="synergistic").values[s, 2] > 0.9


def test_common_driver_is_shared():
    net = infer_multiplex(datasets.common_driver(3000, seed=1), max_lag=3, skeleton_settings=FAST, prng=0)
    kinds = net.dataset["kind"].values
    assert kinds[2, 0] == "parent" and kinds[2, 1] == "parent"
    assert _sig(net, "intrinsic", 2, 0) and _sig(net, "intrinsic", 2, 1)
    # x -> y is not a parent once z is conditioned on; its dependence is shared.
    assert kinds[0, 1] == "shared_candidate"
    assert _sig(net, "shared", 0, 1) and not _sig(net, "intrinsic", 0, 1)


def test_chain_layers_and_delays():
    net = infer_multiplex(datasets.chain(3000, seed=2), max_lag=3, skeleton_settings=FAST, prng=0)
    delay = net.dataset["delay"].values
    assert _sig(net, "intrinsic", 0, 1) and delay[0, 1] == 1
    assert _sig(net, "intrinsic", 1, 2) and delay[1, 2] == 2
    assert not _sig(net, "intrinsic", 0, 2)


def test_mute_network_true_edges_are_intrinsic():
    x = datasets.mute_network(3000, seed=0)
    net = infer_multiplex(
        discretize(x, EqualFrequency(4)), max_lag=4, n_boot=50, n_null=50, skeleton_settings=FAST, prng=0
    )
    recovered = [e for e in datasets.MUTE_EDGES if _sig(net, "intrinsic", *e)]
    assert len(recovered) >= 4
    assert all(int(net.dataset["delay"].values[e]) == lag for e, lag in datasets.MUTE_EDGES.items() if e in recovered)
    intrinsic_edges = {(s, t) for (s, t) in net.edges if _sig(net, "intrinsic", s, t)}
    assert len(intrinsic_edges - set(datasets.MUTE_EDGES)) <= 3


def test_result_objects():
    net = infer_multiplex(
        datasets.chain(2000, seed=3), max_lag=3, names=["a", "b", "c"], skeleton_settings=FAST, prng=0
    )
    ds = net.dataset
    assert set(ds.dims) >= {"layer", "source", "target", "node"}
    assert list(ds.coords["source"].values) == ["a", "b", "c"]
    g = net.to_networkx()
    assert isinstance(g, nx.MultiDiGraph)
    assert ("a", "b", "intrinsic") in g.edges(keys=True)
    assert net.layer("intrinsic").has_edge("a", "b")
    assert "a" in net.summary()
    assert net.settings["estimator"] == "miller_madow"
    te = ds["weight"].sel(layer="te").values
    parts = ds["weight"].sel(layer="intrinsic").values + ds["weight"].sel(layer="synergistic").values
    mask = np.isfinite(te)
    assert np.allclose(te[mask], parts[mask])


def test_holdout_runs():
    net = infer_multiplex(datasets.chain(4000, seed=4), max_lag=3, skeleton_settings=FAST, holdout=0.5, prng=0)
    assert _sig(net, "intrinsic", 0, 1)
    assert net.settings["holdout"] == 0.5
    assert net.edges[(0, 1)].flow.n_samples < 2000


@pytest.mark.parametrize(
    "make",
    [
        lambda: datasets.common_driver(3000, seed=0),
        lambda: datasets.chain(3000, seed=1),
        lambda: datasets.orthogonal_features(3000, seed=2),
    ],
)
def test_layers_respect_te_bound_and_flag_unconfirmed_parents(make):
    """
    Intrinsic and synergistic flow are at most TE: never significant on edges whose TE
    selection rejected; parents with neither layer significant are flagged.
    """
    net = infer_multiplex(make(), max_lag=3, prng=0)
    ds = net.dataset
    parent = ds["kind"].values == "parent"
    intrinsic = ds["significant"].sel(layer="intrinsic").values
    synergistic = ds["significant"].sel(layer="synergistic").values
    assert not ((intrinsic | synergistic) & ~parent).any()
    unconfirmed = np.char.find(ds["flags"].values.astype(str), "layers-unconfirmed") >= 0
    assert np.array_equal(unconfirmed, parent & ~intrinsic & ~synergistic)
