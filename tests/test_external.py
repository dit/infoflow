"""
Tests for the Lorenz-96 simulator and the external-benchmark loaders and exporters.
"""

import json
import pickle

import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from infoflow.benchmarks import lorenz96_graph
from infoflow.datasets import lorenz96
from infoflow.io import (
    adjacency_graph,
    causeme_matrices,
    load_causalrivers,
    load_causaltime,
    load_netsim,
    sample_subgraph,
    to_causeme,
)


def test_lorenz96_shape_and_chaos():
    x = lorenz96(500, n_vars=8, forcing=10.0, seed=0)
    assert x.shape == (500, 8) and np.all(np.isfinite(x))
    assert x.std() > 1.0  # chaotic, not at the fixed point x_i = F
    assert np.allclose(lorenz96(200, n_vars=6, forcing=0.5, noise_std=0, seed=0)[-1], 0.5, atol=1e-3)


def test_lorenz96_graph():
    g = lorenz96_graph(6)
    assert g.number_of_edges() == 18
    assert set(g.predecessors(0)) == {4, 5, 1}
    assert all(d["lag"] == 1 for _, _, d in g.edges(data=True))


def test_lorenz96_parent_drives_child():
    from infoflow.te import transfer_entropy

    x = lorenz96(3000, n_vars=6, seed=1)
    sym = (x > np.median(x, axis=0)).astype(int)
    assert transfer_entropy(sym[:, 5], sym[:, 0]) > transfer_entropy(sym[:, 3], sym[:, 0])


def test_load_netsim(tmp_path):
    from scipy.io import savemat

    net = np.zeros((2, 3, 3))
    net[:, 0, 1] = 0.5
    net[1, 1, 2] = 0.4
    net[:, [0, 1, 2], [0, 1, 2]] = -1
    ts = np.arange(2 * 4 * 3, dtype=float).reshape(8, 3)
    savemat(tmp_path / "sim.mat", {"ts": ts, "net": net, "Ntimepoints": 4, "Nnodes": 3, "Nsubjects": 2})
    data, graph = load_netsim(tmp_path / "sim.mat", subject=1)
    assert np.array_equal(data, ts[4:])
    assert set(graph.edges()) == {(0, 1), (1, 2)}


def test_load_causaltime_keeps_observed_block():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 50, 4))
    A = np.zeros((4, 4))
    A[0, 1] = 1
    A[2, 3] = 1  # residual block: dropped
    data, graph = load_causaltime(X, A)
    assert len(data) == 3 and data[0].shape == (50, 2)
    assert set(graph.edges()) == {(0, 1)}
    _, flipped = load_causaltime(X[0], A, orientation="target_source")
    assert set(flipped.edges()) == {(1, 0)}
    _, full = load_causaltime(X[0], A, residual_block=False)
    assert set(full.edges()) == {(0, 1), (2, 3)}


def test_load_causalrivers(tmp_path):
    g = nx.DiGraph([(101, 102), (102, 103), (104, 103)])
    with open(tmp_path / "g.p", "wb") as fh:
        pickle.dump(g, fh)
    rows = ["datetime,101,102,103,104", "t0,1,2,3,4", "t1,1.5,,3,4", "t2,2,3,4,5"]
    (tmp_path / "ts.csv").write_text("\n".join(rows))
    data, graph, names = load_causalrivers(tmp_path / "ts.csv", tmp_path / "g.p", nodes=[102, 103, 104])
    assert names == ["102", "103", "104"]
    assert data.shape == (3, 3) and data[1, 0] == 2.5  # the gap is interpolated
    dropped, _, _ = load_causalrivers(tmp_path / "ts.csv", tmp_path / "g.p", nodes=[102, 103, 104], gaps="drop")
    assert dropped.shape == (2, 3)
    assert set(graph.edges()) == {(0, 1), (2, 1)}
    assert graph.nodes[0]["station"] == 102


@settings(max_examples=25, deadline=None)
@given(st.integers(2, 6), st.integers(0, 2**31 - 1))
def test_sample_subgraph_is_weakly_connected(k, seed):
    g = nx.gnp_random_graph(15, 0.3, seed=seed % 1000, directed=True)
    big = max(nx.weakly_connected_components(g), key=len)
    if len(big) < k:
        return
    g = g.subgraph(big).copy()
    nodes = sample_subgraph(g, k, prng=seed)
    assert len(set(nodes)) == k and nx.is_weakly_connected(g.subgraph(nodes))


def test_causeme_export_round_trip(tmp_path):
    from types import SimpleNamespace

    import xarray as xr

    from infoflow.selection import TargetSkeleton

    score = np.array([[np.nan, 0.3, -0.1], [0.0, np.nan, 0.2], [0.05, 0.0, np.nan]])
    delay = np.zeros((3, 3), dtype=int)
    delay[0, 1] = 2
    ds = xr.Dataset({"score": (("source", "target"), score), "delay": (("source", "target"), delay)})
    skeleton = {1: TargetSkeleton(target=1, sources=[(0, 2)], source_pvalues={(0, 2): 0.01}, significant=True)}
    net = SimpleNamespace(dataset=ds, skeleton=skeleton)
    m = causeme_matrices(net)
    assert m["scores"][0, 1] == 0.3 and m["scores"][0, 2] == 0.0 and m["scores"][1, 1] == 0.0
    assert m["pvalues"][0, 1] == 0.01 and m["pvalues"][1, 2] == 1.0
    assert m["lags"][0, 1] == 2 and m["lags"][1, 2] == 0
    out = to_causeme([net, {"scores": np.eye(3)}], "abc", path=tmp_path / "r.json", experiment="x")
    assert "pvalues" not in out and len(out["scores"]) == 2 and len(out["scores"][0]) == 9
    loaded = json.loads((tmp_path / "r.json").read_text())
    assert loaded["method_sha"] == "abc" and loaded["experiment"] == "x"
    assert np.allclose(np.reshape(loaded["scores"][0], (3, 3)), m["scores"])


def test_adjacency_graph_ignores_diagonal():
    g = adjacency_graph(np.array([[1, 1], [0, 1]]), lag=3)
    assert list(g.edges(data=True)) == [(0, 1, {"lag": 3})]


def test_score_edges_without_lags():
    from infoflow.benchmarks import score_edges

    g = adjacency_graph(np.array([[0, 1], [0, 0]]))
    out = score_edges({(0, 1)}, np.zeros((2, 2), dtype=int), None, g)
    assert out["precision"] == 1.0 and np.isnan(out["lag_error"])


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])


def test_causalrivers_subset_resamples_and_drops_gaps():
    import networkx as nx

    from infoflow.io import causalrivers_subset

    values = np.arange(16, dtype=np.float32).reshape(8, 2)
    values[1, 0] = np.nan
    values[4:6, 1] = np.nan
    g = nx.DiGraph([(7, 9)])
    data, truth = causalrivers_subset(values, g, ["7", "9"], [7, 9], resample=2, gaps="drop")
    # Block means ignore missing values; the block with no value for station 9 is dropped.
    assert np.allclose(data, [[0.0, 2.0], [5.0, 6.0], [13.0, 14.0]])
    filled, _ = causalrivers_subset(values, g, ["7", "9"], [7, 9], resample=2)
    assert np.allclose(filled, [[0.0, 2.0], [5.0, 6.0], [9.0, 10.0], [13.0, 14.0]])
    assert set(truth.edges()) == {(0, 1)}
