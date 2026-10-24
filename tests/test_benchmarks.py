"""
Tests for the conditioning-scaling benchmark.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from infoflow.benchmarks import conditioning_scaling, scaling_data


def test_analytic_values():
    assert scaling_data(10, 0, "gaussian")[3] == pytest.approx(-0.5 * np.log2(1 - 0.36))
    h2 = lambda p: -p * np.log2(p) - (1 - p) * np.log2(1 - p)  # noqa: E731
    assert scaling_data(10, 0, "discrete")[3] == pytest.approx(h2(0.34) - h2(0.1))
    assert scaling_data(10, 3, "discrete", coupled=False)[3] == 0.0
    ds = conditioning_scaling("discrete", ("plugin",), n_samples=(200_000,), dims=(0,), n_reps=1, prng=0)
    assert float(ds["bias"].squeeze()) == pytest.approx(0, abs=0.005)


def test_symbol_estimates_inflate_then_collapse():
    ds = conditioning_scaling(
        "gaussian", ("plugin",), n_samples=(1000,), dims=(0, 4, 12), n_reps=3, coupled=False, prng=0
    )
    spurious = ds["mean"].sel(estimator="plugin", n=1000).values
    assert spurious[0] < 0.02 and spurious[1] > 0.3 and spurious[2] < 0.05


def test_ksg_never_manufactures_dependence_and_decays_downward():
    null = conditioning_scaling(
        "gaussian", ("ksg",), n_samples=(1000,), dims=(0, 4, 8), n_reps=3, coupled=False, prng=0
    )
    assert np.all(np.abs(null["mean"].values) < 0.03)
    ds = conditioning_scaling("gaussian", ("ksg",), n_samples=(1000,), dims=(0, 8), n_reps=3, prng=0)
    truth = ds.attrs["truth"]
    near, far = ds["mean"].sel(estimator="ksg", n=1000).values
    assert near == pytest.approx(truth, abs=0.06)
    assert 0.1 < far < truth


def test_random_network_and_simulation():
    from infoflow.benchmarks import random_network, simulate_network

    sizes = [random_network(30, 3.0, prng=s).number_of_edges() / 30 for s in range(20)]
    assert 2.4 < np.mean(sizes) < 3.4
    graph = random_network(12, prng=1)
    assert all(1 <= d["lag"] <= 5 for _, _, d in graph.edges(data=True))
    assert not any(s == t for s, t in graph.edges())
    var = simulate_network(graph, 2000, "var", prng=0)
    assert var.shape == (2000, 12) and np.all(np.isfinite(var)) and var.std() < 1
    logistic = simulate_network(graph, 500, "logistic", prng=0)
    assert np.all((logistic >= 0) & (logistic < 1))
    assert random_network(10, 0.0, prng=0).number_of_edges() == 0


def test_score_network_counts():
    from types import SimpleNamespace

    import networkx as nx
    import xarray as xr

    from infoflow.benchmarks import score_network
    from infoflow.selection import TargetSkeleton

    graph = nx.DiGraph()
    graph.add_nodes_from(range(3))
    graph.add_edge(0, 1, lag=2)
    graph.add_edge(1, 2, lag=1)
    skeleton = {
        1: TargetSkeleton(target=1, sources=[(0, 2)], significant=True),
        2: TargetSkeleton(target=2, sources=[(0, 3)], significant=True),
    }
    sig = np.zeros((1, 3, 3), dtype=bool)
    sig[0, 0, 1] = True
    delay = np.zeros((3, 3), dtype=int)
    delay[0, 1] = 3
    dataset = xr.Dataset(
        {"significant": (("layer", "source", "target"), sig), "delay": (("source", "target"), delay)},
        coords={"layer": ["intrinsic"]},
    )
    scores = score_network(SimpleNamespace(skeleton=skeleton, dataset=dataset), graph)
    sk = scores["skeleton"]
    assert sk["precision"] == 0.5 and sk["recall"] == 0.5 and sk["specificity"] == pytest.approx(3 / 4)
    assert sk["lag_error"] == pytest.approx(1 / 0.5)  # |3 - 2| over the chance error (L^2 - 1) / 3L for L = 2
    assert scores["intrinsic"]["n_found"] == 1 and scores["intrinsic"]["recall"] == 0.5


def test_rank_metrics_examples():
    from infoflow.benchmarks import rank_metrics

    assert rank_metrics([3, 2, 1, 0], [True, True, False, False]) == (1.0, 1.0)
    assert rank_metrics([0, 1, 2, 3], [True, True, False, False])[0] == 0.0
    auroc, auprc = rank_metrics([1, 1, 1, 1], [True, False, True, False])
    assert auroc == 0.5 and auprc == pytest.approx((1 / 3 + 2 / 4) / 2)
    assert rank_metrics([np.nan, 2.0, 1.0], [True, False, False]) == (0.0, pytest.approx(1 / 3))
    assert all(np.isnan(rank_metrics([1, 2], [True, True])))


@given(st.lists(st.tuples(st.floats(-5, 5), st.booleans()), min_size=2, max_size=40))
def test_rank_metrics_matches_pairwise_definition(items):
    from infoflow.benchmarks import rank_metrics

    score = np.array([s for s, _ in items])
    truth = np.array([t for _, t in items])
    auroc, auprc = rank_metrics(score, truth)
    if truth.all() or not truth.any():
        assert np.isnan(auroc)
        return
    pos, neg = score[truth], score[~truth]
    pairwise = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    assert auroc == pytest.approx(pairwise)
    assert 0 <= auprc <= 1


def test_score_edges_ranks_with_scores():
    import networkx as nx

    from infoflow.benchmarks import score_edges

    graph = nx.DiGraph()
    graph.add_nodes_from(range(3))
    graph.add_edge(0, 1, lag=1)
    score = np.full((3, 3), 0.0)
    score[0, 1] = 1.0
    out = score_edges({(0, 1)}, None, score, graph)
    assert out["auroc"] == 1.0 and out["auprc"] == 1.0 and np.isnan(out["lag_error"])


def test_edge_scores_rank_true_parent_first():
    from infoflow import datasets
    from infoflow.network import infer_multiplex
    from infoflow.selection import SkeletonSettings

    data = datasets.chain(2000, seed=0)
    net = infer_multiplex(
        data, skeleton_settings=SkeletonSettings(edge_scores=True), interpret=False, node_layer=False, prng=0
    )
    score = net.dataset["score"].values
    assert np.isfinite(score[0, 1]) and score[0, 1] > score[2, 1]
    assert np.isfinite(score[1, 2]) and score[1, 2] > score[0, 2]


@pytest.mark.slow
def test_small_network_validation():
    from infoflow.benchmarks import network_validation

    ds = network_validation("logistic", n_nodes=(6,), n_samples=(2000,), n_reps=1, prng=0)
    skeleton = ds.sel(level="skeleton")
    assert float(skeleton["precision"].squeeze()) >= 0.9
    assert float(skeleton["specificity"].squeeze()) >= 0.95
    assert float(skeleton["recall"].squeeze()) >= 0.5
