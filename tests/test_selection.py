"""
Tests for skeleton inference.
"""

import dataclasses
from typing import Any

import numpy as np
import pytest

from infoflow.data import DiscreteData
from infoflow.embedding import Embedding
from infoflow.preprocess import Ordinal, discretize
from infoflow.selection import SkeletonSettings, check_n_perm, infer_skeleton, select_parents

FAST: dict[str, Any] = dict(
    n_perm_max_stat=50,
    n_perm_min_stat=50,
    n_perm_omnibus=100,
    n_perm_max_seq=100,
    n_perm_pairs=50,
    n_perm_tdmi=100,
)


def _chain(n, rng):
    a = rng.integers(0, 2, n)
    b = np.roll(a, 1) ^ (rng.random(n) < 0.1)
    c = np.roll(b, 2) ^ (rng.random(n) < 0.1)
    return np.stack([a, b, c], axis=1)


def test_chain_recovers_parents_and_lags():
    data = DiscreteData.from_discrete(_chain(3000, np.random.default_rng(0)))
    result = infer_skeleton(data, Embedding(max_lag=3), settings=SkeletonSettings(**FAST), prng=0)
    assert result[0].parents() == []
    assert result[1].parents() == [0] and result[1].sources == [(0, 1)]
    assert result[2].parents() == [1] and result[2].sources == [(1, 2)]


def test_xor_needs_pair_search():
    rng = np.random.default_rng(1)
    n = 3000
    x1, x2 = rng.integers(0, 2, n), rng.integers(0, 2, n)
    y = np.roll(x1 ^ x2, 1)
    data = DiscreteData.from_discrete(np.stack([x1, x2, y], axis=1))
    with_pairs = select_parents(data, 2, Embedding(max_lag=2), settings=SkeletonSettings(**FAST), prng=0)
    assert sorted(with_pairs.sources) == [(0, 1), (1, 1)]
    assert with_pairs.pairs and with_pairs.significant
    without = select_parents(
        data, 2, Embedding(max_lag=2), settings=SkeletonSettings(**FAST, synergy_search=False), prng=0
    )
    assert not {(0, 1), (1, 1)} <= set(without.sources)


def test_target_past_selected():
    rng = np.random.default_rng(2)
    n = 3000
    x = np.zeros(n, dtype=int)
    for t in range(2, n):
        x[t] = x[t - 2] if rng.random() < 0.9 else int(rng.integers(2))
    data = DiscreteData.from_discrete(np.stack([x, rng.integers(0, 2, n)], axis=1))
    result = select_parents(data, 0, Embedding(max_lag=3), settings=SkeletonSettings(**FAST), prng=0)
    assert result.target_past == [(0, 2)]
    assert result.sources == []


def test_tdmi_screen_finds_indirect_dependence():
    data = DiscreteData.from_discrete(_chain(3000, np.random.default_rng(3)))
    result = infer_skeleton(
        data, Embedding(max_lag=3), settings=dataclasses.replace(SkeletonSettings(**FAST), n_perm_tdmi=200), prng=0
    )
    candidate = result[2].tdmi_candidates[0]
    assert candidate["variable"] == (0, 3)
    assert candidate["significant"]


def test_forced_conditionals_and_faes():
    rng = np.random.default_rng(4)
    n = 3000
    z = rng.integers(0, 2, n)
    x = np.roll(z, 1) ^ (rng.random(n) < 0.05)
    y = np.roll(z, 2) ^ (rng.random(n) < 0.05)
    data = DiscreteData.from_discrete(np.stack([x, y, z], axis=1))
    # Without z as a candidate, x's past predicts y (common driver).
    hidden = select_parents(data, 1, Embedding(max_lag=2), sources=[0], settings=SkeletonSettings(**FAST), prng=0)
    assert hidden.sources
    forced = select_parents(
        data,
        1,
        Embedding(max_lag=2),
        sources=[0],
        settings=SkeletonSettings(**FAST, forced_conditionals=((2, 2),)),
        prng=0,
    )
    assert forced.sources == [] and (2, 2) in forced.conditionals
    faes = select_parents(data, 1, Embedding(max_lag=2), settings=SkeletonSettings(**FAST, faes=True), prng=0)
    assert (0, 0) in faes.conditionals and (2, 0) in faes.conditionals


@pytest.mark.parametrize("scheme", ["random", "circular", "block", "local", "trials"])
def test_permutation_schemes(scheme):
    rng = np.random.default_rng(5)
    trials = [_chain(400, rng) for _ in range(6)]
    data = DiscreteData.from_discrete(trials)
    result = select_parents(
        data, 1, Embedding(max_lag=2), settings=SkeletonSettings(**FAST, permutation=scheme), prng=0
    )
    assert result.sources == [(0, 1)]


def test_ordinal_lag_grid_is_non_overlapping():
    x = np.random.default_rng(6).normal(size=(500, 2))
    data = discretize(x, Ordinal(3, 1))
    from infoflow.embedding import candidate_lags

    lags = candidate_lags(data, Embedding(max_lag=9), 0)
    assert lags == [1, 4, 7]


def test_check_n_perm():
    with pytest.raises(ValueError):
        check_n_perm(10, 0.05)
    check_n_perm(19, 0.05)
    with pytest.raises(ValueError):
        SkeletonSettings(n_perm_omnibus=5).check()


def test_ksg_selection_on_raw_values():
    from infoflow.preprocess import preprocess

    rng = np.random.default_rng(4)
    n = 800
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.4 * y[t - 1] + 0.8 * x[t - 1] + 0.5 * rng.normal()
    result = preprocess(np.stack([x, y], axis=1), max_lag=2, prng=0)
    settings = dataclasses.replace(SkeletonSettings(**FAST), estimator="ksg", synergy_search=False, tdmi_screen=False)
    sk = select_parents(result.data, 1, result.embeddings, None, settings, 0)
    assert sk.parents() == [0] and (0, 1) in sk.sources
    with pytest.raises(ValueError, match="raw series"):
        select_parents(
            DiscreteData.from_discrete([np.stack([x > 0, y > 0], axis=1).astype(int)]), 1, None, None, settings, 0
        )


def test_ksg_local_null_keeps_children_out_of_parent_sets():
    from infoflow.preprocess import preprocess

    rng = np.random.default_rng(6)
    n = 800
    x, child = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal()
        child[t] = x[t - 1] + 0.3 * rng.normal()
    result = preprocess(np.stack([x, child], axis=1), max_lag=2, prng=0)
    settings = dataclasses.replace(
        SkeletonSettings(**FAST), estimator="ksg", ksg_null="local", synergy_search=False, tdmi_screen=False
    )
    assert select_parents(result.data, 0, result.embeddings, None, settings, 0).sources == []
    assert select_parents(result.data, 1, result.embeddings, None, settings, 0).parents() == [0]
    with pytest.raises(ValueError, match="ksg_null"):
        SkeletonSettings(ksg_null="nearby").check()


def test_curtailed_tests_reach_the_same_decisions():
    data = DiscreteData.from_discrete(_chain(3000, np.random.default_rng(5)))
    for t in range(3):
        full = select_parents(data, t, Embedding(max_lag=3), None, SkeletonSettings(**FAST, curtail=False), 0)
        cut = select_parents(data, t, Embedding(max_lag=3), None, SkeletonSettings(**FAST, curtail=True), 0)
        assert full.target_past == cut.target_past and full.sources == cut.sources
        assert full.significant == cut.significant


def test_ksg_local_permutation_and_prescreen():
    from infoflow.preprocess import preprocess
    from infoflow.selection import _KsgColumns

    rng = np.random.default_rng(7)
    n = 800
    x, w = rng.normal(size=n), rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.4 * y[t - 1] + 0.8 * x[t - 1] + 0.5 * rng.normal()
    result = preprocess(np.stack([x, w, y], axis=1), max_lag=2, prng=0)
    cols = _KsgColumns(result.data, 2, [(2, 1), (0, 1), (1, 1)], 2)
    z = cols.joint([(2, 1)])[0]
    choice = cols.local_permutation(z, np.random.default_rng(0))
    neighbors = cols._contexts[("neighbors", z.shape, hash(np.asarray(z, dtype=float).tobytes()))]
    assert all(c in row for c, row in zip(choice, neighbors, strict=True))
    assert len(np.unique(choice)) > 0.9 * len(choice)
    settings = dataclasses.replace(
        SkeletonSettings(**FAST), estimator="ksg", prescreen_alpha=0.05, synergy_search=False, tdmi_screen=False
    )
    sk = select_parents(result.data, 2, result.embeddings, None, settings, 0)
    assert sk.parents() == [0] and 0 in sk.prescreened
    full = select_parents(result.data, 2, result.embeddings, None, dataclasses.replace(settings, curtail=False), 0)
    assert full.sources == sk.sources


def test_gaussian_estimator():
    from infoflow.benchmarks import random_network, scaling_data, simulate_network
    from infoflow.preprocess import preprocess
    from infoflow.selection import _GaussianColumns

    g = _GaussianColumns.__new__(_GaussianColumns)
    x, y, z, truth = scaling_data(20000, 3, "gaussian", prng=0)
    g.y, g.n, g._bases = (y - y.mean()) / y.std(), len(y), {}
    assert g.cmi(x, 1, z, 3) == pytest.approx(truth, abs=0.02)
    idx = [np.random.default_rng(s).permutation(g.n) for s in range(3)]
    assert np.allclose(g.cmi_batch(x, 1, z, 3, idx), [g.cmi(x[i], 1, z, 3) for i in idx])
    xi, yi, zi, _ = scaling_data(20000, 3, "gaussian", coupled=False, prng=1)
    g.y, g._bases = (yi - yi.mean()) / yi.std(), {}
    assert g.cmi(xi, 1, zi, 3) < 0.002

    graph = random_network(6, prng=3)
    result = preprocess(simulate_network(graph, 4000, "var", prng=3), max_lag=5, prng=0)
    settings = dataclasses.replace(SkeletonSettings(**FAST), estimator="gaussian", synergy_search=False)
    for t in range(6):
        sk = select_parents(result.data, t, result.embeddings, None, settings, 0)
        true = {s for s, _ in graph.in_edges(t)}
        assert set(sk.parents()) <= true  # no false parents
        assert len(set(sk.parents()) & true) >= len(true) - 1  # at most one weak link missed
    with pytest.raises(ValueError, match="raw series"):
        select_parents(DiscreteData.from_discrete([np.zeros((100, 2), dtype=int)]), 1, None, None, settings, 0)


def test_screened_candidates_enter_null():
    from infoflow.benchmarks import random_network, simulate_network
    from infoflow.preprocess import preprocess
    from infoflow.selection import _GaussianColumns, _max_statistic

    graph = random_network(12, mean_in_degree=0.0, prng=2)
    result = preprocess(simulate_network(graph, 3000, "var", prng=2), max_lag=2, prng=0)
    cands = [(p, 1) for p in range(1, 12)]
    cols = _GaussianColumns(result.data, 0, [(0, 1), *cands], 2)
    z, Kz = cols.context([(0, 1)])
    obs = [cols.cmi(cols.column(v)[0], cols.alphabet[v], z, Kz) for v in cands]
    top = [cands[i] for i in np.argsort(obs)[-2:]]
    rest = [v for v in cands if v not in top]

    def pvalue(screened):
        rng = np.random.default_rng(0)
        return _max_statistic(cols, top, [(0, 1)], 200, lambda z=None: rng.permutation(cols.n), screened=screened)[2]

    biased, corrected = pvalue(()), pvalue(rest)
    assert corrected >= biased
    assert corrected > 0.05


def test_trend_and_coarse_estimators():
    from infoflow.benchmarks import random_network, simulate_network
    from infoflow.preprocess import preprocess
    from infoflow.selection import _CoarseColumns, _TrendColumns

    graph = random_network(6, prng=3)
    result = preprocess(simulate_network(graph, 4000, "var", prng=3), max_lag=5, prng=0)
    target = max(graph.nodes, key=graph.in_degree)
    variables = [(target, 1), *[(p, lag) for p in graph.nodes if p != target for lag in (1, 2)]]
    trend = _TrendColumns(result.data, target, variables, 2)
    x, _ = trend.column(variables[1])
    z, Kz = trend.context([variables[0], variables[2]])
    dummies = np.eye(Kz)[z]
    y = trend._yscore

    def residual(a, b):
        return a - b @ np.linalg.lstsq(b, a, rcond=None)[0]

    ey, ex = residual(y, dummies), residual(x, dummies)
    explicit = 0.5 * np.log2((ey @ ey) / (residual(ey, ex[:, None]) @ residual(ey, ex[:, None])))
    assert trend.cmi(x, 1, z, Kz) == pytest.approx(explicit, rel=1e-8)
    idx = [np.random.default_rng(s).permutation(trend.n) for s in range(3)]
    assert np.allclose(trend.cmi_batch(x, 1, z, Kz, idx), [trend.cmi(x[i], 1, z, Kz) for i in idx])
    pair, _ = trend.joint(variables[1:3])
    assert trend.cmi(pair, 2, z, Kz) >= trend.cmi(x, 1, z, Kz) - 1e-12

    coarse = _CoarseColumns(result.data, target, variables, 2)
    bins = [coarse.context_bins(d) for d in range(1, 6)]
    assert bins == sorted(bins, reverse=True) and bins[0] <= 4 and bins[-1] >= 2
    assert coarse.context(variables[:3])[1] <= coarse.context_bins(3) ** 3

    for estimator in ("trend", "coarse"):
        settings = dataclasses.replace(SkeletonSettings(**FAST), estimator=estimator, synergy_search=False)
        sk = select_parents(result.data, target, result.embeddings, None, settings, 0)
        assert set(sk.parents()) <= set(graph.predecessors(target))
        assert sk.parents()
