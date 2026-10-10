"""
Tests for estimator ensembles in parent selection and the automatic pipeline.
"""

import dataclasses
from typing import Any

import numpy as np
import pytest

from infoflow.selection import SkeletonSettings, _EnsembleStages, select_parents

FAST: dict[str, Any] = dict(
    n_perm_max_stat=200,
    n_perm_min_stat=200,
    n_perm_omnibus=200,
    n_perm_max_seq=200,
    n_perm_pairs=200,
    n_perm_tdmi=200,
)


class _Stub:
    def __init__(self, name, sequential, omnibus):
        self.name = name
        self._sequential = sequential
        self._omnibus = omnibus

    def sequential(self, sources, cond, alpha=None):
        return self._sequential

    def omnibus(self, sources, cond, alpha=None):
        return self._omnibus


def test_ensemble_combines_by_bonferroni():
    settings = SkeletonSettings()
    members = [
        _Stub("a", {"u": 0.004, "v": 0.5}, (0.2, 0.03)),
        _Stub("b", {"u": 0.2, "v": 0.01}, (0.1, 0.004)),
        _Stub("c", {"u": 0.9, "v": 0.9}, (0.3, 0.6)),
    ]
    stages = _EnsembleStages(members, settings)
    assert stages.sequential(["u", "v"], []) == pytest.approx({"u": 0.012, "v": 0.03})
    assert stages.omnibus(["u", "v"], []) == pytest.approx((0.1, 0.012))
    assert stages._combine(0.5) == 1.0


def test_ensemble_settings_are_checked():
    with pytest.raises(ValueError):
        SkeletonSettings(estimator=("gaussian", "gaussian")).check()
    with pytest.raises(ValueError):
        SkeletonSettings(estimator=("gaussian", "nope")).check()
    with pytest.raises(ValueError, match="cannot reach"):
        # 200 permutations reach 0.005 but not 0.005 / 2.
        SkeletonSettings(estimator=("gaussian", "plugin"), alpha_max_stat=0.005).check()
    SkeletonSettings(estimator="gaussian", alpha_max_stat=0.005).check()


def _mixed(n=3000, seed=0):
    """
    x0 drives y linearly; x1 drives y through x1**2, which no linear or rank statistic sees.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2))
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + rng.normal(size=2)
        y[t] = 0.3 * y[t - 1] + 0.25 * x[t - 1, 0] + 0.5 * (x[t - 1, 1] ** 2 - 1) + rng.normal()
    return np.column_stack([x, y])


def test_ensemble_recovers_what_its_members_miss():
    from infoflow.preprocess import preprocess

    result = preprocess(_mixed(), max_lag=2, prng=0)
    base = SkeletonSettings(**FAST, synergy_search=False, tdmi_screen=False)

    def parents(estimator):
        settings = dataclasses.replace(base, estimator=estimator)
        return select_parents(result.data, 2, result.embeddings, None, settings, 0)

    gaussian = parents("gaussian")
    assert 1 not in gaussian.parents()
    both = parents(("gaussian", "plugin"))
    assert both.parents() == [0, 1]
    assert {both.admitted_by[v] for v in both.sources if v[0] == 1} == {"plugin"}
    assert set(both.admitted_by) == set(both.sources)


def test_ensemble_runs_every_stage():
    from infoflow.benchmarks import random_network, simulate_network
    from infoflow.preprocess import preprocess

    graph = random_network(6, prng=3)
    result = preprocess(simulate_network(graph, 3000, "var", prng=3), max_lag=3, prng=0)
    target = max(graph.nodes, key=graph.in_degree)
    settings = SkeletonSettings(**FAST, estimator=("gaussian", "coarse"), prescreen_alpha=0.2)
    sk = select_parents(result.data, target, result.embeddings, None, settings, 0)
    assert sk.parents() and set(sk.parents()) <= set(graph.predecessors(target))
    assert set(sk.admitted_by) == set(sk.sources)
    assert set(sk.admitted_by.values()) <= {"gaussian", "coarse"}
    assert set(sk.timing) == {"gaussian", "coarse"} and all(t > 0 for t in sk.timing.values())
    assert all(0.0 <= c["pvalue"] <= 1.0 for c in sk.tdmi_candidates.values())
