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


# -- the automatic pipeline ------------------------------------------------------------

from datetime import timedelta  # noqa: E402

from infoflow import auto, datasets  # noqa: E402


def test_parse_budget():
    assert auto._parse_budget(None) is None
    assert auto._parse_budget(90) == 90.0
    assert auto._parse_budget("90s") == 90.0
    assert auto._parse_budget("30m") == 1800.0
    assert auto._parse_budget("2h") == 7200.0
    assert auto._parse_budget("1.5d") == 1.5 * 86400
    assert auto._parse_budget(timedelta(minutes=5)) == 300.0
    with pytest.raises(ValueError):
        auto._parse_budget("two hours")


PILOT: dict[str, Any] = {
    "member_seconds": {"plugin": 30.0, "gaussian": 20.0, "ksg": 1000.0},
    "usage": {"gaussian": 0.5},
    "prescreen": False,
    "parents_per_target": 2.0,
    "shared_per_target": 3.0,
    "edge_seconds": 5.0,
    "interpret_seconds": 1.0,
}
OPTIONS: dict[str, Any] = {
    "estimators": ("plugin", "gaussian", "ksg"),
    "prescreen": False,
    "include_shared_candidates": True,
    "interpret": True,
    "hyperedges": True,
}


def test_predict_counts_rounds_of_parallel_work():
    # 10 targets on 4 threads: 3 rounds of selection (Gaussian on half the targets);
    # 20 parent + 30 shared edges: 13 rounds.
    selection = (30.0 + 0.5 * 20.0 + 1000.0) * 3
    layers = 5.0 * 13
    expected = selection + layers + 1.0 * 10 + auto._HYPEREDGE_FACTOR * layers
    assert auto._predict(OPTIONS, PILOT, 10, 4) == pytest.approx(expected)
    # A pilot of 2 concurrent targets saturates the machine: 10 targets take 5 pilot times.
    saturated = dict(PILOT, concurrency=2)
    assert auto._predict(OPTIONS, saturated, 10, 4) == pytest.approx(expected + (5 - 3) * 1040.0)


def test_fit_budget_downgrades_in_order():
    options, downgrades, predicted = auto._fit_budget(OPTIONS, PILOT, 10, 4, 1.0, {})
    assert [d.split(":")[0] for d in downgrades] == [
        "dropped KSG selection",
        "dropped layer estimates for shared-only candidates",
        "dropped hyperedge decomposition",
        "dropped no source pre-screen (pre-screen turned on)",
    ]
    assert options["estimators"] == ("plugin", "gaussian")
    assert options["prescreen"] and not options["include_shared_candidates"] and not options["hyperedges"]
    assert predicted == pytest.approx(auto._predict(options, PILOT, 10, 4))
    # A generous budget changes nothing; pinned options are never dropped.
    assert auto._fit_budget(OPTIONS, PILOT, 10, 4, 1e9, {})[1] == []
    pinned, downgrades, _ = auto._fit_budget(OPTIONS, PILOT, 10, 4, 1.0, {"estimator": OPTIONS["estimators"]})
    assert pinned["estimators"] == OPTIONS["estimators"]
    assert not any("selection" in d for d in downgrades)
    # Stops as soon as the prediction fits.
    budget = auto._predict(dict(OPTIONS, estimators=("plugin", "gaussian")), PILOT, 10, 4)
    options, downgrades, _ = auto._fit_budget(OPTIONS, PILOT, 10, 4, budget, {})
    assert len(downgrades) == 1 and options["include_shared_candidates"]


def test_infer_discrete_chain():
    result = auto.infer(datasets.chain(2000, seed=1), preset="fast", prng=0)
    kind = result.network.dataset["kind"].values
    assert kind[0, 1] == "parent" and kind[1, 2] == "parent" and kind[0, 2] != "parent"
    report = result.report
    assert report.diagnostics["kind"] == "discrete"
    assert report.estimators == ("plugin",)
    assert set(report.admitted) == {(0, 1), (1, 2)}
    assert "preset: fast" in str(report) and np.isfinite(report.predicted_seconds)


def test_infer_options_are_checked_and_pinned():
    data = datasets.chain(1000, seed=2)
    with pytest.raises(ValueError):
        auto.infer(data, preset="extreme")
    with pytest.raises(TypeError):
        auto.infer(data, not_an_option=1)
    result = auto.infer(data, preset="fast", estimator="coarse", interpret=False, time_budget="1s", prng=0)
    assert result.report.estimators == ("coarse",)
    skeleton, infer_kwargs = auto._split_overrides({"estimator": "gaussian", "layer_estimator": "plugin"})
    assert skeleton == {"estimator": "gaussian"} and infer_kwargs == {"estimator": "plugin"}
    assert result.report.options["interpret"] is False


@pytest.mark.slow
def test_infer_reuses_pilot_skeletons_exactly():
    from dit.inference._symbols import as_generator

    import infoflow as inf
    from infoflow.benchmarks import random_network, simulate_network
    from infoflow.data import as_trials
    from infoflow.selection import SkeletonSettings

    graph = random_network(5, prng=4)
    x = simulate_network(graph, 1500, "var", prng=4)
    result = auto.infer(x, preset="fast", prng=3)
    pre, run = (int(s) for s in as_generator(3).integers(0, 2**32, size=2))
    discrete, embeddings, _, _ = auto._diagnose(as_trials(x), None, pre)
    settings = auto._skeleton_settings(
        result.report.estimators, False, result.report.options["threads"], result.report.options["device"], {}
    )
    assert isinstance(settings, SkeletonSettings)
    plain = inf.infer_multiplex(
        discrete,
        embeddings=embeddings,
        skeleton_settings=settings,
        include_shared_candidates=False,
        interpret=False,
        alpha=auto._LAYER_ALPHA,
        max_resamples=auto._MAX_RESAMPLES,
        map_fn=inf.thread_map(result.report.options["threads"]),
        prng=run,
    )
    for t in range(5):
        assert plain.skeleton[t].sources == result.network.skeleton[t].sources
        assert plain.skeleton[t].target_past == result.network.skeleton[t].target_past


def test_linearity_gate_flags_nonlinear_targets():
    from infoflow.benchmarks import random_network, simulate_network
    from infoflow.data import as_trials

    mute, *_ = auto._diagnose(as_trials(datasets.mute_network(3000, seed=0)), None, 0, 5)
    pvalues = auto.linearity_pvalues(mute)
    assert {t for t, p in pvalues.items() if p < auto.LINEARITY_ALPHA} == {1, 3}
    graph = random_network(8, prng=1)
    linear, *_ = auto._diagnose(as_trials(simulate_network(graph, 3000, "var", prng=1)), None, 0, 5)
    assert sum(p < auto.LINEARITY_ALPHA for p in auto.linearity_pvalues(linear).values()) <= 1
    logistic, *_ = auto._diagnose(as_trials(simulate_network(graph, 3000, "logistic", prng=1)), None, 0, 5)
    assert all(p < auto.LINEARITY_ALPHA for p in auto.linearity_pvalues(logistic).values())


def test_infer_gates_gaussian_per_target():
    result = auto.infer(datasets.mute_network(3000, seed=0), preset="fast", max_lag=5, prng=0)
    report = result.report
    assert report.diagnostics["nonlinear_targets"] == [1, 3]
    assert result.network.settings["skeleton_settings"].target_estimators == {1: ("plugin",), 3: ("plugin",)}
    assert all(name != "gaussian" for (s, t), name in report.admitted.items() if t in (1, 3))
