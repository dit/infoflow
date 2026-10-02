"""
Tests for the PID measure registry and multivariate hyperedges.
"""

import dit.pid
import numpy as np
import pytest
from dit.pid.distributions import bivariates, trivariates

from infoflow.data import DiscreteData
from infoflow.hyperedges import (
    PID_MEASURES,
    PIDMeasure,
    _decompose,
    hypergraph,
    multivariate_hyperedges,
    pid_measure,
)
from infoflow.selection import TargetSkeleton

USABLE = [m for m in PID_MEASURES.values() if m.cost in ("fast", "medium") and not m.requires and not m.fails_on_data]


@pytest.mark.parametrize("measure", USABLE, ids=lambda m: m.name)
def test_textbook_decompositions(measure):
    rdn = _decompose(measure, bivariates["redundant"], 2)
    assert rdn["redundancy"] == pytest.approx(1.0, abs=1e-3)
    if measure.name == "sx":
        return  # signed pointwise atoms: XOR is not purely synergistic under SxPID
    xor = _decompose(measure, bivariates["synergy"], 2)
    assert xor["synergy"] == pytest.approx(1.0, abs=1e-3)
    if measure.nonnegative:
        unq = _decompose(measure, bivariates["unique 1"], 2)
        assert unq["unique"][1] == pytest.approx(1.0, abs=1e-3)
        assert min(unq["unique"][0], unq["redundancy"], unq["synergy"]) > -1e-6
    if (measure.max_sources or 0) >= 3:
        assert _decompose(measure, trivariates["synergy"], 3)["synergy"] == pytest.approx(1.0, abs=1e-3)


def test_pid_measure_resolution():
    assert pid_measure("BROJA") is PID_MEASURES["broja"]
    assert pid_measure(dit.pid.PID_MMI) is PID_MEASURES["mmi"]
    assert pid_measure(PID_MEASURES["ccs"]).cls is dit.pid.PID_CCS

    class CustomPID(dit.pid.PID_MMI):
        pass

    custom = pid_measure(CustomPID)
    assert isinstance(custom, PIDMeasure) and custom.max_sources is None and custom.cost == "unknown"
    with pytest.raises(ValueError, match="unknown PID measure"):
        pid_measure("nope")
    with pytest.raises(ValueError, match="fails on sample"):
        pid_measure("proj")


def _data(columns):
    return DiscreteData.from_discrete([np.stack(columns, axis=1)])


def _skeleton(target, sources):
    return {target: TargetSkeleton(target=target, sources=sources, significant=True)}


def test_three_way_synergy_detected_pairs_not():
    rng = np.random.default_rng(0)
    n = 4000
    a, b, c = (rng.integers(0, 2, n) for _ in range(3))
    y = np.roll(a ^ b ^ c, 1)
    data = _data([a, b, c, y])
    h = multivariate_hyperedges(data, _skeleton(3, [(0, 1), (1, 1), (2, 1)]), prng=0)
    triple = h.where(h["order"] == 3, drop=True)
    assert triple.sizes["hyperedge"] == 1
    assert bool(triple["significant"].item()) and str(triple["consensus"].item()) == "synergistic"
    assert float(triple["synergy"].squeeze()) == pytest.approx(1.0, abs=0.02)
    assert float(triple["o_information"].item()) < 0
    pairs = h.where(h["order"] == 2, drop=True)
    assert pairs.sizes["hyperedge"] == 3 and not pairs["significant"].values.any()


def test_redundant_sources_classified():
    rng = np.random.default_rng(1)
    n = 4000
    a = rng.integers(0, 2, n)
    b = a ^ (rng.random(n) < 0.02)
    y = np.roll(a, 1)
    h = multivariate_hyperedges(_data([a, b, y]), _skeleton(2, [(0, 1), (1, 1)]), n_null=49, prng=0)
    assert str(h["consensus"].item()) == "redundant"
    assert not bool(h["significant"].item())
    assert float(h["o_information"].item()) > 0


def test_ensemble_agreement_and_hypergraph():
    rng = np.random.default_rng(2)
    n = 3000
    a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
    data = _data([a, b, np.roll(a ^ b, 1)])
    h = multivariate_hyperedges(
        data, _skeleton(2, [(0, 1), (1, 1)]), measure=["ccs", "broja", "mmi"], n_null=49, prng=0
    )
    assert list(h.measure.values) == ["ccs", "broja", "mmi"]
    assert float(h["agreement"].item()) == 1.0 and str(h["consensus"].item()) == "synergistic"
    g = hypergraph(h)
    hyper = [n for n, d in g.nodes(data=True) if d["kind"] == "hyperedge"]
    assert len(hyper) == 1 and set(g.predecessors(hyper[0])) == {"x0", "x1"}
    assert list(g.successors(hyper[0])) == ["x2"]
