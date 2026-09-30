"""
Tests for edge measures, context merging, and cross-fitting.
"""

import dit
import numpy as np
import pytest
from dit.multivariate import intrinsic_total_correlation
from hypothesis import given, settings
from hypothesis import strategies as st

from infoflow.context import apply_merge, merge_contexts_exact, merge_contexts_statistical
from infoflow.crossfit import crossfit_flows
from infoflow.measures import cmi_from_joint, edge_flows, intrinsic_flow

joints = st.tuples(st.integers(2, 3), st.integers(2, 3), st.integers(1, 4), st.integers(0, 10**6))


def _random_joint(ky, kx, kw, seed):
    return np.random.default_rng(seed).dirichlet(np.full(ky * kx * kw, 0.5)).reshape(ky, kx, kw)


@pytest.mark.parametrize("seed", range(5))
def test_intrinsic_matches_dit(seed):
    p = _random_joint(2, 2, 3, seed)
    outcomes = [(a, b, c) for a in range(2) for b in range(2) for c in range(3)]
    d = dit.Distribution(outcomes, [p[o] for o in outcomes])
    assert intrinsic_flow(p, prng=0)[0] == pytest.approx(intrinsic_total_correlation(d, [[0], [1]], [2]), abs=1e-4)


@settings(max_examples=40, deadline=None)
@given(joints)
def test_bounds_and_identities(args):
    """
    0 <= intrinsic <= min(TE, TDMI), and the layers add up exactly.
    """
    p = _random_joint(*args)
    iif, Q = intrinsic_flow(p, prng=0)
    te, tdmi = cmi_from_joint(p), cmi_from_joint(p.sum(axis=2, keepdims=True))
    assert -1e-12 <= iif <= min(te, tdmi) + 1e-9
    assert np.allclose(Q.sum(axis=1), 1)


@settings(max_examples=30, deadline=None)
@given(joints, st.integers(1, 3))
def test_exact_merging_preserves_intrinsic(args, copies):
    """
    Duplicating context values with the same conditional law and merging them back changes nothing.
    """
    p = _random_joint(*args)
    scales = np.linspace(1.0, 0.3, copies)
    expanded = np.concatenate([p * s for s in scales], axis=2)
    expanded /= expanded.sum()
    mapping = merge_contexts_exact(expanded)
    assert mapping.max() + 1 <= p.shape[2]
    merged = apply_merge(expanded, mapping)
    merged_iif = intrinsic_flow(merged, restarts=20, prng=0)[0]
    assert merged_iif == pytest.approx(intrinsic_flow(expanded, restarts=20, prng=0)[0], abs=1e-4)
    assert cmi_from_joint(merged) == pytest.approx(cmi_from_joint(expanded), abs=1e-9)


def test_statistical_merging_recovers_groups():
    rng = np.random.default_rng(0)
    n = 6000
    w = rng.integers(0, 6, n)
    group = w % 2  # two true conditional laws
    x = rng.integers(0, 2, n)
    y = np.where(group == 0, x, rng.integers(0, 2, n))
    table = np.zeros((2, 2, 6))
    np.add.at(table, (y, x, w), 1)
    mapping = merge_contexts_statistical(table, alpha=0.01)
    assert mapping.max() + 1 == 2
    assert len(set(mapping[[0, 2, 4]])) == 1 and len(set(mapping[[1, 3, 5]])) == 1


def _samples(kind, n, rng):
    x = rng.integers(0, 2, n)
    w = rng.integers(0, 2, n)
    if kind == "copy":
        return np.where(rng.random(n) < 0.9, x, 1 - x), x, w
    if kind == "xor":
        return x ^ w, x, w
    z = rng.integers(0, 2, n)
    return z ^ (rng.random(n) < 0.1), z ^ (rng.random(n) < 0.1), z


@pytest.mark.parametrize(
    ("kind", "layer"),
    [("copy", "intrinsic"), ("xor", "synergistic"), ("shared", "shared")],
)
def test_crossfit_pure_layers(kind, layer):
    y, x, w = _samples(kind, 4000, np.random.default_rng(1))
    flow = crossfit_flows(y, x, w, prng=0).as_dict()
    others = [k for k in ("intrinsic", "synergistic", "shared") if k != layer]
    assert flow[layer] > 0.25
    assert all(flow[k] < 0.02 for k in others)
    assert flow["te"] == pytest.approx(flow["intrinsic"] + flow["synergistic"])
    assert flow["tdmi"] == pytest.approx(flow["intrinsic"] + flow["shared"])


def test_crossfit_reduces_bias_on_independent_data():
    """
    With independent variables and a large context, plug-in TE is inflated; cross-fitted TE is near zero.
    """
    rng = np.random.default_rng(2)
    n = 600
    y, x, w = rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 16, n)
    flow = crossfit_flows(y, x, w, prng=0)
    assert flow.raw["te"] > 0.01
    assert flow.te < flow.raw["te"] / 2


def test_edge_flows_single_joint():
    y, x, w = _samples("copy", 3000, np.random.default_rng(3))
    flow = edge_flows(y, x, w, prng=0)
    assert flow.intrinsic == pytest.approx(flow.raw["intrinsic"], abs=1e-9)
    assert flow.te == pytest.approx(flow.intrinsic + flow.synergistic)
    assert edge_flows(y, x, None, prng=0).synergistic == pytest.approx(0, abs=1e-9)


def test_intrinsic_flow_guards_oversized_channels():
    p = np.random.default_rng(0).random((2, 3, 40))
    te, tdmi = cmi_from_joint(p), cmi_from_joint(p.sum(axis=2, keepdims=True))
    with pytest.warns(UserWarning, match="upper bound"):
        reduced, channel = intrinsic_flow(p, max_parameters=200, prng=0)
    assert channel.shape == (40, 5) and reduced <= min(te, tdmi) + 1e-9
    with pytest.warns(UserWarning, match="min\\(TE, TDMI\\)"):
        skipped, _ = intrinsic_flow(p, max_parameters=50, prng=0)
    assert skipped == pytest.approx(min(te, tdmi))
