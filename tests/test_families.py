"""
Tests for explanation families and the lag-0 family layer.
"""

import itertools

import dit
import networkx as nx
import numpy as np
import pytest

from infoflow.contemporaneous import contemporaneous_families, contemporaneous_layer
from infoflow.data import DiscreteData
from infoflow.embedding import Embedding
from infoflow.families import (
    MAX_NODES,
    CIDecisions,
    ExplanationFamily,
    _dag_table,
    decisions_from_distribution,
    explanation_family,
    statements,
)
from infoflow.selection import SkeletonSettings, infer_skeleton

FAST = SkeletonSettings(
    n_perm_max_stat=50, n_perm_min_stat=50, n_perm_omnibus=100, n_perm_max_seq=100, n_perm_pairs=50, n_perm_tdmi=50
)

X, Y, Z, W = range(4)


def _b(p, bit):
    return p if bit else 1 - p


def _law(n, weight):
    """
    A binary dit distribution over `n` variables with probabilities proportional to `weight`.
    """
    outcomes = [o for o in itertools.product((0, 1), repeat=n) if weight(o) > 0]
    pmf = np.array([weight(o) for o in outcomes])
    return dit.Distribution(outcomes, list(pmf / pmf.sum()))


# The mechanisms of Hourican & Quax (2026), App. D; "flip" is an independent error added mod 2.
LAWS = {
    "xor": _law(3, lambda o: _b(0.1, o[2] ^ o[0] ^ o[1])),
    "xor + independent": _law(4, lambda o: _b(0.1, o[2] ^ o[0] ^ o[1]) * _b(1 / 3, o[3])),
    "biased xor": _law(3, lambda o: _b(1 / 3, o[0]) * _b(0.1, o[2] ^ o[0] ^ o[1])),
    "xor + descendant": _law(4, lambda o: _b(0.1, o[2] ^ o[0] ^ o[1]) * _b(0.1, o[3] ^ o[2])),
    "chain": _law(4, lambda o: _b(1 / 3, o[0]) * _b(0.1, o[1] ^ o[0]) * _b(0.1, o[2] ^ o[1]) * _b(0.1, o[3] ^ o[2])),
    "or": _law(4, lambda o: _b(0.1, o[2] ^ (o[0] | o[1])) * _b(1 / 3, o[3])),
    "asymmetric": _law(
        4, lambda o: _b(0.35, o[0]) * _b(0.6, o[1]) * _b((0.1, 0.9, 0.8, 0.2)[2 * o[0] + o[1]], o[2]) * _b(1 / 3, o[3])
    ),
    "independent baseline": _law(4, lambda o: _b(0.1, o[2] ^ o[0] ^ o[1]) * _b(0.1, o[3] ^ o[1])),
    "dependent baseline": _law(4, lambda o: _b(0.2, o[1] ^ o[0]) * _b(0.1, o[2] ^ o[0] ^ o[1]) * _b(0.1, o[3] ^ o[0])),
}

COLLIDERS = {frozenset({(X, Z), (Y, Z)}), frozenset({(X, Y), (Z, Y)}), frozenset({(Y, X), (Z, X)})}
XOR_DESCENDANT = {
    frozenset({(X, Y), (Z, Y), (Z, W)}),
    frozenset({(X, Y), (Z, Y), (W, Z)}),
    frozenset({(X, Z), (Y, Z), (Z, W)}),
    frozenset({(Y, X), (Z, X), (Z, W)}),
    frozenset({(Y, X), (Z, X), (W, Z)}),
}
CHAIN = {
    frozenset({(X, Y), (Y, Z), (Z, W)}),
    frozenset({(Y, X), (Y, Z), (Z, W)}),
    frozenset({(Y, X), (Z, Y), (Z, W)}),
    frozenset({(Y, X), (Z, Y), (W, Z)}),
}
COLLIDER_AT_Z = {frozenset({(X, Z), (Y, Z)})}


# -- the DAG table -------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "count"), [(1, 1), (2, 3), (3, 25), (4, 543), (5, 29281)])
def test_dag_counts(n, count):
    table = _dag_table(n)
    assert len(table.arcs) == count
    assert table.separated.shape == (count, len(statements(n)))


def test_dag_table_size_limit():
    with pytest.raises(ValueError, match="1 to 5"):
        _dag_table(MAX_NODES + 1)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_separation_matches_networkx(n):
    d_separated = getattr(nx, "is_d_separator", None) or nx.d_separated
    table = _dag_table(n)
    rng = np.random.default_rng(n)
    rows = range(len(table.arcs)) if n < 5 else rng.choice(len(table.arcs), 150, replace=False)
    for d in rows:
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        g.add_edges_from((int(u), int(v)) for u, v in zip(*np.nonzero(table.arcs[d]), strict=True))
        for k, (i, j, S) in enumerate(statements(n)):
            assert table.separated[d, k] == d_separated(g, {i}, {j}, set(S))


def test_without_removes_one_arc():
    n = 4
    table = _dag_table(n)
    for d in range(len(table.arcs)):
        for u, v in zip(*np.nonzero(table.arcs[d]), strict=True):
            fewer = table.arcs[d].copy()
            fewer[u, v] = False
            assert (table.arcs[table.without[d, u * n + v]] == fewer).all()
    assert (table.without[~table.arcs.reshape(len(table.arcs), -1)] == -1).all()


# -- exact families ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "tiers", "convention", "expected"),
    [
        ("xor", None, "neither", COLLIDERS),
        ("xor + independent", None, "neither", COLLIDERS),
        ("biased xor", None, "neither", set()),
        ("biased xor", None, "not_both", 4),
        ("xor + descendant", None, "neither", XOR_DESCENDANT),
        ("xor + descendant", (0, 0, 1, 2), "neither", {frozenset({(X, Z), (Y, Z), (Z, W)})}),
        ("chain", None, "neither", CHAIN),
        ("or", None, "neither", COLLIDER_AT_Z),
        ("or", None, "not_both", 5),
        ("asymmetric", None, "neither", COLLIDER_AT_Z),
        ("asymmetric", None, "not_both", 5),
        ("independent baseline", None, "neither", 5),
        ("independent baseline", (0, 0, 1, 1), "neither", {frozenset({(X, Z), (Y, Z), (Y, W)})}),
        ("dependent baseline", None, "neither", set()),
        ("dependent baseline", None, "not_both", 6),
        ("dependent baseline", (0, 0, 1, 1), "not_both", 2),
    ],
)
def test_population_families(name, tiers, convention, expected):
    """
    Every reference family of Hourican & Quax (2026), App. A and Sec. 3, under both conventions.
    """
    family = explanation_family(decisions_from_distribution(LAWS[name]), convention=convention, tiers=tiers)
    if isinstance(expected, int):
        assert len(family) == expected
    else:
        assert set(family.candidates) == expected
    assert family.status == ("complete" if len(family) else "empty")


def _minimal_imap(decisions, order):
    """
    The minimal I-map for an order: each earlier node dependent on x given the rest of x's predecessors.
    """
    arcs = set()
    for k, x in enumerate(order):
        before = set(order[:k])
        arcs |= {(u, x) for u in before if not decisions.is_independent(x, u, before - {u})}
    return frozenset(arcs)


@pytest.mark.parametrize("name", ["xor + independent", "xor + descendant", "chain", "or", "asymmetric"])
def test_every_explanation_is_a_minimal_imap_of_an_order(name):
    """
    For positive laws every SGS-minimal DAG is the minimal I-map of one of its orders.
    """
    decisions = decisions_from_distribution(LAWS[name])
    imaps = {_minimal_imap(decisions, order) for order in itertools.permutations(range(4))}
    family = explanation_family(decisions)
    assert family.candidates and set(family.candidates) <= imaps


def test_family_summaries():
    xor = explanation_family(decisions_from_distribution(LAWS["xor"]))
    assert xor.compelled_arrows == frozenset() and xor.compelled_adjacencies == frozenset()
    assert xor.possible_adjacencies == {frozenset(p) for p in itertools.combinations(range(3), 2)}
    assert (
        xor.describe(names="XYZ") == "{X, Y, Z}: 3 explanations: {X -> Y, Z -> Y} | {X -> Z, Y -> Z} | {Y -> X, Z -> X}"
    )
    assert [g.number_of_edges() for g in xor.graphs()] == [2, 2, 2]
    chain = explanation_family(decisions_from_distribution(LAWS["chain"]))
    assert chain.compelled_adjacencies == {frozenset({X, Y}), frozenset({Y, Z}), frozenset({Z, W})}
    assert chain.compelled_arrows == frozenset()
    gate = explanation_family(decisions_from_distribution(LAWS["or"]))
    assert gate.compelled_arrows == {(X, Z), (Y, Z)}
    assert gate.describe() == "{0, 1, 2, 3}: 1 explanation: {0 -> 2, 1 -> 2}"
    assert ExplanationFamily((0, 1, 2), status="too_large").describe() == "{0, 1, 2}: too_large"


def test_background_knowledge_and_errors():
    decisions = decisions_from_distribution(LAWS["xor"])
    assert set(explanation_family(decisions, forbidden=[(X, Z)]).candidates) == COLLIDERS - COLLIDER_AT_Z
    assert set(explanation_family(decisions, tiers={X: 0, Y: 0, Z: 1}).candidates) == COLLIDER_AT_Z
    assert len(explanation_family(decisions, tiers={Z: 1})) == 3  # nodes without a tier are unconstrained
    with pytest.raises(ValueError):
        explanation_family(decisions, tiers=(0, 1))
    with pytest.raises(ValueError, match="convention"):
        explanation_family(decisions, convention="both")
    with pytest.raises(ValueError, match="no decision"):
        CIDecisions((0, 1, 2), {})
    six = CIDecisions(tuple(range(6)), dict.fromkeys(statements(6), True))
    with pytest.raises(ValueError, match="at most 5"):
        explanation_family(six)
    assert decisions.is_independent(Z, X) == decisions.is_independent(X, Z) is True
    assert decisions.is_independent(X, Z, [Y]) is False


# -- lag-0 families from data --------------------------------------------------------------


def _flip(rng, n, p):
    return (rng.random(n) < p).astype(int)


def _lag0(kind, n, seed):
    """
    Static samples of the motifs, so every interaction is within one sample.
    """
    rng = np.random.default_rng(seed)
    x, y = rng.integers(0, 2, n), rng.integers(0, 2, n)
    if kind == "xor":
        columns = [x, y, x ^ y ^ _flip(rng, n, 0.1)]
    elif kind == "xor + descendant":
        z = x ^ y ^ _flip(rng, n, 0.1)
        columns = [x, y, z, z ^ _flip(rng, n, 0.1)]
    elif kind == "or":
        columns = [x, y, (x | y) ^ _flip(rng, n, 0.1)]
    else:
        a = (rng.random(n) < 1 / 3).astype(int)
        b = a ^ _flip(rng, n, 0.1)
        c = b ^ _flip(rng, n, 0.1)
        columns = [a, b, c, c ^ _flip(rng, n, 0.1)]
    return DiscreteData.from_discrete(np.stack(columns, axis=1))


def _fit(kind, n=4000, seed=0, **kwargs):
    data = _lag0(kind, n, seed)
    skeleton = infer_skeleton(data, Embedding(max_lag=1), settings=FAST, prng=seed)
    return data, skeleton, contemporaneous_families(data, skeleton, prng=seed, **kwargs)


def _graph(n, undirected=(), directed=()):
    g = np.zeros((n, n), dtype=int)
    for u, v in undirected:
        g[u, v] = g[v, u] = 1
    for u, v in directed:
        g[u, v] = 2
    return g


@pytest.mark.parametrize(
    ("kind", "expected", "graph"),
    [
        ("xor", COLLIDERS, _graph(3)),
        ("xor + descendant", XOR_DESCENDANT, _graph(4, undirected=[(Z, W)])),
        ("or", COLLIDER_AT_Z, _graph(3, directed=[(X, Z), (Y, Z)])),
        ("chain", CHAIN, _graph(4, undirected=[(X, Y), (Y, Z), (Z, W)])),
    ],
)
def test_lag0_families_recover_the_population_family(kind, expected, graph):
    """
    The graph marks only what every explanation shares; ``possible`` marks what any has.
    """
    _, _, result = _fit(kind)
    (family,) = result.families
    assert family.status == "complete" and set(family.candidates) == expected
    assert family.decisions is not None
    assert family.decisions.level == pytest.approx(0.05 / len(statements(len(family.nodes))))
    assert (result.graph == graph).all()
    assert result.possible.sum() == 2 * len(family.possible_adjacencies)


def test_pc_layer_misses_the_lag0_xor():
    data, skeleton, result = _fit("xor")
    assert not contemporaneous_layer(data, skeleton, n_perm=99, prng=0)[0].any()
    assert result.possible.sum() == 6


def test_lag0_families_options():
    _, _, timed = _fit("xor + descendant", tiers=(0, 0, 1, 2))
    assert timed.families[0].candidates == (frozenset({(X, Z), (Y, Z), (Z, W)}),)
    assert timed.graph[X, Z] == timed.graph[Y, Z] == timed.graph[Z, W] == 2
    _, _, asymptotic = _fit("xor", test="asymptotic")
    assert set(asymptotic.families[0].candidates) == COLLIDERS
    _, _, forbidden = _fit("xor", forbidden=[(X, Z)])
    assert set(forbidden.families[0].candidates) == COLLIDERS - COLLIDER_AT_Z


def test_lag0_families_fall_back_without_enumeration():
    data, skeleton, large = _fit("xor", max_family_size=2)
    assert [f.status for f in large.families] == ["too_large"]
    pc = contemporaneous_layer(data, skeleton, n_perm=200, prng=0)[0]
    assert (large.graph == pc).all()
    _, _, sparse = _fit("xor", min_cell_count=10**6)
    assert [f.status for f in sparse.families] == ["unavailable"]
    assert not sparse.possible.any()


def test_lag0_families_arguments():
    data = _lag0("xor", 500, 0)
    skeleton = infer_skeleton(data, Embedding(max_lag=1), settings=FAST, prng=0)
    for kwargs in ({"max_family_size": 6}, {"test": "exact"}, {"convention": "both"}, {"n_perm": 10}):
        with pytest.raises(ValueError):
            contemporaneous_families(data, skeleton, prng=0, **kwargs)
    with pytest.raises(ValueError, match="cannot reach"):
        contemporaneous_families(data, skeleton, prng=0, n_perm=50)  # the 6 statements need 119


def test_infer_multiplex_lag0_families(tmp_path):
    from infoflow.network import MultiplexNetwork, infer_multiplex

    net = infer_multiplex(
        _xor_columns(3000),
        names=["X", "Y", "Z"],
        max_lag=1,
        skeleton_settings=FAST,
        contemporaneous="family",
        prng=0,
    )
    (family,) = net.contemporaneous_families
    assert set(family.candidates) == COLLIDERS
    assert "contemporaneous_possible" in net.dataset and not net.dataset["contemporaneous"].values.any()
    assert "{X, Y, Z}: 3 explanations" in net.summary()
    assert net.settings["contemporaneous"] == "family"
    net.save(tmp_path / "net.pkl")
    assert MultiplexNetwork.load(tmp_path / "net.pkl").contemporaneous_families == net.contemporaneous_families
    with pytest.raises(ValueError, match="contemporaneous"):
        infer_multiplex(_xor_columns(500), max_lag=1, skeleton_settings=FAST, contemporaneous="lag0", prng=0)


def _xor_columns(n, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.integers(0, 2, n), rng.integers(0, 2, n)
    return np.stack([x, y, x ^ y ^ _flip(rng, n, 0.1)], axis=1)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["xor", "xor + descendant", "or", "chain"])
def test_lag0_families_are_reliable(kind):
    """
    With Bonferroni-corrected decisions the family equals the population family nearly always.
    """
    expected = {"xor": COLLIDERS, "xor + descendant": XOR_DESCENDANT, "or": COLLIDER_AT_Z, "chain": CHAIN}[kind]
    hits = 0
    for seed in range(20):
        _, _, result = _fit(kind, n=2000, seed=100 + seed)
        if result.families and set(result.families[0].candidates) == expected:
            hits += 1
    assert hits >= 18
