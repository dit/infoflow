"""
Explanation families: every DAG that a table of conditional-independence decisions allows.

Faithfulness lets a structure learner read adjacencies from dependences, so it deletes a link
whose dependence shows only jointly. In noisy XOR, :math:`Z = X \\oplus Y \\oplus E`, every
pair of variables is independent although :math:`X` and :math:`Y` together determine
:math:`Z`. Weaker adjacency faithfulness :cite:`Marx2021weaker` lets an adjacency be supported
by a triple association instead, and :cite:`Hourican2026` take as the target of structure
learning the *family* of DAGs that are Markov to the distribution, satisfy that condition, are
SGS-minimal :cite:`Spirtes2000`, and agree with background knowledge. Without time order XOR
has three explanations, a collider at each variable. No single skeleton represents them, and
their union, a triangle, is none of them.

:func:`explanation_family` evaluates those predicates over every DAG on up to
:data:`MAX_NODES` nodes, for decisions from tests on data
(:func:`infoflow.contemporaneous.contemporaneous_families`) or from an exact distribution
(:func:`decisions_from_distribution`).
"""

from dataclasses import dataclass
from functools import cache
from itertools import combinations, permutations, product

import numpy as np

__all__ = (
    "MAX_NODES",
    "CIDecisions",
    "ExplanationFamily",
    "decisions_from_distribution",
    "explanation_family",
    "statements",
)

#: The most nodes :func:`explanation_family` enumerates DAGs on (29,281 DAGs).
MAX_NODES = 5


def _subsets(items):
    items = sorted(items)
    for k in range(len(items) + 1):
        yield from (frozenset(c) for c in combinations(items, k))


@cache
def statements(n):
    """
    Every pairwise conditional-independence statement on `n` nodes.

    Returns
    -------
    tuple
        ``(i, j, S)`` for positions ``i < j`` and every frozenset ``S`` of the other positions.
    """
    return tuple((i, j, S) for i, j in combinations(range(n), 2) for S in _subsets(set(range(n)) - {i, j}))


@dataclass(frozen=True)
class CIDecisions:
    """
    A decision for every pairwise conditional-independence statement on a set of nodes.

    Attributes
    ----------
    nodes : tuple
        Node labels. Statements refer to nodes by position.
    independent : dict
        ``(i, j, S) -> bool`` for every statement of :func:`statements`: True when
        :math:`X_i` and :math:`X_j` are judged independent given :math:`X_S`.
    pvalues : dict, None
        The p-value behind each decision, when the decisions come from tests.
    level : float, None
        The level each p-value was compared with.
    """

    nodes: tuple
    independent: dict
    pvalues: dict | None = None
    level: float | None = None

    def __post_init__(self):
        missing = [s for s in statements(len(self.nodes)) if s not in self.independent]
        if missing:
            raise ValueError(f"no decision for {len(missing)} statements, e.g. {missing[0]}")

    def is_independent(self, i, j, given=()):
        """
        The decision for :math:`X_i \\perp X_j \\mid X_{given}` (positions).
        """
        return bool(self.independent[min(i, j), max(i, j), frozenset(given)])


def decisions_from_distribution(dist, tol=1e-10):
    """
    Exact decisions for a dit distribution.

    Parameters
    ----------
    dist : dit.Distribution
    tol : float
        A statement is independent when its conditional mutual information is at most
        `tol` bits.

    Returns
    -------
    CIDecisions
        Nodes are the random variables' names, or their indices.
    """
    from dit.multivariate import coinformation

    n = dist.outcome_length()
    names = dist.get_rv_names()
    independent = {(i, j, S): coinformation(dist, [[i], [j]], sorted(S)) <= tol for i, j, S in statements(n)}
    return CIDecisions(tuple(names) if names else tuple(range(n)), independent)


# -- every DAG on a few nodes ----------------------------------------------------------


@dataclass(frozen=True)
class _DagTable:
    """
    Every DAG on `n` nodes, with what each implies.

    ``arcs[d, u, v]`` is the arc ``u -> v`` of DAG ``d``; ``without[d, u * n + v]`` is the index
    of DAG ``d`` with that arc removed (-1 if absent); ``separated[d, k]`` is whether DAG ``d``
    implies statement ``k`` of :func:`statements` by d-separation.
    """

    n: int
    arcs: np.ndarray
    without: np.ndarray
    separated: np.ndarray


def _closure(arcs):
    """
    ``reach[d, u, v]``: a directed path of length at least one from ``u`` to ``v``.
    """
    reach = arcs.copy()
    for _ in range(arcs.shape[1] - 1):
        longer = reach | ((reach.astype(np.uint8) @ arcs.astype(np.uint8)) > 0)
        if (longer == reach).all():
            break
        reach = longer
    return reach


@cache
def _dag_table(n):
    if not 1 <= n <= MAX_NODES:
        raise ValueError(f"DAGs are enumerated on 1 to {MAX_NODES} nodes, not {n}")
    pairs = list(combinations(range(n), 2))
    states = np.array(list(product(range(3), repeat=len(pairs))), dtype=np.int8)
    arcs = np.zeros((len(states), n, n), dtype=bool)
    for k, (i, j) in enumerate(pairs):
        arcs[:, i, j] = states[:, k] == 1
        arcs[:, j, i] = states[:, k] == 2
    reach = _closure(arcs)
    acyclic = ~reach[:, np.arange(n), np.arange(n)].any(axis=1)
    arcs, reach = arcs[acyclic], reach[acyclic]
    D = len(arcs)

    # Removing an arc leaves a DAG: find it by its arc bitmask.
    flat = arcs.reshape(D, n * n)
    bits = np.left_shift(np.int64(1), np.arange(n * n, dtype=np.int64))
    masks = flat.astype(np.int64) @ bits
    order = np.argsort(masks)
    position = np.clip(np.searchsorted(masks[order], masks[:, None] ^ bits[None, :]), 0, D - 1)
    without = np.where(flat, order[position], -1)

    # d-separation: X_i and X_j are separated by X_S iff no path joins them in the moral
    # graph of the ancestral set of {i, j} | S once S is removed.
    ancestors = reach.transpose(0, 2, 1) | np.eye(n, dtype=bool)  # [d, v, u]: u is v or an ancestor of v
    linked = arcs | arcs.transpose(0, 2, 1)
    coparents = arcs[:, :, None, :] & arcs[:, None, :, :]  # [d, u, v, c]: u -> c <- v
    queries = statements(n)
    separated = np.zeros((D, len(queries)), dtype=bool)
    for k, (i, j, S) in enumerate(queries):
        keep = ancestors[:, [i, j, *S], :].any(axis=1)
        moral = (linked | (coparents & keep[:, None, None, :]).any(axis=3)) & keep[:, :, None] & keep[:, None, :]
        allowed = keep.copy()
        allowed[:, sorted(S)] = False
        reached = np.zeros((D, n), dtype=bool)
        reached[:, i] = True
        for _ in range(n - 1):
            reached |= (reached[:, :, None] & moral).any(axis=1) & allowed
        separated[:, k] = ~reached[:, j]
    return _DagTable(n, arcs, without, separated)


# -- the predicates ----------------------------------------------------------------------


def _associations(decisions, convention):
    """
    ``a1[x, u]``: dependent given every conditioning set. ``s2[x, u, v]``: a strict
    two-association of ``x`` with ``u`` and ``v`` :cite:`Marx2021weaker,Hourican2026`.
    """
    n = len(decisions.nodes)
    every = set(range(n))

    def dependent(i, j, given):
        return not decisions.is_independent(i, j, given)

    a1 = np.zeros((n, n), dtype=bool)
    for x, u in permutations(range(n), 2):
        a1[x, u] = all(dependent(x, u, T) for T in _subsets(every - {x, u}))
    s2 = np.zeros((n, n, n), dtype=bool)
    for x, u, v in permutations(range(n), 3):
        a2 = all(
            dependent(x, u, T | {v}) and dependent(x, v, T | {u}) and dependent(u, v, T | {x})
            for T in _subsets(every - {x, u, v})
        )
        if convention == "neither":
            s2[x, u, v] = a2 and not a1[x, u] and not a1[x, v]
        else:
            s2[x, u, v] = a2 and not (a1[x, u] and a1[x, v])
    return a1, s2


def _weakly_faithful(arcs, a1, s2):
    """
    Every edge is supported at both endpoints: by a one-association, or by a strict
    two-association with a third member of that endpoint's Markov blanket.
    """
    n = len(arcs)
    parents = [set(np.flatnonzero(arcs[:, v]).tolist()) for v in range(n)]
    children = [set(np.flatnonzero(arcs[v]).tolist()) for v in range(n)]
    blanket = [(parents[x] | children[x] | set().union(*(parents[c] for c in children[x]))) - {x} for x in range(n)]
    for u, v in zip(*np.nonzero(arcs), strict=True):
        for x, w in ((int(u), int(v)), (int(v), int(u))):
            if not (a1[x, w] or any(s2[x, w, z] for z in blanket[x] - {w})):
                return False
    return True


def _permitted(table, nodes, tiers, forbidden):
    """
    DAGs with no arc from a later tier to an earlier one and no forbidden arc.
    """
    n = table.n
    ok = np.ones((n, n), dtype=bool)
    if tiers is not None:
        tier = dict(tiers) if isinstance(tiers, dict) else dict(zip(nodes, tiers, strict=True))
        for u, v in permutations(range(n), 2):
            tu, tv = tier.get(nodes[u]), tier.get(nodes[v])
            if tu is not None and tv is not None and tu > tv:
                ok[u, v] = False
    position = {node: k for k, node in enumerate(nodes)}
    for u, v in forbidden:
        if u in position and v in position:
            ok[position[u], position[v]] = False
    return ~(table.arcs & ~ok).any(axis=(1, 2))


def _skeleton(arcs):
    return frozenset(frozenset(arc) for arc in arcs)


@dataclass(frozen=True)
class ExplanationFamily:
    """
    The DAGs that a set of decisions allows on one set of nodes.

    Attributes
    ----------
    nodes : tuple
        Node labels.
    candidates : tuple of frozenset
        Each explanation as a frozenset of arcs ``(source, target)``.
    status : str
        ``'complete'``: every DAG on the nodes was checked, and these are all that qualify.
        ``'empty'``: every DAG was checked and none qualifies, because the assumptions fail for
        these variables or the decisions contradict each other.
        ``'too_large'`` or ``'unavailable'``: nothing was enumerated (too many nodes, or a
        statement that could not be tested reliably), so no structural claim is made.
    convention : str
        The strictness of the two-association: ``'neither'`` or ``'not_both'``.
    decisions : CIDecisions, None
    """

    nodes: tuple
    candidates: tuple = ()
    status: str = "complete"
    convention: str = "neither"
    decisions: CIDecisions | None = None

    def __len__(self):
        return len(self.candidates)

    @property
    def compelled_arrows(self):
        """
        Arcs in every explanation.
        """
        return frozenset.intersection(*self.candidates) if self.candidates else frozenset()

    @property
    def compelled_adjacencies(self):
        """
        Pairs (frozensets) linked in every explanation.
        """
        return frozenset.intersection(*map(_skeleton, self.candidates)) if self.candidates else frozenset()

    @property
    def possible_adjacencies(self):
        """
        Pairs linked in at least one explanation. Their union is not itself an explanation.
        """
        return frozenset().union(*map(_skeleton, self.candidates))

    def graphs(self):
        """
        Each explanation as a networkx ``DiGraph``.
        """
        import networkx as nx

        out = []
        for arcs in self.candidates:
            g = nx.DiGraph()
            g.add_nodes_from(self.nodes)
            g.add_edges_from(sorted(arcs))
            out.append(g)
        return out

    def describe(self, names=None):
        """
        One line: the nodes and their explanations, with `names[node]` as labels if given.
        """

        def label(v):
            return str(names[v]) if names is not None else str(v)

        head = "{" + ", ".join(label(v) for v in self.nodes) + "}"
        if self.status != "complete":
            return f"{head}: {self.status}"
        body = " | ".join(
            "{" + ", ".join(f"{label(u)} -> {label(v)}" for u, v in sorted(arcs)) + "}" for arcs in self.candidates
        )
        return f"{head}: {len(self)} explanation{'' if len(self) == 1 else 's'}: {body}"


def explanation_family(decisions, *, convention="neither", tiers=None, forbidden=()):
    """
    Every DAG on the decisions' nodes that weaker adjacency faithfulness allows.

    A DAG qualifies when it is :cite:`Marx2021weaker,Hourican2026`

    * **Markov:** every statement it implies by d-separation was decided independent;
    * **weakly adjacency faithful:** at each endpoint ``x`` of each edge ``x - w``, either
      ``x`` and ``w`` are dependent given every conditioning set, or they form a strict
      two-association with a third member ``z`` of ``x``'s Markov blanket: ``x, w``, ``x, z``
      and ``w, z`` are each dependent given the third and any further conditioning set, while
      (``convention='neither'``) neither ``x, w`` nor ``x, z`` is dependent given every set,
      or (``'not_both'``) not both are;
    * **SGS-minimal** :cite:`Spirtes2000`: no DAG with one arc fewer is Markov;
    * **consistent with background knowledge:** no arc from a later tier to an earlier one,
      and no forbidden arc.

    Parameters
    ----------
    decisions : CIDecisions
        At most :data:`MAX_NODES` nodes.
    convention : {'neither', 'not_both'}
    tiers : dict or sequence, None
        A tier per node label (a sequence follows ``decisions.nodes``); arcs may not point to
        an earlier tier. Nodes without a tier are unconstrained.
    forbidden : iterable of (source, target)
        Arcs (node labels) that no explanation may contain.

    Returns
    -------
    ExplanationFamily
    """
    if convention not in ("neither", "not_both"):
        raise ValueError("convention must be 'neither' or 'not_both'")
    nodes = tuple(decisions.nodes)
    n = len(nodes)
    if n > MAX_NODES:
        raise ValueError(f"explanation families are enumerated on at most {MAX_NODES} nodes, not {n}")
    table = _dag_table(n)
    dependent = np.array([not decisions.independent[s] for s in statements(n)], dtype=bool)
    markov = ~(table.separated & dependent).any(axis=1)
    reducible = np.where(table.without >= 0, markov[table.without], False).any(axis=1)
    keep = markov & ~reducible & _permitted(table, nodes, tiers, forbidden)
    a1, s2 = _associations(decisions, convention)
    candidates = [
        frozenset((nodes[u], nodes[v]) for u, v in zip(*np.nonzero(table.arcs[d]), strict=True))
        for d in np.flatnonzero(keep)
        if _weakly_faithful(table.arcs[d], a1, s2)
    ]
    return ExplanationFamily(
        nodes=nodes,
        candidates=tuple(sorted(candidates, key=sorted)),
        status="complete" if candidates else "empty",
        convention=convention,
        decisions=decisions,
    )
