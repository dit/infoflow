"""
An optional lag-0 layer, and an adapter to Tigramite.

With coarse sampling (or after Faes compensation) processes can depend on each
other within one sample, and time order no longer orients those links. The lag-0
layer tests each pair's contemporaneous dependence given both processes' selected
pasts and lagged parents (a momentary conditional independence test in the spirit
of PCMCI+ :cite:`Runge2020`), removes links explained by a third contemporaneous
neighbour (PC-style), and orients unshielded triples ``i - k - j`` with ``k``
outside the separating set as colliders ``i -> k <- j``. That reads adjacencies from
dependence, so it cannot see links whose dependence shows only jointly (a lag-0 XOR);
:func:`contemporaneous_families` reports every explanation that weaker adjacency
faithfulness allows instead :cite:`Marx2021weaker,Hourican2026`. For full
latent-variable output (partial ancestral graphs) use :func:`run_tigramite` with LPCMCI.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from dit.inference._symbols import as_generator

from .families import MAX_NODES, CIDecisions, ExplanationFamily, explanation_family, statements
from .selection import _cmi, _Columns, _null_statistics, _pvalue, check_n_perm

__all__ = (
    "ContemporaneousFamilies",
    "contemporaneous_families",
    "contemporaneous_layer",
    "instantaneous_exchange",
    "run_tigramite",
    "to_tigramite",
)


def _conditioning(skeleton, i, j):
    """
    Lagged conditioning set for the pair (i, j): both targets' pasts and lagged parents.
    """
    cond = []
    for p in (i, j):
        sk = skeleton[p]
        cond += list(sk.target_past) + list(sk.sources if sk.significant else [])
    return list(dict.fromkeys(cond))


def _test(data, i, j, cond, n_perm, rng):
    variables = list(dict.fromkeys([(i, 0), *cond]))
    max_lag = max((v[1] for v in variables), default=0)
    cols = _Columns(data, j, variables, max_lag)
    x, Kx = cols.column((i, 0))
    z, Kz = cols.joint([v for v in variables if v != (i, 0)])
    value = _cmi(cols.y, cols.Ky, x, Kx, z, Kz)
    null = np.array([_cmi(cols.y, cols.Ky, x[rng.permutation(cols.n)], Kx, z, Kz) for _ in range(n_perm)])
    return value, float((1 + np.sum(null >= value - 1e-12)) / (1 + n_perm))


def instantaneous_exchange(data, skeleton, n_perm=200, prng=None):
    """
    Instantaneous information exchange of every pair, with a permutation p-value.

    :math:`I[X_t : Y_t \\mid \\text{both selected pasts and lagged parents}]`, the
    causally conditioned instantaneous term of directed information
    :cite:`Amblard2013`: in the bivariate case directed information is transfer
    entropy plus this exchange. It is symmetric in the pair.

    Returns
    -------
    exchange : np.ndarray
        (processes, processes) in bits, symmetric, zero on the diagonal.
    pvalues : np.ndarray
        Permutation p-values, symmetric, one on the diagonal.
    """
    rng = as_generator(prng)
    P = data.n_processes
    values = np.zeros((P, P))
    pvalues = np.ones((P, P))
    for i, j in combinations(range(P), 2):
        v, p = _test(data, i, j, _conditioning(skeleton, i, j), n_perm, rng)
        values[i, j] = values[j, i] = v
        pvalues[i, j] = pvalues[j, i] = p
    return values, pvalues


def contemporaneous_layer(data, skeleton, alpha=0.05, n_perm=200, prng=None, exchange=None):
    """
    Contemporaneous links and their PC-style orientation.

    Parameters
    ----------
    data : DiscreteData
    skeleton : dict
        ``target -> TargetSkeleton``.
    alpha : float
    n_perm : int
    prng : None, int, Generator
    exchange : tuple, None
        The ``(exchange, pvalues)`` pair from :func:`instantaneous_exchange`, to reuse.

    Returns
    -------
    graph : np.ndarray
        (i, j): 0 no link, 1 undirected link, 2 oriented ``i -> j``.
    values : np.ndarray
        The contemporaneous CMI (bits) given the lagged conditioning set.
    pvalues : np.ndarray
    """
    rng = as_generator(prng)
    P = data.n_processes
    values, pvalues = exchange if exchange is not None else instantaneous_exchange(data, skeleton, n_perm, rng)
    adjacent = (pvalues <= alpha) & ~np.eye(P, dtype=bool)
    sepset = {}
    for i, j in combinations(range(P), 2):
        if not adjacent[i, j]:
            continue
        for k in range(P):
            if k in (i, j) or not (adjacent[i, k] or adjacent[j, k]):
                continue
            _, p = _test(data, i, j, _conditioning(skeleton, i, j) + [(k, 0)], n_perm, rng)
            if p > alpha:
                adjacent[i, j] = adjacent[j, i] = False
                sepset[i, j] = sepset[j, i] = {k}
                break
    graph = adjacent.astype(np.int64)
    for k in range(P):
        neighbours = [i for i in range(P) if adjacent[i, k]]
        for i, j in combinations(neighbours, 2):
            if not adjacent[i, j] and k not in sepset.get((i, j), set()):
                graph[i, k], graph[k, i] = 2, 0
                graph[j, k], graph[k, j] = 2, 0
    return graph, values, pvalues


# -- explanation families -----------------------------------------------------------------


def _levels(z, x):
    """
    The number of distinct values of `x` in each stratum of `z`, in the order of ``np.unique(z)``.
    """
    base = int(x.max()) + 1
    pairs = np.unique(z.astype(np.int64) * base + x)
    return np.unique(pairs // base, return_counts=True)[1]


def _lag0_test(data, i, j, given, cond, n_perm, rng, alpha=None, batch=None, test="permutation", min_cell_count=None):
    """
    :math:`I[X^i_t : X^j_t \\mid X^{given}_t, cond]` in bits, its p-value, and whether the
    table has at least `min_cell_count` samples per cell on average.

    ``test='permutation'`` permutes :math:`X^i_t` freely (curtailed at `alpha` with `batch`);
    ``'asymptotic'`` compares :math:`G^2 = 2 N \\ln 2 \\, \\hat I` with a chi-squared law
    whose degrees of freedom count the observed levels in each stratum :cite:`Agresti2013`.
    """
    variables = list(dict.fromkeys([(i, 0), *[(k, 0) for k in given], *cond]))
    max_lag = max((v[1] for v in variables), default=0)
    cols = _Columns(data, j, variables, max_lag)
    x, Kx = cols.column((i, 0))
    z, Kz = cols.joint([v for v in variables if v != (i, 0)])
    value = _cmi(cols.y, cols.Ky, x, Kx, z, Kz)
    available = True
    if min_cell_count is not None:
        cells = len(np.unique(x)) * len(np.unique(cols.y)) * len(np.unique(z))
        available = cols.n >= min_cell_count * cells
    if test == "asymptotic":
        from scipy.stats import chi2

        dof = int(np.sum((_levels(z, x) - 1) * (_levels(z, cols.y) - 1)))
        p = float(chi2.sf(2 * cols.n * np.log(2) * value, dof)) if dof > 0 else 1.0
        return value, p, available

    def evaluate(idx):
        return [_cmi(cols.y, cols.Ky, x[k], Kx, z, Kz) for k in idx]

    null = _null_statistics(n_perm, lambda: rng.permutation(cols.n), evaluate, value, alpha, batch)
    return value, _pvalue(null, value), available


def _components(linked):
    """
    Connected components (sorted tuples) of a boolean adjacency matrix.
    """
    P = len(linked)
    seen, out = set(), []
    for start in range(P):
        if start in seen:
            continue
        stack, members = [start], {start}
        while stack:
            for nb in np.flatnonzero(linked[stack.pop()]).tolist():
                if nb not in members:
                    members.add(nb)
                    stack.append(nb)
        seen |= members
        out.append(tuple(sorted(members)))
    return out


@dataclass
class ContemporaneousFamilies:
    """
    Lag-0 explanation families (:func:`contemporaneous_families`).

    Attributes
    ----------
    graph : np.ndarray
        (processes, processes): 2 where every explanation has ``i -> j``; 1 (both ways) where
        every explanation links ``i`` and ``j`` but they disagree on the direction; 0
        otherwise. Components that were not enumerated (``too_large``, ``unavailable``) keep
        the entries of :func:`contemporaneous_layer`.
    possible : np.ndarray
        (processes, processes), boolean and symmetric: ``i`` and ``j`` are linked in at least
        one explanation (adjacent in :func:`contemporaneous_layer` for components that were
        not enumerated). The union of the explanations is not itself an explanation.
    families : list of ExplanationFamily
        One per connected component of the screen with at least two processes; nodes are
        process indices.
    values, pvalues : np.ndarray
        Each pair's contemporaneous CMI (bits) given the lagged conditioning set, and its
        permutation p-value (:func:`instantaneous_exchange`).
    """

    graph: np.ndarray
    possible: np.ndarray
    families: list
    values: np.ndarray
    pvalues: np.ndarray


def _family_permutations(n_perm, n_statements, alpha):
    """
    Permutations per statement, enough that the smallest p-value clears ``alpha / n_statements``.
    """
    if n_perm is None:
        return max(200, 2 * int(np.ceil(n_statements / alpha)))
    check_n_perm(n_perm, alpha / n_statements)
    return n_perm


def contemporaneous_families(
    data,
    skeleton,
    alpha=0.05,
    n_perm=None,
    prng=None,
    exchange=None,
    max_family_size=MAX_NODES,
    convention="neither",
    tiers=None,
    forbidden=(),
    test="permutation",
    min_cell_count=5,
):
    """
    Every lag-0 explanation that weaker adjacency faithfulness allows, per component.

    A PC-style lag-0 layer (:func:`contemporaneous_layer`) cannot see a link whose
    dependence shows only jointly: if :math:`Z_t = X_t \\oplus Y_t \\oplus E_t`, every pair is
    independent. Following :cite:`Hourican2026`, this reports instead the family of DAGs
    that are Markov, weakly adjacency faithful :cite:`Marx2021weaker`, SGS-minimal and
    consistent with background knowledge (:func:`infoflow.families.explanation_family`):

    1. **Screen.** Two processes are linked when their present values are dependent given
       the lagged conditioning set (both processes' selected pasts and lagged parents), or
       given that set and one more process's present value, as in XOR. Linked processes form
       components.
    2. **Decisions.** In each component of at most `max_family_size` processes, every
       pairwise statement :math:`X^i_t \\perp X^j_t \\mid X^S_t` (also given the pair's lagged
       conditioning set) is tested at level :math:`\\alpha / m` for :math:`m` statements
       (Bonferroni). One false rejection among the many independences a true DAG implies
       removes it from the family, so uncorrected levels leave many families empty.
    3. **Family.** Every DAG on the component is checked against the decisions.

    Parameters
    ----------
    data : DiscreteData
    skeleton : dict
        ``target -> TargetSkeleton``.
    alpha : float
        The screen's level, and the family-wise level of each component's decisions.
    n_perm : int, None
        Permutations per test; by default 200 for the screen and twice what each
        component's corrected level needs for its decisions.
    prng : None, int, Generator
    exchange : tuple, None
        The ``(exchange, pvalues)`` pair from :func:`instantaneous_exchange`, to reuse.
    max_family_size : int
        Larger components are not enumerated (status ``too_large``) and keep the entries of
        :func:`contemporaneous_layer`. At most :data:`~infoflow.families.MAX_NODES`.
    convention : {'neither', 'not_both'}
        The strictness of the two-association (:func:`~infoflow.families.explanation_family`).
    tiers : dict or sequence, None
        A tier per process (a sequence follows process order); no lag-0 arc may point to an
        earlier tier, e.g. when a within-sample order is known.
    forbidden : iterable of (int, int)
        Lag-0 arcs (process indices) that no explanation may contain.
    test : {'permutation', 'asymptotic'}
        How the decisions are tested (see the screen and :func:`_lag0_test`).
    min_cell_count : int, None
        A statement whose table has fewer samples per cell on average cannot be decided
        reliably; its component gets status ``unavailable`` and keeps the entries of
        :func:`contemporaneous_layer`, since a non-rejection there is no evidence of
        independence. None disables the check.

    Returns
    -------
    ContemporaneousFamilies
    """
    if not 1 <= max_family_size <= MAX_NODES:
        raise ValueError(f"max_family_size must be between 1 and {MAX_NODES}")
    if test not in ("permutation", "asymptotic"):
        raise ValueError("test must be 'permutation' or 'asymptotic'")
    if convention not in ("neither", "not_both"):
        raise ValueError("convention must be 'neither' or 'not_both'")
    rng = as_generator(prng)
    P = data.n_processes
    screen_perm = n_perm or 200
    check_n_perm(screen_perm, alpha)
    values, pvalues = exchange if exchange is not None else instantaneous_exchange(data, skeleton, screen_perm, rng)
    tier = None if tiers is None else (dict(tiers) if isinstance(tiers, dict) else dict(enumerate(tiers)))
    forbidden = [tuple(arc) for arc in forbidden]

    linked = (pvalues <= alpha) & ~np.eye(P, dtype=bool)
    for i, j in combinations(range(P), 2):
        if linked[i, j]:
            continue
        cond = _conditioning(skeleton, i, j)
        for k in range(P):
            if k in (i, j):
                continue
            _, p, _ = _lag0_test(data, i, j, [k], cond, screen_perm, rng, alpha, 20, test)
            if p <= alpha:
                linked[i, j] = linked[j, i] = True
                break

    families = []
    for nodes in _components(linked):
        if len(nodes) < 2:
            continue
        if len(nodes) > max_family_size:
            families.append(ExplanationFamily(nodes, status="too_large", convention=convention))
            continue
        queries = statements(len(nodes))
        level = alpha / len(queries)
        perms = _family_permutations(n_perm, len(queries), alpha)
        independent, found = {}, {}
        for i, j, S in queries:
            a, b = nodes[i], nodes[j]
            given = [nodes[k] for k in sorted(S)]
            cond = _conditioning(skeleton, a, b)
            _, p, ok = _lag0_test(data, a, b, given, cond, perms, rng, level, 20, test, min_cell_count)
            if not ok:
                break
            independent[i, j, S], found[i, j, S] = p > level, p
        else:
            decisions = CIDecisions(nodes, independent, found, level)
            families.append(explanation_family(decisions, convention=convention, tiers=tier, forbidden=forbidden))
            continue
        families.append(ExplanationFamily(nodes, status="unavailable", convention=convention))

    graph = np.zeros((P, P), dtype=np.int64)
    possible = np.zeros((P, P), dtype=bool)
    fallback = [f for f in families if f.status in ("too_large", "unavailable")]
    if fallback:
        pc, _, _ = contemporaneous_layer(data, skeleton, alpha, screen_perm, rng, (values, pvalues))
        for family in fallback:
            idx = np.ix_(family.nodes, family.nodes)
            graph[idx] = pc[idx]
            possible[idx] = (pc[idx] != 0) | (pc[idx].T != 0)
    for family in families:
        if family.status != "complete":
            continue
        arrows = family.compelled_arrows
        for pair in family.compelled_adjacencies:
            u, v = sorted(pair)
            if (u, v) in arrows:
                graph[u, v] = 2
            elif (v, u) in arrows:
                graph[v, u] = 2
            else:
                graph[u, v] = graph[v, u] = 1
        for pair in family.possible_adjacencies:
            u, v = sorted(pair)
            possible[u, v] = possible[v, u] = True
    return ContemporaneousFamilies(graph, possible, families, values, pvalues)


def to_tigramite(data, names=None):
    """
    A ``tigramite.data_processing.DataFrame`` of the discrete symbols (one per trial
    becomes Tigramite's multiple-dataset format).
    """
    try:
        from tigramite import data_processing as pp
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "to_tigramite needs the optional dependency tigramite (pip install infoflow[causal])"
        ) from error
    arrays = {i: np.asarray(p.T, dtype=float) for i, p in enumerate(data.past)}
    return pp.DataFrame(arrays, analysis_mode="multiple", var_names=names or data.names)


def run_tigramite(data, tau_max=3, method="pcmciplus", pc_alpha=0.01, **kwargs):  # pragma: no cover - optional
    """
    Run Tigramite's PCMCI+ (or LPCMCI) with the symbolic CMI test on the symbols.

    Returns the Tigramite results dict.
    """
    from tigramite.independence_tests.cmisymb import CMIsymb

    frame = to_tigramite(data)
    test = CMIsymb(significance="shuffle_test", **kwargs)
    if method == "lpcmci":
        from tigramite.lpcmci import LPCMCI

        return LPCMCI(dataframe=frame, cond_ind_test=test).run_lpcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    from tigramite.pcmci import PCMCI

    return PCMCI(dataframe=frame, cond_ind_test=test).run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)
