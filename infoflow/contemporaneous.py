"""
An optional lag-0 layer, and an adapter to Tigramite.

With coarse sampling (or after Faes compensation) processes can depend on each
other within one sample, and time order no longer orients those links. The lag-0
layer tests each pair's contemporaneous dependence given both processes' selected
pasts and lagged parents (a momentary conditional independence test in the spirit
of PCMCI+ :cite:`Runge2020`), removes links explained by a third contemporaneous
neighbour (PC-style), and orients unshielded triples ``i - k - j`` with ``k``
outside the separating set as colliders ``i -> k <- j``. For full latent-variable
output (partial ancestral graphs) use :func:`run_tigramite` with LPCMCI.
"""

from itertools import combinations

import numpy as np
from dit.inference._symbols import as_generator

from .selection import _cmi, _Columns

__all__ = ("contemporaneous_layer", "run_tigramite", "to_tigramite")


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


def contemporaneous_layer(data, skeleton, alpha=0.05, n_perm=200, prng=None):
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
    values = np.zeros((P, P))
    pvalues = np.ones((P, P))
    adjacent = np.zeros((P, P), dtype=bool)
    for i, j in combinations(range(P), 2):
        v, p = _test(data, i, j, _conditioning(skeleton, i, j), n_perm, rng)
        values[i, j] = values[j, i] = v
        pvalues[i, j] = pvalues[j, i] = p
        adjacent[i, j] = adjacent[j, i] = p <= alpha
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
