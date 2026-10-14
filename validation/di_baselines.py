"""
Directed-information network-inference baselines on infoflow's symbols.

Three classic structure learners, all on the same fixed lag grid (each process's past
is the block of its lags ``1..max_lag``), the plug-in CMI of
:func:`infoflow.selection._cmi`, and permutation tests that shuffle the candidate's
past block:

* :func:`pairwise_di`: an edge ``X -> Y`` when the directed information rate
  ``I[Y_t : X_past | Y_past]`` (pairwise transfer entropy) is significant, with
  Benjamini-Hochberg over all pairs.
* :func:`dig_bounded_indegree`: the directed information graph with in-degree at most
  ``K`` (Quinn, Kiyavash & Coleman 2015): per target, the parent set of size ``<= K``
  maximizing the causally conditioned directed information
  ``I[Y_t : A_past | Y_past]``, each member kept if ``I[Y_t : X_past | Y_past,
  (A - X)_past]`` is significant.
* :func:`ocse`: optimal causation entropy (Sun, Taylor & Bollt 2015): greedy forward
  discovery of the source maximizing ``I[Y_t : X_past | Y_past, S_past]`` while
  significant, then backward removal of members no longer significant given the rest.

These are validation baselines, not part of infoflow's API.
"""

from itertools import combinations

import numpy as np

from infoflow.selection import _cmi, _Columns
from infoflow.stats import benjamini_hochberg

__all__ = ("dig_bounded_indegree", "ocse", "pairwise_di")


class _Target:
    """
    One target's aligned realizations of every process's past block.
    """

    def __init__(self, data, target, max_lag):
        P = data.n_processes
        self.blocks = {p: [(p, lag) for lag in range(1, max_lag + 1)] for p in range(P)}
        variables = [v for p in range(P) for v in self.blocks[p]]
        self.cols = _Columns(data, target, variables, max_lag)
        self.target = target
        self._codes = {}

    def code(self, processes):
        key = tuple(sorted(processes))
        if key not in self._codes:
            self._codes[key] = self.cols.joint([v for p in key for v in self.blocks[p]])
        return self._codes[key]

    def cmi(self, sources, given):
        c, Kc = self.code(sources)
        z, Kz = self.code([self.target, *given])
        return _cmi(self.cols.y, self.cols.Ky, c, Kc, z, Kz)

    def test(self, source, given, n_perm, rng):
        """
        ``I[Y_t : X_past | Y_past, given_past]`` and its permutation p-value.
        """
        c, Kc = self.code([source])
        z, Kz = self.code([self.target, *given])
        value = _cmi(self.cols.y, self.cols.Ky, c, Kc, z, Kz)
        null = [_cmi(self.cols.y, self.cols.Ky, c[rng.permutation(len(c))], Kc, z, Kz) for _ in range(n_perm)]
        return value, float((1 + np.sum(np.asarray(null) >= value - 1e-12)) / (1 + n_perm))


def pairwise_di(data, max_lag=1, alpha=0.05, n_perm=200, prng=0):
    """
    Edges ``(source, target)`` whose pairwise directed information is significant (BH).
    """
    rng = np.random.default_rng(prng)
    P = data.n_processes
    pairs, pvalues = [], []
    for t in range(P):
        tg = _Target(data, t, max_lag)
        for s in range(P):
            if s != t:
                pairs.append((s, t))
                pvalues.append(tg.test(s, [], n_perm, rng)[1])
    reject, _ = benjamini_hochberg(pvalues, alpha)
    return {pair for pair, r in zip(pairs, reject, strict=True) if r}


def dig_bounded_indegree(data, K=2, max_lag=1, alpha=0.05, n_perm=200, prng=0):
    """
    The directed information graph with in-degree at most `K` (exhaustive per target).
    """
    rng = np.random.default_rng(prng)
    P = data.n_processes
    edges = set()
    for t in range(P):
        tg = _Target(data, t, max_lag)
        others = [p for p in range(P) if p != t]
        best, best_value = (), 0.0
        for k in range(1, min(K, len(others)) + 1):
            for subset in combinations(others, k):
                value = tg.cmi(subset, [])
                if value > best_value + 1e-12:
                    best, best_value = subset, value
        for s in best:
            if tg.test(s, [p for p in best if p != s], n_perm, rng)[1] <= alpha:
                edges.add((s, t))
    return edges


def ocse(data, max_lag=1, alpha=0.05, n_perm=200, prng=0):
    """
    Optimal causation entropy: forward discovery, then backward removal, per target.
    """
    rng = np.random.default_rng(prng)
    P = data.n_processes
    edges = set()
    for t in range(P):
        tg = _Target(data, t, max_lag)
        chosen = []
        candidates = [p for p in range(P) if p != t]
        while candidates:
            values = {s: tg.cmi([s], chosen) for s in candidates}
            s = max(values, key=values.get)
            if tg.test(s, chosen, n_perm, rng)[1] > alpha:
                break
            chosen.append(s)
            candidates.remove(s)
        for s in list(chosen):
            if tg.test(s, [p for p in chosen if p != s], n_perm, rng)[1] > alpha:
                chosen.remove(s)
        edges |= {(s, t) for s in chosen}
    return edges
