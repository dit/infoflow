"""
An optional hyperedge layer: two-source partial information decompositions.

For each target with at least two parents, each pair of parents' selected
variables and the target's present are decomposed into redundant, unique, and
synergistic information with one of dit's PID measures :cite:`Williams2010`
(default :class:`dit.pid.PID_BROJA` :cite:`Bertschinger2014`). The decomposition
is of :math:`I[Y_t : (A, B)]`, not conditioned on the target's past; it
complements the synergistic edge layer, which is synergy *with the context*.
"""

from itertools import combinations

import dit
import numpy as np

from .data import realizations
from .measures import dense_codes

__all__ = ("pid_hyperedges",)


def _distribution(y, a, b, max_outcomes):
    y, _ = dense_codes(y)
    a, _ = dense_codes(a)
    b, _ = dense_codes(b)
    keys, counts = np.unique(np.stack([a, b, y], axis=1), axis=0, return_counts=True)
    if len(keys) > max_outcomes:
        return None
    return dit.Distribution([tuple(int(v) for v in k) for k in keys], counts / counts.sum())


def pid_hyperedges(data, skeleton, measure=None, max_outcomes=512):
    """
    Two-source PIDs for every pair of parents of every target.

    Parameters
    ----------
    data : DiscreteData
    skeleton : dict
    measure : class, None
        A dit PID class (default ``dit.pid.PID_BROJA``).
    max_outcomes : int
        Skip pairs whose joint has more outcomes (the optimizations grow quickly).

    Returns
    -------
    list of dict
        ``target``, ``sources`` (process pair), ``redundancy``, ``unique`` (pair),
        ``synergy`` in bits.
    """
    from dit.pid import PID_BROJA

    measure = measure or PID_BROJA
    out = []
    for t, sk in skeleton.items():
        parents = sk.parents()
        for a, b in combinations(parents, 2):
            va, vb = sk.source_variables(a), sk.source_variables(b)
            variables = list(dict.fromkeys(va + vb))
            r = realizations(data, t, variables, max_lag=max(v[1] for v in variables))
            d = _distribution(r.present, r.columns(va), r.columns(vb), max_outcomes)
            if d is None:
                continue
            pid = measure(d, [[0], [1]], [2])
            out.append(
                {
                    "target": t,
                    "sources": (a, b),
                    "redundancy": float(pid.get_pi(((0,), (1,)))),
                    "unique": (float(pid.get_pi(((0,),))), float(pid.get_pi(((1,),)))),
                    "synergy": float(pid.get_pi(((0, 1),))),
                }
            )
    return out
