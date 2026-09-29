"""
The node layer: active information storage and predictability.

Active information storage :math:`A = I[X_t : X_{\\text{past}}]` :cite:`Lizier2012storage`
uses the node's selected (non-uniform) past from the skeleton, is tested against
permutations of the past, and is corrected across nodes. Where raw data are
available, the normalized weighted permutation entropy :cite:`Fadlallah2013`
gives a model-free predictability score :cite:`Garland2014`; a node whose
weighted permutation entropy is near its maximum and whose storage is not
significant is flagged ``near-unpredictable``. Such nodes stay in the network,
since they can still receive (e.g. synergistic) flow.
"""

import numpy as np
from dit.inference import weighted_permutation_entropy
from dit.inference._symbols import as_generator

from .selection import _cmi, _Columns

__all__ = ("node_layer",)


def _ais(data, target, past, n_perm, rng):
    if not past:
        return 0.0, 1.0, 0
    max_lag = max(v[1] for v in past)
    cols = _Columns(data, target, list(past), max_lag)
    s, Ks = cols.joint(list(past))
    z = np.zeros(cols.n, dtype=np.int64)
    value = _cmi(cols.y, cols.Ky, s, Ks, z, 1)
    null = np.array([_cmi(cols.y, cols.Ky, s[rng.permutation(cols.n)], Ks, z, 1) for _ in range(n_perm)])
    return value, float((1 + np.sum(null >= value - 1e-12)) / (1 + n_perm)), cols.n


def node_layer(data, skeleton, n_perm=200, alpha=0.05, wpe_threshold=0.95, prng=None):
    """
    Active information storage, its significance, and predictability flags per node.

    Parameters
    ----------
    data : DiscreteData
    skeleton : dict
        ``target -> TargetSkeleton`` (its ``target_past`` is the node's embedding).
    n_perm : int
    alpha : float
        FDR level across nodes.
    wpe_threshold : float
        Normalized weighted permutation entropy above which a node counts as near-random.
    prng : None, int, Generator

    Returns
    -------
    dict
        Arrays over nodes: ``ais``, ``ais_pvalue``, ``ais_significant``, ``wpe``,
        and ``flags`` (strings).
    """
    from .stats import benjamini_hochberg

    rng = as_generator(prng)
    P = data.n_processes
    ais = np.zeros(P)
    pvalues = np.ones(P)
    for p in range(P):
        past = skeleton[p].target_past if p in skeleton else []
        ais[p], pvalues[p], _ = _ais(data, p, past, n_perm, rng)
    reject, _ = benjamini_hochberg(pvalues, alpha)
    wpe = np.full(P, np.nan)
    if data.raw is not None:
        from dit.inference import Trials

        for p in range(P):
            series = [r[:, p] for r in data.raw]
            wpe[p] = weighted_permutation_entropy(
                Trials(series) if len(series) > 1 else series[0], 3, 1, normalize=True
            )
    flags = []
    for p in range(P):
        f = []
        if np.isfinite(wpe[p]) and wpe[p] > wpe_threshold and not reject[p]:
            f.append("near-unpredictable")
        flags.append(",".join(f))
    return {"ais": ais, "ais_pvalue": pvalues, "ais_significant": reject, "wpe": wpe, "flags": np.array(flags)}
