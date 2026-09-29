"""
Merging redundant context values.

The intrinsic flow depends on the context :math:`W` only through
:math:`p(w)` and the conditional laws :math:`p(y, x \\mid w)`. Merging context
values with equal conditional laws is a sufficient statistic, so it leaves the
intrinsic flow (and TE) unchanged while shrinking the channel optimization. On
data, values whose conditional laws are statistically indistinguishable are
merged instead, as CSSR merges histories with equal morphs :cite:`Shalizi2004`.
"""

import numpy as np
from scipy import stats

__all__ = (
    "apply_merge",
    "merge_contexts_exact",
    "merge_contexts_statistical",
)


def merge_contexts_exact(p, atol=1e-12):
    """
    Map each context value to a representative with an identical conditional law.

    Parameters
    ----------
    p : np.ndarray
        Joint table ``p[y, x, w]``.
    atol : float
        Tolerance on the conditional probabilities.

    Returns
    -------
    mapping : np.ndarray
        ``mapping[w]`` is the merged index of context value ``w``.
    """
    p = np.asarray(p, dtype=float)
    Kw = p.shape[2]
    flat = p.reshape(-1, Kw)
    marg = flat.sum(axis=0)
    cond = np.where(marg > 0, flat / np.where(marg > 0, marg, 1), 0)
    mapping = np.full(Kw, -1, dtype=np.int64)
    reps = []
    for w in range(Kw):
        for k, r in enumerate(reps):
            if np.allclose(cond[:, w], cond[:, r], atol=atol):
                mapping[w] = k
                break
        else:
            mapping[w] = len(reps)
            reps.append(w)
    return mapping


def _g_test(table):
    """
    G statistic p-value for a 2 x K table of counts (columns with no counts dropped).
    """
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2 or np.any(table.sum(axis=1) == 0):
        return 1.0
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 2 * np.sum(np.where(table > 0, table * np.log(table / expected), 0.0))
    return float(stats.chi2.sf(g, table.shape[1] - 1))


def merge_contexts_statistical(counts, alpha=0.01, min_count=5):
    """
    Greedily merge context values whose conditional laws are indistinguishable.

    Context values are visited from most to least frequent. Each joins the most
    similar existing cluster whose pooled conditional counts a G-test cannot tell
    apart at level `alpha`, or starts a new cluster. Values seen fewer than
    `min_count` times join the cluster nearest in total-variation distance.

    Parameters
    ----------
    counts : np.ndarray
        Count table ``C[y, x, w]``.
    alpha : float
    min_count : int

    Returns
    -------
    mapping : np.ndarray
        ``mapping[w]`` is the cluster of context value ``w``.
    """
    counts = np.asarray(counts, dtype=float)
    Kw = counts.shape[2]
    flat = counts.reshape(-1, Kw)
    totals = flat.sum(axis=0)
    order = np.argsort(-totals, kind="stable")
    mapping = np.full(Kw, -1, dtype=np.int64)
    clusters = []  # pooled count vectors
    rare = []
    for w in order:
        if totals[w] == 0:
            continue
        if totals[w] < min_count:
            rare.append(w)
            continue
        best, best_p = None, alpha
        for k, pooled in enumerate(clusters):
            pvalue = _g_test(np.stack([pooled, flat[:, w]]))
            if pvalue > best_p:
                best, best_p = k, pvalue
        if best is None:
            mapping[w] = len(clusters)
            clusters.append(flat[:, w].copy())
        else:
            mapping[w] = best
            clusters[best] += flat[:, w]
    for w in rare:
        if not clusters:
            mapping[w] = 0
            clusters.append(flat[:, w].copy())
            continue
        law = flat[:, w] / totals[w]
        distances = [0.5 * np.abs(law - c / c.sum()).sum() for c in clusters]
        k = int(np.argmin(distances))
        mapping[w] = k
        clusters[k] += flat[:, w]
    unseen = mapping < 0
    mapping[unseen] = 0 if clusters else np.arange(unseen.sum())
    return mapping


def apply_merge(table, mapping, n_clusters=None):
    """
    Sum the context axis of ``table[y, x, w]`` according to `mapping`.
    """
    n_clusters = int(mapping.max()) + 1 if n_clusters is None else n_clusters
    merged = np.zeros(table.shape[:2] + (n_clusters,))
    for w, k in enumerate(mapping):
        merged[:, :, k] += table[:, :, w]
    return merged
