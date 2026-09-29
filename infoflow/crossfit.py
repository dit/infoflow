"""
Cross-fitted estimation of the flow layers.

Plug-in conditional mutual information is biased upward, and minimizing it over
channels on a noisy distribution biases the intrinsic flow downward. Cross-fitting
separates the two :cite:`Chernozhukov2018`: the context merge and the minimizing
channel are fitted on one fold and evaluated, with a bias-corrected estimator, on
the other. Because the channel was not tuned to the evaluation fold's noise, its
value there is an honest (feasible) upper bound on that fold's intrinsic flow;
the identity and constant channels are always feasible too, so the evaluated
intrinsic flow is the minimum of the three. The folds are then swapped and averaged.
"""

import numpy as np

from .context import apply_merge, merge_contexts_statistical
from .measures import (
    EdgeFlow,
    _degrade,
    cmi_from_joint,
    dense_codes,
    intrinsic_flow,
    joint_table,
    layers_from_values,
)

__all__ = ("crossfit_flows",)


def _fold_split(n, gap):
    """
    Two contiguous halves with `gap` realizations dropped at the boundary.
    """
    half = n // 2
    a = np.arange(0, max(half - gap // 2, 0))
    b = np.arange(min(half + (gap - gap // 2), n), n)
    return a, b


def _evaluate(table_b, mapping, Q, estimator):
    merged_b = apply_merge(table_b, mapping, n_clusters=Q.shape[0])
    n = merged_b.sum()
    te = cmi_from_joint(table_b, estimator)
    tdmi = cmi_from_joint(table_b.sum(axis=2, keepdims=True), estimator)
    iif = cmi_from_joint(_degrade(merged_b, Q), estimator, n=n)
    return te, tdmi, min(iif, te, tdmi)


def crossfit_flows(
    y,
    x,
    w,
    estimator="miller_madow",
    merge=True,
    merge_alpha=0.01,
    gap=0,
    restarts=3,
    prng=None,
    fold_index=None,
):
    """
    Cross-fitted intrinsic, synergistic, and shared flow for one edge.

    Parameters
    ----------
    y, x, w : array_like
        The target present, source past, and context of each aligned realization
        (1D codes or 2D rows of joint values); `w` may be None.
    estimator : {'miller_madow', 'plugin'}
        The evaluation estimator on the held-out fold.
    merge : bool
        Merge statistically indistinguishable context values before fitting.
    merge_alpha : float
        Level of the merging G-tests.
    gap : int
        Realizations dropped between the folds (use the embedding span so the
        folds share no windows).
    restarts, prng
        Passed to :func:`~infoflow.measures.intrinsic_flow`.
    fold_index : (np.ndarray, np.ndarray), None
        Explicit fold indices, e.g. whole trials; defaults to contiguous halves.

    Returns
    -------
    EdgeFlow
        Averaged over the two fold orderings, with the full-data plug-in values in
        ``raw``.
    """
    y, Ky = dense_codes(y)
    x, Kx = dense_codes(x)
    w, Kw = dense_codes(np.zeros(len(y), dtype=np.int64) if w is None else w)
    shape = (Ky, Kx, Kw)
    full = joint_table(y, x, w, shape)
    a, b = _fold_split(len(y), gap) if fold_index is None else fold_index
    results, channels, contexts = [], [], []
    for fit, ev in ((a, b), (b, a)):
        if len(fit) == 0 or len(ev) == 0:
            continue
        table_fit = joint_table(y[fit], x[fit], w[fit], shape)
        table_ev = joint_table(y[ev], x[ev], w[ev], shape)
        mapping = merge_contexts_statistical(table_fit, merge_alpha) if merge else np.arange(Kw)
        merged_fit = apply_merge(table_fit, mapping)
        _, Q = intrinsic_flow(merged_fit, restarts=restarts, prng=prng)
        results.append(_evaluate(table_ev, mapping, Q, estimator))
        channels.append(Q)
        contexts.append(merged_fit.shape[2])
    if not results:
        raise ValueError("too few realizations to cross-fit")
    te, tdmi, iif = (float(np.mean(v)) for v in zip(*results, strict=True))
    te, tdmi, iif, syn, shared, clipped = layers_from_values(te, tdmi, iif)
    raw_iif, _ = intrinsic_flow(full, restarts=restarts, prng=prng)
    raw = {"te": cmi_from_joint(full), "tdmi": cmi_from_joint(full.sum(axis=2, keepdims=True)), "intrinsic": raw_iif}
    return EdgeFlow(te, tdmi, iif, syn, shared, channels[0], len(y), int(np.mean(contexts)), clipped, raw)
