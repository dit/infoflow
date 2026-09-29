"""
Cross-validated predictive scores for choosing discretizations and history budgets.

A candidate encoding of one process is scored by how well its past symbols predict
fixed *reference targets*: the next value binned by equal-frequency bins of the
raw series at several resolutions (2, 4, and 8 bins, as the data allow). At
resolution :math:`B` the held-out score is

.. math::

    R_B = 1 - \\frac{\\text{log-loss of the context model}}{\\text{log-loss of the marginal model}},

the fraction of the reference target's uncertainty explained by the past, i.e. the
history dependence :math:`R = I[\\text{past} : \\text{present}] / H(\\text{present})`
of Rudelt et al. :cite:`Rudelt2021`, estimated out of sample. The multi-resolution
score :math:`\\bar R` averages :math:`R_B` over resolutions. Because every
candidate predicts the same targets, ordinal and binned candidates with different
alphabets are compared on equal terms; this is the discrete analogue of the
forecast error behind information-storage embedding selection
:cite:`Garland2016`. Averaging over resolutions is specific to this package.
"""

import numpy as np

__all__ = ("context_codes", "reference_codes", "score_contexts")


def reference_codes(series, bins):
    """
    Equal-frequency codes of the raw series (pooled edges), per trial.
    """
    pooled = np.concatenate(series)
    edges = np.quantile(pooled, np.linspace(0, 1, bins + 1)[1:-1])
    return [np.searchsorted(edges, s, side="right") for s in series]


def context_codes(symbol_trials, lags, alphabet):
    """
    Joint codes of the past symbols at `lags` (relative to each time t), per trial.

    Returns a list of (codes, times) with codes defined for all t with every lag valid.
    """
    out = []
    lags = list(lags)
    max_lag = max(lags)
    for sym in symbol_trials:
        N = len(sym)
        t = np.arange(max_lag, N)
        code = np.zeros(len(t), dtype=np.int64)
        valid = np.ones(len(t), dtype=bool)
        for lag in lags:
            col = sym[t - lag]
            valid &= col >= 0
            code = code * alphabet + np.maximum(col, 0)
        out.append((code[valid], t[valid]))
    return out


def _fold_scores(contexts, targets, B, folds, gap, alpha=0.5):
    """
    Per-fold held-out R_B for one reference resolution.
    """
    ctx = np.concatenate([c for c, _ in contexts])
    tgt = np.concatenate([targets[i][t] for i, (_, t) in enumerate(contexts)])
    trial = np.concatenate([np.full(len(c), i) for i, (c, _) in enumerate(contexts)])
    _, ctx = np.unique(ctx, return_inverse=True)
    ctx = ctx.ravel()
    n = len(ctx)
    position = np.zeros(n, dtype=np.int64)
    for i in np.unique(trial):
        m = trial == i
        position[m] = np.arange(m.sum())
    lengths = np.bincount(trial)
    fold = np.minimum(folds - 1, (folds * position / np.maximum(lengths[trial], 1)).astype(np.int64))
    scores = []
    K = int(ctx.max()) + 1
    for f in range(folds):
        test = fold == f
        # Drop `gap` realizations on either side of the test block from training.
        near = np.zeros(n, dtype=bool)
        for i in np.unique(trial):
            m = np.flatnonzero((trial == i) & test)
            if len(m):
                lo, hi = position[m].min(), position[m].max()
                near |= (trial == i) & (position >= lo - gap) & (position <= hi + gap)
        train = ~near
        if test.sum() < B * 10 or train.sum() < B * 10:
            continue
        joint = np.zeros((K, B))
        np.add.at(joint, (ctx[train], tgt[train]), 1.0)
        marginal = np.bincount(tgt[train], minlength=B).astype(float)
        cond = (joint + alpha) / (joint.sum(axis=1, keepdims=True) + alpha * B)
        marg = (marginal + alpha) / (marginal.sum() + alpha * B)
        loss_model = -np.mean(np.log(cond[ctx[test], tgt[test]]))
        loss_base = -np.mean(np.log(marg[tgt[test]]))
        scores.append(1.0 - loss_model / loss_base if loss_base > 0 else 0.0)
    return np.array(scores)


def score_contexts(contexts, raw_series, resolutions=(2, 4, 8), folds=5, gap=1, min_per_bin=50):
    """
    The multi-resolution score of one candidate context encoding.

    Parameters
    ----------
    contexts : list of (codes, times)
        From :func:`context_codes`.
    raw_series : list of np.ndarray
        The raw trials of the process (for the reference targets).
    resolutions : tuple of int
        Reference resolutions; one is used only if every bin has at least
        `min_per_bin` held-out observations on average.
    folds : int
        Blocked cross-validation folds per trial.
    gap : int
        Realizations dropped between training and test blocks.

    Returns
    -------
    mean : float
        :math:`\\bar R`.
    se : float
        Standard error across folds.
    per_resolution : dict
        ``B -> (mean R_B, fold scores)``.
    fold_scores : np.ndarray
        Per-fold scores averaged over resolutions (for bootstrapping).
    """
    n = sum(len(c) for c, _ in contexts)
    usable = [B for B in resolutions if n / folds / B >= min_per_bin] or [min(resolutions)]
    per = {}
    for B in usable:
        targets = reference_codes(raw_series, B)
        per[B] = _fold_scores(contexts, targets, B, folds, gap)
    k = min(len(v) for v in per.values())
    if k == 0:
        return 0.0, np.inf, per, np.zeros(0)
    fold_scores = np.mean([v[:k] for v in per.values()], axis=0)
    se = float(np.std(fold_scores, ddof=1) / np.sqrt(k)) if k > 1 else np.inf
    return float(np.mean(fold_scores)), se, {B: (float(np.mean(v)), v) for B, v in per.items()}, fold_scores
