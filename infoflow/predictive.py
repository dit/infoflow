"""
Transfer entropy as a difference in held-out predictive log-likelihood.

AGM-TE :cite:`Kornai2025` estimates transfer entropy as the difference between the
log-likelihoods that two probabilistic forecasters of the target assign to held-out
data, one given the source's past and one without it:

.. math::

    \\widehat{TE} = \\frac{1}{|\\text{test}|} \\sum_{t \\in \\text{test}}
    \\log q_{\\text{full}}(y_t \\mid y_{\\text{past}}, x_{\\text{past}}, z_{\\text{past}})
    - \\log q_{\\text{restricted}}(y_t \\mid y_{\\text{past}}, z_{\\text{past}}).

Any forecaster can be used, so the estimator scales to long histories and many
conditions where binned estimates fail; held-out evaluation means overfitting
lowers the estimate instead of inflating it, so independent processes give values
near (or below) zero. Folds are contiguous blocks with the embedding span dropped
at their boundaries, so training and evaluation windows never overlap.

Built-in forecasters are ``'gaussian'`` (linear regression with Gaussian
residuals, whose TE is Granger causality :cite:`Barnett2009`) and
``'categorical'`` (smoothed conditional frequency tables for discrete series). A
callable ``model(X_train, y_train)`` returning ``logpdf(X_test, y_test)`` (per-sample
natural-log densities) plugs in anything else, e.g. the neural forecasters of
:cite:`Kornai2025`.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ("PredictiveTE", "predictive_transfer_entropy")


@dataclass
class PredictiveTE:
    """
    A predictive transfer-entropy estimate.

    Attributes
    ----------
    value : float
        Bits per sample, averaged over held-out samples.
    folds : numpy.ndarray
        Per-fold estimates.
    stderr : float
        Standard error from the spread of the fold estimates.
    pvalue : float
        One-sided t-test of the fold estimates against zero (approximate: folds of
        one series are only roughly independent).
    """

    value: float
    folds: np.ndarray
    stderr: float
    pvalue: float


def _lags(x, start, stop, L, lag=1):
    """
    Rows ``x[t - lag - L + 1 : t - lag + 1]`` for ``t`` in ``range(start, stop)``.
    """
    x = np.asarray(x)
    x = x[:, None] if x.ndim == 1 else x
    if L == 0:
        return np.zeros((stop - start, 0))
    return np.column_stack([x[start - lag - j : stop - lag - j] for j in range(L)])


def _gaussian(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    var = max(float(np.mean((y - A @ beta) ** 2)), 1e-12)

    def logpdf(Xt, yt):
        r = yt - np.column_stack([np.ones(len(Xt)), Xt]) @ beta
        return -0.5 * np.log(2 * np.pi * var) - r**2 / (2 * var)

    return logpdf


def _categorical(prior=0.5, n_symbols=None):
    def fit(X, y):
        K = n_symbols or len(np.unique(y))
        table = {}
        for row, v in zip(map(tuple, X), y, strict=True):
            table.setdefault(row, {}).setdefault(v, 0)
            table[row][v] += 1

        def logpdf(Xt, yt):
            out = np.empty(len(yt))
            for i, (row, v) in enumerate(zip(map(tuple, Xt), yt, strict=True)):
                counts = table.get(row, {})
                out[i] = np.log((counts.get(v, 0) + prior) / (sum(counts.values()) + prior * K))
            return out

        return logpdf

    return fit


def _folds(n, n_folds, gap):
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        test = np.arange(a, b)
        train = np.concatenate([np.arange(0, max(a - gap, 0)), np.arange(min(b + gap, n), n)])
        out.append((train, test))
    return out


def predictive_transfer_entropy(
    source,
    target,
    history_length=1,
    source_history=1,
    lag=1,
    conditions=(),
    model="gaussian",
    n_folds=5,
):
    """
    Held-out predictive transfer entropy :cite:`Kornai2025` (module docstring).

    Parameters
    ----------
    source, target : array_like
        Equal-length series (rows of 2D arrays are joint values).
    history_length : int
        Target past :math:`k \\ge 0`.
    source_history : int
        Source past :math:`l \\ge 1`, ending `lag` steps before the present.
    lag : int
    conditions : sequence of array_like
        Further series whose pasts (length ``max(k, 1)``) both forecasters see.
    model : {'gaussian', 'categorical'} or callable
        ``model(X_train, y_train) -> logpdf(X_test, y_test)``.
    n_folds : int

    Returns
    -------
    PredictiveTE
    """
    y_all = np.asarray(target)
    N = len(y_all)
    k, l, m = int(history_length), int(source_history), max(int(history_length), 1)
    if l < 1 or lag < 1 or k < 0:
        raise ValueError("need history_length >= 0, source_history >= 1, and lag >= 1")
    t0 = max(k, l + lag - 1, m if conditions else 0)
    restricted = [_lags(y_all, t0, N, k)] + [_lags(z, t0, N, m) for z in conditions]
    R = np.column_stack(restricted)
    F = np.column_stack([R, _lags(source, t0, N, l, lag)])
    y = y_all[t0:N]
    if model == "gaussian":
        fit = _gaussian
        y = y.astype(float)
    elif model == "categorical":
        fit = _categorical(n_symbols=len(np.unique(y)))
    elif callable(model):
        fit = model
    else:
        raise ValueError("model must be 'gaussian', 'categorical', or a callable")
    values, weights = [], []
    for train, test in _folds(len(y), n_folds, gap=t0 + 1):
        if len(train) < 2 or len(test) == 0:
            continue
        full = fit(F[train], y[train])(F[test], y[test])
        base = fit(R[train], y[train])(R[test], y[test])
        values.append(float(np.mean(full - base)) / np.log(2))
        weights.append(len(test))
    if not values:
        raise ValueError("too few samples for the folds")
    folds = np.array(values)
    value = float(np.average(folds, weights=weights))
    stderr = float(folds.std(ddof=1) / np.sqrt(len(folds))) if len(folds) > 1 else np.nan
    pvalue = float(stats.t.sf(value / stderr, len(folds) - 1)) if len(folds) > 1 and stderr > 0 else np.nan
    return PredictiveTE(value, folds, stderr, pvalue)
