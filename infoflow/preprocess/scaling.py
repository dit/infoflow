"""
Automated scaling regions :cite:`Deshmukh2021`.

Every interval of at least `min_points` consecutive points is fitted by least
squares; each fit is weighted by the length of its fitted line and the inverse
square of its fit error. The weighted distribution of slopes has its mode at the
slope of the scaling region, and the intervals near the mode give its extent;
a wide distribution means there is no clear scaling region.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ("ScalingRegion", "block_entropy_scaling", "scaling_region")


@dataclass
class ScalingRegion:
    """
    The result of :func:`scaling_region`.

    Attributes
    ----------
    slope, intercept : float
        The weighted mode of the fitted slopes, and the weighted median intercept of
        fits near that mode.
    slope_iqr : tuple
        Weighted interquartile range of the slopes (uncertainty).
    start, stop : float
        The weighted median endpoints of fits near the mode.
    n_fits : int
    """

    slope: float
    intercept: float
    slope_iqr: tuple
    start: float
    stop: float
    n_fits: int


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w) / w.sum()
    return np.interp(q, cdf, v)


def scaling_region(x, y, min_points=3, bins=50):
    """
    Estimate the slope and extent of the scaling region of the curve ``y(x)``.

    Parameters
    ----------
    x, y : array_like
    min_points : int
        The smallest interval fitted.
    bins : int
        Histogram bins for locating the mode of the slope distribution.

    Returns
    -------
    ScalingRegion
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < min_points:
        raise ValueError("too few points for a scaling region")
    slopes, intercepts, weights, starts, stops = [], [], [], [], []
    for i in range(n):
        for j in range(i + min_points, n + 1):
            xs, ys = x[i:j], y[i:j]
            A = np.stack([xs, np.ones_like(xs)], axis=1)
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            resid = ys - A @ coef
            err = np.sqrt(np.mean(resid**2)) + 1e-12 * max(1.0, np.abs(ys).max())
            length = np.hypot(xs[-1] - xs[0], coef[0] * (xs[-1] - xs[0]))
            slopes.append(coef[0])
            intercepts.append(coef[1])
            weights.append(length / err**2)
            starts.append(xs[0])
            stops.append(xs[-1])
    slopes, intercepts, weights = np.array(slopes), np.array(intercepts), np.array(weights)
    starts, stops = np.array(starts), np.array(stops)
    weights = weights / weights.sum()
    hist, edges = np.histogram(slopes, bins=bins, weights=weights)
    k = int(np.argmax(hist))
    lo, hi = edges[k], edges[k + 1]
    near = (slopes >= lo) & (slopes <= hi)
    mode = float(_weighted_quantile(slopes[near], weights[near], 0.5))
    return ScalingRegion(
        slope=mode,
        intercept=float(_weighted_quantile(intercepts[near], weights[near], 0.5)),
        slope_iqr=(float(_weighted_quantile(slopes, weights, 0.25)), float(_weighted_quantile(slopes, weights, 0.75))),
        start=float(_weighted_quantile(starts[near], weights[near], 0.5)),
        stop=float(_weighted_quantile(stops[near], weights[near], 0.5)),
        n_fits=len(slopes),
    )


def block_entropy_scaling(symbols, max_length=None, estimator="grassberger", min_points=3):
    """
    Entropy rate and excess entropy from the scaling region of the block entropy.

    ``H(L) ~ E + h L`` in the scaling region, so its slope estimates the entropy rate
    :math:`h_\\mu` and its intercept the excess entropy :math:`E`. Lengths are capped
    where the words become undersampled (at most ``N / 5`` distinct words).

    Parameters
    ----------
    symbols : array_like or Trials
    max_length : int, None
    estimator : str
        dit entropy estimator for the block entropies.
    min_points : int

    Returns
    -------
    region : ScalingRegion
    lengths, entropies : np.ndarray
    """
    import warnings

    from dit.inference import UndersamplingWarning, block_entropy

    lengths, entropies = [], []
    L = 1
    while max_length is None or max_length >= L:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UndersamplingWarning)
            h = block_entropy(symbols, L, estimator)
        if any(issubclass(w.category, UndersamplingWarning) for w in caught) and min_points < L:
            break
        lengths.append(L)
        entropies.append(h)
        L += 1
        if L > 64:
            break
    lengths, entropies = np.array(lengths), np.array(entropies)
    return scaling_region(lengths, entropies, min_points=min(min_points, len(lengths))), lengths, entropies
