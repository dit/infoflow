"""
Screening diagnostics run before discretization :cite:`Bradley2015`.
"""

from dataclasses import dataclass

import numpy as np
from dit.inference import ordinal_patterns
from scipy import stats

__all__ = ("ScreenResult", "screen", "split_missing")


@dataclass
class ScreenResult:
    """
    Per-process screening results.

    Attributes
    ----------
    tie_fraction : float
        Fraction of consecutive equal values (quantization).
    stationarity_pvalue : float
        p-value of a G-test that non-overlapping ordinal-pattern frequencies are the
        same across segments.
    nonstationary : bool
    n_missing : int
    """

    tie_fraction: float
    stationarity_pvalue: float
    nonstationary: bool
    n_missing: int


def split_missing(trials, min_length=20):
    """
    Split trials at missing samples (NaN in any process); drop pieces shorter than `min_length`.
    """
    out = []
    for t in trials:
        bad = ~np.isfinite(t).all(axis=1)
        if not bad.any():
            out.append(t)
            continue
        edges = np.flatnonzero(np.diff(np.concatenate([[1], bad.astype(int), [1]])))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            if stop - start >= min_length:
                out.append(t[start:stop])
    return out


def _stationarity(series, segments=4, order=3):
    rows = []
    for t in series:
        n = len(t) // segments
        for k in range(segments):
            seg = t[k * n : (k + 1) * n]
            if len(seg) < 3 * order:
                continue
            patterns = ordinal_patterns(seg, order)[::order]
            rows.append(np.bincount(patterns, minlength=6))
    if len(rows) < 2:
        return 1.0
    table = np.array(rows, dtype=float)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 1.0
    g, p, _, _ = stats.chi2_contingency(table, lambda_="log-likelihood")
    return float(p)


def screen(trials, alpha=0.01, segments=4):
    """
    Screen raw trials for ties, nonstationarity, and missing data.

    Parameters
    ----------
    trials : list of np.ndarray
        Raw trials, shape (samples, processes), possibly with NaNs.
    alpha : float
        Level at which the stationarity test flags a process.
    segments : int

    Returns
    -------
    list of ScreenResult
        One per process.
    """
    P = trials[0].shape[1]
    missing = [int(sum(np.sum(~np.isfinite(t[:, p])) for t in trials)) for p in range(P)]
    clean = split_missing(trials)
    results = []
    for p in range(P):
        series = [t[:, p] for t in clean]
        diffs = np.concatenate([np.diff(s) for s in series if len(s) > 1])
        ties = float(np.mean(diffs == 0)) if len(diffs) else 0.0
        pval = _stationarity(series, segments)
        results.append(ScreenResult(ties, pval, pval < alpha, missing[p]))
    return results
