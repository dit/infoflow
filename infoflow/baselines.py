"""
Baseline network-inference methods for the validation harness.

Recent benchmarks find that simple baselines are hard to beat: on real river
networks a plain vector autoregression is the most robust method
:cite:`Stein2025`, and on CausalTime few methods reach an AUROC of 0.9
:cite:`Cheng2024`. :func:`var_granger` is that baseline (pairwise-conditional
linear Granger causality, equal to transfer entropy for Gaussian processes
:cite:`Barnett2009`); :func:`tigramite_baseline` runs PCMCI+ :cite:`Runge2020` or
LPCMCI :cite:`Gerhardus2020` with partial correlation on the raw values. Every
baseline returns a :class:`BaselineNetwork`, which
:func:`~infoflow.benchmarks.score_edges` scores like an inferred network.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = ("BaselineNetwork", "run_baseline", "tigramite_baseline", "var_granger")


@dataclass
class BaselineNetwork:
    """
    A baseline's links, lags, scores, and p-values over (source, target).

    Attributes
    ----------
    found : set of (int, int)
        Links declared significant.
    lags : numpy.ndarray
        Most likely lag of each pair.
    score : numpy.ndarray
        Continuous link score (larger is more confident), for rank metrics.
    pvalue : numpy.ndarray
    latent : set of (int, int)
        Pairs marked as sharing a latent confounder (LPCMCI's ``<->``, either order).
    """

    found: set
    lags: np.ndarray
    score: np.ndarray
    pvalue: np.ndarray
    latent: set = field(default_factory=set)


def _lagged(x, max_lag):
    T = len(x)
    return np.stack([x[max_lag - k : T - k] for k in range(1, max_lag + 1)], axis=1)  # (T', lag, P)


def var_granger(data, max_lag=5, alpha=0.05):
    """
    Pairwise-conditional linear Granger causality from a VAR(`max_lag`) fit.

    For each target, the full model regresses it on ``max_lag`` lags of every process;
    each source's restricted model drops that source's lags. The score is the
    Gaussian transfer entropy :math:`\\frac12 \\log(\\mathrm{RSS}_r / \\mathrm{RSS}_f)`
    in nats :cite:`Barnett2009`, the p-value is the F-test of the restriction, links
    are declared at Benjamini--Hochberg level `alpha` over all pairs
    :cite:`Benjamini1995`, and the delay is the lag with the largest absolute
    standardized coefficient.

    Parameters
    ----------
    data : array_like
        (samples, processes).
    max_lag : int
    alpha : float

    Returns
    -------
    BaselineNetwork
    """
    from .stats import benjamini_hochberg

    x = np.asarray(data, dtype=float)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
    T, P = x.shape
    L = _lagged(x, max_lag)
    n = len(L)
    X = np.column_stack([np.ones(n), L.reshape(n, -1)])  # column 1 + k * P + p is (lag k + 1, process p)
    score = np.zeros((P, P))
    pvalue = np.ones((P, P))
    delay = np.zeros((P, P), dtype=np.int64)
    df_full = n - X.shape[1]
    for t in range(P):
        y = x[max_lag:, t]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss_f = float(np.sum((y - X @ beta) ** 2))
        for s in range(P):
            if s == t:
                continue
            cols = [1 + k * P + s for k in range(max_lag)]
            keep = np.setdiff1d(np.arange(X.shape[1]), cols)
            br, *_ = np.linalg.lstsq(X[:, keep], y, rcond=None)
            rss_r = float(np.sum((y - X[:, keep] @ br) ** 2))
            score[s, t] = 0.5 * np.log(max(rss_r, 1e-300) / max(rss_f, 1e-300))
            F = ((rss_r - rss_f) / max_lag) / (rss_f / df_full) if rss_f > 0 else np.inf
            pvalue[s, t] = float(stats.f.sf(F, max_lag, df_full))
            delay[s, t] = 1 + int(np.argmax(np.abs(beta[cols])))
    pairs = [(s, t) for t in range(P) for s in range(P) if s != t]
    reject, _ = benjamini_hochberg([pvalue[s, t] for s, t in pairs], alpha)
    found = {e for e, ok in zip(pairs, reject, strict=True) if ok}
    return BaselineNetwork(found, delay, score, pvalue)


def tigramite_baseline(data, max_lag=5, method="pcmciplus", alpha=0.01):  # pragma: no cover - optional
    """
    PCMCI+ :cite:`Runge2020` or LPCMCI :cite:`Gerhardus2020` with partial correlation
    on the raw values (needs ``infoflow[causal]``).

    Only lagged links (lag at least 1) are scored. The score of a pair is its largest
    absolute partial correlation over lags and the delay the lag of its smallest
    p-value. PCMCI+ declares a link where some lag is significant at `alpha`; LPCMCI
    where some lagged edge mark is ``-->`` or ``o->`` (``<->`` marks a latent
    confounder, not a link).

    Returns
    -------
    BaselineNetwork
    """
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr

    x = np.asarray(data, dtype=float)
    frame = pp.DataFrame(x)
    test = ParCorr(significance="analytic")
    if method == "lpcmci":
        from tigramite.lpcmci import LPCMCI

        res = LPCMCI(dataframe=frame, cond_ind_test=test, verbosity=0).run_lpcmci(tau_max=max_lag, pc_alpha=alpha)
    elif method == "pcmciplus":
        from tigramite.pcmci import PCMCI

        res = PCMCI(dataframe=frame, cond_ind_test=test, verbosity=0).run_pcmciplus(tau_max=max_lag, pc_alpha=alpha)
    else:
        raise ValueError("method must be 'pcmciplus' or 'lpcmci'")
    val = np.abs(np.nan_to_num(res["val_matrix"][:, :, 1:]))
    p = np.nan_to_num(res["p_matrix"][:, :, 1:], nan=1.0)
    graph = res["graph"][:, :, 1:]
    P = x.shape[1]
    score = val.max(axis=2)
    pvalue = p.min(axis=2)
    delay = 1 + p.argmin(axis=2)
    marks = ("-->", "o->") if method == "lpcmci" else ("-->",)
    linked = np.isin(graph, marks).any(axis=2)
    found = {(s, t) for s in range(P) for t in range(P) if s != t and linked[s, t]}
    bidirected = (res["graph"] == "<->").any(axis=2)
    latent = {(s, t) for s in range(P) for t in range(P) if s != t and (bidirected[s, t] or bidirected[t, s])}
    np.fill_diagonal(score, 0.0)
    return BaselineNetwork(found, delay.astype(np.int64), score, pvalue, latent)


def run_baseline(method, data, max_lag=5, alpha=0.05):
    """
    Run a baseline by name: ``'var'``, ``'pcmciplus'``, or ``'lpcmci'``.
    """
    if method == "var":
        return var_granger(data, max_lag=max_lag, alpha=alpha)
    if method in ("pcmciplus", "lpcmci"):
        return tigramite_baseline(data, max_lag=max_lag, method=method, alpha=alpha)
    raise ValueError(f"unknown baseline {method!r}")
