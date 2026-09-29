"""
Candidate delays for delay reconstruction.

Three nonparametric heuristics propose delays; the joint discretizer and embedding
selection then chooses among them by predictive score.

* the first minimum of the auto-mutual information (with equal-frequency bins);
* the first minimum of the mean Menger curvature of the two-dimensional
  reconstruction :cite:`Deshmukh2020` (too-small delays flatten the reconstruction
  along the diagonal and too-large delays fold it, both raising the curvature).
  With strong observational noise, consecutive points differ mostly by noise and
  the curve flattens; the candidate is then uninformative but harmless, since the
  predictive score decides;
* the first decorrelation time of the autocorrelation (below 1/e).
"""

import numpy as np

__all__ = ("autocorrelation_delay", "auto_mi_delay", "curvature_delay", "delay_candidates")


def _trials(series):
    return [np.asarray(s, dtype=float) for s in (series if isinstance(series, list) else [series])]


def _first_minimum(values, offset=1):
    values = np.asarray(values)
    for k in range(1, len(values) - 1):
        if values[k] < values[k - 1] and values[k] <= values[k + 1]:
            return k + offset
    return int(np.argmin(values)) + offset


def auto_mi_delay(series, max_delay=20, bins=8):
    """
    The first minimum of the auto-mutual information ``I[x_t : x_{t+tau}]``.
    """
    from dit.inference import conditional_mutual_information

    trials = _trials(series)
    pooled = np.concatenate(trials)
    edges = np.quantile(pooled, np.linspace(0, 1, bins + 1)[1:-1])
    codes = [np.searchsorted(edges, t, side="right") for t in trials]
    mis = []
    for tau in range(1, max_delay + 1):
        a = np.concatenate([c[:-tau] for c in codes if len(c) > tau])
        b = np.concatenate([c[tau:] for c in codes if len(c) > tau])
        mis.append(conditional_mutual_information(a, b, estimator="miller_madow") if len(a) else 0.0)
    return _first_minimum(mis), np.array(mis)


def curvature_delay(series, max_delay=20):
    """
    The first minimum of the mean Menger curvature of ``(x_t, x_{t+tau})`` :cite:`Deshmukh2020`.
    """
    trials = _trials(series)
    scale = np.std(np.concatenate(trials)) or 1.0
    means = []
    for tau in range(1, max_delay + 1):
        values = []
        for t in trials:
            z = (t - t.mean()) / scale
            if len(z) <= tau + 2:
                continue
            pts = np.stack([z[:-tau], z[tau:]], axis=1)
            a, b, c = pts[:-2], pts[1:-1], pts[2:]
            ab, bc, ca = np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)
            area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            denom = ab * bc * ca
            ok = denom > 1e-12
            values.append(2 * area2[ok] / denom[ok])
        means.append(np.mean(np.concatenate(values)) if values else np.nan)
    means = np.nan_to_num(np.array(means), nan=np.inf)
    # A light moving average keeps sampling noise from creating spurious first minima.
    smooth = np.convolve(np.pad(means, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    return _first_minimum(smooth), means


def autocorrelation_delay(series, max_delay=20):
    """
    The first delay at which the autocorrelation drops below 1/e.
    """
    trials = _trials(series)
    acf = []
    for tau in range(1, max_delay + 1):
        a = np.concatenate([t[:-tau] - t.mean() for t in trials if len(t) > tau])
        b = np.concatenate([t[tau:] - t.mean() for t in trials if len(t) > tau])
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        acf.append(np.sum(a * b) / denom if denom > 0 else 0.0)
    acf = np.array(acf)
    below = np.flatnonzero(acf < 1 / np.e)
    return (int(below[0]) + 1 if len(below) else max_delay), acf


def _rank_transform(series):
    """
    Pooled ranks scaled to (0, 1), per trial: the copula transform.
    """
    from scipy.stats import rankdata

    trials = _trials(series)
    pooled = np.concatenate(trials)
    # Average ranks keep tied values tied, so binning ranks matches binning values.
    ranks = (rankdata(pooled) - 0.5) / len(pooled)
    out, start = [], 0
    for t in trials:
        out.append(ranks[start : start + len(t)])
        start += len(t)
    return out


def delay_candidates(series, max_delay=20):
    """
    The distinct candidate delays from the three heuristics, plus 1.

    All heuristics are computed on the rank (copula) transform of the series, so the
    candidates, like the rank-based discretizers, are invariant to monotone
    transforms of the channel. (The curvature criterion is geometric, so on ranks it
    can differ from its value on the raw series; the predictive score decides.)

    Returns
    -------
    candidates : list of int
    details : dict
        Each heuristic's choice and curve.
    """
    series = _rank_transform(series)
    mi, mi_curve = auto_mi_delay(series, max_delay)
    curv, curv_curve = curvature_delay(series, max_delay)
    acf, acf_curve = autocorrelation_delay(series, max_delay)
    details = {
        "auto_mi": {"delay": mi, "curve": mi_curve},
        "curvature": {"delay": curv, "curve": curv_curve},
        "autocorrelation": {"delay": acf, "curve": acf_curve},
    }
    return sorted({1, mi, curv, acf}), details
