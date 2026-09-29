"""
Surrogate significance tests, bootstrap confidence intervals, and multiple-testing
control.
"""

from dataclasses import dataclass

import numpy as np
from dit.inference import stationary_bootstrap, total_correlation_ksg
from dit.inference._symbols import as_generator, is_trials
from dit.inference.estimators import _check_joint, _cmi_codes, _dense
from scipy.spatial import cKDTree

__all__ = (
    "SurrogateTest",
    "benjamini_hochberg",
    "bootstrap_ci",
    "conditional_mutual_information_test",
    "conditional_mutual_information_test_knn",
)


@dataclass(frozen=True)
class SurrogateTest:
    """
    The result of a surrogate significance test.

    Attributes
    ----------
    value : float
        The observed statistic.
    pvalue : float
        ``(1 + #{null >= value}) / (1 + len(null))``.
    null : np.ndarray
        The statistic evaluated on each surrogate.
    n_samples : int or None
        The number of aligned samples (windows) the statistic was computed from.
    """

    value: float
    pvalue: float
    null: np.ndarray
    n_samples: int | None = None


def _within_strata(x, z, rng):
    """
    Permute `x` uniformly within each stratum of `z`.
    """
    grouped = np.argsort(z, kind="stable")
    shuffled = np.lexsort((rng.random(len(z)), z))
    out = np.empty_like(x)
    out[grouped] = x[shuffled]
    return out


def conditional_mutual_information_test(x, y, z=None, n_surrogates=1000, estimator="plugin", prng=None):
    """
    Test :math:`I[X : Y \\mid Z] = 0` by permuting `x` within strata of `z`.

    Parameters
    ----------
    x, y : array_like
        Samples of each variable; rows of 2D arrays are joint outcomes.
    z : array_like, None
        Samples of the conditioning variable. If None, `x` is permuted freely.
    n_surrogates : int
        The number of permutations.
    estimator : str
        See :func:`~dit.inference.entropy_from_counts`.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    result : SurrogateTest

    Notes
    -----
    Permuting within strata preserves the joint distributions of :math:`(X, Z)`
    and :math:`(Y, Z)` while breaking any dependence between :math:`X` and
    :math:`Y` given :math:`Z`. The test is exact for exchangeable samples; for
    serially dependent samples it is approximate.
    """
    rng = as_generator(prng)
    x, y = _dense(x), _dense(y)
    z = np.zeros_like(x) if z is None else _dense(z)
    _check_joint(x, y, z)
    value = _cmi_codes(x, y, z, estimator)
    null = np.array([_cmi_codes(_within_strata(x, z, rng), y, z, estimator) for _ in range(n_surrogates)])
    return _result(value, null, len(x))


def _result(value, null, n_samples=None):
    tol = 1e-12 * max(1.0, abs(value))
    pvalue = float((1 + np.sum(null >= value - tol)) / (1 + len(null)))
    return SurrogateTest(float(value), pvalue, null, n_samples)


def bootstrap_ci(data, statistic, n_boot=1000, confidence=0.95, mean_block_length=None, method="percentile", prng=None):
    """
    A confidence interval for `statistic` under the stationary bootstrap.

    Parameters
    ----------
    data : array_like
        The series (along the first axis). For statistics of several series,
        stack them as columns so they are resampled jointly.
    statistic : callable
        Maps a series shaped like `data` to a float, e.g.
        ``lambda d: transfer_entropy(d[:, 0], d[:, 1])``.
    n_boot : int
        The number of resamples.
    confidence : float
        The coverage of the interval.
    mean_block_length : float, None
        See :func:`stationary_bootstrap`.
    method : {'basic', 'percentile'}
        ``'percentile'`` returns quantiles of the resampled statistic.
        ``'basic'`` reflects them about the observed value,
        :math:`(2\\hat\\theta - q_{hi}, 2\\hat\\theta - q_{lo})`, which corrects
        for the shift between resampled and observed estimates.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    low, high : float
        The interval endpoints.

    Notes
    -----
    Every block junction in a resampled series creates length-``L`` words that
    straddle two unrelated positions, so statistics built from lagged windows
    (block entropies, transfer entropy) are biased toward independence unless
    `mean_block_length` is much larger than ``L``. For transfer entropy prefer
    :func:`transfer_entropy_ci`, which resamples the aligned windows instead.
    """
    if not is_trials(data):
        data = np.asarray(data)
    samples = [statistic(r) for r in stationary_bootstrap(data, n_boot, mean_block_length, prng)]
    return _interval(samples, statistic(data) if method == "basic" else None, confidence, method)


def _interval(samples, value, confidence, method):
    if method not in ("basic", "percentile"):
        raise ValueError(f"Unknown method {method!r}.")
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    if method == "basic":
        low, high = 2 * value - high, 2 * value - low
    return float(low), float(high)


def benjamini_hochberg(pvalues, alpha=0.05, dependent=False):
    """
    Control the false discovery rate over a fixed family of tests.

    Parameters
    ----------
    pvalues : array_like
        One p-value per test, e.g. every pairwise :func:`transfer_entropy_test`
        in a network.
    alpha : float
        The target false discovery rate.
    dependent : bool
        If False, the Benjamini–Hochberg step-up procedure, valid for
        independent or positively dependent tests :cite:`Benjamini1995`. If
        True, the Benjamini–Yekutieli correction, valid under any dependence
        :cite:`Benjamini2001`.

    Returns
    -------
    reject : np.ndarray of bool
        Which hypotheses are rejected.
    adjusted : np.ndarray
        Adjusted p-values (q-values); ``reject == (adjusted <= alpha)``.

    Notes
    -----
    The family must be fixed before looking at the results. Procedures that
    choose later tests from earlier outcomes, like CSSR's state splitting, are
    not covered.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p)
    scale = np.sum(1.0 / np.arange(1, m + 1)) if dependent else 1.0
    ranked = p[order] * m * scale / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted <= alpha, adjusted


def _local_permutation(x, z, k_perm, rng):
    """
    Runge's local permutation: each sample takes the `x` of a distinct sample among
    its `k_perm` nearest neighbors in `z`, so the dependence of `x` on `z` survives.
    """
    n = len(x)
    if z.shape[1] == 0:
        return x[rng.permutation(n)]
    k_perm = min(k_perm, n)
    neighbors = cKDTree(z).query(z, k_perm, p=np.inf)[1].reshape(n, -1)
    used = np.zeros(n, dtype=bool)
    choice = np.empty(n, dtype=np.int64)
    for i in rng.permutation(n):
        candidates = neighbors[i][rng.permutation(neighbors.shape[1])]
        free = candidates[~used[candidates]]
        j = free[0] if len(free) else candidates[0]
        used[j] = True
        choice[i] = j
    return x[choice]


def conditional_mutual_information_test_knn(
    data, rvs, crvs=None, k=4, k_perm=5, n_surrogates=200, noise=1e-10, prng=None
):
    """
    Test :math:`I[X : Y \\mid Z] = 0` for continuous data with the KSG estimator
    and local-permutation surrogates :cite:`Runge2018`.

    Each surrogate replaces :math:`X` in sample :math:`i` by the :math:`X` of a
    (mostly distinct) sample among the `k_perm` nearest neighbors of :math:`i` in
    :math:`Z`. That keeps the dependence of :math:`X` on :math:`Z` while breaking
    any further dependence on :math:`Y`. It is the continuous analogue of
    :func:`~dit.inference.conditional_mutual_information_test`, which permutes
    within exact strata of a discrete :math:`Z`.

    Parameters
    ----------
    data : np.ndarray
        Samples, one per row.
    rvs : list of two lists
        The columns of :math:`X` and of :math:`Y`.
    crvs : list, None
        The columns of :math:`Z`. If None, :math:`X` is permuted freely.
    k : int
        Nearest neighbors for the KSG estimate.
    k_perm : int
        Neighborhood size for the local permutation; small values (5–10) keep the
        null conditional on :math:`Z`.
    n_surrogates : int
        The number of surrogates.
    noise : float
        Symmetry-breaking noise for the KSG estimator.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    result : SurrogateTest
    """
    rng = as_generator(prng)
    data = np.asarray(data, dtype=np.float64)
    x_cols, y_cols = (list(r) for r in rvs)
    crvs = [] if crvs is None else list(crvs)
    value = total_correlation_ksg(data, [x_cols, y_cols], crvs, k=k, noise=noise, prng=rng)
    z = data[:, crvs]
    null = np.empty(n_surrogates)
    for i in range(n_surrogates):
        shuffled = data.copy()
        shuffled[:, x_cols] = _local_permutation(data[:, x_cols], z, k_perm, rng)
        null[i] = total_correlation_ksg(shuffled, [x_cols, y_cols], crvs, k=k, noise=noise, prng=rng)
    return _result(value, null, len(data))
