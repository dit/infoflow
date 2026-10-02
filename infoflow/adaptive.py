"""
Adaptive MDL histograms for conditional mutual information.

Bins are chosen by the minimum description length principle. A histogram with
:math:`K` cells, counts :math:`c` and volumes :math:`V` costs

.. math::

   L = \\sum_{\\text{cells}} c \\log \\frac{n V}{c} + \\log R(n, K) + \\sum_j \\log \\binom{|C'_j|}{|C_j|},

the negative log-likelihood of the histogram density, the parametric complexity of
a :math:`K`-cell multinomial (normalized maximum likelihood), and the cost of naming
dimension :math:`j`'s cut points :math:`C_j` among its candidates :math:`C'_j`
:cite:`Kontkanen2007,Marx2021`. One-dimensional histograms are optimal by dynamic
programming :cite:`Kontkanen2007`. A joint histogram over several dimensions is built
greedily, re-cutting one dimension at a time (by the same dynamic program, with the
other dimensions' bins fixed) and keeping the change that shortens the code most
:cite:`Marx2021`.

For conditional mutual information :math:`I[X : Y \\mid Z]` inside a permutation test,
:cite:`Marx2021`'s joint histogram over :math:`(X, Y, Z)` would adapt to the very
dependence being tested. Here the joint histogram is learned over :math:`(Y, Z)`, the
target and the conditioning set, which is what lets a strongly nonlinear parent in
:math:`Z` be conditioned away finely, and each candidate :math:`X` gets its own
one-dimensional histogram. No partition depends on how :math:`X` is paired with
:math:`(Y, Z)`, so permuting :math:`X` gives a valid null.
"""

from functools import lru_cache

import numpy as np
from scipy.special import gammaln

__all__ = ("joint_histogram", "log_regret", "mdl_histogram")

LOG2 = np.log(2)


@lru_cache(maxsize=64)
def _regret_table(n, k_max):
    """
    :math:`\\log R(n, K)` (nats) for K = 1..k_max by the Kontkanen-Myllymaki recurrence.
    """
    out = np.zeros(k_max + 1)
    if k_max >= 2:
        h = np.arange(n + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = (
                gammaln(n + 1)
                - gammaln(h + 1)
                - gammaln(n - h + 1)
                + np.where(h > 0, h * np.log(h / n), 0.0)
                + np.where(n - h > 0, (n - h) * np.log((n - h) / n), 0.0)
            )
        out[2] = np.logaddexp.reduce(terms)
    for K in range(1, k_max - 1):
        # R(n, K + 2) = R(n, K + 1) + n / K * R(n, K)
        out[K + 2] = np.logaddexp(out[K + 1], np.log(n / K) + out[K])
    return out


def log_regret(n, K):
    """
    The parametric complexity :math:`\\log R(n, K)` of a K-cell multinomial, in nats.
    """
    K = int(K)
    if K <= 1:
        return 0.0
    return float(_regret_table(int(n), max(K, 2))[K])


def _log_binom(a, b):
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)


def _candidates(x, k_init):
    """
    Candidate cut points for a column: equal-width over its range, or between its values if it has few.
    """
    values = np.unique(x)
    if len(values) <= k_init:
        return (values[:-1] + values[1:]) / 2
    return np.linspace(x.min(), x.max(), k_init + 1)[1:-1]


def _segment_dp(counts, widths, n, k_max, rest_cells, n_candidates):
    """
    Best segmentation of consecutive elementary bins (rows of `counts`, one column per
    cell of the other dimensions with its volume folded into `rest_log_volume`).

    Returns the boundaries (indices into the elementary bins) and the score (nats).
    """
    E = counts.shape[0]
    cum = np.vstack([np.zeros((1, counts.shape[1])), np.cumsum(counts, axis=0)])
    wcum = np.concatenate([[0.0], np.cumsum(widths)])
    i, j = np.triu_indices(E + 1, k=1)
    seg = cum[j] - cum[i]  # (segments, rest)
    w = wcum[j] - wcum[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.where(seg > 0, seg * (np.log(n * w)[:, None] - np.log(np.where(seg > 0, seg, 1))), 0.0).sum(1)
    cost = np.full((E + 1, E + 1), np.inf)
    cost[i, j] = ll
    k_max = min(k_max, E)
    best = np.full((k_max + 1, E + 1), np.inf)
    arg = np.zeros((k_max + 1, E + 1), dtype=np.int64)
    best[0, 0] = 0.0
    for m in range(1, k_max + 1):
        total = best[m - 1][:, None] + cost  # (from, to)
        arg[m] = np.argmin(total, axis=0)
        best[m] = total[arg[m], np.arange(E + 1)]
    scores = [best[m, E] + log_regret(n, m * rest_cells) + _log_binom(n_candidates, m - 1) for m in range(1, k_max + 1)]
    m = int(np.argmin(scores)) + 1
    bounds, end = [], E
    for level in range(m, 0, -1):
        start = arg[level, end]
        bounds.append(end)
        end = start
    return sorted(bounds)[:-1], float(scores[m - 1])


def _codes(x, cuts):
    return np.searchsorted(np.asarray(cuts), x, side="right")


def mdl_histogram(x, k_init=None, k_max=None):
    """
    The MDL-optimal one-dimensional histogram of `x` :cite:`Kontkanen2007`.

    Parameters
    ----------
    x : array_like
    k_init : int, None
        Elementary equal-width bins (default ``20 log n``).
    k_max : int, None
        Most bins (default ``5 log n``).

    Returns
    -------
    np.ndarray
        The chosen cut points (bins are ``x <= cuts[0]``, ..., ``x > cuts[-1]``).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    k_init = k_init or max(2, int(20 * np.log(n)))
    k_max = k_max or max(2, int(5 * np.log(n)))
    cand = _candidates(x, k_init)
    if not len(cand):
        return cand
    edges = np.concatenate([[x.min()], cand, [x.max()]])
    widths = np.maximum(np.diff(edges), 1e-12)
    counts = np.bincount(_codes(x, cand), minlength=len(cand) + 1).astype(float)[:, None]
    bounds, _ = _segment_dp(counts, widths, n, k_max, 1, len(cand))
    return cand[np.asarray(bounds, dtype=np.int64) - 1] if bounds else cand[:0]


def joint_histogram(columns, k_init=64, k_max=None, max_iter=None, init_bins=4, fixed=None, initial=None):
    """
    A greedy MDL joint histogram over several columns :cite:`Marx2021`.

    Starting from `init_bins` equal-frequency bins per column, each iteration re-cuts
    every column optimally with the others fixed (possibly down to a single bin) and
    keeps the column whose re-cut shortens the code most, until no re-cut helps or
    `max_iter` iterations have run. Starting from several bins matters for columns with
    uniform margins (e.g. ranks): dependence between two columns is invisible while
    either has a single bin, so a greedy search from one bin each would never start.

    Parameters
    ----------
    columns : sequence of array_like
        Equal-length columns.
    k_init : int, None
        Candidate cut points per column (equal-width over its range; None: ``20 log n``).
    fixed : dict, None
        ``column index -> number of equal-frequency bins`` for columns that are binned
        once and never re-cut (e.g. the target, whose resolution must not depend on how
        informative the other columns are).
    initial : list, None
        Starting cut points per column (None entries start from `init_bins`), e.g. the
        partition of a conditioning set that differs from this one by one variable.

    Returns
    -------
    list of np.ndarray
        Cut points per column.
    """
    fixed = fixed or {}
    cols = [np.asarray(c, dtype=float) for c in columns]
    n = len(cols[0])
    k_init = k_init or max(2, int(20 * np.log(n)))
    k_max = k_max or max(2, int(5 * np.log(n)))
    cands = [_candidates(c, k_init) for c in cols]
    elem = [_codes(c, cand) for c, cand in zip(cols, cands, strict=True)]
    widths = []
    for c, cand in zip(cols, cands, strict=True):
        edges = np.concatenate([[c.min()], cand, [c.max()]])
        widths.append(np.maximum(np.diff(edges), 1e-12))
    # Too fine a starting grid leaves almost every cell empty; start with at most n / 20 cells.
    free_dims = sum(1 for j in range(len(cols)) if j not in fixed and (initial is None or initial[j] is None))
    fixed_cells = int(np.prod([fixed[j] for j in fixed])) if fixed else 1
    if free_dims:
        budget = max(n / 20 / fixed_cells, 1.0)
        init_bins = int(max(1, min(init_bins, np.floor(budget ** (1 / free_dims)))))
    # Too fine a starting grid leaves almost every cell empty; start with at most n / 20 cells.
    free_dims = sum(1 for j in range(len(cols)) if j not in fixed and (initial is None or initial[j] is None))
    fixed_cells = int(np.prod([fixed[j] for j in fixed])) if fixed else 1
    if free_dims:
        budget = max(n / 20 / fixed_cells, 1.0)
        init_bins = int(max(1, min(init_bins, np.floor(budget ** (1 / free_dims)))))
    # Too fine a starting grid leaves almost every cell empty; start with at most n / 20 cells.
    free_dims = sum(1 for j in range(len(cols)) if j not in fixed and (initial is None or initial[j] is None))
    fixed_cells = int(np.prod([fixed[j] for j in fixed])) if fixed else 1
    if free_dims:
        budget = max(n / 20 / fixed_cells, 1.0)
        init_bins = int(max(1, min(init_bins, np.floor(budget ** (1 / free_dims)))))
    # Too fine a starting grid leaves almost every cell empty; start with at most n / 20 cells.
    free_dims = sum(1 for j in range(len(cols)) if j not in fixed and (initial is None or initial[j] is None))
    fixed_cells = int(np.prod([fixed[j] for j in fixed])) if fixed else 1
    if free_dims:
        budget = max(n / 20 / fixed_cells, 1.0)
        init_bins = int(max(1, min(init_bins, np.floor(budget ** (1 / free_dims)))))
    bounds = []  # chosen boundaries in elementary-bin units
    for j, (e, w) in enumerate(zip(elem, widths, strict=True)):
        start = None if initial is None or j in fixed else initial[j]
        if start is not None:
            b = sorted({int(i) + 1 for i in np.searchsorted(cands[j], np.asarray(start))} - {0, len(w)})
        else:
            counts = np.bincount(e, minlength=len(w)).cumsum()
            k = fixed.get(j, init_bins)
            targets = np.arange(1, k) * n / k
            b = sorted({int(v) + 1 for v in np.searchsorted(counts, targets)} - {0, len(w)})
        bounds.append(b)

    def binned(j):
        return np.searchsorted(np.asarray(bounds[j], dtype=np.int64), elem[j], side="right")

    def bin_log_widths(j):
        b = np.concatenate([[0], bounds[j], [len(widths[j])]]).astype(np.int64)
        return np.log(np.add.reduceat(widths[j], b[:-1]))

    def rest_cells_of(j):
        """
        Mixed-radix codes of the other columns' cells, their distinct values, and log volumes.
        """
        others = [r for r in range(len(cols)) if r != j]
        code = np.zeros(n, dtype=np.int64)
        log_vol_full = np.zeros(1)
        for r in others:
            k = len(bounds[r]) + 1
            code = code * k + binned(r)
            log_vol_full = (log_vol_full[:, None] + bin_log_widths(r)[None, :]).ravel()
        keys, inverse = np.unique(code, return_inverse=True)
        return others, inverse.ravel(), len(keys), log_vol_full[keys]

    def score():
        codes = np.stack([binned(j) for j in range(len(cols))], axis=1)
        keys, counts = np.unique(codes, axis=0, return_counts=True)
        logv = np.zeros(len(keys))
        for j in range(len(cols)):
            b = np.concatenate([[0], bounds[j], [len(widths[j])]]).astype(np.int64)
            logv += np.log(np.add.reduceat(widths[j], b[:-1])[keys[:, j]])
        ll = float(np.sum(counts * (np.log(n) + logv - np.log(counts))))
        K = int(np.prod([len(b) + 1 for b in bounds]))
        model = sum(_log_binom(len(c), len(b)) for c, b in zip(cands, bounds, strict=True))
        return ll + log_regret(n, K) + model

    current = score()
    for _ in range(max_iter or 4 * len(cols)):
        best = (current, None, None)
        for j in range(len(cols)):
            if not len(cands[j]) or j in fixed:
                continue
            others, rest_code, n_rest, log_vol = rest_cells_of(j)
            rest_cells = int(np.prod([len(bounds[r]) + 1 for r in others]))
            E = len(widths[j])
            counts = np.bincount(elem[j] * n_rest + rest_code, minlength=E * n_rest).reshape(E, n_rest).astype(float)
            # Fold each rest cell's volume into the likelihood through a per-column offset.
            offset = float(np.sum(counts.sum(0) * log_vol))
            new_bounds, s = _segment_dp(counts, widths[j], n, k_max, rest_cells, len(cands[j]))
            others_model = sum(_log_binom(len(cands[r]), len(bounds[r])) for r in others)
            total = s + offset + others_model
            if total < best[0] - 1e-9:
                best = (total, j, new_bounds)
        if best[1] is None:
            break
        current, j, new_bounds = best
        bounds[j] = list(new_bounds)
    return [cand[np.asarray(b, dtype=np.int64) - 1] if b else cand[:0] for cand, b in zip(cands, bounds, strict=True)]


def codes(x, cuts):
    """
    Integer bin codes of `x` under the cut points `cuts`.
    """
    return _codes(np.asarray(x, dtype=float), cuts)
