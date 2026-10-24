"""
Stability selection of network links with e-value false-discovery-rate control.

Parent selection on one realization is unstable for weak links, and its permutation
p-values are computed after selection on the same data. Following time-series
knockoffs :cite:`Chi2025`, which handle serial dependence by selecting on
subsamples and aggregating with e-values :cite:`Vovk2021`, and derandomized
knockoffs :cite:`Ren2024`:

1. Each of `n_subsamples` subsamples is a contiguous block of `fraction` of every
   trial (contiguous, so serial dependence is preserved), split in time into a
   selection half and a testing half.
2. The skeleton is selected on the selection half; each selected parent's transfer
   entropy (given the target's selected past and its other selected parents) is
   permutation-tested on the testing half, so its p-value is valid despite the
   selection. Unselected pairs get p-value 1.
3. Benjamini--Hochberg at level ``alpha_bh`` over all ``m`` candidate links gives a
   set :math:`S_b` and the e-values :math:`e_{jb} = m\\, 1[j \\in S_b] / (\\alpha_{bh}
   |S_b|)`, whose null sum has expectation at most ``m`` :cite:`Wang2022`.
4. The e-values are averaged over subsamples (averages of e-values remain e-values
   under any dependence :cite:`Vovk2021`) and the e-BH procedure at level `alpha`
   selects the final links :cite:`Wang2022`.

A link is then kept roughly when it is selected in at least a fraction
``alpha_bh / alpha`` of the subsamples (one half by default), which is stability
selection :cite:`Meinshausen2010` with a threshold set by the FDR level. The
selection frequency is a continuous edge score for rank metrics.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from dit.inference._symbols import as_generator

from .selection import SkeletonSettings, _estimators, _make_columns, _make_perm, _pvalue, infer_skeleton
from .stats import benjamini_hochberg

__all__ = ("StabilityResult", "e_bh", "stability_selection")


@dataclass
class StabilityResult:
    """
    Stability-selection results over (source, target).

    Attributes
    ----------
    dataset : xarray.Dataset
        ``frequency`` (fraction of subsamples selecting the link), ``evalue``
        (averaged e-value), ``selected`` (e-BH), and ``pvalues`` over
        (subsample, source, target).
    selected : set of (int, int)
    """

    dataset: xr.Dataset
    selected: set


def e_bh(evalues, alpha):
    """
    The e-BH procedure :cite:`Wang2022`: reject the :math:`k^*` hypotheses with the
    largest e-values, :math:`k^* = \\max\\{k : e_{(k)} \\ge m / (\\alpha k)\\}`.

    Returns
    -------
    numpy.ndarray of bool
    """
    e = np.asarray(evalues, dtype=float)
    m = e.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(-e)
    ok = e[order] >= m / (alpha * np.arange(1, m + 1))
    k = int(np.max(np.nonzero(ok)[0]) + 1) if ok.any() else 0
    reject = np.zeros(m, dtype=bool)
    reject[order[:k]] = True
    return reject


def _holdout_pvalue(data, sk, p, settings, n_perm, rng):
    base = list(sk.conditionals) + list(sk.target_past)
    x_vars = sk.source_variables(p)
    others = [v for v in sk.sources if v[0] != p]
    variables = list(dict.fromkeys(base + others + x_vars))
    name = _estimators(settings, sk.target)[0]
    cols = _make_columns(name, data, sk.target, variables, max(v[1] for v in variables), settings, rng)
    if cols.n < 10:
        return 1.0
    x, Kx = cols.joint(x_vars)
    z, Kz = cols.context(base + others)
    observed = cols.cmi(x, Kx, z, Kz)
    perm = _make_perm(cols, settings, rng)
    null = cols.cmi_batch(x, Kx, z, Kz, [perm(z) for _ in range(n_perm)])
    return _pvalue(null, observed)


def stability_selection(
    data,
    n_subsamples=20,
    fraction=0.5,
    alpha=0.1,
    alpha_bh=None,
    embeddings=None,
    settings=None,
    n_perm=200,
    max_lag=None,
    preprocess="auto",
    names=None,
    targets=None,
    map_fn=map,
    prng=None,
):
    """
    Links selected stably across subsamples, with e-value FDR control (module docstring).

    Parameters
    ----------
    data : array_like, Trials, or DiscreteData
        As for :func:`~infoflow.network.infer_multiplex`; raw data are preprocessed once.
    n_subsamples : int
    fraction : float
        Length of each subsample as a fraction of every trial (half selects, half tests).
    alpha : float
        Final FDR level (e-BH).
    alpha_bh : float, None
        Per-subsample BH level (default ``alpha / 2`` :cite:`Ren2024`).
    embeddings, settings, max_lag, preprocess, names, targets, map_fn :
        As for :func:`~infoflow.network.infer_multiplex`.
    n_perm : int
        Permutations of each holdout test.
    prng : None, int, Generator

    Returns
    -------
    StabilityResult
    """
    from .embedding import Embedding
    from .network import _as_discrete

    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    rng = as_generator(prng)
    alpha_bh = alpha / 2 if alpha_bh is None else alpha_bh
    discrete, chosen, _ = _as_discrete(data, names, preprocess, max_lag, rng)
    if embeddings is None:
        embeddings = chosen if chosen is not None else Embedding(max_lag=max_lag or 3)
    settings = settings or SkeletonSettings()
    P = discrete.n_processes
    targets = list(range(P)) if targets is None else list(targets)
    pairs = [(s, t) for t in targets for s in range(P) if s != t]
    m = len(pairs)
    pvalues = np.ones((n_subsamples, P, P))
    chosen_mask = np.zeros((n_subsamples, P, P), dtype=bool)
    evalues = np.zeros((n_subsamples, P, P))
    for b in range(n_subsamples):
        lo = float(rng.uniform(0, 1 - fraction)) if fraction < 1 else 0.0
        block = discrete.window(lo, lo + fraction)
        select, test = block.split_time(0.5)
        skeleton = infer_skeleton(select, embeddings, targets, settings, rng, map_fn=map_fn)
        for t, sk in skeleton.items():
            for p in sk.parents():
                pvalues[b, p, t] = _holdout_pvalue(test, sk, p, settings, n_perm, rng)
        reject, _ = benjamini_hochberg([pvalues[b, s, t] for s, t in pairs], alpha_bh)
        k = int(reject.sum())
        for (s, t), ok in zip(pairs, reject, strict=True):
            if ok:
                chosen_mask[b, s, t] = True
                evalues[b, s, t] = m / (alpha_bh * k)
    mean_e = evalues.mean(axis=0)
    frequency = chosen_mask.mean(axis=0)
    final = e_bh([mean_e[s, t] for s, t in pairs], alpha)
    selected = {e for e, ok in zip(pairs, final, strict=True) if ok}
    sel = np.zeros((P, P), dtype=bool)
    for s, t in selected:
        sel[s, t] = True
    unscored = np.ones((P, P), dtype=bool)
    for s, t in pairs:
        unscored[s, t] = False
    ds = xr.Dataset(
        {
            "frequency": (("source", "target"), np.where(unscored, np.nan, frequency)),
            "evalue": (("source", "target"), np.where(unscored, np.nan, mean_e)),
            "selected": (("source", "target"), sel),
            "pvalues": (("subsample", "source", "target"), pvalues),
        },
        coords={"source": discrete.names, "target": discrete.names, "subsample": range(n_subsamples)},
        attrs={"alpha": alpha, "alpha_bh": alpha_bh, "fraction": fraction, "n_subsamples": n_subsamples},
    )
    return StabilityResult(ds, selected)
