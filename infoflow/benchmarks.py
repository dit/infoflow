"""
Estimator benchmarks: how conditional mutual information estimates degrade with the
size of the conditioning set.

Following the scaling analysis of :cite:`Young2021`, :math:`Y` depends on :math:`X`
with a known mutual information, and the conditioning set :math:`Z` holds `d`
independent nuisance variables, so :math:`I[X : Y \\mid Z] = I[X : Y]` for every `d`
and any drift of an estimate with `d` is estimator error. Uncoupled data
(:math:`I = 0`) measure the spurious dependence that conditioning manufactures.
"""

import numpy as np
import xarray as xr
from dit.inference._symbols import as_generator

from .measures import cmi_from_joint, dense_codes, joint_table

__all__ = ("conditioning_scaling", "scaling_data")

ESTIMATORS = ("plugin", "miller_madow", "ksg")


def _h2(p):
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def scaling_data(n, d, kind="discrete", coupled=True, prng=None):
    """
    Samples of :math:`(X, Y, Z)` with :math:`d` independent nuisance variables in :math:`Z`.

    ``kind='discrete'``: :math:`X \\sim \\mathrm{Bernoulli}(0.3)`, :math:`Y` is :math:`X`
    through a binary symmetric channel with flip probability 0.1, and :math:`Z` is
    uniform on :math:`\\{0, 1\\}^d`. ``kind='gaussian'``: :math:`(X, Y)` standard
    bivariate normal with correlation 0.6 and :math:`Z \\sim N(0, I_d)`. Both follow
    :cite:`Young2021`.

    Parameters
    ----------
    n, d : int
    kind : {'discrete', 'gaussian'}
    coupled : bool
        If False, :math:`Y` is drawn independently of :math:`X` (same marginal).
    prng : None, int, Generator

    Returns
    -------
    x, y : np.ndarray
        Shape (n,).
    z : np.ndarray
        Shape (n, d).
    truth : float
        :math:`I[X : Y \\mid Z]` in bits.
    """
    rng = as_generator(prng)
    if kind == "discrete":
        x = (rng.random(n) < 0.3).astype(np.int64)
        source = x if coupled else (rng.random(n) < 0.3).astype(np.int64)
        y = source ^ (rng.random(n) < 0.1)
        z = rng.integers(0, 2, size=(n, d))
        truth = _h2(0.3 * 0.9 + 0.7 * 0.1) - _h2(0.1) if coupled else 0.0
    elif kind == "gaussian":
        rho = 0.6
        x = rng.normal(size=n)
        source = x if coupled else rng.normal(size=n)
        y = rho * source + np.sqrt(1 - rho**2) * rng.normal(size=n)
        z = rng.normal(size=(n, d))
        truth = -0.5 * np.log2(1 - rho**2) if coupled else 0.0
    else:
        raise ValueError("kind must be 'discrete' or 'gaussian'")
    return x, y, z, float(truth)


def _bin(values, bins):
    edges = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1], axis=0)
    if values.ndim == 1:
        return np.searchsorted(edges, values, side="right")
    return np.stack([np.searchsorted(edges[:, j], values[:, j], side="right") for j in range(values.shape[1])], axis=1)


def _estimate(x, y, z, estimator, kind, bins, k, rng):
    if estimator == "ksg":
        from dit.inference import total_correlation_ksg

        data = np.column_stack([x, y, z]).astype(float)
        crvs = list(range(2, data.shape[1])) or None
        return float(total_correlation_ksg(data, [[0], [1]], crvs, k=k, prng=rng))
    if kind == "gaussian":
        x, y, z = _bin(x, bins), _bin(y, bins), _bin(z, bins) if z.shape[1] else z
    yc, _ = dense_codes(y)
    xc, _ = dense_codes(x)
    zc, _ = dense_codes(z)
    return cmi_from_joint(joint_table(yc, xc, zc), estimator)


def conditioning_scaling(
    kind="discrete",
    estimators=ESTIMATORS,
    n_samples=(500, 1000, 2500),
    dims=(0, 1, 2, 4, 8),
    n_reps=10,
    coupled=True,
    bins=4,
    k=4,
    prng=None,
):
    """
    Conditional mutual information estimates against sample size and conditioning dimension.

    Parameters
    ----------
    kind : {'discrete', 'gaussian'}
        See :func:`scaling_data`.
    estimators : sequence of {'plugin', 'miller_madow', 'ksg'}
        Plug-in and Miller--Madow estimates use the symbols (Gaussian data are cut
        into `bins` equal-frequency bins per variable); KSG uses the raw values
        with `k` neighbours.
    n_samples, dims : sequences of int
    n_reps : int
        Independent datasets per (sample size, dimension).
    coupled : bool
        See :func:`scaling_data`.
    bins, k : int
    prng : None, int, Generator

    Returns
    -------
    xarray.Dataset
        ``mean``, ``std``, and ``bias`` (mean minus truth) over (estimator, n, d), in
        bits, with the true value in ``attrs['truth']``.
    """
    rng = as_generator(prng)
    estimators = list(estimators)
    shape = (len(estimators), len(n_samples), len(dims), n_reps)
    values = np.full(shape, np.nan)
    truth = 0.0
    for i, n in enumerate(n_samples):
        for j, d in enumerate(dims):
            for r in range(n_reps):
                x, y, z, truth = scaling_data(n, d, kind, coupled, rng)
                for e, est in enumerate(estimators):
                    values[e, i, j, r] = _estimate(x, y, z, est, kind, bins, k, rng)
    coords = {"estimator": estimators, "n": list(n_samples), "d": list(dims)}
    mean = values.mean(axis=-1)
    return xr.Dataset(
        {
            "mean": (("estimator", "n", "d"), mean),
            "std": (("estimator", "n", "d"), values.std(axis=-1)),
            "bias": (("estimator", "n", "d"), mean - truth),
        },
        coords=coords,
        attrs={"kind": kind, "coupled": int(coupled), "truth": truth, "n_reps": n_reps},
    )
