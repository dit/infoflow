"""
Higher-order structure from the O-information rate.

The O-information :cite:`Rosas2019`,
:math:`\\Omega(V) = (n - 2) H(V) + \\sum_j [H(V_j) - H(V_{-j})]`, is positive when
dependencies among `n` variables are dominated by redundancy and negative when
dominated by synergy. The O-information rate (OIR) :cite:`Faes2022` replaces the
entropies by entropy rates. With a finite history :math:`L`, the entropy rate of a
set :math:`S` of processes is :math:`H(X^S_t, X^S_{\\text{past}}) - H(X^S_{\\text{past}})`,
and since :math:`\\Omega` is linear in entropies,

.. math::

    \\mathrm{OIR}(S) = \\Omega\\big(\\{(X_{j,t}, X_{j,\\text{past}})\\}_{j \\in S}\\big)
    - \\Omega\\big(\\{X_{j,\\text{past}}\\}_{j \\in S}\\big),

two static O-informations of block variables, computed here with dit on plug-in
distributions. Node and link descriptors are its gradients :cite:`Scagliarini2023`:
the first-order gradient :math:`\\partial_j \\mathrm{OIR} = \\mathrm{OIR}(S) -
\\mathrm{OIR}(S \\setminus j)` is node `j`'s contribution, and the second-order gradient
:math:`\\partial_{ij} \\mathrm{OIR} = \\mathrm{OIR}(S) - \\mathrm{OIR}(S \\setminus i) -
\\mathrm{OIR}(S \\setminus j) + \\mathrm{OIR}(S \\setminus ij)` is the higher-order
interaction of `i` and `j` with the rest, the link-wise representation of
:cite:`Mijatovic2025`. These describe statistical (undirected) structure; they
complement the directed synergistic layer rather than replace it
(:func:`compare_with_layers`).

Plug-in distributions over all processes grow as :math:`(K^{L+1})^n`; keep the
number of processes and the history small.
"""

from itertools import combinations

import dit
import numpy as np
import xarray as xr

from .measures import dense_codes

__all__ = ("compare_with_layers", "o_information_rate", "oir_gradients")


def _aligned(data, processes, history):
    """
    Per process, the dense code of its present and of its last `history` past values.
    """
    lags = {p: [j * int(data.lag_step[p]) for j in range(1, history + 1)] for p in processes}
    start = max(
        [int(data.present_offset[p]) for p in processes]
        + [int(data.past_offset[p]) + max(lags[p], default=0) for p in processes]
    )
    present = {p: [] for p in processes}
    past = {p: [] for p in processes}
    for pa, pr in zip(data.past, data.present, strict=True):
        N = pa.shape[1]
        if start >= N:
            continue
        t = np.arange(start, N)
        for p in processes:
            present[p].append(pr[p, t])
            past[p].append(np.stack([pa[p, t - lag] for lag in lags[p]], axis=1))
    present = {p: np.concatenate(v) for p, v in present.items()}
    past = {p: dense_codes(np.concatenate(v))[0] for p, v in past.items()}
    return present, past


def _omega(columns):
    if len(columns) < 3:
        return 0.0
    codes = np.stack(columns, axis=1)
    keys, counts = np.unique(codes, axis=0, return_counts=True)
    d = dit.Distribution([tuple(int(v) for v in k) for k in keys], counts / counts.sum())
    return float(dit.multivariate.o_information(d))


def _as_data(data, preprocess, history, prng):
    from .network import _as_discrete

    return _as_discrete(data, None, preprocess, history, prng)[0]


class _Rates:
    def __init__(self, data, processes, history):
        self.present, self.past = _aligned(data, processes, history)
        self.block = {p: dense_codes(np.stack([self.present[p], self.past[p]], axis=1))[0] for p in processes}
        self.cache = {}

    def __call__(self, subset):
        subset = tuple(sorted(subset))
        if subset not in self.cache:
            blocks = [self.block[p] for p in subset]
            pasts = [self.past[p] for p in subset]
            self.cache[subset] = _omega(blocks) - _omega(pasts)
        return self.cache[subset]


def o_information_rate(data, processes=None, history=1, preprocess="auto", prng=None):
    """
    The O-information rate of a set of processes, in bits (positive: redundancy-dominated,
    negative: synergy-dominated) :cite:`Faes2022`.

    Parameters
    ----------
    data : array_like, Trials, or DiscreteData
        Raw data are preprocessed as by :func:`~infoflow.network.infer_multiplex`.
    processes : sequence of int, None
        Default: all.
    history : int
        Past values per process.
    """
    d = _as_data(data, preprocess, history, prng)
    processes = list(range(d.n_processes)) if processes is None else list(processes)
    return _Rates(d, processes, history)(processes)


def oir_gradients(data, processes=None, history=1, preprocess="auto", prng=None):
    """
    The O-information rate with its first- and second-order gradients :cite:`Scagliarini2023`.

    Returns
    -------
    xarray.Dataset
        ``oir`` (scalar), ``gradient`` over (node,), and ``link`` over (source, target)
        (symmetric, zero diagonal), in bits.
    """
    d = _as_data(data, preprocess, history, prng)
    processes = list(range(d.n_processes)) if processes is None else list(processes)
    rates = _Rates(d, processes, history)
    full = rates(processes)
    n = len(processes)
    gradient = np.array([full - rates([q for q in processes if q != p]) for p in processes])
    link = np.zeros((n, n))
    for a, b in combinations(range(n), 2):
        i, j = processes[a], processes[b]
        rest = [q for q in processes if q not in (i, j)]
        value = full - rates([q for q in processes if q != i]) - rates([q for q in processes if q != j]) + rates(rest)
        link[a, b] = link[b, a] = value
    names = [d.names[p] for p in processes]
    return xr.Dataset(
        {
            "oir": ((), full),
            "gradient": (("node",), gradient),
            "link": (("source", "target"), link),
        },
        coords={"node": names, "source": names, "target": names},
        attrs={"history": history, "units": "bits"},
    )


def compare_with_layers(network, data, history=1, preprocess="auto", prng=None):
    """
    Second-order OIR gradients next to the synergistic and shared layers of an inferred network.

    For each pair of processes, the undirected link gradient (negative: the pair's
    dependence with the rest is synergy-dominated) is reported with the larger of the
    two directed synergistic and shared flows between them.

    Returns
    -------
    xarray.Dataset
        ``link``, ``synergistic``, ``shared`` over (source, target).
    """
    grads = oir_gradients(data, history=history, preprocess=preprocess, prng=prng)
    ds = network.dataset
    out = {"link": (("source", "target"), grads["link"].values)}
    for layer in ("synergistic", "shared"):
        w = np.nan_to_num(ds["weight"].sel(layer=layer).values)
        out[layer] = (("source", "target"), np.maximum(w, w.T))
    names = list(ds.coords["source"].values)
    return xr.Dataset(out, coords={"source": names, "target": names})
