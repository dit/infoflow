"""
Benchmarks and validations.

*Estimator scaling*: how conditional mutual information estimates degrade with the
size of the conditioning set.

Following the scaling analysis of :cite:`Young2021`, :math:`Y` depends on :math:`X`
with a known mutual information, and the conditioning set :math:`Z` holds `d`
independent nuisance variables, so :math:`I[X : Y \\mid Z] = I[X : Y]` for every `d`
and any drift of an estimate with `d` is estimator error. Uncoupled data
(:math:`I = 0`) measure the spurious dependence that conditioning manufactures.

*Network validation*: random networks with known structure, following the design of
:cite:`Novelli2019` (directed Erdos-Renyi graphs with expected in-degree 3, one random
coupling lag per link, vector-autoregressive or coupled-logistic-map dynamics), scored
by precision, recall, specificity, and lag error; and motif networks whose layers are
known (XOR synergy, a common driver, a chain), scored by how often the right layer,
role, or flag is reported.
"""

import time

import numpy as np
import xarray as xr
from dit.inference._symbols import as_generator

from .measures import cmi_from_joint, dense_codes, joint_table

__all__ = (
    "conditioning_scaling",
    "motif_validation",
    "network_validation",
    "random_network",
    "scaling_data",
    "score_network",
    "simulate_network",
)

ESTIMATORS = ("plugin", "miller_madow", "adaptive", "ksg")


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


def _adaptive(x, y, z, bins):
    """
    Plug-in CMI with the target and source in fixed equal-frequency bins of their ranks
    and the conditioning set in the MDL partition learned with the target.
    """
    from .adaptive import codes, joint_histogram
    from .selection import _equal_frequency, _ranks

    yr, xr = _ranks(y), _ranks(x)
    yc, xc = _equal_frequency(yr, bins), _equal_frequency(xr, bins)
    if z.shape[1]:
        zr = [_ranks(z[:, i]) for i in range(z.shape[1])]
        cuts = joint_histogram([yr, *zr], fixed={0: bins})
        zc = dense_codes(np.stack([codes(c, q) for c, q in zip(zr, cuts[1:], strict=True)], axis=1))[0]
    else:
        zc = np.zeros(len(y), dtype=np.int64)
    return cmi_from_joint(joint_table(dense_codes(yc)[0], dense_codes(xc)[0], dense_codes(zc)[0]), "plugin")


def _estimate(x, y, z, estimator, kind, bins, k, rng):
    if estimator == "adaptive":
        return _adaptive(x, y, z, bins)
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
    estimators : sequence of {'plugin', 'miller_madow', 'adaptive', 'ksg'}
        Plug-in and Miller--Madow estimates use the symbols (Gaussian data are cut
        into `bins` equal-frequency bins per variable); ``'adaptive'`` bins the target
        and source the same way and partitions the conditioning set by MDL together
        with the target (:mod:`infoflow.adaptive`); KSG uses the raw values with `k`
        neighbours.
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


# -- network validation -----------------------------------------------------------------


def random_network(n_nodes, mean_in_degree=3.0, max_lag=5, prng=None):
    """
    A directed Erdos-Renyi graph with a random coupling lag on every link :cite:`Novelli2019`.

    Each ordered pair of distinct nodes is linked with probability
    ``mean_in_degree / n_nodes`` and given a lag drawn uniformly from ``1..max_lag``.

    Returns
    -------
    networkx.DiGraph
        Nodes ``0..n_nodes-1``; each edge has a ``lag`` attribute.
    """
    import networkx as nx

    rng = as_generator(prng)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    p = min(mean_in_degree / n_nodes, 1.0) if n_nodes else 0.0
    for s in range(n_nodes):
        for t in range(n_nodes):
            if s != t and rng.random() < p:
                graph.add_edge(s, t, lag=int(rng.integers(1, max_lag + 1)))
    return graph


def simulate_network(
    graph, n_samples, kind="var", self_coupling=0.5, cross_coupling=0.4, noise=0.1, burn_in=500, prng=None
):
    """
    Simulate the dynamics of :cite:`Novelli2019` on `graph`.

    Every node is driven by its own past with weight `self_coupling` and by each parent
    at that link's lag, the parents' weights equal and summing to `cross_coupling`:
    :math:`a_t = \\beta Y_{t-1} + \\sum_X \\alpha_X X_{t - l_X}`. For ``kind='var'``,
    :math:`Y_t = a_t + \\eta_t`; for ``kind='logistic'``,
    :math:`Y_t = (4 a_t (1 - a_t) + \\eta_t) \\bmod 1`, with :math:`\\eta_t \\sim N(0, \\text{noise}^2)`.

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_nodes).
    """
    rng = as_generator(prng)
    n = graph.number_of_nodes()
    max_lag = max([d["lag"] for _, _, d in graph.edges(data=True)] + [1])
    parents = [[(s, d["lag"]) for s, _, d in graph.in_edges(t, data=True)] for t in range(n)]
    total = n_samples + burn_in
    x = np.zeros((total + max_lag, n))
    x[:max_lag] = rng.random((max_lag, n)) if kind == "logistic" else 0.0
    weights = [cross_coupling / len(ps) if ps else 0.0 for ps in parents]
    for t in range(max_lag, total + max_lag):
        a = self_coupling * x[t - 1]
        for j, ps in enumerate(parents):
            for s, lag in ps:
                a[j] += weights[j] * x[t - lag, s]
        eta = noise * rng.normal(size=n)
        if kind == "var":
            x[t] = a + eta
        elif kind == "logistic":
            x[t] = np.mod(4 * a * (1 - a) + eta, 1.0)
        else:
            raise ValueError("kind must be 'var' or 'logistic'")
    return x[-n_samples:]


def _edges(net, level):
    if level == "skeleton":
        found = {(p, t) for t, sk in net.skeleton.items() if sk.significant for p in sk.parents()}
    else:
        sig = net.dataset["significant"].sel(layer=level).values
        found = {(int(s), int(t)) for s, t in zip(*np.nonzero(sig), strict=True)}
    return found


def score_network(net, graph, levels=("skeleton", "intrinsic")):
    """
    Precision, recall, specificity, and normalized lag error of an inferred network.

    The lag error is the mean absolute difference between true and inferred lags over
    recalled links, divided by its value for two independent uniform lags on the
    graph's lag range (:cite:`Novelli2019`), so 0 is perfect and 1 is chance.

    Returns
    -------
    dict
        ``level -> {"precision", "recall", "specificity", "lag_error", "n_found"}``.
    """
    n = graph.number_of_nodes()
    truth = {(s, t) for s, t in graph.edges()}
    lags = [d["lag"] for _, _, d in graph.edges(data=True)]
    L = max(lags) if lags else 1
    chance = (L**2 - 1) / (3 * L) if L > 1 else 1.0
    out = {}
    for level in levels:
        found = _edges(net, level)
        tp, fp = len(found & truth), len(found - truth)
        fn = len(truth - found)
        tn = n * (n - 1) - tp - fp - fn
        recalled = sorted(found & truth)
        delay = net.dataset["delay"].values
        err = [abs(int(delay[s, t]) - graph.edges[s, t]["lag"]) for s, t in recalled]
        out[level] = {
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "lag_error": float(np.mean(err)) / chance if err else np.nan,
            "n_found": len(found),
        }
    return out


METRICS = ("precision", "recall", "specificity", "lag_error", "n_found", "runtime")


def network_validation(
    kind="var",
    n_nodes=(10,),
    n_samples=(1000,),
    n_reps=3,
    mean_in_degree=3.0,
    max_lag=5,
    levels=("skeleton", "intrinsic"),
    infer_kwargs=None,
    prng=None,
):
    """
    Infer random networks of known structure and score them :cite:`Novelli2019`.

    Parameters
    ----------
    kind : {'var', 'logistic'}
    n_nodes, n_samples : sequences of int
    n_reps : int
        Independent graphs and simulations per (size, length).
    mean_in_degree : float
        0 gives empty networks, where every inferred link is a false positive.
    max_lag : int
        Largest coupling lag, also the inference lag limit.
    levels : sequence of str
        ``'skeleton'`` (selected parents) and/or edge layers such as ``'intrinsic'``.
    infer_kwargs : dict, None
        Passed to :func:`~infoflow.network.infer_multiplex`.
    prng : None, int, Generator

    Returns
    -------
    xarray.Dataset
        One variable per metric over (level, n_nodes, n_samples, rep).
    """
    from .network import infer_multiplex

    rng = as_generator(prng)
    kwargs = {"max_lag": max_lag, "interpret": False, "node_layer": False, **(infer_kwargs or {})}
    shape = (len(levels), len(n_nodes), len(n_samples), n_reps)
    values = {m: np.full(shape, np.nan) for m in METRICS}
    for i, N in enumerate(n_nodes):
        for j, T in enumerate(n_samples):
            for r in range(n_reps):
                graph = random_network(N, mean_in_degree, max_lag, rng)
                data = simulate_network(graph, T, kind, prng=rng)
                t0 = time.perf_counter()
                net = infer_multiplex(data, prng=int(rng.integers(2**32)), **kwargs)
                elapsed = time.perf_counter() - t0
                scores = score_network(net, graph, levels)
                for li, level in enumerate(levels):
                    for m in METRICS[:-1]:
                        values[m][li, i, j, r] = scores[level][m]
                    values["runtime"][li, i, j, r] = elapsed
    dims = ("level", "n_nodes", "n_samples", "rep")
    return xr.Dataset(
        {m: (dims, v) for m, v in values.items()},
        coords={"level": list(levels), "n_nodes": list(n_nodes), "n_samples": list(n_samples), "rep": range(n_reps)},
        attrs={"kind": kind, "mean_in_degree": mean_in_degree, "max_lag": max_lag},
    )


def motif_validation(n_samples=3000, n_reps=5, infer_kwargs=None, prng=None):
    """
    How often each layer, role, and flag is reported correctly on motif networks.

    * ``xor``: the target of an XOR of two sources; both edges should be synergistic.
    * ``common_driver``: z drives x and y; x -> y should carry shared flow only and be
      labelled ``confounded``, z's edges intrinsic.
    * ``chain``: a -> b -> c; a -> c should not be intrinsic and should be ``mediated``.

    Returns
    -------
    xarray.Dataset
        ``rate`` over (check,) with the fraction of replications that pass.
    """
    from . import datasets
    from .network import infer_multiplex

    rng = as_generator(prng)
    kwargs = {"max_lag": 3, **(infer_kwargs or {})}
    checks = {}

    def record(name, ok):
        checks.setdefault(name, []).append(bool(ok))

    for _ in range(n_reps):
        seed = int(rng.integers(2**32))
        ds = infer_multiplex(datasets.xor_synergy(n_samples, seed=seed), prng=seed, **kwargs).dataset
        syn = ds["significant"].sel(layer="synergistic").values
        record("xor: both edges synergistic", syn[0, 2] and syn[1, 2])
        ds = infer_multiplex(datasets.common_driver(n_samples, seed=seed), prng=seed, **kwargs).dataset
        sig = ds["significant"]
        record(
            "common driver: z edges intrinsic",
            sig.sel(layer="intrinsic").values[2, 0] and sig.sel(layer="intrinsic").values[2, 1],
        )
        record(
            "common driver: x->y shared only",
            sig.sel(layer="shared").values[0, 1] and not sig.sel(layer="intrinsic").values[0, 1],
        )
        record("common driver: x->y confounded", ds["role"].values[0, 1] == "confounded")
        ds = infer_multiplex(datasets.chain(n_samples, seed=seed), prng=seed, **kwargs).dataset
        record("chain: a->c not intrinsic", not ds["significant"].sel(layer="intrinsic").values[0, 2])
        record("chain: a->c mediated", ds["role"].values[0, 2] == "mediated")
    names = list(checks)
    return xr.Dataset(
        {"rate": (("check",), [float(np.mean(checks[c])) for c in names])},
        coords={"check": names},
        attrs={"n_reps": n_reps, "n_samples": n_samples},
    )
