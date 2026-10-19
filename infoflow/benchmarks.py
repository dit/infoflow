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
    "latent_validation",
    "lorenz96_graph",
    "motif_validation",
    "network_validation",
    "random_network",
    "rank_metrics",
    "scaling_data",
    "score_edges",
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


def lorenz96_graph(n_vars):
    """
    The ground-truth graph of :func:`~infoflow.datasets.lorenz96`: each variable's
    parents are ``i - 2``, ``i - 1``, and ``i + 1`` on the ring, at lag 1 :cite:`Tank2022`.

    Returns
    -------
    networkx.DiGraph
    """
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_vars))
    for i in range(n_vars):
        for offset in (-2, -1, 1):
            s = (i + offset) % n_vars
            if s != i:
                graph.add_edge(s, i, lag=1)
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


def _edges(net, level, targets=None):
    if level == "skeleton":
        found = {(p, t) for t, sk in net.skeleton.items() if sk.significant for p in sk.parents()}
    else:
        sig = net.dataset["significant"].sel(layer=level).values
        found = {(int(s), int(t)) for s, t in zip(*np.nonzero(sig), strict=True)}
    if targets is not None:
        found = {e for e in found if e[1] in targets}
    return found


def rank_metrics(score, truth):
    """
    AUROC and AUPRC (average precision) of a continuous edge score.

    Rank-based scores do not depend on a significance threshold, so methods with
    different thresholds can be compared :cite:`Cheng2024,Stein2025`. Missing scores
    (NaN) rank below every observed score; ties share their average rank.

    Parameters
    ----------
    score : array_like
        One score per candidate link.
    truth : array_like of bool
        Whether each candidate link is real.

    Returns
    -------
    tuple of float
        ``(auroc, auprc)``; NaN if every score is missing or one class is empty.
    """
    from scipy.stats import rankdata

    score = np.asarray(score, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    if not n_pos or not n_neg or np.all(np.isnan(score)):
        return np.nan, np.nan
    s = np.where(np.isnan(score), np.nanmin(score) - 1.0, score)
    ranks = rankdata(s)
    auroc = (ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # Average precision with ties broken pessimistically (negatives first).
    order = np.lexsort((truth, -s))
    hits = np.cumsum(truth[order])
    precision = hits / np.arange(1, len(s) + 1)
    auprc = float(np.sum(precision[truth[order]]) / n_pos)
    return float(auroc), auprc


def score_edges(found, delay, score, graph, targets=None):
    """
    Score a set of inferred links against `graph`.

    Parameters
    ----------
    found : set of (int, int)
        Inferred ``(source, target)`` links.
    delay : array_like, None
        ``(source, target)`` inferred lags (for the lag error).
    score : array_like, None
        ``(source, target)`` continuous edge scores (for AUROC and AUPRC).
    graph : networkx.DiGraph
        Ground truth with a ``lag`` attribute per edge.
    targets : iterable of int, None
        Score only links into these targets.

    Returns
    -------
    dict
        ``precision``, ``recall``, ``specificity``, ``lag_error``, ``auroc``, ``auprc``, ``n_found``.
    """
    n = graph.number_of_nodes()
    targets = set(range(n)) if targets is None else set(targets)
    found = {e for e in found if e[1] in targets}
    truth = {(s, t) for s, t in graph.edges() if t in targets and s != t}
    lags = [d["lag"] for _, _, d in graph.edges(data=True) if "lag" in d]
    L = max(lags) if lags else 1
    chance = (L**2 - 1) / (3 * L) if L > 1 else 1.0
    tp, fp = len(found & truth), len(found - truth)
    fn = len(truth - found)
    tn = len(targets) * (n - 1) - tp - fp - fn
    err = []
    if delay is not None:
        delay = np.asarray(delay)
        err = [
            abs(int(delay[s, t]) - graph.edges[s, t]["lag"])
            for s, t in sorted(found & truth)
            if "lag" in graph.edges[s, t]
        ]
    auroc = auprc = np.nan
    if score is not None:
        score = np.asarray(score, dtype=float)
        pairs = [(s, t) for t in sorted(targets) for s in range(n) if s != t]
        auroc, auprc = rank_metrics([score[s, t] for s, t in pairs], [p in truth for p in pairs])
    return {
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "lag_error": float(np.mean(err)) / chance if err else np.nan,
        "auroc": auroc,
        "auprc": auprc,
        "n_found": len(found),
    }


def score_network(net, graph, levels=("skeleton", "intrinsic"), targets=None):
    """
    Precision, recall, specificity, normalized lag error, AUROC, and AUPRC of an inferred network.

    The lag error is the mean absolute difference between true and inferred lags over
    recalled links, divided by its value for two independent uniform lags on the
    graph's lag range (:cite:`Novelli2019`), so 0 is perfect and 1 is chance. AUROC
    and AUPRC rank every candidate link by the dataset's ``score``
    (``SkeletonSettings(edge_scores=True)``; NaN otherwise) and are the same for every
    level (:func:`rank_metrics`). With `targets`, only links into those targets are scored.

    Returns
    -------
    dict
        ``level -> {"precision", "recall", "specificity", "lag_error", "auroc", "auprc", "n_found"}``.
    """
    ds = net.dataset
    score = ds["score"].values if "score" in ds else None
    return {
        level: score_edges(_edges(net, level, targets), ds["delay"].values, score, graph, targets) for level in levels
    }


METRICS = ("precision", "recall", "specificity", "lag_error", "auroc", "auprc", "n_found", "runtime")


def network_validation(
    kind="var",
    n_nodes=(10,),
    n_samples=(1000,),
    n_reps=3,
    mean_in_degree=3.0,
    max_lag=5,
    levels=("skeleton", "intrinsic"),
    infer_kwargs=None,
    n_targets=None,
    methods=("infoflow",),
    baseline_alpha=0.05,
    edge_scores=True,
    prng=None,
):
    """
    Infer random networks of known structure and score them :cite:`Novelli2019`.

    Every method sees the same graphs and simulations. Besides infoflow, `methods` may
    name the baselines of :mod:`infoflow.baselines` (``'var'``, ``'pcmciplus'``,
    ``'lpcmci'``), which a fair comparison needs: simple vector autoregression is often
    the strongest method on real benchmarks :cite:`Stein2025`. Baselines have only
    the ``skeleton`` level; their other levels are NaN.

    Parameters
    ----------
    kind : {'var', 'logistic', 'lorenz96'}
        Novelli's random networks simulated as a VAR or coupled logistic maps, or the
        Lorenz-96 ring (:func:`~infoflow.datasets.lorenz96`, forcing 10;
        `mean_in_degree` and `max_lag` then do not shape the graph).
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
    n_targets : int, None
        Infer and score only this many randomly chosen targets per network (links into
        them), which keeps slow estimators tractable on large networks.
    methods : sequence of str
        ``'infoflow'`` and/or baseline names.
    baseline_alpha : float
        Significance level of the baselines.
    edge_scores : bool
        Run infoflow with ``SkeletonSettings(edge_scores=True)`` so that its AUROC and
        AUPRC are defined (at the cost of a few permutations per candidate).
    prng : None, int, Generator

    Returns
    -------
    xarray.Dataset
        One variable per metric over (algorithm, level, n_nodes, n_samples, rep);
        the dimension is not called ``method``, which ``Dataset.sel`` reserves.
    """
    import dataclasses

    from .baselines import run_baseline
    from .datasets import lorenz96
    from .network import infer_multiplex
    from .selection import SkeletonSettings

    if kind not in ("var", "logistic", "lorenz96"):
        raise ValueError("kind must be 'var', 'logistic', or 'lorenz96'")
    rng = as_generator(prng)
    kwargs = {"max_lag": max_lag, "interpret": False, "node_layer": False, **(infer_kwargs or {})}
    if edge_scores:
        kwargs["skeleton_settings"] = dataclasses.replace(
            kwargs.get("skeleton_settings") or SkeletonSettings(), edge_scores=True
        )
    methods = list(methods)
    shape = (len(methods), len(levels), len(n_nodes), len(n_samples), n_reps)
    values = {m: np.full(shape, np.nan) for m in METRICS}
    for i, N in enumerate(n_nodes):
        for j, T in enumerate(n_samples):
            for r in range(n_reps):
                if kind == "lorenz96":
                    graph = lorenz96_graph(N)
                    data = lorenz96(T, n_vars=N, seed=int(rng.integers(2**32)))
                else:
                    graph = random_network(N, mean_in_degree, max_lag, rng)
                    data = simulate_network(graph, T, kind, prng=rng)
                targets = None
                if n_targets is not None and n_targets < N:
                    targets = sorted(int(t) for t in rng.choice(N, n_targets, replace=False))
                seed = int(rng.integers(2**32))
                for mi, method in enumerate(methods):
                    t0 = time.perf_counter()
                    if method == "infoflow":
                        net = infer_multiplex(data, targets=targets, prng=seed, **kwargs)
                        elapsed = time.perf_counter() - t0
                        scores = score_network(net, graph, levels, targets)
                    else:
                        base = run_baseline(method, data, max_lag=max_lag, alpha=baseline_alpha)
                        elapsed = time.perf_counter() - t0
                        scores = {"skeleton": score_edges(base.found, base.lags, base.score, graph, targets)}
                    for li, level in enumerate(levels):
                        if level not in scores:
                            continue
                        for m in METRICS[:-1]:
                            values[m][mi, li, i, j, r] = scores[level][m]
                        values["runtime"][mi, li, i, j, r] = elapsed
    dims = ("algorithm", "level", "n_nodes", "n_samples", "rep")
    return xr.Dataset(
        {m: (dims, v) for m, v in values.items()},
        coords={
            "algorithm": methods,
            "level": list(levels),
            "n_nodes": list(n_nodes),
            "n_samples": list(n_samples),
            "rep": range(n_reps),
        },
        attrs={"kind": kind, "mean_in_degree": mean_in_degree, "max_lag": max_lag},
    )


LATENT_FLAGS = ("shared-dominant", "unexplained-tdmi", "zero-lag", "history-collider")
LATENT_METRICS = ("precision", "recall", "spurious", "flagged", "n_confounded")


def _hide(graph, n_hidden, rng):
    import networkx as nx

    drivers = [v for v in graph.nodes if graph.out_degree(v) >= 2]
    if len(drivers) < n_hidden:
        return None
    hidden = sorted(int(v) for v in rng.choice(drivers, n_hidden, replace=False))
    observed = [v for v in graph.nodes if v not in hidden]
    index = {v: k for k, v in enumerate(observed)}
    truth = nx.relabel_nodes(graph.subgraph(observed).copy(), index)
    confounded = set()
    for h in hidden:
        children = [index[c] for c in graph.successors(h) if c in index]
        for a in children:
            for b in children:
                if a != b and not truth.has_edge(a, b) and not truth.has_edge(b, a):
                    confounded.add((a, b))
    return observed, truth, confounded


def latent_validation(
    kind="var",
    n_nodes=8,
    n_hidden=1,
    n_samples=3000,
    n_reps=3,
    mean_in_degree=2.0,
    max_lag=3,
    methods=("infoflow", "lpcmci"),
    baseline_alpha=0.01,
    infer_kwargs=None,
    prng=None,
):
    """
    How methods treat pairs confounded by an unobserved driver.

    A random network :cite:`Novelli2019` is simulated, `n_hidden` nodes with at least
    two children are removed from the data, and the induced graph over the observed
    nodes is the truth. Ordered pairs of observed nodes that share a hidden parent
    but have no link either way are *confounded*: a method should not report a link
    there, and ideally it flags the confounding. infoflow flags it through
    :func:`~infoflow.interpret.latent_flags` (any of :data:`LATENT_FLAGS`); LPCMCI
    :cite:`Gerhardus2020` through a ``<->`` mark. The VAR and PCMCI+ baselines cannot
    flag, so their ``flagged`` is NaN.

    Returns
    -------
    xarray.Dataset
        ``precision`` and ``recall`` (observed links), ``spurious`` (fraction of
        confounded pairs reported as links), ``flagged`` (fraction flagged), and
        ``n_confounded``, over (algorithm, rep).
    """
    from .baselines import run_baseline
    from .network import infer_multiplex

    rng = as_generator(prng)
    kwargs = {"max_lag": max_lag, "node_layer": False, **(infer_kwargs or {}), "interpret": True}
    methods = list(methods)
    values = {m: np.full((len(methods), n_reps), np.nan) for m in LATENT_METRICS}
    for r in range(n_reps):
        for _ in range(100):
            graph = random_network(n_nodes, mean_in_degree, max_lag, rng)
            hidden = _hide(graph, n_hidden, rng)
            if hidden is not None and hidden[2]:
                break
        else:
            raise RuntimeError("no network with a confounding hidden driver was drawn")
        observed, truth, confounded = hidden
        data = simulate_network(graph, n_samples, kind, prng=rng)[:, observed]
        seed = int(rng.integers(2**32))
        for mi, method in enumerate(methods):
            if method == "infoflow":
                net = infer_multiplex(data, prng=seed, **kwargs)
                found = _edges(net, "skeleton")
                flags = net.dataset["flags"].values
                flagged = [any(f in str(flags[a, b]).split(",") for f in LATENT_FLAGS) for a, b in confounded]
                delay = net.dataset["delay"].values
            else:
                base = run_baseline(method, data, max_lag=max_lag, alpha=baseline_alpha)
                found, delay = base.found, base.lags
                flagged = [(a, b) in base.latent for a, b in confounded] if method == "lpcmci" else None
            s = score_edges(found, delay, None, truth)
            values["precision"][mi, r] = s["precision"]
            values["recall"][mi, r] = s["recall"]
            values["spurious"][mi, r] = np.mean([(a, b) in found for a, b in confounded])
            values["flagged"][mi, r] = np.mean(flagged) if flagged is not None else np.nan
            values["n_confounded"][mi, r] = len(confounded)
    return xr.Dataset(
        {m: (("algorithm", "rep"), v) for m, v in values.items()},
        coords={"algorithm": methods, "rep": range(n_reps)},
        attrs={"kind": kind, "n_nodes": n_nodes, "n_hidden": n_hidden, "n_samples": n_samples},
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
