"""
Regime-dependent networks and pooling across datasets.

*Regimes.* Following Regime-PCMCI :cite:`Saggioro2020`, :func:`infer_regimes`
alternates between learning one skeleton per regime and reassigning time to the
regime whose skeleton predicts it best. Time is cut into windows of `window`
samples, the unit of persistence: every window belongs to one regime. Windows are
initialised by k-means on their pairwise lagged mutual information, then each
iteration (1) infers a skeleton per regime from its windows and (2) reassigns every
window to the regime maximizing the plug-in predictive log-likelihood of all
targets given their selected parents, with the window's own counts left out of its
current regime's model so that a window is not kept by fitting itself. Iteration
stops when no window moves. :func:`compare_regimes` tests the layer differences
between two regimes with :func:`~infoflow.compare.compare_networks`, the windows
being the units.

*Multiple datasets.* J-PCMCI+ :cite:`Gunther2023` pools datasets of one system and
adds dataset-index variables, constant within a dataset, that absorb differences
between datasets (unobserved contexts) which would otherwise confound the pooled
estimates. :func:`add_context` appends such a context process to discretized
trials and :func:`infer_with_context` infers a network with it in every
conditioning set. Whether it is needed depends on the selection null: a
within-trial permutation (the default with few or unequal trials) keeps a
dataset-level association in the null, so it inflates the estimates but not the
tests, while swapping whole trials (the default with many equal-length trials)
breaks it and reports it as links.
"""

from dataclasses import dataclass, replace

import numpy as np
from dit.inference._symbols import as_generator

from .data import DiscreteData, realizations
from .embedding import Embedding
from .selection import SkeletonSettings, infer_skeleton

__all__ = (
    "RegimeResult",
    "add_context",
    "compare_regimes",
    "infer_regimes",
    "infer_with_context",
    "regime_data",
)


@dataclass
class RegimeResult:
    """
    Regime assignment and per-regime skeletons.

    Attributes
    ----------
    labels : numpy.ndarray
        Regime of each window.
    windows : list of (int, int, int)
        ``(trial, start, stop)`` sample range of each window.
    skeletons : list of dict
        Per regime, ``target -> TargetSkeleton``.
    log_likelihood : numpy.ndarray
        (windows, regimes) predictive log-likelihood (nats) at the last iteration.
    data : DiscreteData
        The windowed data, one trial per window.
    embeddings : Embedding or list of Embedding
    n_iter : int
    """

    labels: np.ndarray
    windows: list
    skeletons: list
    log_likelihood: np.ndarray
    data: DiscreteData
    embeddings: object
    n_iter: int


def _windowed(data, window):
    def cut(arrays):
        return [a[:, s : s + window] for a in arrays for s in range(0, a.shape[1] - window + 1, window)]

    windows = [(i, s, s + window) for i, a in enumerate(data.past) for s in range(0, a.shape[1] - window + 1, window)]
    raw = None
    if data.raw is not None:
        raw = [r[s : s + window] for r in data.raw for s in range(0, len(r) - window + 1, window)]
    out = DiscreteData(
        past=cut(data.past),
        present=cut(data.present),
        past_offset=data.past_offset,
        present_offset=data.present_offset,
        past_alphabet=data.past_alphabet,
        present_alphabet=data.present_alphabet,
        lag_step=data.lag_step,
        names=data.names,
        discretizers=data.discretizers,
        raw=raw,
    )
    return out, windows


def _features(data):
    """
    Per window, the plug-in mutual information between every process's lag-1 past and
    every target's present.
    """

    def entropy(*columns):
        p = np.unique(np.stack(columns), axis=1, return_counts=True)[1] / len(columns[0])
        return -np.sum(p * np.log(p))

    P = data.n_processes
    feats = []
    for past, present in zip(data.past, data.present, strict=True):
        f = np.zeros((P, P))
        for s in range(P):
            for t in range(P):
                x, y = past[s, :-1], present[t, 1:]
                ok = (x >= 0) & (y >= 0)
                if ok.sum() >= 2:
                    f[s, t] = entropy(x[ok]) + entropy(y[ok]) - entropy(x[ok], y[ok])
        feats.append(f.ravel())
    return np.array(feats)


def _target_loglik(data, target, sk, members, prior=0.5):
    """
    Per window, the log-likelihood of `target` under the model fitted on `members`
    (each member window's own counts excluded from its fit).
    """
    variables = list(dict.fromkeys(list(sk.conditionals) + list(sk.target_past) + list(sk.sources)))
    max_lag = max([v[1] for v in variables] + [1])
    r = realizations(data, target, variables, max_lag=max_lag)
    n_windows = data.n_trials
    if len(r) == 0:
        return np.zeros(n_windows)
    K = int(data.present_alphabet[target])
    if variables:
        _, ctx = np.unique(r.values, axis=0, return_inverse=True)
        ctx = ctx.ravel()
    else:
        ctx = np.zeros(len(r), dtype=np.int64)
    C = int(ctx.max()) + 1
    cell = ctx * K + r.present
    in_fit = np.isin(r.trial, members)
    counts = np.bincount(cell[in_fit], minlength=C * K).astype(float)
    # Each window's own counts in its cells, to leave it out of its own regime's fit.
    own = np.zeros(len(r))
    own_ctx = np.zeros(len(r))
    order = np.lexsort((cell, r.trial))
    key = r.trial[order] * (C * K) + cell[order]
    _, n = np.unique(key, return_counts=True)
    own[order] = np.repeat(n, n)
    key_c = r.trial[order] * C + ctx[order]
    _, n_c = np.unique(key_c, return_counts=True)
    own_ctx[order] = np.repeat(n_c, n_c)
    ctx_counts = counts.reshape(C, K).sum(axis=1)
    num = counts[cell] - np.where(in_fit, own, 0.0)
    den = ctx_counts[ctx] - np.where(in_fit, own_ctx, 0.0)
    logp = np.log((num + prior) / (den + prior * K))
    return np.bincount(r.trial, weights=logp, minlength=n_windows)


def infer_regimes(
    data,
    n_regimes=2,
    window=200,
    n_iter=10,
    embeddings=None,
    settings=None,
    max_lag=None,
    preprocess="auto",
    names=None,
    targets=None,
    labels=None,
    map_fn=map,
    prng=None,
):
    """
    Regime-dependent skeletons by alternating assignment and inference (module docstring).

    Parameters
    ----------
    data : array_like, Trials, or DiscreteData
        As for :func:`~infoflow.network.infer_multiplex`; raw data are preprocessed once.
    n_regimes : int
    window : int
        Samples per window (the shortest regime episode that can be resolved).
    n_iter : int
        Maximum number of assignment / inference iterations.
    labels : array_like, None
        Initial regime per window (default: k-means on lagged mutual information).
    embeddings, settings, max_lag, preprocess, names, targets, map_fn :
        As for :func:`~infoflow.network.infer_multiplex`.
    prng : None, int, Generator

    Returns
    -------
    RegimeResult
    """
    from scipy.cluster.vq import kmeans2

    from .network import _as_discrete

    rng = as_generator(prng)
    discrete, chosen, _ = _as_discrete(data, names, preprocess, max_lag, rng)
    if embeddings is None:
        embeddings = chosen if chosen is not None else Embedding(max_lag=max_lag or 3)
    settings = settings or SkeletonSettings()
    windowed, windows = _windowed(discrete, window)
    W = len(windows)
    if n_regimes > W:
        raise ValueError(f"{W} windows of {window} samples cannot hold {n_regimes} regimes")
    targets = list(range(discrete.n_processes)) if targets is None else list(targets)
    if labels is None:
        feats = _features(windowed)
        feats = (feats - feats.mean(axis=0)) / np.where(feats.std(axis=0) > 0, feats.std(axis=0), 1.0)
        _, labels = kmeans2(feats, n_regimes, minit="++", seed=int(rng.integers(2**32)))
    labels = np.asarray(labels, dtype=np.int64)
    skeletons, loglik = [], np.zeros((W, n_regimes))
    it = 0
    for _ in range(n_iter):
        it += 1
        if any(not np.any(labels == k) for k in range(n_regimes)):
            raise RuntimeError("a regime lost all its windows; use fewer regimes or shorter windows")
        skeletons = [
            infer_skeleton(
                windowed.subset_trials(np.flatnonzero(labels == k)), embeddings, targets, settings, rng, map_fn
            )
            for k in range(n_regimes)
        ]
        loglik = np.zeros((W, n_regimes))
        for k in range(n_regimes):
            members = np.flatnonzero(labels == k)
            for t in targets:
                loglik[:, k] += _target_loglik(windowed, t, skeletons[k][t], members)
        new = np.argmax(loglik, axis=1)
        if np.array_equal(new, labels) or any(not np.any(new == k) for k in range(n_regimes)):
            break
        labels = new
    return RegimeResult(labels, windows, skeletons, loglik, windowed, embeddings, it)


def regime_data(result, regime):
    """
    The windows of one regime as a DiscreteData with one trial per window.
    """
    return result.data.subset_trials(np.flatnonzero(result.labels == regime))


def compare_regimes(result, a=0, b=1, infer_kwargs=None, compare_kwargs=None, prng=None):
    """
    Infer a network per regime and compare their layers (:func:`~infoflow.compare.compare_networks`,
    windows as units of a ``'within'`` design).

    Returns
    -------
    tuple
        ``(network_a, network_b, NetworkComparison)``.
    """
    from .compare import compare_networks
    from .network import infer_multiplex

    rng = as_generator(prng)
    kwargs = {"embeddings": result.embeddings, "interpret": False, "node_layer": False, **(infer_kwargs or {})}
    data_a, data_b = regime_data(result, a), regime_data(result, b)
    net_a = infer_multiplex(data_a, prng=rng, **kwargs)
    net_b = infer_multiplex(data_b, prng=rng, **kwargs)
    comparison = compare_networks(net_a, net_b, data_a, data_b, design="within", prng=rng, **(compare_kwargs or {}))
    return net_a, net_b, comparison


def add_context(data, contexts=None, embeddings=None, max_lag=None, preprocess="auto", names=None, prng=None):
    """
    Append a context process, constant within each trial, to (discretized) trials
    :cite:`Gunther2023`.

    Parameters
    ----------
    data : array_like, Trials, or DiscreteData
        One trial per dataset; raw data are preprocessed first.
    contexts : sequence, None
        One label per trial (default: the trial index, i.e. one dummy per dataset).
        Trials with equal labels share a context.

    Returns
    -------
    data : DiscreteData
        The trials with the context process appended last (named ``'context'``).
    embeddings : list of Embedding
        Per-process lag grids, the context's being lag 1.
    context : int
        The context process's index.
    """
    from .network import _as_discrete

    rng = as_generator(prng)
    discrete, chosen, _ = _as_discrete(data, names, preprocess, max_lag, rng)
    P = discrete.n_processes
    labels = list(range(discrete.n_trials)) if contexts is None else list(contexts)
    if len(labels) != discrete.n_trials:
        raise ValueError("contexts needs one label per trial")
    codes = {c: i for i, c in enumerate(dict.fromkeys(labels))}

    def extend(arrays):
        return [
            np.vstack([a, np.full((1, a.shape[1]), codes[c], dtype=a.dtype)])
            for a, c in zip(arrays, labels, strict=True)
        ]

    K = max(len(codes), 1)
    out = DiscreteData(
        past=extend(discrete.past),
        present=extend(discrete.present),
        past_offset=np.append(discrete.past_offset, 0),
        present_offset=np.append(discrete.present_offset, 0),
        past_alphabet=np.append(discrete.past_alphabet, K),
        present_alphabet=np.append(discrete.present_alphabet, K),
        lag_step=np.append(discrete.lag_step, 1),
        names=list(discrete.names) + ["context"],
        discretizers=list(discrete.discretizers) + ["identity"],
        raw=None
        if discrete.raw is None
        else [np.column_stack([r, np.full(len(r), codes[c])]) for r, c in zip(discrete.raw, labels, strict=True)],
    )
    if embeddings is None:
        embeddings = chosen if chosen is not None else Embedding(max_lag=max_lag or 3)
    per = list(embeddings) if isinstance(embeddings, (list, tuple)) else [embeddings] * P
    return out, per + [Embedding(max_lag=1)], P


def infer_with_context(data, contexts=None, skeleton_settings=None, max_lag=None, prng=None, **infer_kwargs):
    """
    Infer a network from pooled datasets with a context process in every conditioning set.

    The context (:func:`add_context`) is a forced conditional of every target's parent
    selection and so enters every layer's context; it is not itself a target.

    Returns
    -------
    MultiplexNetwork
        Over the original processes plus ``'context'``.
    """
    from .network import infer_multiplex

    rng = as_generator(prng)
    pooled, embeddings, c = add_context(data, contexts, max_lag=max_lag, prng=rng)
    settings = skeleton_settings or SkeletonSettings()
    settings = replace(settings, forced_conditionals=tuple(settings.forced_conditionals) + ((c, 1),))
    return infer_multiplex(
        pooled,
        embeddings=embeddings,
        skeleton_settings=settings,
        targets=list(range(c)),
        max_lag=max_lag,
        prng=rng,
        **infer_kwargs,
    )
