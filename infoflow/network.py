"""
Multiplex network inference and the :class:`MultiplexNetwork` result.
"""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import xarray as xr
from dit.inference._symbols import as_generator

from .data import DiscreteData, realizations
from .embedding import Embedding
from .layers import LAYERS, adjust_layer_pvalues, layer_statistics
from .selection import SkeletonSettings, _cmi, _Columns, infer_skeleton

__all__ = (
    "EDGE_LAYERS",
    "MultiplexNetwork",
    "estimate_edge",
    "infer_multiplex",
)

EDGE_LAYERS = ("te", "tdmi", *LAYERS)


@dataclass
class MultiplexNetwork:
    """
    An inferred multiplex information-flow network.

    Attributes
    ----------
    dataset : xarray.Dataset
        Edge variables over dimensions (layer, source, target): ``weight``,
        ``ci_low``, ``ci_high``, ``pvalue``, ``qvalue``, ``significant``; edge
        variables over (source, target): ``kind`` (``"parent"``,
        ``"shared_candidate"``, or ``""``), ``delay``, ``n_samples``, ``omnibus_pvalue``;
        and node variables over (node,).
    skeleton : dict
        ``target -> TargetSkeleton``.
    edges : dict
        ``(source, target) -> LayerStatistics``.
    settings : dict
        Every setting used, for provenance.
    report : object, None
        The preprocessing report, when raw data were preprocessed.
    hyperedges : list of dict
        Two-source PID hyperedges, when requested.
    """

    dataset: xr.Dataset
    skeleton: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    report: object = None
    hyperedges: list = field(default_factory=list)

    @property
    def names(self):
        return [str(n) for n in self.dataset.coords["source"].values]

    def layer(self, name, significant_only=True):
        """
        One layer as a weighted networkx ``DiGraph``.
        """
        g = nx.DiGraph(layer=name)
        g.add_nodes_from(self.names)
        ds = self.dataset.sel(layer=name)
        for s, t in self.edges:
            src, tgt = self.names[s], self.names[t]
            sig = bool(ds["significant"].sel(source=src, target=tgt))
            if significant_only and not sig:
                continue
            g.add_edge(
                src,
                tgt,
                weight=float(ds["weight"].sel(source=src, target=tgt)),
                ci=(float(ds["ci_low"].sel(source=src, target=tgt)), float(ds["ci_high"].sel(source=src, target=tgt))),
                pvalue=float(ds["pvalue"].sel(source=src, target=tgt)),
                delay=int(self.dataset["delay"].sel(source=src, target=tgt)),
            )
        return g

    def to_networkx(self, significant_only=True, layers=LAYERS):
        """
        All flow layers as a networkx ``MultiDiGraph`` with a ``layer`` edge attribute.
        """
        g = nx.MultiDiGraph()
        g.add_nodes_from(self.names)
        for name in layers:
            for u, v, attrs in self.layer(name, significant_only).edges(data=True):
                g.add_edge(u, v, key=name, layer=name, **attrs)
        for node in self.names:
            for var in self.dataset.data_vars:
                if self.dataset[var].dims == ("node",):
                    g.nodes[node][var] = self.dataset[var].sel(node=node).item()
        return g

    def save(self, path):
        """
        Pickle the network (dataset, skeleton, edge statistics, settings, report).
        """
        from .io import save

        save(self, path)

    @staticmethod
    def load(path):
        """
        Load a network saved with :meth:`save`.
        """
        from .io import load

        return load(path)

    def summary(self):
        """
        A text table of the edges with any significant layer.
        """
        lines = ["source -> target  kind              delay  " + "  ".join(f"{k:>11}" for k in LAYERS)]
        for s, t in sorted(self.edges):
            src, tgt = self.names[s], self.names[t]
            cells = []
            for k in LAYERS:
                w = float(self.dataset["weight"].sel(layer=k, source=src, target=tgt))
                sig = bool(self.dataset["significant"].sel(layer=k, source=src, target=tgt))
                cells.append(f"{w:10.3f}{'*' if sig else ' '}")
            kind = str(self.dataset["kind"].sel(source=src, target=tgt).item())
            delay = int(self.dataset["delay"].sel(source=src, target=tgt))
            lines.append(f"{src:>6} -> {tgt:<6}  {kind:<16}  {delay:5d}  " + "  ".join(cells))
        return "\n".join(lines)


def _interaction_delay(cols, x_vars, w_vars):
    """
    The lag of the source variable with the largest CMI given everything else.
    """
    if len(x_vars) == 1:
        return int(x_vars[0][1])
    best, best_value = x_vars[0], -1.0
    for v in x_vars:
        others = [u for u in x_vars if u != v] + list(w_vars)
        z, Kz = cols.joint(others)
        c, a = cols.column(v)
        value = _cmi(cols.y, cols.Ky, c, a, z, Kz)
        if value > best_value:
            best, best_value = v, value
    return int(best[1])


def estimate_edge(data, target, x_vars, w_vars, estimator="miller_madow", n_boot=100, n_null=100, prng=None):
    """
    Layer statistics for one edge given its source and context variables.

    Returns
    -------
    stats : LayerStatistics
    delay : int
    """
    variables = list(dict.fromkeys(list(x_vars) + list(w_vars)))
    max_lag = max((v[1] for v in variables), default=0)
    r = realizations(data, target, variables, max_lag=max_lag)
    x = r.columns(list(x_vars))
    w = r.columns(list(w_vars)) if w_vars else None
    stats = layer_statistics(r.present, x, w, n_boot=n_boot, n_null=n_null, estimator=estimator, gap=max_lag, prng=prng)
    cols = _Columns(data, target, variables, max_lag)
    return stats, _interaction_delay(cols, list(x_vars), list(w_vars))


def _as_discrete(data, names, preprocess, max_lag, prng):
    if isinstance(data, DiscreteData):
        return data, None, None
    from .data import as_trials

    trials = as_trials(data)
    if preprocess is None or (preprocess == "auto" and all(np.issubdtype(t.dtype, np.integer) for t in trials)):
        return DiscreteData.from_discrete(trials, names), None, None
    from .preprocess.pipeline import preprocess as run_preprocess

    kwargs = {} if preprocess == "auto" else dict(preprocess)
    result = run_preprocess(trials, names=names, max_lag=max_lag, prng=prng, **kwargs)
    return result.data, result.embeddings, result.report


def infer_multiplex(
    data,
    names=None,
    embeddings=None,
    max_lag=None,
    preprocess="auto",
    estimator="miller_madow",
    alpha=0.05,
    fdr_dependent=False,
    n_boot=100,
    n_null=100,
    skeleton_settings=None,
    include_shared_candidates=True,
    targets=None,
    map_fn=map,
    adaptive_resamples=True,
    holdout=None,
    node_layer=True,
    interpret=True,
    contemporaneous=False,
    hyperedges=False,
    checkpoint=None,
    prng=None,
):
    """
    Infer a multiplex information-flow network.

    Parameters
    ----------
    data : array_like, Trials, or DiscreteData
        Raw series (samples, processes), independent trials of them, or
        already-discretized data. Integer arrays are used as symbols directly; float
        arrays are preprocessed (discretizer and embedding chosen per node).
    names : list of str, None
    embeddings : Embedding or list of Embedding, None
        Candidate lag grids; by default the preprocessing lag budgets, or
        ``Embedding(max_lag)``.
    max_lag : int, None
        Lag budget when none is chosen by preprocessing (default 3), and an upper
        bound on the budgets that preprocessing chooses.
    preprocess : 'auto', dict, or None
        ``'auto'`` preprocesses float data with :func:`~infoflow.preprocess.preprocess`;
        a dict passes options to it; None treats values as symbols.
    estimator : {'miller_madow', 'plugin'}
        Evaluation estimator for the flow layers.
    alpha : float
        FDR level for the layer tests.
    fdr_dependent : bool
        Benjamini–Yekutieli instead of Benjamini–Hochberg for the layers.
    n_boot, n_null : int
        Bootstrap and parametric-null resamples per edge.
    skeleton_settings : SkeletonSettings, None
    include_shared_candidates : bool
        Also estimate layers for significant TDMI-screen edges (shared-only candidates).
    targets : list of int, None
    map_fn : callable
        ``map``-like function used over targets (e.g. from :mod:`infoflow.parallel`).
    adaptive_resamples : bool
        Raise `n_boot` and `n_null` to at least ``ceil(m / alpha)`` for ``m`` tested
        edges, so that the smallest attainable p-value can survive the FDR correction.
    node_layer : bool
        Add active information storage and predictability flags per node.
    interpret : bool
        Add edge roles (direct / confounded / mediated) and latent-confounding flags.
    contemporaneous : bool
        Add the optional lag-0 layer (:mod:`infoflow.contemporaneous`) as the
        ``contemporaneous`` variable (0 none, 1 undirected, 2 oriented source -> target).
    hyperedges : bool
        Add two-source PID hyperedges (:mod:`infoflow.hyperedges`) as ``network.hyperedges``.
    checkpoint : str, Path, or Checkpoint, None
        A directory where per-target skeletons and per-edge estimates are stored as
        they finish; rerunning with the same directory and seed resumes.
    holdout : float, None
        If given, select the skeleton on the first `holdout` fraction of each trial
        and estimate the layers on the rest. A falsely selected parent enters its
        siblings' contexts because of a chance fluctuation in the same data, which
        biases their synergy upward; holding out the estimation data removes this
        post-selection bias at the cost of power.
    prng : None, int, Generator

    Returns
    -------
    MultiplexNetwork
    """
    from .checkpoint import Checkpoint

    if checkpoint is not None and not isinstance(checkpoint, Checkpoint):
        checkpoint = Checkpoint(checkpoint)
    rng = as_generator(prng)
    discrete, chosen, report = _as_discrete(data, names, preprocess, max_lag, rng)
    names = discrete.names if names is None else list(names)
    discrete.names = names
    if embeddings is None:
        embeddings = chosen if chosen is not None else Embedding(max_lag=max_lag or 3)
    settings = skeleton_settings or SkeletonSettings()
    selection_data, estimation_data = (discrete, discrete) if holdout is None else discrete.split_time(holdout)
    skeleton = infer_skeleton(selection_data, embeddings, targets, settings, rng, map_fn=map_fn, checkpoint=checkpoint)

    jobs = []
    for t, sk in skeleton.items():
        parents = sk.parents()
        base = list(sk.conditionals) + list(sk.target_past)
        for p in parents:
            x_vars = sk.source_variables(p)
            w_vars = base + [v for v in sk.sources if v[0] != p]
            jobs.append((p, t, x_vars, w_vars, "parent"))
        if include_shared_candidates:
            for p, cand in sk.tdmi_candidates.items():
                if cand.get("significant") and p not in parents:
                    jobs.append(
                        (
                            p,
                            t,
                            [cand["variable"]],
                            base + list(sk.sources if sk.significant else []),
                            "shared_candidate",
                        )
                    )

    seeds = rng.integers(0, 2**32, size=len(jobs))
    if adaptive_resamples and jobs:
        # With m tested edges, BH can only reject a p-value at its floor 1 / (B + 1) if
        # B + 1 >= m / alpha, so raise the resample budget to make that possible.
        floor = int(np.ceil(len(jobs) / alpha))
        n_boot, n_null = max(n_boot, floor), max(n_null, floor)

    def run(job):
        (p, t, x_vars, w_vars, kind), seed = job

        def compute():
            return estimate_edge(estimation_data, t, x_vars, w_vars, estimator, n_boot, n_null, int(seed))

        stats, delay = compute() if checkpoint is None else checkpoint.cached(f"edge-{p}-{t}-{kind}", compute)
        return (p, t), (stats, delay, kind)

    results = dict(map_fn(run, zip(jobs, seeds)))
    dataset = _assemble(discrete, names, skeleton, results, alpha, fdr_dependent)
    if node_layer:
        from .nodes import node_layer as compute_nodes

        nodes = compute_nodes(estimation_data, skeleton, alpha=alpha, prng=rng)
        for key, values in nodes.items():
            dataset[f"node_{key}" if key == "flags" else key] = (("node",), values)
    if interpret:
        from .interpret import edge_roles, latent_flags

        roles, explained = edge_roles(selection_data, skeleton, embeddings, settings, prng=rng)
        dataset["role"] = (("source", "target"), roles)
        dataset["explained_by"] = (("source", "target"), explained)
        edges_run = set(results)
        dataset["flags"] = (
            ("source", "target"),
            latent_flags(estimation_data, skeleton, edges_run, dataset, embeddings, roles),
        )
    if contemporaneous:
        from .contemporaneous import contemporaneous_layer

        graph, values, _ = contemporaneous_layer(estimation_data, skeleton, alpha=alpha, prng=rng)
        dataset["contemporaneous"] = (("source", "target"), graph)
        dataset["contemporaneous_cmi"] = (("source", "target"), values)
    pid = []
    if hyperedges:
        from .hyperedges import pid_hyperedges

        pid = pid_hyperedges(estimation_data, skeleton)
    import datetime
    from importlib.metadata import version

    provenance = {
        "versions": {pkg: version(pkg) for pkg in ("infoflow", "dit", "numpy", "scipy")},
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "estimator": estimator,
        "alpha": alpha,
        "fdr_dependent": fdr_dependent,
        "n_boot": n_boot,
        "n_null": n_null,
        "adaptive_resamples": adaptive_resamples,
        "holdout": holdout,
        "node_layer": node_layer,
        "interpret": interpret,
        "contemporaneous": contemporaneous,
        "hyperedges": hyperedges,
        "max_lag": max_lag,
        "embeddings": embeddings if isinstance(embeddings, list) else [embeddings] * discrete.n_processes,
        "skeleton_settings": settings,
        "discretizers": [getattr(d, "params", lambda d=d: {"name": str(d)})() for d in discrete.discretizers],
    }
    return MultiplexNetwork(
        dataset=dataset,
        skeleton=skeleton,
        edges={k: v[0] for k, v in results.items()},
        settings=provenance,
        report=report,
        hyperedges=pid,
    )


def _assemble(data, names, skeleton, results, alpha, fdr_dependent):
    P = data.n_processes
    shape = (len(EDGE_LAYERS), P, P)
    weight = np.full(shape, np.nan)
    low = np.full(shape, np.nan)
    high = np.full(shape, np.nan)
    pvalue = np.full(shape, np.nan)
    kind = np.full((P, P), "", dtype=object)
    delay = np.zeros((P, P), dtype=np.int64)
    n_samples = np.zeros((P, P), dtype=np.int64)
    omnibus = np.full(P, np.nan)
    for t, sk in skeleton.items():
        omnibus[t] = sk.omnibus_pvalue if sk.sources else np.nan
    for (s, t), (stats, d, k) in results.items():
        flow = stats.flow.as_dict()
        for i, layer in enumerate(EDGE_LAYERS):
            weight[i, s, t] = flow[layer]
            if layer in stats.ci:
                low[i, s, t], high[i, s, t] = stats.ci[layer]
            if layer in stats.pvalue:
                pvalue[i, s, t] = stats.pvalue[layer]
        kind[s, t] = k
        delay[s, t] = d
        n_samples[s, t] = stats.flow.n_samples
    layer_p = {layer: pvalue[EDGE_LAYERS.index(layer)] for layer in LAYERS}
    significant, adjusted = adjust_layer_pvalues(layer_p, alpha, fdr_dependent)
    sig = np.zeros(shape, dtype=bool)
    q = np.full(shape, np.nan)
    for layer in LAYERS:
        i = EDGE_LAYERS.index(layer)
        sig[i], q[i] = significant[layer], adjusted[layer]
    # TE and TDMI rows: significant when their component layers are.
    sig[0] = sig[EDGE_LAYERS.index("intrinsic")] | sig[EDGE_LAYERS.index("synergistic")]
    sig[1] = sig[EDGE_LAYERS.index("intrinsic")] | sig[EDGE_LAYERS.index("shared")]
    coords = {"layer": list(EDGE_LAYERS), "source": names, "target": names}
    ds = xr.Dataset(
        {
            "weight": (("layer", "source", "target"), weight),
            "ci_low": (("layer", "source", "target"), low),
            "ci_high": (("layer", "source", "target"), high),
            "pvalue": (("layer", "source", "target"), pvalue),
            "qvalue": (("layer", "source", "target"), q),
            "significant": (("layer", "source", "target"), sig),
            "kind": (("source", "target"), kind.astype(str)),
            "delay": (("source", "target"), delay),
            "n_samples": (("source", "target"), n_samples),
            "omnibus_pvalue": (("node",), omnibus),
        },
        coords={**coords, "node": names},
        attrs={"description": "Multiplex information-flow network (bits)"},
    )
    return ds
