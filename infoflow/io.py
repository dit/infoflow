"""
Input and output: MATLAB / FieldTrip import, BrainNet Viewer export, save and load,
and loaders and exporters for external network-inference benchmarks (NetSim,
CausalTime, CausalRivers, CauseMe). Benchmark data are not shipped; download them
from the benchmarks' own sites.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from dit.inference import Trials

__all__ = (
    "adjacency_graph",
    "causalrivers_subset",
    "causeme_matrices",
    "export_brainnet",
    "load",
    "load_causalrivers",
    "load_causaltime",
    "load_fieldtrip",
    "load_mat",
    "load_netsim",
    "read_causalrivers",
    "sample_subgraph",
    "save",
    "to_causeme",
)


def load_mat(path, variable, dim_order="sp"):
    """
    Load an array from a MATLAB file as (samples, processes) or trials of them.

    Parameters
    ----------
    path : str or Path
    variable : str
        The variable name in the file.
    dim_order : str
        The order of the array's axes, using ``s`` (samples), ``p`` (processes),
        and ``r`` (replications/trials), e.g. ``'ps'`` or ``'psr'``.
    """
    from scipy.io import loadmat

    arr = np.asarray(loadmat(path)[variable], dtype=float)
    order = [dim_order.index(c) for c in ("s", "p", "r") if c in dim_order]
    arr = np.transpose(arr, order)
    if arr.ndim == 3:
        return Trials(arr[:, :, r] for r in range(arr.shape[2]))
    return arr


def load_fieldtrip(path, variable="data"):
    """
    Load a FieldTrip raw-data structure (``data.trial`` cell array of channels x samples).

    Returns
    -------
    trials : Trials
        One (samples, channels) array per trial.
    labels : list of str
        Channel labels.
    """
    from scipy.io import loadmat

    struct = loadmat(path, squeeze_me=True, struct_as_record=False)[variable]
    trials = np.atleast_1d(struct.trial)
    labels = [str(label) for label in np.atleast_1d(struct.label)]
    return Trials(np.asarray(t, dtype=float).T for t in trials), labels


def export_brainnet(network, layer, coordinates, path, significant_only=True):
    """
    Write BrainNet Viewer ``.node`` and ``.edge`` files for one layer.

    Parameters
    ----------
    network : MultiplexNetwork
    layer : str
    coordinates : array_like
        (nodes, 3) MNI coordinates.
    path : str or Path
        Output path without extension.
    """
    path = Path(path)
    names = network.names
    coordinates = np.asarray(coordinates, dtype=float)
    ds = network.dataset.sel(layer=layer)
    weights = np.nan_to_num(ds["weight"].values)
    if significant_only:
        weights = np.where(ds["significant"].values, weights, 0.0)
    with open(path.with_suffix(".node"), "w") as fh:
        for (x, y, z), name in zip(coordinates, names, strict=True):
            fh.write(f"{x:.3f}\t{y:.3f}\t{z:.3f}\t1\t1\t{name.replace(' ', '_')}\n")
    np.savetxt(path.with_suffix(".edge"), weights, delimiter="\t", fmt="%.6f")
    return path.with_suffix(".node"), path.with_suffix(".edge")


def save(obj, path):
    """
    Pickle a network, comparison, or preprocessing result.
    """
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def load(path):
    """
    Load an object saved with :func:`save`.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)


# -- external benchmarks -----------------------------------------------------------


def adjacency_graph(adjacency, lag=None):
    """
    A ``networkx.DiGraph`` with an edge ``i -> j`` wherever ``adjacency[i, j]`` is nonzero
    (the diagonal is ignored), optionally with a ``lag`` attribute on every edge.
    """
    import networkx as nx

    A = np.asarray(adjacency)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(A)))
    for i, j in zip(*np.nonzero(A), strict=True):
        if i != j:
            graph.add_edge(int(i), int(j), **({} if lag is None else {"lag": lag}))
    return graph


def load_netsim(path, subject=0):
    """
    One subject of a NetSim simulation :cite:`Smith2011` (``sim<k>.mat``).

    ``ts`` stacks the subjects' (time, node) blocks and ``net[s, i, j]`` is the
    ground-truth connection from node ``i`` to node ``j``; its diagonal holds the
    nodes' self-decay and is ignored.

    Returns
    -------
    data : numpy.ndarray
        (Ntimepoints, Nnodes).
    graph : networkx.DiGraph
        The subject's ground truth (no lags: BOLD convolution blurs them).
    """
    from scipy.io import loadmat

    d = loadmat(path)
    T = int(np.squeeze(d["Ntimepoints"]))
    ts = np.asarray(d["ts"], dtype=float)
    net = np.asarray(d["net"], dtype=float)
    net = net[None] if net.ndim == 2 else net
    return ts[subject * T : (subject + 1) * T], adjacency_graph(net[subject])


def load_causaltime(data, graph, orientation="source_target", residual_block=True):
    """
    A CausalTime dataset :cite:`Cheng2024` (AQI, Traffic, or Medical).

    CausalTime's ground truth is a :math:`2N \\times 2N` graph over the :math:`N`
    observed series and :math:`N` residual terms; as in the paper, only the observed
    :math:`N \\times N` block is scored, and only the first :math:`N` columns of the
    data (the observed series) are kept.

    Parameters
    ----------
    data : str, Path, or array_like
        A ``.npy`` file or array of shape (time, columns) or (samples, time, columns).
    graph : str, Path, or array_like
        The :math:`2N \\times 2N` (or :math:`N \\times N`) ground-truth matrix.
    orientation : {'source_target', 'target_source'}
        Whether ``graph[i, j]`` marks ``i -> j`` or ``j -> i``; check it against the
        release you downloaded.
    residual_block : bool
        `graph` is :math:`2N \\times 2N` with the residual terms (default); False for an
        :math:`N \\times N` graph.

    Returns
    -------
    data : numpy.ndarray or Trials
        (time, N), or one such array per sample.
    graph : networkx.DiGraph
    """
    X = np.load(data) if isinstance(data, (str, Path)) else np.asarray(data)
    A = np.load(graph) if isinstance(graph, (str, Path)) else np.asarray(graph)
    if orientation not in ("source_target", "target_source"):
        raise ValueError("orientation must be 'source_target' or 'target_source'")
    if residual_block and A.shape[0] % 2:
        raise ValueError("a graph with residual terms is 2N x 2N")
    N = A.shape[0] // 2 if residual_block else A.shape[0]
    A = A[:N, :N]
    if orientation == "target_source":
        A = A.T
    X = np.asarray(X[..., :N], dtype=float)
    out = Trials(X[i] for i in range(len(X))) if X.ndim == 3 else X
    return out, adjacency_graph(A)


def read_causalrivers(ts_csv, graph_pickle):
    """
    The full CausalRivers tables :cite:`Stein2025`, read once.

    Returns
    -------
    values : numpy.ndarray
        (time, stations), float32, NaN where missing.
    graph : networkx.DiGraph
        Ground truth over station ids (upstream -> downstream).
    stations : list of str
        Station id of each column.
    """
    import csv

    with open(graph_pickle, "rb") as fh:
        graph = pickle.load(fh)
    with open(ts_csv, newline="") as fh:
        reader = csv.reader(fh)
        stations = [_station(h) for h in next(reader)[1:]]
        values = np.array([[float(v) if v != "" else np.nan for v in r[1:]] for r in reader], dtype=np.float32)
    return values, graph, stations


def _resample(values, block):
    n = len(values) // block
    blocks = values[: n * block].reshape(n, block, values.shape[1])
    with np.errstate(invalid="ignore"):
        counts = np.sum(~np.isnan(blocks), axis=1)
        means = np.nansum(blocks, axis=1) / np.maximum(counts, 1)
    return np.where(counts > 0, means, np.nan)


def _interpolate(values):
    """
    Linear interpolation of interior gaps per column; rows before the first or after the
    last complete row are trimmed.
    """
    out = values.copy()
    t = np.arange(len(out))
    for j in range(out.shape[1]):
        ok = ~np.isnan(out[:, j])
        if ok.sum() >= 2:
            out[:, j] = np.interp(t, t[ok], out[ok, j], left=np.nan, right=np.nan)
    complete = np.flatnonzero(~np.isnan(out).any(axis=1))
    return out[complete[0] : complete[-1] + 1] if len(complete) else out[:0]


def causalrivers_subset(values, graph, stations, nodes, resample=1, gaps="interpolate"):
    """
    The series and induced ground truth of a subset of CausalRivers stations.

    Parameters
    ----------
    values, graph, stations
        From :func:`read_causalrivers`.
    nodes : list
        Station ids, e.g. from :func:`sample_subgraph`.
    resample : int
        Average blocks of this many samples (NaN-aware), e.g. 24 for 6-hour
        resolution of the 15-minute data.
    gaps : {'interpolate', 'drop'}
        Fill remaining gaps by linear interpolation, as the benchmark's own
        preprocessing does (default), or drop incomplete rows, which splices
        separate stretches of time together.

    Returns
    -------
    data : numpy.ndarray
        (time, len(nodes)).
    graph : networkx.DiGraph
        The induced ground truth relabelled ``0..len(nodes)-1`` in the order of `nodes`,
        each node carrying its station id as ``station``.
    """
    import networkx as nx

    columns = {s: i for i, s in enumerate(stations)}
    sub = values[:, [columns[_station(n)] for n in nodes]].astype(float)
    if resample > 1:
        sub = _resample(sub, resample)
    if gaps == "interpolate":
        sub = _interpolate(sub)
    elif gaps == "drop":
        sub = sub[~np.isnan(sub).any(axis=1)]
    else:
        raise ValueError("gaps must be 'interpolate' or 'drop'")
    relabelled = nx.relabel_nodes(graph.subgraph(nodes).copy(), {n: k for k, n in enumerate(nodes)})
    truth = nx.DiGraph()
    truth.add_nodes_from((k, {"station": n}) for k, n in enumerate(nodes))
    truth.add_edges_from(relabelled.edges())
    return sub, truth


def load_causalrivers(ts_csv, graph_pickle, nodes=None, resample=1, gaps="interpolate"):
    """
    River-discharge series and ground truth from CausalRivers :cite:`Stein2025`.

    Parameters
    ----------
    ts_csv : str or Path
        The time-series table (first column time stamps, one column per station id).
    graph_pickle : str or Path
        The pickled ``networkx.DiGraph`` over station ids (upstream -> downstream).
    nodes : list, None
        Station ids to keep, e.g. from :func:`sample_subgraph` (default: every station
        present in both files).
    resample : int
        Average blocks of this many samples (see :func:`causalrivers_subset`).
    gaps : {'interpolate', 'drop'}
        See :func:`causalrivers_subset`.

    Returns
    -------
    data : numpy.ndarray
        (time, len(nodes)).
    graph : networkx.DiGraph
        The induced ground truth relabelled ``0..len(nodes)-1`` in the order of `nodes`,
        each node carrying its station id as ``station``.
    names : list of str
    """
    values, full, stations = read_causalrivers(ts_csv, graph_pickle)
    present = set(stations)
    if nodes is None:
        nodes = [n for n in full.nodes if _station(n) in present]
    data, truth = causalrivers_subset(values, full, stations, nodes, resample, gaps)
    return data, truth, [str(n) for n in nodes]


def _station(x):
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def sample_subgraph(graph, n_nodes, prng=None, max_tries=1000):
    """
    A random weakly connected set of `n_nodes` nodes of `graph`, as CausalRivers samples
    its benchmark subgraphs :cite:`Stein2025`.

    Grows a node set from a random seed by adding a random neighbour (in either
    direction) of the current set.

    Returns
    -------
    list
        Node labels of `graph`.
    """
    from dit.inference._symbols import as_generator

    rng = as_generator(prng)
    nodes = list(graph.nodes)
    for _ in range(max_tries):
        chosen = [nodes[int(rng.integers(len(nodes)))]]
        while len(chosen) < n_nodes:
            frontier = sorted(
                {m for c in chosen for m in (*graph.successors(c), *graph.predecessors(c))} - set(chosen), key=str
            )
            if not frontier:
                break
            chosen.append(frontier[int(rng.integers(len(frontier)))])
        if len(chosen) == n_nodes:
            return chosen
    raise ValueError(f"no weakly connected set of {n_nodes} nodes found")


def causeme_matrices(network):
    """
    CauseMe's ``scores``, ``pvalues``, and ``lags`` matrices for one inferred network
    :cite:`Runge2019nc`; entry ``[i, j]`` is the link ``i -> j``.

    Scores are the network's ``score`` clipped at 0 where it was computed
    (``SkeletonSettings(edge_scores=True)``), otherwise 1 for selected parents and 0
    elsewhere. P-values are each parent's smallest sequential p-value (1 for
    non-parents) and lags the skeleton's delay (0 for non-parents).
    """
    ds = network.dataset
    P = ds.sizes["source"]
    parents = np.zeros((P, P), dtype=bool)
    pvalues = np.ones((P, P))
    for t, sk in network.skeleton.items():
        for p in sk.parents():
            parents[p, t] = True
            pvalues[p, t] = min(sk.source_pvalues.get(v, 1.0) for v in sk.source_variables(p))
    score = ds["score"].values if "score" in ds else np.full((P, P), np.nan)
    scores = np.where(np.isfinite(score), np.clip(score, 0, None), parents.astype(float))
    lags = np.where(parents, ds["delay"].values, 0)
    np.fill_diagonal(scores, 0.0)
    return {"scores": scores, "pvalues": pvalues, "lags": lags.astype(int)}


def to_causeme(results, method_sha, path=None, **metadata):
    """
    The CauseMe upload dictionary for one experiment :cite:`Runge2019nc`.

    Parameters
    ----------
    results : list
        One entry per dataset, in the order of the downloaded archive: a
        MultiplexNetwork, or a dict with ``scores`` (and optionally ``pvalues``,
        ``lags``) square arrays.
    method_sha : str
        The method's hash from CauseMe.
    path : str, Path, None
        Write the dictionary as JSON here.
    **metadata
        Further fields, e.g. ``model``, ``experiment``, ``parameter_values``.

    Returns
    -------
    dict
        ``method_sha``, ``scores``, and (when every entry has them) ``pvalues`` and
        ``lags``, each a list of row-major flattened matrices.
    """
    mats = [r if isinstance(r, dict) else causeme_matrices(r) for r in results]
    out = {"method_sha": method_sha, **metadata}
    for key in ("scores", "pvalues", "lags"):
        if all(key in m for m in mats):
            out[key] = [np.asarray(m[key]).ravel(order="C").tolist() for m in mats]
    if path is not None:
        with open(path, "w") as fh:
            json.dump(out, fh)
    return out
