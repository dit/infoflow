"""
Run infoflow's automatic inference with layers on the benchmark suites and save, per
network, everything needed to compare layer reporting strategies.

For each instance the record holds the true edges, the selected edges (transfer
entropy significant in parent selection, or shared-only candidates passing the TDMI
screen), and every edge's layer estimates, intervals, per-layer p-values, and FDR
decisions.

    python validation/layer_report.py motifs synthetic mute --out .bench_data/results/layers
    python validation/layer_report.py netsim --netsim .bench_data/netsim --out ...
    python validation/layer_report.py rivers --rivers .bench_data/rivers/product --out ...
"""

import argparse
import pickle
import time
import warnings
import zlib
from pathlib import Path

import networkx as nx
import numpy as np
from dit.inference._symbols import as_generator

from infoflow import auto, datasets
from infoflow.benchmarks import lorenz96_graph, random_network, simulate_network

# Expected layers of selected edges in motifs whose composition is known by construction.
MOTIFS = {
    # y = x1 xor x2: each source informs y only jointly with the other.
    "xor": (
        lambda s: datasets.xor_synergy(3000, noise=0.05, seed=s),
        [(0, 2), (1, 2)],
        {(0, 2): "synergistic", (1, 2): "synergistic"},
    ),
    # z drives x and y: x -> y carries shared flow only.
    "common_driver": (
        lambda s: datasets.common_driver(3000, seed=s),
        [(2, 0), (2, 1)],
        {(2, 0): "intrinsic", (2, 1): "intrinsic", (0, 1): "shared"},
    ),
    "chain": (lambda s: datasets.chain(3000, seed=s), [(0, 1), (1, 2)], {(0, 1): "intrinsic", (1, 2): "intrinsic"}),
    "orthogonal": (
        lambda s: datasets.orthogonal_features(4000, seed=s),
        [(0, 1), (0, 2)],
        {(0, 1): "intrinsic", (0, 2): "intrinsic"},
    ),
    # Weaker versions: noisier couplings and shorter series.
    "xor_weak": (
        lambda s: datasets.xor_synergy(800, noise=0.25, seed=s),
        [(0, 2), (1, 2)],
        {(0, 2): "synergistic", (1, 2): "synergistic"},
    ),
    "common_driver_weak": (
        lambda s: datasets.common_driver(800, noise=0.25, seed=s),
        [(2, 0), (2, 1)],
        {(2, 0): "intrinsic", (2, 1): "intrinsic", (0, 1): "shared"},
    ),
    "chain_weak": (
        lambda s: datasets.chain(800, noise=0.3, seed=s),
        [(0, 1), (1, 2)],
        {(0, 1): "intrinsic", (1, 2): "intrinsic"},
    ),
    "orthogonal_weak": (
        lambda s: datasets.orthogonal_features(1000, noise=1.0, seed=s),
        [(0, 1), (0, 2)],
        {(0, 1): "intrinsic", (0, 2): "intrinsic"},
    ),
}


def graph_of(edges, n):
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return g


def record(net, truth, expected=None):
    ds = net.dataset
    selected = {(p, t) for t, sk in net.skeleton.items() if sk.significant for p in sk.parents()}
    kind = ds["kind"].values
    shared = {(int(s), int(t)) for s, t in zip(*np.nonzero(kind == "shared_candidate"), strict=True)}
    return {
        "truth": sorted(truth),
        "expected": expected or {},
        "parents": sorted(selected),
        "shared_candidates": sorted(shared),
        "layers": list(ds.coords["layer"].values),
        "weight": ds["weight"].values,
        "ci_low": ds["ci_low"].values,
        "ci_high": ds["ci_high"].values,
        "pvalue": ds["pvalue"].values,
        "qvalue": ds["qvalue"].values,
        "significant": ds["significant"].values,
        "score": ds["score"].values,
    }


def run(data, max_lag, seed):
    return auto.infer(
        data, preset="fast", max_lag=max_lag, include_shared_candidates=True, edge_scores=True, prng=seed
    ).network


def suite_motifs(n_seeds):
    for name, (gen, edges, expected) in MOTIFS.items():
        for s in range(n_seeds):
            yield f"motif/{name}/{s}", gen(s), set(edges), expected, 3


def suite_synthetic(n_seeds, n_nodes=10):
    for kind, lag in (("var", 5), ("logistic", 5), ("lorenz96", 2)):
        for s in range(n_seeds):
            if kind == "lorenz96":
                yield (
                    f"synthetic/{kind}/{s}",
                    datasets.lorenz96(1000, n_vars=n_nodes, seed=s),
                    set(lorenz96_graph(n_nodes).edges()),
                    None,
                    lag,
                )
            else:
                g = random_network(n_nodes, 3.0, 5, prng=s)
                yield f"synthetic/{kind}/{s}", simulate_network(g, 1000, kind, prng=s), set(g.edges()), None, lag


def suite_mute(n_seeds):
    for s in range(n_seeds):
        yield f"mute/{s}", datasets.mute_network(3000, seed=s), set(datasets.MUTE_EDGES), None, 5


def suite_netsim(directory, sims, subjects):
    from infoflow.io import load_netsim

    for k in sims:
        for s in range(subjects):
            data, truth = load_netsim(Path(directory) / f"sim{k}.mat", s)
            yield f"netsim/sim{k}/{s}", data, set(truth.edges()), None, 2


def suite_rivers(directory, sizes, graphs, seed=0):
    from infoflow.io import causalrivers_subset, read_causalrivers, sample_subgraph

    directory = Path(directory)
    values, graph, stations = read_causalrivers(
        directory / "rivers_ts_east_germany.csv", directory / "rivers_east_germany.p"
    )
    graph = graph.subgraph([n for n in graph.nodes if str(n) in set(stations)]).copy()
    for size in sizes:
        rng, seen, k = as_generator(size), set(), 0
        while k < graphs:
            nodes = tuple(sample_subgraph(graph, size, prng=rng))
            if nodes in seen:
                continue
            seen.add(nodes)
            data, truth = causalrivers_subset(values, graph, stations, list(nodes), 24)
            if len(data) < 200 or truth.number_of_edges() == 0:
                continue
            k += 1
            yield f"rivers/n{size}/{k}", data, set(truth.edges()), None, 3


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="+", choices=("motifs", "synthetic", "mute", "netsim", "rivers"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--netsim", default=".bench_data/netsim")
    parser.add_argument("--sims", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--subjects", type=int, default=5)
    parser.add_argument("--rivers", default=".bench_data/rivers/product")
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--graphs", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suites = {
        "motifs": lambda: suite_motifs(args.seeds),
        "synthetic": lambda: suite_synthetic(args.seeds),
        "mute": lambda: suite_mute(args.seeds),
        "netsim": lambda: suite_netsim(args.netsim, args.sims, args.subjects),
        "rivers": lambda: suite_rivers(args.rivers, args.sizes, args.graphs),
    }
    for suite in args.suites:
        for label, data, truth, expected, max_lag in suites[suite]():
            path = out / (label.replace("/", "__") + ".p")
            if path.exists():
                continue
            t0 = time.perf_counter()
            net = run(data, max_lag, zlib.crc32(label.encode()))
            rec = record(net, truth, expected)
            rec["seconds"] = time.perf_counter() - t0
            rec["label"] = label
            with open(path, "wb") as fh:
                pickle.dump(rec, fh)
            print(label, f"{rec['seconds']:.0f}s", len(rec["parents"]), "parents", flush=True)


if __name__ == "__main__":
    main()
