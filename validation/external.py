"""
infoflow's automatic inference against baselines on external ground-truth benchmarks.

NetSim (Smith et al. 2011): per subject of the chosen simulations. CausalRivers
(Stein et al. 2025): connected subgraphs of the East German river network sampled with
``sample_subgraph``; as in that paper, AUROC is computed per graph (self-links
ignored) and averaged.

Data are not shipped: download NetSim's ``sims.tar.gz`` and CausalRivers'
``product.zip`` and pass their directories.

    python validation/external.py netsim --data .bench_data/netsim --sims 1 2 3 4 --subjects 10
    python validation/external.py rivers --data .bench_data/rivers/product --sizes 3 5 --graphs 40 --resample 24 96
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from dit.inference._symbols import as_generator

from infoflow import auto
from infoflow.baselines import run_baseline
from infoflow.benchmarks import rank_metrics, score_edges
from infoflow.io import causalrivers_subset, load_netsim, read_causalrivers, sample_subgraph

METRICS = ("auroc", "auprc", "auroc_undirected", "precision", "recall", "runtime")


def run_method(method, data, max_lag, seed):
    t0 = time.perf_counter()
    if method == "auto":
        net = auto.infer(data, preset="fast", max_lag=max_lag, edge_scores=True, prng=seed).network
        found = {(p, t) for t, sk in net.skeleton.items() if sk.significant for p in sk.parents()}
        delay, score = net.dataset["delay"].values, net.dataset["score"].values
    else:
        base = run_baseline(method, data, max_lag=max_lag, alpha=0.01)
        found, delay, score = base.found, base.lags, base.score
    return found, delay, score, time.perf_counter() - t0


def undirected_auroc(score, truth):
    """
    AUROC for detecting a link in either direction, from the larger of the two scores.
    """
    P = len(score)
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=-np.inf)
    s = np.maximum(s, s.T)
    linked = np.zeros((P, P), dtype=bool)
    for a, b in truth.edges():
        if a != b:
            linked[a, b] = linked[b, a] = True
    iu = np.triu_indices(P, 1)
    return rank_metrics(np.where(np.isfinite(s[iu]), s[iu], np.nan), linked[iu])[0]


def evaluate(instances, methods, max_lag, seed):
    """
    ``instances`` yields ``(label, data, truth)``; returns metrics over (method, instance).
    """
    rng = as_generator(seed)
    labels, rows = [], []
    for label, data, truth in instances:
        labels.append(label)
        s = int(rng.integers(2**32))
        row = []
        for method in methods:
            found, delay, score, elapsed = run_method(method, data, max_lag, s)
            r = score_edges(found, None, score, truth)
            row.append([r["auroc"], r["auprc"], undirected_auroc(score, truth), r["precision"], r["recall"], elapsed])
        rows.append(row)
        print(label, {m: round(v[0], 3) for m, v in zip(methods, row, strict=True)}, flush=True)
    values = np.array(rows)  # (instance, method, metric)
    return xr.Dataset(
        {m: (("instance", "algorithm"), values[:, :, i]) for i, m in enumerate(METRICS)},
        coords={"instance": labels, "algorithm": list(methods)},
    )


def netsim_instances(directory, sims, subjects):
    from scipy.io import loadmat

    for k in sims:
        path = Path(directory) / f"sim{k}.mat"
        n_subjects = int(np.squeeze(loadmat(path)["Nsubjects"]))
        for subject in range(min(subjects, n_subjects)):
            data, truth = load_netsim(path, subject)
            yield f"sim{k}/s{subject}", data, truth


def rivers_instances(directory, sizes, graphs, resample, seed):
    directory = Path(directory)
    values, graph, stations = read_causalrivers(
        directory / "rivers_ts_east_germany.csv", directory / "rivers_east_germany.p"
    )
    present = set(stations)
    graph = graph.subgraph([n for n in graph.nodes if str(n) in present]).copy()
    rng = as_generator(seed)
    for size in sizes:
        drawn = set()
        while len(drawn) < graphs:
            nodes = sample_subgraph(graph, size, prng=rng)
            key = tuple(sorted(nodes))
            if key in drawn:
                continue
            drawn.add(key)
            for block in resample:
                data, truth = causalrivers_subset(values, graph, stations, nodes, block)
                if len(data) < 200 or truth.number_of_edges() == 0:
                    continue
                yield f"n{size}/r{block}/{'-'.join(map(str, key))}", data, truth


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench", choices=("netsim", "rivers"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--methods", nargs="+", default=["auto", "var", "pcmciplus"])
    parser.add_argument("--sims", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--subjects", type=int, default=10)
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--graphs", type=int, default=40)
    parser.add_argument("--resample", type=int, nargs="+", default=[24])
    parser.add_argument("--max-lag", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.bench == "netsim":
        instances = netsim_instances(args.data, args.sims, args.subjects)
        max_lag = args.max_lag or 2
    else:
        instances = rivers_instances(args.data, args.sizes, args.graphs, args.resample, args.seed)
        max_lag = args.max_lag or 3
    ds = evaluate(instances, args.methods, max_lag, args.seed)
    group = [
        lab.split("/")[0] if args.bench == "netsim" else "/".join(lab.split("/")[:2]) for lab in ds["instance"].values
    ]
    ds = ds.assign_coords(group=("instance", group))
    print(ds.groupby("group").mean().to_dataframe().round(3))
    if args.out:
        ds.to_netcdf(args.out, engine="scipy")


if __name__ == "__main__":
    main()
