"""
Validation of :func:`infoflow.infer` against its member estimators run alone.

Every arm runs through :func:`~infoflow.infer` with the balanced preset, so all arms
share the significance level, pre-screen, layer settings, and sampled targets; the
``auto`` arm uses the preset's estimator ensemble and the others pin one estimator.
Interpretation and the node layer are off in every arm (they do not change the
scored levels).

Example::

    python validation/auto.py --scenario mute --arm auto --out mute_auto.nc
"""

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import xarray as xr
from dit.inference._symbols import as_generator

sys.path.insert(0, str(Path(__file__).parent))

from novelli import progress_map  # noqa: E402

from infoflow import datasets  # noqa: E402
from infoflow.auto import infer  # noqa: E402
from infoflow.benchmarks import random_network, score_network, simulate_network  # noqa: E402
from infoflow.parallel import thread_map  # noqa: E402

LEVELS = ("skeleton", "intrinsic")
METRICS = ("precision", "recall", "specificity", "lag_error", "n_found")


def _graph(n, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for s, t, lag in edges:
        graph.add_edge(s, t, lag=lag)
    return graph


# Small scenarios: (data, true graph). Lags from the generators' docstrings.
SMALL = {
    "mute": lambda: (
        datasets.mute_network(3000, seed=0),
        _graph(5, [(0, 1, 2), (0, 2, 3), (0, 3, 2), (3, 4, 1), (4, 3, 1)]),
    ),
    "orthogonal": lambda: (datasets.orthogonal_features(4000, seed=0), _graph(3, [(0, 1, 1), (0, 2, 1)])),
    "xor": lambda: (datasets.xor_synergy(3000, noise=0.05, seed=0), _graph(3, [(0, 2, 1), (1, 2, 1)])),
    "common_driver": lambda: (datasets.common_driver(3000, seed=0), _graph(3, [(2, 0, 1), (2, 1, 2)])),
    "chain": lambda: (datasets.chain(3000, seed=0), _graph(3, [(0, 1, 1), (1, 2, 2)])),
}

# Large scenarios: (kind, samples, mean in-degree), drawn as network_validation(prng=0) draws them.
LARGE = {
    "var_T3000": ("var", 3000, 3.0),
    "var_T10000": ("var", 10000, 3.0),
    "logistic_T10000": ("logistic", 10000, 3.0),
    "empty_T10000": ("var", 10000, 0.0),
}


def scenario(name, n_targets):
    """
    The data, true graph, scored targets, and inference seed of a scenario.
    """
    if name in SMALL:
        data, graph = SMALL[name]()
        return data, graph, None, 0
    kind, T, degree = LARGE[name]
    rng = as_generator(0)
    graph = random_network(100, degree, 5, rng)
    data = simulate_network(graph, T, kind, prng=rng)
    targets = sorted(int(t) for t in rng.choice(100, n_targets, replace=False))
    return data, graph, targets, int(rng.integers(2**32))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--scenario", choices=(*SMALL, *LARGE), required=True)
    parser.add_argument("--arm", choices=("auto", "gaussian", "trend", "coarse", "plugin"), required=True)
    parser.add_argument("--n-targets", type=int, default=20, help="sampled targets of the 100-node scenarios")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data, graph, targets, seed = scenario(args.scenario, args.n_targets)
    pinned = {} if args.arm == "auto" else {"estimator": args.arm}
    threads = args.threads
    result = infer(
        data,
        preset="balanced",
        targets=targets,
        max_lag=5,
        threads=threads,
        interpret=False,
        node_layer=False,
        map_fn=progress_map(thread_map(threads)),
        prng=seed,
        **pinned,
    )
    print(result.report, flush=True)
    scores = score_network(result.network, graph, LEVELS, targets)
    ds = xr.Dataset(
        {m: (("level",), np.array([scores[level][m] for level in LEVELS], dtype=float)) for m in METRICS},
        coords={"level": list(LEVELS)},
        attrs={
            "scenario": args.scenario,
            "arm": args.arm,
            "estimators": ",".join(result.report.estimators),
            "runtime": float(result.report.actual_seconds),
            "predicted": float(result.report.predicted_seconds),
            "n_targets": len(targets) if targets is not None else graph.number_of_nodes(),
            "admitted": "; ".join(f"{s}->{t}:{e}" for (s, t), e in sorted(result.report.admitted.items())),
        },
    )
    print(ds[list(METRICS)].to_dataframe().round(3), flush=True)
    ds.to_netcdf(args.out, engine="scipy")


if __name__ == "__main__":
    main()
