"""
Benchmark infoflow against directed-information baselines on networks of known structure.

Every method sees the same integer symbols and the same fixed lag grid ``1..L``
(infoflow with ``max_lag = max_source_lag = L``). Continuous data are binned into 3
equal-frequency bins first. Cases: the coupled tent-map ring (weak, moderate, and
moderate at lag 1), noisy elementary cellular automata (rules 90 and 110), and random
10-node VAR and logistic networks with lag-1 links. Writes
``validation/results/di/results.json`` and prints an RST table.

Example::

    python validation/di_benchmark.py            # all cases
    python validation/di_benchmark.py --cases eca90 tent_weak
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from di_baselines import dig_bounded_indegree, ocse, pairwise_di  # noqa: E402

from infoflow import datasets  # noqa: E402
from infoflow.benchmarks import random_network, simulate_network  # noqa: E402
from infoflow.data import DiscreteData  # noqa: E402
from infoflow.network import infer_multiplex  # noqa: E402
from infoflow.selection import SkeletonSettings, rank_bins  # noqa: E402

SLOPES = [1.7692923542386314, 1.695620769559862, 1.8737085638967708, 1.8535602775842457]
RING = {(0, 1), (1, 2), (1, 3), (2, 0), (3, 0)}
ECA = {((i - 1) % 6, i) for i in range(6)} | {((i + 1) % 6, i) for i in range(6)}


def _binned(x, bins=3):
    return np.column_stack([rank_bins(x[:, j], bins) for j in range(x.shape[1])])


def cases():
    out = {}
    for name, eps, lag in (("tent_weak", 0.1, 2), ("tent_moderate", 0.3, 2), ("tent_moderate_lag1", 0.3, 1)):
        out[name] = lambda eps=eps, lag=lag: (
            datasets.coupled_tent(20000, SLOPES, datasets.tent_ring_coupling(eps), seed=0),
            RING,
            lag,
            2,
        )
    for rule in (90, 110):
        out[f"eca{rule}"] = lambda rule=rule: (datasets.noisy_eca(20000, 6, rule, 0.05, seed=0), ECA, 1, 2)
    for kind in ("var", "logistic"):
        for rep in range(2):

            def make(kind=kind, rep=rep):
                graph = random_network(10, mean_in_degree=2.0, max_lag=1, prng=rep)
                x = simulate_network(graph, 5000, kind, prng=rep)
                return _binned(x), set(graph.edges()), 1, 3

            out[f"{kind}10_rep{rep}"] = make
    return out


def score(found, truth):
    tp = len(found & truth)
    return {
        "precision": tp / len(found) if found else float("nan"),
        "recall": tp / len(truth) if truth else float("nan"),
        "n_found": len(found),
        "n_true": len(truth),
    }


def run_infoflow(symbols, lag):
    settings = SkeletonSettings(
        n_perm_max_stat=200, n_perm_min_stat=200, n_perm_omnibus=200, n_perm_max_seq=200, max_source_lag=lag
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = infer_multiplex(
            symbols,
            max_lag=lag,
            skeleton_settings=settings,
            n_boot=50,
            n_null=50,
            node_layer=False,
            interpret=False,
            include_shared_candidates=False,
            prng=0,
        )
    kind = net.dataset["kind"].values
    return {(int(s), int(t)) for s, t in zip(*np.nonzero(kind == "parent"), strict=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--out", default=str(Path(__file__).parent / "results" / "di" / "results.json"))
    args = parser.parse_args()
    all_cases = cases()
    names = args.cases or list(all_cases)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        x, truth, lag, K = all_cases[name]()
        data = DiscreteData.from_discrete(x)
        methods = {
            "infoflow": lambda x=x, lag=lag: run_infoflow(x, lag),
            "dig": lambda data=data, lag=lag, K=K: dig_bounded_indegree(data, K=K, max_lag=lag),
            "ocse": lambda data=data, lag=lag: ocse(data, max_lag=lag),
            "pairwise_di": lambda data=data, lag=lag: pairwise_di(data, max_lag=lag),
        }
        results[name] = {"lag": lag, "K": K}
        for method, fn in methods.items():
            t0 = time.perf_counter()
            found = fn()
            results[name][method] = {**score(found, truth), "seconds": time.perf_counter() - t0}
            r = results[name][method]
            print(
                f"{name:22s} {method:12s} P={r['precision']:.2f} R={r['recall']:.2f} n={r['n_found']:3d} "
                f"({r['seconds']:.0f}s)",
                flush=True,
            )
        out_path.write_text(json.dumps(results, indent=2))
    print_table(results)


def print_table(results):
    methods = ("infoflow", "dig", "ocse", "pairwise_di")
    print("\n" + "  ".join(["case".ljust(22)] + [m.ljust(12) for m in methods]))
    for name, row in results.items():
        cells = [f"{row[m]['precision']:.2f}/{row[m]['recall']:.2f}".ljust(12) for m in methods]
        print("  ".join([name.ljust(22), *cells]))


if __name__ == "__main__":
    main()
