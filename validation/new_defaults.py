"""
Current against proposed estimator defaults on infoflow's benchmarks.

``current``: plug-in selection statistic, Miller--Madow layers (the defaults).
``proposed``: Miller--Madow selection statistic (``SkeletonSettings(entropy_estimator=
"miller_madow")``), Chao--Wang--Jost layers (``estimator="chao_wang_jost"``).

Each benchmark runs both with the same seeds and data:

* ``synthetic``: :func:`~infoflow.benchmarks.network_validation` on VAR,
  coupled-logistic and Lorenz-96 networks (``infer_multiplex``);
* ``motifs``: :func:`~infoflow.benchmarks.motif_validation` (layers and roles);
* ``netsim``, ``rivers``: ``auto.infer`` (fast preset) on NetSim and CausalRivers, as
  ``validation/external.py`` does.

    python validation/new_defaults.py synthetic --kind lorenz96 --config proposed --out-dir results/
"""

import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr

CONFIGS = {
    "current": {"selection": "plugin", "layers": "miller_madow"},
    "proposed": {"selection": "miller_madow", "layers": "chao_wang_jost"},
    "selection_only": {"selection": "miller_madow", "layers": "miller_madow"},
    "layers_only": {"selection": "plugin", "layers": "chao_wang_jost"},
}


def infer_kwargs(config):
    from infoflow.selection import SkeletonSettings

    c = CONFIGS[config]
    return {"skeleton_settings": SkeletonSettings(entropy_estimator=c["selection"]), "estimator": c["layers"]}


def synthetic(kind, config, reps, seed):
    from infoflow.benchmarks import network_validation

    samples = (1000,) if kind == "lorenz96" else (500, 2000)
    return network_validation(
        kind=kind, n_nodes=(10,), n_samples=samples, n_reps=reps, max_lag=3,
        levels=("skeleton", "intrinsic"), infer_kwargs=infer_kwargs(config), prng=seed,
    )  # fmt: skip


def motifs(config, reps, seed):
    from infoflow.benchmarks import motif_validation

    return motif_validation(n_samples=3000, n_reps=reps, infer_kwargs=infer_kwargs(config), prng=seed)


def external(bench, config, data_dir, seed, sims, subjects, sizes, graphs):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from external import METRICS, netsim_instances, rivers_instances, undirected_auroc

    from infoflow import auto
    from infoflow.benchmarks import score_edges

    c = CONFIGS[config]
    if bench == "netsim":
        instances = netsim_instances(data_dir, sims, subjects)
    else:
        instances = rivers_instances(data_dir, sizes, graphs, [24], seed)
    rng = np.random.default_rng(seed)
    labels, rows = [], []
    for label, data, truth in instances:
        s = int(rng.integers(2**32))
        t0 = time.perf_counter()
        net = auto.infer(
            data, preset="fast", edge_scores=True, entropy_estimator=c["selection"], layer_estimator=c["layers"],
            prng=s,
        ).network  # fmt: skip
        elapsed = time.perf_counter() - t0
        found = {(p, t) for t, sk in net.skeleton.items() if sk.significant for p in sk.parents()}
        score = net.dataset["score"].values
        r = score_edges(found, None, score, truth)
        rows.append([r["auroc"], r["auprc"], undirected_auroc(score, truth), r["precision"], r["recall"], elapsed])
        labels.append(label)
        print(label, round(r["auroc"], 3), flush=True)
    values = np.array(rows)
    return xr.Dataset({m: (("instance",), values[:, i]) for i, m in enumerate(METRICS)}, coords={"instance": labels})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bench", choices=("synthetic", "motifs", "netsim", "rivers"))
    parser.add_argument("--config", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--kind", choices=("var", "logistic", "lorenz96"), default="var")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--data", default=None)
    parser.add_argument("--sims", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--subjects", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=[3])
    parser.add_argument("--graphs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    start = time.perf_counter()
    if args.bench == "synthetic":
        ds, name = synthetic(args.kind, args.config, args.reps, args.seed), f"synthetic_{args.kind}"
    elif args.bench == "motifs":
        ds, name = motifs(args.config, args.reps, args.seed), "motifs"
    else:
        ds = external(args.bench, args.config, args.data, args.seed, args.sims, args.subjects, args.sizes, args.graphs)
        name = args.bench
    ds.attrs["seconds"] = time.perf_counter() - start
    out = Path(args.out_dir) / f"{name}_{args.config}.nc"
    ds.to_netcdf(out)
    print(out, round(ds.attrs["seconds"]), "s", flush=True)


if __name__ == "__main__":
    main()
