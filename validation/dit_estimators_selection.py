"""
Parent-selection accuracy with dit's discrete entropy estimators.

Runs :func:`~infoflow.benchmarks.network_validation` (random networks of known
structure, simulated as a VAR or coupled logistic maps) with the plug-in selection
statistic replaced by one of dit's estimators (``SkeletonSettings.entropy_estimator``).
Every estimator sees the same graphs and simulations (same ``--seed``); the layer
estimator stays Miller--Madow, so differences come from selection.

    python validation/dit_estimators_selection.py --kind var --estimator zhang --out-dir results/
"""

import argparse
import time
from pathlib import Path

from infoflow.benchmarks import network_validation
from infoflow.selection import SkeletonSettings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("var", "logistic"), required=True)
    parser.add_argument("--estimator", required=True)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--samples", type=int, nargs="+", default=[500, 2000])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    start = time.perf_counter()
    ds = network_validation(
        kind=args.kind,
        n_nodes=(args.nodes,),
        n_samples=tuple(args.samples),
        n_reps=args.reps,
        max_lag=args.max_lag,
        levels=("skeleton", "intrinsic"),
        infer_kwargs={"skeleton_settings": SkeletonSettings(entropy_estimator=args.estimator)},
        prng=args.seed,
    )
    ds = ds.expand_dims(estimator=[args.estimator])
    ds.attrs["seconds"] = time.perf_counter() - start
    out = Path(args.out_dir) / f"selection_{args.kind}_{args.estimator}.nc"
    ds.to_netcdf(out)
    print(out, round(ds.attrs["seconds"]), "s", flush=True)


if __name__ == "__main__":
    main()
