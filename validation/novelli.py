"""
Random-network validation after Novelli et al. (2019).

Example::

    python validation/novelli.py --kind var --nodes 10 30 --samples 300 1000 3000 --reps 3 --out var.nc
"""

import argparse
import threading
import time
from datetime import timedelta

import numpy as np

from infoflow.benchmarks import network_validation
from infoflow.parallel import thread_map
from infoflow.selection import SkeletonSettings


def progress_map(base):
    """
    Wrap a ``map_fn`` so every finished target or edge prints a progress line with a
    linear estimate of the time left in that phase.
    """

    def map_fn(fn, iterable):
        items = list(iterable)
        if not items:
            return base(fn, items)
        phase = "targets" if isinstance(items[0][0], (int, np.integer)) else "edges"
        lock, done, start = threading.Lock(), [0], time.monotonic()
        print(f"[{time.strftime('%H:%M:%S')}] {phase}: 0/{len(items)} started", flush=True)

        def tracked(item):
            result = fn(item)
            with lock:
                done[0] += 1
                elapsed = time.monotonic() - start
                left = elapsed / done[0] * (len(items) - done[0])
                print(
                    f"[{time.strftime('%H:%M:%S')}] {phase}: {done[0]}/{len(items)} done, "
                    f"elapsed {timedelta(seconds=round(elapsed))}, ~{timedelta(seconds=round(left))} left",
                    flush=True,
                )
            return result

        return base(tracked, items)

    return map_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--kind", choices=("var", "logistic"), default="var")
    parser.add_argument("--nodes", type=int, nargs="+", default=[10])
    parser.add_argument("--samples", type=int, nargs="+", default=[1000])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--in-degree", type=float, default=3.0, help="0 gives empty networks (false-positive rate)")
    parser.add_argument("--alpha", type=float, default=0.01, help="significance level of every selection stage")
    parser.add_argument(
        "--estimator", choices=("plugin", "ksg", "adaptive", "gaussian", "trend", "coarse"), default="plugin"
    )
    parser.add_argument("--null", choices=("free", "strata"), default="free", help="selection null (plug-in/adaptive)")
    parser.add_argument("--statistic", choices=("raw", "debiased"), default="raw", help="selection statistic")
    parser.add_argument("--threads", type=int, default=8, help="targets inferred in parallel")
    parser.add_argument("--prescreen", type=float, default=None, help="pre-screen level for source processes")
    parser.add_argument("--n-targets", type=int, default=None, help="infer and score this many sampled targets")
    parser.add_argument(
        "--no-shared", action="store_true", help="skip layer estimates for shared-only candidates (not scored)"
    )
    parser.add_argument("--device", default=None, help="batched kernels: cpu, mps, cuda, or auto (default: NumPy path)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the results to this netCDF file")
    args = parser.parse_args()

    n_perm = max(200, int(np.ceil(2 / args.alpha)))
    settings = SkeletonSettings(
        estimator=args.estimator,
        null=args.null,
        statistic=args.statistic,
        prescreen_alpha=args.prescreen,
        **{f"alpha_{s}": args.alpha for s in ("max_stat", "min_stat", "omnibus", "max_seq", "pairs")},
        **{f"n_perm_{s}": n_perm for s in ("max_stat", "min_stat", "omnibus", "max_seq", "pairs")},
    )
    ds = network_validation(
        args.kind,
        n_nodes=args.nodes,
        n_samples=args.samples,
        n_reps=args.reps,
        mean_in_degree=args.in_degree,
        n_targets=args.n_targets,
        infer_kwargs={
            "include_shared_candidates": not args.no_shared,
            "skeleton_settings": settings,
            "map_fn": progress_map(thread_map(args.threads)),
            "device": args.device,
        },
        prng=args.seed,
    )
    ds.attrs.update(
        alpha=args.alpha,
        estimator=args.estimator,
        null=args.null,
        statistic=args.statistic,
        prescreen=str(args.prescreen),
        n_targets=str(args.n_targets),
        n_perm=n_perm,
        device=str(args.device),
    )
    print(ds[["precision", "recall", "specificity", "lag_error", "runtime"]].mean("rep").to_dataframe().round(3))
    if args.out:
        ds.to_netcdf(args.out, engine="scipy")


if __name__ == "__main__":
    main()
